"""Build, run and verify Docker inference against validated local outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "data" / "weatherAUS_2026C1.csv"
DOCKERFILE = ROOT / "docker" / "Dockerfile"
ATOL = 1e-7
RTOL = 1e-6


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="prediccion-lluvia:parity")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if shutil.which("docker") is None:
        raise RuntimeError("Docker CLI no esta disponible")
    run(["docker", "version"], capture_output=True)

    local_summary = json.loads(
        (ARTIFACTS / "local_inference_parity.json").read_text(encoding="utf-8")
    )
    local_predictions = np.load(ARTIFACTS / "local_inference_parity_predictions.npz")
    final_test_predictions = np.load(ARTIFACTS / "final_test_predictions.npz")
    row_index = local_predictions["row_index"]
    final_test_index = final_test_predictions["row_index"]
    if not set(row_index).isdisjoint(set(final_test_index)):
        raise AssertionError("La muestra Docker contiene filas del test final")

    selection = json.loads(
        (ARTIFACTS / "oof_selection.json").read_text(encoding="utf-8")
    )
    current_hashes = {
        "selection": sha256_file(ARTIFACTS / "oof_selection.json"),
        "model": sha256_file(ROOT / selection["frozen_model_artifact"]),
        "preprocessor": sha256_file(
            ROOT / selection["frozen_preprocessor_artifact"]
        ),
    }
    if current_hashes != local_summary["frozen_artifact_sha256"]:
        raise AssertionError("Los artefactos congelados cambiaron desde la paridad local")

    data = pd.read_csv(DATASET)
    sample = data.loc[row_index].drop(
        columns=["Unnamed: 0", "RainTomorrow", "RainfallTomorrow", "Region"],
        errors="ignore",
    )
    reference_probability = local_predictions["candidate_probability"].astype(float)
    reference_label = np.where(
        local_predictions["candidate_positive"], "Llueve", "No llueve"
    )

    if not args.skip_build:
        run(
            [
                "docker",
                "build",
                "--file",
                str(DOCKERFILE),
                "--tag",
                args.image,
                str(ROOT),
            ]
        )

    with tempfile.TemporaryDirectory(prefix="weather_docker_parity_") as temp_dir:
        temp_path = Path(temp_dir).resolve()
        sample.to_csv(temp_path / "input.csv", index=False)
        run(
            [
                "docker",
                "run",
                "--rm",
                "--mount",
                f"type=bind,source={temp_path},target=/files",
                args.image,
            ]
        )
        candidate = pd.read_csv(temp_path / "output.csv")

    if list(candidate.columns) != ["prediccion", "Probabilidad"]:
        raise AssertionError("El contenedor produjo un esquema de salida inesperado")
    candidate_probability = candidate["Probabilidad"].to_numpy()
    candidate_label = candidate["prediccion"].to_numpy()
    labels_match = bool(np.array_equal(candidate_label, reference_label))
    probabilities_match = bool(
        np.allclose(
            candidate_probability,
            reference_probability,
            rtol=RTOL,
            atol=ATOL,
        )
    )
    if not labels_match or not probabilities_match:
        raise AssertionError("La inferencia Docker no tiene paridad con la inferencia local")

    absolute_delta = np.abs(candidate_probability - reference_probability)
    image_id = run(
        ["docker", "image", "inspect", "--format={{.Id}}", args.image],
        capture_output=True,
    ).stdout.strip()
    summary = {
        "protocol": "docker_end_to_end_inference_parity",
        "container_parity_status": "verified",
        "image": args.image,
        "image_id": image_id,
        "sample_rows": int(len(row_index)),
        "sample_test_overlap": 0,
        "sample_index_sha256": local_summary["sample_index_sha256"],
        "candidate_id": selection["winner"]["candidate_id"],
        "threshold": float(selection["winner"]["threshold"]),
        "labels_match": labels_match,
        "probabilities_match": probabilities_match,
        "max_absolute_probability_delta": float(absolute_delta.max()),
        "mean_absolute_probability_delta": float(absolute_delta.mean()),
        "rtol": RTOL,
        "atol": ATOL,
        "frozen_artifact_sha256": current_hashes,
    }
    (ARTIFACTS / "docker_inference_parity.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        ARTIFACTS / "docker_inference_parity_predictions.npz",
        row_index=row_index,
        local_probability=reference_probability.astype(np.float32),
        docker_probability=candidate_probability.astype(np.float32),
        local_positive=(reference_label == "Llueve"),
        docker_positive=(candidate_label == "Llueve"),
    )

    print(f"OK: {len(row_index)} filas de desarrollo; interseccion con test final=0")
    print(f"OK: etiquetas identicas={labels_match}")
    print(
        "OK: probabilidades dentro de tolerancia; "
        f"max_abs_delta={absolute_delta.max():.12g}"
    )
    print(f"OK: imagen={args.image} id={image_id}")


if __name__ == "__main__":
    main()
