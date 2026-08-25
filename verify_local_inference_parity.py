"""Paridad local entre el flujo validado y la nueva ruta de inferencia."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATASET = ROOT / "weatherAUS_2026C1.csv"
INFERENCE_SCRIPT = ROOT / "docker" / "inferencia.py"
SAMPLE_ROWS = 512


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


selection = json.loads((ARTIFACTS / "oof_selection.json").read_text(encoding="utf-8"))
final_report = json.loads(
    (ARTIFACTS / "final_test_metrics.json").read_text(encoding="utf-8")
)
threshold = float(selection["winner"]["threshold"])
assert selection["winner"]["candidate_id"] == "nn_config_5"
assert threshold == final_report["threshold"]

current_hashes = {
    "selection": sha256_file(ARTIFACTS / "oof_selection.json"),
    "model": sha256_file(ROOT / selection["frozen_model_artifact"]),
    "preprocessor": sha256_file(ROOT / selection["frozen_preprocessor_artifact"]),
}
assert current_hashes == final_report["frozen_artifact_sha256_after"]

data = pd.read_csv(DATASET)
data = data.dropna(subset=["RainTomorrow"])
y_all = data["RainTomorrow"].map({"No": 0, "Yes": 1})
development_index, final_test_index = train_test_split(
    data.index.to_numpy(),
    test_size=0.20,
    random_state=42,
    stratify=y_all,
)
development = data.loc[np.sort(development_index)].sort_values(["Date", "Location"])
sample_positions = np.linspace(0, len(development) - 1, SAMPLE_ROWS, dtype=int)
sample = development.iloc[sample_positions].copy()
sample_index = sample.index.to_numpy()
assert len(np.unique(sample_index)) == SAMPLE_ROWS
assert set(sample_index).isdisjoint(set(final_test_index))
sample = sample.drop(
    columns=["Unnamed: 0", "RainTomorrow", "RainfallTomorrow", "Region"],
    errors="ignore",
)

# Referencia independiente: misma secuencia validada en prioridad 5.
preprocessor = joblib.load(ROOT / selection["frozen_preprocessor_artifact"])
model = keras.models.load_model(ROOT / selection["frozen_model_artifact"])
reference_transformed = preprocessor.transform(sample)
reference_probability = model.predict(
    reference_transformed,
    batch_size=1024,
    verbose=0,
).reshape(-1)
reference_label = np.where(
    reference_probability >= threshold,
    "Llueve",
    "No llueve",
)

with tempfile.TemporaryDirectory(prefix="weather_local_parity_") as temp_dir:
    temp_dir = Path(temp_dir)
    input_path = temp_dir / "input.csv"
    output_path = temp_dir / "output.csv"
    sample.to_csv(input_path, index=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--artifacts-dir",
            str(ARTIFACTS),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    candidate_output = pd.read_csv(output_path)

assert list(candidate_output.columns) == ["prediccion", "Probabilidad"]
assert len(candidate_output) == SAMPLE_ROWS
candidate_probability = candidate_output["Probabilidad"].to_numpy()
candidate_label = candidate_output["prediccion"].to_numpy()
absolute_delta = np.abs(candidate_probability - reference_probability)
max_absolute_delta = float(absolute_delta.max())
mean_absolute_delta = float(absolute_delta.mean())
probabilities_match = bool(
    np.allclose(candidate_probability, reference_probability, rtol=1e-6, atol=1e-7)
)
labels_match = bool(np.array_equal(candidate_label, reference_label))
threshold_labels_match = bool(
    np.array_equal(
        candidate_label,
        np.where(candidate_probability >= threshold, "Llueve", "No llueve"),
    )
)
assert probabilities_match
assert labels_match
assert threshold_labels_match

source = INFERENCE_SCRIPT.read_text(encoding="utf-8")
for forbidden in (".fit(", "pycaret", "preprocesar_datos", "pipeline.pkl"):
    assert forbidden not in source.lower()
assert source.count("preprocessor.transform(frame)") == 1
assert source.count("model.predict(") == 1
assert 'threshold = float(winner["threshold"])' in source

summary = {
    "protocol": "local_inference_parity",
    "container_parity_status": "pending_docker_runtime",
    "sample_rows": SAMPLE_ROWS,
    "sample_source": "random_primary_development_only",
    "sample_test_overlap": 0,
    "sample_index_sha256": hashlib.sha256(sample_index.tobytes()).hexdigest(),
    "candidate_id": selection["winner"]["candidate_id"],
    "threshold": threshold,
    "probability_semantics": "P(RainTomorrow=Yes)",
    "probabilities_match": probabilities_match,
    "labels_match": labels_match,
    "threshold_labels_match": threshold_labels_match,
    "max_absolute_probability_delta": max_absolute_delta,
    "mean_absolute_probability_delta": mean_absolute_delta,
    "rtol": 1e-6,
    "atol": 1e-7,
    "frozen_artifact_sha256": current_hashes,
    "subprocess_stdout": completed.stdout.strip(),
    "subprocess_stderr": completed.stderr.strip(),
}
(ARTIFACTS / "local_inference_parity.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
np.savez_compressed(
    ARTIFACTS / "local_inference_parity_predictions.npz",
    row_index=sample_index,
    reference_probability=reference_probability.astype(np.float32),
    candidate_probability=candidate_probability.astype(np.float32),
    reference_positive=(reference_probability >= threshold),
    candidate_positive=(candidate_probability >= threshold),
)

print(f"OK: {SAMPLE_ROWS} filas de desarrollo; interseccion con test final=0")
print(f"OK: etiquetas identicas={labels_match}")
print(
    "OK: probabilidades dentro de tolerancia; "
    f"max_abs_delta={max_absolute_delta:.12g}"
)
print(f"OK: threshold cargado del manifiesto={threshold:.12f}")
print("PENDIENTE: build/run Docker no disponible en este entorno")
