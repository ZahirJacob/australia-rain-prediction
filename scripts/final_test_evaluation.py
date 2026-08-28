"""Evaluacion unica e irreversible del test final congelado.

No contiene fit, seleccion de modelo, ajuste de hiperparametros ni busqueda de
threshold. Una vez creado el marcador exclusivo, el script rechaza reejecuciones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]


def repo_relative(path):
    """Repository-relative POSIX path when `path` is inside the repo, absolute otherwise."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RANDOM_STATE = 42
FINAL_TEST_SIZE = 0.20


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(index):
    payload = ",".join(map(str, sorted(index))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "final_test_evaluation.completed.json"
    metrics_path = output_dir / "final_test_metrics.json"
    predictions_path = output_dir / "final_test_predictions.npz"

    if completion_path.exists() or metrics_path.exists() or predictions_path.exists():
        raise RuntimeError(
            "El test final ya fue abierto o existen resultados finales; "
            "se rechaza una segunda evaluacion."
        )
    completion_payload = {
        "status": "started",
        "policy": "single_final_test_evaluation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with completion_path.open("x", encoding="utf-8") as handle:
        json.dump(completion_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    selection_path = args.selection.resolve()
    temporal_path = args.temporal_summary.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    winner = selection["winner"]
    threshold = float(winner["threshold"])

    frozen_paths = {
        "selection": selection_path,
        "model": ROOT / selection["frozen_model_artifact"],
        "preprocessor": ROOT / selection["frozen_preprocessor_artifact"],
    }
    frozen_hashes_before = {
        name: sha256_file(path) for name, path in frozen_paths.items()
    }
    if temporal["frozen_artifact_sha256_after"] != frozen_hashes_before:
        raise RuntimeError("Los artefactos no coinciden con los verificados en prioridad 4")
    if temporal["decision_impact"] != "none":
        raise RuntimeError("La evaluacion temporal no es puramente descriptiva")

    data = pd.read_csv(args.dataset)
    data = data.dropna(subset=["RainTomorrow"])
    data = data.drop(columns=["Unnamed: 0", "RainfallTomorrow"], errors="ignore")
    y_all = data["RainTomorrow"].map({"No": 0, "Yes": 1})
    if y_all.isna().any():
        raise ValueError("RainTomorrow contiene etiquetas no reconocidas")

    all_indices = data.index.to_numpy()
    development_indices, final_test_indices = train_test_split(
        all_indices,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    development_indices = np.sort(development_indices)
    final_test_indices = np.sort(final_test_indices)
    if index_hash(development_indices) != selection["development_index_hash"]:
        raise RuntimeError("Desarrollo no coincide con el conjunto congelado")
    if len(final_test_indices) != selection["final_test_rows_reserved"]:
        raise RuntimeError("El numero de filas del test final no coincide")
    assert set(development_indices).isdisjoint(set(final_test_indices))

    preprocessor = joblib.load(frozen_paths["preprocessor"])
    learned_indices_before = tuple(
        preprocessor.named_steps["weather_features"].fit_indices_
    )
    if index_hash(learned_indices_before) != selection["development_index_hash"]:
        raise RuntimeError("El preprocesador no esta ajustado sobre desarrollo completo")
    if not set(learned_indices_before).isdisjoint(set(final_test_indices)):
        raise RuntimeError("El preprocesador contiene indices del test final")
    model = keras.models.load_model(frozen_paths["model"])

    # Primera y unica apertura del bloque final. Desde aqui solo transform/predict.
    X_test_final = data.loc[final_test_indices].drop(columns="RainTomorrow")
    y_test_final = y_all.loc[final_test_indices].to_numpy()
    X_test_transformed = preprocessor.transform(X_test_final)
    probabilities = model.predict(
        X_test_transformed,
        batch_size=1024,
        verbose=0,
    ).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int8)

    if tuple(preprocessor.named_steps["weather_features"].fit_indices_) != (
        learned_indices_before
    ):
        raise RuntimeError("Transform altero los indices aprendidos del preprocesador")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("El modelo produjo probabilidades no finitas")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError("El modelo produjo probabilidades fuera de [0, 1]")

    tn, fp, fn, tp = confusion_matrix(y_test_final, predictions, labels=[0, 1]).ravel()
    test_metrics = {
        "f1_positive": float(
            f1_score(y_test_final, predictions, pos_label=1, zero_division=0)
        ),
        "precision": float(
            precision_score(y_test_final, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test_final, predictions, pos_label=1, zero_division=0)
        ),
        "pr_auc": float(average_precision_score(y_test_final, probabilities)),
        "roc_auc": float(roc_auc_score(y_test_final, probabilities)),
    }
    prevalence = {
        "positive_rows": int(y_test_final.sum()),
        "negative_rows": int(len(y_test_final) - y_test_final.sum()),
        "positive_rate": float(y_test_final.mean()),
    }
    matrix = {
        "labels": ["No", "Yes"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    frozen_hashes_after = {
        name: sha256_file(path) for name, path in frozen_paths.items()
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("La evaluacion final altero un artefacto congelado")

    reference_metrics = {
        "oof": winner["oof_metrics"],
        "temporal_macro": temporal["temporal_macro_mean"],
        "temporal_pooled": temporal["temporal_pooled_metrics"],
    }
    comparisons = {
        reference: {
            metric: float(test_metrics[metric] - values[metric])
            for metric in test_metrics
        }
        for reference, values in reference_metrics.items()
    }
    result = {
        "protocol": "single_final_test_evaluation",
        "status": "complete",
        "decision_impact": "none",
        "model_candidate_id": winner["candidate_id"],
        "model_family": winner["family"],
        "model_parameters": winner["parameters"],
        "threshold": threshold,
        "threshold_source": "priority_3_oof_selection",
        "primary_metric": "f1_positive",
        "complementary_metrics": ["precision", "recall", "pr_auc", "roc_auc"],
        "test_rows": len(y_test_final),
        "test_index_hash": index_hash(final_test_indices),
        "development_index_hash": index_hash(development_indices),
        "prevalence": prevalence,
        "confusion_matrix": matrix,
        "test_metrics": test_metrics,
        "reference_metrics": reference_metrics,
        "test_minus_reference": comparisons,
        "fit_calls": 0,
        "models_compared": 1,
        "hyperparameter_search_performed": False,
        "threshold_search_performed": False,
        "test_prediction_calls": 1,
        "frozen_artifact_sha256_before": frozen_hashes_before,
        "frozen_artifact_sha256_after": frozen_hashes_after,
        "frozen_artifacts_unchanged": True,
    }
    np.savez_compressed(
        predictions_path,
        row_index=final_test_indices,
        y_true=y_test_final.astype(np.int8),
        probability=probabilities.astype(np.float32),
        prediction=predictions,
    )
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion_payload.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics_artifact": repo_relative(metrics_path),
            "predictions_artifact": repo_relative(predictions_path),
            "test_prediction_calls": 1,
        }
    )
    completion_path.write_text(
        json.dumps(completion_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("FINAL_TEST_METRICS", json.dumps(test_metrics, sort_keys=True), flush=True)
    print("CONFUSION_MATRIX", json.dumps(matrix, sort_keys=True), flush=True)
    print("PREVALENCE", json.dumps(prevalence, sort_keys=True), flush=True)
    print("TEST_PREDICTION_CALLS 1", flush=True)
    print("FIT_CALLS 0", flush=True)
    print("PRIMARY_DECISION_CHANGED 0", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data" / "weatherAUS_2026C1.csv"
    )
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "artifacts" / "oof_selection.json"
    )
    parser.add_argument(
        "--temporal-summary",
        type=Path,
        default=ROOT / "artifacts" / "temporal_robustness.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
