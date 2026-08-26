"""Verificacion sin nueva inferencia de la evaluacion unica del test final."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


selection = json.loads((ARTIFACTS / "oof_selection.json").read_text(encoding="utf-8"))
temporal = json.loads(
    (ARTIFACTS / "temporal_robustness.json").read_text(encoding="utf-8")
)
result = json.loads((ARTIFACTS / "final_test_metrics.json").read_text(encoding="utf-8"))
completion = json.loads(
    (ARTIFACTS / "final_test_evaluation.completed.json").read_text(encoding="utf-8")
)
stored = np.load(ARTIFACTS / "final_test_predictions.npz")

assert result["protocol"] == "single_final_test_evaluation"
assert result["status"] == completion["status"] == "complete"
assert result["decision_impact"] == "none"
assert result["primary_metric"] == "f1_positive"
assert result["model_candidate_id"] == selection["winner"]["candidate_id"]
assert result["model_parameters"] == selection["winner"]["parameters"]
assert result["threshold"] == selection["winner"]["threshold"]
assert result["threshold_source"] == "priority_3_oof_selection"
assert result["fit_calls"] == 0
assert result["models_compared"] == 1
assert result["hyperparameter_search_performed"] is False
assert result["threshold_search_performed"] is False
assert result["test_prediction_calls"] == completion["test_prediction_calls"] == 1

row_index = stored["row_index"]
y_true = stored["y_true"]
probability = stored["probability"]
prediction = stored["prediction"]
threshold = result["threshold"]
assert len(row_index) == len(np.unique(row_index)) == result["test_rows"]
assert np.isfinite(probability).all()
assert np.array_equal(prediction, (probability >= threshold).astype(np.int8))

recomputed_metrics = {
    "f1_positive": float(f1_score(y_true, prediction, pos_label=1, zero_division=0)),
    "precision": float(
        precision_score(y_true, prediction, pos_label=1, zero_division=0)
    ),
    "recall": float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
    "pr_auc": float(average_precision_score(y_true, probability)),
    "roc_auc": float(roc_auc_score(y_true, probability)),
}
for name, value in recomputed_metrics.items():
    assert np.isclose(value, result["test_metrics"][name], rtol=0, atol=1e-8)

tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
assert result["confusion_matrix"]["matrix"] == [
    [int(tn), int(fp)],
    [int(fn), int(tp)],
]
assert result["prevalence"]["positive_rows"] == int(y_true.sum())
assert result["prevalence"]["negative_rows"] == int(len(y_true) - y_true.sum())
assert np.isclose(result["prevalence"]["positive_rate"], y_true.mean(), atol=1e-12)

data = pd.read_csv(
    ROOT / "data" / "weatherAUS_2026C1.csv", usecols=["RainTomorrow"]
)
data = data.dropna(subset=["RainTomorrow"])
y_all = data["RainTomorrow"].map({"No": 0, "Yes": 1})
development_index, expected_test_index = train_test_split(
    data.index.to_numpy(),
    test_size=0.20,
    random_state=42,
    stratify=y_all,
)
assert set(row_index) == set(expected_test_index)
assert set(row_index).isdisjoint(set(development_index))
assert len(row_index) == selection["final_test_rows_reserved"]

current_hashes = {
    "selection": sha256_file(ARTIFACTS / "oof_selection.json"),
    "model": sha256_file(ROOT / selection["frozen_model_artifact"]),
    "preprocessor": sha256_file(ROOT / selection["frozen_preprocessor_artifact"]),
}
assert result["frozen_artifact_sha256_before"] == current_hashes
assert result["frozen_artifact_sha256_after"] == current_hashes
assert result["frozen_artifacts_unchanged"] is True
assert current_hashes == temporal["frozen_artifact_sha256_after"]

for reference_name, reference_values in result["reference_metrics"].items():
    for metric_name, test_value in result["test_metrics"].items():
        expected_delta = test_value - reference_values[metric_name]
        assert np.isclose(
            result["test_minus_reference"][reference_name][metric_name],
            expected_delta,
            rtol=0,
            atol=1e-12,
        )

implementation = (ROOT / "scripts" / "final_test_evaluation.py").read_text(
    encoding="utf-8"
)
assert implementation.count("X_test_final = data.loc[final_test_indices]") == 1
assert implementation.count("preprocessor.transform(X_test_final)") == 1
assert implementation.count("model.predict(") == 1
for forbidden in (
    ".fit(",
    "select_threshold",
    "GridSearchCV",
    "compare_models",
    "tune_model",
    "candidate_catalog",
):
    assert forbidden not in implementation
assert 'with completion_path.open("x"' in implementation

notebook = json.loads((ROOT / "TP_clasificacion_AA1.ipynb").read_text(encoding="utf-8"))
notebook_source = "\n".join(
    "".join(cell.get("source", [])) for cell in notebook["cells"]
)
assert 'Path("artifacts/final_test_metrics.json")' in notebook_source
assert 'resultado_test_final["decision_impact"] == "none"' in notebook_source
assert 'resultado_test_final["test_prediction_calls"] == 1' in notebook_source

print("OK: las 28.431 predicciones almacenadas corresponden exactamente al test final")
print("OK: metricas, prevalencia y matriz reconstruidas sin nueva inferencia")
print("OK: una apertura, un transform, un predict y cero fit/seleccion/busquedas")
print("OK: modelo, hiperparametros y threshold coinciden con prioridad 3")
print("OK: hashes de seleccion, modelo y preprocesador permanecen sin cambios")
print("OK: comparaciones OOF/temporales son descriptivas y decision_impact=none")
