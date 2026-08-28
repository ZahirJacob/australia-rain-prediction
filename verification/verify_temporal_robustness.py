"""Verifica la evaluacion temporal expansiva secundaria de prioridad 4."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from temporal_robustness_evaluation import metrics_for  # noqa: E402


ARTIFACTS = ROOT / "artifacts"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_hashes_match(recorded, current, label):
    """Same check as before, but the failure names the artifact whose hash differs."""
    mismatched = {k: (recorded.get(k), current.get(k)) for k in set(recorded) | set(current) if recorded.get(k) != current.get(k)}
    assert not mismatched, (
        f"{label}: hash distinto para {sorted(mismatched)} "
        + "; ".join(f"{k}: registrado={r[:16] if r else None} actual={c[:16] if c else None}" for k, (r, c) in mismatched.items())
    )


selection = json.loads((ARTIFACTS / "oof_selection.json").read_text(encoding="utf-8"))
summary = json.loads(
    (ARTIFACTS / "temporal_robustness.json").read_text(encoding="utf-8")
)
folds = pd.read_csv(ARTIFACTS / "temporal_fold_metrics.csv")
predictions = np.load(ARTIFACTS / "temporal_predictions.npz")

assert summary["protocol"] == "expanding_window_temporal_robustness"
assert summary["role"] == "secondary_analysis_only"
assert summary["decision_impact"] == "none"
assert summary["fold_count"] == len(folds) == 5
assert summary["models_compared"] == 1
assert summary["winner_candidate_id"] == selection["winner"]["candidate_id"]
assert summary["winner_parameters"] == selection["winner"]["parameters"]
assert summary["threshold"] == selection["winner"]["threshold"]
assert summary["threshold_source"] == "priority_3_oof_selection"
assert summary["threshold_optimized_temporally"] is False
assert summary["development_index_hash"] == selection["development_index_hash"]
assert summary["final_test_features_used"] == 0
assert summary["final_test_predictions_computed"] == 0

assert list(folds["fold"]) == [1, 2, 3, 4, 5]
for column in (
    "train_validation_overlap",
    "train_test_overlap",
    "validation_test_overlap",
):
    assert (folds[column] == 0).all()
assert folds["chronology_valid"].all()
assert (
    pd.to_datetime(folds["train_end"])
    < pd.to_datetime(folds["validation_start"])
).all()
assert (folds["preprocessor_fit_rows"] == folds["train_rows"]).all()
assert folds["train_rows"].is_monotonic_increasing
for fold_number in range(1, 5):
    assert folds.loc[fold_number, "train_rows"] == (
        folds.loc[fold_number - 1, "train_rows"]
        + folds.loc[fold_number - 1, "validation_rows"]
    )

row_index = predictions["row_index"]
y_true = predictions["y_true"]
probability = predictions["probability"]
prediction_fold = predictions["fold"]
assert len(row_index) == len(np.unique(row_index))
assert len(row_index) == summary["temporal_evaluation_rows"]
assert len(row_index) + summary["initial_training_rows_not_scored"] == summary[
    "development_rows"
]
assert np.isfinite(probability).all()
assert ((probability >= 0) & (probability <= 1)).all()

metric_names = ("f1_positive", "precision", "recall", "pr_auc", "roc_auc")
for fold_number in range(1, 6):
    mask = prediction_fold == fold_number
    recomputed = metrics_for(y_true[mask], probability[mask], summary["threshold"])
    stored = folds.loc[folds["fold"] == fold_number].iloc[0]
    assert mask.sum() == stored["validation_rows"]
    for name in metric_names:
        assert np.isclose(recomputed[name], stored[name], rtol=0, atol=1e-8)

pooled = metrics_for(y_true, probability, summary["threshold"])
for name in metric_names:
    assert np.isclose(
        pooled[name], summary["temporal_pooled_metrics"][name], rtol=0, atol=1e-8
    )
    assert np.isclose(
        folds[name].mean(), summary["temporal_macro_mean"][name], rtol=0, atol=1e-12
    )
    assert np.isclose(
        folds[name].std(ddof=1),
        summary["temporal_macro_std"][name],
        rtol=0,
        atol=1e-12,
    )

data = pd.read_csv(ROOT / "data" / "weatherAUS_2026C1.csv")
data = data.dropna(subset=["RainTomorrow"])
y_all = data["RainTomorrow"].map({"No": 0, "Yes": 1})
development_index, final_test_index = train_test_split(
    data.index.to_numpy(),
    test_size=0.20,
    random_state=42,
    stratify=y_all,
)
assert set(row_index).issubset(set(development_index))
assert set(row_index).isdisjoint(set(final_test_index))
assert len(final_test_index) == summary["final_test_rows_reserved"]

current_hashes = {
    "selection": sha256_file(ARTIFACTS / "oof_selection.json"),
    "model": sha256_file(ROOT / selection["frozen_model_artifact"]),
    "preprocessor": sha256_file(ROOT / selection["frozen_preprocessor_artifact"]),
}
assert_hashes_match(summary["frozen_artifact_sha256_before"], current_hashes, "temporal_robustness.frozen_artifact_sha256_before")
assert_hashes_match(summary["frozen_artifact_sha256_after"], current_hashes, "temporal_robustness.frozen_artifact_sha256_after")
assert summary["frozen_artifacts_unchanged"] is True

implementation = (ROOT / "scripts" / "temporal_robustness_evaluation.py").read_text(
    encoding="utf-8"
)
assert "select_threshold" not in implementation
assert 'threshold = float(winner["threshold"])' in implementation
assert "date_blocks = np.array_split(unique_dates, N_TEMPORAL_FOLDS + 1)" in implementation
assert "X_train[\"Date\"].max() < X_validation[\"Date\"].min()" in implementation
assert "learned_indices == train_index" in implementation

notebook = json.loads((ROOT / "TP_clasificacion_AA1.ipynb").read_text(encoding="utf-8"))
notebook_source = "\n".join(
    "".join(cell.get("source", [])) for cell in notebook["cells"]
)
assert '"status": "implemented"' in notebook_source
assert 'Path("artifacts/temporal_robustness.json")' in notebook_source
assert 'resumen_temporal["decision_impact"] == "none"' in notebook_source
assert 'resumen_temporal["threshold"] == selection_oof["winner"]["threshold"]' in notebook_source

print("OK: cinco ventanas temporales expansivas, contiguas y cronologicas")
print("OK: preprocessing y SMOTE se ajustaron solo con pasado en cada fold")
print("OK: metricas por fold y pooled reconstruidas desde predicciones temporales")
print("OK: candidato, hiperparametros y threshold coinciden con prioridad 3")
print("OK: hashes del manifiesto, modelo y preprocesador permanecen sin cambios")
print("OK: predicciones temporales no contienen ninguna fila del test final")
