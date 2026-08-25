"""Verificacion reproducible de la seleccion OOF de prioridad 3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oof_model_selection import metrics_for, select_threshold


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

selection = json.loads((ARTIFACTS / "oof_selection.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(ARTIFACTS / "oof_candidate_metrics.csv")
audit = pd.read_csv(ARTIFACTS / "oof_fold_audit.csv")
oof = np.load(ARTIFACTS / "oof_predictions.npz")

assert selection["protocol"] == "single_stratified_5fold_oof"
assert selection["primary_metric"] == "f1_positive"
assert selection["complementary_metrics"] == [
    "precision",
    "recall",
    "pr_auc",
    "roc_auc",
]
assert selection["candidate_count"] == len(metrics) == 23
assert metrics["candidate_id"].is_unique

development_index = oof["development_index"]
y_true = oof["y_true"]
fold = oof["fold"]
assert len(development_index) == selection["development_rows"]
assert len(np.unique(development_index)) == len(development_index)
unique_folds, fold_counts = np.unique(fold, return_counts=True)
assert np.array_equal(unique_folds, np.arange(1, 6))
assert fold_counts.max() - fold_counts.min() <= 1

metric_names = ("f1_positive", "precision", "recall", "pr_auc", "roc_auc")
for row in metrics.itertuples(index=False):
    probabilities = oof[row.candidate_id]
    assert probabilities.shape == y_true.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    threshold = select_threshold(y_true, probabilities)
    recomputed = metrics_for(y_true, probabilities, threshold)
    assert np.isclose(threshold, row.threshold, rtol=0, atol=1e-12)
    for name in metric_names:
        assert np.isclose(recomputed[name], getattr(row, name), rtol=0, atol=1e-12)

expected_winner = metrics.sort_values(
    ["f1_positive", "candidate_id"], ascending=[False, True]
).iloc[0]
winner = selection["winner"]
assert winner["candidate_id"] == expected_winner["candidate_id"]
assert np.isclose(winner["threshold"], expected_winner["threshold"], atol=1e-12)
for name in metric_names:
    assert np.isclose(winner["oof_metrics"][name], expected_winner[name], atol=1e-12)

assert len(audit) == 5
assert set(audit["fold"]) == set(range(1, 6))
assert audit[
    ["fit_validation_overlap", "fit_test_overlap", "validation_test_overlap"]
].to_numpy().sum() == 0
assert audit["fit_rows"].sum() == 4 * selection["development_rows"]
assert audit["validation_rows"].sum() == selection["development_rows"]

assert selection["final_test_features_used"] == 0
assert selection["final_test_predictions_computed"] == 0
assert selection["final_test_rows_reserved"] > 0
assert (ROOT / selection["frozen_model_artifact"]).is_file()
if selection["frozen_model_format"] == "keras_plus_joblib_preprocessor":
    assert (ROOT / selection["frozen_preprocessor_artifact"]).is_file()

notebook = json.loads((ROOT / "TP_clasificacion_AA1.ipynb").read_text(encoding="utf-8"))
notebook_source = "\n".join(
    "".join(cell.get("source", [])) for cell in notebook["cells"]
)
assert 'Path("artifacts/oof_selection.json")' in notebook_source
assert 'Path("artifacts/oof_candidate_metrics.csv")' in notebook_source
assert 'selection_oof["winner"]' in notebook_source
assert 'selection_oof["primary_metric"] == "f1_positive"' in notebook_source
assert "MODELO, HIPERPARAMETROS Y THRESHOLD CONGELADOS" in notebook_source

print("OK: 23 candidatos tienen probabilidades OOF completas y finitas")
print("OK: cada fila de desarrollo pertenece exactamente a uno de 5 folds")
print("OK: umbrales y metricas fueron reconstruidos desde las probabilidades OOF")
print("OK: el ganador es el maximo F1 positivo; las otras metricas son complementarias")
print("OK: preprocessing no solapa validation ni test en ningun fold")
print("OK: el selector uso 0 features y produjo 0 predicciones del test final")
print(f"GANADOR: {winner['candidate_id']} threshold={winner['threshold']:.12f}")
