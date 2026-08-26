"""Verifica estaticamente el contrato de aislamiento del test final del notebook."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "TP_clasificacion_AA1.ipynb"
TEST_IDENTIFIERS = re.compile(
    r"\b(?:X_test_final(?:_raw|_processed|_pycaret)?|y_test_final(?:_encoded)?|df_test_final_raw)\b"
)
LEGACY_TEST_IDENTIFIERS = re.compile(r"\b(?:X_test|y_test|df_test|X_test_scaled)\b")


def source(cell):
    return "".join(cell.get("source", []))


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]

assert len(cells) == 263, "Cambio inesperado en la estructura del notebook"

protocol = source(cells[36])
assert '"name": "random_stratified"' in protocol
assert "FINAL_TEST_SIZE = 0.20" in protocol
assert "VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.20" in protocol
assert '"status": "implemented"' in protocol
assert '"approach": "expanding_window_5fold"' in protocol
assert '"decision_impact": "none"' in protocol
assert '"model_selection"' in protocol
assert '"hyperparameter_selection"' in protocol
assert '"threshold_selection"' in protocol
assert 'random_split_manifest.to_csv("random_split_manifest.csv"' in protocol
assert '"status": "complete"' in protocol
assert '"evaluation_count": 1' in protocol

# Antes de la evaluación final solo se permiten referencias de test para crearlo,
# conservarlo crudo y comprobar que no se solapa con desarrollo.
allowed_setup_cells = {36, 37, 38}
unexpected_test_references = []
for index, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    identifiers = sorted(set(TEST_IDENTIFIERS.findall(source(cell))))
    if identifiers and index not in allowed_setup_cells:
        unexpected_test_references.append((index, identifiers))

assert not unexpected_test_references, (
    "El test final aparece antes de la evaluacion final: "
    f"{unexpected_test_references}"
)

# No deben sobrevivir los nombres antiguos usados como banco general de evaluacion.
legacy_references = []
for index, cell in enumerate(cells[36:], start=36):
    if cell["cell_type"] != "code":
        continue
    identifiers = sorted(set(LEGACY_TEST_IDENTIFIERS.findall(source(cell))))
    if identifiers:
        legacy_references.append((index, identifiers))
assert not legacy_references, f"Persisten referencias al test anterior: {legacy_references}"

# La selección se reconstruye exclusivamente desde los artefactos OOF.
oof_load = source(cells[246])
assert 'Path("artifacts/oof_candidate_metrics.csv")' in oof_load
assert 'Path("artifacts/oof_selection.json")' in oof_load
assert 'selection_oof["primary_metric"] == "f1_positive"' in oof_load
assert 'selection_oof["final_test_features_used"] == 0' in oof_load
assert 'selection_oof["final_test_predictions_computed"] == 0' in oof_load

final = source(cells[253])
assert 'ganador_oof = selection_oof["winner"]' in final
assert 'df_comparacion["f1_positive"].max()' in final
assert 'Path(selection_oof["frozen_model_artifact"])' in final
assert "CONFIGURACIÓN SELECCIONADA MEDIANTE OOF" in final
assert not TEST_IDENTIFIERS.search(final)

selector = (ROOT / "scripts" / "oof_model_selection.py").read_text(
    encoding="utf-8"
)
assert "development_indices, final_test_indices = train_test_split(" in selector
assert "X_development = data.loc[development_indices]" in selector
assert '"final_test_features_used": 0' in selector
assert '"final_test_predictions_computed": 0' in selector
assert "final_test_index_set" in selector
assert "data.loc[final_test_indices]" not in selector

# Las secciones OOF, temporal y test solo reconstruyen resultados guardados.
final_section_source = "\n".join(
    source(cell) for cell in cells[246:] if cell["cell_type"] == "code"
)
for required_path in (
    "artifacts/oof_candidate_metrics.csv",
    "artifacts/oof_selection.json",
    "artifacts/temporal_robustness.json",
    "artifacts/temporal_fold_metrics.csv",
    "artifacts/final_test_metrics.json",
    "artifacts/final_test_evaluation.completed.json",
):
    assert required_path in final_section_source
for forbidden_call in (".fit(", ".predict(", ".predict_proba(", "subprocess"):
    assert forbidden_call not in final_section_source

print("OK: protocolo aleatorio primario reproducible")
print("OK: evaluacion temporal implementada como analisis secundario no decisional")
print("OK: train, validation y test final usan identificadores separados")
print("OK: modelo, hiperparametros y umbral se leen del manifiesto OOF")
print("OK: el selector no materializa features ni predicciones del test final")
print("OK: el test se evaluo una sola vez despues de completar la seleccion OOF")
print("OK: las secciones finales leen artefactos sin ejecutar fit ni predicciones")
