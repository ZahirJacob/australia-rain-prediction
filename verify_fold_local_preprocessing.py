"""Prueba estatica y dinamica del contrato fold-local de prioridad 2."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from weather_preprocessing import build_weather_preprocessor


ROOT = Path(__file__).parent
NOTEBOOK = ROOT / "TP_clasificacion_AA1.ipynb"
DATASET = ROOT / "weatherAUS_2026C1.csv"


def source(cell):
    return "".join(cell.get("source", []))


# 1) Todas las celdas Python siguen siendo sintacticamente validas.
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
cells = notebook["cells"]
parsed_code_cells = 0
for index, cell in enumerate(cells):
    if cell["cell_type"] != "code" or index == 1:  # %pip es una magic de Jupyter
        continue
    ast.parse(source(cell), filename=f"cell_{index}")
    parsed_code_cells += 1

# 2) Contratos estaticos de las tres rutas de entrenamiento.
assert 'drop(columns=["RainTomorrow", "Region"])' in source(cells[35])
assert "build_weather_preprocessor(" in source(cells[79])
assert ".fit(" not in source(cells[79])

manual_preprocessing = "\n".join(source(cell) for cell in cells[80:105])
for forbidden in (
    "StandardScaler().fit",
    "KNNImputer(",
    "medianas_region_season = X_train.groupby",
    "modas_region_season_docker =",
):
    assert forbidden not in manual_preprocessing, forbidden

grid = source(cells[160])
assert '("preprocesamiento", clone(preprocesamiento))' in grid
assert "GridSearchCV(" in grid
assert "grid_search.fit(X_train, y_train)" in grid

pycaret_setup = source(cells[192]) + source(cells[194])
assert "prepare_pycaret_frame(X_train)" in pycaret_setup
assert "normalize=True" in pycaret_setup
assert "fix_imbalance=True" in pycaret_setup
assert "fold_strategy=StratifiedKFold(" in pycaret_setup
assert "data=df_pycaret_train" in pycaret_setup

neural = source(cells[209])
split_position = neural.index("train_test_split(")
fit_position = neural.index("fit_transform(X_train_nn_raw)")
valid_position = neural.index("transform(X_valid_nn_raw)")
external_position = neural.index("transform(X_validation)")
assert split_position < fit_position < valid_position < external_position
assert "fit_indices_nn.isdisjoint(X_valid_nn_raw.index)" in neural
assert "fit_indices_nn.isdisjoint(X_validation.index)" in neural

all_modeling_code = "\n".join(source(cell) for cell in cells[35:253])
assert "fit_transform(X_validation" not in all_modeling_code
assert "fit(X_validation" not in all_modeling_code

oof_selector = (ROOT / "oof_model_selection.py").read_text(encoding="utf-8")
oof_fit = oof_selector.index("X_subtrain_transformed = preprocessor.fit_transform")
oof_validation = oof_selector.index(
    "X_validation_transformed = preprocessor.transform(X_fold_validation)"
)
assert oof_fit < oof_validation
assert "learned_indices == set(X_subtrain.index)" in oof_selector
assert "learned_indices.isdisjoint(expected_validation_index)" in oof_selector
assert "learned_indices.isdisjoint(final_test_index_set)" in oof_selector
assert "smote.fit_resample(" in oof_selector

# 3) Evidencia dinamica sobre indices reales del repositorio.
data = pd.read_csv(DATASET)
data = data.dropna(subset=["RainTomorrow"])
sample_size = min(600, int(data["RainTomorrow"].value_counts().min()))
sample = (
    data.groupby("RainTomorrow", group_keys=False)
    .sample(n=sample_size, random_state=42)
    .sort_index()
)
X = sample.drop(
    columns=["Unnamed: 0", "RainTomorrow", "RainfallTomorrow"], errors="ignore"
)
y = sample["RainTomorrow"].map({"No": 0, "Yes": 1})

# Coordenadas deterministicas de prueba: permiten ejecutar KMeans sin red.
locations = sorted(X["Location"].dropna().unique())
coordinates = {
    location: (
        float(np.cos(2 * np.pi * position / len(locations))),
        float(np.sin(2 * np.pi * position / len(locations))),
    )
    for position, location in enumerate(locations)
}

X_development, X_external_test, y_development, y_external_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(cv.split(X_development, y_development))
candidate = Pipeline(
    steps=[
        ("preprocesamiento", build_weather_preprocessor(coordinates, random_state=42)),
        ("clasificador", LogisticRegression(max_iter=300, random_state=42)),
    ]
)
result = cross_validate(
    candidate,
    X_development,
    y_development,
    cv=splits,
    scoring="f1",
    return_estimator=True,
    n_jobs=1,
    error_score="raise",
)

fold_evidence = []
for fold_number, ((train_positions, valid_positions), estimator) in enumerate(
    zip(splits, result["estimator"]), start=1
):
    expected_train = set(X_development.iloc[train_positions].index)
    expected_valid = set(X_development.iloc[valid_positions].index)
    transformer = estimator.named_steps["preprocesamiento"].named_steps[
        "weather_features"
    ]
    observed_fit = set(transformer.fit_indices_)
    assert observed_fit == expected_train
    assert observed_fit.isdisjoint(expected_valid)
    assert transformer.fit_row_count_ == len(expected_train)

    centers_before = transformer.kmeans_.cluster_centers_.copy()
    means_before = transformer.knn_scaler_.mean_.copy()
    transformed_valid = estimator.named_steps["preprocesamiento"].transform(
        X_development.iloc[valid_positions]
    )
    assert observed_fit == set(transformer.fit_indices_)
    np.testing.assert_allclose(centers_before, transformer.kmeans_.cluster_centers_)
    np.testing.assert_allclose(means_before, transformer.knn_scaler_.mean_)
    assert transformed_valid.shape[0] == len(expected_valid)
    assert np.isfinite(transformed_valid).all()
    fold_evidence.append((fold_number, len(expected_train), len(expected_valid), 0))

# Refit permitido sobre desarrollo; el test externo solo recibe transform/predict.
final_candidate = clone(candidate).fit(X_development, y_development)
final_transformer = final_candidate.named_steps["preprocesamiento"].named_steps[
    "weather_features"
]
final_fit_indices = set(final_transformer.fit_indices_)
centers_before_test = final_transformer.kmeans_.cluster_centers_.copy()
means_before_test = final_transformer.knn_scaler_.mean_.copy()
_ = final_candidate.predict(X_external_test)
assert final_fit_indices == set(final_transformer.fit_indices_)
assert final_fit_indices.isdisjoint(X_external_test.index)
np.testing.assert_allclose(centers_before_test, final_transformer.kmeans_.cluster_centers_)
np.testing.assert_allclose(means_before_test, final_transformer.knn_scaler_.mean_)

print(
    f"OK: las {parsed_code_cells} celdas Python "
    "(excepto la magic %pip) tienen sintaxis valida"
)
print("OK: no queda preprocesamiento aprendido manual antes de GridSearchCV")
print("OK: GridSearchCV contiene el preprocesador completo dentro del Pipeline")
print("OK: PyCaret recibe datos sin preprocesamiento aprendido externo")
print("OK: la red divide datos crudos antes de ejecutar fit_transform")
for fold_number, train_count, valid_count, overlap in fold_evidence:
    print(
        f"OK fold {fold_number}: fit={train_count}, valid={valid_count}, "
        f"interseccion={overlap}, validacion=solo transform"
    )
print(
    "OK test externo: fit=desarrollo, "
    f"test={len(X_external_test)}, interseccion=0, test=solo transform/predict"
)
