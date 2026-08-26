"""Seleccion OOF de prioridad 3 sin acceso predictivo al test final.

El protocolo usa un unico StratifiedKFold de cinco particiones sobre desarrollo.
Cada transformacion aprendida y SMOTE se ajustan solo con el subtrain del fold.
El modelo, sus hiperparametros y el threshold se eligen exclusivamente por F1
de la clase positiva calculado sobre probabilidades OOF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weather_preprocessing import build_weather_preprocessor


RANDOM_STATE = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5
POSITIVE_LABEL = 1


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    parameters: dict
    kind: str = "sklearn"


def candidate_catalog():
    candidates = []
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1, 10, 100, 1000, 10000):
            candidates.append(
                Candidate(
                    candidate_id=f"lr_C={c_value:g}_class_weight={class_weight}",
                    family="LogisticRegression",
                    parameters={"C": c_value, "class_weight": class_weight},
                )
            )

    candidates.extend(
        [
            Candidate("dt_default", "DecisionTree", {}),
            Candidate("rf_default", "RandomForest", {"n_estimators": 100}),
            Candidate("ada_default", "AdaBoost", {"n_estimators": 50, "learning_rate": 1.0}),
            Candidate(
                "gbc_default",
                "GradientBoosting",
                {"n_estimators": 100, "learning_rate": 0.1},
            ),
        ]
    )

    neural_configs = (
        {"epochs": 50, "batch_size": 64, "learning_rate": 0.001, "dropout_rate": 0.2},
        {"epochs": 50, "batch_size": 64, "learning_rate": 0.001, "dropout_rate": 0.3},
        {"epochs": 50, "batch_size": 128, "learning_rate": 0.001, "dropout_rate": 0.3},
        {"epochs": 50, "batch_size": 64, "learning_rate": 0.0005, "dropout_rate": 0.3},
        {"epochs": 50, "batch_size": 128, "learning_rate": 0.0005, "dropout_rate": 0.4},
    )
    for number, config in enumerate(neural_configs, start=1):
        candidates.append(
            Candidate(
                candidate_id=f"nn_config_{number}",
                family="NeuralNetwork",
                parameters=config,
                kind="keras",
            )
        )
    return candidates


def build_sklearn_estimator(candidate):
    params = candidate.parameters
    if candidate.family == "LogisticRegression":
        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if candidate.family == "DecisionTree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if candidate.family == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if candidate.family == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )
    if candidate.family == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Familia sklearn desconocida: {candidate.family}")


def build_keras_model(input_shape, config, seed):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(26, activation="relu"),
            tf.keras.layers.Dropout(config["dropout_rate"]),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
    )
    return model


def positive_probability(estimator, features):
    probabilities = estimator.predict_proba(features)
    class_positions = np.flatnonzero(np.asarray(estimator.classes_) == POSITIVE_LABEL)
    if len(class_positions) != 1:
        raise RuntimeError(f"No se encontro la clase positiva en {estimator.classes_}")
    return probabilities[:, class_positions[0]]


def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    best_f1 = np.max(f1_values)
    tied = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0, atol=1e-12))
    closest_to_half = tied[np.argmin(np.abs(thresholds[tied] - 0.5))]
    return float(thresholds[closest_to_half])


def metrics_for(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "f1_positive": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "precision": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def index_hash(index):
    payload = ",".join(map(str, sorted(index))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def camel_case(value):
    output = ""
    for position, character in enumerate(value):
        if position > 0 and character.isupper() and value[position - 1].islower():
            output += " "
        output += character
    return output


def load_or_geocode_coordinates(locations, coordinates_path):
    if coordinates_path.exists():
        raw = json.loads(coordinates_path.read_text(encoding="utf-8"))
        coordinates = {key: tuple(value) for key, value in raw.items()}
    else:
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="climate_oof_priority3", timeout=10)
        geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1.2,
            max_retries=3,
            error_wait_seconds=5,
            swallow_exceptions=True,
            return_value_on_exception=None,
        )
        coordinates = {}
        for number, location_name in enumerate(sorted(locations), start=1):
            result = geocode(f"{camel_case(location_name)}, Australia")
            if result is None:
                raise RuntimeError(f"No se pudo geocodificar {location_name}")
            coordinates[location_name] = (result.latitude, result.longitude)
            print(f"COORD {number}/{len(locations)} {location_name}", flush=True)
        coordinates_path.write_text(
            json.dumps(coordinates, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    missing = set(locations).difference(coordinates)
    if missing:
        raise RuntimeError(f"Faltan coordenadas para: {sorted(missing)}")
    return coordinates


def fit_final_sklearn_pipeline(candidate, X_development, y_development, coordinates):
    pipeline = ImbalancedPipeline(
        steps=[
            (
                "preprocessing",
                build_weather_preprocessor(coordinates, random_state=RANDOM_STATE),
            ),
            ("balance", SMOTE(random_state=RANDOM_STATE)),
            ("model", build_sklearn_estimator(candidate)),
        ]
    )
    pipeline.fit(X_development, y_development)
    return pipeline


def run(args):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    final_test_index_set = set(final_test_indices.tolist())

    # A partir de aqui el selector recibe exclusivamente desarrollo. No se crea
    # ningun dataframe de features del test final ni se calculan predicciones.
    X_development = data.loc[development_indices].drop(columns="RainTomorrow")
    y_development = y_all.loc[development_indices]
    assert set(X_development.index).isdisjoint(final_test_index_set)

    coordinates = load_or_geocode_coordinates(
        X_development["Location"].dropna().unique(), args.coordinates
    )
    candidates = candidate_catalog()
    oof_probabilities = {
        candidate.candidate_id: np.full(len(X_development), np.nan, dtype=np.float32)
        for candidate in candidates
    }
    fold_assignments = np.full(len(X_development), -1, dtype=np.int8)
    audit_rows = []
    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    started = time.time()
    for fold_number, (train_positions, validation_positions) in enumerate(
        folds.split(X_development, y_development), start=1
    ):
        X_subtrain = X_development.iloc[train_positions]
        y_subtrain = y_development.iloc[train_positions]
        X_fold_validation = X_development.iloc[validation_positions]
        expected_validation_index = set(X_fold_validation.index)
        assert set(X_subtrain.index).isdisjoint(expected_validation_index)
        assert set(X_subtrain.index).isdisjoint(final_test_index_set)
        assert expected_validation_index.isdisjoint(final_test_index_set)

        preprocessor = build_weather_preprocessor(
            coordinates,
            random_state=RANDOM_STATE,
        )
        X_subtrain_transformed = preprocessor.fit_transform(X_subtrain)
        learned_indices = set(
            preprocessor.named_steps["weather_features"].fit_indices_
        )
        assert learned_indices == set(X_subtrain.index)
        assert learned_indices.isdisjoint(expected_validation_index)
        assert learned_indices.isdisjoint(final_test_index_set)
        X_validation_transformed = preprocessor.transform(X_fold_validation)

        smote = SMOTE(random_state=RANDOM_STATE + fold_number)
        X_balanced, y_balanced = smote.fit_resample(
            X_subtrain_transformed,
            y_subtrain.to_numpy(),
        )
        fold_assignments[validation_positions] = fold_number

        print(
            f"FOLD {fold_number}/{N_SPLITS} "
            f"fit={len(train_positions)} validation={len(validation_positions)} "
            f"balanced={len(y_balanced)}",
            flush=True,
        )

        for candidate_number, candidate in enumerate(candidates, start=1):
            candidate_started = time.time()
            if candidate.kind == "sklearn":
                estimator = build_sklearn_estimator(candidate)
                estimator.fit(X_balanced, y_balanced)
                probabilities = positive_probability(estimator, X_validation_transformed)
            else:
                model = build_keras_model(
                    X_balanced.shape[1],
                    candidate.parameters,
                    seed=RANDOM_STATE + fold_number,
                )
                model.fit(
                    X_balanced,
                    y_balanced,
                    epochs=candidate.parameters["epochs"],
                    batch_size=candidate.parameters["batch_size"],
                    verbose=0,
                    shuffle=True,
                )
                probabilities = model.predict(
                    X_validation_transformed,
                    batch_size=1024,
                    verbose=0,
                ).reshape(-1)

            oof_probabilities[candidate.candidate_id][validation_positions] = probabilities
            elapsed = time.time() - candidate_started
            print(
                f"CANDIDATE {candidate_number}/{len(candidates)} "
                f"{candidate.candidate_id} fold={fold_number} seconds={elapsed:.1f}",
                flush=True,
            )

        audit_rows.append(
            {
                "fold": fold_number,
                "fit_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "fit_validation_overlap": 0,
                "fit_test_overlap": 0,
                "validation_test_overlap": 0,
                "preprocessor_fit_index_hash": index_hash(learned_indices),
                "validation_index_hash": index_hash(expected_validation_index),
            }
        )

    assert np.all(fold_assignments > 0)
    metrics_rows = []
    y_values = y_development.to_numpy()
    for candidate in candidates:
        probabilities = oof_probabilities[candidate.candidate_id]
        if np.isnan(probabilities).any():
            raise RuntimeError(f"OOF incompleto para {candidate.candidate_id}")
        threshold = select_threshold(y_values, probabilities)
        row = {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "parameters": json.dumps(candidate.parameters, sort_keys=True),
            "threshold": threshold,
            **metrics_for(y_values, probabilities, threshold),
        }
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).sort_values(
        by=["f1_positive", "candidate_id"], ascending=[False, True]
    )
    winner_id = metrics.iloc[0]["candidate_id"]
    winner = next(candidate for candidate in candidates if candidate.candidate_id == winner_id)
    winner_threshold = float(metrics.iloc[0]["threshold"])

    metrics_path = output_dir / "oof_candidate_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "oof_fold_audit.csv", index=False)
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        development_index=X_development.index.to_numpy(),
        y_true=y_values,
        fold=fold_assignments,
        **{
            candidate.candidate_id: probabilities
            for candidate, probabilities in (
                (candidate, oof_probabilities[candidate.candidate_id])
                for candidate in candidates
            )
        },
    )

    selection = {
        "protocol": "single_stratified_5fold_oof",
        "primary_metric": "f1_positive",
        "complementary_metrics": ["precision", "recall", "pr_auc", "roc_auc"],
        "random_state": RANDOM_STATE,
        "development_rows": len(X_development),
        "development_index_hash": index_hash(X_development.index),
        "final_test_rows_reserved": len(final_test_indices),
        "final_test_features_used": 0,
        "final_test_predictions_computed": 0,
        "candidate_count": len(candidates),
        "winner": {
            "candidate_id": winner.candidate_id,
            "family": winner.family,
            "parameters": winner.parameters,
            "threshold": winner_threshold,
            "oof_metrics": {
                key: float(metrics.iloc[0][key])
                for key in ("f1_positive", "precision", "recall", "pr_auc", "roc_auc")
            },
        },
        "tie_break": "threshold_closest_to_0.5_then_candidate_id",
        "selection_bias_note": "OOF se usa para seleccionar y reportar; test final confirma despues.",
        "elapsed_seconds": time.time() - started,
    }

    if winner.kind == "sklearn":
        final_pipeline = fit_final_sklearn_pipeline(
            winner,
            X_development,
            y_development,
            coordinates,
        )
        model_path = output_dir / "selected_model.joblib"
        joblib.dump(final_pipeline, model_path)
        selection["frozen_model_artifact"] = str(model_path.as_posix())
        selection["frozen_model_format"] = "joblib_imblearn_pipeline"
    else:
        preprocessor = build_weather_preprocessor(coordinates, random_state=RANDOM_STATE)
        X_transformed = preprocessor.fit_transform(X_development)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X_transformed, y_development)
        model = build_keras_model(
            X_balanced.shape[1],
            winner.parameters,
            seed=RANDOM_STATE,
        )
        model.fit(
            X_balanced,
            y_balanced,
            epochs=winner.parameters["epochs"],
            batch_size=winner.parameters["batch_size"],
            verbose=0,
        )
        joblib.dump(preprocessor, output_dir / "selected_nn_preprocessor.joblib")
        model.save(output_dir / "selected_nn_model.keras")
        selection["frozen_model_artifact"] = "artifacts/selected_nn_model.keras"
        selection["frozen_preprocessor_artifact"] = (
            "artifacts/selected_nn_preprocessor.joblib"
        )
        selection["frozen_model_format"] = "keras_plus_joblib_preprocessor"

    selection_path = output_dir / "oof_selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False), flush=True)
    print("WINNER", json.dumps(selection["winner"], sort_keys=True), flush=True)
    print("TEST_FINAL_USED 0", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data" / "weatherAUS_2026C1.csv"
    )
    parser.add_argument(
        "--coordinates", type=Path, default=ROOT / "data" / "location_coordinates.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
