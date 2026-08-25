"""Evaluacion temporal expansiva secundaria del ganador congelado en prioridad 3.

Esta rutina no selecciona modelos, hiperparametros ni threshold. Reentrena clones
de la configuracion ganadora en ventanas de pasado creciente y evalua siempre el
bloque cronologico inmediatamente posterior con el threshold OOF ya congelado.
El test aleatorio final permanece reservado y no recibe predicciones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from oof_model_selection import build_keras_model
from weather_preprocessing import build_weather_preprocessor


RANDOM_STATE = 42
FINAL_TEST_SIZE = 0.20
N_TEMPORAL_FOLDS = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(index):
    payload = ",".join(map(str, sorted(index))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def metrics_for(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "f1_positive": float(
            f1_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def date_string(value):
    return pd.Timestamp(value).date().isoformat()


def run(args):
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_path = args.selection.resolve()
    selection_before = selection_path.read_bytes()
    selection = json.loads(selection_before.decode("utf-8"))
    winner = selection["winner"]
    if winner["family"] != "NeuralNetwork":
        raise RuntimeError("La evaluacion implementada espera el ganador NN congelado")

    threshold = float(winner["threshold"])
    frozen_paths = {
        "selection": selection_path,
        "model": Path(selection["frozen_model_artifact"]).resolve(),
        "preprocessor": Path(selection["frozen_preprocessor_artifact"]).resolve(),
    }
    frozen_hashes_before = {
        name: sha256_file(path) for name, path in frozen_paths.items()
    }

    data = pd.read_csv(args.dataset)
    data = data.dropna(subset=["RainTomorrow"])
    data = data.drop(columns=["Unnamed: 0", "RainfallTomorrow"], errors="ignore")
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
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
    X_development = data.loc[development_indices].drop(columns="RainTomorrow")
    y_development = y_all.loc[development_indices]
    if index_hash(X_development.index) != selection["development_index_hash"]:
        raise RuntimeError("El conjunto de desarrollo no coincide con prioridad 3")
    assert set(X_development.index).isdisjoint(final_test_index_set)

    coordinates = {
        key: tuple(value)
        for key, value in json.loads(args.coordinates.read_text(encoding="utf-8")).items()
    }
    unique_dates = np.sort(X_development["Date"].unique())
    date_blocks = np.array_split(unique_dates, N_TEMPORAL_FOLDS + 1)
    if any(len(block) == 0 for block in date_blocks):
        raise RuntimeError("No hay suficientes fechas para cinco folds temporales")

    fold_rows = []
    prediction_indices = []
    prediction_y = []
    prediction_probabilities = []
    prediction_folds = []

    for fold_number in range(1, N_TEMPORAL_FOLDS + 1):
        train_dates = np.concatenate(date_blocks[:fold_number])
        validation_dates = date_blocks[fold_number]
        train_mask = X_development["Date"].isin(train_dates)
        validation_mask = X_development["Date"].isin(validation_dates)
        X_train = X_development.loc[train_mask]
        y_train = y_development.loc[X_train.index]
        X_validation = X_development.loc[validation_mask]
        y_validation = y_development.loc[X_validation.index]

        train_index = set(X_train.index)
        validation_index = set(X_validation.index)
        assert train_index.isdisjoint(validation_index)
        assert train_index.isdisjoint(final_test_index_set)
        assert validation_index.isdisjoint(final_test_index_set)
        assert X_train["Date"].max() < X_validation["Date"].min()

        preprocessor = build_weather_preprocessor(
            coordinates,
            random_state=RANDOM_STATE,
        )
        X_train_transformed = preprocessor.fit_transform(X_train)
        learned_indices = set(
            preprocessor.named_steps["weather_features"].fit_indices_
        )
        assert learned_indices == train_index
        assert learned_indices.isdisjoint(validation_index)
        assert learned_indices.isdisjoint(final_test_index_set)
        X_validation_transformed = preprocessor.transform(X_validation)

        smote = SMOTE(random_state=RANDOM_STATE + fold_number)
        X_balanced, y_balanced = smote.fit_resample(
            X_train_transformed,
            y_train.to_numpy(),
        )
        model = build_keras_model(
            X_balanced.shape[1],
            winner["parameters"],
            seed=RANDOM_STATE + fold_number,
        )
        model.fit(
            X_balanced,
            y_balanced,
            epochs=winner["parameters"]["epochs"],
            batch_size=winner["parameters"]["batch_size"],
            verbose=0,
            shuffle=True,
        )
        probabilities = model.predict(
            X_validation_transformed,
            batch_size=1024,
            verbose=0,
        ).reshape(-1)
        fold_metrics = metrics_for(y_validation.to_numpy(), probabilities, threshold)

        fold_rows.append(
            {
                "fold": fold_number,
                "train_start": date_string(X_train["Date"].min()),
                "train_end": date_string(X_train["Date"].max()),
                "validation_start": date_string(X_validation["Date"].min()),
                "validation_end": date_string(X_validation["Date"].max()),
                "train_unique_dates": int(X_train["Date"].nunique()),
                "validation_unique_dates": int(X_validation["Date"].nunique()),
                "train_rows": len(X_train),
                "validation_rows": len(X_validation),
                "balanced_train_rows": len(y_balanced),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_validation.mean()),
                "threshold": threshold,
                **fold_metrics,
                "train_validation_overlap": 0,
                "train_test_overlap": 0,
                "validation_test_overlap": 0,
                "chronology_valid": True,
                "preprocessor_fit_rows": len(learned_indices),
                "preprocessor_fit_index_hash": index_hash(learned_indices),
            }
        )
        prediction_indices.append(X_validation.index.to_numpy())
        prediction_y.append(y_validation.to_numpy())
        prediction_probabilities.append(probabilities.astype(np.float32))
        prediction_folds.append(
            np.full(len(X_validation), fold_number, dtype=np.int8)
        )
        print(
            f"TEMPORAL_FOLD {fold_number}/{N_TEMPORAL_FOLDS} "
            f"train={len(X_train)} validation={len(X_validation)} "
            f"f1={fold_metrics['f1_positive']:.6f}",
            flush=True,
        )

    fold_metrics_frame = pd.DataFrame(fold_rows)
    fold_metrics_path = output_dir / "temporal_fold_metrics.csv"
    fold_metrics_frame.to_csv(fold_metrics_path, index=False)

    temporal_index = np.concatenate(prediction_indices)
    temporal_y = np.concatenate(prediction_y)
    temporal_probabilities = np.concatenate(prediction_probabilities)
    temporal_folds = np.concatenate(prediction_folds)
    pooled_metrics = metrics_for(temporal_y, temporal_probabilities, threshold)
    metric_names = ("f1_positive", "precision", "recall", "pr_auc", "roc_auc")
    macro_mean = {
        name: float(fold_metrics_frame[name].mean()) for name in metric_names
    }
    macro_std = {
        name: float(fold_metrics_frame[name].std(ddof=1)) for name in metric_names
    }
    macro_min = {
        name: float(fold_metrics_frame[name].min()) for name in metric_names
    }
    np.savez_compressed(
        output_dir / "temporal_predictions.npz",
        row_index=temporal_index,
        y_true=temporal_y,
        probability=temporal_probabilities,
        fold=temporal_folds,
    )

    frozen_hashes_after = {
        name: sha256_file(path) for name, path in frozen_paths.items()
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("La evaluacion temporal altero un artefacto congelado")
    if selection_path.read_bytes() != selection_before:
        raise RuntimeError("La evaluacion temporal altero el manifiesto OOF")

    summary = {
        "protocol": "expanding_window_temporal_robustness",
        "role": "secondary_analysis_only",
        "decision_impact": "none",
        "random_state": RANDOM_STATE,
        "fold_count": N_TEMPORAL_FOLDS,
        "date_partition_rule": "six_contiguous_equal_unique_date_blocks",
        "development_rows": len(X_development),
        "development_index_hash": index_hash(X_development.index),
        "temporal_evaluation_rows": len(temporal_y),
        "initial_training_rows_not_scored": int(
            len(X_development) - len(temporal_y)
        ),
        "winner_source": str(selection_path.as_posix()),
        "winner_candidate_id": winner["candidate_id"],
        "winner_parameters": winner["parameters"],
        "threshold": threshold,
        "threshold_source": "priority_3_oof_selection",
        "threshold_optimized_temporally": False,
        "models_compared": 1,
        "primary_oof_metrics_reference": winner["oof_metrics"],
        "temporal_pooled_metrics": pooled_metrics,
        "temporal_macro_mean": macro_mean,
        "temporal_macro_std": macro_std,
        "temporal_worst_fold": macro_min,
        "frozen_artifact_sha256_before": frozen_hashes_before,
        "frozen_artifact_sha256_after": frozen_hashes_after,
        "frozen_artifacts_unchanged": True,
        "final_test_rows_reserved": len(final_test_indices),
        "final_test_features_used": 0,
        "final_test_predictions_computed": 0,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output_dir / "temporal_robustness.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(fold_metrics_frame.to_string(index=False), flush=True)
    print("TEMPORAL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("PRIMARY_DECISION_CHANGED 0", flush=True)
    print("TEST_FINAL_USED 0", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("weatherAUS_2026C1.csv"))
    parser.add_argument(
        "--selection", type=Path, default=Path("artifacts/oof_selection.json")
    )
    parser.add_argument(
        "--coordinates", type=Path, default=Path("location_coordinates.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
