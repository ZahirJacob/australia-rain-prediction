"""Inferencia con el bundle final validado (NN + preprocesador + threshold OOF)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weather_preprocessing import MODE_COLUMNS, NUMERIC_COLUMNS  # noqa: E402


LOGGER = logging.getLogger("weather_inference")
REQUIRED_COLUMNS = frozenset(("Date", "Location", *NUMERIC_COLUMNS, *MODE_COLUMNS))


def load_frozen_bundle(artifacts_dir):
    """Carga exclusivamente los tres componentes congelados de prioridad 3."""

    artifacts_dir = Path(artifacts_dir).resolve()
    selection_path = artifacts_dir / "oof_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner = selection["winner"]
    if winner["candidate_id"] != "nn_config_5":
        raise RuntimeError("El manifiesto no contiene el ganador nn_config_5")
    if selection["frozen_model_format"] != "keras_plus_joblib_preprocessor":
        raise RuntimeError("Formato de bundle final no soportado")

    project_root = artifacts_dir.parent
    model_path = project_root / selection["frozen_model_artifact"]
    preprocessor_path = project_root / selection["frozen_preprocessor_artifact"]
    if not model_path.is_file() or not preprocessor_path.is_file():
        raise FileNotFoundError("Faltan artefactos congelados de inferencia")

    preprocessor = joblib.load(preprocessor_path)
    model = keras.models.load_model(model_path)
    threshold = float(winner["threshold"])
    return preprocessor, model, threshold, selection


def validate_input(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("La entrada debe ser un pandas.DataFrame")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    if frame.empty:
        raise ValueError("El archivo de entrada no contiene filas")


def predict_dataframe(frame, artifacts_dir):
    """Devuelve etiqueta y P(lluvia) sin ajustar ningun componente."""

    validate_input(frame)
    preprocessor, model, threshold, selection = load_frozen_bundle(artifacts_dir)
    transformed = preprocessor.transform(frame)
    probabilities = model.predict(transformed, batch_size=1024, verbose=0).reshape(-1)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("El modelo produjo probabilidades no finitas")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise RuntimeError("El modelo produjo probabilidades fuera de [0, 1]")

    positive = probabilities >= threshold
    output = pd.DataFrame(
        {
            "prediccion": np.where(positive, "Llueve", "No llueve"),
            "Probabilidad": probabilities,
        },
        index=frame.index,
    )
    metadata = {
        "candidate_id": selection["winner"]["candidate_id"],
        "threshold": threshold,
        "probability_semantics": "P(RainTomorrow=Yes)",
    }
    return output, metadata


def run_cli(args):
    frame = pd.read_csv(args.input)
    LOGGER.info("Entrada cargada: %s filas", len(frame))
    output, metadata = predict_dataframe(frame, args.artifacts_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    LOGGER.info(
        "Salida guardada: %s; modelo=%s threshold=%.12f",
        args.output,
        metadata["candidate_id"],
        metadata["threshold"],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/files/input.csv"))
    parser.add_argument("--output", type=Path, default=Path("/files/output.csv"))
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_cli(parse_args())
