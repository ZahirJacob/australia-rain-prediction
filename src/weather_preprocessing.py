"""Preprocesamiento clonable y libre de fugas para el modelo meteorologico.

Las transformaciones que aprenden parametros implementan la API de scikit-learn.
Al incluir este objeto dentro de un Pipeline, GridSearchCV ajusta una copia
independiente exclusivamente con el subtrain de cada fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


NUMERIC_COLUMNS = (
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
)

CATEGORICAL_COLUMNS = (
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday",
    "Season",
    "Region",
)

MEDIAN_COLUMNS = (
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Sunshine",
    "WindGustSpeed",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Temp9am",
    "Temp3pm",
)

KNN_COLUMNS = ("Evaporation", "Cloud9am", "Cloud3pm")

MODE_COLUMNS = ("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")


def _season_from_month(month: pd.Series) -> pd.Series:
    conditions = (
        month.isin((12, 1, 2)),
        month.isin((3, 4, 5)),
        month.isin((6, 7, 8)),
        month.isin((9, 10, 11)),
    )
    values = ("Summer", "Autumn", "Winter", "Spring")
    return pd.Series(np.select(conditions, values, default=None), index=month.index)


def prepare_pycaret_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica solamente transformaciones deterministicas para PyCaret.

    La imputacion, codificacion, normalizacion y balanceo permanecen dentro del
    pipeline de PyCaret y, por lo tanto, se ajustan en el subtrain de cada fold.
    """

    frame = data.copy()
    frame = frame.drop(
        columns=("Unnamed: 0", "RainTomorrow", "RainfallTomorrow", "Region"),
        errors="ignore",
    )
    if "Date" not in frame:
        raise ValueError("Falta la columna requerida 'Date'.")
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Season"] = _season_from_month(dates.dt.month)
    return frame.drop(columns="Date")


class WeatherFeatureTransformer(BaseEstimator, TransformerMixin):
    """Ingenieria e imputacion meteorologica con estado aprendido en ``fit``."""

    def __init__(
        self,
        location_coordinates,
        n_regions=10,
        n_neighbors=5,
        random_state=42,
    ):
        self.location_coordinates = location_coordinates
        self.n_regions = n_regions
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @staticmethod
    def _require_dataframe(data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("WeatherFeatureTransformer requiere un pandas.DataFrame.")
        missing = {"Date", "Location"}.difference(data.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    def _coordinates_for(self, locations):
        coordinates = locations.map(self.location_coordinates)
        latitude = coordinates.map(
            lambda value: value[0] if isinstance(value, (tuple, list)) else np.nan
        )
        longitude = coordinates.map(
            lambda value: value[1] if isinstance(value, (tuple, list)) else np.nan
        )
        return pd.DataFrame(
            {"Latitude": latitude, "Longitude": longitude}, index=locations.index
        ).astype(float)

    def _prepare_base(self, data):
        frame = data.copy()
        frame = frame.drop(
            columns=("Unnamed: 0", "RainTomorrow", "RainfallTomorrow", "Region"),
            errors="ignore",
        )
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Season"] = _season_from_month(dates.dt.month)

        coordinates = self._coordinates_for(frame["Location"])
        valid_coordinates = coordinates.notna().all(axis=1)
        regions = pd.Series(self.region_default_, index=frame.index, dtype="object")
        if valid_coordinates.any():
            predicted = self.kmeans_.predict(coordinates.loc[valid_coordinates])
            regions.loc[valid_coordinates] = [f"Region_{value}" for value in predicted]
        frame["Region"] = regions
        return frame.drop(columns=("Date", "Location"), errors="ignore")

    @staticmethod
    def _group_values(table, frame):
        keys = pd.MultiIndex.from_frame(frame[["Region", "Season"]])
        values = table.reindex(keys)
        values.index = frame.index
        return values

    @staticmethod
    def _safe_mode(series):
        modes = series.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else np.nan

    def fit(self, data, y=None):
        self._require_dataframe(data)
        self.fit_indices_ = tuple(data.index.tolist())
        self.fit_row_count_ = len(data)

        coordinates = self._coordinates_for(data["Location"]).dropna().drop_duplicates()
        if coordinates.empty:
            raise ValueError("No hay coordenadas validas para ajustar las regiones.")
        cluster_count = min(int(self.n_regions), len(coordinates))
        self.kmeans_ = KMeans(
            n_clusters=cluster_count,
            random_state=self.random_state,
            n_init=10,
        ).fit(coordinates)
        training_regions = self.kmeans_.predict(coordinates)
        region_counts = pd.Series(training_regions).value_counts()
        self.region_default_ = f"Region_{region_counts.index[0]}"

        frame = self._prepare_base(data)
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        self.group_medians_ = frame.groupby(
            ["Region", "Season"], observed=True
        )[list(MEDIAN_COLUMNS)].median()
        self.global_medians_ = frame[list(MEDIAN_COLUMNS)].median()
        frame = self._apply_medians(frame)

        self.knn_scaler_ = StandardScaler().fit(frame[list(KNN_COLUMNS)])
        scaled_knn = pd.DataFrame(
            self.knn_scaler_.transform(frame[list(KNN_COLUMNS)]),
            columns=KNN_COLUMNS,
            index=frame.index,
        )
        self.global_knn_imputer_ = KNNImputer(
            n_neighbors=self.n_neighbors, keep_empty_features=True
        ).fit(scaled_knn)
        self.group_knn_imputers_ = {}
        for key, indices in frame.groupby(["Region", "Season"], observed=True).groups.items():
            self.group_knn_imputers_[key] = KNNImputer(
                n_neighbors=self.n_neighbors, keep_empty_features=True
            ).fit(scaled_knn.loc[indices])

        self.group_modes_ = frame.groupby(
            ["Region", "Season"], observed=True
        )[list(MODE_COLUMNS)].agg(self._safe_mode)
        self.global_modes_ = frame[list(MODE_COLUMNS)].agg(self._safe_mode)
        self.output_columns_ = np.asarray(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS, dtype=object)
        return self

    def _apply_medians(self, frame):
        group_values = self._group_values(self.group_medians_, frame)
        for column in MEDIAN_COLUMNS:
            frame[column] = (
                frame[column]
                .fillna(group_values[column])
                .fillna(self.global_medians_[column])
            )
        return frame

    def _apply_knn(self, frame):
        scaled = pd.DataFrame(
            self.knn_scaler_.transform(frame[list(KNN_COLUMNS)]),
            columns=KNN_COLUMNS,
            index=frame.index,
        )
        imputed = scaled.copy()
        for key, indices in frame.groupby(["Region", "Season"], observed=True).groups.items():
            imputer = self.group_knn_imputers_.get(key, self.global_knn_imputer_)
            imputed.loc[indices, list(KNN_COLUMNS)] = imputer.transform(
                scaled.loc[indices, list(KNN_COLUMNS)]
            )
        frame.loc[:, list(KNN_COLUMNS)] = self.knn_scaler_.inverse_transform(imputed)
        return frame

    def _apply_modes(self, frame):
        group_values = self._group_values(self.group_modes_, frame)
        for column in MODE_COLUMNS:
            frame[column] = (
                frame[column]
                .fillna(group_values[column])
                .fillna(self.global_modes_[column])
            )
        frame["Season"] = frame["Season"].fillna("Unknown")
        frame["Region"] = frame["Region"].fillna(self.region_default_)
        return frame

    def transform(self, data):
        check_is_fitted(
            self,
            attributes=(
                "kmeans_",
                "group_medians_",
                "knn_scaler_",
                "global_knn_imputer_",
                "group_modes_",
            ),
        )
        self._require_dataframe(data)
        frame = self._prepare_base(data)
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = self._apply_medians(frame)
        frame = self._apply_knn(frame)
        frame = self._apply_modes(frame)
        return frame.loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=("output_columns_",))
        return self.output_columns_.copy()


def build_weather_preprocessor(location_coordinates, random_state=42):
    """Construye un preprocesador completo que puede clonarse dentro de CV."""

    numeric_pipeline = Pipeline(
        steps=(
            (
                "fallback_imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            (
                "fallback_imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        )
    )
    encode_and_scale = ColumnTransformer(
        transformers=(
            ("num", numeric_pipeline, list(NUMERIC_COLUMNS)),
            ("cat", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ),
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=(
            (
                "weather_features",
                WeatherFeatureTransformer(
                    location_coordinates=location_coordinates,
                    random_state=random_state,
                ),
            ),
            ("encode_and_scale", encode_and_scale),
        )
    )
