from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ComfortTrainingConfig, SurveyCalibration


@dataclass
class ComfortModelResult:
    model_name: str
    pipeline: Optional[Pipeline]
    metrics: Dict[str, float]
    region_metrics: pd.DataFrame
    predictions: pd.DataFrame


class HeuristicComfortModel:
    """Transparent rule-based comfort baseline.

    This is included to answer reviewer requests for an ablation where the
    heuristic formula is used directly, without a learned surrogate.
    """

    def __init__(self, survey: SurveyCalibration):
        self.survey = survey

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return apply_survey_informed_heuristics(df, self.survey).to_numpy()


class SurveyInformedComfortFactory:
    def __init__(self, config: ComfortTrainingConfig, survey: SurveyCalibration):
        self.config = config
        self.survey = survey

    def generate_synthetic_dataset(self, n_samples: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
        n = n_samples or self.config.n_samples
        rng = np.random.default_rng(self.config.random_state if seed is None else seed)

        distance = rng.gamma(shape=2.0, scale=max(self.survey.mean_distance_to_campus_km / 2.0, 0.1), size=n)
        walk_share = np.clip(rng.beta(2, 5, size=n), 0, 1)
        bike_share = np.clip(rng.beta(2, 6, size=n), 0, 1)
        bus_share = np.clip(rng.beta(3, 3, size=n), 0, 1)
        tram_share = np.clip(rng.beta(2, 4, size=n), 0, 1)
        car_share = np.clip(1 - (walk_share + bike_share + bus_share + tram_share), 0, 1)
        row_sum = walk_share + bike_share + bus_share + tram_share + car_share
        walk_share, bike_share, bus_share, tram_share, car_share = [x / row_sum for x in (walk_share, bike_share, bus_share, tram_share, car_share)]

        df = pd.DataFrame(
            {
                "walk_share": walk_share,
                "bike_share": bike_share,
                "bus_share": bus_share,
                "tram_share": tram_share,
                "car_share": car_share,
                "crowding": rng.uniform(0, 1, size=n),
                "transfers": rng.integers(0, 5, size=n),
                "distance_km": distance,
                "rain": rng.binomial(1, self.survey.weather_importance, size=n),
                "temperature_c": rng.normal(16, 8, size=n),
                "age": np.clip(rng.normal(self.survey.mean_age, 7, size=n), 18, 70),
                "mobility_restriction": rng.binomial(1, 0.08, size=n),
                "reliability_penalty": np.clip(rng.normal(1 - self.survey.reliability_importance, 0.15, size=n), 0, 1),
                "safety_penalty": np.clip(rng.normal(1 - self.survey.safety_importance, 0.15, size=n), 0, 1),
                "fare_eur": np.clip(rng.normal(self.survey.mean_daily_budget_eur / 2, 1.2, size=n), 0, 15),
                "travel_time_min": np.clip(rng.normal(28, 12, size=n), 5, 120),
                "weather_label": rng.choice(["sunny", "cloudy", "rainy"], size=n, p=[0.35, 0.35, 0.30]),
                "dominant_mode": rng.choice(["walk", "bike", "bus", "tram", "car"], size=n, p=[0.15, 0.15, 0.32, 0.22, 0.16]),
            }
        )

        df["comfort_score"] = apply_survey_informed_heuristics(df, self.survey)
        return df

    def train_models(self, df: pd.DataFrame) -> List[ComfortModelResult]:
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        feature_cols = [c for c in df.columns if c != "comfort_score"]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ]
        )

        models: List[Tuple[str, Optional[Pipeline]]] = [
            ("heuristic_direct", None),
            (
                "linear_regression",
                Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("model", LinearRegression()),
                    ]
                ),
            ),
            (
                "mlp_surrogate",
                Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        (
                            "model",
                            MLPRegressor(
                                hidden_layer_sizes=self.config.hidden_layers,
                                activation=self.config.activation,
                                alpha=self.config.alpha,
                                random_state=self.config.random_state,
                                max_iter=self.config.max_iter,
                                early_stopping=self.config.early_stopping,
                                learning_rate_init=self.config.learning_rate_init,
                            ),
                        ),
                    ]
                ),
            ),
        ]

        results: List[ComfortModelResult] = []
        for model_name, pipeline in models:
            if model_name == "heuristic_direct":
                predictor = HeuristicComfortModel(self.survey)
                y_pred = predictor.predict(test_df)
            else:
                pipeline.fit(train_df[feature_cols], train_df["comfort_score"])
                y_pred = pipeline.predict(test_df[feature_cols])

            pred_df = test_df[feature_cols].copy()
            pred_df["y_true"] = test_df["comfort_score"].to_numpy()
            pred_df["y_pred"] = np.clip(y_pred, 0.0, 1.0)
            pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])

            results.append(
                ComfortModelResult(
                    model_name=model_name,
                    pipeline=pipeline,
                    metrics=compute_prediction_metrics(pred_df["y_true"], pred_df["y_pred"]),
                    region_metrics=compute_region_wise_errors(pred_df),
                    predictions=pred_df,
                )
            )
        return results

    def noise_robustness(self, model_result: ComfortModelResult, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c != "comfort_score"]
        base = df.copy()
        records: List[Dict[str, float]] = []

        for noise_level in self.config.noise_levels:
            noisy = base.copy()
            if noise_level > 0:
                rng = np.random.default_rng(self.config.random_state + int(noise_level * 1000))
                noisy["comfort_score"] = np.clip(
                    noisy["comfort_score"] + rng.normal(0, noise_level, size=len(noisy)),
                    0.0,
                    1.0,
                )

            if model_result.model_name == "heuristic_direct":
                y_pred = HeuristicComfortModel(self.survey).predict(noisy)
            else:
                train_df, test_df = train_test_split(
                    noisy,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                )
                model_result.pipeline.fit(train_df[feature_cols], train_df["comfort_score"])
                y_pred = model_result.pipeline.predict(test_df[feature_cols])
                noisy = test_df

            metrics = compute_prediction_metrics(noisy["comfort_score"], np.clip(y_pred, 0.0, 1.0))
            metrics["noise_level"] = noise_level
            metrics["model_name"] = model_result.model_name
            records.append(metrics)
        return pd.DataFrame(records)


def apply_survey_informed_heuristics(df: pd.DataFrame, survey: SurveyCalibration) -> pd.Series:
    score = np.full(len(df), 0.95, dtype=float)

    rain = df.get("rain", 0).astype(float).to_numpy() if isinstance(df, pd.DataFrame) else 0
    walk_share = df.get("walk_share", 0).astype(float).to_numpy()
    bike_share = df.get("bike_share", 0).astype(float).to_numpy()
    crowding = df.get("crowding", 0).astype(float).to_numpy()
    transfers = df.get("transfers", 0).astype(float).to_numpy()
    age = df.get("age", survey.mean_age).astype(float).to_numpy()
    mobility_restriction = df.get("mobility_restriction", 0).astype(float).to_numpy()
    reliability_penalty = df.get("reliability_penalty", 0).astype(float).to_numpy()
    safety_penalty = df.get("safety_penalty", 0).astype(float).to_numpy()
    fare_eur = df.get("fare_eur", survey.mean_daily_budget_eur / 2).astype(float).to_numpy()
    distance_km = df.get("distance_km", survey.mean_distance_to_campus_km).astype(float).to_numpy()
    travel_time_min = df.get("travel_time_min", 25).astype(float).to_numpy()

    score -= rain * (0.22 * walk_share + 0.30 * bike_share) * survey.weather_importance
    score -= 0.16 * crowding
    score -= 0.06 * np.clip(transfers, 0, 4)
    score -= 0.09 * (age > 60).astype(float) * bike_share
    score -= 0.12 * mobility_restriction * (walk_share + bike_share)
    score -= 0.10 * reliability_penalty * survey.reliability_importance
    score -= 0.08 * safety_penalty * survey.safety_importance
    score -= 0.05 * np.maximum(fare_eur - survey.mean_daily_budget_eur, 0)
    score -= 0.002 * np.maximum(distance_km - survey.walking_threshold_km, 0) * walk_share
    score -= 0.0015 * np.maximum(travel_time_min - 45, 0)

    sunny_bonus = (1 - rain) * np.clip(walk_share + bike_share, 0, 1) * 0.03
    score += sunny_bonus

    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)


def compute_prediction_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def compute_region_wise_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    region_df = predictions.copy()
    region_df["comfort_band"] = pd.cut(
        region_df["y_true"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["low", "medium", "high"],
    )
    return (
        region_df.groupby("comfort_band", observed=False)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
        )
        .reset_index()
    )
