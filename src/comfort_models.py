"""comfort_models.py
====================
The survey-calibrated comfort surrogate of Section 4.1 and its two baselines.

Three models are compared on the same twelve features and the same
respondent-level split:

``heuristic_direct``
    a transparent rule-based baseline, no learning;
``linear_regression``
    additive baseline, quantifies the contribution of non-linear interactions;
``mlp_surrogate``
    the proposed model: two hidden layers of 100 and 50 units, ReLU
    activation, :math:`L_2` regularisation at :math:`\\alpha = 10^{-2}`, and a
    sigmoid output, trained with Adam and early stopping on a 20 % held-out
    validation split.

The split is performed at the **respondent** level, so that all trip-comfort
pairs of a given respondent land either in the training set or in the test set
but never in both.  The reported :math:`R^2` therefore measures generalisation
to unseen respondents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import COMFORT_FEATURES, ComfortTrainingConfig, SurveyCalibration

logger = logging.getLogger(__name__)

#: Squeeze applied before the logit so that labels at the boundaries of the
#: ordinal scale (a perfect or a minimal comfort rating) do not map to an
#: unbounded target.  Without it a single label at 1.0 would dominate the loss.
_SQUEEZE = 0.02


@dataclass
class ComfortModelResult:
    model_name: str
    pipeline: Optional[Pipeline]
    metrics: Dict[str, float]
    region_metrics: pd.DataFrame
    predictions: pd.DataFrame


# --------------------------------------------------------------------------
# Sigmoid output layer
# --------------------------------------------------------------------------

def _logit(y: np.ndarray, squeeze: float = _SQUEEZE) -> np.ndarray:
    y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0) * (1.0 - 2.0 * squeeze) + squeeze
    return np.log(y / (1.0 - y))


def _expit(z: np.ndarray, squeeze: float = _SQUEEZE) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -30.0, 30.0)))
    return np.clip((s - squeeze) / (1.0 - 2.0 * squeeze), 0.0, 1.0)


class SigmoidOutputMLP(MLPRegressor):
    """Multilayer perceptron with a sigmoid output unit.

    ``scikit-learn`` regressors expose a linear output layer.  Fitting the
    network on the (squeezed) logit of the target and squashing its prediction
    back through the logistic function is equivalent to placing a sigmoid on
    the output unit: predictions lie inside ``[0, 1]`` by construction rather
    than by clipping, and the loss is expressed on the same scale as a
    genuinely sigmoid-headed network.
    """

    def fit(self, X, y):  # type: ignore[override]
        return super().fit(X, _logit(y))

    def predict(self, X):  # type: ignore[override]
        return _expit(super().predict(X))


# --------------------------------------------------------------------------
# Heuristic baseline
# --------------------------------------------------------------------------

def apply_survey_informed_heuristics(
    df: pd.DataFrame, survey: SurveyCalibration
) -> pd.Series:
    """Rule-based comfort score on the twelve features, in ``[0, 1]``.

    The rules follow the published comfort determinants (weather exposure for
    active modes, in-vehicle crowding, interchange burden, exposure of older
    and mobility-restricted travellers to active modes, and trip length).  No
    parameter is fitted.
    """
    def col(name: str, default: float) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=float)
        return np.full(len(df), float(default))

    walk = col("walk_share", 0.0)
    bike = col("bike_share", 0.0)
    bus = col("bus_share", 0.0)
    tram = col("tram_share", 0.0)
    crowding = col("crowding", 0.0)
    transfers = col("transfers", 0.0)
    distance = col("distance_km", survey.mean_distance_to_campus_km)
    rain = col("rain", 0.0)
    temperature = col("temperature_c", 14.0)
    age = col("age", survey.mean_age)
    restriction = col("mobility_restriction", 0.0)

    score = np.full(len(df), 0.95, dtype=float)
    score -= rain * (0.22 * walk + 0.30 * bike) * survey.weather_importance
    score -= 0.16 * crowding
    score -= 0.06 * np.clip(transfers, 0, 4)
    score -= 0.09 * (age > 60).astype(float) * bike
    score -= 0.12 * restriction * (walk + bike)
    score -= 0.10 * (bus + tram) * (1.0 - survey.reliability_importance)
    score -= 0.08 * walk * (1.0 - survey.safety_importance)
    score -= 0.002 * np.maximum(distance - survey.walking_threshold_km, 0.0) * walk
    score -= 0.004 * np.maximum(8.0 - temperature, 0.0) * (walk + bike)
    score += (1.0 - rain) * np.clip(walk + bike, 0.0, 1.0) * 0.03
    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)


class HeuristicComfortModel:
    """Wrapper exposing the heuristic through the estimator interface."""

    def __init__(self, survey: SurveyCalibration):
        self.survey = survey

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return apply_survey_informed_heuristics(df, self.survey).to_numpy()


# --------------------------------------------------------------------------
# Training factory
# --------------------------------------------------------------------------

class SurveyInformedComfortFactory:
    """Trains and evaluates the three comfort models on the real survey data."""

    def __init__(self, config: ComfortTrainingConfig, survey: SurveyCalibration):
        self.config = config
        self.survey = survey
        self.feature_cols: List[str] = list(config.feature_columns)

    # -- preprocessing -----------------------------------------------------

    def _preprocessor(self) -> Pipeline:
        """Median imputation then min-max normalization to ``[0, 1]``.

        Both are fitted on the training fold only; the same bounds are applied
        to the test fold (Section 4.1).
        """
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(feature_range=(0.0, 1.0), clip=True)),
        ])

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Leakage-free split at the respondent level."""
        groups = df["respondent_id"] if "respondent_id" in df.columns else np.arange(len(df))
        gss = GroupShuffleSplit(
            n_splits=1, test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        train_idx, test_idx = next(gss.split(df, groups=groups))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

    def _models(self) -> List[Tuple[str, Optional[Pipeline]]]:
        return [
            ("heuristic_direct", None),
            ("linear_regression", Pipeline([
                ("preprocess", self._preprocessor()),
                ("model", LinearRegression()),
            ])),
            ("mlp_surrogate", Pipeline([
                ("preprocess", self._preprocessor()),
                ("model", SigmoidOutputMLP(
                    hidden_layer_sizes=self.config.hidden_layers,
                    activation=self.config.activation,
                    alpha=self.config.alpha,
                    solver="adam",
                    learning_rate_init=self.config.learning_rate_init,
                    max_iter=self.config.max_iter,
                    early_stopping=self.config.early_stopping,
                    validation_fraction=self.config.validation_fraction,
                    random_state=self.config.random_state,
                )),
            ])),
        ]

    # -- training ----------------------------------------------------------

    def train_models(self, df: pd.DataFrame) -> List[ComfortModelResult]:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"comfort training frame is missing features: {missing}")

        train_df, test_df = self._split(df)
        logger.info(
            "Comfort split: %d train / %d test pairs from %d / %d respondents",
            len(train_df), len(test_df),
            train_df.get("respondent_id", pd.Series(dtype=object)).nunique(),
            test_df.get("respondent_id", pd.Series(dtype=object)).nunique(),
        )

        results: List[ComfortModelResult] = []
        for name, pipeline in self._models():
            if name == "heuristic_direct":
                y_pred = HeuristicComfortModel(self.survey).predict(test_df)
            else:
                pipeline.fit(train_df[self.feature_cols], train_df["comfort_score"])
                y_pred = pipeline.predict(test_df[self.feature_cols])

            pred_df = test_df[self.feature_cols].copy()
            pred_df["y_true"] = test_df["comfort_score"].to_numpy()
            pred_df["y_pred"] = np.clip(y_pred, 0.0, 1.0)
            pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])

            results.append(ComfortModelResult(
                model_name=name,
                pipeline=pipeline,
                metrics=compute_prediction_metrics(pred_df["y_true"], pred_df["y_pred"]),
                region_metrics=compute_region_wise_errors(pred_df),
                predictions=pred_df,
            ))
            logger.info("  %-18s R2=%.3f RMSE=%.4f MAE=%.4f", name,
                        results[-1].metrics["r2"], results[-1].metrics["rmse"],
                        results[-1].metrics["mae"])
        return results

    # -- robustness --------------------------------------------------------

    def noise_robustness(
        self,
        model_result: ComfortModelResult,
        df: pd.DataFrame,
        noise_levels: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """Degradation of predictive accuracy under **input-feature** noise.

        Additive Gaussian noise of standard deviation ``sigma`` is applied to
        the twelve input features of the *test* fold, after min-max scaling has
        been fitted on the clean training fold, so the perturbation is
        expressed on the same scale for every feature (Section 6.6).  The
        labels are never modified.
        """
        levels = list(noise_levels if noise_levels is not None else self.config.noise_levels)
        train_df, test_df = self._split(df)

        scaler = self._preprocessor()
        scaler.fit(train_df[self.feature_cols])

        if model_result.model_name != "heuristic_direct":
            model_result.pipeline.fit(train_df[self.feature_cols], train_df["comfort_score"])

        y_true = test_df["comfort_score"].to_numpy(dtype=float)
        records: List[Dict[str, float]] = []

        for sigma in levels:
            rng = np.random.default_rng(self.config.random_state + int(round(sigma * 1000)))
            X_scaled = scaler.transform(test_df[self.feature_cols])
            if sigma > 0:
                X_scaled = np.clip(X_scaled + rng.normal(0.0, sigma, size=X_scaled.shape), 0.0, 1.0)
            X_noisy = pd.DataFrame(
                scaler.named_steps["scaler"].inverse_transform(X_scaled),
                columns=self.feature_cols, index=test_df.index,
            )

            if model_result.model_name == "heuristic_direct":
                y_pred = HeuristicComfortModel(self.survey).predict(X_noisy)
            else:
                y_pred = model_result.pipeline.predict(X_noisy[self.feature_cols])

            metrics = compute_prediction_metrics(y_true, np.clip(y_pred, 0.0, 1.0))
            metrics["noise_level"] = float(sigma)
            metrics["model_name"] = model_result.model_name
            records.append(metrics)

        return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_prediction_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def compute_region_wise_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    region = predictions.copy()
    region["comfort_band"] = pd.cut(
        region["y_true"], bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["low", "medium", "high"],
    )
    return (
        region.groupby("comfort_band", observed=False)
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"),
             mean_true=("y_true", "mean"), mean_pred=("y_pred", "mean"))
        .reset_index()
    )


# --------------------------------------------------------------------------
# Runtime predictor used inside the optimization loop
# --------------------------------------------------------------------------

class TrainedComfortPredictor:
    """Adapter exposing a trained model to :class:`PathMultimodalEvaluator`."""

    def __init__(self, comfort_results: Sequence[ComfortModelResult],
                 model_name: str = "mlp_surrogate"):
        match = [r for r in comfort_results if r.model_name == model_name]
        if not match:
            raise ValueError(f"model '{model_name}' not found among trained comfort models")
        self.model_name = model_name
        self.pipeline = match[0].pipeline
        self.feature_cols = list(COMFORT_FEATURES)

    def predict(self, comfort_df: pd.DataFrame, survey: SurveyCalibration) -> np.ndarray:
        if self.pipeline is None:  # heuristic baseline
            return apply_survey_informed_heuristics(comfort_df, survey).to_numpy()
        preds = self.pipeline.predict(comfort_df[self.feature_cols])
        return np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
