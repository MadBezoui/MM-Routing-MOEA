"""survey_data_loader.py
=======================
Loads and calibrates real survey data from the ``survey_results/`` directory.

Five CSV files are consumed:
  01_demographics_profiles.csv      – 749 student records
  02_mode_usage_frequency.csv       – per-mode usage frequency (1-5 Likert)
  03_comfort_factor_importance.csv  – 10 comfort-factor importance scores (1-5)
  04_comfort_scenario_ratings.csv   – 8988 human comfort ratings + MLP predictions
  05_pairwise_objective_preferences.csv – pairwise objective preferences

Public API
----------
  load_real_survey_calibration(survey_dir)  -> SurveyCalibration
  load_real_profiles(survey_dir)            -> pd.DataFrame (749 rows)
  load_comfort_training_data(survey_dir)    -> pd.DataFrame (8988 rows, training-ready)
  compute_objective_weights(survey_dir)     -> Dict[str, float]
  load_all(survey_dir)                      -> SurveyData (named-tuple bundle)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import numpy as np
import pandas as pd

from config import SurveyCalibration

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SEP = ";"
_DEMO_FILE = "01_demographics_profiles.csv"
_MODE_FILE = "02_mode_usage_frequency.csv"
_COMFORT_FACTOR_FILE = "03_comfort_factor_importance.csv"
_SCENARIO_FILE = "04_comfort_scenario_ratings.csv"
_PAIRWISE_FILE = "05_pairwise_objective_preferences.csv"


def _read(survey_dir: Path, filename: str) -> pd.DataFrame:
    path = survey_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected survey file not found: {path}")
    return pd.read_csv(path, sep=_SEP)


# ---------------------------------------------------------------------------
# 1. SurveyCalibration — empirically calibrated from real data
# ---------------------------------------------------------------------------

def load_real_survey_calibration(survey_dir: str | Path) -> SurveyCalibration:
    """Return a :class:`SurveyCalibration` whose parameters are derived from
    the 749 real respondents rather than hard-coded defaults.

    Derivation of each parameter:
      sample_size               – exact number of rows in demographics file
      mean_age                  – mean(age)
      mean_distance_to_campus_km– mean(distance_km)
      mean_daily_budget_eur     – mean(max_budget_eur)
      walking_threshold_km      – mean(max_walking_distance_m) / 1000
      weather_importance        – mean of (weather_rain_walking + weather_rain_cycling) / 5
      safety_importance         – mean(safety_perception) / 5
      reliability_importance    – mean(waiting_time_uncertainty) / 5
      comfort_over_emission_prob– fraction who prefer Comfort over Emissions
      comfort_over_cost_prob    – fraction who prefer Comfort over Cost
      eco_attitude_mean         – mean(environmental_attitude) / 5
    """
    survey_dir = Path(survey_dir)

    demo = _read(survey_dir, _DEMO_FILE)
    cf = _read(survey_dir, _COMFORT_FACTOR_FILE)
    pw = _read(survey_dir, _PAIRWISE_FILE)

    n = len(demo)

    # Weather importance: average of "rain affects walking" and "rain affects cycling"
    weather_importance = float(
        (cf["weather_rain_walking"].mean() + cf["weather_rain_cycling"].mean()) / 2 / 5.0
    )

    safety_importance = float(cf["safety_perception"].mean() / 5.0)
    reliability_importance = float(cf["waiting_time_uncertainty"].mean() / 5.0)

    # Pairwise objective preferences
    # comfort_over_cost_prob: fraction preferring Comfort (=Prefer_Second in Cost_vs_Comfort)
    cost_vs_comfort = pw["Cost_vs_Comfort"].value_counts(normalize=True)
    comfort_over_cost_prob = float(cost_vs_comfort.get("Prefer_Second", 0.0))

    # comfort_over_emission_prob: fraction preferring Comfort (=Prefer_Second in Emissions_vs_Comfort)
    em_vs_comfort = pw["Emissions_vs_Comfort"].value_counts(normalize=True)
    comfort_over_emission_prob = float(em_vs_comfort.get("Prefer_Second", 0.0))

    # Eco attitude: environmental_attitude is already on 1-5 scale
    eco_attitude_mean = float(demo["environmental_attitude"].mean() / 5.0)

    return SurveyCalibration(
        sample_size=n,
        mean_age=float(demo["age"].mean()),
        mean_distance_to_campus_km=float(demo["distance_km"].mean()),
        mean_daily_budget_eur=float(demo["max_budget_eur"].mean()),
        walking_threshold_km=float(demo["max_walking_distance_m"].mean() / 1000.0),
        weather_importance=weather_importance,
        safety_importance=safety_importance,
        reliability_importance=reliability_importance,
        comfort_over_emission_prob=comfort_over_emission_prob,
        comfort_over_cost_prob=comfort_over_cost_prob,
        eco_attitude_mean=eco_attitude_mean,
    )


# ---------------------------------------------------------------------------
# 2. Real profiles — 749 student records as optimization profiles
# ---------------------------------------------------------------------------

def load_real_profiles(survey_dir: str | Path) -> pd.DataFrame:
    """Return a DataFrame with one row per student, ready to be used as
    optimization profiles in the pipeline.

    The returned columns match those expected by :class:`DefaultMultimodalEvaluator`:
      profile_id, archetype, trip_distance_bin, distance_km, age,
      budget_eur, rain, temperature_c, mobility_restriction, seed_offset,
      campus, student_status
    """
    survey_dir = Path(survey_dir)
    demo = _read(survey_dir, _DEMO_FILE)

    # Normalise boolean columns
    mobility_bool = demo["mobility_restriction"].map(
        lambda x: 1 if str(x).strip().lower() in ("yes", "oui", "1", "true") else 0
    )

    # rain is not in the survey directly; we impute from season distribution (30 % rainy days)
    rng = np.random.default_rng(42)
    rain_col = rng.binomial(1, 0.30, size=len(demo))

    # temperature_c: seasonal mean for Strasbourg/Nancy area
    temperature_col = rng.normal(14.0, 8.0, size=len(demo))

    profiles = pd.DataFrame(
        {
            "profile_id": demo["student_id"],
            "archetype": demo["archetype"].str.lower().str.replace(" ", "_"),
            "trip_distance_bin": demo["distance_category"].str.lower(),
            "distance_km": demo["distance_km"].clip(lower=0.3),
            "age": demo["age"],
            "budget_eur": demo["max_budget_eur"].clip(lower=0.5),
            "rain": rain_col,
            "temperature_c": temperature_col,
            "mobility_restriction": mobility_bool,
            "seed_offset": range(len(demo)),
            "campus": demo["campus"],
            "student_status": demo["student_status"],
        }
    )
    return profiles.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Comfort training data — real human ratings as supervised labels
# ---------------------------------------------------------------------------

_MODE_MIX_COLS = ["walk_share", "bike_share", "bus_share", "tram_share", "car_share"]

_MODE_MIX_MAP: Dict[str, Dict[str, float]] = {
    # pure-mode scenarios
    "walk_100":    {"walk_share": 1.0, "bike_share": 0.0, "bus_share": 0.0, "tram_share": 0.0, "car_share": 0.0},
    "bike_100":    {"walk_share": 0.0, "bike_share": 1.0, "bus_share": 0.0, "tram_share": 0.0, "car_share": 0.0},
    "bus_100":     {"walk_share": 0.0, "bike_share": 0.0, "bus_share": 1.0, "tram_share": 0.0, "car_share": 0.0},
    "tram_100":    {"walk_share": 0.0, "bike_share": 0.0, "bus_share": 0.0, "tram_share": 1.0, "car_share": 0.0},
    "car_100":     {"walk_share": 0.0, "bike_share": 0.0, "bus_share": 0.0, "tram_share": 0.0, "car_share": 1.0},
    # multimodal scenarios (approximate splits)
    "walk_bus_50_50":      {"walk_share": 0.50, "bike_share": 0.0,  "bus_share": 0.50, "tram_share": 0.0,  "car_share": 0.0},
    "walk_tram_50_50":     {"walk_share": 0.50, "bike_share": 0.0,  "bus_share": 0.0,  "tram_share": 0.50, "car_share": 0.0},
    "bike_tram_50_50":     {"walk_share": 0.0,  "bike_share": 0.50, "bus_share": 0.0,  "tram_share": 0.50, "car_share": 0.0},
    "walk_bus_tram_33_33": {"walk_share": 0.33, "bike_share": 0.0,  "bus_share": 0.33, "tram_share": 0.34, "car_share": 0.0},
    "car_bus_50_50":       {"walk_share": 0.0,  "bike_share": 0.0,  "bus_share": 0.50, "tram_share": 0.0,  "car_share": 0.50},
    "ridesharing":         {"walk_share": 0.0,  "bike_share": 0.0,  "bus_share": 0.0,  "tram_share": 0.0,  "car_share": 1.0},
}


def _parse_mode_mix(series: pd.Series) -> pd.DataFrame:
    """Parse mode_mix strings into share columns (5 floats that sum to 1)."""
    rows = []
    for val in series:
        val_lower = str(val).strip().lower()
        if val_lower in _MODE_MIX_MAP:
            rows.append(_MODE_MIX_MAP[val_lower])
        else:
            # Fallback: try to detect dominant mode from the string
            shares = {"walk_share": 0.0, "bike_share": 0.0, "bus_share": 0.0, "tram_share": 0.0, "car_share": 0.0}
            for mode, col in [("walk", "walk_share"), ("bike", "bike_share"), ("bus", "bus_share"),
                               ("tram", "tram_share"), ("car", "car_share")]:
                if mode in val_lower:
                    shares[col] = 1.0
                    break
            else:
                shares["bus_share"] = 1.0  # safest default
            rows.append(shares)
    return pd.DataFrame(rows)


def load_comfort_training_data(survey_dir: str | Path) -> pd.DataFrame:
    """Return a training-ready DataFrame for the comfort surrogate model.

    The target column is ``comfort_score`` (= ``human_comfort_normalized_0_1``).
    Feature columns match :attr:`ComfortTrainingConfig.route_feature_columns` plus
    ``weather_label`` and ``dominant_mode`` (categorical).

    The MLP will be re-trained from scratch on the 8988 real human ratings.
    """
    survey_dir = Path(survey_dir)
    scen = _read(survey_dir, _SCENARIO_FILE)

    # --- mode shares ---
    mode_df = _parse_mode_mix(scen["mode_mix"])

    # --- weather: map string to rain (binary) and label ---
    weather_map = {"sunny": 0, "cloudy": 0, "rain": 1, "rainy": 1}
    rain_col = scen["weather"].str.lower().map(weather_map).fillna(0).astype(int)

    # --- crowding: map text levels to numeric ---
    crowding_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "NA": 0.3, "na": 0.3}
    crowding_col = scen["crowding"].astype(str).str.lower().map(crowding_map).fillna(0.3)

    # --- reliability penalty: inferred from transfers (more transfers → more uncertainty) ---
    reliability_penalty = np.clip(scen["transfers"].fillna(0) * 0.15, 0, 0.6)

    # --- dominant_mode from mode_mix string ---
    dominant_mode_col = scen["mode_mix"].apply(
        lambda x: str(x).split("_")[0] if isinstance(x, str) else "bus"
    )

    df = pd.DataFrame(
        {
            "walk_share":        mode_df["walk_share"].values,
            "bike_share":        mode_df["bike_share"].values,
            "bus_share":         mode_df["bus_share"].values,
            "tram_share":        mode_df["tram_share"].values,
            "car_share":         mode_df["car_share"].values,
            "crowding":          crowding_col.values,
            "transfers":         scen["transfers"].fillna(0).astype(int).values,
            "distance_km":       scen["distance_km"].fillna(scen["distance_km"].mean()).values,
            "rain":              rain_col.values,
            "temperature_c":     np.full(len(scen), 14.0),   # not recorded per scenario → use mean
            "age":               np.full(len(scen), 25.8),   # mean from demographics
            "mobility_restriction": np.zeros(len(scen), dtype=int),
            "reliability_penalty": reliability_penalty.values,
            "safety_penalty":    np.full(len(scen), 0.10),   # fixed baseline
            "fare_eur":          scen["distance_km"].fillna(3.0) * 0.25,  # rough estimate
            "travel_time_min":   scen["distance_km"].fillna(3.0) / 20.0 * 60.0,
            "weather_label":     scen["weather"].str.lower().fillna("sunny").values,
            "dominant_mode":     dominant_mode_col.values,
            # Target
            "comfort_score":     scen["human_comfort_normalized_0_1"].clip(0.0, 1.0).values,
        }
    )

    # Drop rows with NaN target
    before = len(df)
    df = df.dropna(subset=["comfort_score"]).reset_index(drop=True)
    if before > len(df):
        warnings.warn(f"Dropped {before - len(df)} rows with missing comfort_score.")

    return df


# ---------------------------------------------------------------------------
# 4. Objective weights — derived from pairwise preferences (Borda scoring)
# ---------------------------------------------------------------------------

def compute_objective_weights(survey_dir: str | Path) -> Dict[str, float]:
    """Compute normalized preference weights for the 4 objectives using a
    Borda-count aggregation of the pairwise preference votes.

    Returns a dict with keys: ``time``, ``cost``, ``emissions``, ``comfort``.
    Values sum to 1.0.

    Scoring rule:
      Prefer_First  → +1 for the first objective, -1 for the second
      Prefer_Second → -1 for the first objective, +1 for the second
      Equal         →  0 for both

    The final score per objective is then min-max normalized to [0.05, 1.0]
    and renormalized to sum to 1.
    """
    survey_dir = Path(survey_dir)
    pw = _read(survey_dir, _PAIRWISE_FILE)

    scores: Dict[str, float] = {"time": 0.0, "cost": 0.0, "emissions": 0.0, "comfort": 0.0}

    pair_map = {
        "Time_vs_Cost":        ("time", "cost"),
        "Time_vs_Emissions":   ("time", "emissions"),
        "Time_vs_Comfort":     ("time", "comfort"),
        "Cost_vs_Emissions":   ("cost", "emissions"),
        "Cost_vs_Comfort":     ("cost", "comfort"),
        "Emissions_vs_Comfort":("emissions", "comfort"),
    }

    for col, (obj_a, obj_b) in pair_map.items():
        if col not in pw.columns:
            continue
        counts = pw[col].value_counts()
        n_first  = counts.get("Prefer_First", 0)
        n_second = counts.get("Prefer_Second", 0)
        scores[obj_a] += n_first  - n_second
        scores[obj_b] += n_second - n_first

    # Shift to positive and normalize
    min_s = min(scores.values())
    shifted = {k: v - min_s + 0.05 for k, v in scores.items()}
    total = sum(shifted.values())
    weights = {k: v / total for k, v in shifted.items()}
    return weights


# ---------------------------------------------------------------------------
# 5. Convenience bundle
# ---------------------------------------------------------------------------

class SurveyData(NamedTuple):
    calibration: SurveyCalibration
    profiles: pd.DataFrame
    comfort_training: pd.DataFrame
    objective_weights: Dict[str, float]


def load_all(survey_dir: str | Path) -> SurveyData:
    """Load and return all survey-derived objects in one call."""
    survey_dir = Path(survey_dir)
    return SurveyData(
        calibration=load_real_survey_calibration(survey_dir),
        profiles=load_real_profiles(survey_dir),
        comfort_training=load_comfort_training_data(survey_dir),
        objective_weights=compute_objective_weights(survey_dir),
    )
