"""survey_data_loader.py
========================
Loading and calibration of the real survey data (Section 4.1).

Five semicolon-separated files are consumed from ``data/survey_results/``:

===================================  ==========================================
``01_demographics_profiles.csv``     one row per respondent
``02_mode_usage_frequency.csv``      per-mode usage frequency (1-5 Likert)
``03_comfort_factor_importance.csv`` comfort-factor importance scores (1-5)
``04_comfort_scenario_ratings.csv``  one row per rated trip scenario
``05_pairwise_objective_preferences.csv``  pairwise objective preferences
===================================  ==========================================

Public API
----------
``load_real_survey_calibration``  aggregate anchors
``load_real_profiles``            optimization profiles with their three bounds
``load_comfort_training_data``    the twelve features plus the comfort label
``compute_objective_weights``     raw elicited priority weights
``describe_survey``               provenance report of every derived field
``load_all``                      all of the above in one bundle
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd

from src.config import COMFORT_FEATURES, SurveyCalibration

logger = logging.getLogger(__name__)

_SEP = ";"
_DEMO_FILE = "01_demographics_profiles.csv"
_MODE_FILE = "02_mode_usage_frequency.csv"
_COMFORT_FACTOR_FILE = "03_comfort_factor_importance.csv"
_SCENARIO_FILE = "04_comfort_scenario_ratings.csv"
_PAIRWISE_FILE = "05_pairwise_objective_preferences.csv"

#: Deterministic temperature associated with each recorded weather condition.
#: Per-scenario temperature is not an elicited field; it is derived from the
#: recorded ``weather`` label using September climate normals for Strasbourg.
#: The mapping is deterministic: no value is sampled.
_TEMPERATURE_BY_WEATHER: Dict[str, float] = {
    "sunny": 21.0, "clear": 21.0,
    "cloudy": 16.0, "overcast": 16.0,
    "rain": 12.0, "rainy": 12.0,
}
_DEFAULT_TEMPERATURE_C = 16.0

_SHARE_COLS = ("walk_share", "bike_share", "bus_share", "tram_share", "car_share")

_MODE_TOKENS = (
    ("walk", "walk_share"), ("bike", "bike_share"), ("bus", "bus_share"),
    ("tram", "tram_share"), ("car", "car_share"),
)

#: Mode-mix labels of the scenario file mapped to the five mode shares.
#: Single-mode scenarios put the whole trip on one mode; multimodal labels
#: split the trip equally across the modes they name, which is how the
#: scenario descriptions present them to respondents.
_MODE_MIX_MAP: Dict[str, Dict[str, float]] = {
    "walk_100": {"walk_share": 1.0},
    "bike_100": {"bike_share": 1.0},
    "bus_100": {"bus_share": 1.0},
    "tram_100": {"tram_share": 1.0},
    "car_100": {"car_share": 1.0},
    "ridesharing": {"car_share": 1.0},
    "walk_bus": {"walk_share": 0.50, "bus_share": 0.50},
    "walk_tram": {"walk_share": 0.50, "tram_share": 0.50},
    "bike_tram": {"bike_share": 0.50, "tram_share": 0.50},
    "car_bus": {"car_share": 0.50, "bus_share": 0.50},
    "walk_bus_tram": {"walk_share": 1 / 3, "bus_share": 1 / 3, "tram_share": 1 / 3},
}

#: Perceived in-vehicle crowding by recorded crowding level.  A missing level
#: on a scenario without a public-transport leg means "not applicable".
_CROWDING_MAP = {"low": 0.2, "medium": 0.5, "high": 0.8}

#: ``any`` marks a weather-independent scenario: no rain is implied and the
#: annual mean temperature is used.
_WEATHER_RAIN = {"rain", "rainy"}


def _read(survey_dir: Path, filename: str) -> pd.DataFrame:
    path = Path(survey_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"expected survey file not found: {path}")
    return pd.read_csv(path, sep=_SEP)


# --------------------------------------------------------------------------
# 1. Aggregate calibration
# --------------------------------------------------------------------------

def load_real_survey_calibration(survey_dir: str | Path) -> SurveyCalibration:
    """Aggregate anchors computed from the respondents, not hard-coded."""
    survey_dir = Path(survey_dir)
    demo = _read(survey_dir, _DEMO_FILE)
    cf = _read(survey_dir, _COMFORT_FACTOR_FILE)
    pw = _read(survey_dir, _PAIRWISE_FILE)

    cost_vs_comfort = pw["Cost_vs_Comfort"].value_counts(normalize=True)
    em_vs_comfort = pw["Emissions_vs_Comfort"].value_counts(normalize=True)

    return SurveyCalibration(
        sample_size=len(demo),
        mean_age=float(demo["age_group"].str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1).mean()),
        mean_distance_to_campus_km=float(demo["distance_km"].mean()),
        mean_daily_budget_eur=float(demo["max_budget_eur"].mean()),
        mean_max_travel_time_min=float(demo["max_travel_time_min"].mean()),
        walking_threshold_km=float(demo["max_walking_distance_m"].mean() / 1000.0),
        weather_importance=float(
            (cf["weather_rain_walking"].mean() + cf["weather_rain_cycling"].mean()) / 2 / 5.0
        ),
        safety_importance=float(cf["safety_perception"].mean() / 5.0),
        reliability_importance=float(cf["waiting_time_uncertainty"].mean() / 5.0),
        comfort_over_emission_prob=float(em_vs_comfort.get("Prefer_Second", 0.0)),
        comfort_over_cost_prob=float(cost_vs_comfort.get("Prefer_Second", 0.0)),
        eco_attitude_mean=float(demo["environmental_attitude"].mean() / 5.0),
    )


# --------------------------------------------------------------------------
# 2. Optimization profiles
# --------------------------------------------------------------------------

def _respondent_weather(survey_dir: Path) -> pd.DataFrame:
    """Modal weather condition each respondent was shown, from real ratings."""
    scen = _read(survey_dir, _SCENARIO_FILE)
    weather = scen["weather"].astype(str).str.lower().str.strip()
    modal = (
        pd.DataFrame({"student_id": scen["student_id"], "weather": weather})
        .groupby("student_id")["weather"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    modal["rain"] = modal["weather"].isin(_WEATHER_RAIN).astype(int)
    modal["temperature_c"] = modal["weather"].map(_TEMPERATURE_BY_WEATHER).fillna(_DEFAULT_TEMPERATURE_C)
    return modal


def load_real_profiles(survey_dir: str | Path) -> pd.DataFrame:
    """Return one optimization profile per respondent.

    The three feasibility bounds of Section 3.3 are carried per profile:

    ``budget_eur``                 :math:`B_u`   from ``max_budget_eur``
    ``max_travel_time_min``        :math:`T_{max,u}` from ``max_travel_time_min``
    ``max_walking_distance_km``    :math:`W_{lim,u}` from ``max_walking_distance_m``

    Weather and temperature are derived from the scenarios the respondent
    actually rated; no field is randomly generated.
    """
    survey_dir = Path(survey_dir)
    demo = _read(survey_dir, _DEMO_FILE)
    weather = _respondent_weather(survey_dir)

    mobility = demo["mobility_restriction"].map(
        lambda x: 1 if str(x).strip().lower() in ("yes", "oui", "1", "true") else 0
    )

    profiles = pd.DataFrame({
        "profile_id": demo["student_id"],
        "archetype": demo["archetype"].str.lower().str.replace(" ", "_"),
        "trip_distance_bin": demo["distance_category"].str.lower(),
        "distance_km": demo["distance_km"].clip(lower=0.3),
        "age": demo["age_group"].str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1),
        # --- the three feasibility bounds of Eq. 5 ---
        "budget_eur": demo["max_budget_eur"].clip(lower=0.5),
        "max_travel_time_min": demo["max_travel_time_min"].clip(lower=5.0),
        "max_walking_distance_km": (demo["max_walking_distance_m"] / 1000.0).clip(lower=0.05),
        "mobility_restriction": mobility,
        "seed_offset": range(len(demo)),
        "campus": demo["campus"],
        "student_status": demo["student_status"],
    })

    profiles = profiles.merge(
        weather[["student_id", "rain", "temperature_c"]],
        left_on="profile_id", right_on="student_id", how="left",
    ).drop(columns=["student_id"])
    profiles["rain"] = profiles["rain"].fillna(0).astype(int)
    profiles["temperature_c"] = profiles["temperature_c"].fillna(_DEFAULT_TEMPERATURE_C)

    return profiles.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Comfort training data
# --------------------------------------------------------------------------

def _parse_mode_mix(series: pd.Series) -> pd.DataFrame:
    """Convert the ``mode_mix`` label of each scenario into five mode shares.

    Labels present in :data:`_MODE_MIX_MAP` are used directly.  Any other label
    is decomposed by the mode tokens it contains and the trip is split equally
    between them, so that a combination never collapses onto a single mode.
    Unrecognisable labels are reported rather than silently defaulted.
    """
    rows: List[Dict[str, float]] = []
    unknown: set = set()
    for value in series:
        key = str(value).strip().lower()
        shares = {c: 0.0 for c in _SHARE_COLS}
        if key in _MODE_MIX_MAP:
            shares.update(_MODE_MIX_MAP[key])
        else:
            hits = [col for token, col in _MODE_TOKENS if token in key]
            if hits:
                for col in hits:
                    shares[col] = 1.0 / len(hits)
            else:
                unknown.add(key)
                shares["bus_share"] = 1.0
        rows.append(shares)
    if unknown:
        logger.warning("unrecognised mode_mix labels defaulted to bus: %s", sorted(unknown))
    return pd.DataFrame(rows, columns=list(_SHARE_COLS))


def load_comfort_training_data(survey_dir: str | Path) -> pd.DataFrame:
    """Return the labelled trip-comfort pairs with the twelve features.

    Every feature is either recorded directly in the survey or derived
    deterministically from a recorded field:

    ===========================  ==========================================
    five mode shares             ``mode_mix``
    ``crowding``                 ``crowding`` level, mapped to [0, 1]
    ``transfers``                ``transfers``
    ``distance_km``              ``distance_km``
    ``rain``                     ``weather``
    ``temperature_c``            ``weather`` via September climate normals
    ``age``                      joined from the demographics file
    ``mobility_restriction``     joined from the demographics file
    ===========================  ==========================================

    ``respondent_id`` is retained so that the train-test split of
    :class:`SurveyInformedComfortFactory` can be performed at respondent level.
    """
    survey_dir = Path(survey_dir)
    scen = _read(survey_dir, _SCENARIO_FILE)
    demo = _read(survey_dir, _DEMO_FILE)

    shares = _parse_mode_mix(scen["mode_mix"])
    weather = scen["weather"].astype(str).str.lower().str.strip()

    demo_lookup = demo.set_index("student_id")
    age = scen["student_id"].map(demo_lookup["age_group"].str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1))
    restriction = scen["student_id"].map(
        demo_lookup["mobility_restriction"].map(
            lambda x: 1 if str(x).strip().lower() in ("yes", "oui", "1", "true") else 0
        )
    )

    df = pd.DataFrame({
        "respondent_id": scen["student_id"].to_numpy(),
        "scenario_id": scen["scenario_id"].to_numpy() if "scenario_id" in scen else np.arange(len(scen)),
        **{c: shares[c].to_numpy() for c in _SHARE_COLS},
        "crowding": weather.index.map(lambda _: np.nan),  # placeholder, filled below
        "transfers": pd.to_numeric(scen["transfers"], errors="coerce").to_numpy(),
        "distance_km": pd.to_numeric(scen["distance_km"], errors="coerce").to_numpy(),
        "rain": weather.isin(_WEATHER_RAIN).astype(int).to_numpy(),
        "temperature_c": weather.map(_TEMPERATURE_BY_WEATHER).fillna(_DEFAULT_TEMPERATURE_C).to_numpy(),
        "age": pd.to_numeric(age, errors="coerce").to_numpy(),
        "mobility_restriction": pd.to_numeric(restriction, errors="coerce").to_numpy(),
        "comfort_score": pd.to_numeric(scen["human_comfort_rating_1_5"], errors="coerce").sub(1).div(4).clip(0, 1).to_numpy(),
    })
    # Crowding is recorded only for scenarios with a public-transport leg.  On
    # a purely active or car-based scenario the field is not applicable and is
    # set to zero; anywhere else an unrecognised level is left missing and
    # median-imputed on the training fold.
    crowding = scen["crowding"].astype(str).str.lower().str.strip().map(_CROWDING_MAP).to_numpy()
    no_transit = (df["bus_share"].to_numpy() + df["tram_share"].to_numpy()) <= 0
    crowding = np.where(np.isnan(crowding) & no_transit, 0.0, crowding)
    df["crowding"] = crowding

    before = len(df)
    df = df.dropna(subset=["comfort_score"]).reset_index(drop=True)
    if before > len(df):
        logger.warning("dropped %d scenario(s) with a missing comfort label", before - len(df))

    n_cells = len(df) * len(COMFORT_FEATURES)
    n_missing = int(df[list(COMFORT_FEATURES)].isna().to_numpy().sum())
    logger.info(
        "Comfort training set: %d pairs from %d respondents; %.2f%% of feature "
        "cells missing (median-imputed on the training fold)",
        len(df), df["respondent_id"].nunique(), 100.0 * n_missing / max(n_cells, 1),
    )
    return df


# --------------------------------------------------------------------------
# 4. Priority weights
# --------------------------------------------------------------------------

def compute_objective_weights(survey_dir: str | Path) -> Dict[str, float]:
    """Raw elicited priority weights from the pairwise preference block.

    Each of the six ordered pairs contributes ``+1`` to the preferred objective
    and ``-1`` to the other; ties contribute nothing.  The resulting Borda
    scores are shifted to be strictly positive and renormalised to the simplex.

    The shift is deliberately small (``0.05`` on the score scale) so that an
    objective dominated in every pair keeps a near-zero weight.  That near-zero
    value is the elicitation artifact Section 4.2 is designed to correct; it is
    *not* smoothed away here.
    """
    pw = _read(Path(survey_dir), _PAIRWISE_FILE)
    scores: Dict[str, float] = {"time": 0.0, "cost": 0.0, "emissions": 0.0, "comfort": 0.0}
    pair_map = {
        "Time_vs_Cost": ("time", "cost"),
        "Time_vs_Emissions": ("time", "emissions"),
        "Time_vs_Comfort": ("time", "comfort"),
        "Cost_vs_Emissions": ("cost", "emissions"),
        "Cost_vs_Comfort": ("cost", "comfort"),
        "Emissions_vs_Comfort": ("emissions", "comfort"),
    }
    for col, (a, b) in pair_map.items():
        if col not in pw.columns:
            continue
        counts = pw[col].value_counts()
        first = counts.get("Prefer_First", 0)
        second = counts.get("Prefer_Second", 0)
        scores[a] += first - second
        scores[b] += second - first

    lowest = min(scores.values())
    shifted = {k: v - lowest + 0.05 for k, v in scores.items()}
    total = sum(shifted.values())
    return {k: v / total for k, v in shifted.items()}


# --------------------------------------------------------------------------
# 5. Provenance report
# --------------------------------------------------------------------------

def describe_survey(survey_dir: str | Path) -> Dict[str, object]:
    """Summarise the survey instrument as it actually is in the released data."""
    survey_dir = Path(survey_dir)
    demo = _read(survey_dir, _DEMO_FILE)
    scen = _read(survey_dir, _SCENARIO_FILE)
    per_respondent = scen.groupby("student_id").size()
    ratings = pd.to_numeric(scen["human_comfort_rating_1_5"], errors="coerce")

    return {
        "n_respondents": int(len(demo)),
        "n_trip_comfort_pairs": int(len(scen)),
        "scenarios_per_respondent": {
            "min": int(per_respondent.min()),
            "max": int(per_respondent.max()),
            "median": float(per_respondent.median()),
        },
        "comfort_rating_scale": {
            "column": "human_comfort_rating_1_5",
            "observed_min": float(ratings.min()),
            "observed_max": float(ratings.max()),
            "n_distinct_levels": int(ratings.nunique()),
            "normalization": "human_comfort_normalized = (rating - 1) / 4",
        },
        "distance_km": {
            "mean": float(demo["distance_km"].mean()),
            "std": float(demo["distance_km"].std()),
        },
        "corr_distance_budget": float(demo["distance_km"].corr(demo["max_budget_eur"])),
        "feature_provenance": {
            "recorded": ["mode_mix", "crowding", "transfers", "distance_km", "weather"],
            "joined_from_demographics": ["age", "mobility_restriction"],
            "derived_deterministically": {
                "rain": "weather in {rain, rainy}",
                "temperature_c": "weather mapped to September climate normals",
            },
            "randomly_generated": [],
        },
    }


# --------------------------------------------------------------------------
# 6. Bundle
# --------------------------------------------------------------------------

class SurveyData(NamedTuple):
    calibration: SurveyCalibration
    profiles: pd.DataFrame
    comfort_training: pd.DataFrame
    objective_weights: Dict[str, float]


def load_all(survey_dir: str | Path) -> SurveyData:
    survey_dir = Path(survey_dir)
    return SurveyData(
        calibration=load_real_survey_calibration(survey_dir),
        profiles=load_real_profiles(survey_dir),
        comfort_training=load_comfort_training_data(survey_dir),
        objective_weights=compute_objective_weights(survey_dir),
    )
