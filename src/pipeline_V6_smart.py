"""pipeline_V6_smart.py
======================
Single-file consolidated pipeline (V5 corrected + integrated HV/IGD recovery).

What it does in one run:
1. Loads the survey calibration (749 respondents).
2. Audits and stabilizes the priority weights (floor 0.08, blend uniform 0.20)
   to fix the "emissions ~ 0" pathology of V4 raw weights.
3. Builds three balanced stratified plans:
     - main: 150 profiles, NSGA-II + NSGA-III, 30 seeds each
     - extended: 30 profiles, NSGA-II + NSGA-III + MOEA/D (10 seeds each),
                 SMS-EMOA (5 seeds — reduced to save ~6h of wall-clock)
     - representative: 10 profiles, NSGA-II + NSGA-III, 30 seeds each
4. Trains the survey-calibrated comfort models (heuristic / linear / MLP).
5. Runs each plan with per-algorithm seed counts and 5 parallel workers.
   Resumes automatically from existing population checkpoints.
6. Recovers HV / IGD / normalized HV post-hoc from checkpoints, with the
   robust bool-dtype coercion that fixes the silent emptiness bug.
7. Prints a final per-plan, per-algorithm summary (mean / median / std / count
   of normalized HV).

Bugs fixed vs V4 / V5:
- Sampling: extended_benchmark used `head(30)` -> mono-archetype bias. Replaced
  by stratified `balanced_sample_by_group`.
- Priority weights: V4 raw weights gave emissions ~ 0.0001 -> NSGA-III ref
  directions were effectively 3-D. Stabilized via floor + uniform blend.
- HV/IGD silent emptiness: when populations were reloaded from CSV, the
  `feasible` column was string ("True"/"False"), not bool. `df.loc[<str>, ...]`
  produced empty selections. Fixed by `_coerce_bool_safe`.
- Per-algorithm seed control: SMS-EMOA was running 10 seeds at ~7 minutes each.
  Reduced to 5 seeds via separate suite calls (one per algorithm).

Run:
    python pipeline_V6_smart.py [--output_dir <dir>] [--survey_dir <dir>]
                                [--sms_seeds 5] [--max_workers 5]

If you point --output_dir to an existing directory containing partial
checkpoints (e.g. `outputs_v5_parallel_3threads_fixed`), already-completed
runs are skipped automatically.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from config import (
    DEFAULT_COMFORT_CONFIG,
    DEFAULT_SCENARIO,
    ComfortTrainingConfig,
    ScenarioConfig,
)
from comfort_models import SurveyInformedComfortFactory, apply_survey_informed_heuristics
from optimization_framework_parallel3 import (
    ProfiledMultimodalProblem,
    run_algorithm_suite_parallel3_checkpointed,
)
from survey_data_loader import load_all


OBJECTIVE_COLUMNS = ["obj_1", "obj_2", "obj_3", "obj_4"]
KEY_COLS = ["profile_id", "algorithm", "seed"]


# ---------------------------------------------------------------------------
# Section 1 - Hamilton apportionment + stratified sampling
# ---------------------------------------------------------------------------

def _hamilton_apportionment(weights: pd.Series, total: int, min_positive: int = 1) -> pd.Series:
    weights = weights.astype(float).clip(lower=0)
    alloc = pd.Series(0, index=weights.index, dtype=int)
    positive = weights[weights > 0].index
    if len(positive) == 0:
        return alloc
    if min_positive > 0:
        base_take = min(min_positive, total // len(positive)) if len(positive) <= total else 0
        if base_take > 0:
            alloc.loc[positive] = base_take
    remaining = total - int(alloc.sum())
    if remaining <= 0:
        return alloc
    frac = weights / weights.sum()
    quotas = frac * remaining
    floors = np.floor(quotas).astype(int)
    alloc += floors
    still = total - int(alloc.sum())
    if still > 0:
        remainders = (quotas - floors).sort_values(ascending=False)
        alloc.loc[remainders.index[:still]] += 1
    return alloc


def balanced_sample_by_group(
    df: pd.DataFrame,
    sample_size: int,
    group_cols: Sequence[str],
    random_state: int = 42,
    force_min_per_group: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    groups = df.groupby(list(group_cols), dropna=False)
    group_sizes = groups.size().sort_index()
    allocation = _hamilton_apportionment(group_sizes, sample_size, min_positive=force_min_per_group)

    selected = []
    remaining_idx = set(df.index)
    for group_key, take in allocation.items():
        if take <= 0:
            continue
        group = groups.get_group(group_key)
        take = min(int(take), len(group))
        chosen = rng.choice(group.index.to_numpy(), size=take, replace=False)
        selected.append(df.loc[chosen])
        remaining_idx -= set(chosen)

    selected_df = pd.concat(selected, ignore_index=False) if selected else df.head(0)
    if len(selected_df) < sample_size and remaining_idx:
        missing = sample_size - len(selected_df)
        remaining = df.loc[sorted(remaining_idx)]
        extra_take = min(missing, len(remaining))
        chosen = rng.choice(remaining.index.to_numpy(), size=extra_take, replace=False)
        selected_df = pd.concat([selected_df, df.loc[chosen]], ignore_index=False)

    return selected_df.head(sample_size).reset_index(drop=True)


def balanced_representative_subset(
    df: pd.DataFrame,
    sample_size: int,
    archetype_col: str = "archetype",
    trip_col: str = "trip_distance_bin",
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    archetypes = list(df[archetype_col].dropna().unique())
    n_arch = max(len(archetypes), 1)
    base = max(1, sample_size // n_arch)

    chosen_parts = []
    chosen_ids = set()
    for archetype in archetypes:
        subset = df[df[archetype_col] == archetype]
        trips = list(subset[trip_col].dropna().unique())
        take_rows = []
        for trip in trips:
            cell = subset[subset[trip_col] == trip]
            if len(cell) == 0:
                continue
            idx = rng.choice(cell.index.to_numpy(), size=1, replace=False)[0]
            take_rows.append(df.loc[[idx]])
            chosen_ids.add(df.loc[idx, "profile_id"])
            if len(take_rows) >= base:
                break
        if take_rows:
            chosen_parts.append(pd.concat(take_rows, ignore_index=False))

    selected = pd.concat(chosen_parts, ignore_index=False) if chosen_parts else df.head(0)
    if len(selected) < sample_size:
        remaining = df.loc[~df["profile_id"].isin(chosen_ids)]
        extra = balanced_sample_by_group(
            remaining,
            sample_size=sample_size - len(selected),
            group_cols=[archetype_col, trip_col],
            random_state=random_state + 17,
            force_min_per_group=0,
        )
        selected = pd.concat([selected, extra], ignore_index=False)

    return selected.head(sample_size).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 2 - Weight audit and stabilization
# ---------------------------------------------------------------------------

@dataclass
class WeightAudit:
    raw_weights: Dict[str, float]
    stabilized_weights: Dict[str, float]
    warnings: List[str]


def audit_and_stabilize_weights(
    raw_weights: Dict[str, float],
    floor: float = 0.08,
    blend_uniform: float = 0.20,
) -> WeightAudit:
    keys = ["time", "cost", "emissions", "comfort"]
    raw = {k: float(raw_weights.get(k, 0.0)) for k in keys}
    total = sum(raw.values())
    warnings: List[str] = []

    if total <= 0:
        warnings.append("All raw objective weights are non-positive; falling back to uniform.")
        raw = {k: 1.0 / len(keys) for k in keys}
    else:
        raw = {k: v / total for k, v in raw.items()}

    if raw["emissions"] < 0.02:
        warnings.append("Emissions weight is near zero; check survey aggregation.")
    if raw["comfort"] < 0.20:
        warnings.append("Comfort weight is unexpectedly low.")
    if max(raw.values()) > 0.75:
        warnings.append("One objective dominates; stabilization will reduce its grip on ref directions.")

    uniform = {k: 1.0 / len(keys) for k in keys}
    blended = {k: (1.0 - blend_uniform) * raw[k] + blend_uniform * uniform[k] for k in keys}
    floored = {k: max(v, floor) for k, v in blended.items()}
    norm = sum(floored.values())
    stabilized = {k: v / norm for k, v in floored.items()}
    return WeightAudit(raw_weights=raw, stabilized_weights=stabilized, warnings=warnings)


# ---------------------------------------------------------------------------
# Section 3 - Reference point factory (per-profile nadir for HV during run)
# ---------------------------------------------------------------------------

def build_reference_point_factory(survey):
    def _factory(profile: Dict[str, object]) -> np.ndarray:
        d = float(profile.get("distance_km", getattr(survey, "mean_distance_to_campus_km", 5.0)))
        budget = float(profile.get("budget_eur", getattr(survey, "mean_daily_budget_eur", 6.0)))
        max_time = max(180.0, 60.0 * d / 3.5 + 30.0)
        max_cost = max(15.0, 2.0 * budget + 3.0)
        max_emissions = max(2.5, 0.22 * d + 0.25)
        max_discomfort = 1.05
        ret = np.array([max_time, max_cost, max_emissions, max_discomfort], dtype=float)
        assert ret.shape == (4,)
        assert np.all(np.isfinite(ret))
        assert np.all(ret > 0)
        return ret
    return _factory


# ---------------------------------------------------------------------------
# Section 4 - Trained comfort predictor + multimodal evaluator
# ---------------------------------------------------------------------------

class TrainedComfortPredictor:
    def __init__(self, comfort_results):
        mlp = [r for r in comfort_results if r.model_name == "mlp_surrogate"]
        self.mlp_pipeline = mlp[0].pipeline if mlp else None

    def predict(self, comfort_df: pd.DataFrame, survey) -> np.ndarray:
        if self.mlp_pipeline is None:
            return apply_survey_informed_heuristics(comfort_df, survey).to_numpy()
        preds = self.mlp_pipeline.predict(comfort_df)
        return np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)


class RealSurveyMultimodalEvaluator:
    MODES = np.array(["walk", "bike", "bus", "tram", "car"])
    EMISSION_FACTORS = {"walk": 0.0, "bike": 0.0, "bus": 0.080, "tram": 0.035, "car": 0.150}
    BASE_COST = {"walk": 0.0, "bike": 0.15, "bus": 1.5, "tram": 1.7, "car": 0.28}
    BASE_SPEED_KMPH = {"walk": 4.5, "bike": 14.0, "bus": 20.0, "tram": 24.0, "car": 26.0}

    def __init__(self, survey, comfort_predictor: TrainedComfortPredictor):
        self.survey = survey
        self.comfort_predictor = comfort_predictor

    def __call__(self, X, profile, extras, scenario: ScenarioConfig):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, 0, 1)
        shares = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)

        distance_km = float(profile.get("distance_km", self.survey.mean_distance_to_campus_km))
        age = float(profile.get("age", self.survey.mean_age))
        budget = float(profile.get("budget_eur", self.survey.mean_daily_budget_eur))
        rain = float(profile.get("rain", 0))
        mobility_restriction = int(profile.get("mobility_restriction", 0))

        transfers = np.round(2 * shares[:, 2] + 2 * shares[:, 3] + shares[:, 4]).astype(int)
        crowding = np.clip(0.7 * shares[:, 2] + 0.5 * shares[:, 3] + 0.2 * shares[:, 4], 0, 1)
        reliability_penalty = np.clip(0.6 * shares[:, 2] + 0.35 * shares[:, 3] + 0.15 * shares[:, 4], 0, 1)
        safety_penalty = np.clip(0.35 * shares[:, 0] + 0.15 * shares[:, 1] + 0.08 * shares[:, 2], 0, 1)

        rng = np.random.default_rng(int(profile.get("seed_offset", 0)) + 123)
        congestion = (
            rng.normal(scenario.congestion_multiplier_mean, scenario.congestion_multiplier_std, size=len(shares))
            if scenario.stochastic_travel_time else np.ones(len(shares))
        )
        occupancy = (
            rng.normal(scenario.occupancy_multiplier_mean, scenario.occupancy_multiplier_std, size=len(shares))
            if scenario.stochastic_crowding else np.ones(len(shares))
        )
        pricing = (
            rng.normal(scenario.pricing_multiplier_mean, scenario.pricing_multiplier_std, size=len(shares))
            if scenario.dynamic_pricing else np.ones(len(shares))
        )

        speed = np.zeros(len(shares))
        cost = np.zeros(len(shares))
        emissions = np.zeros(len(shares))
        for idx, mode in enumerate(self.MODES):
            speed += shares[:, idx] * self.BASE_SPEED_KMPH[mode]
            cost += shares[:, idx] * self.BASE_COST[mode]
            emissions += shares[:, idx] * self.EMISSION_FACTORS[mode]

        speed = np.maximum(speed / np.maximum(congestion, 0.4), 1.0)
        travel_time_min = 60 * distance_km / speed + 4 * transfers
        cost = cost * distance_km * np.maximum(pricing, 0.5)
        emissions = emissions * distance_km * np.maximum(occupancy, 0.5)

        comfort_df = pd.DataFrame({
            "walk_share": shares[:, 0], "bike_share": shares[:, 1],
            "bus_share": shares[:, 2], "tram_share": shares[:, 3],
            "car_share": shares[:, 4],
            "crowding": np.clip(crowding * occupancy, 0, 1),
            "transfers": transfers, "distance_km": distance_km,
            "rain": rain, "temperature_c": float(profile.get("temperature_c", 14.0)),
            "age": age, "mobility_restriction": mobility_restriction,
            "reliability_penalty": reliability_penalty,
            "safety_penalty": safety_penalty,
            "fare_eur": cost, "travel_time_min": travel_time_min,
            "weather_label": np.where(rain > 0.5, "rainy", "dry"),
            "dominant_mode": self.MODES[np.argmax(shares, axis=1)],
        })
        comfort_score = self.comfort_predictor.predict(comfort_df, self.survey)

        max_time = float(profile.get("max_time_min", 120.0))
        max_walk = float(profile.get("max_walk_km", self.survey.walking_threshold_km))
        
        g1_budget = np.maximum(0, (cost - budget) / max(budget, 0.1))
        g2_time = np.maximum(0, (travel_time_min - max_time) / max(max_time, 1.0))
        g3_walk = np.maximum(0, (shares[:, 0] * distance_km - max_walk) / max(max_walk, 0.1))

        F = np.column_stack([travel_time_min, cost, emissions, 1.0 - comfort_score])
        G = np.column_stack([g1_budget, g2_time, g3_walk])
        meta = {
            "dominant_mode": self.MODES[np.argmax(shares, axis=1)],
            "travel_time_min": travel_time_min,
            "cost": cost,
            "emissions": emissions,
            "comfort_score": comfort_score,
        }
        return F, G, meta


def build_problem_factory(survey, comfort_predictor, n_var: int = 5, n_obj: int = 4, evaluator_type: str = "continuous", G=None):
    if evaluator_type != "discrete":
        evaluator = RealSurveyMultimodalEvaluator(survey, comfort_predictor)

    def _factory(profile, scenario: ScenarioConfig) -> ProfiledMultimodalProblem:
        if evaluator_type == "discrete" and G is not None:
            # Subgraph extraction for performance
            O = profile.get('origin_node')
            D = profile.get('dest_node')
            if O and D and O in G and D in G:
                o_data = G.nodes[O]
                d_data = G.nodes[D]
                # Fallback coordinates if x, y are not present (though OSMnx guarantees them)
                ox, oy = float(o_data.get('x', 0)), float(o_data.get('y', 0))
                dx, dy = float(d_data.get('x', 0)), float(d_data.get('y', 0))
                
                min_x, max_x = min(ox, dx) - 0.05, max(ox, dx) + 0.05
                min_y, max_y = min(oy, dy) - 0.05, max(oy, dy) + 0.05
                
                sub_nodes = [n for n, d in G.nodes(data=True) if min_x <= float(d.get('x', 0)) <= max_x and min_y <= float(d.get('y', 0)) <= max_y]
                subG = G.subgraph(sub_nodes).copy()
            else:
                subG = G
            
            import copy
            scenario_copy = copy.copy(scenario)
            scenario_copy.G = subG
            
            from src.network.evaluator import PathMultimodalEvaluator
            local_evaluator = PathMultimodalEvaluator(subG, survey, comfort_predictor)
            local_n_var = 1
        else:
            scenario_copy = scenario
            local_evaluator = evaluator
            local_n_var = n_var
            
        return ProfiledMultimodalProblem(
            n_var=local_n_var, n_obj=n_obj,
            xl=[0.0] * local_n_var if evaluator_type != "discrete" else None, 
            xu=[1.0] * local_n_var if evaluator_type != "discrete" else None,
            evaluator=local_evaluator,
            profile=profile, extras={}, scenario=scenario_copy,
            n_ieq_constr=4 if evaluator_type == "discrete" else 3,
        )

    return _factory

VALID_ALGORITHMS = {
    "nsga2",
    "canonical_nsga3",
    "pi_nsga3_raw",
    "pi_nsga3_stab",
    "moead",
    "smsemoa",
}


# ---------------------------------------------------------------------------
# Section 5 - Smart run plans (per-algorithm seed counts)
# ---------------------------------------------------------------------------

@dataclass
class SmartRunPlan:
    name: str
    profiles_df: pd.DataFrame
    seeds_by_algorithm: Dict[str, Sequence[int]]
    n_generations: int
    population_size: int
    nsga3_partitions: int


def build_smart_plans(profiles_df: pd.DataFrame, sms_seeds: int = 5) -> List[SmartRunPlan]:
    main_profiles = balanced_sample_by_group(
        profiles_df, sample_size=150,
        group_cols=["archetype", "trip_distance_bin"],
        random_state=42, force_min_per_group=1,
    )
    extended_profiles = balanced_sample_by_group(
        main_profiles, sample_size=30,
        group_cols=["archetype", "trip_distance_bin"],
        random_state=57, force_min_per_group=1,
    )
    representative_profiles = balanced_representative_subset(
        main_profiles, sample_size=10,
        archetype_col="archetype", trip_col="trip_distance_bin",
        random_state=91,
    )
    return [
        SmartRunPlan(
            name="main_nsga2_vs_nsga3_150profiles",
            profiles_df=main_profiles,
            seeds_by_algorithm={
                "nsga2": tuple(range(20)),
                "pi_nsga3_stab": tuple(range(20)),
            },
            n_generations=150, population_size=168, nsga3_partitions=8,
        ),
        SmartRunPlan(
            name="extended_benchmark_30profiles",
            profiles_df=extended_profiles,
            seeds_by_algorithm={
                "nsga2": tuple(range(20)),
                "canonical_nsga3": tuple(range(20)),
                "pi_nsga3_raw": tuple(range(20)),
                "pi_nsga3_stab": tuple(range(20)),
                "moead": tuple(range(20)),
                "smsemoa": tuple(range(sms_seeds)),
            },
            n_generations=120, population_size=128, nsga3_partitions=7,
        ),
        SmartRunPlan(
            name="representative_curves_10profiles",
            profiles_df=representative_profiles,
            seeds_by_algorithm={
                "nsga2": tuple(range(20)),
                "pi_nsga3_stab": tuple(range(20)),
            },
            n_generations=150, population_size=168, nsga3_partitions=8,
        ),
    ]


def print_strata_audit(plan: SmartRunPlan) -> None:
    df = plan.profiles_df
    if "archetype" in df.columns and "trip_distance_bin" in df.columns:
        table = df.groupby(["archetype", "trip_distance_bin"]).size().unstack(fill_value=0)
        print(f"  [strata audit] {plan.name}:")
        for line in table.to_string().splitlines():
            print(f"    {line}")
        print(f"    archetypes={df['archetype'].nunique()}, trip_bins={df['trip_distance_bin'].nunique()}")


# ---------------------------------------------------------------------------
# Section 6 - HV/IGD recovery (with bool-dtype bug fix)
# ---------------------------------------------------------------------------

def _coerce_bool_safe(series: pd.Series) -> pd.Series:
    """Robustly convert to bool. Series.astype(bool) returns True for any
    non-empty string (including 'False'), which is the silent bug fixed here.
    """
    if series.dtype == bool:
        return series
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"true", "1", "yes", "y", "t"})


def _is_nondominated(points: np.ndarray) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = points - points[i]
        dominates_i = np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)
        if np.any(dominates_i):
            keep[i] = False
            continue
        diff_i = points[i] - points
        dominated_by_i = np.all(diff_i <= 0, axis=1) & np.any(diff_i < 0, axis=1)
        keep[dominated_by_i] = False
    return keep


def _load_populations(plan_dir: Path):
    ckpt_dir = plan_dir / "checkpoints" / "population"
    frames = []
    if ckpt_dir.exists():
        for f in sorted(ckpt_dir.glob("*.csv")):
            try:
                frames.append(pd.read_csv(f))
            except Exception as exc:
                print(f"    [warn] {f.name}: {exc}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    fallback = plan_dir / "all_population_results.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    return None


def recover_hv_igd_for_plan(plan_dir: Path, output_dir: Path, epsilon: float = 1e-6):
    plan_name = plan_dir.name
    populations_df = _load_populations(plan_dir)
    if populations_df is None or len(populations_df) == 0:
        print(f"  [skip] {plan_name}: no population data")
        return None

    missing = [c for c in OBJECTIVE_COLUMNS + KEY_COLS if c not in populations_df.columns]
    if missing:
        print(f"  [skip] {plan_name}: missing columns {missing}")
        return None

    # >>> THE BUG FIX <<<
    populations_df["feasible"] = (
        _coerce_bool_safe(populations_df["feasible"])
        if "feasible" in populations_df.columns
        else pd.Series(True, index=populations_df.index)
    )

    records = []
    for profile_id, pop_grp in populations_df.groupby("profile_id"):
        feas = pop_grp[pop_grp["feasible"]][OBJECTIVE_COLUMNS].dropna()
        if len(feas) == 0:
            continue
        union = feas.to_numpy(dtype=float)
        nadir = union.max(axis=0) + epsilon
        nondom = _is_nondominated(union)
        ref_front = union[nondom] if nondom.any() else union
        hv_ind = HV(ref_point=nadir)
        igd_ind = IGD(ref_front)

        for (algo, seed), run_pop in pop_grp.groupby(["algorithm", "seed"]):
            run_feas = run_pop[run_pop["feasible"]][OBJECTIVE_COLUMNS].dropna().to_numpy(dtype=float)
            if len(run_feas) == 0:
                hv_val, igd_val = np.nan, np.nan
            else:
                hv_val = float(hv_ind(run_feas))
                igd_val = float(igd_ind(run_feas))
            records.append({
                "profile_id": profile_id,
                "algorithm": algo,
                "seed": int(seed) if pd.notna(seed) else seed,
                "hypervolume": hv_val,
                "igd": igd_val,
                "n_feasible_run": int(len(run_feas)),
                "n_nondom_union": int(nondom.sum()),
                "ref_point_obj1": float(nadir[0]),
                "ref_point_obj2": float(nadir[1]),
                "ref_point_obj3": float(nadir[2]),
                "ref_point_obj4": float(nadir[3]),
            })

    if not records:
        print(f"  [warn] {plan_name}: no metric records")
        return None

    metrics_df = pd.DataFrame(records)
    metrics_df["max_hv_profile"] = metrics_df.groupby("profile_id")["hypervolume"].transform("max")
    metrics_df["normalized_hv"] = metrics_df["hypervolume"] / metrics_df["max_hv_profile"].replace(0, np.nan)
    metrics_df["normalization_scheme"] = "union_observed_max_per_profile"
    metrics_df["plan_name"] = plan_name

    plan_out = plan_dir / f"{plan_name}_final_generation_recovered.csv"
    flat_out = output_dir / f"{plan_name}_final_generation_recovered.csv"
    metrics_df.to_csv(plan_out, index=False)
    metrics_df.to_csv(flat_out, index=False)

    summary = (
        metrics_df.groupby("algorithm")["normalized_hv"]
        .agg(["mean", "median", "std", "count"])
        .round(4)
    )
    print(f"  [ok] {plan_name}: {len(metrics_df)} rows -> {flat_out.name}")
    for line in summary.to_string().splitlines():
        print(f"    {line}")
    return metrics_df


# ---------------------------------------------------------------------------
# Section 7 - Plan execution (one suite call per algorithm to honor seed counts)
# ---------------------------------------------------------------------------

def execute_plan(
    plan: SmartRunPlan,
    problem_factory,
    scenario: ScenarioConfig,
    output_path: Path,
    priority_weights: Sequence[float],
    ref_point_factory,
    max_workers: int,
    plan_type: str = "main",
) -> None:
    plan_dir = output_path / plan.name
    plan_dir.mkdir(parents=True, exist_ok=True)

    for algo, seeds in plan.seeds_by_algorithm.items():
        if len(seeds) == 0:
            continue
        if algo not in VALID_ALGORITHMS:
            raise ValueError(f"Ambiguous identifier '{algo}'. Use 'canonical_nsga3', 'pi_nsga3_raw', or 'pi_nsga3_stab'.")
            
        print(f"  [exec] {plan.name} | algo={algo} | seeds={len(seeds)} | gens={plan.n_generations} | pop={plan.population_size} | workers={max_workers}")
        try:
            run_algorithm_suite_parallel3_checkpointed(
                problem_factory=problem_factory,
                profiles=plan.profiles_df.to_dict(orient="records"),
                scenario=scenario,
                output_dir=str(plan_dir),
                algorithms=(algo,),
                seeds=seeds,
                n_generations=plan.n_generations,
                population_size=plan.population_size,
                n_partitions=plan.nsga3_partitions,
                crossover_prob=0.9,
                crossover_eta=15.0,
                mutation_eta=20.0,
                priority_weights=priority_weights,
                reference_point_factory=ref_point_factory,
                max_workers=max_workers,
                plan=plan_type,
            )
        except TypeError:
            # Older framework signatures may not accept reference_point_factory
            # or max_workers; retry with a minimal call.
            run_algorithm_suite_parallel3_checkpointed(
                problem_factory=problem_factory,
                profiles=plan.profiles_df.to_dict(orient="records"),
                scenario=scenario,
                output_dir=str(plan_dir),
                algorithms=(algo,),
                seeds=seeds,
                n_generations=plan.n_generations,
                population_size=plan.population_size,
                n_partitions=plan.nsga3_partitions,
                crossover_prob=0.9,
                crossover_eta=15.0,
                mutation_eta=20.0,
                priority_weights=priority_weights,
                plan=plan_type,
            )


# ---------------------------------------------------------------------------
# Section 8 - Main orchestrator
# ---------------------------------------------------------------------------

def run_smart_pipeline(
    output_dir: str = "outputs_v5_parallel_3threads_fixed",
    survey_dir: str = "survey_results",
    sms_seeds: int = 5,
    max_workers: int = 5,
    comfort_cfg: ComfortTrainingConfig = DEFAULT_COMFORT_CONFIG,
    scenario: ScenarioConfig = DEFAULT_SCENARIO,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"[V6-smart] output_dir={output_dir} | workers={max_workers} | sms_seeds={sms_seeds}")
    print("=" * 78)

    # --- Load survey ---
    print(f"[V6-smart] Loading survey from {survey_dir} ...")
    survey_data = load_all(Path(survey_dir))
    real_survey = survey_data.calibration
    profiles_df = survey_data.profiles.copy()
    real_training_df = survey_data.comfort_training.copy()
    raw_obj_weights = survey_data.objective_weights

    with open(output_path / "survey_calibration_real.json", "w", encoding="utf-8") as f:
        json.dump(real_survey.__dict__, f, indent=2, default=float)

    # --- Stabilize weights ---
    weight_audit = audit_and_stabilize_weights(raw_obj_weights)
    with open(output_path / "objective_weights_raw.json", "w", encoding="utf-8") as f:
        json.dump(weight_audit.raw_weights, f, indent=2)
    with open(output_path / "objective_weights_stabilized.json", "w", encoding="utf-8") as f:
        json.dump(weight_audit.stabilized_weights, f, indent=2)
    pd.DataFrame({"warning": weight_audit.warnings}).to_csv(
        output_path / "objective_weight_warnings.csv", index=False
    )
    print("[V6-smart] Priority weights:")
    print(f"  raw        : {weight_audit.raw_weights}")
    print(f"  stabilized : {weight_audit.stabilized_weights}")
    if weight_audit.warnings:
        for w in weight_audit.warnings:
            print(f"  [!] {w}")

    # --- Build plans ---
    plans = build_smart_plans(profiles_df, sms_seeds=sms_seeds)
    pd.concat(
        [p.profiles_df.assign(run_plan=p.name) for p in plans],
        ignore_index=True,
    ).to_csv(output_path / "profiles_all_plans.csv", index=False)

    print("\n[V6-smart] Strata audit per plan:")
    for plan in plans:
        print_strata_audit(plan)

    # --- Train comfort models ---
    print("\n[V6-smart] Training comfort models on the real survey training set ...")
    comfort_factory = SurveyInformedComfortFactory(comfort_cfg, real_survey)
    comfort_factory.generate_synthetic_dataset = lambda n_samples=None, seed=None: real_training_df.copy()
    comfort_results = comfort_factory.train_models(real_training_df)
    pd.DataFrame([{**r.metrics, "model_name": r.model_name} for r in comfort_results]).to_csv(
        output_path / "comfort_model_comparison.csv", index=False
    )
    for result in comfort_results:
        result.region_metrics.to_csv(output_path / f"comfort_region_errors_{result.model_name}.csv", index=False)
        result.predictions.to_csv(output_path / f"comfort_predictions_{result.model_name}.csv", index=False)
        try:
            comfort_factory.noise_robustness(result, real_training_df).to_csv(
                output_path / f"comfort_noise_robustness_{result.model_name}.csv", index=False
            )
        except Exception as exc:
            print(f"  [warn] noise_robustness failed for {result.model_name}: {exc}")

    comfort_predictor = TrainedComfortPredictor(comfort_results)
    problem_factory = build_problem_factory(real_survey, comfort_predictor)

    priority_weights = [
        weight_audit.stabilized_weights["time"],
        weight_audit.stabilized_weights["cost"],
        weight_audit.stabilized_weights["emissions"],
        weight_audit.stabilized_weights["comfort"],
    ]
    ref_point_factory = build_reference_point_factory(real_survey)

    # --- Execute plans + recover ---
    summaries = {}
    manifest = {
        "survey_calibration_real": str(output_path / "survey_calibration_real.json"),
        "objective_weights_raw": str(output_path / "objective_weights_raw.json"),
        "objective_weights_stabilized": str(output_path / "objective_weights_stabilized.json"),
        "comfort_model_comparison": str(output_path / "comfort_model_comparison.csv"),
        "profiles_all_plans": str(output_path / "profiles_all_plans.csv"),
    }

    for plan in plans:
        print("\n" + "-" * 78)
        print(f"[V6-smart] Plan: {plan.name}")
        print(f"  profiles={len(plan.profiles_df)} | seeds={ {a: len(s) for a, s in plan.seeds_by_algorithm.items()} }")
        print("-" * 78)
        execute_plan(
            plan=plan,
            problem_factory=problem_factory,
            scenario=scenario,
            output_path=output_path,
            priority_weights=priority_weights,
            ref_point_factory=ref_point_factory,
            max_workers=max_workers,
        )
        print(f"\n[V6-smart] Recovering HV/IGD for {plan.name} ...")
        metrics_df = recover_hv_igd_for_plan(
            plan_dir=output_path / plan.name,
            output_dir=output_path,
        )
        if metrics_df is not None:
            summaries[plan.name] = (
                metrics_df.groupby("algorithm")["normalized_hv"]
                .agg(["mean", "median", "std", "count"])
                .round(4)
            )
            manifest[f"{plan.name}_final_generation_recovered"] = str(
                output_path / f"{plan.name}_final_generation_recovered.csv"
            )

    # --- Final manifest + summary ---
    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    print("[V6-smart] FINAL SUMMARY")
    print("=" * 78)
    for name, sm in summaries.items():
        print(f"\n{name}:")
        for line in sm.to_string().splitlines():
            print(f"  {line}")
    print(f"\n[V6-smart] Done. Results in '{output_dir}/'.")
    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V6 smart pipeline: V5 corrected + integrated HV/IGD recovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", default="outputs_v5_parallel_3threads_fixed",
                        help="Output directory (point to existing one to resume).")
    parser.add_argument("--survey_dir", default="survey_results",
                        help="Path to survey calibration results.")
    parser.add_argument("--sms_seeds", type=int, default=5,
                        help="Number of seeds for SMS-EMOA in extended benchmark (default 5).")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="Number of parallel worker threads (default 5).")
    args = parser.parse_args()

    run_smart_pipeline(
        output_dir=args.output_dir,
        survey_dir=args.survey_dir,
        sms_seeds=args.sms_seeds,
        max_workers=args.max_workers,
    )
