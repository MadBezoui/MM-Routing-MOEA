"""pipeline_V6_smart.py
=======================
Main orchestrator of the four experimental plans of Section 5.3.

Steps
-----
1. Load the survey (749 respondents) and derive the optimization profiles with
   their three feasibility bounds.
2. Audit and stabilize the elicited priority weights (Eq. 6-7).
3. Load the consolidated multimodal graph and assign each profile an
   origin-destination pair whose great-circle distance matches the commuting
   distance the respondent reported.
4. Build the stratified plans by Hamilton apportionment on the
   ``archetype x trip-distance`` grid.
5. Train the three comfort models on the real trip-comfort pairs.
6. Run every plan, resuming from existing checkpoints.
7. Recover hypervolume and IGD post hoc under the algorithm-agnostic
   normalization of Eq. 12-13.

Usage
-----
    python -m src.pipeline_V6_smart \\
        --survey-dir data/survey_results \\
        --graph      data/processed/strasbourg_multimodal.graphml \\
        --output-dir results/outputs_main \\
        --plans main extended convergence
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from src.comfort_models import (
    SurveyInformedComfortFactory,
    TrainedComfortPredictor,
)
from src.config import (
    DEFAULT_COMFORT_CONFIG,
    DEFAULT_EXPERIMENT,
    DEFAULT_NORMALIZATION,
    DEFAULT_REFDIRS,
    DEFAULT_SCENARIO,
    DIVISIONS,
    GENERATIONS,
    SEEDS,
    ComfortTrainingConfig,
    ScenarioConfig,
    canonical_algorithm,
)
from src.network.evaluator import PathMultimodalEvaluator
from src.optimization_framework_parallel3 import (
    ProfiledMultimodalProblem,
    run_algorithm_suite_parallel3_checkpointed,
)
from src.preferences.stabilization import WeightAudit, audit_and_stabilize_weights
from src.reference_directions import audit_reference_directions
from src.survey_data_loader import describe_survey, load_all

logger = logging.getLogger(__name__)

OBJECTIVE_COLUMNS = ["obj_1", "obj_2", "obj_3", "obj_4"]
KEY_COLS = ["profile_id", "algorithm", "seed"]
EARTH_RADIUS_KM = 6371.0088


# ==========================================================================
# 1. Stratified sampling by Hamilton apportionment
# ==========================================================================

def _hamilton_apportionment(weights: pd.Series, total: int, min_positive: int = 1) -> pd.Series:
    """Largest-remainder allocation of ``total`` units across strata."""
    weights = weights.astype(float).clip(lower=0)
    alloc = pd.Series(0, index=weights.index, dtype=int)
    positive = weights[weights > 0].index
    if len(positive) == 0:
        return alloc
    if min_positive > 0 and len(positive) <= total:
        alloc.loc[positive] = min(min_positive, total // len(positive))
    remaining = total - int(alloc.sum())
    if remaining <= 0:
        return alloc
    quotas = weights / weights.sum() * remaining
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
    """Stratified sample guaranteeing every non-empty cell is represented."""
    rng = np.random.default_rng(random_state)
    groups = df.groupby(list(group_cols), dropna=False)
    allocation = _hamilton_apportionment(groups.size().sort_index(), sample_size,
                                         min_positive=force_min_per_group)

    selected: List[pd.DataFrame] = []
    remaining_idx = set(df.index)
    for key, take in allocation.items():
        if take <= 0:
            continue
        group = groups.get_group(key)
        take = min(int(take), len(group))
        chosen = rng.choice(group.index.to_numpy(), size=take, replace=False)
        selected.append(df.loc[chosen])
        remaining_idx -= set(chosen)

    out = pd.concat(selected) if selected else df.head(0)
    if len(out) < sample_size and remaining_idx:
        extra = min(sample_size - len(out), len(remaining_idx))
        chosen = rng.choice(sorted(remaining_idx), size=extra, replace=False)
        out = pd.concat([out, df.loc[chosen]])
    return out.head(sample_size).reset_index(drop=True)


def balanced_representative_subset(
    df: pd.DataFrame,
    sample_size: int,
    archetype_col: str = "archetype",
    trip_col: str = "trip_distance_bin",
    random_state: int = 91,
) -> pd.DataFrame:
    """Ten-profile subset spanning every archetype and every distance bin."""
    return balanced_sample_by_group(
        df, sample_size, [archetype_col, trip_col],
        random_state=random_state, force_min_per_group=0,
    )


def strata_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format archetype and distance-bin counts (Table 5)."""
    rows = []
    for dim in ("archetype", "trip_distance_bin"):
        for level, count in df[dim].value_counts().items():
            rows.append({"dimension": dim, "level": level, "count": int(count)})
    return pd.DataFrame(rows)


# ==========================================================================
# 2. Origin-destination assignment
# ==========================================================================

def _node_coordinates(G: nx.MultiDiGraph) -> Tuple[List[str], np.ndarray]:
    nodes = list(G.nodes())
    coords = np.asarray([[float(G.nodes[n]["y"]), float(G.nodes[n]["x"])] for n in nodes])
    return nodes, coords


def _haversine_matrix(coords: np.ndarray, target: np.ndarray) -> np.ndarray:
    lat1 = np.radians(coords[:, 0])[:, None]
    lon1 = np.radians(coords[:, 1])[:, None]
    lat2 = np.radians(target[:, 0])[None, :]
    lon2 = np.radians(target[:, 1])[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def assign_od_pairs(
    G: nx.MultiDiGraph,
    profiles: pd.DataFrame,
    random_state: int = 42,
    n_destination_candidates: int = 40,
) -> pd.DataFrame:
    """Attach an origin and a destination node to every profile.

    The destination is drawn among the best-served facilities of the network,
    which stand for the campus and city-centre attractors of a commuting trip.
    The origin is then the node whose great-circle distance to that destination
    is closest to the commuting distance the respondent actually reported, so
    that the origin-destination distance distribution of the experiment
    reproduces the survey distribution rather than being uniform.
    """
    nodes, coords = _node_coordinates(G)
    service = np.asarray([float(G.nodes[n].get("n_trips", 0.0)) for n in nodes])
    attractors = np.argsort(-service)[:max(n_destination_candidates, 1)]

    rng = np.random.default_rng(random_state)
    out = profiles.copy()
    origins: List[str] = []
    destinations: List[str] = []
    realised: List[float] = []

    dest_choice = rng.choice(attractors, size=len(out), replace=True)
    dist_to_dest = _haversine_matrix(coords, coords[attractors])
    attractor_pos = {int(a): i for i, a in enumerate(attractors)}

    for row_i, (_, row) in enumerate(out.iterrows()):
        d_idx = int(dest_choice[row_i])
        column = dist_to_dest[:, attractor_pos[d_idx]]
        target = float(row["distance_km"])
        candidates = np.argsort(np.abs(column - target))
        
        # Enforce OD feasibility
        origin_idx = None
        for c in candidates:
            c = int(c)
            if c != d_idx and nx.has_path(G, nodes[c], nodes[d_idx]):
                origin_idx = c
                break
                
        if origin_idx is None:
            raise RuntimeError(f"No valid origin found with a path to destination {nodes[d_idx]}")
            
        origins.append(nodes[origin_idx])
        destinations.append(nodes[d_idx])
        realised.append(float(column[origin_idx]))

    out["origin_node"] = origins
    out["dest_node"] = destinations
    out["od_distance_km"] = realised
    logger.info(
        "OD assignment: realised distance mean %.2f km (std %.2f); reported mean %.2f km (std %.2f)",
        out["od_distance_km"].mean(), out["od_distance_km"].std(),
        out["distance_km"].mean(), out["distance_km"].std(),
    )
    return out


# ==========================================================================
# 3. Plans
# ==========================================================================

@dataclass
class RunPlan:
    name: str
    plan_type: str
    profiles_df: pd.DataFrame
    algorithms: Tuple[str, ...]
    seeds_by_algorithm: Dict[str, Sequence[int]]
    n_generations: int
    n_partitions: int


def build_plans(
    profiles_df: pd.DataFrame,
    which: Sequence[str] = ("main", "extended", "convergence", "ablation"),
    experiment=DEFAULT_EXPERIMENT,
) -> List[RunPlan]:
    """Construct the requested plans from nested stratified subsets."""
    main_profiles = balanced_sample_by_group(
        profiles_df, experiment.total_profiles_expected,
        ["archetype", "trip_distance_bin"], random_state=42, force_min_per_group=1,
    )
    extended_profiles = balanced_sample_by_group(
        main_profiles, experiment.extended_subset_size,
        ["archetype", "trip_distance_bin"], random_state=57, force_min_per_group=1,
    )
    ablation_profiles = balanced_sample_by_group(
        main_profiles, experiment.ablation_subset_size,
        ["archetype", "trip_distance_bin"], random_state=57, force_min_per_group=1,
    )
    convergence_profiles = balanced_representative_subset(
        main_profiles, experiment.representative_subset_size, random_state=91,
    )

    specs = {
        "main": ("main_pi_nsga3_vs_nsga2_150profiles", main_profiles, ("nsga2", "pi_nsga3")),
        "extended": ("extended_benchmark_30profiles", extended_profiles,
                     ("nsga2", "pi_nsga3", "moead", "smsemoa")),
        "convergence": ("convergence_curves_10profiles", convergence_profiles,
                        ("nsga2", "pi_nsga3")),
        "ablation": ("four_way_ablation_30profiles", ablation_profiles,
                     ("nsga2", "canonical_nsga3", "pi_nsga3_raw", "pi_nsga3_stab")),
    }

    plans: List[RunPlan] = []
    for key in which:
        if key not in specs:
            raise ValueError(f"unknown plan '{key}'")
        name, frame, algorithms = specs[key]
        plans.append(RunPlan(
            name=name,
            plan_type=key,
            profiles_df=frame,
            algorithms=algorithms,
            seeds_by_algorithm={a: tuple(range(SEEDS[key][a])) for a in algorithms},
            n_generations=GENERATIONS[key],
            n_partitions=DIVISIONS[key],
        ))
    return plans


# ==========================================================================
# 4. Problem factory
# ==========================================================================

class ReferencePointFactory:
    def __init__(self, survey):
        self.survey = survey
        
    def __call__(self, profile: Dict[str, object]) -> np.ndarray:
        d = float(profile.get("od_distance_km", profile.get("distance_km", 5.0)))
        return np.array([
            max(180.0, 60.0 * d / 3.5 + 30.0),
            max(15.0, 2.0 * float(profile.get("budget_eur", 6.0)) + 3.0),
            max(2.5, 0.22 * d + 0.25),
            1.05,
        ], dtype=float)

def build_reference_point_factory(survey):
    """Per-profile nadir used by the in-run instrumentation only."""
    return ReferencePointFactory(survey)


class ProblemFactory:
    def __init__(self, G, survey, comfort_predictor, scenario: ScenarioConfig):
        from src.network.operators import MultimodalIndex
        self.G = G
        self.survey = survey
        self.comfort_predictor = comfort_predictor
        self.scenario = scenario
        self.shared_index = MultimodalIndex(G)

    def __call__(self, profile: Dict[str, object], run_scenario: ScenarioConfig, seed: int = 0) -> ProfiledMultimodalProblem:
        sc = run_scenario or self.scenario
        evaluator = PathMultimodalEvaluator(
            self.G, self.survey, self.comfort_predictor, scenario=sc,
            n_monte_carlo=getattr(sc, "n_monte_carlo", 1),
            comfort_bias=getattr(sc, "comfort_bias", 0.0),
            algorithm_seed=seed,
        )
        problem = ProfiledMultimodalProblem(
            n_var=1, n_obj=4, xl=None, xu=None,
            evaluator=evaluator, profile=profile, extras={},
            scenario=sc, vtype=object,
        )
        problem.graph = self.G
        problem.graph_index = self.shared_index
        return problem

def build_problem_factory(G, survey, comfort_predictor, scenario: ScenarioConfig):
    """Return a factory producing one path-encoded problem per profile."""
    return ProblemFactory(G, survey, comfort_predictor, scenario)


# ==========================================================================
# 5. Hypervolume recovery (Eq. 12-13)
# ==========================================================================

def _coerce_bool_safe(series: pd.Series) -> pd.Series:
    """Robust boolean coercion.

    ``Series.astype(bool)`` returns ``True`` for any non-empty string,
    including ``"False"``.  Population checkpoints are reloaded from CSV, where
    the feasibility flag is a string, so the naive cast silently marks every
    individual feasible.
    """
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _non_dominated(points: np.ndarray) -> np.ndarray:
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = points - points[i]
        if np.any(np.all(diff <= 0, axis=1) & np.any(diff < 0, axis=1)):
            keep[i] = False
            continue
        diff_i = points[i] - points
        keep[np.all(diff_i <= 0, axis=1) & np.any(diff_i < 0, axis=1)] = False
    return keep


def _load_populations(plan_dir: Path) -> Optional[pd.DataFrame]:
    ckpt = plan_dir / "checkpoints" / "population"
    frames = []
    if ckpt.exists():
        for f in sorted(ckpt.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                # Filenames are formatted as: profile_id__algorithm__seed{seed}.csv
                # Extract the design seed from the filename to override internal hash_algo
                stem = f.stem
                if "__seed" in stem:
                    parts = stem.split("__")
                    if len(parts) >= 3:
                        design_seed = int(parts[-1].replace("seed", ""))
                        df["seed"] = design_seed
                frames.append(df)
            except Exception as exc:  # pragma: no cover - corrupt checkpoint
                logger.warning("skipping %s: %s", f.name, exc)
    return pd.concat(frames, ignore_index=True) if frames else None


def recover_hv_igd_for_plan(
    plan_dir: Path,
    output_dir: Path,
    epsilon: float = DEFAULT_NORMALIZATION.epsilon,
    schemes: Sequence[str] = (DEFAULT_NORMALIZATION.default_scheme,),
    survey_nadir: Optional[Sequence[float]] = None,
) -> Optional[pd.DataFrame]:
    """Recompute HV, IGD and normalized HV from the population checkpoints.

    The default scheme is Eq. 12-13: the reference point and the normalization
    denominator are both built from the union of *final feasible populations*
    across all algorithms and seeds of the profile, so neither favours any
    algorithm. The alternatives are the ones compared in Section 6.6.
    """
    plan_name = plan_dir.name
    populations = _load_populations(plan_dir)
    if populations is None or populations.empty:
        logger.warning("[skip] %s: no population data", plan_name)
        return None

    missing = [c for c in OBJECTIVE_COLUMNS + KEY_COLS if c not in populations.columns]
    if missing:
        logger.warning("[skip] %s: missing columns %s", plan_name, missing)
        return None

    populations["feasible"] = (
        _coerce_bool_safe(populations["feasible"])
        if "feasible" in populations.columns
        else pd.Series(True, index=populations.index)
    )
    populations["algorithm"] = populations["algorithm"].map(canonical_algorithm)

    records: List[Dict[str, object]] = []
    for profile_id, group in populations.groupby("profile_id"):
        feasible = group[group["feasible"]][OBJECTIVE_COLUMNS].dropna()
        if feasible.empty:
            continue
        union = feasible.to_numpy(dtype=float)
        nadir = union.max(axis=0) + epsilon                       # Eq. 12
        ideal = union.min(axis=0)
        reference_front = union[_non_dominated(union)]

        hv_ind = HV(ref_point=nadir)
        igd_ind = IGD(reference_front)

        fixed_ref = None
        if survey_nadir is not None:
            fixed_ref = np.asarray(survey_nadir, dtype=float)
        ideal_ref = ideal + (nadir - ideal) * 1.1

        for (algo, seed), run in group.groupby(["algorithm", "seed"]):
            run_f = run[run["feasible"]][OBJECTIVE_COLUMNS].dropna().to_numpy(dtype=float)
            if run_f.size == 0:
                hv = igd = np.nan
                hv_fixed = hv_ideal = np.nan
            else:
                hv = float(hv_ind(run_f))
                igd = float(igd_ind(run_f))
                hv_fixed = float(HV(ref_point=fixed_ref)(run_f)) if fixed_ref is not None else np.nan
                hv_ideal = float(HV(ref_point=ideal_ref)(run_f))
            records.append({
                "profile_id": profile_id, "algorithm": algo,
                "seed": int(seed) if pd.notna(seed) else seed,
                "hypervolume": hv, "igd": igd,
                "hv_fixed_survey_nadir": hv_fixed,
                "hv_ideal_bounded": hv_ideal,
                "n_feasible_run": int(len(run_f)),
                "n_nondom_union": int(len(reference_front)),
                **{f"ref_point_obj{j + 1}": float(nadir[j]) for j in range(len(nadir))},
            })

    if not records:
        logger.warning("[warn] %s: no metric records", plan_name)
        return None

    metrics = pd.DataFrame(records)

    # --- normalization schemes -------------------------------------------
    metrics["max_hv_profile"] = metrics.groupby("profile_id")["hypervolume"].transform("max")
    metrics["normalized_hv"] = metrics["hypervolume"] / metrics["max_hv_profile"].replace(0, np.nan)
    metrics["normalization_scheme"] = DEFAULT_NORMALIZATION.default_scheme

    if "per_algorithm_max" in schemes:
        per_algo = metrics.groupby(["profile_id", "algorithm"])["hypervolume"].transform("max")
        metrics["nhv_per_algorithm_max"] = metrics["hypervolume"] / per_algo.replace(0, np.nan)
    if "fixed_survey_nadir" in schemes and survey_nadir is not None:
        denom = metrics.groupby("profile_id")["hv_fixed_survey_nadir"].transform("max")
        metrics["nhv_fixed_survey_nadir"] = metrics["hv_fixed_survey_nadir"] / denom.replace(0, np.nan)
    if "ideal_bounded" in schemes:
        denom = metrics.groupby("profile_id")["hv_ideal_bounded"].transform("max")
        metrics["nhv_ideal_bounded"] = metrics["hv_ideal_bounded"] / denom.replace(0, np.nan)

    metrics["plan_name"] = plan_name
    out_file = output_dir / f"{plan_name}_final_generation_recovered.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_file, index=False)
    metrics.to_csv(plan_dir / out_file.name, index=False)

    summary = metrics.groupby("algorithm")["normalized_hv"].agg(["mean", "median", "std", "count"]).round(4)
    logger.info("[ok] %s -> %s", plan_name, out_file.name)
    for line in summary.to_string().splitlines():
        logger.info("    %s", line)
    return metrics


# ==========================================================================
# 6. Orchestration
# ==========================================================================

def execute_plan(
    plan: RunPlan,
    problem_factory,
    scenario: ScenarioConfig,
    output_path: Path,
    audit: WeightAudit,
    ref_point_factory,
    max_workers: int,
    rho: float = DEFAULT_REFDIRS.rho,
    instrumented: bool = True,
) -> None:
    """Run every algorithm of ``plan`` with its own seed count."""
    plan_dir = output_path / plan.name
    plan_dir.mkdir(parents=True, exist_ok=True)

    for algorithm in plan.algorithms:
        seeds = plan.seeds_by_algorithm[algorithm]
        if not seeds:
            continue
        logger.info(
            "[exec] %s | %s | seeds=%d | gens=%d | p=%d",
            plan.name, algorithm, len(seeds), plan.n_generations, plan.n_partitions,
        )
        run_algorithm_suite_parallel3_checkpointed(
            problem_factory=problem_factory,
            profiles=plan.profiles_df.to_dict(orient="records"),
            scenario=scenario,
            output_dir=str(plan_dir),
            algorithms=(algorithm,),
            seeds=seeds,
            n_generations=plan.n_generations,
            plan=plan.plan_type,
            n_partitions=plan.n_partitions,
            stabilized_weights=audit.as_vector(stabilized=True),
            raw_weights=audit.as_vector(stabilized=False),
            rho=rho,
            reference_point_factory=ref_point_factory,
            max_workers=max_workers,
            encoding="path",
            instrumented=instrumented,
        )


def run_pipeline(
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    output_dir: str = "results/outputs_main",
    plans: Sequence[str] = ("main", "extended", "convergence", "ablation"),
    max_workers: int = 3,
    comfort_model: str = "mlp_surrogate",
    comfort_cfg: ComfortTrainingConfig = DEFAULT_COMFORT_CONFIG,
    scenario: ScenarioConfig = DEFAULT_SCENARIO,
    rho: float = DEFAULT_REFDIRS.rho,
    instrumented: bool = True,
    profile_limit: Optional[int] = None,
    seed_limit: Optional[int] = None,
    generations_override: Optional[int] = None,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- survey --------------------------------------------------------
    logger.info("Loading survey from %s", survey_dir)
    survey_data = load_all(Path(survey_dir))
    survey = survey_data.calibration
    profiles_df = survey_data.profiles
    training_df = survey_data.comfort_training

    with open(output_path / "survey_calibration_real.json", "w", encoding="utf-8") as fh:
        json.dump(survey.__dict__, fh, indent=2, default=float)
    with open(output_path / "survey_instrument_report.json", "w", encoding="utf-8") as fh:
        json.dump(describe_survey(survey_dir), fh, indent=2)

    # --- weights -------------------------------------------------------
    audit = audit_and_stabilize_weights(survey_data.objective_weights)
    with open(output_path / "objective_weights_raw.json", "w", encoding="utf-8") as fh:
        json.dump(audit.raw_weights, fh, indent=2)
    with open(output_path / "objective_weights_stabilized.json", "w", encoding="utf-8") as fh:
        json.dump(audit.stabilized_weights, fh, indent=2)
    pd.DataFrame({"warning": audit.warnings}).to_csv(
        output_path / "objective_weight_warnings.csv", index=False)
    logger.info("raw        : %s", {k: round(v, 6) for k, v in audit.raw_weights.items()})
    logger.info("stabilized : %s", {k: round(v, 6) for k, v in audit.stabilized_weights.items()})

    # --- graph and OD assignment ---------------------------------------
    logger.info("Loading multimodal graph from %s", graph_path)
    G = nx.read_graphml(graph_path)
    logger.info("  |V|=%d, |E|=%d", G.number_of_nodes(), G.number_of_edges())
    profiles_df = assign_od_pairs(G, profiles_df, random_state=42)

    with open(output_path / "reference_direction_audit.json", "w", encoding="utf-8") as fh:
        json.dump(audit_reference_directions(4, DIVISIONS["main"],
                                             audit.as_vector(True), rho=rho), fh, indent=2)

    # --- plans ---------------------------------------------------------
    run_plans = build_plans(profiles_df, which=plans)
    for plan in run_plans:
        if profile_limit:
            plan.profiles_df = plan.profiles_df.head(profile_limit)
        if seed_limit:
            plan.seeds_by_algorithm = {
                a: tuple(s[:seed_limit]) for a, s in plan.seeds_by_algorithm.items()
            }
        if generations_override:
            plan.n_generations = int(generations_override)
    pd.concat([p.profiles_df.assign(run_plan=p.name) for p in run_plans],
              ignore_index=True).to_csv(output_path / "profiles_all_plans.csv", index=False)
    for plan in run_plans:
        strata_audit(plan.profiles_df).assign(plan_name=plan.name).to_csv(
            output_path / f"{plan.name}_sampling_audit.csv", index=False)

    # --- comfort models -------------------------------------------------
    logger.info("Training comfort models on %d real trip-comfort pairs", len(training_df))
    factory = SurveyInformedComfortFactory(comfort_cfg, survey)
    comfort_results = factory.train_models(training_df)
    pd.DataFrame([{**r.metrics, "model_name": r.model_name} for r in comfort_results]).to_csv(
        output_path / "comfort_model_comparison.csv", index=False)
    for result in comfort_results:
        result.region_metrics.to_csv(output_path / f"comfort_region_errors_{result.model_name}.csv", index=False)
        result.predictions.to_csv(output_path / f"comfort_predictions_{result.model_name}.csv", index=False)
        try:
            factory.noise_robustness(result, training_df).to_csv(
                output_path / f"comfort_noise_robustness_{result.model_name}.csv", index=False)
        except Exception as exc:  # pragma: no cover
            logger.warning("noise robustness failed for %s: %s", result.model_name, exc)

    predictor = TrainedComfortPredictor(comfort_results, model_name=comfort_model)
    problem_factory = build_problem_factory(G, survey, predictor, scenario)
    ref_point_factory = build_reference_point_factory(survey)

    # --- execute --------------------------------------------------------
    manifest: Dict[str, str] = {}
    for plan in run_plans:
        logger.info("=" * 70)
        logger.info("Plan %s: %d profiles, seeds %s", plan.name, len(plan.profiles_df),
                    {a: len(s) for a, s in plan.seeds_by_algorithm.items()})
        execute_plan(plan, problem_factory, scenario, output_path, audit,
                     ref_point_factory, max_workers, rho=rho, instrumented=instrumented)
        metrics = recover_hv_igd_for_plan(
            output_path / plan.name, output_path,
            schemes=DEFAULT_NORMALIZATION.schemes,
        )
        if metrics is not None:
            manifest[plan.name] = str(output_path / f"{plan.name}_final_generation_recovered.csv")

    with open(output_path / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Done. Results in %s", output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experimental plans of Section 5.3.")
    parser.add_argument("--survey-dir", default="data/survey_results")
    parser.add_argument("--graph", default="data/processed/strasbourg_multimodal.graphml")
    parser.add_argument("--output-dir", default="results/outputs_main")
    parser.add_argument("--plans", nargs="+",
                        default=["main", "extended", "convergence", "ablation"])
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--comfort-model", default="mlp_surrogate",
                        choices=["mlp_surrogate", "linear_regression", "heuristic_direct"])
    parser.add_argument("--rho", type=float, default=DEFAULT_REFDIRS.rho)
    parser.add_argument("--no-instrumentation", action="store_true",
                        help="disable the per-generation callback (Table 16, 'bare')")
    parser.add_argument("--profile-limit", type=int, default=None,
                        help="truncate every plan to this many profiles (smoke tests)")
    parser.add_argument("--seed-limit", type=int, default=None,
                        help="truncate every plan to this many seeds (smoke tests)")
    parser.add_argument("--generations", type=int, default=None,
                        help="override the generation budget (smoke tests)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    run_pipeline(
        survey_dir=args.survey_dir, graph_path=args.graph, output_dir=args.output_dir,
        plans=args.plans, max_workers=args.max_workers, comfort_model=args.comfort_model,
        rho=args.rho, instrumented=not args.no_instrumentation,
        profile_limit=args.profile_limit, seed_limit=args.seed_limit,
        generations_override=args.generations,
    )


if __name__ == "__main__":
    main()
