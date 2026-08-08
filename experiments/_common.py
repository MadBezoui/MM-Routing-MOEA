"""_common.py
==============
Shared setup for the sensitivity and ablation experiments of Sections 6.4-6.6.

Every experiment in this package starts from the same objects -- the survey,
the stabilized weights, the multimodal graph, the comfort surrogate and a
stratified profile subset -- so they are built once here.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import networkx as nx
import pandas as pd

from src.comfort_models import SurveyInformedComfortFactory, TrainedComfortPredictor
from src.config import (
    DEFAULT_COMFORT_CONFIG,
    DEFAULT_REFDIRS,
    DIVISIONS,
    ScenarioConfig,
    SurveyCalibration,
)
from src.pipeline_V6_smart import (
    assign_od_pairs,
    balanced_sample_by_group,
    build_problem_factory,
    build_reference_point_factory,
    recover_hv_igd_for_plan,
)
from src.optimization_framework_parallel3 import run_algorithm_suite_parallel3_checkpointed
from src.preferences.stabilization import WeightAudit, audit_and_stabilize_weights
from src.survey_data_loader import load_all

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Everything an experiment needs, built once."""

    survey: SurveyCalibration
    audit: WeightAudit
    graph: nx.MultiDiGraph
    profiles: pd.DataFrame
    comfort_results: list
    problem_factory: object
    ref_point_factory: object
    output_dir: Path

    @property
    def stabilized(self) -> List[float]:
        return self.audit.as_vector(stabilized=True)

    @property
    def raw(self) -> List[float]:
        return self.audit.as_vector(stabilized=False)


def build_context(
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    output_dir: str = "results/experiments",
    n_profiles: int = 30,
    comfort_model: str = "mlp_surrogate",
    random_state: int = 57,
    scenario: Optional[ScenarioConfig] = None,
) -> ExperimentContext:
    """Load the survey and graph, train the surrogate and draw a stratified subset."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    survey_data = load_all(Path(survey_dir))
    audit = audit_and_stabilize_weights(survey_data.objective_weights)

    G = nx.read_graphml(graph_path)
    profiles = assign_od_pairs(G, survey_data.profiles, random_state=42)

    main_profiles = balanced_sample_by_group(
        profiles, 150, ["archetype", "trip_distance_bin"], random_state=42,
    )
    subset = balanced_sample_by_group(
        main_profiles, n_profiles, ["archetype", "trip_distance_bin"],
        random_state=random_state,
    )

    factory = SurveyInformedComfortFactory(DEFAULT_COMFORT_CONFIG, survey_data.calibration)
    comfort_results = factory.train_models(survey_data.comfort_training)
    predictor = TrainedComfortPredictor(comfort_results, model_name=comfort_model)

    scenario = scenario or ScenarioConfig()
    problem_factory = build_problem_factory(G, survey_data.calibration, predictor, scenario)

    logger.info("Context ready: %d profiles, |V|=%d, comfort=%s",
                len(subset), G.number_of_nodes(), comfort_model)

    return ExperimentContext(
        survey=survey_data.calibration,
        audit=audit,
        graph=G,
        profiles=subset,
        comfort_results=comfort_results,
        problem_factory=problem_factory,
        ref_point_factory=build_reference_point_factory(survey_data.calibration),
        output_dir=out,
    )


def run_variant(
    ctx: ExperimentContext,
    variant_dir: Path,
    algorithms: Sequence[str],
    n_seeds: int,
    n_generations: int,
    plan: str = "sensitivity",
    n_partitions: Optional[int] = None,
    stabilized_weights: Optional[Sequence[float]] = None,
    raw_weights: Optional[Sequence[float]] = None,
    rho: float = DEFAULT_REFDIRS.rho,
    scenario: Optional[ScenarioConfig] = None,
    max_workers: int = 3,
    instrumented: bool = False,
    problem_factory=None,
) -> Optional[pd.DataFrame]:
    """Run one experimental variant and recover its normalized hypervolume."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    partitions = n_partitions if n_partitions is not None else DIVISIONS.get(plan, 8)

    for algorithm in algorithms:
        run_algorithm_suite_parallel3_checkpointed(
            problem_factory=problem_factory or ctx.problem_factory,
            profiles=ctx.profiles.to_dict(orient="records"),
            scenario=scenario or ScenarioConfig(),
            output_dir=str(variant_dir),
            algorithms=(algorithm,),
            seeds=tuple(range(n_seeds)),
            n_generations=n_generations,
            plan=plan,
            n_partitions=partitions,
            stabilized_weights=stabilized_weights if stabilized_weights is not None else ctx.stabilized,
            raw_weights=raw_weights if raw_weights is not None else ctx.raw,
            rho=rho,
            reference_point_factory=ctx.ref_point_factory,
            max_workers=max_workers,
            encoding="path",
            instrumented=instrumented,
            show_progress=False,
        )

    return recover_hv_igd_for_plan(variant_dir, variant_dir.parent)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--survey-dir", default="data/survey_results")
    parser.add_argument("--graph", default="data/processed/strasbourg_multimodal.graphml")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--comfort-model", default="mlp_surrogate",
                        choices=["mlp_surrogate", "linear_regression", "heuristic_direct"])
    return parser


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
    logger.info("wrote %s", path)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
