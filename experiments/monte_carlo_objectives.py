"""monte_carlo_objectives.py
=============================
Monte Carlo evaluation of the stochastic objectives (Section 6.6).

The main experiments are deterministic: the fare-variability factor
:math:`\\pi_m` of Eq. 2 and the occupancy multiplier :math:`\\omega_m` of Eq. 3
are both fixed at one.  This experiment switches them on -- :math:`\\pi` with
standard deviation 0.12, :math:`\\omega` with 0.15, both centred on one -- draws
``--n-samples`` realisations per candidate evaluation, and averages
:math:`f_2` and :math:`f_3` before they reach the selection operator.

With a small profile subset the smallest attainable two-sided Wilcoxon
p-value at the profile level is :math:`2^{-(n-1)}`, so this check is reported
as descriptive rather than confirmatory; the script prints that bound
explicitly.

Usage
-----
    python -m experiments.monte_carlo_objectives --n-profiles 5 --n-samples 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from experiments._common import (
    add_common_arguments, build_context, run_variant, setup_logging, write_json,
)
from src.config import ScenarioConfig
from src.statistics import compare

logger = logging.getLogger(__name__)


def run(
    n_profiles: int = 5,
    n_seeds: int = 10,
    n_samples: int = 20,
    n_generations: int = 100,
    output_dir: str = "results/experiments/monte_carlo",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "mlp_surrogate",
    max_workers: int = 3,
) -> Dict[str, object]:
    import networkx as nx
    from src.comfort_models import TrainedComfortPredictor
    from src.pipeline_V6_smart import build_problem_factory
    from src.survey_data_loader import load_all

    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model)

    survey_data = load_all(Path(survey_dir))
    G = nx.read_graphml(graph_path)
    predictor = TrainedComfortPredictor(ctx.comfort_results, model_name=comfort_model)

    scenarios = {
        "deterministic": ScenarioConfig(),
        "monte_carlo": ScenarioConfig(
            dynamic_pricing=True, stochastic_crowding=True,
            n_monte_carlo=int(n_samples),
        ),
    }

    summaries: Dict[str, object] = {}
    tables = []
    for label, scenario in scenarios.items():
        logger.info("--- %s (n_monte_carlo=%d) ---", label, scenario.n_monte_carlo)
        problem_factory = build_problem_factory(G, survey_data.calibration, predictor, scenario)
        metrics = run_variant(
            ctx, out / label, algorithms=("nsga2", "pi_nsga3"), n_seeds=n_seeds,
            n_generations=n_generations, plan="sensitivity", max_workers=max_workers,
            scenario=scenario, problem_factory=problem_factory,
        )
        if metrics is None:
            continue
        metrics["setting"] = label
        tables.append(metrics)

        result = compare(metrics, algo_a="nsga2", algo_b="pi_nsga3")
        per_profile = metrics.groupby(["algorithm", "profile_id"])["normalized_hv"].mean().unstack(0)
        summaries[label] = {
            "mean_nhv_pi_nsga3": float(per_profile["pi_nsga3"].mean()),
            "std_nhv_pi_nsga3": float(per_profile["pi_nsga3"].std(ddof=1)),
            "mean_nhv_nsga2": float(per_profile["nsga2"].mean()),
            "std_nhv_nsga2": float(per_profile["nsga2"].std(ddof=1)),
            "dz_profile_level": result["dz_profile_level_confirmatory"],
            "wilcoxon_profile_p": result["wilcoxon_profile_p"],
            "n_profiles": result["n_profiles"],
            "ranking_preserved": bool(result["dz_profile_level_confirmatory"] < 0),
        }

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out / "monte_carlo_raw.csv", index=False)

    smallest_p = 2.0 ** (-(n_profiles - 1)) if n_profiles > 1 else 1.0
    payload = {
        "n_profiles": n_profiles,
        "n_seeds": n_seeds,
        "n_monte_carlo_samples": n_samples,
        "n_generations": n_generations,
        "smallest_attainable_profile_level_p": float(smallest_p),
        "inference_status": (
            "descriptive: with this few profiles the profile-level Wilcoxon "
            "test cannot reach alpha = 0.05"
            if smallest_p > 0.05 else "confirmatory"
        ),
        "settings": summaries,
    }
    write_json(out / "monte_carlo_summary.json", payload)

    print(pd.DataFrame(summaries).T.to_string())
    print(f"\nSmallest attainable two-sided p at the profile level: {smallest_p:.4f}")
    return payload


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-profiles", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--out", default="results/experiments/monte_carlo")
    args = parser.parse_args()

    setup_logging()
    run(n_profiles=args.n_profiles, n_seeds=args.n_seeds, n_samples=args.n_samples,
        n_generations=args.n_generations, output_dir=args.out,
        survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
