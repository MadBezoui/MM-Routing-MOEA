"""decoupling.py
=================
Controlled decoupling of the inter-objective correlation structure (Section 6.4).

Section 6.4 attributes part of the advantage of PI-NSGA-III to the low
effective dimensionality of the sampled feasible objective archive, itself a
consequence of the correlations between time, cost, emissions and discomfort.

This experiment removes those correlations by construction.  On a problem of
identical decision-space size -- the same graph, the same profiles, the same
routes -- the four objective values of every candidate are replaced by
independent Gaussian draws, matched to the marginal mean and standard deviation
of the real objectives so that only the *dependence* structure changes.  The
draw is deterministic in the route, so the synthetic landscape is a genuine
function of the decision variables rather than pure noise.

Interpretation, stated in the manuscript and repeated here: the result is
suggestive but not conclusive.  Resampling from i.i.d. Gaussians removes the
correlations, but it also alters the objective semantics and the mapping from
decision space to objective space, so it does not isolate dimensionality as the
sole causal factor.

Usage
-----
    python -m experiments.decoupling --n-profiles 20 --n-seeds 10
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from experiments._common import (
    add_common_arguments, build_context, run_variant, setup_logging, write_json,
)
from src.config import ScenarioConfig
from src.optimization_framework_parallel3 import ProfiledMultimodalProblem
from src.statistics import compare

logger = logging.getLogger(__name__)


class DecoupledEvaluator:
    """Wrap an evaluator and replace its objectives by independent draws.

    The wrapped evaluator is still called, so the constraint values and the
    feasible region are exactly those of the real problem.  Only the objective
    vector is substituted, by a deterministic pseudo-random draw seeded on the
    route itself: identical routes always receive identical objective values,
    which is what makes the surrogate landscape a well-defined function.
    """

    def __init__(self, base_evaluator, moments: Tuple[np.ndarray, np.ndarray]):
        self.base = base_evaluator
        self.mean, self.std = moments

    @staticmethod
    def _seed_of(route) -> int:
        key = "|".join(route.nodes) + "#" + "|".join(route.modes)
        return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % (2 ** 32)

    def __call__(self, X, profile, extras, scenario):
        _, G, meta = self.base(X, profile, extras, scenario)
        n = X.shape[0]
        F = np.empty((n, len(self.mean)), dtype=float)
        for i in range(n):
            rng = np.random.default_rng(self._seed_of(X[i, 0]))
            F[i] = np.maximum(rng.normal(self.mean, self.std), 0.0)
        return F, G, meta


def _estimate_moments(ctx, n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Marginal mean and standard deviation of the four real objectives."""
    from src.network.operators import MultimodalIndex, PathSampling

    profile = ctx.profiles.iloc[0].to_dict()
    problem = ctx.problem_factory(profile, ScenarioConfig())
    sampler = PathSampling(ctx.graph, index=MultimodalIndex(ctx.graph))
    X = sampler._do(problem, n_samples)
    F, _, _ = problem._evaluator(X, profile, {}, ScenarioConfig())
    return F.mean(axis=0), F.std(axis=0) + 1e-9


def build_decoupled_factory(ctx, moments):
    def factory(profile: Dict[str, Any], scenario: ScenarioConfig) -> ProfiledMultimodalProblem:
        base_problem = ctx.problem_factory(profile, scenario)
        problem = ProfiledMultimodalProblem(
            n_var=1, n_obj=4, xl=None, xu=None,
            evaluator=DecoupledEvaluator(base_problem._evaluator, moments),
            profile=profile, extras={}, scenario=scenario, vtype=object,
        )
        problem.graph = base_problem.graph
        problem.graph_index = base_problem.graph_index
        return problem
    return factory


def run(
    n_profiles: int = 20,
    n_seeds: int = 10,
    n_generations: int = 100,
    output_dir: str = "results/experiments/decoupling",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "mlp_surrogate",
    max_workers: int = 3,
) -> Dict[str, object]:
    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model)

    moments = _estimate_moments(ctx)
    logger.info("Objective moments: mean=%s std=%s",
                np.round(moments[0], 3).tolist(), np.round(moments[1], 3).tolist())

    settings = {
        "correlated_real": None,
        "decoupled_gaussian": build_decoupled_factory(ctx, moments),
    }

    summaries: Dict[str, object] = {}
    correlations: Dict[str, object] = {}
    for label, factory in settings.items():
        logger.info("--- %s ---", label)
        metrics = run_variant(
            ctx, out / label, algorithms=("nsga2", "pi_nsga3"), n_seeds=n_seeds,
            n_generations=n_generations, plan="sensitivity",
            max_workers=max_workers, problem_factory=factory,
        )
        if metrics is None:
            continue
        result = compare(metrics, algo_a="nsga2", algo_b="pi_nsga3")
        summaries[label] = {
            "mean_diff": result["mean_diff"],
            "dz_run_level": result["dz_run_level_descriptive"],
            "dz_profile_level": result["dz_profile_level_confirmatory"],
            "wilcoxon_profile_p": result["wilcoxon_profile_p"],
        }

        populations = []
        for f in sorted((out / label / "checkpoints" / "population").glob("*.csv")):
            populations.append(pd.read_csv(f, usecols=["obj_1", "obj_2", "obj_3", "obj_4"]))
        if populations:
            pooled = pd.concat(populations, ignore_index=True)
            correlations[label] = pooled.corr().round(4).to_dict()

    real_dz = summaries.get("correlated_real", {}).get("dz_run_level", float("nan"))
    decoupled_dz = summaries.get("decoupled_gaussian", {}).get("dz_run_level", float("nan"))

    payload = {
        "n_profiles": n_profiles,
        "n_seeds": n_seeds,
        "n_generations": n_generations,
        "objective_moments": {"mean": moments[0].tolist(), "std": moments[1].tolist()},
        "settings": summaries,
        "objective_correlations": correlations,
        "dz_shrinkage": float(abs(real_dz) - abs(decoupled_dz)),
        "caveat": (
            "Suggestive, not conclusive: resampling objectives from i.i.d. "
            "Gaussians removes inter-objective correlation but also changes "
            "the problem geometry, the objective semantics and the "
            "decision-to-objective mapping."
        ),
    }
    write_json(out / "decoupling_summary.json", payload)
    print(pd.DataFrame(summaries).T.to_string())
    return payload


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-profiles", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--out", default="results/experiments/decoupling")
    args = parser.parse_args()

    setup_logging()
    run(n_profiles=args.n_profiles, n_seeds=args.n_seeds,
        n_generations=args.n_generations, output_dir=args.out,
        survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
