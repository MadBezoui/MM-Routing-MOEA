"""surrogate_perturbations.py
=============================
Robustness of the algorithmic ranking to the comfort surrogate (Section 6.6).

Three perturbations are applied, each answering a different question.

``--study ablation``
    Replace the multilayer perceptron by the linear regression and by the
    heuristic baseline, and re-run the comparison.  Question: is the advantage
    of the proposed variant an artifact of the surrogate's non-linearities?

``--study noise``
    Re-run the comparison with a surrogate trained on inputs corrupted by
    additive Gaussian noise of standard deviation ``sigma``, applied to the
    twelve input features.  Question: how does the ranking degrade as the
    surrogate's predictive accuracy degrades?

``--study bias``
    Add a constant bias ``delta`` to every comfort prediction before Eq. 4.
    A positive bias overestimates comfort uniformly and favours high-comfort
    routes; a negative one penalises all routes uniformly.  Question: is the
    advantage an artifact of the surrogate's calibration level?

Usage
-----
    python -m experiments.surrogate_perturbations --study ablation
    python -m experiments.surrogate_perturbations --study noise --sigmas 0 0.05 0.10
    python -m experiments.surrogate_perturbations --study bias --deltas -0.10 0.10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from experiments._common import (
    add_common_arguments, build_context, run_variant, setup_logging, write_json,
)
from src.comfort_models import (
    SurveyInformedComfortFactory, TrainedComfortPredictor,
)
from src.config import DEFAULT_COMFORT_CONFIG, ScenarioConfig
from src.pipeline_V6_smart import build_problem_factory
from src.statistics import compare
from src.survey_data_loader import load_all

logger = logging.getLogger(__name__)

SURROGATES = ("heuristic_direct", "linear_regression", "mlp_surrogate")


def _pair_stats(metrics: pd.DataFrame) -> Dict[str, float]:
    result = compare(metrics, algo_a="nsga2", algo_b="pi_nsga3")
    means = metrics.groupby("algorithm")["normalized_hv"].mean()
    return {
        "mean_nhv_pi_nsga3": float(means.get("pi_nsga3", np.nan)),
        "mean_nhv_nsga2": float(means.get("nsga2", np.nan)),
        "dz_profile_level": result["dz_profile_level_confirmatory"],
        "dz_run_level_descriptive": result["dz_run_level_descriptive"],
        "wilcoxon_profile_p": result["wilcoxon_profile_p"],
        "ranking_preserved": bool(result["dz_profile_level_confirmatory"] < 0),
    }


# --------------------------------------------------------------------------
# (a) surrogate ablation
# --------------------------------------------------------------------------

def study_ablation(ctx, out: Path, n_seeds: int, n_generations: int,
                   max_workers: int, survey_dir: str, graph_path: str) -> pd.DataFrame:
    import networkx as nx

    survey_data = load_all(Path(survey_dir))
    G = nx.read_graphml(graph_path)

    rows: List[Dict[str, object]] = []
    for name in SURROGATES:
        logger.info("--- comfort surrogate: %s ---", name)
        predictor = TrainedComfortPredictor(ctx.comfort_results, model_name=name)
        factory = build_problem_factory(G, survey_data.calibration, predictor, ScenarioConfig())
        metrics = run_variant(
            ctx, out / f"surrogate_{name}", algorithms=("nsga2", "pi_nsga3"),
            n_seeds=n_seeds, n_generations=n_generations, plan="sensitivity",
            max_workers=max_workers, problem_factory=factory,
        )
        if metrics is None:
            continue
        model_metrics = next(r.metrics for r in ctx.comfort_results if r.model_name == name)
        rows.append({"surrogate": name, "r2": model_metrics["r2"],
                     "rmse": model_metrics["rmse"], **_pair_stats(metrics)})

    table = pd.DataFrame(rows)
    table.to_csv(out / "surrogate_ablation.csv", index=False)
    return table


# --------------------------------------------------------------------------
# (b) input-feature noise
# --------------------------------------------------------------------------

def study_noise(ctx, out: Path, sigmas: Sequence[float], n_seeds: int,
                n_generations: int, max_workers: int, survey_dir: str,
                graph_path: str) -> pd.DataFrame:
    import networkx as nx

    survey_data = load_all(Path(survey_dir))
    G = nx.read_graphml(graph_path)
    training = survey_data.comfort_training
    factory_builder = SurveyInformedComfortFactory(DEFAULT_COMFORT_CONFIG, survey_data.calibration)

    rows: List[Dict[str, object]] = []
    for sigma in sigmas:
        logger.info("--- input noise sigma = %.3f ---", sigma)

        noisy = training.copy()
        if sigma > 0:
            rng = np.random.default_rng(DEFAULT_COMFORT_CONFIG.random_state + int(sigma * 1000))
            cols = list(DEFAULT_COMFORT_CONFIG.feature_columns)
            # Scale the perturbation by each feature's own spread, so that
            # sigma has the same meaning for a mode share and for a temperature.
            spread = noisy[cols].std(ddof=0).replace(0, 1.0).to_numpy()
            noise = rng.normal(0.0, sigma, size=(len(noisy), len(cols))) * spread
            noisy[cols] = noisy[cols].to_numpy() + noise

        results = factory_builder.train_models(noisy)
        predictor = TrainedComfortPredictor(results, model_name="mlp_surrogate")
        problem_factory = build_problem_factory(G, survey_data.calibration, predictor, ScenarioConfig())

        metrics = run_variant(
            ctx, out / f"noise_{sigma:.3f}", algorithms=("nsga2", "pi_nsga3"),
            n_seeds=n_seeds, n_generations=n_generations, plan="sensitivity",
            max_workers=max_workers, problem_factory=problem_factory,
        )
        if metrics is None:
            continue
        mlp_metrics = next(r.metrics for r in results if r.model_name == "mlp_surrogate")
        rows.append({"sigma": float(sigma), "surrogate_r2": mlp_metrics["r2"],
                     "surrogate_rmse": mlp_metrics["rmse"], **_pair_stats(metrics)})

    table = pd.DataFrame(rows)
    table.to_csv(out / "surrogate_noise.csv", index=False)
    return table


# --------------------------------------------------------------------------
# (c) bias injection
# --------------------------------------------------------------------------

def study_bias(ctx, out: Path, deltas: Sequence[float], n_seeds: int,
               n_generations: int, max_workers: int, survey_dir: str,
               graph_path: str) -> pd.DataFrame:
    import networkx as nx

    survey_data = load_all(Path(survey_dir))
    G = nx.read_graphml(graph_path)
    predictor = TrainedComfortPredictor(ctx.comfort_results, model_name="mlp_surrogate")

    rows: List[Dict[str, object]] = []
    for delta in deltas:
        logger.info("--- comfort bias delta = %+0.2f ---", delta)
        scenario = ScenarioConfig(comfort_bias=float(delta))
        problem_factory = build_problem_factory(G, survey_data.calibration, predictor, scenario)
        metrics = run_variant(
            ctx, out / f"bias_{delta:+0.2f}", algorithms=("nsga2", "pi_nsga3"),
            n_seeds=n_seeds, n_generations=n_generations, plan="sensitivity",
            max_workers=max_workers, scenario=scenario, problem_factory=problem_factory,
        )
        if metrics is None:
            continue
        rows.append({"delta": float(delta), **_pair_stats(metrics)})

    table = pd.DataFrame(rows)
    table.to_csv(out / "surrogate_bias.csv", index=False)
    return table


# --------------------------------------------------------------------------

def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--study", required=True, choices=["ablation", "noise", "bias"])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--deltas", nargs="+", type=float, default=[-0.10, 0.10])
    parser.add_argument("--n-profiles", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--out", default="results/experiments/surrogate")
    args = parser.parse_args()

    setup_logging()
    out = Path(args.out) / args.study
    ctx = build_context(args.survey_dir, args.graph, str(out),
                        n_profiles=args.n_profiles, comfort_model="mlp_surrogate")

    if args.study == "ablation":
        table = study_ablation(ctx, out, args.n_seeds, args.n_generations,
                               args.max_workers, args.survey_dir, args.graph)
    elif args.study == "noise":
        table = study_noise(ctx, out, args.sigmas, args.n_seeds, args.n_generations,
                            args.max_workers, args.survey_dir, args.graph)
    else:
        table = study_bias(ctx, out, args.deltas, args.n_seeds, args.n_generations,
                           args.max_workers, args.survey_dir, args.graph)

    write_json(out / f"{args.study}_summary.json", {
        "study": args.study,
        "n_profiles": args.n_profiles,
        "n_seeds": args.n_seeds,
        "n_generations": args.n_generations,
        "ranking_preserved_everywhere": bool(table["ranking_preserved"].all()) if len(table) else False,
        "dz_range": [float(table["dz_profile_level"].min()),
                     float(table["dz_profile_level"].max())] if len(table) else [],
    })
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
