"""popsize_equalization.py
==========================
Population-size controls (Section 5.3, Table 7).

NSGA-II runs at :math:`N = 168` while PI-NSGA-III is raised to :math:`N = 170`,
because the implementation matches the population to the cardinality of the
augmented reference set so that every direction is associated with at least one
individual.  The asymmetry is two individuals per generation, 1.2 % of the
evaluation budget.  Two questions follow.

**Q1 -- equalization.**  Does the asymmetry explain the advantage?  Both
algorithms are re-run at a *common* population size on a stratified subset, and
the asymmetric configuration is reproduced on the same subset as a control.

**Q2 -- sweep.**  Is :math:`N = 170` in any way special for NSGA-II?  NSGA-II
is run across a grid of population sizes on the same subset.

The hypervolume protocol is the one of Eq. 12-13, unchanged.

Usage
-----
    python -m experiments.popsize_equalization --n-profiles 30 --n-seeds 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from experiments._common import (
    add_common_arguments, build_context, run_variant, setup_logging, write_json,
)
from src.config import POPULATION_SIZES
from src.reference_directions import audit_reference_directions
from src.statistics import compare

logger = logging.getLogger(__name__)

DEFAULT_SWEEP = (120, 140, 150, 160, 168, 170, 180, 200)


def _register_population(plan: str, algorithm: str, size: int) -> None:
    """Temporarily declare a population size for a sweep cell."""
    POPULATION_SIZES.setdefault(plan, {})[algorithm] = int(size)


def run(
    n_profiles: int = 30,
    n_seeds: int = 10,
    n_generations: int = 150,
    sweep_sizes: Sequence[int] = DEFAULT_SWEEP,
    sweep_profiles: int = 10,
    sweep_seeds: int = 5,
    output_dir: str = "results/experiments/popsize",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "mlp_surrogate",
    max_workers: int = 3,
) -> Dict[str, object]:
    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model)

    payload: Dict[str, object] = {}

    # ---- Q1: equalization at a common population size --------------------
    logger.info("=== Q1: equalized population size ===")
    equalized = run_variant(
        ctx, out / "equalized", algorithms=("nsga2", "pi_nsga3"), n_seeds=n_seeds,
        n_generations=n_generations, plan="equalization", n_partitions=8,
        max_workers=max_workers,
    )
    if equalized is not None:
        equalized.to_csv(out / "equalization_raw.csv", index=False)
        result = compare(equalized, algo_a="nsga2", algo_b="pi_nsga3")
        per_profile = equalized.groupby(["algorithm", "profile_id"])["normalized_hv"].mean().unstack(0)
        payload["equalized"] = {
            "population_size": POPULATION_SIZES["equalization"]["nsga2"],
            "mean_nhv_pi_nsga3": float(per_profile["pi_nsga3"].mean()),
            "std_nhv_pi_nsga3": float(per_profile["pi_nsga3"].std(ddof=1)),
            "mean_nhv_nsga2": float(per_profile["nsga2"].mean()),
            "std_nhv_nsga2": float(per_profile["nsga2"].std(ddof=1)),
            "mean_difference": result["mean_diff"],
            "dz_profile_level": result["dz_profile_level_confirmatory"],
            "wilcoxon_statistic": result["wilcoxon_profile_statistic"],
            "wilcoxon_p": result["wilcoxon_profile_p"],
            "profile_wins_pi_nsga3": result["profile_wins_a"],
            "n_profiles": result["n_profiles"],
            "sign_convention": "NSGA-II minus PI-NSGA-III; negative favours PI-NSGA-III",
        }

    # ---- control: asymmetric configuration on the same subset ------------
    logger.info("=== Q1 control: asymmetric main-plan configuration ===")
    control = run_variant(
        ctx, out / "asymmetric_control", algorithms=("nsga2", "pi_nsga3"),
        n_seeds=n_seeds, n_generations=n_generations, plan="main", n_partitions=8,
        max_workers=max_workers,
    )
    if control is not None:
        result = compare(control, algo_a="nsga2", algo_b="pi_nsga3")
        payload["asymmetric_control"] = {
            "population_nsga2": POPULATION_SIZES["main"]["nsga2"],
            "population_pi_nsga3": POPULATION_SIZES["main"]["pi_nsga3"],
            "mean_difference": result["mean_diff"],
            "dz_profile_level": result["dz_profile_level_confirmatory"],
        }
        if "equalized" in payload:
            payload["equalization_effect"] = float(
                payload["equalized"]["mean_difference"] - result["mean_diff"]
            )

    # ---- Q2: NSGA-II population sweep ------------------------------------
    logger.info("=== Q2: NSGA-II population sweep ===")
    sweep_ctx = build_context(survey_dir, graph_path, str(out / "sweep"),
                              n_profiles=sweep_profiles, comfort_model=comfort_model,
                              random_state=91)
    rows: List[Dict[str, object]] = []
    for size in sweep_sizes:
        plan = f"sweep_{size}"
        _register_population(plan, "nsga2", size)
        _register_population(plan, "pi_nsga3", 170)
        metrics = run_variant(
            sweep_ctx, out / "sweep" / f"N{size}", algorithms=("nsga2",),
            n_seeds=sweep_seeds, n_generations=n_generations, plan=plan,
            n_partitions=8, max_workers=max_workers,
        )
        if metrics is None:
            continue
        values = metrics[metrics["algorithm"] == "nsga2"]["normalized_hv"]
        rows.append({
            "algorithm": "nsga2", "pop_size": int(size), "config": "sweep",
            "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "count": int(len(values)),
        })
        logger.info("  N=%d -> mean nHV %.4f", size, values.mean())

    sweep_table = pd.DataFrame(rows)
    sweep_table.to_csv(out / "table7_popsize_sweep.csv", index=False)
    if len(sweep_table):
        best = sweep_table.loc[sweep_table["mean"].idxmax()]
        payload["sweep"] = {
            "sizes": [int(s) for s in sweep_sizes],
            "best_size": int(best["pop_size"]),
            "best_mean": float(best["mean"]),
            "rank_of_170": int(
                (sweep_table["mean"] > sweep_table.loc[
                    sweep_table["pop_size"] == 170, "mean"].iloc[0]).sum() + 1
            ) if (sweep_table["pop_size"] == 170).any() else None,
            "note": "no optimum for NSGA-II is expected at N = 170",
        }

    payload["reference_direction_audit"] = audit_reference_directions(4, 8, ctx.stabilized)
    write_json(out / "popsize_summary.json", payload)

    print(sweep_table.to_string(index=False))
    return payload


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-profiles", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=150)
    parser.add_argument("--sweep-sizes", nargs="+", type=int, default=list(DEFAULT_SWEEP))
    parser.add_argument("--sweep-profiles", type=int, default=10)
    parser.add_argument("--sweep-seeds", type=int, default=5)
    parser.add_argument("--out", default="results/experiments/popsize")
    args = parser.parse_args()

    setup_logging()
    run(n_profiles=args.n_profiles, n_seeds=args.n_seeds,
        n_generations=args.n_generations, sweep_sizes=args.sweep_sizes,
        sweep_profiles=args.sweep_profiles, sweep_seeds=args.sweep_seeds,
        output_dir=args.out, survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
