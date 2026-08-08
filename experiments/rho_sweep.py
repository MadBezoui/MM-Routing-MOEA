"""rho_sweep.py
================
Sensitivity to the anchor-spread parameter rho (Section 6.6, Table 13).

The parameter :math:`\\rho` of Eq. 9 controls how far the :math:`M` directional
anchors depart from :math:`\\mathbf{w}_{stab}` towards each objective axis.  A
small :math:`\\rho` clusters the anchors tightly around the elicited priority
vector; a large one pushes them half-way to the axes.

This experiment re-runs PI-NSGA-III and NSGA-II on a reduced stratified
test-bed for each value of :math:`\\rho` and reports the paired effect size.
The question it answers is the **flatness** of the response, not the absolute
level: the reduced budget is not the headline configuration, so the ranking it
produces should not be read as a ranking.

Sign convention, as in the main plan:
:math:`\\Delta = \\widehat{HV}_{NSGA-II} - \\widehat{HV}_{PI-NSGA-III}`, so
:math:`d_z < 0` favours the proposed variant.

Usage
-----
    python -m experiments.rho_sweep --n-profiles 20 --n-seeds 10
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
from src.config import DEFAULT_REFDIRS
from src.reference_directions import audit_reference_directions
from src.statistics import compare

logger = logging.getLogger(__name__)


def run(
    rho_grid: Sequence[float] = DEFAULT_REFDIRS.rho_grid,
    n_profiles: int = 20,
    n_seeds: int = 10,
    n_generations: int = 100,
    output_dir: str = "results/experiments/rho_sweep",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "linear_regression",
    max_workers: int = 3,
) -> pd.DataFrame:
    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model)

    rows: List[Dict[str, object]] = []
    frames: List[pd.DataFrame] = []

    for rho in rho_grid:
        logger.info("--- rho = %.2f ---", rho)
        metrics = run_variant(
            ctx, out / f"rho_{rho:.2f}",
            algorithms=("nsga2", "pi_nsga3"), n_seeds=n_seeds,
            n_generations=n_generations, plan="sensitivity", n_partitions=8,
            rho=float(rho), max_workers=max_workers,
        )
        if metrics is None:
            logger.warning("rho=%.2f produced no metrics; skipping", rho)
            continue
        metrics["rho"] = float(rho)
        frames.append(metrics)

        result = compare(metrics, algo_a="nsga2", algo_b="pi_nsga3")
        audit = audit_reference_directions(4, 8, ctx.stabilized, rho=float(rho))
        pivot = metrics.groupby("algorithm")["normalized_hv"].agg(["mean", "std"])

        rows.append({
            "rho": float(rho),
            "n_ref_dirs": audit["n_ref_dirs"],
            "mean_nhv_pi_nsga3": float(pivot.loc["pi_nsga3", "mean"]),
            "std_nhv_pi_nsga3": float(pivot.loc["pi_nsga3", "std"]),
            "mean_nhv_nsga2": float(pivot.loc["nsga2", "mean"]),
            "std_nhv_nsga2": float(pivot.loc["nsga2", "std"]),
            "dz_profile_level": result["dz_profile_level_confirmatory"],
            "dz_run_level_descriptive": result["dz_run_level_descriptive"],
            "wilcoxon_profile_p": result["wilcoxon_profile_p"],
            "n_paired_runs": result["n_paired_runs"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(out / "table13_rho_sensitivity.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "rho_sweep_raw.csv", index=False)

    dz = table["dz_profile_level"].to_numpy()
    write_json(out / "rho_sweep_summary.json", {
        "rho_grid": [float(r) for r in rho_grid],
        "n_profiles": int(n_profiles),
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "comfort_model": comfort_model,
        "dz_range": [float(np.min(dz)), float(np.max(dz))] if len(dz) else [],
        "dz_amplitude": float(np.max(dz) - np.min(dz)) if len(dz) else float("nan"),
        "sign_consistent": bool(np.all(dz < 0) or np.all(dz > 0)) if len(dz) else False,
        "note": (
            "The quantity of interest is the amplitude of d_z across the grid, "
            "not its absolute level: this reduced test-bed is not the headline "
            "configuration."
        ),
    })

    print(table.to_string(index=False))
    return table


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--rho-grid", nargs="+", type=float,
                        default=list(DEFAULT_REFDIRS.rho_grid))
    parser.add_argument("--n-profiles", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--out", default="results/experiments/rho_sweep")
    args = parser.parse_args()

    setup_logging()
    run(rho_grid=args.rho_grid, n_profiles=args.n_profiles, n_seeds=args.n_seeds,
        n_generations=args.n_generations, output_dir=args.out,
        survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
