"""beta_phi_grid.py
====================
Sensitivity to the stabilization parameters (beta, phi) (Section 6.6, Fig. 18).

Eq. 6 has two free parameters: the uniform-blend intensity :math:`\\beta` and
the per-component floor :math:`\\varphi`.  This experiment sweeps the
calibration grid, re-anchors the PI-NSGA-III reference set on the resulting
:math:`\\mathbf{w}_{stab}`, and reports the paired effect size against NSGA-II
in every cell.

The cell :math:`(\\beta, \\varphi) = (0, 0)` is the *degenerate limit*: the
floor is inactive and the raw elicited weight vector -- including its near-zero
emissions component -- is used directly for anchoring.  It is the most
informative cell, because it tests whether the anchoring mechanism survives
with stabilization fully disabled.

The grid also records, per cell, on how many objectives the floor actually
binds; a cell where it binds on more than the single near-degenerate objective
lies outside the admissible set of Eq. 7.

Usage
-----
    python -m experiments.beta_phi_grid --n-profiles 8 --n-seeds 5
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
from src.config import DEFAULT_STABILIZATION
from src.preferences.stabilization import (
    OBJECTIVES, admissible_pairs, blended_weights, stabilize,
)
from src.statistics import compare

logger = logging.getLogger(__name__)


def run(
    beta_grid: Sequence[float] = DEFAULT_STABILIZATION.beta_grid,
    phi_grid: Sequence[float] = DEFAULT_STABILIZATION.phi_grid,
    n_profiles: int = 8,
    n_seeds: int = 5,
    n_generations: int = 100,
    output_dir: str = "results/experiments/beta_phi_grid",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "linear_regression",
    max_workers: int = 3,
) -> pd.DataFrame:
    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model)

    raw_vector = ctx.raw
    admissibility = admissible_pairs(ctx.audit.raw_weights)
    admissibility.to_csv(out / "admissible_grid.csv", index=False)

    rows: List[Dict[str, object]] = []
    for beta in beta_grid:
        blended = blended_weights(raw_vector, float(beta))
        for phi in phi_grid:
            label = f"b{beta:.2f}_p{phi:.2f}"
            weights = stabilize(raw_vector, float(beta), float(phi))
            logger.info("--- beta=%.2f phi=%.2f -> w=%s ---",
                        beta, phi, np.round(weights, 4).tolist())

            metrics = run_variant(
                ctx, out / label,
                algorithms=("nsga2", "pi_nsga3"), n_seeds=n_seeds,
                n_generations=n_generations, plan="sensitivity", n_partitions=8,
                stabilized_weights=list(weights), max_workers=max_workers,
            )
            if metrics is None:
                logger.warning("cell %s produced no metrics; skipping", label)
                continue

            result = compare(metrics, algo_a="nsga2", algo_b="pi_nsga3")
            cell = admissibility[
                np.isclose(admissibility["beta"], beta) & np.isclose(admissibility["phi"], phi)
            ]
            rows.append({
                "beta": float(beta),
                "phi": float(phi),
                "dz": result["dz_profile_level_confirmatory"],
                "dz_run_level": result["dz_run_level_descriptive"],
                "wilcoxon_profile_p": result["wilcoxon_profile_p"],
                "floor_binds_on_n_objectives": int((blended < phi).sum()),
                "admissible": bool(cell["admissible"].iloc[0]) if len(cell) else False,
                "degenerate_limit": bool(beta == 0.0 and phi == 0.0),
                **{f"w_{name}": float(w) for name, w in zip(OBJECTIVES, weights)},
            })

    table = pd.DataFrame(rows)
    table.to_csv(out / "fig18_beta_phi_grid.csv", index=False)

    pivot = table.pivot(index="phi", columns="beta", values="dz")
    pivot.to_csv(out / "fig18_dz_matrix.csv")

    dz = table["dz"].to_numpy()
    degenerate = table[table["degenerate_limit"]]
    write_json(out / "beta_phi_summary.json", {
        "beta_grid": [float(b) for b in beta_grid],
        "phi_grid": [float(p) for p in phi_grid],
        "n_cells": int(len(table)),
        "n_profiles": int(n_profiles),
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "dz_range": [float(dz.min()), float(dz.max())] if len(dz) else [],
        "dz_amplitude": float(dz.max() - dz.min()) if len(dz) else float("nan"),
        "sign_consistent": bool(np.all(dz < 0) or np.all(dz > 0)) if len(dz) else False,
        "degenerate_limit_dz": float(degenerate["dz"].iloc[0]) if len(degenerate) else None,
        "calibrated_cell": {"beta": DEFAULT_STABILIZATION.blend_uniform,
                            "phi": DEFAULT_STABILIZATION.floor},
    })

    print(table.to_string(index=False))
    print()
    print("d_z matrix (rows = phi, columns = beta):")
    print(pivot.round(3).to_string())
    return table


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--beta-grid", nargs="+", type=float,
                        default=list(DEFAULT_STABILIZATION.beta_grid))
    parser.add_argument("--phi-grid", nargs="+", type=float,
                        default=list(DEFAULT_STABILIZATION.phi_grid))
    parser.add_argument("--n-profiles", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--out", default="results/experiments/beta_phi_grid")
    args = parser.parse_args()

    setup_logging()
    run(beta_grid=args.beta_grid, phi_grid=args.phi_grid, n_profiles=args.n_profiles,
        n_seeds=args.n_seeds, n_generations=args.n_generations, output_dir=args.out,
        survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
