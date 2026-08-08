"""four_way_ablation.py
=======================
Controlled four-way ablation on the real pipeline (Section 6.6, Tables 14-15).

    NSGA-II  x  canonical NSGA-III  x  PI-NSGA-III (raw)  x  PI-NSGA-III (stab)

The chain separates three additive contributions (Eq. 15):

.. math::
    \\underbrace{\\Delta_{total}}_{\\text{PI-stab} - \\text{NSGA-II}}
    = \\underbrace{\\Delta_1}_{\\text{reference directions}}
    + \\underbrace{\\Delta_2}_{\\text{priority anchoring}}
    + \\underbrace{\\Delta_3}_{\\text{weight stabilization}}

* :math:`\\Delta_1` = canonical NSGA-III minus NSGA-II -- the generic gain of
  moving from crowding-distance selection to a reference-direction method;
* :math:`\\Delta_2` = PI-NSGA-III(raw) minus canonical -- the gain of anchoring
  the reference set on the elicited priorities, *before* stabilization;
* :math:`\\Delta_3` = PI-NSGA-III(stab) minus PI-NSGA-III(raw) -- the gain of
  stabilizing those priorities.

All confirmatory tests use the per-profile mean difference as the unit of
inference; the seed-level observations only stabilize each profile mean.

Usage
-----
    python -m experiments.four_way_ablation --n-profiles 30 --n-seeds 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments._common import (
    add_common_arguments, build_context, run_variant, setup_logging, write_json,
)
from src.statistics import compare, holm_correction

logger = logging.getLogger(__name__)

#: Ordered ablation chain.  Each row gains exactly one mechanism over the previous.
CHAIN = ("nsga2", "canonical_nsga3", "pi_nsga3_raw", "pi_nsga3_stab")

LABELS = {
    "nsga2": "NSGA-II",
    "canonical_nsga3": "Canonical NSGA-III",
    "pi_nsga3_raw": "PI-NSGA-III (raw)",
    "pi_nsga3_stab": "PI-NSGA-III (stab)",
}


def run(
    n_profiles: int = 30,
    n_seeds: int = 10,
    n_generations: int = 150,
    output_dir: str = "results/experiments/four_way_ablation",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "mlp_surrogate",
    max_workers: int = 3,
) -> Dict[str, object]:
    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model)

    metrics = run_variant(
        ctx, out / "runs", algorithms=CHAIN, n_seeds=n_seeds,
        n_generations=n_generations, plan="ablation", n_partitions=8,
        max_workers=max_workers,
    )
    if metrics is None:
        raise RuntimeError("ablation produced no metrics")
    metrics.to_csv(out / "ablation_raw.csv", index=False)

    # ---- Table 14: ordered chain with incremental gains -------------------
    per_profile = (
        metrics.groupby(["algorithm", "profile_id"])["normalized_hv"].mean().unstack(0)
    )
    rows: List[Dict[str, object]] = []
    total_gain = float(per_profile[CHAIN[-1]].mean() - per_profile[CHAIN[0]].mean())

    for i, algorithm in enumerate(CHAIN):
        row: Dict[str, object] = {
            "algorithm": LABELS[algorithm],
            "key": algorithm,
            "mean_nhv": float(per_profile[algorithm].mean()),
            "std_nhv": float(per_profile[algorithm].std(ddof=1)),
        }
        if i == 0:
            row.update({"delta": np.nan, "share_pct": np.nan,
                        "profile_wins": "n/a", "p_vs_previous": np.nan})
        else:
            previous = CHAIN[i - 1]
            stats_row = compare(metrics, algo_a=previous, algo_b=algorithm)
            delta = float(per_profile[algorithm].mean() - per_profile[previous].mean())
            row.update({
                "delta": delta,
                "share_pct": 100.0 * delta / total_gain if total_gain else np.nan,
                "profile_wins": f"{stats_row['profile_wins_b']}/{stats_row['n_profiles']}",
                "p_vs_previous": stats_row["wilcoxon_profile_p"],
                "dz_vs_previous": stats_row["dz_profile_level_confirmatory"],
            })
        rows.append(row)

    table14 = pd.DataFrame(rows)
    table14.to_csv(out / "table14_ablation_chain.csv", index=False)

    # ---- Table 15: all six pairwise comparisons ---------------------------
    pairwise: List[Dict[str, object]] = []
    for i in range(len(CHAIN)):
        for j in range(i + 1, len(CHAIN)):
            a, b = CHAIN[j], CHAIN[i]          # sign convention: d_z < 0 favours a
            result = compare(metrics, algo_a=a, algo_b=b)
            pairwise.append({
                "algorithm_a": LABELS[a], "algorithm_b": LABELS[b],
                "dz": result["dz_profile_level_confirmatory"],
                "p": result["wilcoxon_profile_p"],
                "wins_a": f"{result['profile_wins_a']}/{result['n_profiles']}",
                "mean_diff": result["mean_diff"],
            })
    table15 = pd.DataFrame(pairwise)
    table15["p_holm"] = holm_correction(table15["p"].to_numpy())["p_holm"]
    table15.to_csv(out / "table15_pairwise.csv", index=False)

    summary = {
        "n_profiles": int(per_profile.shape[0]),
        "n_seeds": int(n_seeds),
        "n_generations": int(n_generations),
        "comfort_model": comfort_model,
        "total_gain": total_gain,
        "decomposition": {
            "delta_1_reference_directions": float(
                per_profile["canonical_nsga3"].mean() - per_profile["nsga2"].mean()),
            "delta_2_priority_anchoring": float(
                per_profile["pi_nsga3_raw"].mean() - per_profile["canonical_nsga3"].mean()),
            "delta_3_weight_stabilization": float(
                per_profile["pi_nsga3_stab"].mean() - per_profile["pi_nsga3_raw"].mean()),
        },
    }
    write_json(out / "ablation_summary.json", summary)

    print(table14.to_string(index=False))
    print()
    print(table15.to_string(index=False))
    return summary


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-profiles", type=int, default=30)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-generations", type=int, default=150)
    parser.add_argument("--out", default="results/experiments/four_way_ablation")
    args = parser.parse_args()

    setup_logging()
    run(n_profiles=args.n_profiles, n_seeds=args.n_seeds,
        n_generations=args.n_generations, output_dir=args.out,
        survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
