"""normalization_schemes.py
============================
Robustness of the ranking to the hypervolume normalization (Section 6.6).

The headline results use the algorithm-agnostic scheme of Eq. 12-13: both the
reference point and the normalization denominator are built from the union of
all algorithms and seeds of a profile.  Three alternatives are compared here:

``per_algorithm_max``
    each algorithm normalised by its own best run on that profile.  This is the
    scheme to avoid: conditioning the denominator on a single algorithm's
    output introduces a bias correlated with that algorithm's performance, and
    it mechanically inflates whichever algorithm leads on each metric.
``fixed_survey_nadir``
    a fixed reference point derived from the survey bounds, identical for every
    profile of a given respondent.
``ideal_bounded``
    a reference point placed 10 % beyond the observed nadir along the
    ideal-to-nadir segment of the profile.

The experiment does not re-run any search: it recomputes the indicator from the
population checkpoints already on disk, which is exactly the point -- the
comparison isolates the measurement instrument from the search.

Usage
-----
    python -m experiments.normalization_schemes --runs results/outputs_main/main_pi_nsga3_vs_nsga2_150profiles
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from experiments._common import setup_logging, write_json
from src.config import DEFAULT_NORMALIZATION
from src.pipeline_V6_smart import recover_hv_igd_for_plan
from src.statistics import compare

logger = logging.getLogger(__name__)

SCHEME_COLUMNS = {
    "union_observed_max_per_profile": "normalized_hv",
    "per_algorithm_max": "nhv_per_algorithm_max",
    "fixed_survey_nadir": "nhv_fixed_survey_nadir",
    "ideal_bounded": "nhv_ideal_bounded",
}


def run(
    run_dirs: Sequence[Path],
    algo_a: str = "nsga2",
    algo_b: str = "pi_nsga3",
    output_dir: str = "results/experiments/normalization",
    survey_nadir: Sequence[float] = (180.0, 20.0, 3.0, 1.05),
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics = recover_hv_igd_for_plan(
            run_dir, out, schemes=DEFAULT_NORMALIZATION.schemes,
            survey_nadir=list(survey_nadir),
        )
        if metrics is None:
            logger.warning("no metrics recovered from %s", run_dir)
            continue

        for scheme, column in SCHEME_COLUMNS.items():
            if column not in metrics.columns or metrics[column].isna().all():
                logger.info("scheme %s unavailable in %s; skipping", scheme, run_dir.name)
                continue
            result = compare(metrics, algo_a=algo_a, algo_b=algo_b, value_col=column)
            means = metrics.groupby("algorithm")[column].mean()
            rows.append({
                "plan": run_dir.name,
                "scheme": scheme,
                "is_algorithm_agnostic": scheme == DEFAULT_NORMALIZATION.default_scheme,
                f"mean_{algo_b}": float(means.get(algo_b, np.nan)),
                f"mean_{algo_a}": float(means.get(algo_a, np.nan)),
                "mean_diff": result["mean_diff"],
                "dz_profile_level": result["dz_profile_level_confirmatory"],
                "wilcoxon_profile_p": result["wilcoxon_profile_p"],
                "profile_wins_b": result["profile_wins_b"],
                "n_profiles": result["n_profiles"],
                "ranking_preserved": bool(result["dz_profile_level_confirmatory"] < 0),
            })

    table = pd.DataFrame(rows)
    table.to_csv(out / "normalization_schemes.csv", index=False)

    baseline = table[table["is_algorithm_agnostic"]]["dz_profile_level"]
    write_json(out / "normalization_summary.json", {
        "schemes_compared": list(SCHEME_COLUMNS),
        "ranking_preserved_everywhere": bool(table["ranking_preserved"].all()) if len(table) else False,
        "baseline_dz": float(baseline.mean()) if len(baseline) else float("nan"),
        "dz_by_scheme": table.groupby("scheme")["dz_profile_level"].mean().round(4).to_dict()
        if len(table) else {},
        "note": (
            "per_algorithm_max conditions the denominator on a single "
            "algorithm's output and is not recommended for unbiased comparison."
        ),
    })
    print(table.to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="plan directories containing checkpoints/population")
    parser.add_argument("--algo-a", default="nsga2")
    parser.add_argument("--algo-b", default="pi_nsga3")
    parser.add_argument("--out", default="results/experiments/normalization")
    parser.add_argument("--survey-nadir", nargs=4, type=float,
                        default=[180.0, 20.0, 3.0, 1.05])
    args = parser.parse_args()

    setup_logging()
    run([Path(r) for r in args.runs], algo_a=args.algo_a, algo_b=args.algo_b,
        output_dir=args.out, survey_nadir=args.survey_nadir)


if __name__ == "__main__":
    main()
