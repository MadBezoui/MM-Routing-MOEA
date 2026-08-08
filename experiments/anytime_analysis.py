"""anytime_analysis.py
======================
Anytime behaviour and computational cost (Section 7.3, Tables 16-17).

Three distinct costs are measured separately, because they answer different
questions and the manuscript keeps them apart:

*benchmark cost*
    the search **with** the per-generation instrumentation callback that
    recomputes hypervolume, IGD and spacing at every generation.  This is a
    property of the experimental protocol, not of a deployed system;
*optimization cost*
    the same configuration with the callback disabled -- what a platform would
    actually pay for one profile on one core;
*truncated cost*
    the same again with the generation budget cut to the point where the
    hypervolume has plateaued.

Table 17 quantifies the plateau: the hypervolume at generation ``g`` as a
percentage of its value at the terminal generation, on every profile.

All timings are single-process with no concurrent workers, so they are not
contaminated by thread contention.

Usage
-----
    python -m experiments.anytime_analysis --n-profiles 10 --n-repeats 5
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from experiments._common import (
    add_common_arguments, build_context, setup_logging, write_json,
)
from src.config import ScenarioConfig
from src.optimization_framework_parallel3 import run_single_algorithm

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS = (40, 60, 80, 90, 110, 130, 150)


# --------------------------------------------------------------------------
# (i)-(iii) wall-clock cost
# --------------------------------------------------------------------------

def measure_runtime(
    ctx,
    algorithms: Sequence[str] = ("pi_nsga3", "nsga2"),
    n_repeats: int = 5,
    n_generations: int = 150,
    truncated_generations: int = 40,
) -> pd.DataFrame:
    """Measure the three costs on every profile of the context."""
    records: List[Dict[str, object]] = []
    conditions = [
        ("instrumented", True, n_generations),
        ("bare", False, n_generations),
        ("bare_truncated", False, truncated_generations),
    ]

    for algorithm in algorithms:
        for condition, instrumented, generations in conditions:
            if condition == "bare_truncated" and algorithm != algorithms[0]:
                continue
            for _, profile in ctx.profiles.iterrows():
                problem = ctx.problem_factory(profile.to_dict(), ScenarioConfig())
                for repeat in range(n_repeats):
                    start = time.perf_counter()
                    run_single_algorithm(
                        problem=problem, algorithm_name=algorithm, seed=repeat,
                        n_generations=generations, plan="convergence", n_partitions=8,
                        reference_point=ctx.ref_point_factory(profile.to_dict()),
                        stabilized_weights=ctx.stabilized, raw_weights=ctx.raw,
                        encoding="path", instrumented=instrumented,
                    )
                    records.append({
                        "algorithm": algorithm,
                        "condition": condition,
                        "n_gen": generations,
                        "profile_id": profile["profile_id"],
                        "repeat": repeat,
                        "seconds": time.perf_counter() - start,
                    })
            logger.info("  %s / %s done", algorithm, condition)

    return pd.DataFrame(records)


def summarise_runtime(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["algorithm", "condition", "n_gen"])["seconds"]
        .agg(count="size", mean="mean", median="median", std="std",
             p95=lambda s: float(np.percentile(s, 95)))
        .reset_index()
    )


# --------------------------------------------------------------------------
# Table 17 - anytime hypervolume
# --------------------------------------------------------------------------

def anytime_table(
    history_dirs: Sequence[Path],
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
) -> pd.DataFrame:
    """Hypervolume at generation ``g`` as a percentage of the terminal value."""
    frames: List[pd.DataFrame] = []
    for root in history_dirs:
        for ckpt in Path(root).rglob("checkpoints/history"):
            for f in sorted(ckpt.glob("*.csv")):
                try:
                    df = pd.read_csv(f)
                except Exception:  # pragma: no cover
                    continue
                if {"generation", "hypervolume", "profile_id"}.issubset(df.columns):
                    frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no instrumented history under {list(history_dirs)}")

    history = pd.concat(frames, ignore_index=True).dropna(subset=["hypervolume"])
    terminal = history["generation"].max()

    per_run = history.pivot_table(
        index=["profile_id", "algorithm", "seed"],
        columns="generation", values="hypervolume",
    )
    final = per_run[terminal].replace(0, np.nan)

    available = [g for g in checkpoints if g in per_run.columns]
    if not available:
        # The run used a shorter budget than the default checkpoints: fall back
        # to evenly spaced generations of the budget that was actually used.
        available = sorted({
            int(q) for q in np.linspace(max(int(terminal * 0.25), 1), terminal, 6)
            if q in per_run.columns
        })
        logger.warning("default checkpoints absent (terminal generation = %s); "
                       "using %s instead", terminal, available)

    rows: List[Dict[str, object]] = []
    for g in available:
        pct = 100.0 * per_run[g] / final
        by_profile = pct.groupby(level=0).mean()
        rows.append({
            "generation": int(g),
            "budget_vs_terminal": round(float(g) / float(terminal), 2),
            "mean_hv_pct_of_final": float(pct.mean()),
            "worst_profile_pct": float(by_profile.min()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def run(
    n_profiles: int = 10,
    n_repeats: int = 5,
    n_generations: int = 150,
    truncated_generations: int = 40,
    output_dir: str = "results/experiments/anytime",
    survey_dir: str = "data/survey_results",
    graph_path: str = "data/processed/strasbourg_multimodal.graphml",
    comfort_model: str = "mlp_surrogate",
    history_dirs: Sequence[str] = (),
) -> Dict[str, object]:
    out = Path(output_dir)
    ctx = build_context(survey_dir, graph_path, str(out), n_profiles=n_profiles,
                        comfort_model=comfort_model, random_state=91)

    raw = measure_runtime(ctx, n_repeats=n_repeats, n_generations=n_generations,
                          truncated_generations=truncated_generations)
    raw.to_csv(out / "runtime_raw.csv", index=False)
    summary = summarise_runtime(raw)
    summary.to_csv(out / "table16_runtime_summary.csv", index=False)

    write_json(out / "environment.json", {
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "n_profiles": n_profiles,
        "n_repeats": n_repeats,
        "note": "single process, no concurrent worker threads",
    })

    payload: Dict[str, object] = {"runtime_summary": summary.to_dict(orient="records")}

    if history_dirs:
        try:
            table17 = anytime_table([Path(d) for d in history_dirs])
            table17.to_csv(out / "table17_anytime_hypervolume.csv", index=False)
            payload["anytime"] = table17.to_dict(orient="records")
            print(table17.to_string(index=False))
        except FileNotFoundError as exc:
            logger.warning("anytime table skipped: %s", exc)

    write_json(out / "anytime_summary.json", payload)
    print(summary.to_string(index=False))
    return payload


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-profiles", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-generations", type=int, default=150)
    parser.add_argument("--truncated-generations", type=int, default=40)
    parser.add_argument("--history-dirs", nargs="*", default=[],
                        help="run directories whose instrumented history feeds Table 17")
    parser.add_argument("--out", default="results/experiments/anytime")
    args = parser.parse_args()

    setup_logging()
    run(n_profiles=args.n_profiles, n_repeats=args.n_repeats,
        n_generations=args.n_generations,
        truncated_generations=args.truncated_generations,
        output_dir=args.out, survey_dir=args.survey_dir, graph_path=args.graph,
        comfort_model=args.comfort_model, history_dirs=args.history_dirs)


if __name__ == "__main__":
    main()
