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
    fractions: Sequence[float] = (0.25, 0.50, 0.75, 1.0),
) -> pd.DataFrame:
    """Hypervolume at budget fractions relative to each run's terminal generation."""
    frames: List[pd.DataFrame] = []
    for root in history_dirs:
        for ckpt in Path(root).rglob("checkpoints/history"):
            for f in sorted(ckpt.glob("*.csv")):
                try:
                    df = pd.read_csv(f)
                except Exception:  # pragma: no cover
                    continue
                if {"generation", "hypervolume", "profile_id"}.issubset(df.columns):
                    # Dedup duplicated generations if any
                    df = df.drop_duplicates(subset=["generation"], keep="last")
                    df["plan"] = root.name
                    frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no instrumented history under {list(history_dirs)}")

    history = pd.concat(frames, ignore_index=True).dropna(subset=["hypervolume"])
    
    RUN_KEY = ["plan", "profile_id", "algorithm", "seed"]
    
    # Terminal generation per run
    history["terminal_generation"] = history.groupby(RUN_KEY)["generation"].transform("max")
    history["budget_fraction"] = history["generation"] / history["terminal_generation"]
    
    rows: List[Dict[str, object]] = []
    
    # For each run, interpolate the hypervolume at the requested fractions
    for run_id, group in history.groupby(RUN_KEY):
        # group is sorted by generation
        g = group.sort_values("generation")
        x = g["budget_fraction"].to_numpy()
        y = g["hypervolume"].to_numpy()
        
        terminal_hv = y[-1]
        if terminal_hv <= 0:
            continue
            
        # Interpolate
        y_interp = np.interp(fractions, x, y)
        
        for frac, val in zip(fractions, y_interp):
            pct = 100.0 * val / terminal_hv
            rows.append({
                "plan": run_id[0],
                "profile_id": run_id[1],
                "algorithm": run_id[2],
                "budget_fraction": frac,
                "pct_of_terminal": pct,
            })
            
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    # Average across seeds and profiles
    summary = (
        df.groupby(["algorithm", "budget_fraction"])["pct_of_terminal"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return summary


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
