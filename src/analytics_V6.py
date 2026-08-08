"""analytics_V6.py
===================
Post-hoc analysis of the experimental plans (Section 6).

Reads the recovered hypervolume files and the population/history checkpoints,
and produces every table and figure of the results section:

======================================  ====================================
``table8_distribution.csv``             Table 8, with profile-stratified
                                        bootstrap confidence intervals
``table9_per_archetype.csv``            Table 9
``table10_extended_benchmark.csv``      Table 10
``table11_friedman_nemenyi.csv``        Table 11 and the CD diagram
``table12_objective_correlations.csv``  Table 12 (Pearson and Spearman)
``per_profile_paired_tests.csv``        per-profile d_z with Holm correction
======================================  ====================================

Usage
-----
    python -m src.analytics_V6 --runs results/outputs_main --out results/analytics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_BENCHMARK, canonical_algorithm
from src.statistics import (
    compare, friedman_nemenyi, per_profile_table, stratified_bootstrap_ci,
)

logger = logging.getLogger(__name__)

OBJECTIVE_COLUMNS = ["obj_1", "obj_2", "obj_3", "obj_4"]
OBJECTIVE_LABELS = ["Time", "Cost", "Emissions", "SC Discomfort"]

LABELS = {
    "nsga2": "NSGA-II", "pi_nsga3": "PI-NSGA-III",
    "pi_nsga3_raw": "PI-NSGA-III (raw)", "pi_nsga3_stab": "PI-NSGA-III (stab)",
    "canonical_nsga3": "Canonical NSGA-III",
    "moead": "MOEA/D", "smsemoa": "SMS-EMOA",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_metrics(run_dirs: Sequence[Path]) -> Dict[str, pd.DataFrame]:
    """Load every ``*_final_generation_recovered.csv`` found under ``run_dirs``."""
    plans: Dict[str, pd.DataFrame] = {}
    for root in run_dirs:
        for f in sorted(Path(root).rglob("*_final_generation_recovered.csv")):
            name = f.name.replace("_final_generation_recovered.csv", "")
            if name in plans:
                continue
            df = pd.read_csv(f)
            df["algorithm"] = df["algorithm"].map(canonical_algorithm)
            plans[name] = df
    if not plans:
        raise FileNotFoundError(f"no recovered metric files under {list(run_dirs)}")
    logger.info("Loaded %d plan(s): %s", len(plans), ", ".join(plans))
    return plans


def load_profiles_metadata(run_dirs: Sequence[Path]) -> Optional[pd.DataFrame]:
    for root in run_dirs:
        f = Path(root) / "profiles_all_plans.csv"
        if f.exists():
            return pd.read_csv(f)
    return None


def load_populations(run_dirs: Sequence[Path], max_files: int = 3000) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    budget = max_files
    for root in run_dirs:
        for ckpt in Path(root).rglob("checkpoints/population"):
            for f in sorted(ckpt.glob("*.csv")):
                if budget <= 0:
                    break
                try:
                    frames.append(pd.read_csv(f))
                    budget -= 1
                except Exception:  # pragma: no cover
                    continue
    return pd.concat(frames, ignore_index=True) if frames else None


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def table_distribution(metrics: pd.DataFrame, value_col: str = "normalized_hv") -> pd.DataFrame:
    """Table 8: distribution of normalized hypervolume with a bootstrap CI."""
    rows: List[Dict[str, object]] = []
    for algorithm, group in metrics.groupby("algorithm"):
        ci = stratified_bootstrap_ci(
            group[value_col], group["profile_id"],
            n_resamples=DEFAULT_BENCHMARK.bootstrap_resamples,
        )
        values = group[value_col]
        rows.append({
            "algorithm": LABELS.get(algorithm, algorithm),
            "key": algorithm,
            "n_runs": int(len(values)),
            "mean": float(values.mean()),
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "median": float(values.median()),
            "std": float(values.std(ddof=1)),
            "q05": float(values.quantile(0.05)),
            "q95": float(values.quantile(0.95)),
            "min": float(values.min()),
        })
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def table_per_stratum(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    stratum: str = "archetype",
) -> pd.DataFrame:
    """Table 9: mean normalized hypervolume per algorithm and stratum."""
    merged = metrics.merge(
        profiles[["profile_id", stratum]].drop_duplicates("profile_id"),
        on="profile_id", how="left",
    )
    pivot = merged.pivot_table(index=stratum, columns="algorithm",
                               values="normalized_hv", aggfunc="mean")
    pivot.columns = [LABELS.get(c, c) for c in pivot.columns]
    return pivot.reset_index()


def table_extended(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Table 10 and Table 11: four-algorithm benchmark with Friedman ranks."""
    _, report = friedman_nemenyi(metrics)
    ranks = report["average_ranks"]

    rows: List[Dict[str, object]] = []
    for algorithm, group in metrics.groupby("algorithm"):
        values = group["normalized_hv"]
        runtime = group["runtime_sec"].mean() if "runtime_sec" in group else np.nan
        rows.append({
            "algorithm": LABELS.get(algorithm, algorithm),
            "n_runs": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)),
            "mean_rank": float(ranks.get(algorithm, np.nan)),
            "mean_runtime_s": float(runtime) if runtime == runtime else np.nan,
        })
    table = pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)
    return table, report


def table_correlations(populations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Table 12: Pearson correlations, with Spearman as a cross-check."""
    feasible = populations
    if "feasible" in populations.columns:
        flag = populations["feasible"]
        keep = flag if flag.dtype == bool else (
            flag.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        )
        feasible = populations[keep]
    F = feasible[OBJECTIVE_COLUMNS].dropna()
    pearson = F.corr(method="pearson")
    spearman = F.corr(method="spearman")
    pearson.index = pearson.columns = OBJECTIVE_LABELS
    spearman.index = spearman.columns = OBJECTIVE_LABELS
    return pearson, spearman


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _figure(out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return plt


def plot_distribution(metrics_by_plan: Dict[str, pd.DataFrame], out_path: Path) -> None:
    plt = _figure(out_path)
    plans = list(metrics_by_plan)
    fig, axes = plt.subplots(1, len(plans), figsize=(5 * len(plans), 4), squeeze=False, sharey=True)
    for ax, plan in zip(axes[0], plans):
        df = metrics_by_plan[plan]
        algorithms = sorted(df["algorithm"].unique())
        data = [df[df["algorithm"] == a]["normalized_hv"].dropna() for a in algorithms]
        ax.boxplot(data, labels=[LABELS.get(a, a) for a in algorithms], showmeans=True, whis=1.5)
        ax.set_title(plan, fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Normalized hypervolume")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_caterpillar(metrics: pd.DataFrame, algo_a: str, algo_b: str, out_path: Path) -> None:
    """Per-profile mean paired difference, sorted; negative favours ``algo_b``."""
    plt = _figure(out_path)
    from src.statistics import paired_differences

    _, profile_level = paired_differences(metrics, algo_a, algo_b)
    ordered = profile_level.sort_values()
    colours = ["tab:red" if v > 0 else "tab:green" for v in ordered]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(ordered)), ordered.to_numpy(), color=colours, width=1.0)
    ax.axhline(ordered.mean(), ls="--", color="tab:blue",
               label=f"mean = {ordered.mean():.4f}")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(f"Profiles sorted by HV gain ({LABELS.get(algo_b, algo_b)} - {LABELS.get(algo_a, algo_a)})")
    ax.set_ylabel("Per-profile mean HV difference")
    ax.set_title(f"{LABELS.get(algo_b, algo_b)} gain on {(ordered < 0).sum()}/{len(ordered)} profiles")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dz_distribution(per_profile: pd.DataFrame, out_path: Path) -> None:
    plt = _figure(out_path)
    dz = per_profile["cohen_dz"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(dz, bins=25, color="tab:red", edgecolor="black", alpha=0.75)
    ax.axvline(dz.mean(), color="red", label=f"mean = {dz.mean():.2f}")
    ax.axvline(dz.median(), color="orange", ls="-.", label=f"median = {dz.median():.2f}")
    ax.axvline(0.0, color="black", ls="--")
    ax.set_xlabel("Cohen's $d_z$ (paired)")
    ax.set_ylabel("Profiles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cd_diagram(report: Dict[str, object], out_path: Path) -> None:
    plt = _figure(out_path)
    ranks = report["average_ranks"]
    cd = report[[k for k in report if k.startswith("critical_difference")][0]]
    ordered = sorted(ranks.items(), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(8, 2.6))
    lo, hi = 1, len(ranks)
    ax.set_xlim(lo - 0.2, hi + 0.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.75, lo, hi, color="black")
    for tick in range(lo, hi + 1):
        ax.vlines(tick, 0.72, 0.78, color="black")
        ax.text(tick, 0.83, str(tick), ha="center", fontsize=9)

    for i, (algorithm, rank) in enumerate(ordered):
        y = 0.55 - 0.11 * i
        ax.plot([rank, rank], [0.75, y], color="black", lw=0.8)
        ax.plot([rank, lo - 0.15], [y, y], color="black", lw=0.8)
        ax.text(lo - 0.2, y, f"{LABELS.get(algorithm, algorithm)} ({rank:.2f})",
                ha="right", va="center", fontsize=9)

    for i in range(len(ordered)):
        group = [r for _, r in ordered if r - ordered[i][1] < cd and r >= ordered[i][1]]
        if len(group) > 1:
            ax.hlines(0.68 - 0.03 * i, min(group), max(group), lw=3, color="grey")

    ax.text((lo + hi) / 2, 0.95, f"CD = {cd:.3f} (alpha = 0.05)", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_convergence(run_dirs: Sequence[Path], out_path: Path) -> Optional[pd.DataFrame]:
    """Hypervolume against generation, averaged over profiles and seeds."""
    frames: List[pd.DataFrame] = []
    for root in run_dirs:
        for ckpt in Path(root).rglob("checkpoints/history"):
            for f in sorted(ckpt.glob("*.csv")):
                try:
                    frames.append(pd.read_csv(f))
                except Exception:  # pragma: no cover
                    continue
    if not frames:
        return None

    history = pd.concat(frames, ignore_index=True)
    history["algorithm"] = history["algorithm"].map(canonical_algorithm)
    history = history.dropna(subset=["hypervolume"])
    peak = history.groupby("profile_id")["hypervolume"].transform("max").replace(0, np.nan)
    history["nhv"] = history["hypervolume"] / peak

    plt = _figure(out_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algorithm, group in history.groupby("algorithm"):
        curve = group.groupby("generation")["nhv"].agg(["mean", "std"])
        ax.plot(curve.index, curve["mean"], label=LABELS.get(algorithm, algorithm))
        ax.fill_between(curve.index, curve["mean"] - curve["std"],
                        curve["mean"] + curve["std"], alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized hypervolume")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    feasibility = history.groupby(["algorithm", "generation"])["feasible_ratio"].mean()
    return feasibility.reset_index()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["results/outputs_main"])
    parser.add_argument("--out", default="results/analytics")
    parser.add_argument("--algo-a", default="nsga2")
    parser.add_argument("--algo-b", default="pi_nsga3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    run_dirs = [Path(r) for r in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics_by_plan = load_metrics(run_dirs)
    profiles = load_profiles_metadata(run_dirs)
    summary: Dict[str, object] = {}

    for plan, metrics in metrics_by_plan.items():
        algorithms = set(metrics["algorithm"].unique())
        prefix = out / plan

        table_distribution(metrics).to_csv(f"{prefix}_table8_distribution.csv", index=False)

        if profiles is not None:
            for stratum in ("archetype", "trip_distance_bin"):
                if stratum in profiles.columns:
                    table_per_stratum(metrics, profiles, stratum).to_csv(
                        f"{prefix}_table9_per_{stratum}.csv", index=False)

        if {args.algo_a, args.algo_b} <= algorithms:
            result = compare(metrics, algo_a=args.algo_a, algo_b=args.algo_b)
            summary[plan] = result

            per_profile = per_profile_table(metrics, args.algo_a, args.algo_b)
            per_profile.to_csv(f"{prefix}_per_profile_paired_tests.csv", index=False)

            plot_caterpillar(metrics, args.algo_a, args.algo_b,
                             out / f"{plan}_fig_caterpillar.png")
            plot_dz_distribution(per_profile, out / f"{plan}_fig_dz_distribution.png")

            logger.info(
                "%s: mean diff %.5f, profile d_z %.3f, p=%.3g, %d/%d profiles",
                plan, result["mean_diff"], result["dz_profile_level_confirmatory"],
                result["wilcoxon_profile_p"], result["profile_wins_a"], result["n_profiles"],
            )

        if len(algorithms) >= 3:
            table, report = table_extended(metrics)
            table.to_csv(f"{prefix}_table10_benchmark.csv", index=False)
            pd.DataFrame(report["pairwise"]).to_csv(f"{prefix}_table11_nemenyi.csv", index=False)
            with open(f"{prefix}_friedman_nemenyi.json", "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            plot_cd_diagram(report, out / f"{plan}_fig_cd_diagram.png")

    plot_distribution(metrics_by_plan, out / "fig_distribution_per_plan.png")

    populations = load_populations(run_dirs)
    if populations is not None and set(OBJECTIVE_COLUMNS) <= set(populations.columns):
        pearson, spearman = table_correlations(populations)
        pearson.to_csv(out / "table12_objective_correlations.csv")
        spearman.to_csv(out / "table12_objective_correlations_spearman.csv")
        logger.info("max |Pearson - Spearman| = %.4f",
                    float((pearson - spearman).abs().to_numpy().max()))

    feasibility = plot_convergence(run_dirs, out / "fig_hv_convergence.png")
    if feasibility is not None:
        feasibility.to_csv(out / "feasibility_ratio_per_generation.csv", index=False)

    with open(out / "analytics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info("Analytics written to %s", out)


if __name__ == "__main__":
    main()
