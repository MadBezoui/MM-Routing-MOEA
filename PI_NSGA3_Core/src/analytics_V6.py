"""analytics_V6.py
==================
Extended analytics + visualizations for V6 outputs.

Produces ~30 figures and ~12 stat tables in <output_dir>/analytics/.

Tables (CSV / JSON)
-------------------
    summary_stats_per_plan.csv
    paired_effect_sizes_per_plan.csv          # Cohen's d_z, Wilcoxon r/p, bootstrap CI
    per_profile_dz_distribution.csv           # paired d_z per profile
    win_tie_loss_per_plan.csv
    friedman_nemenyi_extended.json            # Friedman chi2 + Nemenyi pairwise
    friedman_pivot_extended.csv
    cd_diagram_data.json                      # average ranks + CD value
    strata_breakdown_per_plan.csv             # archetype x trip_distance x algo
    per_archetype_summary.csv
    per_trip_distance_summary.csv
    objective_correlations.csv                # Pearson, all 4 obj
    intrinsic_dimensionality.json             # TwoNN estimate
    per_algorithm_runtime_summary.csv

Figures (PNG)
-------------
    A. Distribution of normalized HV
        fig_box_normalized_hv.png
        fig_violin_per_plan.png
        fig_ecdf_normalized_hv.png
        fig_strip_per_plan.png
        fig_box_per_archetype.png
        fig_box_per_trip_distance.png
    B. Pairwise comparisons
        fig_scatter_nsga2_vs_nsga3.png
        fig_scatter_matrix_all_pairs.png
        fig_caterpillar_nsga3_minus_nsga2.png
        fig_dz_distribution.png
    C. Convergence (per generation)
        fig_hv_convergence.png
        fig_hv_convergence_per_profile.png
        fig_spacing_convergence.png
        fig_feasible_ratio_convergence.png
    D. Benchmarks
        fig_cd_diagram.png
        fig_runtime_comparison.png
    E. Strata
        fig_heatmap_strata.png
        fig_bar_per_archetype.png
        fig_bar_per_trip_distance.png
    F. Solution space
        fig_pareto_fronts_2d.png
        fig_pareto_3d.png
        fig_pca_solutions.png
        fig_objective_correlations.png
        fig_parallel_coordinates.png
        fig_mode_share_per_algorithm.png
    G. Comfort surrogate
        fig_comfort_model_comparison.png
        fig_comfort_region_errors.png

Run:
    python analytics_V6.py [--output_dir <dir>] [--analytics_dir <dir>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_3D = True
except ImportError:
    HAS_3D = False


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
OBJECTIVE_COLUMNS = ["obj_1", "obj_2", "obj_3", "obj_4"]
OBJECTIVE_LABELS = ["Travel time (min)", "Cost (€)", "Emissions (kg)", "Discomfort"]
OBJECTIVE_SHORT = ["Time", "Cost", "Emissions", "Discomfort"]
ALGORITHM_ORDER = ["nsga2", "nsga3", "moead", "smsemoa"]
ALGORITHM_LABELS = {
    "nsga2": "NSGA-II",
    "nsga3": "NSGA-III (informed)",
    "moead": "MOEA/D",
    "smsemoa": "SMS-EMOA",
}
ALGORITHM_COLORS = {
    "nsga2": "#1f77b4",
    "nsga3": "#d62728",
    "moead": "#2ca02c",
    "smsemoa": "#ff7f0e",
}
PLAN_NAMES = [
    "main_nsga2_vs_nsga3_150profiles",
    "extended_benchmark_30profiles",
    "representative_curves_10profiles",
]
MODE_COLUMNS = ["walk", "bike", "bus", "tram", "car"]
MODE_COLORS = {
    "walk": "#9467bd",
    "bike": "#17becf",
    "bus": "#1f77b4",
    "tram": "#2ca02c",
    "car": "#d62728",
}
ARCHETYPE_ORDER = ["comfort_seeking", "cost_sensitive", "eco_conscious", "time_sensitive"]
TRIP_DISTANCE_ORDER = ["short", "medium", "long"]


# ============================================================================
# Section 0 — Loading helpers
# ============================================================================
def _coerce_bool_safe(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"true", "1", "yes", "y", "t"})


def load_recovered_metrics(output_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for plan in PLAN_NAMES:
        path = output_dir / f"{plan}_final_generation_recovered.csv"
        if path.exists():
            out[plan] = pd.read_csv(path)
        else:
            print(f"  [warn] missing recovered file: {path.name}")
    return out


def load_profiles_metadata(output_dir: Path) -> Optional[pd.DataFrame]:
    path = output_dir / "profiles_all_plans.csv"
    return pd.read_csv(path) if path.exists() else None


def _load_csvs_from_dir(dir_path: Path, max_files: Optional[int] = None) -> Optional[pd.DataFrame]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None


def load_population_sample(plan_dir: Path, max_files: int = 600) -> Optional[pd.DataFrame]:
    df = _load_csvs_from_dir(plan_dir / "checkpoints" / "population", max_files=max_files)
    if df is None:
        fallback = plan_dir / "all_population_results.csv"
        if fallback.exists():
            df = pd.read_csv(fallback)
    if df is not None and "feasible" in df.columns:
        df["feasible_bool"] = _coerce_bool_safe(df["feasible"])
    return df


def load_history_sample(plan_dir: Path, max_files: int = 1500) -> Optional[pd.DataFrame]:
    df = _load_csvs_from_dir(plan_dir / "checkpoints" / "history", max_files=max_files)
    if df is None:
        fallback = plan_dir / "all_history_results.csv"
        if fallback.exists():
            df = pd.read_csv(fallback)
    return df


# ============================================================================
# Section 1 — Statistics
# ============================================================================
def summary_stats_per_plan(metrics_by_plan: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for plan, df in metrics_by_plan.items():
        for algo, sub in df.groupby("algorithm"):
            for col in ["normalized_hv", "hypervolume", "igd", "n_feasible_run"]:
                if col not in sub.columns:
                    continue
                values = sub[col].dropna()
                if len(values) == 0:
                    continue
                rows.append({
                    "plan": plan,
                    "algorithm": algo,
                    "metric": col,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                })
    return pd.DataFrame(rows)


def _paired_align(df: pd.DataFrame, a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
    da = df[df["algorithm"] == a].set_index(["profile_id", "seed"])["normalized_hv"]
    db = df[df["algorithm"] == b].set_index(["profile_id", "seed"])["normalized_hv"]
    common = da.index.intersection(db.index)
    if len(common) == 0:
        return np.array([]), np.array([])
    x = da.loc[common].to_numpy()
    y = db.loc[common].to_numpy()
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask], y[mask]


def paired_stats(metrics_df: pd.DataFrame, plan_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    algos = sorted(metrics_df["algorithm"].unique())
    rows_eff, rows_wtl = [], []
    rng = np.random.default_rng(20260509)
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            x, y = _paired_align(metrics_df, a, b)
            n = len(x)
            if n < 5:
                continue
            diff = x - y
            mean_diff = float(diff.mean())
            sd_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")
            cohen_dz = mean_diff / sd_diff if sd_diff and sd_diff > 0 else float("nan")

            try:
                wilc = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                p_val = float(wilc.pvalue)
            except Exception:
                p_val = float("nan")
            if not np.isnan(p_val) and 0 < p_val < 1:
                z_abs = abs(stats.norm.ppf(1 - p_val / 2))
                wilc_r = z_abs / np.sqrt(n)
                if mean_diff < 0:
                    wilc_r = -wilc_r
            else:
                wilc_r = float("nan")

            wins_a = int((diff > 1e-6).sum())
            wins_b = int((diff < -1e-6).sum())
            ties = int((np.abs(diff) <= 1e-6).sum())

            try:
                idx = rng.integers(0, n, size=(2000, n))
                boot = diff[idx].mean(axis=1)
                ci_low = float(np.percentile(boot, 2.5))
                ci_high = float(np.percentile(boot, 97.5))
            except Exception:
                ci_low = ci_high = float("nan")

            rows_eff.append({
                "plan": plan_name, "algo_a": a, "algo_b": b, "n_pairs": n,
                "mean_diff_a_minus_b": mean_diff, "sd_diff": sd_diff,
                "cohen_dz_paired": cohen_dz,
                "wilcoxon_p": p_val, "wilcoxon_r_signed": wilc_r,
                "ci_low_95_mean_diff": ci_low, "ci_high_95_mean_diff": ci_high,
            })
            rows_wtl.append({
                "plan": plan_name, "algo_a": a, "algo_b": b, "n_pairs": n,
                "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                "win_rate_a": wins_a / n, "win_rate_b": wins_b / n,
            })
    return pd.DataFrame(rows_eff), pd.DataFrame(rows_wtl)


def per_profile_dz(metrics_df: pd.DataFrame, plan_name: str) -> pd.DataFrame:
    algos = sorted(metrics_df["algorithm"].unique())
    rows = []
    for profile_id, sub in metrics_df.groupby("profile_id"):
        for i, a in enumerate(algos):
            for b in algos[i + 1:]:
                xa = sub[sub["algorithm"] == a].set_index("seed")["normalized_hv"]
                xb = sub[sub["algorithm"] == b].set_index("seed")["normalized_hv"]
                common = xa.index.intersection(xb.index)
                if len(common) < 3:
                    continue
                diff = (xa.loc[common] - xb.loc[common]).to_numpy()
                diff = diff[~np.isnan(diff)]
                if len(diff) < 3:
                    continue
                mean_d = float(diff.mean())
                sd_d = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
                dz = mean_d / sd_d if sd_d and sd_d > 0 else float("nan")
                rows.append({
                    "plan": plan_name, "profile_id": profile_id,
                    "algo_a": a, "algo_b": b, "n": int(len(diff)),
                    "mean_diff": mean_d, "cohen_dz_per_profile": dz,
                })
    return pd.DataFrame(rows)


def friedman_nemenyi(metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    pivot = (
        metrics_df.groupby(["profile_id", "algorithm"])["normalized_hv"]
        .mean().unstack("algorithm").dropna()
    )
    if pivot.shape[1] < 3:
        return pivot, {"error": "Friedman requires >= 3 groups"}
    samples = [pivot[c].to_numpy() for c in pivot.columns]
    try:
        chi2, p = stats.friedmanchisquare(*samples)
    except Exception as e:
        return pivot, {"error": str(e)}
    k, n = pivot.shape[1], pivot.shape[0]
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = {algo: float(ranks[algo].mean()) for algo in pivot.columns}
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    q = q_table.get(k, 2.569)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n)))
    pairs = []
    cols = list(pivot.columns)
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(avg_ranks[cols[i]] - avg_ranks[cols[j]])
            pairs.append({
                "algo_a": cols[i], "algo_b": cols[j],
                "rank_diff": d, "significant_at_0.05": bool(d > cd),
            })
    return pivot, {
        "n_profiles": int(n), "n_algorithms": int(k),
        "friedman_chi2": float(chi2), "friedman_p": float(p),
        "average_ranks": avg_ranks, "critical_difference_0.05": cd,
        "pairwise_rank_comparisons": pairs,
    }


def strata_breakdown(metrics_df: pd.DataFrame, profiles_meta: pd.DataFrame, plan_name: str) -> pd.DataFrame:
    if "run_plan" in profiles_meta.columns:
        meta = profiles_meta[profiles_meta["run_plan"] == plan_name][
            ["profile_id", "archetype", "trip_distance_bin"]
        ].drop_duplicates()
    else:
        meta = profiles_meta[["profile_id", "archetype", "trip_distance_bin"]].drop_duplicates()
    merged = metrics_df.merge(meta, on="profile_id", how="left")
    grouped = (
        merged.groupby(["archetype", "trip_distance_bin", "algorithm"])
        .agg(
            mean_norm_hv=("normalized_hv", "mean"),
            median_norm_hv=("normalized_hv", "median"),
            std_norm_hv=("normalized_hv", "std"),
            n_runs=("normalized_hv", "count"),
        ).reset_index()
    )
    grouped["plan"] = plan_name
    return grouped


def estimate_intrinsic_dim_twonn(points: np.ndarray) -> float:
    n = len(points)
    if n < 50:
        return float("nan")
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(points))
    np.fill_diagonal(D, np.inf)
    sorted_D = np.sort(D, axis=1)
    r1 = sorted_D[:, 0]
    r2 = sorted_D[:, 1]
    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 20:
        return float("nan")
    mu = r2[valid] / r1[valid]
    mu_sorted = np.sort(mu)
    F = (np.arange(1, len(mu_sorted) + 1)) / (len(mu_sorted) + 1)
    log_mu = np.log(mu_sorted)
    log_F = -np.log(1 - F)
    valid_lf = np.isfinite(log_F) & np.isfinite(log_mu)
    if valid_lf.sum() < 10:
        return float("nan")
    num = np.sum(log_mu[valid_lf] * log_F[valid_lf])
    den = np.sum(log_mu[valid_lf] ** 2)
    return float(num / den) if den > 0 else float("nan")


# ============================================================================
# Section 2 — Distribution figures
# ============================================================================
def plot_box_normalized_hv(metrics_by_plan: Dict[str, pd.DataFrame], out_path: Path):
    n = len(metrics_by_plan)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (plan, df) in zip(axes, metrics_by_plan.items()):
        algos = [a for a in ALGORITHM_ORDER if a in df["algorithm"].unique()]
        data = [df[df["algorithm"] == a]["normalized_hv"].dropna().to_numpy() for a in algos]
        bp = ax.boxplot(
            data, labels=[ALGORITHM_LABELS[a] for a in algos],
            patch_artist=True, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"},
        )
        for patch, a in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGORITHM_COLORS[a])
            patch.set_alpha(0.75)
        ax.set_title(plan, fontsize=10)
        ax.set_ylabel("Normalized HV" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Normalized hypervolume per algorithm — by plan", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_violin_per_plan(metrics_by_plan: Dict[str, pd.DataFrame], out_path: Path):
    n = len(metrics_by_plan)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (plan, df) in zip(axes, metrics_by_plan.items()):
        algos = [a for a in ALGORITHM_ORDER if a in df["algorithm"].unique()]
        data = [df[df["algorithm"] == a]["normalized_hv"].dropna().to_numpy() for a in algos]
        if any(len(d) > 1 for d in data):
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            for body, a in zip(parts["bodies"], algos):
                body.set_facecolor(ALGORITHM_COLORS[a])
                body.set_alpha(0.6)
        ax.set_xticks(range(1, len(algos) + 1))
        ax.set_xticklabels([ALGORITHM_LABELS[a] for a in algos], rotation=15)
        ax.set_title(plan, fontsize=10)
        ax.set_ylabel("Normalized HV" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Density of normalized HV (violin)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf(metrics_by_plan: Dict[str, pd.DataFrame], out_path: Path):
    n = len(metrics_by_plan)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (plan, df) in zip(axes, metrics_by_plan.items()):
        for algo in [a for a in ALGORITHM_ORDER if a in df["algorithm"].unique()]:
            vals = np.sort(df[df["algorithm"] == algo]["normalized_hv"].dropna().to_numpy())
            if len(vals) == 0:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, y, color=ALGORITHM_COLORS[algo],
                    label=ALGORITHM_LABELS[algo], linewidth=2)
        ax.set_title(plan, fontsize=10)
        ax.set_xlabel("Normalized HV")
        ax.set_ylabel("Empirical CDF" if ax is axes[0] else "")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Empirical CDF of normalized HV", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_strip_per_plan(metrics_by_plan: Dict[str, pd.DataFrame], out_path: Path):
    n = len(metrics_by_plan)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, (plan, df) in zip(axes, metrics_by_plan.items()):
        algos = [a for a in ALGORITHM_ORDER if a in df["algorithm"].unique()]
        for i, algo in enumerate(algos):
            vals = df[df["algorithm"] == algo]["normalized_hv"].dropna().to_numpy()
            jitter = rng.normal(0, 0.07, size=len(vals))
            ax.scatter(np.full_like(vals, i) + jitter, vals,
                       color=ALGORITHM_COLORS[algo], alpha=0.35, s=8, edgecolors="none")
            ax.scatter([i], [np.median(vals)] if len(vals) else [],
                       color="black", marker="_", s=180, linewidth=2)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels([ALGORITHM_LABELS[a] for a in algos], rotation=15)
        ax.set_title(plan, fontsize=10)
        ax.set_ylabel("Normalized HV" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Strip plot of normalized HV (each dot = one (profile, seed))", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_box_per_archetype(metrics_df: pd.DataFrame, profiles_meta: pd.DataFrame, out_path: Path, plan_name: str):
    if "run_plan" in profiles_meta.columns:
        meta = profiles_meta[profiles_meta["run_plan"] == plan_name]
    else:
        meta = profiles_meta
    merged = metrics_df.merge(meta[["profile_id", "archetype"]].drop_duplicates(), on="profile_id", how="left")
    archetypes = [a for a in ARCHETYPE_ORDER if a in merged["archetype"].unique()]
    if not archetypes:
        return
    fig, axes = plt.subplots(1, len(archetypes), figsize=(4 * len(archetypes), 4.5), sharey=True)
    if len(archetypes) == 1:
        axes = [axes]
    for ax, arche in zip(axes, archetypes):
        sub = merged[merged["archetype"] == arche]
        algos = [a for a in ALGORITHM_ORDER if a in sub["algorithm"].unique()]
        data = [sub[sub["algorithm"] == a]["normalized_hv"].dropna().to_numpy() for a in algos]
        bp = ax.boxplot(data, labels=[ALGORITHM_LABELS[a] for a in algos],
                        patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"})
        for patch, a in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGORITHM_COLORS[a])
            patch.set_alpha(0.7)
        ax.set_title(arche, fontsize=10)
        ax.set_ylabel("Normalized HV" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Normalized HV per archetype — {plan_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_box_per_trip_distance(metrics_df: pd.DataFrame, profiles_meta: pd.DataFrame, out_path: Path, plan_name: str):
    if "run_plan" in profiles_meta.columns:
        meta = profiles_meta[profiles_meta["run_plan"] == plan_name]
    else:
        meta = profiles_meta
    merged = metrics_df.merge(meta[["profile_id", "trip_distance_bin"]].drop_duplicates(), on="profile_id", how="left")
    bins = [b for b in TRIP_DISTANCE_ORDER if b in merged["trip_distance_bin"].unique()]
    if not bins:
        return
    fig, axes = plt.subplots(1, len(bins), figsize=(4 * len(bins), 4.5), sharey=True)
    if len(bins) == 1:
        axes = [axes]
    for ax, b in zip(axes, bins):
        sub = merged[merged["trip_distance_bin"] == b]
        algos = [a for a in ALGORITHM_ORDER if a in sub["algorithm"].unique()]
        data = [sub[sub["algorithm"] == a]["normalized_hv"].dropna().to_numpy() for a in algos]
        bp = ax.boxplot(data, labels=[ALGORITHM_LABELS[a] for a in algos],
                        patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black"})
        for patch, a in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGORITHM_COLORS[a])
            patch.set_alpha(0.7)
        ax.set_title(f"trip = {b}", fontsize=10)
        ax.set_ylabel("Normalized HV" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Normalized HV per trip-distance bin — {plan_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Section 3 — Pairwise comparison figures
# ============================================================================
def plot_scatter_pair(metrics_df: pd.DataFrame, a: str, b: str, out_path: Path):
    pivot = metrics_df.groupby(["profile_id", "algorithm"])["normalized_hv"].median().unstack("algorithm")
    if a not in pivot.columns or b not in pivot.columns:
        return
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(pivot[a], pivot[b], alpha=0.55, s=42, color=ALGORITHM_COLORS.get(b, "C0"), edgecolor="white")
    lo = float(min(pivot[a].min(), pivot[b].min())) - 0.005
    hi = float(max(pivot[a].max(), pivot[b].max())) + 0.005
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.7, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{ALGORITHM_LABELS.get(a, a)} median normalized HV")
    ax.set_ylabel(f"{ALGORITHM_LABELS.get(b, b)} median normalized HV")
    above = int((pivot[b] > pivot[a]).sum()); below = int((pivot[b] < pivot[a]).sum())
    ax.set_title(f"Per-profile median HV: {ALGORITHM_LABELS.get(b, b)} > {ALGORITHM_LABELS.get(a, a)} on {above}/{above + below} profiles")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_scatter_matrix_all_pairs(metrics_df: pd.DataFrame, out_path: Path):
    pivot = metrics_df.groupby(["profile_id", "algorithm"])["normalized_hv"].median().unstack("algorithm")
    algos = [a for a in ALGORITHM_ORDER if a in pivot.columns]
    if len(algos) < 2:
        return
    n = len(algos)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n), sharex=True, sharey=True)
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            ax = axes[i, j]
            if i == j:
                vals = pivot[a].dropna().to_numpy()
                ax.hist(vals, bins=30, color=ALGORITHM_COLORS[a], alpha=0.7, edgecolor="white")
                ax.set_title(ALGORITHM_LABELS[a], fontsize=9)
            else:
                ax.scatter(pivot[b], pivot[a], alpha=0.5, s=18, color=ALGORITHM_COLORS[a], edgecolor="none")
                lo, hi = pivot.values.min() - 0.01, pivot.values.max() + 0.01
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(ALGORITHM_LABELS[b], fontsize=8)
            if j == 0:
                ax.set_ylabel(ALGORITHM_LABELS[a], fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("Per-profile median normalized HV — pairwise scatter matrix", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_caterpillar(metrics_df: pd.DataFrame, a: str, b: str, out_path: Path, label: str):
    pivot = metrics_df.groupby(["profile_id", "algorithm"])["normalized_hv"].median().unstack("algorithm")
    if a not in pivot.columns or b not in pivot.columns:
        return
    diff = (pivot[b] - pivot[a]).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in diff.values]
    ax.bar(range(len(diff)), diff.values, color=colors, alpha=0.85, edgecolor="none", width=0.9)
    ax.axhline(0, color="black", linewidth=0.7)
    mean_d = diff.mean()
    ax.axhline(mean_d, color="blue", linestyle="--", linewidth=1.2, label=f"mean = {mean_d:.4f}")
    ax.set_xlabel(f"Profiles (sorted by HV gain {ALGORITHM_LABELS.get(b, b)} − {ALGORITHM_LABELS.get(a, a)})")
    ax.set_ylabel("Per-profile median HV difference")
    ax.set_title(f"Caterpillar plot — {label}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    n_above = int((diff > 0).sum()); n_below = int((diff < 0).sum())
    ax.text(0.02, 0.95, f"NSGA-III gain on {n_above}/{n_above + n_below} profiles",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_dz_distribution(dz_df: pd.DataFrame, out_path: Path):
    if dz_df.empty:
        return
    pairs = dz_df[["algo_a", "algo_b"]].drop_duplicates().to_numpy()
    if len(pairs) == 0:
        return
    cols = min(len(pairs), 3)
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx // cols, idx % cols]
        sub = dz_df[(dz_df["algo_a"] == a) & (dz_df["algo_b"] == b)]
        vals = sub["cohen_dz_per_profile"].dropna().to_numpy()
        if len(vals) == 0:
            ax.set_visible(False); continue
        ax.hist(vals, bins=20, alpha=0.75, edgecolor="black", color=ALGORITHM_COLORS.get(b, "C0"))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(np.mean(vals), color="red", linewidth=1.4, label=f"mean={np.mean(vals):.2f}")
        ax.axvline(np.median(vals), color="orange", linewidth=1.2, linestyle="-.", label=f"median={np.median(vals):.2f}")
        ax.set_title(f"d_z per profile: {ALGORITHM_LABELS.get(a, a)} − {ALGORITHM_LABELS.get(b, b)}", fontsize=10)
        ax.set_xlabel("Cohen's d_z (paired)")
        ax.set_ylabel("Profiles")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for idx in range(len(pairs), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================================
# Section 4 — Convergence figures (history-based)
# ============================================================================
def plot_history_metric(history_df: pd.DataFrame, metric: str, out_path: Path, ylabel: str, title: str):
    if history_df is None or history_df.empty or metric not in history_df.columns:
        return
    valid = history_df.dropna(subset=[metric])
    if valid.empty:
        print(f"  [skip] {metric} has no non-NaN values; cannot plot {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algo in [a for a in ALGORITHM_ORDER if a in valid["algorithm"].unique()]:
        sub = valid[valid["algorithm"] == algo]
        agg = sub.groupby("generation")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(agg["generation"], agg["mean"],
                color=ALGORITHM_COLORS[algo], label=ALGORITHM_LABELS[algo], linewidth=2)
        ax.fill_between(agg["generation"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=ALGORITHM_COLORS[algo], alpha=0.18)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_hv_per_profile_grid(history_df: pd.DataFrame, out_path: Path, max_profiles: int = 10):
    if history_df is None or "hypervolume" not in history_df.columns:
        return
    valid = history_df.dropna(subset=["hypervolume"])
    if valid.empty:
        return
    profiles = sorted(valid["profile_id"].unique())[:max_profiles]
    if not profiles:
        return
    cols = 5; rows = int(np.ceil(len(profiles) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), sharex=True, squeeze=False)
    for idx, profile in enumerate(profiles):
        ax = axes[idx // cols, idx % cols]
        sub = valid[valid["profile_id"] == profile]
        for algo in [a for a in ALGORITHM_ORDER if a in sub["algorithm"].unique()]:
            sub_a = sub[sub["algorithm"] == algo]
            agg = sub_a.groupby("generation")["hypervolume"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["generation"], agg["mean"], color=ALGORITHM_COLORS[algo], linewidth=1.4,
                    label=ALGORITHM_LABELS[algo])
            ax.fill_between(agg["generation"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=ALGORITHM_COLORS[algo], alpha=0.15)
        ax.set_title(profile, fontsize=8)
        ax.grid(alpha=0.3)
        if idx // cols == rows - 1:
            ax.set_xlabel("Gen", fontsize=8)
        if idx % cols == 0:
            ax.set_ylabel("HV", fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(len(profiles), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"HV vs generation — per profile ({len(profiles)} profiles)", fontsize=11, y=1.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================================
# Section 5 — Benchmark figures
# ============================================================================
def plot_cd_diagram(ftest: dict, out_path: Path):
    if "average_ranks" not in ftest:
        return
    ranks = ftest["average_ranks"]
    cd = ftest["critical_difference_0.05"]
    algos = sorted(ranks.keys(), key=lambda a: ranks[a])  # ascending: best first
    rank_vals = [ranks[a] for a in algos]
    k = len(algos)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x_min = max(0.5, min(rank_vals) - 0.3)
    x_max = min(k + 0.5, max(rank_vals) + 0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, 1.3)
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 1.0))
    ax.set_xticks(np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1))
    ax.tick_params(axis="x", direction="in", top=False)

    # Place algos at their ranks
    label_y = [0.7, 0.4] * k  # alternating heights
    for i, a in enumerate(algos):
        x = ranks[a]
        ax.plot([x, x], [1.0, 0.95], "k-", linewidth=1)
        if x < (x_min + x_max) / 2:
            ax.plot([x, x_min - 0.1], [0.95, 0.6 + 0.15 * (i % 2)], "k-", linewidth=0.8)
            ax.text(x_min - 0.15, 0.6 + 0.15 * (i % 2),
                    f"{ALGORITHM_LABELS.get(a, a)}\n(rank {ranks[a]:.2f})",
                    ha="right", va="center", fontsize=9)
        else:
            ax.plot([x, x_max + 0.1], [0.95, 0.6 + 0.15 * (i % 2)], "k-", linewidth=0.8)
            ax.text(x_max + 0.15, 0.6 + 0.15 * (i % 2),
                    f"{ALGORITHM_LABELS.get(a, a)}\n(rank {ranks[a]:.2f})",
                    ha="left", va="center", fontsize=9)

    # CD bar
    bar_y = 1.18
    ax.plot([x_min + 0.05, x_min + 0.05 + cd], [bar_y, bar_y], "k-", linewidth=2.2)
    ax.plot([x_min + 0.05, x_min + 0.05], [bar_y - 0.04, bar_y + 0.04], "k-", linewidth=2.2)
    ax.plot([x_min + 0.05 + cd, x_min + 0.05 + cd], [bar_y - 0.04, bar_y + 0.04], "k-", linewidth=2.2)
    ax.text(x_min + 0.05 + cd / 2, bar_y + 0.08, f"CD = {cd:.3f} (α=0.05)",
            ha="center", va="bottom", fontsize=10)

    # Connect non-significantly-different groups with horizontal bars
    rank_y = 0.9
    sorted_algos = algos
    sorted_ranks = rank_vals
    used = set()
    for i in range(k):
        for j in range(i + 1, k):
            if (sorted_ranks[j] - sorted_ranks[i]) <= cd:
                key = (i, j)
                if key in used:
                    continue
                used.add(key)
                ax.plot([sorted_ranks[i], sorted_ranks[j]], [rank_y, rank_y],
                        "k-", linewidth=4, alpha=0.55, solid_capstyle="round")
                rank_y -= 0.05

    ax.set_title(
        f"Critical Difference diagram (Friedman χ²={ftest['friedman_chi2']:.2f}, "
        f"p={ftest['friedman_p']:.2e}, n={ftest['n_profiles']} profiles)",
        fontsize=10,
    )
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_runtime_comparison(populations_df: pd.DataFrame, out_path: Path) -> Optional[pd.DataFrame]:
    if populations_df is None or "runtime_sec" not in populations_df.columns:
        return None
    runtime = (
        populations_df.drop_duplicates(["profile_id", "algorithm", "seed"])
        .groupby("algorithm")["runtime_sec"]
        .agg(["mean", "median", "std", "min", "max", "count"]).reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = [a for a in ALGORITHM_ORDER if a in runtime["algorithm"].to_numpy()]
    means = [float(runtime[runtime["algorithm"] == a]["mean"].iloc[0]) for a in algos]
    stds = [float(runtime[runtime["algorithm"] == a]["std"].iloc[0]) for a in algos]
    ax.bar([ALGORITHM_LABELS[a] for a in algos], means, yerr=stds, capsize=5,
           color=[ALGORITHM_COLORS[a] for a in algos], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Runtime per run (s, log scale)")
    ax.set_title("Wall-clock cost per run, by algorithm")
    ax.grid(axis="y", alpha=0.3, which="both")
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return runtime


# ============================================================================
# Section 6 — Strata figures
# ============================================================================
def plot_heatmap_strata(strata_df: pd.DataFrame, out_path: Path, title: str = ""):
    if strata_df.empty:
        return
    algos = [a for a in ALGORITHM_ORDER if a in strata_df["algorithm"].unique()]
    n = len(algos)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    last_im = None
    for ax, algo in zip(axes, algos):
        sub = strata_df[strata_df["algorithm"] == algo]
        pivot = sub.pivot(index="archetype", columns="trip_distance_bin", values="mean_norm_hv")
        pivot = pivot.reindex(
            index=[a for a in ARCHETYPE_ORDER if a in pivot.index],
            columns=[c for c in TRIP_DISTANCE_ORDER if c in pivot.columns],
        )
        last_im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_title(ALGORITHM_LABELS.get(algo, algo), fontsize=10)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="black" if v > 0.75 else "white", fontsize=8)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[-1], shrink=0.8, label="Mean normalized HV")
    fig.suptitle(title or "Mean normalized HV by archetype × trip_distance", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_bar_per_strata(strata_df: pd.DataFrame, group_col: str, out_path: Path, title: str):
    if strata_df.empty or group_col not in strata_df.columns:
        return
    order = ARCHETYPE_ORDER if group_col == "archetype" else TRIP_DISTANCE_ORDER
    groups = [g for g in order if g in strata_df[group_col].unique()]
    algos = [a for a in ALGORITHM_ORDER if a in strata_df["algorithm"].unique()]
    if not groups or not algos:
        return
    agg = strata_df.groupby([group_col, "algorithm"]).agg(
        mean=("mean_norm_hv", "mean"),
        std=("std_norm_hv", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(groups) * len(algos), 5))
    width = 0.8 / len(algos)
    x = np.arange(len(groups))
    for i, algo in enumerate(algos):
        sub = agg[agg["algorithm"] == algo].set_index(group_col).reindex(groups)
        ax.bar(x + i * width - 0.4 + width / 2, sub["mean"], width=width,
               yerr=sub["std"].fillna(0), capsize=3,
               color=ALGORITHM_COLORS[algo], alpha=0.85, label=ALGORITHM_LABELS[algo])
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel("Mean normalized HV")
    ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================================
# Section 7 — Solution-space figures
# ============================================================================
def plot_pareto_fronts_2d(populations_df: pd.DataFrame, out_path: Path, n_profiles: int = 4):
    if populations_df is None or populations_df.empty:
        return
    profiles = list(populations_df["profile_id"].unique()[:n_profiles])
    feasible = populations_df.copy()
    if "feasible_bool" in feasible.columns:
        feasible = feasible[feasible["feasible_bool"]]
    pair_specs = [
        (1, 2, OBJECTIVE_LABELS[0], OBJECTIVE_LABELS[1]),
        (1, 3, OBJECTIVE_LABELS[0], OBJECTIVE_LABELS[2]),
        (1, 4, OBJECTIVE_LABELS[0], OBJECTIVE_LABELS[3]),
        (2, 3, OBJECTIVE_LABELS[1], OBJECTIVE_LABELS[2]),
        (2, 4, OBJECTIVE_LABELS[1], OBJECTIVE_LABELS[3]),
        (3, 4, OBJECTIVE_LABELS[2], OBJECTIVE_LABELS[3]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (i, j, lx, ly) in zip(axes.ravel(), pair_specs):
        for algo in feasible["algorithm"].unique():
            sub = feasible[(feasible["algorithm"] == algo) & (feasible["profile_id"].isin(profiles))]
            if sub.empty:
                continue
            ax.scatter(sub[f"obj_{i}"], sub[f"obj_{j}"], alpha=0.35, s=10,
                       color=ALGORITHM_COLORS.get(algo, "gray"),
                       label=ALGORITHM_LABELS.get(algo, algo))
        ax.set_xlabel(lx); ax.set_ylabel(ly); ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"2D objective projections — {len(profiles)} sample profiles", y=1.04, fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_pareto_3d(populations_df: pd.DataFrame, out_path: Path, n_profiles: int = 3):
    if not HAS_3D or populations_df is None or populations_df.empty:
        return
    profiles = list(populations_df["profile_id"].unique()[:n_profiles])
    feasible = populations_df.copy()
    if "feasible_bool" in feasible.columns:
        feasible = feasible[feasible["feasible_bool"]]
    feasible = feasible[feasible["profile_id"].isin(profiles)]
    if feasible.empty:
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for algo in feasible["algorithm"].unique():
        sub = feasible[feasible["algorithm"] == algo]
        sc = ax.scatter(sub["obj_1"], sub["obj_2"], sub["obj_3"],
                        c=sub["obj_4"], cmap="viridis", alpha=0.6, s=18,
                        edgecolor=ALGORITHM_COLORS.get(algo, "gray"), linewidth=0.4,
                        label=ALGORITHM_LABELS.get(algo, algo))
    ax.set_xlabel(OBJECTIVE_LABELS[0]); ax.set_ylabel(OBJECTIVE_LABELS[1]); ax.set_zlabel(OBJECTIVE_LABELS[2])
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(OBJECTIVE_LABELS[3])
    ax.set_title(f"3D projection of feasible solutions (color = {OBJECTIVE_SHORT[3]})\n"
                 f"{len(profiles)} sample profiles", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_pca_solutions(populations_df: pd.DataFrame, out_path: Path, max_points: int = 6000):
    if not HAS_SKLEARN or populations_df is None or populations_df.empty:
        return
    df = populations_df.copy()
    if "feasible_bool" in df.columns:
        df = df[df["feasible_bool"]]
    df = df.dropna(subset=OBJECTIVE_COLUMNS)
    if df.empty:
        return
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
    X = df[OBJECTIVE_COLUMNS].to_numpy()
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    pca = PCA(n_components=2); Y = pca.fit_transform(Xs)
    if all(f"x{i}" in df.columns for i in range(5)):
        shares = df[[f"x{i}" for i in range(5)]].to_numpy()
        shares = shares / np.maximum(shares.sum(axis=1, keepdims=True), 1e-12)
        dom = np.array(MODE_COLUMNS)[np.argmax(shares, axis=1)]
    else:
        dom = np.array(["unknown"] * len(df))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for mode in MODE_COLUMNS:
        m = dom == mode
        if not m.any():
            continue
        axes[0].scatter(Y[m, 0], Y[m, 1], alpha=0.45, s=10, color=MODE_COLORS[mode], label=mode)
    axes[0].set_title(
        f"PCA — by dominant mode\nVar explained: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}", fontsize=10)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for algo in df["algorithm"].unique():
        m = (df["algorithm"] == algo).to_numpy()
        if not m.any():
            continue
        axes[1].scatter(Y[m, 0], Y[m, 1], alpha=0.45, s=10,
                        color=ALGORITHM_COLORS.get(algo, "gray"),
                        label=ALGORITHM_LABELS.get(algo, algo))
    axes[1].set_title("PCA — by algorithm", fontsize=10)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_objective_correlations(populations_df: pd.DataFrame, out_path: Path) -> Optional[pd.DataFrame]:
    if populations_df is None or populations_df.empty:
        return None
    df = populations_df.copy()
    if "feasible_bool" in df.columns:
        df = df[df["feasible_bool"]]
    df = df.dropna(subset=OBJECTIVE_COLUMNS)
    if df.empty:
        return None
    corr = df[OBJECTIVE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(4)); ax.set_xticklabels(OBJECTIVE_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(OBJECTIVE_LABELS)
    for i in range(4):
        for j in range(4):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=10)
    ax.set_title("Pearson correlations between objectives\n(over feasible solutions)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return corr


def plot_parallel_coordinates(populations_df: pd.DataFrame, out_path: Path, max_points_per_algo: int = 800):
    if populations_df is None or populations_df.empty:
        return
    df = populations_df.copy()
    if "feasible_bool" in df.columns:
        df = df[df["feasible_bool"]]
    df = df.dropna(subset=OBJECTIVE_COLUMNS)
    if df.empty:
        return
    # Normalize each objective to [0, 1] for comparable axes
    Xnorm = df[OBJECTIVE_COLUMNS].copy()
    for col in OBJECTIVE_COLUMNS:
        lo, hi = Xnorm[col].min(), Xnorm[col].max()
        Xnorm[col] = (Xnorm[col] - lo) / (hi - lo) if hi > lo else 0.5
    Xnorm["algorithm"] = df["algorithm"].values
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for algo in [a for a in ALGORITHM_ORDER if a in Xnorm["algorithm"].unique()]:
        sub = Xnorm[Xnorm["algorithm"] == algo]
        if len(sub) > max_points_per_algo:
            sub = sub.sample(n=max_points_per_algo, random_state=42)
        for _, row in sub.iterrows():
            ax.plot(range(4), [row[c] for c in OBJECTIVE_COLUMNS],
                    color=ALGORITHM_COLORS[algo], alpha=0.04, linewidth=0.6)
    # Median lines per algorithm
    for algo in [a for a in ALGORITHM_ORDER if a in Xnorm["algorithm"].unique()]:
        med = Xnorm[Xnorm["algorithm"] == algo][OBJECTIVE_COLUMNS].median().to_numpy()
        ax.plot(range(4), med, color=ALGORITHM_COLORS[algo], linewidth=3,
                marker="o", markersize=8, label=f"{ALGORITHM_LABELS[algo]} median")
    ax.set_xticks(range(4)); ax.set_xticklabels(OBJECTIVE_LABELS)
    ax.set_ylabel("Min-max normalized value")
    ax.set_title("Parallel coordinates of feasible solutions (median per algorithm overlaid)")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_mode_share_per_algorithm(populations_df: pd.DataFrame, out_path: Path):
    if populations_df is None or populations_df.empty:
        return
    df = populations_df.copy()
    if "feasible_bool" in df.columns:
        df = df[df["feasible_bool"]]
    if not all(f"x{i}" in df.columns for i in range(5)):
        return
    shares = df[[f"x{i}" for i in range(5)]].to_numpy()
    shares = shares / np.maximum(shares.sum(axis=1, keepdims=True), 1e-12)
    df = df.copy()
    for i, mode in enumerate(MODE_COLUMNS):
        df[f"share_{mode}"] = shares[:, i]
    means = df.groupby("algorithm")[[f"share_{m}" for m in MODE_COLUMNS]].mean()
    algos = [a for a in ALGORITHM_ORDER if a in means.index]
    means = means.reindex(algos)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(algos))
    for mode in MODE_COLUMNS:
        ax.bar([ALGORITHM_LABELS[a] for a in algos], means[f"share_{mode}"].values,
               bottom=bottom, label=mode, color=MODE_COLORS[mode], alpha=0.85)
        bottom += means[f"share_{mode}"].values
    ax.set_ylabel("Average mode share")
    ax.set_title("Mean mode share across feasible solutions, by algorithm")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================================
# Section 8 — Comfort surrogate figures
# ============================================================================
def plot_comfort_models(comfort_df: pd.DataFrame, out_path: Path):
    if comfort_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ["r2", "rmse", "mae"]
    titles = ["R² (higher is better)", "RMSE (lower is better)", "MAE (lower is better)"]
    palette = ["#9b9b9b", "#4c72b0", "#dd8452"]
    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in comfort_df.columns:
            ax.set_visible(False); continue
        ax.bar(comfort_df["model_name"], comfort_df[metric], color=palette[: len(comfort_df)])
        ax.set_title(title, fontsize=10); ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(comfort_df[metric].to_numpy()):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Comfort surrogate model comparison", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_comfort_region_errors(output_dir: Path, out_path: Path):
    """Heatmap of R²/RMSE per (model, region). Builds from comfort_region_errors_*.csv."""
    files = sorted(output_dir.glob("comfort_region_errors_*.csv"))
    if not files:
        return
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["model_name"] = f.stem.replace("comfort_region_errors_", "")
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)
    if "region" not in all_df.columns or "rmse" not in all_df.columns:
        return
    pivot_rmse = all_df.pivot(index="region", columns="model_name", values="rmse")
    fig, ax = plt.subplots(figsize=(7, 4 + 0.3 * len(pivot_rmse)))
    im = ax.imshow(pivot_rmse.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot_rmse.columns))); ax.set_xticklabels(pivot_rmse.columns, rotation=20)
    ax.set_yticks(range(len(pivot_rmse.index))); ax.set_yticklabels(pivot_rmse.index)
    for i in range(pivot_rmse.shape[0]):
        for j in range(pivot_rmse.shape[1]):
            v = pivot_rmse.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v > pivot_rmse.values.mean() else "black")
    fig.colorbar(im, ax=ax, label="RMSE")
    ax.set_title("Comfort surrogate RMSE per region of objective space")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ============================================================================
# Main orchestrator
# ============================================================================
def main(output_dir: Path, analytics_dir: Path):
    analytics_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analytics] reading from {output_dir}")
    print(f"[analytics] writing to  {analytics_dir}")

    metrics_by_plan = load_recovered_metrics(output_dir)
    if not metrics_by_plan:
        raise SystemExit("No recovered metrics found. Run pipeline_V6_smart first.")
    profiles_meta = load_profiles_metadata(output_dir)

    # ---------- 1. Stats ----------
    summary = summary_stats_per_plan(metrics_by_plan)
    summary.to_csv(analytics_dir / "summary_stats_per_plan.csv", index=False)
    print(f"  [stats] summary_stats_per_plan.csv ({len(summary)} rows)")

    eff_frames, wtl_frames, dz_frames = [], [], []
    for plan, df in metrics_by_plan.items():
        eff, wtl = paired_stats(df, plan)
        eff_frames.append(eff); wtl_frames.append(wtl)
        dz_frames.append(per_profile_dz(df, plan))
    paired_eff = pd.concat(eff_frames, ignore_index=True) if eff_frames else pd.DataFrame()
    wtl = pd.concat(wtl_frames, ignore_index=True) if wtl_frames else pd.DataFrame()
    dz_all = pd.concat(dz_frames, ignore_index=True) if dz_frames else pd.DataFrame()
    paired_eff.to_csv(analytics_dir / "paired_effect_sizes_per_plan.csv", index=False)
    wtl.to_csv(analytics_dir / "win_tie_loss_per_plan.csv", index=False)
    dz_all.to_csv(analytics_dir / "per_profile_dz_distribution.csv", index=False)
    print(f"  [stats] paired_effect_sizes ({len(paired_eff)}), win_tie_loss ({len(wtl)}), per_profile_dz ({len(dz_all)})")

    ftest = None
    if "extended_benchmark_30profiles" in metrics_by_plan:
        pivot, ftest = friedman_nemenyi(metrics_by_plan["extended_benchmark_30profiles"])
        with open(analytics_dir / "friedman_nemenyi_extended.json", "w", encoding="utf-8") as f:
            json.dump(ftest, f, indent=2, default=str)
        pivot.to_csv(analytics_dir / "friedman_pivot_extended.csv")
        if "average_ranks" in ftest:
            with open(analytics_dir / "cd_diagram_data.json", "w", encoding="utf-8") as f:
                json.dump({
                    "average_ranks": ftest["average_ranks"],
                    "critical_difference_0.05": ftest["critical_difference_0.05"],
                    "n_profiles": ftest["n_profiles"],
                }, f, indent=2)
        print(f"  [stats] friedman_nemenyi_extended.json (chi2={ftest.get('friedman_chi2', 'NA')})")

    strata_all = pd.DataFrame()
    if profiles_meta is not None:
        strata_frames = [strata_breakdown(df, profiles_meta, plan) for plan, df in metrics_by_plan.items()]
        strata_all = pd.concat(strata_frames, ignore_index=True) if strata_frames else pd.DataFrame()
        strata_all.to_csv(analytics_dir / "strata_breakdown_per_plan.csv", index=False)

        # Per-archetype and per-trip aggregates (across plans)
        if not strata_all.empty:
            arche_agg = strata_all.groupby(["archetype", "algorithm"]).agg(
                mean_norm_hv=("mean_norm_hv", "mean"),
                std_norm_hv=("std_norm_hv", "mean"),
                n_runs=("n_runs", "sum"),
            ).reset_index()
            arche_agg.to_csv(analytics_dir / "per_archetype_summary.csv", index=False)
            trip_agg = strata_all.groupby(["trip_distance_bin", "algorithm"]).agg(
                mean_norm_hv=("mean_norm_hv", "mean"),
                std_norm_hv=("std_norm_hv", "mean"),
                n_runs=("n_runs", "sum"),
            ).reset_index()
            trip_agg.to_csv(analytics_dir / "per_trip_distance_summary.csv", index=False)
        print(f"  [stats] strata_breakdown_per_plan.csv + per_archetype + per_trip_distance")

    # ---------- 2. Distribution figures ----------
    plot_box_normalized_hv(metrics_by_plan, analytics_dir / "fig_box_normalized_hv.png")
    plot_violin_per_plan(metrics_by_plan, analytics_dir / "fig_violin_per_plan.png")
    plot_ecdf(metrics_by_plan, analytics_dir / "fig_ecdf_normalized_hv.png")
    plot_strip_per_plan(metrics_by_plan, analytics_dir / "fig_strip_per_plan.png")
    print(f"  [fig] distributions: box + violin + ecdf + strip")

    if profiles_meta is not None and "main_nsga2_vs_nsga3_150profiles" in metrics_by_plan:
        plot_box_per_archetype(
            metrics_by_plan["main_nsga2_vs_nsga3_150profiles"], profiles_meta,
            analytics_dir / "fig_box_per_archetype.png", "main_nsga2_vs_nsga3_150profiles",
        )
        plot_box_per_trip_distance(
            metrics_by_plan["main_nsga2_vs_nsga3_150profiles"], profiles_meta,
            analytics_dir / "fig_box_per_trip_distance.png", "main_nsga2_vs_nsga3_150profiles",
        )
        print(f"  [fig] per-archetype + per-trip_distance box plots")

    # ---------- 3. Pairwise comparison figures ----------
    if "main_nsga2_vs_nsga3_150profiles" in metrics_by_plan:
        plot_scatter_pair(metrics_by_plan["main_nsga2_vs_nsga3_150profiles"], "nsga2", "nsga3",
                          analytics_dir / "fig_scatter_nsga2_vs_nsga3.png")
        plot_caterpillar(metrics_by_plan["main_nsga2_vs_nsga3_150profiles"], "nsga2", "nsga3",
                         analytics_dir / "fig_caterpillar_nsga3_minus_nsga2.png",
                         "main 150-profile plan")
        print(f"  [fig] NSGA-II vs NSGA-III scatter + caterpillar")

    if "extended_benchmark_30profiles" in metrics_by_plan:
        plot_scatter_matrix_all_pairs(
            metrics_by_plan["extended_benchmark_30profiles"],
            analytics_dir / "fig_scatter_matrix_all_pairs.png",
        )
        print(f"  [fig] scatter matrix (all algo pairs)")

    if not dz_all.empty:
        dz_main = dz_all[dz_all["plan"] == "main_nsga2_vs_nsga3_150profiles"]
        if not dz_main.empty:
            plot_dz_distribution(dz_main, analytics_dir / "fig_dz_distribution.png")
            print(f"  [fig] d_z distribution (main plan)")

    # ---------- 4. Convergence figures ----------
    repr_dir = output_dir / "representative_curves_10profiles"
    hist_repr = load_history_sample(repr_dir, max_files=2000)
    if hist_repr is not None:
        plot_history_metric(hist_repr, "hypervolume",
                            analytics_dir / "fig_hv_convergence.png",
                            "Hypervolume",
                            "HV vs generation — representative_curves_10profiles (mean ± std)")
        plot_history_metric(hist_repr, "spacing",
                            analytics_dir / "fig_spacing_convergence.png",
                            "Spacing (lower is better)",
                            "Spacing vs generation — representative plan")
        plot_history_metric(hist_repr, "feasible_ratio",
                            analytics_dir / "fig_feasible_ratio_convergence.png",
                            "Feasible ratio",
                            "Feasible solutions ratio vs generation")
        plot_hv_per_profile_grid(hist_repr, analytics_dir / "fig_hv_convergence_per_profile.png", max_profiles=10)
        print(f"  [fig] convergence: HV + spacing + feasibility + per-profile grid")
    else:
        print(f"  [skip] no history checkpoints in {repr_dir}")

    # ---------- 5. Benchmark figures ----------
    if ftest is not None and "average_ranks" in ftest:
        plot_cd_diagram(ftest, analytics_dir / "fig_cd_diagram.png")
        print(f"  [fig] Critical Difference diagram")

    # ---------- 6. Strata figures ----------
    if not strata_all.empty:
        sb_main = strata_all[strata_all["plan"] == "main_nsga2_vs_nsga3_150profiles"]
        if not sb_main.empty:
            plot_heatmap_strata(sb_main, analytics_dir / "fig_heatmap_strata.png",
                                title="Mean normalized HV by archetype × trip_distance — main plan")
        plot_bar_per_strata(strata_all, "archetype",
                            analytics_dir / "fig_bar_per_archetype.png",
                            "Mean normalized HV per archetype (across plans)")
        plot_bar_per_strata(strata_all, "trip_distance_bin",
                            analytics_dir / "fig_bar_per_trip_distance.png",
                            "Mean normalized HV per trip-distance bin (across plans)")
        print(f"  [fig] strata heatmap + bar per archetype + bar per trip_distance")

    # ---------- 7. Solution-space figures (from extended populations) ----------
    plan_dir = output_dir / "extended_benchmark_30profiles"
    pop_sample = load_population_sample(plan_dir, max_files=600)
    if pop_sample is not None:
        plot_pareto_fronts_2d(pop_sample, analytics_dir / "fig_pareto_fronts_2d.png", n_profiles=4)
        plot_pareto_3d(pop_sample, analytics_dir / "fig_pareto_3d.png", n_profiles=3)
        plot_pca_solutions(pop_sample, analytics_dir / "fig_pca_solutions.png")
        corr = plot_objective_correlations(pop_sample, analytics_dir / "fig_objective_correlations.png")
        if corr is not None:
            corr.to_csv(analytics_dir / "objective_correlations.csv")
        plot_parallel_coordinates(pop_sample, analytics_dir / "fig_parallel_coordinates.png")
        plot_mode_share_per_algorithm(pop_sample, analytics_dir / "fig_mode_share_per_algorithm.png")
        rt = plot_runtime_comparison(pop_sample, analytics_dir / "fig_runtime_comparison.png")
        if rt is not None:
            rt.to_csv(analytics_dir / "per_algorithm_runtime_summary.csv", index=False)
        print(f"  [fig] solution-space: pareto2D + pareto3D + PCA + correlations + parallel-coords + mode-share + runtime")

        # Intrinsic dimensionality
        feas = pop_sample.copy()
        if "feasible_bool" in feas.columns:
            feas = feas[feas["feasible_bool"]]
        feas = feas.dropna(subset=OBJECTIVE_COLUMNS)
        if len(feas) > 200:
            X = feas[OBJECTIVE_COLUMNS].to_numpy()
            Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
            n_use = min(2500, len(Xs))
            idx = np.random.default_rng(42).choice(len(Xs), size=n_use, replace=False)
            d = estimate_intrinsic_dim_twonn(Xs[idx])
            with open(analytics_dir / "intrinsic_dimensionality.json", "w", encoding="utf-8") as f:
                json.dump({
                    "n_objectives_nominal": 4,
                    "twonn_intrinsic_dim_estimate": d,
                    "n_points_used": int(n_use),
                    "interpretation": (
                        f"Effective dimensionality estimated at {d:.2f} via the TwoNN "
                        "estimator of Facco et al. 2017. A value substantially below 4 "
                        "indicates strong inter-objective correlations or a constraint-induced "
                        "low-dimensional manifold in objective space — consistent with the "
                        "geometric explanation for NSGA-III's reference-direction efficiency."
                    ),
                }, f, indent=2)
            print(f"  [stats] intrinsic_dimensionality.json (TwoNN ≈ {d:.3f})")
    else:
        print(f"  [skip] no population checkpoints in {plan_dir}")

    # ---------- 8. Comfort surrogate figures ----------
    comfort_path = output_dir / "comfort_model_comparison.csv"
    if comfort_path.exists():
        comfort_df = pd.read_csv(comfort_path)
        plot_comfort_models(comfort_df, analytics_dir / "fig_comfort_model_comparison.png")
        print(f"  [fig] comfort model comparison")
    plot_comfort_region_errors(output_dir, analytics_dir / "fig_comfort_region_errors.png")

    # Manifest
    manifest = sorted([p.name for p in analytics_dir.iterdir() if p.is_file()])
    with open(analytics_dir / "manifest.txt", "w", encoding="utf-8") as f:
        for name in manifest:
            f.write(name + "\n")
    print(f"\n[analytics] Done. {len(manifest)} artifacts in {analytics_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extended analytics for V6 outputs (~30 figures, ~12 tables)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", default="outputs_v5_parallel_3threads_fixed",
                        help="Directory containing V6 outputs.")
    parser.add_argument("--analytics_dir", default=None,
                        help="Output directory for tables and figures (default: <output_dir>/analytics).")
    args = parser.parse_args()
    out_path = Path(args.output_dir)
    an_path = Path(args.analytics_dir) if args.analytics_dir else out_path / "analytics"
    main(out_path, an_path)
