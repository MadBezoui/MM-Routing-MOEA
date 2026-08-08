"""statistics.py
=================
The statistical evaluation protocol of Section 4.5.

The unit of confirmatory inference is the **profile**: seed-level paired
differences are averaged within each profile before any test is applied, since
runs sharing a profile are clustered.  Seed-level statistics are computed too,
but only ever reported as descriptive summaries.

Provides
--------
``paired_differences``        seed-level and profile-level paired differences
``cohens_dz``                 paired effect size (Eq. 14)
``wilcoxon_report``           Wilcoxon signed-rank with r = |Z| / sqrt(n)
``holm_correction``           step-down family-wise error control
``stratified_bootstrap_ci``   profile-stratified percentile bootstrap
``friedman_nemenyi``          omnibus rank test with the Critical Difference
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

#: Studentised range statistic q_alpha / sqrt(2) for the Nemenyi test at
#: alpha = 0.05, indexed by the number of algorithms compared.
NEMENYI_Q05: Dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


# --------------------------------------------------------------------------
# Paired differences
# --------------------------------------------------------------------------

def paired_differences(
    metrics: pd.DataFrame,
    algo_a: str,
    algo_b: str,
    value_col: str = "normalized_hv",
    profile_col: str = "profile_id",
    seed_col: str = "seed",
) -> Tuple[pd.Series, pd.Series]:
    """Return ``(seed_level, profile_level)`` differences ``a - b``.

    Runs are aligned by ``(profile, seed)``; profile-level values are the mean
    of the seed-level differences within each profile.
    """
    pivot = metrics.pivot_table(
        index=[profile_col, seed_col], columns="algorithm", values=value_col,
    )
    if algo_a not in pivot.columns or algo_b not in pivot.columns:
        raise KeyError(f"missing algorithm column: {algo_a} or {algo_b}")
    seed_level = (pivot[algo_a] - pivot[algo_b]).dropna()
    profile_level = seed_level.groupby(level=0).mean()
    return seed_level, profile_level


def cohens_dz(differences: Sequence[float]) -> float:
    """Paired Cohen's :math:`d_z` (Eq. 14): mean difference over its own SD."""
    d = np.asarray(differences, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def wilcoxon_report(differences: Sequence[float], zero_method: str = "wilcox") -> Dict[str, float]:
    """One-sample Wilcoxon signed-rank test with its rank-biserial effect size.

    ``r = |Z| / sqrt(n)`` carries the sign of the mean difference, so that a
    negative ``r`` favours the same algorithm as a negative ``d_z``.
    """
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if n < 3 or np.allclose(d, 0):
        return {"n": n, "statistic": float("nan"), "p_value": float("nan"),
                "z": float("nan"), "r": float("nan"),
                "hodges_lehmann": float(np.median(d)) if n else float("nan")}

    result = stats.wilcoxon(d, zero_method=zero_method, alternative="two-sided")
    z = float(stats.norm.isf(result.pvalue / 2.0))
    walsh = (d[:, None] + d[None, :]) / 2.0
    hl = float(np.median(walsh[np.triu_indices(n)]))
    return {
        "n": n,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "z": z,
        "r": float(np.sign(d.mean()) * abs(z) / np.sqrt(n)),
        "hodges_lehmann": hl,
    }


def compare(
    metrics: pd.DataFrame,
    algo_a: str,
    algo_b: str,
    value_col: str = "normalized_hv",
) -> Dict[str, object]:
    """Full paired comparison of two algorithms under the Section 4.5 protocol."""
    seed_level, profile_level = paired_differences(metrics, algo_a, algo_b, value_col)
    confirmatory = wilcoxon_report(profile_level.to_numpy())
    descriptive = wilcoxon_report(seed_level.to_numpy())
    return {
        "algo_a": algo_a,
        "algo_b": algo_b,
        "n_paired_runs": int(len(seed_level)),
        "n_profiles": int(len(profile_level)),
        "mean_diff": float(seed_level.mean()),
        "sd_diff": float(seed_level.std(ddof=1)),
        "dz_run_level_descriptive": cohens_dz(seed_level),
        "dz_profile_level_confirmatory": cohens_dz(profile_level),
        "wilcoxon_profile_statistic": confirmatory["statistic"],
        "wilcoxon_profile_p": confirmatory["p_value"],
        "wilcoxon_profile_r": confirmatory["r"],
        "hodges_lehmann_profile": confirmatory["hodges_lehmann"],
        "wilcoxon_seed_p_descriptive": descriptive["p_value"],
        "run_win_rate_b": float((seed_level > 0).mean()),
        "profile_wins_b": int((profile_level > 0).sum()),
        "profile_wins_a": int((profile_level < 0).sum()),
        "pct_2_5": float(np.percentile(seed_level, 2.5)),
        "pct_97_5": float(np.percentile(seed_level, 97.5)),
    }


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni step-down correction.

    Returns the adjusted p-values and the rejection decisions, in the input
    order.  Adjusted values are made monotone, as the procedure requires.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return pd.DataFrame({
        "p_value": p,
        "p_holm": adjusted,
        "reject": adjusted < alpha,
    })


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def stratified_bootstrap_ci(
    values: pd.Series,
    strata: pd.Series,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
    statistic=np.mean,
) -> Dict[str, float]:
    """Percentile bootstrap CI that resamples **whole strata**.

    Runs sharing a profile are clustered, so resampling individual runs would
    understate the variance.  Profiles are resampled with replacement and all
    their runs are carried along.
    """
    rng = np.random.default_rng(random_state)
    values = pd.Series(values).reset_index(drop=True)
    strata = pd.Series(strata).reset_index(drop=True)
    groups = [values[strata == s].to_numpy() for s in strata.unique()]
    n_groups = len(groups)

    draws = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        picks = rng.integers(0, n_groups, size=n_groups)
        draws[b] = statistic(np.concatenate([groups[i] for i in picks]))

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(statistic(values.to_numpy())),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": int(n_resamples),
        "n_strata": int(n_groups),
    }


# --------------------------------------------------------------------------
# Omnibus
# --------------------------------------------------------------------------

def friedman_nemenyi(
    metrics: pd.DataFrame,
    value_col: str = "normalized_hv",
    profile_col: str = "profile_id",
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Friedman omnibus test, Nemenyi post-hoc and the Critical Difference.

    Ranks are computed on the per-profile mean of ``value_col``, so that every
    profile contributes one observation per algorithm.
    """
    pivot = metrics.pivot_table(index=profile_col, columns="algorithm",
                                values=value_col, aggfunc="mean").dropna()
    algorithms = list(pivot.columns)
    k, n = len(algorithms), len(pivot)
    if k < 2 or n < 2:
        raise ValueError("Friedman test needs at least two algorithms and two profiles")

    # higher hypervolume is better -> rank 1 to the largest value
    ranks = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0)

    chi2, p_value = stats.friedmanchisquare(*[pivot[a].to_numpy() for a in algorithms])
    cd = NEMENYI_Q05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6.0 * n))

    rows: List[Dict[str, object]] = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(abs(mean_ranks.iloc[i] - mean_ranks.iloc[j]))
            rows.append({
                "algo_a": algorithms[i], "algo_b": algorithms[j],
                "rank_diff": diff, "critical_difference": float(cd),
                "significant": bool(diff > cd),
            })

    report = {
        "n_profiles": int(n),
        "n_algorithms": int(k),
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_value),
        "average_ranks": {a: float(mean_ranks[a]) for a in algorithms},
        f"critical_difference_{alpha}": float(cd),
        "pairwise": rows,
    }
    return pivot, report


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

def per_profile_table(
    metrics: pd.DataFrame,
    algo_a: str,
    algo_b: str,
    value_col: str = "normalized_hv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-profile paired statistics with Holm correction across profiles."""
    seed_level, _ = paired_differences(metrics, algo_a, algo_b, value_col)
    rows: List[Dict[str, object]] = []
    for profile, diffs in seed_level.groupby(level=0):
        report = wilcoxon_report(diffs.to_numpy())
        rows.append({
            "profile_id": profile,
            "n_seeds": int(len(diffs)),
            "mean_diff": float(diffs.mean()),
            "cohen_dz": cohens_dz(diffs),
            "wilcoxon_p": report["p_value"],
        })
    table = pd.DataFrame(rows)
    corrected = holm_correction(table["wilcoxon_p"].fillna(1.0).to_numpy(), alpha=alpha)
    table["p_holm"] = corrected["p_holm"]
    table["significant_holm"] = corrected["reject"]
    return table.sort_values("mean_diff").reset_index(drop=True)
