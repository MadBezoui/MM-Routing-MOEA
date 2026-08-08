"""intrinsic_dimensionality.py
===============================
Effective dimensionality of the sampled feasible objective archive (Section 6.4).

Two estimators are computed on the same samples, because they disagree and the
disagreement is itself informative:

``TwoNN`` (Facco et al., 2017)
    Uses only the ratio :math:`\\mu = r_2 / r_1` of the two nearest-neighbour
    distances.  The maximum-likelihood estimate is obtained from the linear fit
    of :math:`-\\log(1 - F(\\mu))` against :math:`\\log\\mu` through the origin.

``Levina-Bickel`` maximum likelihood
    Averages the local estimator over the ``k`` nearest neighbours.

Both are archive-level diagnostics.  They characterise the geometry the
algorithms actually explored; they are **not** an estimate of the intrinsic
dimensionality of the full feasible decision space, nor a statement about the
true Pareto front.  The estimate depends on archive construction, on the
surrogate and on the generation budget, which is why per-algorithm estimates
are reported alongside the pooled one.

Usage
-----
    python -m experiments.intrinsic_dimensionality --runs results/outputs_main
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments._common import setup_logging, write_json

logger = logging.getLogger(__name__)

OBJECTIVE_COLUMNS = ["obj_1", "obj_2", "obj_3", "obj_4"]


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def twonn(points: np.ndarray, retained_fraction: float = 0.95) -> Dict[str, float]:
    """TwoNN estimator of Facco et al. (2017).

    The largest (1 - retained_fraction) of the ratios is dropped before the fit,
    as recommended by the authors, to limit the influence of the tail where the
    local-density assumption breaks down.
    """
    X = np.unique(np.asarray(points, dtype=float), axis=0)
    n = len(X)
    if n < 10:
        return {"id": float("nan")}

    from scipy.spatial import cKDTree

    dist, _ = cKDTree(X).query(X, k=3)
    r1, r2 = dist[:, 1], dist[:, 2]
    valid = (r1 > 0) & (r2 > 0)
    
    invalid_or_dup = int((~valid).sum())
    
    mu = np.sort(r2[valid] / r1[valid])
    mu = mu[mu > 1.0]
    if len(mu) < 10:
        return {"id": float("nan")}

    N = len(mu)
    keep = max(2, int(np.floor(retained_fraction * N)))
    mu_trimmed = mu[:keep]
    
    ranks = np.arange(1, keep + 1, dtype=float)
    f_emp = (ranks - 0.5) / N
    
    x = np.log(mu_trimmed)
    y = -np.log1p(-f_emp)
    
    # least squares through the origin
    id_est = float(np.dot(x, y) / np.dot(x, x))
    
    # Calculate R2
    y_pred = id_est * x
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "id": id_est,
        "n_original": float(n),
        "n_retained": float(keep),
        "invalid_or_dup": float(invalid_or_dup),
        "r2": float(r2)
    }


def levina_bickel(points: np.ndarray, k: int = 10) -> float:
    """Levina-Bickel maximum-likelihood estimator averaged over ``k`` neighbours."""
    X = np.unique(np.asarray(points, dtype=float), axis=0)
    n = len(X)
    if n < k + 2:
        return float("nan")

    from scipy.spatial import cKDTree

    dist, _ = cKDTree(X).query(X, k=k + 1)
    dist = np.clip(dist[:, 1:], 1e-12, None)
    log_ratio = np.log(dist[:, [-1]]) - np.log(dist[:, :-1])
    inv = log_ratio.sum(axis=1) / (k - 1)
    inv = inv[np.isfinite(inv) & (inv > 0)]
    return float(1.0 / np.mean(inv)) if len(inv) else float("nan")


# --------------------------------------------------------------------------
# Archive assembly
# --------------------------------------------------------------------------

def load_archive(run_dirs: Sequence[Path], max_files_per_dir: int = 4000) -> pd.DataFrame:
    """Pool the feasible final populations of every checkpoint found."""
    frames: List[pd.DataFrame] = []
    for root in run_dirs:
        for ckpt in sorted(Path(root).rglob("checkpoints/population")):
            files = sorted(ckpt.glob("*.csv"))[:max_files_per_dir]
            for f in files:
                try:
                    df = pd.read_csv(f)
                except Exception:  # pragma: no cover
                    continue
                if not set(OBJECTIVE_COLUMNS).issubset(df.columns):
                    continue
                if "feasible" in df.columns:
                    flag = df["feasible"]
                    keep = flag if flag.dtype == bool else (
                        flag.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                    )
                    df = df[keep]
                frames.append(df[OBJECTIVE_COLUMNS + ["algorithm"]])
    if not frames:
        raise FileNotFoundError(f"no population checkpoints under {list(run_dirs)}")
    return pd.concat(frames, ignore_index=True).dropna()


def _standardise(F: np.ndarray) -> np.ndarray:
    """Unit-variance scaling, so no objective dominates by its numeric range."""
    std = F.std(axis=0)
    std[std == 0] = 1.0
    return (F - F.mean(axis=0)) / std


def estimate(
    archive: pd.DataFrame,
    n_points: int = 2500,
    k: int = 10,
    random_state: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_state)

    def sample(frame: pd.DataFrame) -> np.ndarray:
        F = frame[OBJECTIVE_COLUMNS].to_numpy(dtype=float)
        if len(F) > n_points:
            F = F[rng.choice(len(F), size=n_points, replace=False)]
        return _standardise(F)

    pooled = sample(archive)
    report: Dict[str, object] = {
        "n_objectives_nominal": len(OBJECTIVE_COLUMNS),
        "n_points_pooled": int(len(pooled)),
        "twonn_pooled": twonn(pooled),
        "levina_bickel_pooled": levina_bickel(pooled, k=k),
        "levina_bickel_k": int(k),
        "per_algorithm": {},
    }

    for algorithm, group in archive.groupby("algorithm"):
        pts = sample(group)
        report["per_algorithm"][str(algorithm)] = {
            "n_points": int(len(pts)),
            "twonn": twonn(pts),
            "levina_bickel": levina_bickel(pts, k=k),
        }

    per_algo = report["per_algorithm"]
    if per_algo:
        tw = [v["twonn"]["id"] for v in per_algo.values() if np.isfinite(v["twonn"]["id"])]
        lb = [v["levina_bickel"] for v in per_algo.values() if np.isfinite(v["levina_bickel"])]
        report["cross_algorithm_spread_twonn"] = float(max(tw) - min(tw)) if tw else float("nan")
        report["cross_algorithm_spread_levina_bickel"] = float(max(lb) - min(lb)) if lb else float("nan")

    report["interpretation"] = (
        "Archive-level diagnostic. An estimate well below the nominal four "
        "indicates strong inter-objective correlation or a constraint-induced "
        "low-dimensional manifold in the region the algorithms explored. It is "
        "not an estimate for the full feasible decision space and should be "
        "confirmed with algorithm-independent feasible-path sampling."
    )
    return report


def objective_correlations(archive: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson and Spearman correlations between the four objectives (Table 12).

    Both are returned because Table 12 reports Pearson while claiming the two
    agree; the maximum absolute difference is attached to the Pearson frame so
    that the claim can be checked rather than assumed.
    """
    F = archive[OBJECTIVE_COLUMNS]
    pearson = F.corr(method="pearson")
    spearman = F.corr(method="spearman")
    pearson.attrs["max_abs_difference_to_spearman"] = float(
        (pearson - spearman).abs().to_numpy().max()
    )
    return pearson, spearman


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["results/outputs_main"],
                        help="directories containing checkpoints/population")
    parser.add_argument("--out", default="results/analytics")
    parser.add_argument("--n-points", type=int, default=2500)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    setup_logging()
    archive = load_archive([Path(r) for r in args.runs])
    logger.info("Pooled archive: %d feasible objective vectors", len(archive))

    report = estimate(archive, n_points=args.n_points, k=args.k)
    out = Path(args.out)
    write_json(out / "intrinsic_dimensionality.json", report)

    pearson, spearman = objective_correlations(archive)
    pearson.to_csv(out / "objective_correlations.csv")
    spearman.to_csv(out / "objective_correlations_spearman.csv")
    logger.info("max |Pearson - Spearman| = %.4f",
                pearson.attrs["max_abs_difference_to_spearman"])

    print(pd.Series({
        "TwoNN (pooled)": report["twonn_pooled"],
        "Levina-Bickel (pooled)": report["levina_bickel_pooled"],
    }).round(3).to_string())
    print()
    print(pd.DataFrame(report["per_algorithm"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
