"""stabilization.py
===================
Stabilization of the elicited priority weights (Section 4.2).

Pairwise elicitation routinely assigns a near-zero weight to one objective --
in this sample, emissions, at :math:`8.9\\times 10^{-5}`.  Anchoring reference
directions on such a vector silently reduces a four-objective search to a
three-objective one.  Eq. 6 blends the elicited vector towards the uniform one
and imposes a per-component floor:

.. math::
    \\mathbf{w}_{stab} = \\Pi\\left(\\max\\{(1-\\beta)\\,
    \\tilde{\\mathbf{w}}_{raw} + \\beta\\mathbf{u},\\ \\varphi\\mathbf{1}\\}\\right)

The floor is applied **before** the final simplex projection :math:`\\Pi`, so a
component set to exactly :math:`\\varphi` is reduced slightly by the
renormalisation.

Eq. 7 defines the admissible set: a pair :math:`(\\beta, \\varphi)` is
admissible when the floor corrects *only* the single near-degenerate objective
:math:`j^\\star = \\arg\\min_j w_j` while remaining inactive on all others.
:math:`j^\\star` is read off the raw elicited weights before any search, so the
calibration is not circular.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_STABILIZATION, StabilizationConfig

logger = logging.getLogger(__name__)

OBJECTIVES: Tuple[str, ...] = ("time", "cost", "emissions", "comfort")


@dataclass
class WeightAudit:
    """Result of :func:`audit_and_stabilize_weights`."""

    raw_weights: Dict[str, float]
    stabilized_weights: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    beta: float = DEFAULT_STABILIZATION.blend_uniform
    phi: float = DEFAULT_STABILIZATION.floor
    degenerate_objective: str = ""
    renormalisation_delta: float = 0.0

    def as_vector(self, stabilized: bool = True) -> List[float]:
        source = self.stabilized_weights if stabilized else self.raw_weights
        return [float(source[k]) for k in OBJECTIVES]


# --------------------------------------------------------------------------
# Eq. 6
# --------------------------------------------------------------------------

def blended_weights(raw: Sequence[float], beta: float) -> np.ndarray:
    """:math:`\\tilde{w}_j(\\beta) = (1-\\beta) w_j + \\beta / M`, before the floor."""
    w = np.asarray(raw, dtype=float)
    w = w / w.sum()
    return (1.0 - beta) * w + beta / len(w)


def stabilize(raw: Sequence[float], beta: float, phi: float) -> np.ndarray:
    """Apply Eq. 6 to a raw weight vector."""
    blended = blended_weights(raw, beta)
    floored = np.maximum(blended, phi)
    return floored / floored.sum()


def audit_and_stabilize_weights(
    raw_weights: Dict[str, float],
    floor: float = DEFAULT_STABILIZATION.floor,
    blend_uniform: float = DEFAULT_STABILIZATION.blend_uniform,
) -> WeightAudit:
    """Normalise, audit and stabilize the elicited weight vector."""
    raw = {k: float(raw_weights.get(k, 0.0)) for k in OBJECTIVES}
    total = sum(raw.values())
    warnings: List[str] = []

    if total <= 0:
        warnings.append("All raw objective weights are non-positive; falling back to uniform.")
        raw = {k: 1.0 / len(OBJECTIVES) for k in OBJECTIVES}
    else:
        raw = {k: v / total for k, v in raw.items()}

    degenerate = min(raw, key=raw.get)
    if raw[degenerate] < 0.02:
        warnings.append(
            f"Objective '{degenerate}' receives a near-zero elicited weight "
            f"({raw[degenerate]:.2e}); the floor will correct it."
        )
    if raw["comfort"] < 0.20:
        warnings.append("Comfort weight is unexpectedly low.")
    if max(raw.values()) > 0.75:
        warnings.append("One objective dominates; stabilization will reduce its grip on the reference set.")

    vector = stabilize([raw[k] for k in OBJECTIVES], blend_uniform, floor)
    stabilized = {k: float(v) for k, v in zip(OBJECTIVES, vector)}

    # Reduction of the floored component caused by the final renormalisation.
    delta = 0.0
    if stabilized[degenerate] > 0:
        delta = float((floor - stabilized[degenerate]) / floor)

    return WeightAudit(
        raw_weights=raw,
        stabilized_weights=stabilized,
        warnings=warnings,
        beta=float(blend_uniform),
        phi=float(floor),
        degenerate_objective=degenerate,
        renormalisation_delta=delta,
    )


# --------------------------------------------------------------------------
# Eq. 7 - admissible calibration set
# --------------------------------------------------------------------------

def admissible_pairs(
    raw_weights: Dict[str, float],
    cfg: StabilizationConfig = DEFAULT_STABILIZATION,
) -> pd.DataFrame:
    """Enumerate the calibration grid and mark the admissible cells (Eq. 7).

    A pair is admissible when

    .. math::
        \\max(\\tilde{w}_{j^\\star}(\\beta), \\varphi_{min}) \\le \\varphi
        \\le \\min_{j \\ne j^\\star} \\tilde{w}_j(\\beta).

    The ``separation`` column reports
    :math:`\\min_{j\\ne j^\\star} \\tilde{w}_j(\\beta) - \\varphi`, the margin by
    which the floor stays inactive on the non-degenerate objectives.
    """
    raw = np.asarray([raw_weights[k] for k in OBJECTIVES], dtype=float)
    raw = raw / raw.sum()
    j_star = int(np.argmin(raw))

    rows: List[Dict[str, object]] = []
    for beta in cfg.beta_grid:
        blended = blended_weights(raw, beta)
        others = np.delete(blended, j_star)
        lower = max(blended[j_star], cfg.floor_min)
        upper = float(others.min())
        for phi in cfg.phi_grid:
            rows.append({
                "beta": float(beta),
                "phi": float(phi),
                "degenerate_objective": OBJECTIVES[j_star],
                "blended_degenerate_weight": float(blended[j_star]),
                "min_other_blended_weight": upper,
                "admissible": bool(lower <= phi <= upper),
                "separation": float(upper - phi),
                "floor_binds_on_n_objectives": int((blended < phi).sum()),
            })
    return pd.DataFrame(rows)


def select_calibration(
    raw_weights: Dict[str, float],
    cfg: StabilizationConfig = DEFAULT_STABILIZATION,
) -> Tuple[float, float, pd.DataFrame]:
    """Return ``(beta, phi)`` chosen inside the admissible set, plus the grid.

    Within the admissible set the floor is set to its smallest admissible value
    and the blend intensity is the one that leaves a workable separation
    between the floor and the smallest non-degenerate blended weight.
    """
    grid = admissible_pairs(raw_weights, cfg)
    admissible = grid[grid["admissible"] & (grid["phi"] >= cfg.floor_min)]
    if admissible.empty:
        logger.warning("No admissible (beta, phi) cell; falling back to the configured defaults.")
        return cfg.blend_uniform, cfg.floor, grid

    phi_star = float(admissible["phi"].min())
    candidates = admissible[np.isclose(admissible["phi"], phi_star)]
    beta_star = float(candidates.loc[candidates["separation"].idxmax(), "beta"])
    return beta_star, phi_star, grid


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_stabilization(raw_weights_path: str, out_dir: str) -> WeightAudit:
    """Stabilize a raw weight file and write the Table 3 artefacts."""
    raw_path = Path(raw_weights_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8") as fh:
        raw_weights = json.load(fh)

    audit = audit_and_stabilize_weights(raw_weights)
    grid = admissible_pairs(audit.raw_weights)

    with open(out / "objective_weights_stabilized.json", "w", encoding="utf-8") as fh:
        json.dump(audit.stabilized_weights, fh, indent=2)

    pd.DataFrame([
        {"objective": k,
         "raw_weight": audit.raw_weights[k],
         "stabilized_weight": audit.stabilized_weights[k]}
        for k in OBJECTIVES
    ]).to_csv(out / "table3_stabilized_weights.csv", index=False)

    grid.to_csv(out / "stabilization_admissible_grid.csv", index=False)

    with open(out / "stabilization_audit.json", "w", encoding="utf-8") as fh:
        json.dump({
            "beta": audit.beta,
            "phi": audit.phi,
            "degenerate_objective": audit.degenerate_objective,
            "renormalisation_delta": audit.renormalisation_delta,
            "warnings": audit.warnings,
        }, fh, indent=2)

    for message in audit.warnings:
        logger.warning(" - %s", message)
    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stabilize elicited priority weights (Eq. 6-7).")
    parser.add_argument("--survey-dir", default="data/survey_results")
    parser.add_argument("--out", default="results/preferences")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    from src.survey_data_loader import compute_objective_weights

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = compute_objective_weights(args.survey_dir)
    raw_path = out / "objective_weights_raw.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)

    audit = run_stabilization(str(raw_path), str(out))
    print(json.dumps({
        "raw": audit.raw_weights,
        "stabilized": audit.stabilized_weights,
        "beta": audit.beta, "phi": audit.phi,
    }, indent=2))


if __name__ == "__main__":
    main()
