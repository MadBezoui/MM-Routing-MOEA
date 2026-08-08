"""reference_directions.py
==========================
Construction of the reference-direction sets of Section 4.3.

Two sets are distinguished, and the distinction is what the four-way ablation
of Section 6.6 rests on:

``canonical_reference_directions``
    the plain Das-Dennis simplex lattice with :math:`p` divisions,
    :math:`n_{ref} = \\binom{M+p-1}{p}` (Eq. 8).  This is the reference set of
    *canonical NSGA-III*: no anchors, uniform objective weights.

``priority_informed_reference_directions``
    the canonical lattice augmented with the :math:`M+1` anchors of Eq. 9 --
    the priority vector :math:`\\mathbf{w}` itself and the :math:`M`
    directional anchors :math:`\\Pi((1-\\rho)\\mathbf{w} + \\rho\\mathbf{e}_i)`.
    Near-duplicate directions are removed after rounding.

The anchor-spread parameter :math:`\\rho` is an explicit argument so that the
sensitivity sweep of Table 13 can be run without editing the source.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

from src.config import DEFAULT_REFDIRS

logger = logging.getLogger(__name__)


def canonical_reference_directions(n_obj: int, n_partitions: int) -> np.ndarray:
    """Das-Dennis lattice of Eq. 8: ``C(M + p - 1, p)`` directions."""
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)


def anchor_directions(
    priority_weights: Sequence[float],
    rho: float = DEFAULT_REFDIRS.rho,
) -> np.ndarray:
    """The ``M + 1`` anchors of Eq. 9.

    Returns ``w`` followed by ``Pi((1 - rho) w + rho e_i)`` for
    ``i = 1..M``, every row projected back onto the unit simplex.
    """
    w = np.asarray(priority_weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("priority_weights must be one-dimensional")
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must lie in (0, 1); got {rho}")

    w = np.clip(w, 1e-12, None)
    w = w / w.sum()

    rows = [w]
    for e in np.eye(len(w)):
        rows.append((1.0 - rho) * w + rho * e)
    anchors = np.asarray(rows, dtype=float)
    return anchors / anchors.sum(axis=1, keepdims=True)


def priority_informed_reference_directions(
    n_obj: int,
    n_partitions: int,
    priority_weights: Optional[Sequence[float]],
    rho: float = DEFAULT_REFDIRS.rho,
    dedup_decimals: int = DEFAULT_REFDIRS.dedup_decimals,
) -> np.ndarray:
    """Canonical lattice augmented with the priority-informed anchors."""
    base = canonical_reference_directions(n_obj, n_partitions)
    if priority_weights is None:
        return base

    w = np.asarray(priority_weights, dtype=float)
    if w.ndim != 1 or len(w) != n_obj:
        logger.warning(
            "priority_weights has shape %s for %d objectives; falling back to "
            "the canonical lattice.", np.shape(w), n_obj,
        )
        return base

    merged = np.vstack([base, anchor_directions(w, rho=rho)])
    # Near-duplicate directions are removed after rounding (Section 4.3).
    _, keep = np.unique(np.round(merged, dedup_decimals), axis=0, return_index=True)
    return merged[np.sort(keep)]


def build_reference_directions(
    algorithm: str,
    n_obj: int,
    n_partitions: int,
    stabilized_weights: Optional[Sequence[float]] = None,
    raw_weights: Optional[Sequence[float]] = None,
    rho: float = DEFAULT_REFDIRS.rho,
) -> np.ndarray:
    """Return the reference set appropriate for ``algorithm``.

    ==========================  ==================================================
    ``canonical_nsga3``         plain Das-Dennis lattice, uniform weights
    ``pi_nsga3`` / ``_stab``    lattice + anchors from the *stabilized* weights
    ``pi_nsga3_raw``            lattice + anchors from the *raw* elicited weights
    ``moead``                   lattice + anchors from the stabilized weights
    ``nsga2`` / ``smsemoa``     lattice (unused: selection is weight-agnostic)
    ==========================  ==================================================
    """
    from src.config import canonical_algorithm

    algo = canonical_algorithm(algorithm)

    if algo in ("canonical_nsga3", "nsga2", "smsemoa"):
        return canonical_reference_directions(n_obj, n_partitions)

    if algo == "pi_nsga3_raw":
        weights = raw_weights
        if weights is None:
            raise ValueError("pi_nsga3_raw requires the raw elicited weights")
    else:
        weights = stabilized_weights

    return priority_informed_reference_directions(
        n_obj, n_partitions, weights, rho=rho,
    )


def audit_reference_directions(
    n_obj: int,
    n_partitions: int,
    stabilized_weights: Sequence[float],
    rho: float = DEFAULT_REFDIRS.rho,
) -> Dict[str, object]:
    """Report the cardinalities quoted in Section 4.3."""
    base = canonical_reference_directions(n_obj, n_partitions)
    full = priority_informed_reference_directions(n_obj, n_partitions, stabilized_weights, rho=rho)
    return {
        "n_objectives": int(n_obj),
        "divisions_p": int(n_partitions),
        "n_das_dennis": int(len(base)),
        "n_anchors_constructed": int(n_obj + 1),
        "n_anchors_retained": int(len(full) - len(base)),
        "n_ref_dirs": int(len(full)),
        "rho": float(rho),
        "priority_weights": [float(v) for v in stabilized_weights],
    }
