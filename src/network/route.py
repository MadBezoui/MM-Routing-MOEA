"""route.py
===========
Solution encoding of Section 3.1.

A solution is a path :math:`P = (v_1, \\dots, v_k)` together with a mode
sequence :math:`M(P) = (m_1, \\dots, m_{k-1})`, where :math:`m_i` is the
transport mode used on the edge :math:`(v_i, v_{i+1})`.

Section 4.3 distinguishes two notions that this module makes explicit:

*topological validity*
    every consecutive node pair is an edge of :math:`G` carrying the requested
    mode, and every mode transition occurs at a node that hosts the required
    infrastructure;
*operational feasibility*
    a topologically valid path that additionally satisfies
    :math:`\\mathrm{CV}(P) = 0` (Eq. 5).

Only topologically valid routes are ever passed to the objective functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import networkx as nx

#: Modes that require boarding a scheduled vehicle at an equipped facility.
TRANSIT_MODES = frozenset({"bus", "tram"})

#: Modes a traveller can start or resume anywhere on the corresponding layer.
PRIVATE_MODES = frozenset({"walk", "bike", "car"})


@dataclass(frozen=True)
class Route:
    """An ordered node sequence with its per-edge mode sequence."""

    nodes: Tuple[str, ...]
    modes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.modes) != max(len(self.nodes) - 1, 0):
            raise ValueError(
                f"mode sequence of length {len(self.modes)} does not match "
                f"a path of {len(self.nodes)} nodes"
            )

    # -- basic properties --------------------------------------------------

    @property
    def origin(self) -> str:
        return self.nodes[0]

    @property
    def destination(self) -> str:
        return self.nodes[-1]

    @property
    def n_edges(self) -> int:
        return len(self.modes)

    def edges(self) -> Iterable[Tuple[str, str, str]]:
        """Yield ``(u, v, mode)`` for every edge of the route."""
        return zip(self.nodes[:-1], self.nodes[1:], self.modes)

    def transfer_indices(self) -> Tuple[int, ...]:
        """Indices ``i`` such that a mode change happens at node ``v_{i+1}``."""
        return tuple(
            i for i in range(len(self.modes) - 1)
            if self.modes[i] != self.modes[i + 1]
        )

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Route({len(self.nodes)} nodes, modes={'->'.join(self.modes)})"


# --------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------

def node_supports_mode(G: nx.MultiDiGraph, node: str, mode: str) -> bool:
    """Whether ``node`` hosts the infrastructure required by ``mode``.

    A private mode is available wherever the corresponding layer touches the
    node.  A transit mode additionally requires the node to be a stop of that
    mode, which in this graph is equivalent to carrying an incident edge of
    that mode.
    """
    if node not in G:
        return False
    for _, _, d in G.out_edges(node, data=True):
        if d["mode"] == mode:
            return True
    for _, _, d in G.in_edges(node, data=True):
        if d["mode"] == mode:
            return True
    return False


def is_topologically_valid(G: nx.MultiDiGraph, route: Route) -> bool:
    """Return ``True`` when ``route`` satisfies conditions (i) and (ii) of §4.3.
    
    Note: All transitions between incident modes are intentionally allowed by the model
    as long as both modes are supported by the node infrastructure.
    """
    if route is None or len(route.nodes) < 2:
        return False
    if len(set(route.nodes)) != len(route.nodes):
        return False  # simple paths only: no repeated node

    for u, v, mode in route.edges():
        data = G.get_edge_data(u, v)
        if not data or mode not in data:
            return False

    # (ii) every mode transition occurs at an equipped node
    for i in route.transfer_indices():
        junction = route.nodes[i + 1]
        if not node_supports_mode(G, junction, route.modes[i]):
            return False
        if not node_supports_mode(G, junction, route.modes[i + 1]):
            return False
    return True


def transfer_quality(G: nx.MultiDiGraph, node: str) -> float:
    """Quality score in ``[0, 1]`` of the transfer facility at ``node``.

    Interpolates the three-to-fifteen-minute penalty range of Section 3.2:
    a well-equipped, frequently served interchange scores near one and incurs
    the lower bound of the range, an unequipped node scores near zero and
    incurs the upper bound.
    """
    if node not in G:
        return 0.0
    n_modes = int(G.nodes[node].get("n_modes", 0))
    n_trips = float(G.nodes[node].get("n_trips", 0.0))
    mode_score = min(n_modes / 5.0, 1.0)
    service_score = min(n_trips / 120.0, 1.0)
    return 0.5 * mode_score + 0.5 * service_score


def route_from_sequence(nodes: Sequence[str], modes: Sequence[str]) -> Route:
    return Route(tuple(nodes), tuple(modes))
