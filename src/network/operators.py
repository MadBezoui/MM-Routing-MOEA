"""operators.py
================
Variation operators for path-encoded solutions (Section 4.3).

``PathSampling``
    Builds topologically valid origin-destination routes by randomised
    shortest-path search, so that the initial population covers several
    mode-dominated route families.

``PathCrossover``
    Selects a node shared by the two parents and swaps the suffix subpaths.
    An offspring is accepted only if the resulting node sequence remains
    topologically valid *and* the mode transition at the concatenation node is
    admissible.

``PathMutation``
    Mutation first attempts a parallel-edge mode substitution between the same endpoints. If no such substitution exists, it introduces a local detour containing at most two intermediate nodes.

Offspring that fail the topological-validity check are rejected and resampled
up to :data:`MAX_RESAMPLE`; if no valid offspring is produced the parent is
retained.  Only topologically valid routes ever leave these operators.

Randomness
----------
``pymoo`` threads a per-run :class:`numpy.random.Generator` through
``Sampling.do``, ``Crossover.do`` and ``Mutation.do`` as the ``random_state``
keyword.  Every operator below draws from that generator and never from the
global ``numpy.random`` state, so concurrent runs in a thread pool stay
reproducible and independent.

Crossover probability is delegated to :class:`pymoo.core.crossover.Crossover`,
which masks the matings itself; applying a second test inside ``_do`` would
silently square the probability.
"""

from __future__ import annotations

import heapq
import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from src.network.route import Route, is_topologically_valid, node_supports_mode

logger = logging.getLogger(__name__)

#: Maximum number of resampling attempts before the parent is retained.
MAX_RESAMPLE = 8

#: Maximum number of intermediate nodes in a mutation detour (Section 4.3).
MAX_DETOUR_NODES = 2


# --------------------------------------------------------------------------
# Graph index
# --------------------------------------------------------------------------

class MultimodalIndex:
    """Adjacency cache: ``node -> [(neighbour, mode, travel_time_min), ...]``."""

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self.out: Dict[str, List[Tuple[str, str, float]]] = {n: [] for n in G.nodes()}
        for u, v, key, data in G.edges(keys=True, data=True):
            self.out[u].append((v, data["mode"], float(data["travel_time_min"])))
        self.nodes: List[str] = list(G.nodes())

    def modes_between(self, u: str, v: str) -> List[str]:
        data = self.G.get_edge_data(u, v)
        return list(data.keys()) if data else []


# --------------------------------------------------------------------------
# Randomised shortest path
# --------------------------------------------------------------------------

#: Mode families used to diversify the initial population.  Section 3.4 (i)
#: describes the feasible objective space as clustering into route families
#: dominated by one mode, plus a mixed family; sampling therefore draws a
#: family per individual rather than treating all modes symmetrically.
MODE_FAMILIES: Tuple[Tuple[str, ...], ...] = (
    ("walk",), ("bike",), ("bus", "tram"), ("car",), (),
)

#: Weight discount applied to the edges of the preferred family, and surcharge
#: applied to the others.  The gap has to be wide enough for a walking- or
#: car-dominated path to win against a transit path that is three times faster.
FAMILY_DISCOUNT = 0.20
FAMILY_SURCHARGE = 5.0

#: The transfer penalty used *inside the sampler* is drawn per individual from
#: the range of Section 3.2.  A traveller who dislikes interchanges explores
#: direct routes, one who tolerates them explores connecting routes; sampling
#: the penalty therefore spreads the initial population along the transfer axis
#: instead of collapsing it onto a single shortest path.
SAMPLER_TRANSFER_PENALTY_RANGE = (3.0, 15.0)


def randomised_route(
    index: MultimodalIndex,
    origin: str,
    destination: str,
    rng: np.random.Generator,
    family: Optional[Tuple[str, ...]] = None,
    transfer_penalty_min: float = 6.0,
    perturbation: Tuple[float, float] = (0.7, 1.5),
    max_expansions: int = 60000,
) -> Optional[Route]:
    """Randomised mode-aware shortest path from ``origin`` to ``destination``.

    The search runs over ``(node, arrival mode)`` states rather than nodes, so
    that changing mode can be charged a transfer penalty.  Without that penalty
    a plain shortest-path search hops between layers at almost every node and
    produces routes with eight or more interchanges, which are both unrealistic
    and dominated once Eq. 1 charges the waiting time.

    Two sources of diversity are combined:

    * multiplicative noise on every edge weight, so repeated calls return
      structurally different paths;
    * an optional preferred mode ``family`` whose edges are discounted and
      whose complement is surcharged, so the initial population spans the
      walking-, cycling-, transit- and car-dominated families of Section 3.4.
    """
    if origin not in index.out or destination not in index.out or origin == destination:
        return None

    lo, hi = perturbation
    bias: Dict[str, float] = {}
    if family:
        for mode in ("walk", "bike", "bus", "tram", "car"):
            bias[mode] = FAMILY_DISCOUNT if mode in family else FAMILY_SURCHARGE

    start = (origin, "")
    dist: Dict[Tuple[str, str], float] = {start: 0.0}
    prev: Dict[Tuple[str, str], Tuple[Tuple[str, str], str]] = {}
    heap: List[Tuple[float, str, str]] = [(0.0, origin, "")]
    visited: set = set()
    expansions = 0
    best_goal: Optional[Tuple[str, str]] = None

    while heap and expansions < max_expansions:
        d, u, m_prev = heapq.heappop(heap)
        state = (u, m_prev)
        if state in visited:
            continue
        visited.add(state)
        expansions += 1
        if u == destination:
            best_goal = state
            break
        for v, mode, t in index.out[u]:
            nxt = (v, mode)
            if nxt in visited:
                continue
            step = t * float(rng.uniform(lo, hi)) * bias.get(mode, 1.0)
            if m_prev and mode != m_prev:
                step += transfer_penalty_min
            nd = d + step
            if nd < dist.get(nxt, np.inf):
                dist[nxt] = nd
                prev[nxt] = (state, mode)
                heapq.heappush(heap, (nd, v, mode))

    if best_goal is None:
        return None

    nodes: List[str] = [destination]
    modes: List[str] = []
    cur = best_goal
    while cur != start:
        parent, mode = prev[cur]
        nodes.append(parent[0])
        modes.append(mode)
        cur = parent
    nodes.reverse()
    modes.reverse()
    if len(set(nodes)) != len(nodes):
        return None
    return Route(tuple(nodes), tuple(modes))


def local_detour(
    index: MultimodalIndex,
    u: str,
    v: str,
    rng: np.random.Generator,
    max_intermediate: int = MAX_DETOUR_NODES,
) -> Optional[Tuple[List[str], List[str]]]:
    """Sample an admissible detour from ``u`` to ``v`` via <= ``max_intermediate`` nodes."""
    first_hops = list(index.out.get(u, []))
    if not first_hops:
        return None
    rng.shuffle(first_hops)

    for w, m1, _ in first_hops[:12]:
        if w == v or w == u:
            continue
        # one intermediate node
        modes_wv = index.modes_between(w, v)
        if modes_wv:
            m2 = str(rng.choice(modes_wv))
            return [u, w, v], [m1, m2]
        if max_intermediate < 2:
            continue
        # two intermediate nodes
        second_hops = list(index.out.get(w, []))
        rng.shuffle(second_hops)
        for z, m2, _ in second_hops[:12]:
            if z in (u, v, w):
                continue
            modes_zv = index.modes_between(z, v)
            if modes_zv:
                m3 = str(rng.choice(modes_zv))
                return [u, w, z, v], [m1, m2, m3]
    return None


# --------------------------------------------------------------------------
# pymoo operators
# --------------------------------------------------------------------------

def _profile_od(problem) -> Tuple[Optional[str], Optional[str]]:
    profile = getattr(problem, "profile", {}) or {}
    return profile.get("origin_node"), profile.get("dest_node")


class PathSampling(Sampling):
    """Initial population of topologically valid origin-destination routes."""

    def __init__(self, G: nx.MultiDiGraph, index: MultimodalIndex | None = None):
        super().__init__()
        self.G = G
        self.index = index or MultimodalIndex(G)

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        origin, destination = _profile_od(problem)
        X = np.empty((n_samples, 1), dtype=object)

        fallback: Optional[Route] = None
        for i in range(n_samples):
            # cycle through the mode families so every one is represented
            family = MODE_FAMILIES[i % len(MODE_FAMILIES)]
            penalty = float(rng.uniform(*SAMPLER_TRANSFER_PENALTY_RANGE))
            route = None
            for _ in range(MAX_RESAMPLE):
                candidate = randomised_route(self.index, origin, destination, rng,
                                             family=family,
                                             transfer_penalty_min=penalty)
                if candidate is not None and is_topologically_valid(self.G, candidate):
                    route = candidate
                    break
            if route is None:
                if fallback is None:
                    fallback = randomised_route(
                        self.index, origin, destination, rng,
                        family=None, perturbation=(1.0, 1.0),
                    )
                route = fallback
            if route is None:
                raise RuntimeError(
                    f"no route connects {origin} to {destination}; the graph "
                    "should be strongly connected"
                )
            X[i, 0] = route
            fallback = fallback or route
        return X


class PathCrossover(Crossover):
    """Suffix exchange at a node shared by both parents."""

    def __init__(self, G: nx.MultiDiGraph, prob: float = 0.9):
        # ``prob`` is handed to the base class, which masks the matings on
        # which crossover is executed.  ``_do`` must therefore attempt the
        # splice unconditionally.
        super().__init__(2, 2, prob=prob)
        self.G = G

    def _splice(self, p1: Route, p2: Route, junction: str) -> Optional[Route]:
        i = p1.nodes.index(junction)
        j = p2.nodes.index(junction)
        nodes = p1.nodes[:i] + p2.nodes[j:]
        modes = p1.modes[:i] + p2.modes[j:]
        if len(set(nodes)) != len(nodes):
            return None  # the splice created a cycle
        try:
            candidate = Route(tuple(nodes), tuple(modes))
        except ValueError:
            return None
        # the mode transition at the concatenation node must be admissible
        if 0 < i < len(modes):
            if not node_supports_mode(self.G, junction, modes[i - 1]):
                return None
            if not node_supports_mode(self.G, junction, modes[i]):
                return None
        return candidate if is_topologically_valid(self.G, candidate) else None

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        _, n_matings, _ = X.shape
        Y = np.empty((2, n_matings, 1), dtype=object)

        for k in range(n_matings):
            p1: Route = X[0, k, 0]
            p2: Route = X[1, k, 0]
            o1, o2 = p1, p2

            shared = sorted(
                n for n in set(p1.nodes) & set(p2.nodes)
                if n not in (p1.origin, p1.destination)
            )
            rng.shuffle(shared)
            for junction in shared[:MAX_RESAMPLE]:
                c1 = self._splice(p1, p2, junction)
                c2 = self._splice(p2, p1, junction)
                if c1 is not None:
                    o1 = c1
                if c2 is not None:
                    o2 = c2
                if c1 is not None or c2 is not None:
                    break

            Y[0, k, 0] = o1
            Y[1, k, 0] = o2
        return Y


class PathMutation(Mutation):
    """Parallel-edge substitution with local-detour fallback."""

    def __init__(self, G: nx.MultiDiGraph, index: MultimodalIndex | None = None, prob: float = 0.2):
        # The base class is set to always call ``_do`` and the per-route test is
        # applied inside it, so that the expensive graph work is skipped on the
        # routes that are not mutated.  Passing ``prob`` upwards instead would
        # mutate every route and then discard most of the results.
        super().__init__(prob=1.0)
        self.G = G
        self.index = index or MultimodalIndex(G)
        self.mutation_prob = float(prob)

    def _mutate(self, route: Route, rng: np.random.Generator) -> Optional[Route]:
        n_edges = route.n_edges
        if n_edges == 0:
            return None
        i = int(rng.integers(0, n_edges))
        u, v, mode = route.nodes[i], route.nodes[i + 1], route.modes[i]

        # (a) parallel edge with the same tail and head
        alternatives = [m for m in self.index.modes_between(u, v) if m != mode]
        if alternatives:
            new_mode = str(rng.choice(alternatives))
            modes = list(route.modes)
            modes[i] = new_mode
            return Route(route.nodes, tuple(modes))

        # (b) local admissible detour through at most two intermediate nodes
        detour = local_detour(self.index, u, v, rng)
        if detour is None:
            return None
        det_nodes, det_modes = detour
        nodes = list(route.nodes[:i]) + det_nodes + list(route.nodes[i + 2:])
        modes = list(route.modes[:i]) + det_modes + list(route.modes[i + 1:])
        if len(set(nodes)) != len(nodes):
            return None
        try:
            return Route(tuple(nodes), tuple(modes))
        except ValueError:
            return None

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        Y = np.empty_like(X, dtype=object)
        for i in range(X.shape[0]):
            route: Route = X[i, 0]
            Y[i, 0] = route
            if rng.random() >= self.mutation_prob:
                continue
            for _ in range(MAX_RESAMPLE):
                candidate = self._mutate(route, rng)
                if candidate is not None and is_topologically_valid(self.G, candidate):
                    Y[i, 0] = candidate
                    break
        return Y


class PathDuplicateElimination(ElementwiseDuplicateElimination):
    """Structural duplicate test used by pymoo's duplicate filter."""

    def is_equal(self, a, b) -> bool:  # pragma: no cover - trivial
        ra, rb = a.X[0], b.X[0]
        return ra.nodes == rb.nodes and ra.modes == rb.modes
