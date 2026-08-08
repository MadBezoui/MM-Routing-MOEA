"""evaluator.py
================
Objective and constraint evaluation on the multimodal graph (Section 3.2-3.3).

Implements Eq. 1 to Eq. 5 literally:

===========  ==========================================================
Eq. 1        :math:`f_1(P)=\\sum_{e\\in P} t(e) + \\sum_{\\tau\\in T(P)} t_{wait}(\\tau)`
(Note: the waiting time term includes both initial waiting time and transfer waiting time.)
Eq. 2        :math:`f_2(P)=\\sum_{e\\in P} c(e)\\,\\pi_{m(e)}`
Eq. 3        :math:`f_3(P)=\\sum_{e\\in P} d(e)\\,\\varepsilon_{m(e)}\\,\\omega_{m(e)}`
Eq. 4        :math:`f_4(P)=1-M_{comfort}(\\phi(P), u)`
Eq. 5        :math:`\\mathrm{CV}(P)=\\frac{[f_2-B_u]_+}{B_u}+\\frac{[f_1-T_{max,u}]_+}{T_{max,u}}+\\frac{[\\mathrm{WalkDist}-W_{lim,u}]_+}{W_{lim,u}}`
===========  ==========================================================

The constraint violation is used **only** for the three operational bounds of
Section 3.3.  Disconnected or illegal graph objects never reach this module:
they are rejected by the topological-validity check of
:mod:`src.network.route` inside the variation operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.network.route import TRANSIT_MODES, Route, transfer_quality

MODES: Tuple[str, ...] = ("walk", "bike", "bus", "tram", "car")

#: Per-mode emission factors in kg CO2 per passenger-kilometre (Section 3.2).
#: Walking and cycling emit nothing; the tram, bus and car factors follow the
#: European average per-passenger-kilometre figures quoted in the manuscript.
EMISSION_FACTORS: Dict[str, float] = {
    "walk": 0.000,
    "bike": 0.000,
    "tram": 0.040,
    "bus": 0.080,
    "car": 0.150,
}

#: Per-mode tariffs.  ``boarding`` is charged once when a scheduled vehicle is
#: boarded; ``per_km`` is charged on every kilometre travelled by that mode.
#: Values reflect the CTS single-ticket fare and standard French running costs.
TARIFFS: Dict[str, Dict[str, float]] = {
    "walk": {"boarding": 0.00, "per_km": 0.00},
    "bike": {"boarding": 0.00, "per_km": 0.03},
    "bus": {"boarding": 1.90, "per_km": 0.00},
    "tram": {"boarding": 1.90, "per_km": 0.00},
    "car": {"boarding": 0.00, "per_km": 0.28},
}

#: Transfer-penalty range of Section 3.2, in minutes.
MIN_TRANSFER_WAIT_MIN = 3.0
MAX_TRANSFER_WAIT_MIN = 15.0

#: Perceived in-vehicle crowding by mode, used to build the comfort feature
#: vector phi(P) of Eq. 4.
CROWDING_BY_MODE: Dict[str, float] = {
    "walk": 0.00, "bike": 0.00, "bus": 0.70, "tram": 0.50, "car": 0.20,
}

#: The twelve components of phi(P) (Section 4.1).
COMFORT_FEATURES: Tuple[str, ...] = (
    "walk_share", "bike_share", "bus_share", "tram_share", "car_share",
    "crowding", "transfers", "distance_km", "rain", "temperature_c",
    "age", "mobility_restriction",
)


@dataclass
class RouteBreakdown:
    """Per-mode decomposition of a route, reused by several objectives."""

    distance_by_mode: Dict[str, float]
    time_by_mode: Dict[str, float]
    total_distance_km: float
    in_vehicle_time_min: float
    wait_time_min: float
    n_transfers: int
    fare_eur: float
    emissions_kg: float


# --------------------------------------------------------------------------
# Route decomposition
# --------------------------------------------------------------------------

def decompose(
    G: nx.MultiDiGraph,
    route: Route,
    pricing: Dict[str, float] | None = None,
    occupancy: Dict[str, float] | None = None,
) -> RouteBreakdown:
    """Walk the edges of ``route`` once and accumulate every quantity needed.

    ``pricing`` and ``occupancy`` carry the multiplicative factors
    :math:`\\pi_{m}` of Eq. 2 and :math:`\\omega_{m}` of Eq. 3.  Both default
    to one, which is the deterministic setting used in the main experiments.
    """
    pricing = pricing or {}
    occupancy = occupancy or {}

    distance = {m: 0.0 for m in MODES}
    time = {m: 0.0 for m in MODES}
    fare = 0.0
    emissions = 0.0
    wait_total = 0.0

    previous_mode: str | None = None

    for idx, (u, v, mode) in enumerate(route.edges()):
        data = G.get_edge_data(u, v)[mode]
        length = float(data["length_km"])
        t_edge = float(data["travel_time_min"])

        distance[mode] += length
        time[mode] += t_edge

        # --- Eq. 2: per-edge tariff times the fare-variability factor ------
        tariff = TARIFFS[mode]
        edge_cost = tariff["per_km"] * length
        boarding = mode in TRANSIT_MODES and mode != previous_mode
        if boarding:
            edge_cost += tariff["boarding"]
        fare += edge_cost * float(pricing.get(mode, 1.0))

        # --- Eq. 3: length times emission factor times occupancy ----------
        emissions += length * EMISSION_FACTORS[mode] * float(occupancy.get(mode, 1.0))

        # --- Eq. 1: waiting term at every transfer point ------------------
        if previous_mode is not None and mode != previous_mode:
            junction = route.nodes[idx]
            quality = transfer_quality(G, junction)
            penalty = MAX_TRANSFER_WAIT_MIN - quality * (MAX_TRANSFER_WAIT_MIN - MIN_TRANSFER_WAIT_MIN)
            if mode in TRANSIT_MODES:
                # a scheduled vehicle also imposes its own headway-derived wait
                penalty = min(max(float(data.get("wait_min", penalty)), MIN_TRANSFER_WAIT_MIN),
                              MAX_TRANSFER_WAIT_MIN)
            wait_total += penalty
        elif previous_mode is None and mode in TRANSIT_MODES:
            wait_total += min(max(float(data.get("wait_min", MIN_TRANSFER_WAIT_MIN)),
                                  MIN_TRANSFER_WAIT_MIN), MAX_TRANSFER_WAIT_MIN)

        previous_mode = mode

    return RouteBreakdown(
        distance_by_mode=distance,
        time_by_mode=time,
        total_distance_km=float(sum(distance.values())),
        in_vehicle_time_min=float(sum(time.values())),
        wait_time_min=float(wait_total),
        n_transfers=len(route.transfer_indices()),
        fare_eur=float(fare),
        emissions_kg=float(emissions),
    )


def comfort_features(
    G: nx.MultiDiGraph,
    route: Route,
    breakdown: RouteBreakdown,
    profile: Dict[str, Any],
) -> Dict[str, float]:
    """Build the twelve-dimensional feature vector phi(P) of Eq. 4."""
    total = max(breakdown.total_distance_km, 1e-9)
    shares = {m: breakdown.distance_by_mode[m] / total for m in MODES}
    crowding = float(sum(shares[m] * CROWDING_BY_MODE[m] for m in MODES))

    return {
        "walk_share": shares["walk"],
        "bike_share": shares["bike"],
        "bus_share": shares["bus"],
        "tram_share": shares["tram"],
        "car_share": shares["car"],
        "crowding": crowding,
        "transfers": float(breakdown.n_transfers),
        "distance_km": breakdown.total_distance_km,
        "rain": float(profile.get("rain", 0.0)),
        "temperature_c": float(profile.get("temperature_c", 14.0)),
        "age": float(profile.get("age", 25.8)),
        "mobility_restriction": float(profile.get("mobility_restriction", 0)),
    }


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------

class PathMultimodalEvaluator:
    """Evaluate Eq. 1 to Eq. 5 for a population of routes.

    Parameters
    ----------
    G
        The multimodal graph.
    survey
        Survey calibration object, used only as a fallback for profile fields.
    comfort_predictor
        Object exposing ``predict(DataFrame, survey) -> ndarray`` in ``[0, 1]``.
    scenario
        :class:`~src.config.ScenarioConfig`; controls the stochastic
        multiplicative factors of Eq. 2 and Eq. 3.
    n_monte_carlo
        When greater than one, the stochastic factors are sampled this many
        times per candidate and the resulting :math:`f_2` and :math:`f_3` are
        averaged before selection (Section 6.6).
    comfort_bias
        Constant bias added to the comfort prediction before Eq. 4, used by the
        bias-injection sensitivity analysis of Section 6.6.
    """

    def __init__(
        self,
        G: nx.MultiDiGraph,
        survey: Any,
        comfort_predictor: Any,
        scenario: Any = None,
        n_monte_carlo: int = 1,
        comfort_bias: float = 0.0,
        algorithm_seed: int = 0,
    ):
        self.G = G
        self.survey = survey
        self.comfort_predictor = comfort_predictor
        self.scenario = scenario
        self.n_monte_carlo = max(int(n_monte_carlo), 1)
        self.comfort_bias = float(comfort_bias)
        self.algorithm_seed = int(algorithm_seed)
        self._pregenerated_scenarios = None

    # -- profile bounds ----------------------------------------------------

    def _bounds(self, profile: Dict[str, Any]) -> Tuple[float, float, float]:
        """Return the per-profile ``(B_u, T_max_u, W_lim_u)`` of Section 3.3."""
        budget = float(profile.get("budget_eur",
                                   getattr(self.survey, "mean_daily_budget_eur", 6.0)))
        t_max = float(profile.get("max_travel_time_min", 60.0))
        w_lim = float(profile.get("max_walking_distance_km",
                                  getattr(self.survey, "walking_threshold_km", 1.16)))
        return max(budget, 1e-6), max(t_max, 1e-6), max(w_lim, 1e-6)

    # -- stochastic multipliers -------------------------------------------

    def _draw_factors(self, rng: np.random.Generator) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        sc = self.scenario
        pricing = {m: 1.0 for m in MODES}
        occupancy = {m: 1.0 for m in MODES}
        congestion = {m: 1.0 for m in MODES}
        if sc is None:
            return pricing, occupancy, congestion
        if getattr(sc, "dynamic_pricing", False):
            pricing = {m: max(rng.normal(sc.pricing_multiplier_mean, sc.pricing_multiplier_std), 0.5) for m in MODES}
        if getattr(sc, "stochastic_crowding", False):
            occupancy = {m: max(rng.normal(sc.occupancy_multiplier_mean, sc.occupancy_multiplier_std), 0.5) for m in MODES}
        if getattr(sc, "stochastic_travel_time", False):
            congestion = {m: max(rng.normal(sc.congestion_multiplier_mean, sc.congestion_multiplier_std), 0.4) for m in MODES}
        return pricing, occupancy, congestion

    # -- main entry point --------------------------------------------------

    def __call__(
        self,
        X: np.ndarray,
        profile: Dict[str, Any],
        extras: Dict[str, Any],
        scenario: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        if scenario is not None:
            self.scenario = scenario

        routes: List[Route] = [X[i, 0] for i in range(X.shape[0])]
        n = len(routes)
        budget, t_max, w_lim = self._bounds(profile)
        
        if self._pregenerated_scenarios is None:
            profile_id = str(profile.get("profile_id", "unknown"))
            # Common random numbers: scenarios fixed per (profile, seed)
            seed = (hash(profile_id) % (2**31)) + self.algorithm_seed
            rng = np.random.default_rng(seed)
            self._pregenerated_scenarios = [self._draw_factors(rng) for _ in range(self.n_monte_carlo)]

        f1 = np.zeros(n)
        f2 = np.zeros(n)
        f3 = np.zeros(n)
        walk_dist = np.zeros(n)
        transfers = np.zeros(n, dtype=int)
        dominant = np.empty(n, dtype=object)
        feature_rows: List[Dict[str, float]] = []

        for i, route in enumerate(routes):
            costs = np.zeros(self.n_monte_carlo)
            emis = np.zeros(self.n_monte_carlo)
            times = np.zeros(self.n_monte_carlo)
            breakdown = None
            for s in range(self.n_monte_carlo):
                pricing, occupancy, congestion = self._pregenerated_scenarios[s]
                breakdown = decompose(self.G, route, pricing=pricing, occupancy=occupancy)
                stochastic_in_vehicle_time = sum(breakdown.time_by_mode[m] * congestion[m] for m in MODES)
                times[s] = stochastic_in_vehicle_time + breakdown.wait_time_min
                costs[s] = breakdown.fare_eur
                emis[s] = breakdown.emissions_kg

            f1[i] = float(times.mean())
            f2[i] = float(costs.mean())
            f3[i] = float(emis.mean())
            walk_dist[i] = breakdown.distance_by_mode["walk"]
            transfers[i] = breakdown.n_transfers
            dominant[i] = max(MODES, key=lambda m: breakdown.distance_by_mode[m])
            feature_rows.append(comfort_features(self.G, route, breakdown, profile))

        # --- Eq. 4 ---------------------------------------------------------
        features = pd.DataFrame(feature_rows, columns=list(COMFORT_FEATURES))
        comfort = np.asarray(self.comfort_predictor.predict(features, self.survey), dtype=float)
        comfort = np.clip(comfort + self.comfort_bias, 0.0, 1.0)
        f4 = 1.0 - comfort

        # --- Eq. 5 ---------------------------------------------------------
        cv = (
            np.maximum(f2 - budget, 0.0) / budget
            + np.maximum(f1 - t_max, 0.0) / t_max
            + np.maximum(walk_dist - w_lim, 0.0) / w_lim
        )

        F = np.column_stack([f1, f2, f3, f4])
        Gc = cv[:, None]
        meta = {
            "dominant_mode": dominant,
            "travel_time_min": f1,
            "cost": f2,
            "emissions": f3,
            "comfort_score": comfort,
            "walk_distance_km": walk_dist,
            "n_transfers": transfers,
        }
        return F, Gc, meta
