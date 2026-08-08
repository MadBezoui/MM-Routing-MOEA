import networkx as nx
import numpy as np
import pandas as pd
import pytest

from pymoo.core.population import Population
from src.network.route import Route
from src.statistics import compare
from src.optimization_framework_parallel3 import MetricsCallback
from src.network.evaluator import PathMultimodalEvaluator
from src.config import ScenarioConfig

def test_statistics_label_inversion():
    """
    Test that compare() correctly attributes wins to algo_a when its metric is better (higher).
    """
    df = pd.DataFrame([
        {"profile_id": "p1", "seed": 1, "algorithm": "algo_a", "normalized_hv": 0.8},
        {"profile_id": "p1", "seed": 1, "algorithm": "algo_b", "normalized_hv": 0.5},
        {"profile_id": "p1", "seed": 2, "algorithm": "algo_a", "normalized_hv": 0.9},
        {"profile_id": "p1", "seed": 2, "algorithm": "algo_b", "normalized_hv": 0.6},
        {"profile_id": "p2", "seed": 1, "algorithm": "algo_a", "normalized_hv": 0.7},
        {"profile_id": "p2", "seed": 1, "algorithm": "algo_b", "normalized_hv": 0.4},
        {"profile_id": "p2", "seed": 2, "algorithm": "algo_a", "normalized_hv": 0.8},
        {"profile_id": "p2", "seed": 2, "algorithm": "algo_b", "normalized_hv": 0.5},
    ])
    
    res = compare(df, "algo_a", "algo_b", higher_is_better=True)
    assert res["run_win_rate_a"] == 1.0
    assert res["profile_wins_a"] == 2
    assert res["profile_wins_b"] == 0
    assert res["profile_ties"] == 0

def test_route_invariants():
    """
    Test route invariants.
    """
    nodes = ("A", "B", "C")
    modes = ("walk", "bus")
    route = Route(nodes, modes)
    
    assert route.origin == "A"
    assert route.destination == "C"
    assert len(route.modes) == len(route.nodes) - 1
    assert len(set(route.nodes)) == len(route.nodes)
    
    with pytest.raises(ValueError):
        Route(("A", "B"), ("walk", "bus"))


def test_metrics_no_feasible():
    """
    Test that the callback calculates metrics appropriately when no solutions are feasible,
    producing NaNs instead of blindly computing HV/IGD on the infeasible set.
    """
    class MockAlgo:
        def __init__(self):
            self.problem = None
            self.n_gen = 1
            self.pop = Population.new("X", np.array([1, 2]))
            self.pop.set("F", np.array([[1.0, 2.0], [3.0, 4.0]]))
            self.pop.set("G", np.array([[1.0], [2.0]])) # > 0 means infeasible
    
    algo = MockAlgo()
    cb = MetricsCallback(reference_front=np.array([[0,0]]), reference_point=np.array([10,10]), enabled=True)
    cb.notify(algo)
    
    history = cb.data["history"][0]
    assert history["n_feasible"] == 0
    assert history["feasible_ratio"] == 0.0
    assert np.isnan(history["hypervolume"])
    assert np.isnan(history["igd"])
    assert np.isnan(history["spacing"])


def test_evaluator_crn():
    """
    Test that Common Random Numbers logic creates identical scenarios for the same seed + profile
    and different scenarios for different seeds or profiles.
    """
    G = nx.MultiDiGraph()
    sc = ScenarioConfig(dynamic_pricing=True, stochastic_crowding=True, stochastic_travel_time=True)
    
    # Same config
    ev1 = PathMultimodalEvaluator(G, None, None, scenario=sc, n_monte_carlo=2, algorithm_seed=42)
    ev2 = PathMultimodalEvaluator(G, None, None, scenario=sc, n_monte_carlo=2, algorithm_seed=42)
    
    # Different config
    ev3 = PathMultimodalEvaluator(G, None, None, scenario=sc, n_monte_carlo=2, algorithm_seed=99)
    
    profile = {"profile_id": "test_profile"}
    profile2 = {"profile_id": "other_profile"}
    
    # Initialize generators manually as would happen in __call__
    def init_scenarios(ev, p):
        import hashlib
        profile_hash = int.from_bytes(hashlib.sha256(str(p.get("profile_id", "unknown")).encode("utf-8")).digest()[:8], "little")
        seed_sequence = np.random.SeedSequence([profile_hash & 0xFFFFFFFF, ev.algorithm_seed & 0xFFFFFFFF])
        rng = np.random.default_rng(seed_sequence)
        return [ev._draw_factors(rng) for _ in range(ev.n_monte_carlo)]
        
    s1 = init_scenarios(ev1, profile)
    s2 = init_scenarios(ev2, profile)
    s3 = init_scenarios(ev3, profile)
    s4 = init_scenarios(ev1, profile2)
    
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4


def test_evaluator_congestion_multiplier():
    """
    Test that the congestion multiplier correctly acts as a multiplier.
    A route of 10 min with a congestion multiplier of 1.5 should be exactly 15 min.
    """
    from src.network.route import Route
    from src.network.evaluator import PathMultimodalEvaluator
    
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", key="car", length_km=0.0,
               speed_kmh=60.0, mode="car", is_transit=False,
               wait_time_min=0.0, fare_eur=0.0,
               travel_time_min=10.0, emissions_kg=0.0)
    # 60 km/h = 1 km/min. Let's make length 10 km so time is 10 min.
    G.edges["A", "B", "car"]["length_km"] = 10.0
    G.nodes["A"]["supported_modes"] = ["car"]
    G.nodes["B"]["supported_modes"] = ["car"]
    
    route = Route(("A", "B"), ("car",))
    
    class MockEvaluator(PathMultimodalEvaluator):
        def _draw_factors(self, rng):
            # Force congestion multiplier for car to 1.5
            pricing = {m: 1.0 for m in ["car", "bus", "walk", "bike", "tram"]}
            occupancy = {m: 1.0 for m in ["car", "bus", "walk", "bike", "tram"]}
            congestion = {m: 1.0 for m in ["car", "bus", "walk", "bike", "tram"]}
            congestion["car"] = 1.5
            return pricing, occupancy, congestion
            
        def _bounds(self, profile):
            return 100, 100, 100
            
    class MockPredictor:
        def predict(self, features, survey):
            return np.ones(len(features))
            
    ev = MockEvaluator(G, None, MockPredictor(), n_monte_carlo=1, algorithm_seed=0)
    
    X = np.empty((1, 1), dtype=object)
    X[0, 0] = route
    profile = {"profile_id": "test", "seed_offset": 0}
    
    F, G_out, _ = ev(X, profile, {}, None)
    
    # F[0, 0] is f1 (time). 10 min * 1.5 = 15.0
    assert abs(F[0, 0] - 15.0) < 1e-6
