import pytest
import networkx as nx
import numpy as np

from src.network.evaluator import PathMultimodalEvaluator

@pytest.fixture
def mock_graph():
    G = nx.MultiDiGraph()
    # Linear path: A -> B -> C -> D
    G.add_edge('A', 'B', mode='walk', length=500.0, travel_time_sec=300)
    G.add_edge('B', 'C', mode='bus', length=2000.0, travel_time_sec=600)
    G.add_edge('C', 'D', mode='walk', length=500.0, travel_time_sec=300)
    
    # Alternate expensive path
    G.add_edge('A', 'D', mode='walk', length=5000.0, travel_time_sec=3000)
    
    # Cycle path
    G.add_edge('C', 'B', mode='walk', length=100.0, travel_time_sec=60)
    
    return G

@pytest.fixture
def evaluator(mock_graph):
    # Dummy comfort predictor that always returns 0
    class MockPredictor:
        def predict(self, *args, **kwargs):
            return np.array([0.0])
            
    class MockSurvey:
        mean_daily_budget_eur = 5.0
        walking_threshold_km = 1.0

    return PathMultimodalEvaluator(mock_graph, MockSurvey(), MockPredictor())

@pytest.fixture
def profile():
    return {
        "origin_node": "A",
        "dest_node": "D",
        "budget_eur": 1.0,  # Bus cost is 1.9, so this will fail if bus is used
        "max_time_min": 15.0, # 900 seconds
        "max_walk_km": 0.8  # 800 meters
    }

def evaluate_path(evaluator, path, profile):
    X = np.empty((1, 1), dtype=object)
    X[0, 0] = path
    F, G, _ = evaluator(X, profile, {}, None)
    return F[0], G[0]

def test_1_invalid_empty_path(evaluator, profile):
    F, G = evaluate_path(evaluator, [], profile)
    assert G[0] > 0.0 # g_invalid > 0
    assert F[0] == 180.0
    
def test_2_invalid_disconnected_path(evaluator, profile):
    # 'A' -> 'C' is disconnected
    F, G = evaluate_path(evaluator, ['A', 'C'], profile)
    assert G[0] > 0.0

def test_3_invalid_cycles(evaluator, profile):
    # 'A' -> 'B' -> 'C' -> 'B' -> 'C' -> 'D'
    F, G = evaluate_path(evaluator, ['A', 'B', 'C', 'B', 'C', 'D'], profile)
    assert G[0] > 0.0

def test_4_invalid_single_node(evaluator, profile):
    F, G = evaluate_path(evaluator, ['A'], profile)
    assert G[0] > 0.0

def test_5_invalid_missing_node(evaluator, profile):
    # 'Z' does not exist
    F, G = evaluate_path(evaluator, ['A', 'Z'], profile)
    assert G[0] > 0.0

def test_6_budget_exceeded(evaluator, profile):
    # 'A' -> 'B' -> 'C' -> 'D' uses bus (costs 1.9), budget is 1.0
    F, G = evaluate_path(evaluator, ['A', 'B', 'C', 'D'], profile)
    assert G[0] <= 0.0 # Structurally valid
    assert G[1] > 0.0 # Budget exceeded
    
def test_7_time_exceeded(evaluator, profile):
    # 'A' -> 'D' takes 3000 seconds = 50 min. Profile max is 15 min.
    profile["budget_eur"] = 10.0 # Relax budget
    profile["max_walk_km"] = 10.0 # Relax walk
    F, G = evaluate_path(evaluator, ['A', 'D'], profile)
    assert G[0] <= 0.0 # Structurally valid
    assert G[2] > 0.0 # Time exceeded
    
def test_8_walk_distance_exceeded(evaluator, profile):
    # 'A' -> 'D' takes 5000 meters = 5km walk. Profile max is 0.8km.
    profile["budget_eur"] = 10.0 # Relax budget
    profile["max_time_min"] = 100.0 # Relax time
    F, G = evaluate_path(evaluator, ['A', 'D'], profile)
    assert G[0] <= 0.0 # Structurally valid
    assert G[3] > 0.0 # Walk exceeded

def test_9_multiple_constraints_violated(evaluator, profile):
    # 'A' -> 'D' takes 50 min, 5km walk.
    profile["budget_eur"] = 10.0 
    # Keep time max=15m, walk max=0.8km
    F, G = evaluate_path(evaluator, ['A', 'D'], profile)
    assert G[0] <= 0.0 # Structurally valid
    assert G[2] > 0.0 # Time exceeded
    assert G[3] > 0.0 # Walk exceeded

def test_10_penalties_applied_for_invalid(evaluator, profile):
    # Check that structurally invalid path has correct large F values
    F, G = evaluate_path(evaluator, [], profile)
    assert F[0] == 180.0
    assert F[1] == 20.0
    assert F[2] == 10.0
    assert F[3] == 1.0
