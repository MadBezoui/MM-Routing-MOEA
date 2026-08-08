import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.network.route import Route
from src.statistics import compare

def test_statistics_label_inversion():
    """
    Test that compare() correctly attributes wins to algo_a when its metric is better (higher).
    """
    # We want a dataframe where algo_a has higher values
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
    
    # algo_a is always +0.3 better.
    # So run_win_rate_a should be 1.0
    # profile_wins_a should be 2
    
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
    
    # Check that route rejects bad sizes
    with pytest.raises(ValueError):
        Route(("A", "B"), ("walk", "bus"))
