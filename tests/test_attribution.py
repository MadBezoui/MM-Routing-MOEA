import pandas as pd
import numpy as np
import pytest
from src.statistics import compare

def test_compare_algorithm_a_always_wins():
    # Create synthetic data where A is strictly better (higher normalized_hv is better)
    df = pd.DataFrame({
        "algorithm": ["A", "A", "A", "B", "B", "B"],
        "profile_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "seed": [1, 1, 1, 1, 1, 1],
        "normalized_hv": [1.0, 0.9, 1.1, 0.5, 0.4, 0.6]
    })
    res = compare(df, "A", "B", higher_is_better=True)
    assert res["profile_wins_a"] == 3
    assert res["profile_wins_b"] == 0
    assert res["profile_ties"] == 0
    assert res["dz_profile_level_confirmatory"] > 0
    assert res["mean_diff"] > 0

def test_compare_algorithm_b_always_wins():
    # B is strictly better
    df = pd.DataFrame({
        "algorithm": ["A", "A", "A", "B", "B", "B"],
        "profile_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "seed": [1, 1, 1, 1, 1, 1],
        "normalized_hv": [0.5, 0.4, 0.6, 1.0, 0.9, 1.1]
    })
    res = compare(df, "A", "B", higher_is_better=True)
    assert res["profile_wins_a"] == 0
    assert res["profile_wins_b"] == 3
    assert res["profile_ties"] == 0
    assert res["dz_profile_level_confirmatory"] < 0
    assert res["mean_diff"] < 0

def test_compare_all_ties():
    # A and B are identical
    df = pd.DataFrame({
        "algorithm": ["A", "A", "A", "B", "B", "B"],
        "profile_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "seed": [1, 1, 1, 1, 1, 1],
        "normalized_hv": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    })
    res = compare(df, "A", "B", higher_is_better=True)
    assert res["profile_wins_a"] == 0
    assert res["profile_wins_b"] == 0
    assert res["profile_ties"] == 3
    assert np.isnan(res["dz_profile_level_confirmatory"])
    assert res["mean_diff"] == 0.0

def test_compare_mixed_wins():
    # A wins p1, B wins p2, tie on p3
    df = pd.DataFrame({
        "algorithm": ["A", "A", "A", "B", "B", "B"],
        "profile_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "seed": [1, 1, 1, 1, 1, 1],
        "normalized_hv": [1.0, 0.5, 0.8, 0.5, 1.0, 0.8]
    })
    res = compare(df, "A", "B", higher_is_better=True)
    assert res["profile_wins_a"] == 1
    assert res["profile_wins_b"] == 1
    assert res["profile_ties"] == 1
    assert res["mean_diff"] == 0.0

def test_compare_lower_is_better():
    # Test for metrics where lower is better (e.g. generation counts or runtime)
    # A has lower values, so A should win.
    df = pd.DataFrame({
        "algorithm": ["A", "A", "A", "B", "B", "B"],
        "profile_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
        "seed": [1, 1, 1, 1, 1, 1],
        "metric_x": [10.0, 10.0, 10.0, 50.0, 50.0, 50.0]
    })
    res = compare(df, "A", "B", value_col="metric_x", higher_is_better=False)
    # If lower is better, diff is a - b. 10 - 50 = -40. A negative diff means a < b, which means a is better.
    # Therefore profile_wins_a should be 3, profile_wins_b should be 0.
    assert res["profile_wins_a"] == 3
    assert res["profile_wins_b"] == 0
    assert res["mean_diff"] < 0
