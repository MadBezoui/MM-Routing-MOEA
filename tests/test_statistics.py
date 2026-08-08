import numpy as np
import pytest
from src.statistics import wilcoxon_report

def test_wilcoxon_all_positive():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    res = wilcoxon_report(d)
    assert res["r"] > 0
    assert res["n_total"] == 4
    assert res["n_effective"] == 4

def test_wilcoxon_all_negative():
    d = np.array([-1.0, -2.0, -3.0, -4.0])
    res = wilcoxon_report(d)
    assert res["r"] < 0

def test_wilcoxon_all_zeros():
    d = np.array([0.0, 0.0, 0.0, 0.0])
    res = wilcoxon_report(d)
    assert np.isnan(res["r"])
    assert res["n_effective"] == 0
    assert res["n_total"] == 4

def test_wilcoxon_mixed():
    d = np.array([1.0, -2.0, 3.0, -4.0])
    res = wilcoxon_report(d)
    assert not np.isnan(res["r"])

def test_wilcoxon_small_effective_sample():
    d = np.array([1.0, -1.0, 0.0, 0.0])
    res = wilcoxon_report(d)
    assert res["n_effective"] == 2
    assert np.isnan(res["r"])

def test_wilcoxon_zeros_with_mixed():
    d = np.array([1.0, -2.0, 0.0, 3.0])
    res = wilcoxon_report(d)
    assert res["n_effective"] == 3
    assert res["n_total"] == 4
