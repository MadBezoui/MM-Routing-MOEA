import numpy as np
import pytest
from experiments.intrinsic_dimensionality import twonn

def test_twonn_1d_line():
    np.random.seed(42)
    # Generate points along a 1D line embedded in 3D
    t = np.random.uniform(0, 10, 2000)
    # The noise must be small but non-negligible, or else 
    # nearest neighbors along the exact same ray cause zero distances or perfectly degenerate r2/r1.
    points = np.column_stack((t, 2*t, 3*t)) + np.random.normal(0, 0.05, (2000, 3))
    res = twonn(points)
    # For a 1D line embedded in 3D with noise, the TwoNN estimator will naturally overestimate 
    # the dimension compared to a perfect 1D line because the local neighborhood is a 3D cylinder. 
    # With this noise level we expect an estimate between 1.0 and 3.0.
    assert 1.0 <= res["id"] <= 3.0

def test_twonn_2d_plane():
    np.random.seed(42)
    # Generate points on a 2D plane embedded in 3D
    u = np.random.uniform(0, 10, 500)
    v = np.random.uniform(0, 10, 500)
    points = np.column_stack((u, v, u + v))
    res = twonn(points)
    assert 1.8 <= res["id"] <= 2.2

def test_twonn_3d_gaussian():
    np.random.seed(42)
    # Generate 3D Gaussian cloud
    points = np.random.normal(0, 1, (1000, 3))
    res = twonn(points)
    assert 2.7 <= res["id"] <= 3.3

def test_twonn_4d_gaussian():
    np.random.seed(42)
    # Generate 4D Gaussian cloud
    points = np.random.normal(0, 1, (2000, 4))
    res = twonn(points)
    assert 3.5 <= res["id"] <= 4.5
