import os
import pandas as pd
import pytest
from pathlib import Path
import json
import numpy as np

from src.optimization_framework_parallel3 import run_algorithm_suite_parallel3_checkpointed
from src.config import ScenarioConfig

from pymoo.problems.single.sphere import Sphere
from src.optimization_framework_parallel3 import ProfiledMultimodalProblem
import numpy as np

def _evaluator(X, profile, extras, scenario):
    out = {}
    Sphere(n_var=2)._evaluate(X, out)
    return out["F"], np.zeros((len(X), 1)), {}

def _factory(profile, scenario, seed=0):
    return ProfiledMultimodalProblem(
        n_var=2, n_obj=1, xl=[0, 0], xu=[1, 1],
        evaluator=_evaluator, profile=profile,
        extras={}, scenario=scenario
    )

def test_sequential_vs_parallel_reproducibility(tmp_path):
    """
    Test that ProcessPoolExecutor with pymoo 0.6.1.1 produces bit-identical
    results as sequential execution (max_workers=1) when given the same seeds,
    proving that no global RNG state is leaking.
    """
    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"

    profiles = [{"profile_id": "P1"}, {"profile_id": "P2"}, {"profile_id": "P3"}]
    seeds = [1, 2, 3]
    
    run_algorithm_suite_parallel3_checkpointed(
        problem_factory=_factory,
        profiles=profiles,
        scenario=ScenarioConfig(),
        output_dir=str(out_seq),
        algorithms=["nsga2"],
        seeds=seeds,
        n_generations=5,
        plan="main",
        n_partitions=1,
        max_workers=1,
        executor_backend="sequential",
        encoding="float",
        instrumented=False,
        show_progress=False
    )
    
    # Parallel
    run_algorithm_suite_parallel3_checkpointed(
        problem_factory=_factory,
        profiles=profiles,
        scenario=ScenarioConfig(),
        output_dir=str(out_par),
        algorithms=["nsga2"],
        seeds=seeds,
        n_generations=5,
        plan="main",
        n_partitions=1,
        max_workers=2,
        executor_backend="process",
        encoding="float",
        instrumented=False,
        show_progress=False
    )
    
    # Compare outputs
    for p in profiles:
        for s in seeds:
            f1 = out_seq / "checkpoints/population" / f"{p['profile_id']}__nsga2__seed{s}.csv"
            f2 = out_par / "checkpoints/population" / f"{p['profile_id']}__nsga2__seed{s}.csv"
            
            df1 = pd.read_csv(f1).drop(columns=["runtime_sec"], errors="ignore")
            df2 = pd.read_csv(f2).drop(columns=["runtime_sec"], errors="ignore")
            
            pd.testing.assert_frame_equal(df1, df2)
