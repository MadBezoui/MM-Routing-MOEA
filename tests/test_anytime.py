import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import json

from experiments.anytime_analysis import anytime_table

def test_anytime_analysis_variable_budgets(tmp_path):
    # Create synthetic history files mimicking different budgets and plans
    p1 = tmp_path / "plan_A" / "checkpoints" / "history"
    p2 = tmp_path / "plan_B" / "checkpoints" / "history"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)
    
    # Run 1 in plan_A: Terminal gen = 100
    df1 = pd.DataFrame({
        "generation": [25, 50, 75, 100],
        "hypervolume": [0.25, 0.50, 0.75, 1.0],
        "profile_id": ["prof1", "prof1", "prof1", "prof1"],
        "algorithm": ["algo1", "algo1", "algo1", "algo1"],
        "seed": [1, 1, 1, 1]
    })
    df1.to_csv(p1 / "prof1_algo1_1.csv", index=False)
    
    # Run 2 in plan_B: Same profile/algo/seed but Terminal gen = 200
    df2 = pd.DataFrame({
        "generation": [50, 100, 150, 200],
        "hypervolume": [0.4, 0.6, 0.8, 2.0],  # Terminal HV is 2.0
        "profile_id": ["prof1", "prof1", "prof1", "prof1"],
        "algorithm": ["algo1", "algo1", "algo1", "algo1"],
        "seed": [1, 1, 1, 1]
    })
    df2.to_csv(p2 / "prof1_algo1_1.csv", index=False)
    
    # Run 3 in plan_A: Terminal gen = 150 (different budget)
    df3 = pd.DataFrame({
        "generation": [50, 100, 150],
        "hypervolume": [1.0, 2.0, 4.0],  # Terminal HV is 4.0
        "profile_id": ["prof2", "prof2", "prof2"],
        "algorithm": ["algo1", "algo1", "algo1"],
        "seed": [2, 2, 2]
    })
    df3.to_csv(p1 / "prof2_algo1_2.csv", index=False)

    df_out = anytime_table([tmp_path / "plan_A", tmp_path / "plan_B"], fractions=[0.25, 0.5, 0.75, 1.0])
    
    # Expect 4 rows for algo1 (one for each fraction), where the mean is the average pct of terminal across 3 runs.
    # Run 1: 0.25 -> 25%, 0.5 -> 50%, 0.75 -> 75%, 1.0 -> 100%
    # Run 2: 0.25 (gen 50) -> 20%, 0.5 (gen 100) -> 30%, 0.75 (gen 150) -> 40%, 1.0 -> 100%
    # Run 3: 0.25 (gen 37.5). Interp between gen 0 and 50... wait, gen 50 is the first. 
    # Interp for run 3: 
    #   gen 50 (frac 0.33) -> HV 1.0
    #   gen 100 (frac 0.66) -> HV 2.0
    #   gen 150 (frac 1.0) -> HV 4.0
    # Frac 0.25 is extrapolated/interpolated as 1.0 since it's outside left bound? np.interp uses left bound.
    # So 0.25 -> 1.0 (25%), 0.5 -> ~1.5 (37.5%), 0.75 -> ~2.5 (62.5%), 1.0 -> 100%
    
    assert len(df_out) == 4
    assert df_out["count"].iloc[0] == 3
    assert (df_out["mean"].iloc[3] == 100.0) # 100% budget should always be exactly 100% of terminal HV
