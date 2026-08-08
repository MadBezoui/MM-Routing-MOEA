import pandas as pd
import json
from scipy.stats import wilcoxon

def main():
    raw = pd.read_csv("outputs_popsize_equalization/equalization_raw.csv")
    
    # Split the raw data
    sweep = raw[raw["experiment"] == "nsga2_sweep"].copy()
    eq = raw[raw["experiment"] == "equalization"].copy()
    
    # 1. Sweep Outputs
    sweep.to_csv("outputs_popsize_equalization/popsize_sweep_raw.csv", index=False)
    sweep_profile = sweep.groupby(["profile_id", "algorithm", "pop_size", "config"])["normalized_hv"].mean().reset_index()
    sweep_profile.to_csv("outputs_popsize_equalization/popsize_sweep_profile_level.csv", index=False)
    sweep_summary = sweep_profile.groupby(["algorithm", "pop_size", "config"])["normalized_hv"].agg(["mean", "std", "count"]).reset_index()
    sweep_summary.to_csv("outputs_popsize_equalization/popsize_sweep_summary.csv", index=False)
    
    # 2. Equalization Outputs
    eq.to_csv("outputs_popsize_equalization/equalization_run_level.csv", index=False)
    eq_profile = eq.groupby(["profile_id", "algorithm", "pop_size", "config"])["normalized_hv"].mean().reset_index()
    eq_profile.to_csv("outputs_popsize_equalization/equalization_profile_level.csv", index=False)
    eq_summary = eq_profile.groupby(["algorithm", "pop_size", "config"])["normalized_hv"].agg(["mean", "std", "count"]).reset_index()
    eq_summary.to_csv("outputs_popsize_equalization/equalization_summary.csv", index=False)
    
    # 3. Equalization Statistics JSON
    stats_dict = {}
    # Compare nsga2 vs nsga3 at equalized popsize (170)
    nsga2_eq = eq_profile[(eq_profile["algorithm"] == "nsga2") & (eq_profile["config"] == "equalized")].set_index("profile_id")["normalized_hv"]
    nsga3_eq = eq_profile[(eq_profile["algorithm"] == "nsga3") & (eq_profile["config"] == "equalized")].set_index("profile_id")["normalized_hv"]
    
    # Align
    common = nsga2_eq.index.intersection(nsga3_eq.index)
    nsga2_eq = nsga2_eq.loc[common]
    nsga3_eq = nsga3_eq.loc[common]
    
    diff = nsga2_eq - nsga3_eq
    dz = diff.mean() / diff.std(ddof=1) if diff.std() > 0 else 0.0
    w, p = wilcoxon(nsga2_eq, nsga3_eq, zero_method="wilcox", alternative="two-sided")
    
    stats_dict = {
        "nsga2_equalized_mean": float(nsga2_eq.mean()),
        "nsga3_equalized_mean": float(nsga3_eq.mean()),
        "mean_difference": float(diff.mean()),
        "d_z_profile_level": float(dz),
        "wilcoxon_w": float(w),
        "wilcoxon_p_value": float(p),
        "nsga2_wins": int((diff > 0).sum()),
        "nsga3_wins": int((diff < 0).sum()),
        "sign_convention": "NSGA-II - NSGA-III (positive means NSGA-II is better)"
    }
    
    with open("outputs_popsize_equalization/equalization_statistics.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    # 4. Config JSON
    config_dict = {
        "n_profiles_equalization": 30,
        "n_seeds_equalization": 10,
        "n_profiles_sweep": 10,
        "n_seeds_sweep": 5,
        "generations": 150
    }
    with open("outputs_popsize_equalization/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

if __name__ == "__main__":
    main()
