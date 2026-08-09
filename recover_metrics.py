import logging
from pathlib import Path
from src.pipeline_V6_smart import recover_hv_igd_for_plan

logging.basicConfig(level=logging.INFO)
out_dir = Path("results/outputs_main")
for plan in ["main_pi_nsga3_vs_nsga2_150profiles", "extended_benchmark_30profiles", "convergence_curves_10profiles", "four_way_ablation_30profiles"]:
    p = out_dir / plan
    if not p.exists(): continue
    print(f"Recovering {plan}...")
    recover_hv_igd_for_plan(p, out_dir, survey_nadir=None)
    print(f"Done {plan}")
