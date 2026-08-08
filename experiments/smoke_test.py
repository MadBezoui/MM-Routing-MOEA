"""smoke_test.py
===================
Executes a small end-to-end smoke campaign to verify that execution and
analytics generation work correctly with ProcessPoolExecutor.
"""

import logging
import argparse
from pathlib import Path
from src.config import ScenarioConfig, POPULATION_SIZES
from experiments._common import build_context, setup_logging
from src.optimization_framework_parallel3 import run_algorithm_suite_parallel3_checkpointed
import pandas as pd

logger = logging.getLogger(__name__)

def run_smoke_test(out_dir: str):
    out = Path(out_dir)
    # 3 profiles
    ctx = build_context(
        n_profiles=3,
        survey_dir="data/survey_results",
        graph_path="data/processed/strasbourg_multimodal.graphml",
        output_dir=str(out),
        comfort_model="mlp_surrogate",
        random_state=42
    )

    profiles_dicts = [p.to_dict() for _, p in ctx.profiles.iterrows()]

    # Configure small population for smoke test
    POPULATION_SIZES["smoke"] = {"nsga2": 20, "nsga3": 20}

    logger.info("Running smoke test optimization...")
    # 2 algorithms, 2 seeds, 10 generations, process executor
    pop, hist = run_algorithm_suite_parallel3_checkpointed(
        problem_factory=ctx.problem_factory,
        profiles=profiles_dicts,
        scenario=ScenarioConfig(),
        output_dir=str(out / "smoke_runs"),
        algorithms=["nsga2", "nsga3"],
        seeds=[1, 2],
        n_generations=10,
        plan="smoke",
        n_partitions=1,
        max_workers=2,
        executor_backend="process",
        encoding="path",
        instrumented=True,
        show_progress=True,
        reference_point_factory=ctx.ref_point_factory,
        stabilized_weights=ctx.stabilized,
        raw_weights=ctx.raw,
    )
    
    logger.info("Optimization complete.")
    logger.info("Found %d final solutions in population frame.", len(pop))
    logger.info("Found %d history records.", len(hist))

    # Verify no failures
    assert len(pop) > 0, "No population generated!"
    assert len(hist) > 0, "No history generated!"

    # Verify provenance and columns
    expected_cols = ["feasible_ratio", "n_feasible", "hypervolume", "algorithm", "seed", "profile_id"]
    for c in expected_cols:
        assert c in hist.columns, f"Missing {c} in history!"
        
    logger.info("Smoke test passed successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/smoke_test")
    args = parser.parse_args()
    setup_logging()
    run_smoke_test(args.out)

if __name__ == "__main__":
    main()
