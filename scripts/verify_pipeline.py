import logging
import argparse
import networkx as nx
from pathlib import Path
import random
import sys

from src.pipeline_V6_smart import (
    build_smart_plans, build_problem_factory, build_reference_point_factory,
    execute_plan, recover_hv_igd_for_plan, audit_and_stabilize_weights,
    TrainedComfortPredictor
)
from src.survey_data_loader import load_all
from src.comfort_models import SurveyInformedComfortFactory
from src.config import ScenarioConfig, ComfortTrainingConfig

def run_verification():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VERIFY")
    
    out_dir = Path("outputs_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading survey...")
    survey_data = load_all(Path("data/survey_results"))
    real_survey = survey_data.calibration
    real_training_df = survey_data.comfort_training.copy()
    profiles_df = survey_data.profiles.copy().head(1) # just 1 profile!
    
    plans = build_smart_plans(profiles_df, sms_seeds=1)
    
    # We only take the first plan, restrict seeds and generations
    plan = plans[0]
    plan.name = "verification_plan"
    plan.profiles_df = profiles_df
    plan.seeds_by_algorithm = {
        "nsga2": [0],
        "pi_nsga3_stab": [0],
        "canonical_nsga3": [0]
    }
    plan.n_generations = 20
    plan.population_size = 12
    
    logger.info("Loading graph...")
    G = nx.read_graphml("data/processed/strasbourg_multimodal.graphml")
    osm_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != 'transit_stop']
    
    for idx, row in plan.profiles_df.iterrows():
        plan.profiles_df.at[idx, 'origin_node'] = random.choice(osm_nodes)
        plan.profiles_df.at[idx, 'dest_node'] = random.choice(osm_nodes)
        
    logger.info("Training comfort models...")
    comfort_cfg = ComfortTrainingConfig()
    comfort_factory = SurveyInformedComfortFactory(comfort_cfg, real_survey)
    comfort_factory.generate_synthetic_dataset = lambda n_samples=None, seed=None: real_training_df.copy()
    comfort_results = comfort_factory.train_models(real_training_df)
    comfort_predictor = TrainedComfortPredictor(comfort_results)
    
    problem_factory = build_problem_factory(real_survey, comfort_predictor, evaluator_type="discrete", G=G)
    weight_audit = audit_and_stabilize_weights({"time": 0.35, "cost": 0.25, "emissions": 0.15, "comfort": 0.25})
    priority_weights = [
        weight_audit.stabilized_weights["time"],
        weight_audit.stabilized_weights["cost"],
        weight_audit.stabilized_weights["emissions"],
        weight_audit.stabilized_weights["comfort"],
    ]
    
    ref_point_factory = build_reference_point_factory(real_survey)
    scenario = ScenarioConfig(G=G, origins=osm_nodes, destinations=osm_nodes)
    
    logger.info("Running verification execution...")
    execute_plan(
        plan=plan,
        problem_factory=problem_factory,
        scenario=scenario,
        output_path=out_dir,
        priority_weights=priority_weights,
        ref_point_factory=ref_point_factory,
        max_workers=1,
        plan_type="verification_plan",
    )
    
    import pandas as pd
    pop_file = out_dir / plan.name / "checkpoints" / "population" / "STU_0001__nsga2__seed0.csv"
    if pop_file.exists():
        df = pd.read_csv(pop_file)
        assert df["feasible"].any(), "No feasible solutions found in verification!"
        assert (df["obj_2"] > 0).any() or (df["obj_3"] > 0).any(), "No multimodal solutions found (all walk)!"
        
    logger.info("Verification successful!")

if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    run_verification()
