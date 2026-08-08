import logging
import argparse
import networkx as nx
import pandas as pd
from pathlib import Path
import sys
import json
import hashlib
import subprocess

from src.pipeline_V6_smart import (
    build_smart_plans, build_problem_factory, build_reference_point_factory,
    execute_plan, recover_hv_igd_for_plan, audit_and_stabilize_weights,
    TrainedComfortPredictor
)
from src.survey_data_loader import load_all
from src.comfort_models import SurveyInformedComfortFactory
from src.config import ScenarioConfig, ComfortTrainingConfig

logger = logging.getLogger(__name__)

def hash_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def run_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey-dir", type=str, default="data/survey_results")
    parser.add_argument("--graph-path", type=str, default="data/processed/strasbourg_multimodal.graphml")
    parser.add_argument("--out-dir", type=str, default="outputs_v6_smart")
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading survey data from {args.survey_dir}...")
    survey_data = load_all(Path(args.survey_dir))
    real_survey = survey_data.calibration
    real_training_df = survey_data.comfort_training.copy()
    profiles_df = survey_data.profiles.copy()
    
    logger.info("Generating synthetic plans...")
    plans = build_smart_plans(profiles_df, sms_seeds=5)
    
    logger.info(f"Loading multimodal graph from {args.graph_path}...")
    graph_path = Path(args.graph_path)
    G = nx.read_graphml(graph_path)
    graph_sha256 = hash_file(graph_path)
    
    logger.info("Assigning Origin and Destination nodes to profiles...")
    od_path = Path("data/processed/od_pairs.csv")
    if not od_path.exists():
        raise FileNotFoundError(f"OD pairs file not found at {od_path}. Run scripts/generate_od_pairs.py first.")
        
    od_df = pd.read_csv(od_path)
    od_dict = od_df.set_index('profile_id').to_dict(orient='index')
    od_sha256 = hash_file(od_path)
    
    for plan in plans:
        for idx, row in plan.profiles_df.iterrows():
            pid = row['profile_id']
            if pid not in od_dict:
                raise ValueError(f"Requested profile {pid} has no OD in {od_path}.")
            
            o = str(od_dict[pid]['origin_node'])
            d = str(od_dict[pid]['dest_node'])
            if o not in G.nodes:
                raise ValueError(f"Origin node {o} not found in graph.")
            if d not in G.nodes:
                raise ValueError(f"Destination node {d} not found in graph.")
                
            plan.profiles_df.at[idx, 'origin_node'] = o
            plan.profiles_df.at[idx, 'dest_node'] = d
            
    # Check if the graph hash matches the one used to generate ODs
    if 'graph_sha256' in od_df.columns:
        expected_hash = od_df['graph_sha256'].iloc[0]
        if expected_hash and expected_hash != graph_sha256:
            raise ValueError(f"Graph hash mismatch! OD pairs were generated for {expected_hash}, but current graph is {graph_sha256}.")

    logger.info("Training comfort models...")
    comfort_cfg = ComfortTrainingConfig()
    comfort_factory = SurveyInformedComfortFactory(comfort_cfg, real_survey)
    comfort_factory.generate_synthetic_dataset = lambda n_samples=None, seed=None: real_training_df.copy()
    comfort_results = comfort_factory.train_models(real_training_df)
    comfort_predictor = TrainedComfortPredictor(comfort_results)
    
    problem_factory = build_problem_factory(real_survey, comfort_predictor, evaluator_type="discrete", G=G)
    
    # Load actual stabilized weights
    raw_weights = survey_data.objective_weights
    weight_audit = audit_and_stabilize_weights(raw_weights)
    priority_weights = [
        weight_audit.stabilized_weights["time"],
        weight_audit.stabilized_weights["cost"],
        weight_audit.stabilized_weights["emissions"],
        weight_audit.stabilized_weights["comfort"],
    ]
    
    ref_point_factory = build_reference_point_factory(real_survey)
    osm_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != 'transit_stop']
    scenario = ScenarioConfig(G=G, origins=osm_nodes, destinations=osm_nodes)
    
    logger.info("Starting Plan Executions (Discrete Path Routing Mode)...")
    for plan in plans:
        logger.info(f"Executing plan {plan.name}")
        
        execute_plan(
            plan=plan,
            problem_factory=problem_factory,
            scenario=scenario,
            output_path=out_dir,
            priority_weights=priority_weights,
            ref_point_factory=ref_point_factory,
            max_workers=args.max_workers,
            plan_type=plan.name,
        )
        
        logger.info(f"Recovering metrics for {plan.name}...")
        recover_hv_igd_for_plan(plan_dir=out_dir / plan.name, output_dir=out_dir)
        
        # Persist run metadata
        metadata = {
            "experiment_schema_version": 2,
            "git_commit": get_git_commit(),
            "graph_sha256": graph_sha256,
            "od_pairs_sha256": od_sha256,
            "algorithms": list(plan.seeds_by_algorithm.keys()),
            "population_size": plan.population_size,
            "generations": plan.n_generations,
            "objective_units": {
                "time": "min",
                "cost": "EUR",
                "emissions": "kgCO2e",
                "discomfort": "dimensionless"
            },
            "constraint_order": [
                "structural_validity",
                "budget",
                "travel_time",
                "walking_distance"
            ]
        }
        with open(out_dir / plan.name / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    logger.info("All experiments completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.setrecursionlimit(5000)
    run_experiments()
