import random
import networkx as nx
import pandas as pd
from pathlib import Path

from src.survey_data_loader import load_all
from src.pipeline_V6_smart import build_smart_plans

def generate_od_pairs():
    # 1. Load profiles (which will eventually be all plans)
    survey_data = load_all(Path("data/survey_results"))
    profiles_df = survey_data.profiles.copy()
    
    # 2. Extract which profiles actually participate in the plans
    plans = build_smart_plans(profiles_df)
    active_profile_ids = set()
    for plan in plans:
        for pid in plan.profiles_df['profile_id']:
            active_profile_ids.add(pid)
            
    print(f"Generating OD pairs for {len(active_profile_ids)} unique profiles.")
    
    # 3. Load valid origin/destination nodes from graph
    G = nx.read_graphml("data/processed/strasbourg_multimodal.graphml")
    osm_nodes = [n for n, d in G.nodes(data=True) if d.get('type') != 'transit_stop']
    
    # 4. Generate pairs deterministically
    random.seed(42) # HARDCODED SEED for reproducibility
    records = []
    
    # Ensure stable sorting so assignments don't shift
    for pid in sorted(list(active_profile_ids)):
        o = random.choice(osm_nodes)
        d = random.choice(osm_nodes)
        # Prevent trivial O=D (though rare in a big graph)
        while d == o:
            d = random.choice(osm_nodes)
            
        records.append({
            "profile_id": pid,
            "origin_node": o,
            "dest_node": d
        })
        
    od_df = pd.DataFrame(records)
    
    # 5. Save the output
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "od_pairs.csv"
    od_df.to_csv(out_path, index=False)
    
    print(f"Successfully generated {len(od_df)} OD pairs and saved to {out_path}.")

if __name__ == "__main__":
    generate_od_pairs()
