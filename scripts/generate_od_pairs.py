import hashlib
import json
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path

from src.survey_data_loader import load_all
from src.pipeline_V6_smart import build_smart_plans

def hash_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_path_stats(G, path):
    p_time = 0.0
    p_walk_dist = 0.0
    p_bus_dist = 0.0
    p_tram_dist = 0.0
    p_transfers = 0
    has_transit = False
    
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data is None:
            return None
        d = edge_data[list(edge_data.keys())[0]]
        
        p_time += d.get('travel_time_sec', 60) / 60.0
        mode = d.get('mode', 'walk')
        length = d.get('length', 0) / 1000.0
        
        if mode == 'walk':
            p_walk_dist += length
        elif mode == 'transfer':
            p_walk_dist += length
            p_transfers += 1
        elif mode == 'bus':
            p_bus_dist += length
            has_transit = True
        elif mode == 'tram':
            p_tram_dist += length
            has_transit = True
        elif mode == 'transit':
            p_bus_dist += length
            has_transit = True
            
    p_cost = 1.90 if has_transit else 0.0
    BUS_GCO2E_PER_PASSENGER_KM = 80.0
    TRAM_GCO2E_PER_PASSENGER_KM = 35.0
    p_emissions = ((p_bus_dist * BUS_GCO2E_PER_PASSENGER_KM) + (p_tram_dist * TRAM_GCO2E_PER_PASSENGER_KM)) / 1000.0
    
    return {
        "time": p_time,
        "walk_dist": p_walk_dist,
        "cost": p_cost,
        "emissions": p_emissions,
        "transfers": p_transfers,
        "has_transit": has_transit
    }

def generate_od_pairs():
    survey_data = load_all(Path("data/survey_results"))
    profiles_df = survey_data.profiles.copy()
    plans = build_smart_plans(profiles_df)
    active_profile_ids = set()
    for plan in plans:
        for pid in plan.profiles_df['profile_id']:
            active_profile_ids.add(pid)
            
    print(f"Generating OD pairs for {len(active_profile_ids)} unique profiles.")
    
    graph_path = Path("data/processed/strasbourg_multimodal.graphml")
    graph_sha256 = hash_file(graph_path)
    G = nx.read_graphml(graph_path)
    
    # Largest weakly connected component
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    G_sub = G.subgraph(largest_wcc).copy()
    
    osm_nodes = sorted([n for n, d in G_sub.nodes(data=True) if d.get('type') != 'transit_stop'])
    
    rng = np.random.default_rng(1)
    records = []
    used_ods = set()
    
    # Verification profile
    VERIFICATION_PROFILE = "VERIFICATION_PROFILE"
    
    def find_valid_od(is_verification=False):
        max_attempts = 1000
        for _ in range(max_attempts):
            o, d = rng.choice(osm_nodes, 2, replace=False)
            if (o, d) in used_ods:
                continue
                
            try:
                # Need a path
                path = nx.shortest_path(G_sub, source=o, target=d, weight='travel_time_sec')
            except nx.NetworkXNoPath:
                continue
                
            if len(path) < 2:
                continue
                
            stats = get_path_stats(G_sub, path)
            if stats is None:
                continue
                
            if is_verification:
                if not stats["has_transit"]:
                    continue
                # verification limits: 120 min, 2 km walk
                if stats["time"] > 120.0 or stats["walk_dist"] > 2.0:
                    continue
                    
            used_ods.add((o, d))
            
            o_data = G_sub.nodes[o]
            d_data = G_sub.nodes[d]
            
            return {
                "origin_node": o,
                "dest_node": d,
                "origin_lon": float(o_data.get('x', 0)),
                "origin_lat": float(o_data.get('y', 0)),
                "destination_lon": float(d_data.get('x', 0)),
                "destination_lat": float(d_data.get('y', 0)),
                "baseline_path": json.dumps(path),
                "baseline_travel_time_min": stats["time"],
                "baseline_cost_eur": stats["cost"],
                "baseline_emissions_kgco2e": stats["emissions"],
                "baseline_walk_distance_km": stats["walk_dist"],
                "baseline_transfer_count": stats["transfers"],
                "generation_seed": 1,
                "graph_sha256": graph_sha256
            }
        raise RuntimeError("Could not find a valid OD pair after max attempts.")

    # Generate for verification profile
    verif_od = find_valid_od(is_verification=True)
    verif_od["profile_id"] = VERIFICATION_PROFILE
    records.append(verif_od)

    for pid in sorted(list(active_profile_ids)):
        od = find_valid_od(is_verification=False)
        od["profile_id"] = pid
        records.append(od)
        
    od_df = pd.DataFrame(records)
    
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "od_pairs.csv"
    od_df.to_csv(out_path, index=False)
    
    print(f"Successfully generated {len(od_df)} OD pairs and saved to {out_path}.")
    print(f"Graph SHA-256: {graph_sha256}")

if __name__ == "__main__":
    generate_od_pairs()
