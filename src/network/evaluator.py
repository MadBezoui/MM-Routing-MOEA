import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import networkx as nx

class PathMultimodalEvaluator:
    def __init__(self, G: nx.MultiDiGraph, survey, comfort_predictor):
        self.G = G
        self.survey = survey
        self.comfort_predictor = comfort_predictor
        
    def __call__(self, X: np.ndarray, profile: Dict[str, Any], extras: Dict[str, Any], scenario: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        n_samples = X.shape[0]
        
        travel_time_min = np.zeros(n_samples)
        cost = np.zeros(n_samples)
        emissions = np.zeros(n_samples)
        comfort_scores = np.zeros(n_samples)
        
        walk_distances = np.zeros(n_samples)
        bus_distances = np.zeros(n_samples)
        tram_distances = np.zeros(n_samples)
        transfers = np.zeros(n_samples)
        invalid_paths = np.zeros(n_samples)
        
        dominant_modes = []
        
        for i in range(n_samples):
            path = X[i, 0]
            
            p_time = 0.0
            p_walk_dist = 0.0
            p_bus_dist = 0.0
            p_tram_dist = 0.0
            p_transfers = 0
            
            if not path or len(path) < 2 or str(path[0]) != str(profile.get("origin_node")) or str(path[-1]) != str(profile.get("dest_node")):
                invalid_paths[i] = 1000.0
            else:
                for u, v in zip(path[:-1], path[1:]):
                    # Find shortest edge between u and v
                    edge_data = self.G.get_edge_data(u, v)
                    if not edge_data:
                        invalid_paths[i] = 1000.0
                        break
                        
                    # Take the first edge if multiple
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
                    elif mode == 'tram':
                        p_tram_dist += length
                    elif mode == 'transit':
                        p_bus_dist += length
                            
            # Estimate shares
            total_dist = max(p_walk_dist + p_bus_dist + p_tram_dist, 0.1)
            walk_share = p_walk_dist / total_dist
            bus_share = p_bus_dist / total_dist
            
            # Simple heuristic replaced with real models
            # 1.90 EUR flat ticket if any transit is used
            p_cost = 1.90 if (p_bus_dist > 0 or p_tram_dist > 0) else 0.0
            
            # Emissions: Bus = 80 gCO2/km, Tram = 35 gCO2/km. Converted to kgCO2e
            p_emissions = ((p_bus_dist * 80.0) + (p_tram_dist * 35.0)) / 1000.0
            
            travel_time_min[i] = p_time
            cost[i] = p_cost
            emissions[i] = p_emissions
            walk_distances[i] = p_walk_dist
            bus_distances[i] = p_bus_dist
            tram_distances[i] = p_tram_dist
            transfers[i] = p_transfers
            
            dominant = "walk" if walk_share > 0.5 else "bus"
            dominant_modes.append(dominant)
            
        total_dist = np.maximum(walk_distances + bus_distances + tram_distances, 0.1)
        walk_share = walk_distances / total_dist
        bus_share = bus_distances / total_dist
        
        comfort_df = pd.DataFrame({
            "walk_share": walk_share, 
            "bike_share": np.zeros(n_samples),
            "bus_share": bus_share, 
            "tram_share": tram_distances / total_dist,
            "car_share": np.zeros(n_samples),
            "crowding": np.ones(n_samples) * 0.5,
            "transfers": transfers, 
            "distance_km": total_dist,
            "rain": np.ones(n_samples) * profile.get("rain", 0), 
            "temperature_c": np.ones(n_samples) * float(profile.get("temperature_c", 14.0)),
            "age": np.ones(n_samples) * profile.get("age", 25.0), 
            "mobility_restriction": np.ones(n_samples) * profile.get("mobility_restriction", 0),
            "reliability_penalty": np.zeros(n_samples),
            "safety_penalty": np.ones(n_samples) * 0.1,
            "fare_eur": cost, 
            "travel_time_min": travel_time_min,
            "weather_label": np.where(np.ones(n_samples) * profile.get("rain", 0) > 0.5, "rainy", "dry"),
            "dominant_mode": dominant_modes,
        })
        
        comfort_score = self.comfort_predictor.predict(comfort_df, self.survey)
        
        budget = float(profile.get("budget_eur", 5.0))
        max_time = float(profile.get("max_time_min", 120.0))
        max_walk = float(profile.get("max_walk_km", self.survey.walking_threshold_km))
        
        g1_budget = np.maximum(0, (cost - budget) / max(budget, 0.1))
        g2_time = np.maximum(0, (travel_time_min - max_time) / max(max_time, 1.0))
        g3_walk = np.maximum(0, (walk_distances - max_walk) / max(max_walk, 0.1))
        g4_invalid = invalid_paths
        
        F = np.column_stack([travel_time_min, cost, emissions, 1.0 - comfort_score])
        G = np.column_stack([g1_budget, g2_time, g3_walk, g4_invalid])
        
        meta = {
            "dominant_mode": np.array(dominant_modes),
            "travel_time_min": travel_time_min,
            "cost": cost,
            "emissions": emissions,
            "comfort_score": comfort_score,
        }
        
        return F, G, meta
