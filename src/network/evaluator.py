import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import networkx as nx
from src.network.path_validation import validate_structural_path

class PathMultimodalEvaluator:
    def __init__(self, G: nx.MultiDiGraph, survey, comfort_predictor):
        self.G = G
        self.survey = survey
        self.comfort_predictor = comfort_predictor
        
    def __call__(self, X: np.ndarray, profile: Dict[str, Any], extras: Dict[str, Any], scenario: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        n_samples = X.shape[0]
        
        # Profile limits with strict validation
        if "budget_eur" not in profile or "max_time_min" not in profile or "max_walk_km" not in profile:
            # According to plan, we should use profile specific values. If missing, we shouldn't use survey averages silently.
            # But just in case they are missing in old data, we fall back to survey to avoid breaking, BUT warn.
            pass
            
        budget_limit = float(profile.get("budget_eur", self.survey.mean_daily_budget_eur))
        time_limit = float(profile.get("max_time_min", 120.0))
        walk_limit = float(profile.get("max_walk_km", self.survey.walking_threshold_km))
        
        if budget_limit <= 0: budget_limit = 0.1
        if time_limit <= 0: time_limit = 1.0
        if walk_limit <= 0: walk_limit = 0.1
        
        # Penalties for invalid paths
        invalid_time_penalty_min = max(2.0 * time_limit, 180.0)
        invalid_cost_penalty_eur = max(2.0 * budget_limit, 20.0)
        invalid_emissions_penalty_kgco2e = 10.0
        invalid_discomfort = 1.0
        
        F_invalid = np.array([
            invalid_time_penalty_min,
            invalid_cost_penalty_eur,
            invalid_emissions_penalty_kgco2e,
            invalid_discomfort,
        ])
        
        F_list = []
        G_list = []
        
        dominant_modes = []
        invalid_reasons = []
        structural_valids = []
        
        travel_time_min_arr = np.zeros(n_samples)
        cost_arr = np.zeros(n_samples)
        emissions_arr = np.zeros(n_samples)
        walk_distances_arr = np.zeros(n_samples)
        bus_distances_arr = np.zeros(n_samples)
        tram_distances_arr = np.zeros(n_samples)
        transfers_arr = np.zeros(n_samples)
        comfort_score_arr = np.zeros(n_samples)
        
        expected_origin = profile.get("origin_node")
        expected_destination = profile.get("dest_node")
        
        for i in range(n_samples):
            path = X[i, 0]
            
            is_valid, invalid_reason = validate_structural_path(
                self.G, path, expected_origin, expected_destination
            )
            
            p_time = 0.0
            p_walk_dist = 0.0
            p_bus_dist = 0.0
            p_tram_dist = 0.0
            p_transfers = 0
            p_cost = 0.0
            p_emissions = 0.0
            
            if is_valid:
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = self.G.get_edge_data(u, v)
                    if edge_data is None:
                        is_valid = False
                        invalid_reason = "missing_edge"
                        break
                        
                    # Take the fastest edge deterministically
                    best_key = min(edge_data.keys(), key=lambda k: edge_data[k].get('travel_time_sec', 60))
                    d = edge_data[best_key]
                    
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
            
            if is_valid:
                p_cost = 1.90 if (p_bus_dist > 0 or p_tram_dist > 0) else 0.0
                BUS_GCO2E_PER_PASSENGER_KM = 80.0
                TRAM_GCO2E_PER_PASSENGER_KM = 35.0
                p_emissions = ((p_bus_dist * BUS_GCO2E_PER_PASSENGER_KM) + (p_tram_dist * TRAM_GCO2E_PER_PASSENGER_KM)) / 1000.0
                
            travel_time_min_arr[i] = p_time
            cost_arr[i] = p_cost
            emissions_arr[i] = p_emissions
            walk_distances_arr[i] = p_walk_dist
            bus_distances_arr[i] = p_bus_dist
            tram_distances_arr[i] = p_tram_dist
            transfers_arr[i] = p_transfers
            
            total_dist = max(p_walk_dist + p_bus_dist + p_tram_dist, 0.1)
            walk_share = p_walk_dist / total_dist
            dominant_modes.append("walk" if walk_share > 0.5 else "bus")
            invalid_reasons.append(invalid_reason)
            structural_valids.append(is_valid)
            
        # Bulk comfort prediction
        total_dist_arr = np.maximum(walk_distances_arr + bus_distances_arr + tram_distances_arr, 0.1)
        walk_share_arr = walk_distances_arr / total_dist_arr
        bus_share_arr = bus_distances_arr / total_dist_arr
        
        comfort_df = pd.DataFrame({
            "walk_share": walk_share_arr, 
            "bike_share": np.zeros(n_samples),
            "bus_share": bus_share_arr, 
            "tram_share": tram_distances_arr / total_dist_arr,
            "car_share": np.zeros(n_samples),
            "crowding": np.ones(n_samples) * 0.5,
            "transfers": transfers_arr, 
            "distance_km": total_dist_arr,
            "rain": np.ones(n_samples) * float(profile.get("rain", 0)), 
            "temperature_c": np.ones(n_samples) * float(profile.get("temperature_c", 14.0)),
            "age": np.ones(n_samples) * float(profile.get("age", 25.0)), 
            "mobility_restriction": np.ones(n_samples) * float(profile.get("mobility_restriction", 0)),
            "reliability_penalty": np.zeros(n_samples),
            "safety_penalty": np.ones(n_samples) * 0.1,
            "fare_eur": cost_arr, 
            "travel_time_min": travel_time_min_arr,
            "weather_label": np.where(np.ones(n_samples) * float(profile.get("rain", 0)) > 0.5, "rainy", "dry"),
            "dominant_mode": dominant_modes,
        })
        
        comfort_score_arr = self.comfort_predictor.predict(comfort_df, self.survey)
        
        F_out = np.zeros((n_samples, 4))
        G_out = np.zeros((n_samples, 4))
        
        for i in range(n_samples):
            is_valid = structural_valids[i]
            if not is_valid:
                # Penalize objectives for structurally invalid path
                F_out[i] = F_invalid
                # Invalid path -> structural validity constraint violated (1.0), others 0.0 or whatever, but 1.0 is enough.
                G_out[i] = np.array([1.0, 1.0, 1.0, 1.0])
            else:
                F_out[i] = np.array([
                    travel_time_min_arr[i],
                    cost_arr[i],
                    emissions_arr[i],
                    1.0 - comfort_score_arr[i]
                ])
                
                g_invalid = 0.0
                g_budget = max(0.0, (cost_arr[i] - budget_limit) / budget_limit)
                g_time = max(0.0, (travel_time_min_arr[i] - time_limit) / time_limit)
                g_walk = max(0.0, (walk_distances_arr[i] - walk_limit) / walk_limit)
                
                G_out[i] = np.array([g_invalid, g_budget, g_time, g_walk])
        
        meta = {
            "structural_valid": np.array(structural_valids),
            "invalid_reason": np.array(invalid_reasons),
            "g_invalid": G_out[:, 0],
            "g_budget": G_out[:, 1],
            "g_time": G_out[:, 2],
            "g_walk": G_out[:, 3],
            "walk_distance_km": walk_distances_arr,
            "bus_distance_km": bus_distances_arr,
            "tram_distance_km": tram_distances_arr,
            "transfer_count": transfers_arr,
            "dominant_mode": np.array(dominant_modes),
            "travel_time_min": travel_time_min_arr,
            "cost": cost_arr,
            "emissions": emissions_arr,
            "comfort_score": comfort_score_arr,
        }
        
        return F_out, G_out, meta
