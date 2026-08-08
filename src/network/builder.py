import json
import logging
import zipfile
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

import subprocess

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # radius of Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_multimodal_graph(osm_path: str, gtfs_path: str, out_path: str) -> None:
    """Builds an integrated multimodal graph from OSM XML/PBF and GTFS ZIP."""
    
    # If the input is PBF, convert it to XML first
    if osm_path.endswith(".pbf"):
        xml_path = osm_path.replace(".pbf", ".osm")
        logger.info(f"Converting {osm_path} to {xml_path} via osmium...")
        subprocess.run(["osmium", "cat", osm_path, "-o", xml_path, "--overwrite"], check=True)
        osm_path = xml_path
        
    logger.info(f"Loading OSM walking network from {osm_path}...")
    
    # We load walking paths from OSM
    G_walk = ox.graph_from_xml(osm_path, simplify=True, retain_all=True)
    
    # Convert all nodes to string IDs to avoid mixing ints and strings
    G_walk = nx.relabel_nodes(G_walk, {n: f"osm_{n}" for n in G_walk.nodes()})
    
    for u, v, k, d in G_walk.edges(keys=True, data=True):
        d['mode'] = 'walk'
        if 'length' not in d:
            d['length'] = 0.0
        # Assume walking speed of 5 km/h = 1.38 m/s
        d['travel_time_sec'] = float(d['length']) / 1.38

    logger.info(f"Loaded {G_walk.number_of_nodes()} OSM nodes and {G_walk.number_of_edges()} edges.")
    
    logger.info(f"Loading GTFS from {gtfs_path}...")
    with zipfile.ZipFile(gtfs_path) as z:
        stops = pd.read_csv(z.open('stops.txt'))
        routes = pd.read_csv(z.open('routes.txt'))
        trips = pd.read_csv(z.open('trips.txt'))
        stop_times = pd.read_csv(z.open('stop_times.txt'))
        
        # Only bus (3) and tram (0)
        valid_routes = routes[routes['route_type'].isin([0, 3])]
        valid_trips = trips[trips['route_id'].isin(valid_routes['route_id'])]
        valid_st = stop_times[stop_times['trip_id'].isin(valid_trips['trip_id'])]
        
        # Parse times
        valid_st['arr_time_sec'] = valid_st['arrival_time'].str.split(':').apply(
            lambda x: int(x[0])*3600 + int(x[1])*60 + int(x[2]) if type(x) == list else 0
        )
        valid_st['dep_time_sec'] = valid_st['departure_time'].str.split(':').apply(
            lambda x: int(x[0])*3600 + int(x[1])*60 + int(x[2]) if type(x) == list else 0
        )
        
        # Filter for AM peak (e.g. 7 AM to 10 AM)
        am_st = valid_st[(valid_st['dep_time_sec'] >= 7 * 3600) & (valid_st['dep_time_sec'] <= 10 * 3600)]
        am_trips = am_st['trip_id'].unique()
        
        # Add transit nodes
        transit_stops = stops[stops['stop_id'].isin(am_st['stop_id'])]
        
        G = nx.MultiDiGraph(G_walk)
        
        logger.info(f"Adding {len(transit_stops)} transit stops...")
        stop_ids = []
        stop_coords = []
        
        for _, row in transit_stops.iterrows():
            nid = f"gtfs_{row['stop_id']}"
            G.add_node(nid, y=row['stop_lat'], x=row['stop_lon'], stop_name=row['stop_name'], type='transit_stop')
            stop_ids.append(nid)
            stop_coords.append((row['stop_lat'], row['stop_lon']))
            
        # Add transit edges (average travel time between stops in AM peak)
        am_st = am_st.sort_values(['trip_id', 'stop_sequence'])
        am_st['next_stop'] = am_st.groupby('trip_id')['stop_id'].shift(-1)
        am_st['travel_time'] = am_st.groupby('trip_id')['arr_time_sec'].shift(-1) - am_st['dep_time_sec']
        
        # Merge route_type into trips to identify mode
        am_trips_df = trips[['trip_id', 'route_id']].merge(valid_routes[['route_id', 'route_type']], on='route_id')
        am_st = am_st.merge(am_trips_df[['trip_id', 'route_type']], on='trip_id', how='left')
        
        edges_df = am_st.dropna(subset=['next_stop']).groupby(['stop_id', 'next_stop', 'route_type']).agg(
            avg_travel_time=('travel_time', 'mean'),
            freq=('trip_id', 'count')
        ).reset_index()
        
        # Build coordinates dictionary for fast distance computation
        stop_coords_dict = {row['stop_id']: (row['stop_lat'], row['stop_lon']) for _, row in transit_stops.iterrows()}
        
        for _, row in edges_df.iterrows():
            u = f"gtfs_{row['stop_id']}"
            v = f"gtfs_{row['next_stop']}"
            
            # Distance computation using haversine
            if row['stop_id'] in stop_coords_dict and row['next_stop'] in stop_coords_dict:
                lat1, lon1 = stop_coords_dict[row['stop_id']]
                lat2, lon2 = stop_coords_dict[row['next_stop']]
                dist_m = haversine(lon1, lat1, lon2, lat2)
            else:
                dist_m = 500.0 # fallback
                
            # Mode identification
            mode_str = "tram" if row['route_type'] == 0 else "bus"
            
            # Add average waiting time of (3 hours / freq) / 2
            wait_time = (3 * 3600 / max(1, row['freq'])) / 2
            
            G.add_edge(u, v, mode=mode_str, travel_time_sec=row['avg_travel_time'] + wait_time, freq=row['freq'], length=dist_m)
            
        logger.info(f"Added {len(edges_df)} transit edges.")
        
        # Create transfer edges using KDTree
        logger.info("Building transfer edges between OSM and GTFS...")
        osm_nodes = [n for n, d in G.nodes(data=True) if 'type' not in d or d['type'] != 'transit_stop']
        osm_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in osm_nodes]
        
        if osm_coords and stop_coords:
            tree = KDTree(osm_coords)
            # Find closest OSM node for each transit stop
            dists, idxs = tree.query(stop_coords, k=1)
            
            for stop_nid, dist, idx in zip(stop_ids, dists, idxs):
                osm_nid = osm_nodes[idx]
                
                # dist is roughly in degrees. Let's convert to meters roughly (1 deg ~ 111km)
                dist_m = dist * 111000
                walk_time = dist_m / 1.38
                
                G.add_edge(osm_nid, stop_nid, mode='transfer', travel_time_sec=walk_time, length=dist_m)
                G.add_edge(stop_nid, osm_nid, mode='transfer', travel_time_sec=walk_time, length=dist_m)
                
        logger.info("Stringifying attributes for GraphML compatibility...")
        valid_types = (int, float, str, bool)
        for n, d in G.nodes(data=True):
            for k, v in d.items():
                if not isinstance(v, valid_types):
                    d[k] = str(v)
        
        for u, v, k, d in G.edges(keys=True, data=True):
            for key, val in d.items():
                if not isinstance(val, valid_types):
                    d[key] = str(val)
                    
        logger.info("Saving integrated graph...")
        nx.write_graphml(G, out_path)
        logger.info(f"Graph saved to {out_path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    osm_path = "data/raw/strasbourg.osm.pbf"
    gtfs_path = "data/raw/strasbourg_gtfs.zip"
    out_path = "data/processed/strasbourg_multimodal.graphml"
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    build_multimodal_graph(osm_path, gtfs_path, out_path)
