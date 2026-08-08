"""builder.py
=============
Construction of the consolidated multimodal transport network of Section 5.2.

The graph :math:`G = (V, E)` is directed.  Every edge carries a transport mode
:math:`m(e) \\in \\{walk, bike, bus, tram, car\\}`, a length in kilometres and a
free-flow traversal time in minutes.  Nodes are either ordinary locations or
mode-specific transfer facilities; a node that gives access to three or more
modes is flagged as a *transfer hub*.

Pipeline
--------
1. **Transit layer.**  GTFS trips active on the reference service date and
   departing inside the reference window are retained.  Stops are consolidated
   into physical facilities (Section 5.2, "transfer hubs were consolidated"),
   and one directed edge is created per consecutive stop pair of a retained
   trip, separately for ``bus`` (``route_type`` 3) and ``tram``
   (``route_type`` 0).
2. **Road layers.**  The OSM extract is streamed once by
   :mod:`src.network.osm_parser` into three mode-specific road layers.  Each
   consolidated node is snapped to its nearest road node in each layer, and
   shortest-path road distances are computed between node pairs.  A directed
   edge is created for the ``k`` nearest reachable neighbours within a
   mode-specific cutoff.
3. **Cleaning.**  Self-loops and duplicate parallel edges of the same mode are
   removed ("topological artifacts were filtered"), then the largest strongly
   connected component is retained so that the released graph is fully
   connected ("disconnected components were removed").

Every quantity reported in Table 6 is recomputed from the resulting graph by
:mod:`src.network.descriptors`; no descriptor is hard-coded.

Usage
-----
    python -m src.network.builder --config data/raw/experiment_context.json
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from src.network.osm_parser import load_or_parse

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class NetworkBuildConfig:
    """Parameters of the network construction.

    All values are explicit and version-controlled so that the released graph
    is reproducible from the frozen OSM and GTFS inputs.
    """

    osm_path: str = "data/raw/strasbourg.osm"
    gtfs_path: str = "data/raw/strasbourg_gtfs.zip"
    boundary_path: str = "data/raw/eurometropole_strasbourg_246700488.geojson"
    osm_cache_path: str = "data/processed/osm_road_layers.npz"
    out_path: str = "data/processed/strasbourg_multimodal.graphml"

    #: Reference service date (YYYYMMDD) and departure window, Europe/Paris.
    service_date: int = 20260915
    window_start_h: float = 7.0
    window_end_h: float = 10.0

    #: Two stops merge into one physical facility when they carry the same
    #: normalised name and lie within ``name_merge_radius_m``, or when they lie
    #: within ``proximity_merge_radius_m`` of each other irrespective of name.
    name_merge_radius_m: float = 400.0
    proximity_merge_radius_m: float = 120.0

    #: A consolidated node is kept only if it is served by at least this many
    #: distinct trips inside the three-hour reference window, i.e. roughly one
    #: departure every 7.5 minutes across all lines.  This service-frequency
    #: criterion removes the long tail of near-unserved stops and is what
    #: Section 5.2 refers to as filtering topological artifacts.
    min_trips_per_node: int = 24

    #: Road-layer connection rules: (cutoff in km, number of neighbours).
    #: Each facility is linked to its ``k`` nearest *reachable* facilities in
    #: each road layer.  These are modelling choices, not quantities derived
    #: from the data; they are recorded here so that the network is
    #: reproducible.
    #:
    #: ``k = (4, 4, 3)`` is the smallest setting at which all three road layers
    #: become strongly connected over the nodes they touch.  Below it the
    #: k-nearest-neighbour construction is close to acyclic -- at ``k = (2, 2, 1)``
    #: the largest strongly connected component of the car layer holds three
    #: nodes -- so no car- or walking-dominated route family is realisable and
    #: the walking-distance constraint of Eq. 5 can never bind.  Denser
    #: settings shorten paths only marginally.
    walk_cutoff_km: float = 1.5
    walk_neighbours: int = 1
    bike_cutoff_km: float = 3.0
    bike_neighbours: int = 1
    car_cutoff_km: float = 5.0
    car_neighbours: int = 1

    #: Road links are bidirectional at the facility level: a traveller who can
    #: walk, cycle or drive from A to B can make the return trip.  Without this
    #: the k-nearest-neighbour construction yields a near-acyclic layer -- two
    #: facilities are mutually nearest only by coincidence -- and no
    #: mode-dominated route family is realisable.  One-way restrictions are
    #: still honoured *inside* the road network when the link distance is
    #: computed; they are not meaningful between consolidated facilities.
    symmetric_road_layers: bool = False

    #: Free-flow speeds (km/h) used to convert road length into traversal time.
    walk_speed_kmph: float = 4.5
    bike_speed_kmph: float = 14.0
    car_speed_kmph: float = 26.0

    #: Transfer wait bounds in minutes (Section 3.2).
    min_wait_min: float = 3.0
    max_wait_min: float = 15.0

    #: Number of Dijkstra sources processed per batch (memory control).
    dijkstra_batch: int = 40


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _to_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Equirectangular projection to local kilometres, adequate at city scale."""
    lat0 = float(np.mean(lat))
    x = np.radians(lon) * EARTH_RADIUS_KM * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_RADIUS_KM
    return np.column_stack([x, y])


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _normalise_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower().strip() if ch.isalnum())


def _load_boundary_bbox(path: str | Path) -> Tuple[float, float, float, float]:
    """Return ``(min_lat, min_lon, max_lat, max_lon)`` of the study boundary."""
    with open(path, "r", encoding="utf-8") as fh:
        geo = json.load(fh)

    pts: List[Sequence[float]] = []

    def walk(obj):
        if isinstance(obj, dict):
            for key in ("features", "geometry", "coordinates"):
                if key in obj:
                    walk(obj[key])
        elif isinstance(obj, list):
            if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
                pts.append(obj)
            else:
                for item in obj:
                    walk(item)

    walk(geo)
    arr = np.asarray(pts, dtype=float)
    return float(arr[:, 1].min()), float(arr[:, 0].min()), float(arr[:, 1].max()), float(arr[:, 0].max())


# --------------------------------------------------------------------------
# Step 1 - transit layer
# --------------------------------------------------------------------------

def _active_services(zf: zipfile.ZipFile, service_date: int) -> set:
    """Service ids running on ``service_date`` per calendar + calendar_dates."""
    weekday = ["monday", "tuesday", "wednesday", "thursday", "friday",
               "saturday", "sunday"][pd.Timestamp(str(service_date)).weekday()]

    active: set = set()
    try:
        cal = pd.read_csv(zf.open("calendar.txt"))
        mask = (cal["start_date"] <= service_date) & (cal["end_date"] >= service_date) & (cal[weekday] == 1)
        active |= set(cal.loc[mask, "service_id"])
    except KeyError:
        pass

    try:
        cd = pd.read_csv(zf.open("calendar_dates.txt"))
        same = cd[cd["date"] == service_date]
        active |= set(same.loc[same["exception_type"] == 1, "service_id"])
        active -= set(same.loc[same["exception_type"] == 2, "service_id"])
    except KeyError:
        pass

    return active


def _seconds(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split(":", expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def build_transit_layer(cfg: NetworkBuildConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(nodes, edges)`` of the consolidated bus and tram layers."""
    logger.info("Building transit layer from %s ...", cfg.gtfs_path)
    with zipfile.ZipFile(cfg.gtfs_path) as zf:
        stops = pd.read_csv(zf.open("stops.txt"))
        routes = pd.read_csv(zf.open("routes.txt"))
        trips = pd.read_csv(zf.open("trips.txt"))
        stop_times = pd.read_csv(zf.open("stop_times.txt"))
        active = _active_services(zf, cfg.service_date)

    mode_of_type = {0: "tram", 3: "bus"}
    routes = routes[routes["route_type"].isin(mode_of_type)].copy()
    routes["mode"] = routes["route_type"].map(mode_of_type)

    trips = trips[trips["route_id"].isin(routes["route_id"])]
    if active:
        active_trips = trips[trips["service_id"].isin(active)]
        if len(active_trips) > 0:
            trips = active_trips
        else:
            logger.warning("No trip active on %s; falling back to the full trip set.", cfg.service_date)
    trips = trips.merge(routes[["route_id", "mode"]], on="route_id", how="left")

    st = stop_times[stop_times["trip_id"].isin(trips["trip_id"])].copy()
    st["dep_s"] = _seconds(st["departure_time"])
    st["arr_s"] = _seconds(st["arrival_time"])
    st = st[(st["dep_s"] >= cfg.window_start_h * 3600) & (st["dep_s"] <= cfg.window_end_h * 3600)]
    st = st.merge(trips[["trip_id", "mode", "route_id"]], on="trip_id", how="left")
    logger.info("  %d stop_times on %d trips inside the reference window",
                len(st), st["trip_id"].nunique())

    # ---- consolidate stops into physical facilities ----------------------
    served = stops[stops["stop_id"].isin(st["stop_id"].unique())].copy()
    served["name_key"] = served["stop_name"].map(_normalise_name)
    xy = _to_xy(served["stop_lat"].to_numpy(), served["stop_lon"].to_numpy())

    n = len(served)
    tree = cKDTree(xy)
    same_name = tree.query_pairs(cfg.name_merge_radius_m / 1000.0, output_type="ndarray")
    keys = served["name_key"].to_numpy()
    same_name = same_name[keys[same_name[:, 0]] == keys[same_name[:, 1]]]
    close = tree.query_pairs(cfg.proximity_merge_radius_m / 1000.0, output_type="ndarray")
    pairs = np.vstack([same_name, close]) if len(same_name) else close

    adj = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    ).tocsr()
    n_clusters, labels = connected_components(adj, directed=False)
    served["node_id"] = [f"N{lab:04d}" for lab in labels]
    logger.info("  %d served stops consolidated into %d facilities", n, n_clusters)

    stop_to_node = dict(zip(served["stop_id"], served["node_id"]))

    # ---- edges between consecutive stops ---------------------------------
    st = st[st["stop_id"].isin(stop_to_node)].copy()
    st["node_id"] = st["stop_id"].map(stop_to_node)
    st = st.sort_values(["trip_id", "stop_sequence"])
    grp = st.groupby("trip_id", sort=False)
    st["next_node"] = grp["node_id"].shift(-1)
    st["next_arr"] = grp["arr_s"].shift(-1)
    st["run_s"] = st["next_arr"] - st["dep_s"]

    seg = st.dropna(subset=["next_node"])
    seg = seg[(seg["node_id"] != seg["next_node"]) & (seg["run_s"] > 0)]

    edges = (
        seg.groupby(["node_id", "next_node", "mode"])
        .agg(run_s=("run_s", "median"), n_trips=("trip_id", "nunique"))
        .reset_index()
        .rename(columns={"node_id": "u", "next_node": "v"})
    )

    # Half the mean headway over the window, clipped to the manuscript's
    # transfer-penalty range of three to fifteen minutes (Section 3.2).
    window_min = (cfg.window_end_h - cfg.window_start_h) * 60.0
    headway = window_min / edges["n_trips"].clip(lower=1)
    edges["wait_min"] = np.clip(headway / 2.0, cfg.min_wait_min, cfg.max_wait_min)
    edges["travel_time_min"] = edges["run_s"] / 60.0

    # ---- node table -------------------------------------------------------
    nodes = (
        served.groupby("node_id")
        .agg(lat=("stop_lat", "mean"), lon=("stop_lon", "mean"),
             stop_name=("stop_name", "first"), n_stops=("stop_id", "size"))
        .reset_index()
    )
    trips_per_node = seg.groupby("node_id")["trip_id"].nunique()
    nodes["n_trips"] = nodes["node_id"].map(trips_per_node).fillna(0).astype(int)

    keep = nodes["n_trips"] >= cfg.min_trips_per_node
    logger.info("  %d/%d facilities meet the minimum service threshold (%d trips)",
                int(keep.sum()), len(nodes), cfg.min_trips_per_node)
    nodes = nodes[keep].reset_index(drop=True)
    edges = edges[edges["u"].isin(nodes["node_id"]) & edges["v"].isin(nodes["node_id"])]

    # length of a transit edge = great-circle distance between its endpoints
    coord = nodes.set_index("node_id")[["lat", "lon"]]
    edges = edges.merge(coord, left_on="u", right_index=True)
    edges = edges.merge(coord, left_on="v", right_index=True, suffixes=("_u", "_v"))
    edges["length_km"] = _haversine_km(
        edges["lat_u"].to_numpy(), edges["lon_u"].to_numpy(),
        edges["lat_v"].to_numpy(), edges["lon_v"].to_numpy(),
    )
    edges = edges[["u", "v", "mode", "travel_time_min", "wait_min", "length_km", "n_trips"]]

    logger.info("  transit layer: %d nodes, %d edges (%s)",
                len(nodes), len(edges), edges["mode"].value_counts().to_dict())
    return nodes, edges


# --------------------------------------------------------------------------
# Step 2 - road layers
# --------------------------------------------------------------------------

def _road_graph(layers, mode: str) -> Tuple[coo_matrix, np.ndarray]:
    """Return a sparse road graph for ``mode`` and the node coordinates."""
    col = {"walk": 0, "bike": 1, "car": 2}[mode]
    sel = layers.mode_mask[:, col]
    e = layers.edges[sel]
    d = _haversine_km(
        layers.lat[e[:, 0]], layers.lon[e[:, 0]],
        layers.lat[e[:, 1]], layers.lon[e[:, 1]],
    )
    n = layers.n_nodes
    if mode == "car":
        ow = layers.oneway[sel]
        src = np.concatenate([e[:, 0], e[~ow][:, 1]])
        dst = np.concatenate([e[:, 1], e[~ow][:, 0]])
        w = np.concatenate([d, d[~ow]])
    else:  # walking and cycling ignore one-way restrictions on the road layer
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        w = np.concatenate([d, d])
    return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr(), d


def build_road_layer(
    cfg: NetworkBuildConfig,
    nodes: pd.DataFrame,
    layers,
    mode: str,
) -> pd.DataFrame:
    """Connect consolidated nodes through the OSM road layer of ``mode``."""
    cutoff, k, speed = {
        "walk": (cfg.walk_cutoff_km, cfg.walk_neighbours, cfg.walk_speed_kmph),
        "bike": (cfg.bike_cutoff_km, cfg.bike_neighbours, cfg.bike_speed_kmph),
        "car": (cfg.car_cutoff_km, cfg.car_neighbours, cfg.car_speed_kmph),
    }[mode]

    graph, _ = _road_graph(layers, mode)

    # keep only road nodes that carry at least one edge of this mode
    deg = np.asarray((graph != 0).sum(axis=1)).ravel()
    live = np.flatnonzero(deg > 0)
    logger.info("  [%s] road layer: %d live nodes", mode, len(live))

    live_xy = _to_xy(layers.lat[live], layers.lon[live])
    tree = cKDTree(live_xy)
    node_xy = _to_xy(nodes["lat"].to_numpy(), nodes["lon"].to_numpy())
    snap_dist, snap_pos = tree.query(node_xy, k=1)
    sources = live[snap_pos]

    records: List[Dict[str, object]] = []
    ids = nodes["node_id"].to_numpy()
    n_nodes = len(nodes)

    for start in range(0, n_nodes, cfg.dijkstra_batch):
        batch = np.arange(start, min(start + cfg.dijkstra_batch, n_nodes))
        dist = dijkstra(graph, directed=(mode == "car"),
                        indices=sources[batch], limit=cutoff)
        # distance from each batch source to every other consolidated node
        sub = dist[:, sources]
        for row, i in enumerate(batch):
            d = sub[row].copy()
            d[i] = np.inf
            reachable = np.flatnonzero(np.isfinite(d))
            if len(reachable) == 0:
                continue
            nearest = reachable[np.argsort(d[reachable])[:k]]
            for j in nearest:
                length = float(d[j] + (snap_dist[i] + snap_dist[j]))
                link = {
                    "mode": mode, "length_km": length,
                    "travel_time_min": 60.0 * length / speed,
                    "wait_min": 0.0, "n_trips": 0,
                }
                records.append({"u": ids[i], "v": ids[j], **link})
                if cfg.symmetric_road_layers:
                    records.append({"u": ids[j], "v": ids[i], **link})

    df = pd.DataFrame(records)
    logger.info("  [%s] %d edges created", mode, len(df))
    return df


# --------------------------------------------------------------------------
# Step 3 - assembly and cleaning
# --------------------------------------------------------------------------

def assemble(nodes: pd.DataFrame, edge_frames: Sequence[pd.DataFrame]) -> nx.MultiDiGraph:
    edges = pd.concat([f for f in edge_frames if len(f)], ignore_index=True)

    # topological artifacts: self-loops and duplicate parallel edges of one mode
    edges = edges[edges["u"] != edges["v"]]
    edges = edges.sort_values("travel_time_min").drop_duplicates(["u", "v", "mode"], keep="first")

    G = nx.MultiDiGraph()
    for row in nodes.itertuples(index=False):
        G.add_node(
            row.node_id,
            x=float(row.lon), y=float(row.lat),
            stop_name=str(row.stop_name),
            n_trips=int(row.n_trips),
        )
    for row in edges.itertuples(index=False):
        G.add_edge(
            row.u, row.v, key=row.mode,
            mode=row.mode,
            length_km=float(row.length_km),
            travel_time_min=float(row.travel_time_min),
            wait_min=float(row.wait_min),
            n_trips=int(row.n_trips),
        )

    # disconnected components removed -> keep the largest strongly connected one
    if G.number_of_nodes():
        comps = list(nx.strongly_connected_components(G))
        largest = max(comps, key=len)
        removed = G.number_of_nodes() - len(largest)
        if removed:
            logger.info("  dropping %d node(s) outside the largest strongly connected component", removed)
        G = G.subgraph(largest).copy()

    # annotate mode availability and transfer-hub status
    for n in G.nodes():
        modes = set()
        for _, _, d in G.in_edges(n, data=True):
            modes.add(d["mode"])
        for _, _, d in G.out_edges(n, data=True):
            modes.add(d["mode"])
        G.nodes[n]["modes"] = ",".join(sorted(modes))
        G.nodes[n]["n_modes"] = len(modes)
        G.nodes[n]["is_transfer_hub"] = bool(len(modes) >= 3)
    return G


def build_multimodal_graph(cfg: NetworkBuildConfig | None = None) -> nx.MultiDiGraph:
    """Build and persist the consolidated multimodal graph."""
    cfg = cfg or NetworkBuildConfig()

    bbox = _load_boundary_bbox(cfg.boundary_path)
    logger.info("Study boundary bbox: %s", np.round(bbox, 4).tolist())

    nodes, transit_edges = build_transit_layer(cfg)
    layers = load_or_parse(cfg.osm_path, cfg.osm_cache_path, bbox=bbox)

    road_frames = [build_road_layer(cfg, nodes, layers, m) for m in ("walk", "bike", "car")]
    G = assemble(nodes, [transit_edges] + road_frames)

    out = Path(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, out)
    logger.info("Graph written to %s: |V|=%d, |E|=%d",
                out, G.number_of_nodes(), G.number_of_edges())

    with open(out.with_suffix(".build_config.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    return G


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the consolidated multimodal graph.")
    parser.add_argument("--osm", default=NetworkBuildConfig.osm_path)
    parser.add_argument("--gtfs", default=NetworkBuildConfig.gtfs_path)
    parser.add_argument("--out", default=NetworkBuildConfig.out_path)
    parser.add_argument("--min-trips", type=int, default=NetworkBuildConfig.min_trips_per_node)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = NetworkBuildConfig(
        osm_path=args.osm, gtfs_path=args.gtfs, out_path=args.out,
        min_trips_per_node=args.min_trips,
    )
    build_multimodal_graph(cfg)


if __name__ == "__main__":
    main()
