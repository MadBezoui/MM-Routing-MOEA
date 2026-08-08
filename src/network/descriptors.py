"""descriptors.py
==================
Quantitative descriptors of the multimodal network (Table 6 of the manuscript).

Everything reported here is recomputed from the GraphML file; no value is
hard-coded.  Running this module also writes the per-mode adjacency matrices
and the node-degree histogram announced in Section 5.2.

Usage
-----
    python -m src.network.descriptors \\
        --graph data/processed/strasbourg_multimodal.graphml \\
        --out   results/network
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODES = ("walk", "bike", "bus", "tram", "car")


def _undirected_degrees(G: nx.MultiDiGraph) -> np.ndarray:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from((u, v) for u, v, _ in G.edges(keys=True))
    return np.asarray([d for _, d in H.degree()], dtype=float)


def mean_shortest_path_edges(G: nx.MultiDiGraph, sample: int = 200, seed: int = 42) -> float:
    """Mean hop count of the shortest path over a random sample of OD pairs."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return float("nan")
    lengths: List[int] = []
    sources = rng.choice(len(nodes), size=min(sample, len(nodes)), replace=False)
    for si in sources:
        src = nodes[si]
        sp = nx.single_source_shortest_path_length(G, src)
        lengths.extend(v for k, v in sp.items() if k != src)
    return float(np.mean(lengths)) if lengths else float("nan")


def count_feasible_paths(
    G: nx.MultiDiGraph,
    n_pairs: int = 60,
    cutoff: int = 8,
    max_paths: int = 400,
    seed: int = 42,
) -> Dict[str, float]:
    """Number of simple multimodal paths per OD pair, capped at ``max_paths``.

    The cap keeps the enumeration tractable; the reported statistics are
    therefore lower bounds on dense pairs and exact on sparse ones.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    simple = nx.DiGraph()
    simple.add_nodes_from(nodes)
    simple.add_edges_from((u, v) for u, v, _ in G.edges(keys=True))

    counts: List[int] = []
    attempts = 0
    while len(counts) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        o, d = rng.choice(len(nodes), size=2, replace=False)
        o, d = nodes[o], nodes[d]
        if not nx.has_path(simple, o, d):
            continue
        n = 0
        for _ in nx.all_simple_paths(simple, o, d, cutoff=cutoff):
            n += 1
            if n >= max_paths:
                break
        counts.append(n)

    arr = np.asarray(counts, dtype=float)
    return {
        "n_od_pairs_sampled": int(len(arr)),
        "mean_feasible_paths_per_od": float(arr.mean()) if len(arr) else float("nan"),
        "median_feasible_paths_per_od": float(np.median(arr)) if len(arr) else float("nan"),
        "path_enumeration_cutoff_edges": cutoff,
        "path_enumeration_cap": max_paths,
    }


def compute_descriptors(G: nx.MultiDiGraph, sample_paths: bool = True) -> Dict[str, object]:
    """Return the full Table 6 descriptor set."""
    n_v = G.number_of_nodes()
    n_e = G.number_of_edges()

    by_mode = {m: 0 for m in MODES}
    for _, _, d in G.edges(data=True):
        by_mode[d["mode"]] = by_mode.get(d["mode"], 0) + 1

    deg = _undirected_degrees(G)
    hubs = sum(1 for _, d in G.nodes(data=True) if d.get("n_modes", 0) >= 3)

    # A node is an *interchange* hub when both scheduled transit modes meet
    # there, i.e. a traveller can change between bus and tram without walking.
    interchange = 0
    for n, d in G.nodes(data=True):
        modes = set(str(d.get("modes", "")).split(","))
        if {"bus", "tram"} <= modes:
            interchange += 1

    # Per-mode reachability: a mode family is realisable end to end only if its
    # own layer is strongly connected over the nodes it touches.
    layer_report: Dict[str, Dict[str, int]] = {}
    for mode in MODES:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d["mode"] == mode)
        touched = [n for n in H if H.degree(n) > 0]
        comps = list(nx.strongly_connected_components(H.subgraph(touched))) if touched else []
        layer_report[mode] = {
            "nodes_touched": len(touched),
            "largest_strongly_connected_component": max((len(c) for c in comps), default=0),
        }

    desc: Dict[str, object] = {
        "n_nodes": n_v,
        "n_edges": n_e,
        "per_mode_layer_connectivity": layer_report,
        "edges_by_mode": {m: int(by_mode.get(m, 0)) for m in MODES},
        "edge_share_by_mode": {
            m: round(100.0 * by_mode.get(m, 0) / n_e, 1) if n_e else 0.0 for m in MODES
        },
        # Two conventions are reported.  ``simple`` collapses the parallel
        # edges of different modes between the same node pair into a single
        # undirected link; ``multi`` is 2|E|/|V|, i.e. every directed edge
        # counted once at each endpoint.
        "mean_node_degree_simple_undirected": round(float(deg.mean()), 2) if n_v else float("nan"),
        "mean_node_degree_multi": round(2.0 * n_e / n_v, 2) if n_v else float("nan"),
        "median_node_degree": float(np.median(deg)) if n_v else float("nan"),
        "max_node_degree": int(deg.max()) if n_v else 0,
        "n_transfer_hubs_ge_3_modes": hubs,
        "n_interchange_hubs_bus_and_tram": interchange,
        "is_strongly_connected": bool(nx.is_strongly_connected(G)) if n_v else False,
    }

    if sample_paths:
        desc["mean_shortest_path_edges"] = round(mean_shortest_path_edges(G), 2)
        desc.update(count_feasible_paths(G))

    return desc


def write_artifacts(G: nx.MultiDiGraph, out_dir: str | Path, sample_paths: bool = True) -> Dict[str, object]:
    """Write descriptors, per-mode adjacency matrices and the degree histogram."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    desc = compute_descriptors(G, sample_paths=sample_paths)
    with open(out / "network_descriptors.json", "w", encoding="utf-8") as fh:
        json.dump(desc, fh, indent=2)

    nodes = sorted(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    for mode in MODES:
        A = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
        for u, v, d in G.edges(data=True):
            if d["mode"] == mode:
                A[index[u], index[v]] = 1
        pd.DataFrame(A, index=nodes, columns=nodes).to_csv(out / f"adjacency_{mode}.csv")

    deg = _undirected_degrees(G)
    hist, edges = np.histogram(deg, bins=np.arange(0, deg.max() + 2) if len(deg) else [0, 1])
    pd.DataFrame({"degree": edges[:-1].astype(int), "count": hist}).to_csv(
        out / "node_degree_histogram.csv", index=False
    )

    rows = [{"descriptor": "Number of nodes |V|", "value": desc["n_nodes"]},
            {"descriptor": "Number of edges |E|", "value": desc["n_edges"]}]
    for m in MODES:
        rows.append({
            "descriptor": f"Edges by mode - {m}",
            "value": f'{desc["edges_by_mode"][m]} ({desc["edge_share_by_mode"][m]}%)',
        })
    for key, label in [
        ("mean_node_degree_multi", "Mean node degree (undirected, 2|E|/|V|)"),
        ("mean_node_degree_simple_undirected", "Mean node degree (simple graph)"),
        ("median_node_degree", "Median node degree"),
        ("max_node_degree", "Max node degree"),
        ("mean_shortest_path_edges", "Mean shortest-path length (all modes, edges)"),
        ("mean_feasible_paths_per_od", "Mean feasible multimodal paths per OD pair"),
        ("median_feasible_paths_per_od", "Median feasible paths per OD pair"),
        ("n_transfer_hubs_ge_3_modes", "Number of transfer hubs (>= 3 modes)"),
        ("n_interchange_hubs_bus_and_tram", "Number of bus/tram interchange hubs"),
    ]:
        if key in desc:
            rows.append({"descriptor": label, "value": desc[key]})
    pd.DataFrame(rows).to_csv(out / "table6_network_descriptors.csv", index=False)

    logger.info("Descriptors written to %s", out)
    return desc


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Table 6 network descriptors.")
    parser.add_argument("--graph", default="data/processed/strasbourg_multimodal.graphml")
    parser.add_argument("--out", default="results/network")
    parser.add_argument("--no-path-sampling", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    G = nx.read_graphml(args.graph)
    desc = write_artifacts(G, args.out, sample_paths=not args.no_path_sampling)
    print(json.dumps(desc, indent=2))


if __name__ == "__main__":
    main()
