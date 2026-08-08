"""osm_parser.py
================
Streaming OpenStreetMap XML reader used by :mod:`src.network.builder`.

The module extracts, in a single pass over the ``.osm`` file, the sub-graph of
ways that carry a ``highway`` tag together with the coordinates of the nodes
those ways reference.  Everything is stored in flat ``numpy`` arrays and cached
to a compressed ``.npz`` archive so that the (expensive) XML pass is performed
only once.

Three mode-specific road layers are derived from the OSM ``highway`` tag:

``walk``
    Pedestrian-accessible ways.  Motorways and their link roads are excluded,
    as are ways explicitly tagged ``foot=no``.
``bike``
    Cycle-accessible ways: dedicated cycleways plus the ordinary road classes
    on which cycling is permitted in France.  Ways tagged ``bicycle=no`` are
    excluded.
``car``
    Ways of the drivable classes, minus ways tagged ``motor_vehicle=no`` or
    ``access=private``.

Only the geometry and the mode masks are produced here; the consolidation into
the compact multimodal graph of the manuscript is performed by
:mod:`src.network.builder`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from xml.etree.ElementTree import iterparse

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# OSM highway classification
# --------------------------------------------------------------------------

#: Way classes on which walking is permitted.
WALK_HIGHWAYS: Set[str] = {
    "footway", "path", "pedestrian", "steps", "living_street", "track",
    "residential", "service", "unclassified", "tertiary", "tertiary_link",
    "secondary", "secondary_link", "primary", "primary_link", "cycleway",
    "corridor", "crossing",
}

#: Way classes on which cycling is permitted.
BIKE_HIGHWAYS: Set[str] = {
    "cycleway", "path", "track", "living_street", "residential", "service",
    "unclassified", "tertiary", "tertiary_link", "secondary", "secondary_link",
    "primary", "primary_link",
}

#: Way classes open to private motor vehicles.
CAR_HIGHWAYS: Set[str] = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "living_street",
    "service",
}

_ALL_HIGHWAYS = WALK_HIGHWAYS | BIKE_HIGHWAYS | CAR_HIGHWAYS


class OSMRoadLayers:
    """Container for the three mode-specific road layers.

    Attributes
    ----------
    node_ids
        Sorted array of the OSM node identifiers actually referenced by a
        retained way.
    lat, lon
        Coordinates aligned with ``node_ids``.
    edges
        ``(n_edges, 2)`` array of *indices into* ``node_ids``.
    mode_mask
        ``(n_edges, 3)`` boolean array whose columns are ``(walk, bike, car)``.
    oneway
        ``(n_edges,)`` boolean array, ``True`` when the underlying way is
        one-way (relevant for the car layer only).
    """

    __slots__ = ("node_ids", "lat", "lon", "edges", "mode_mask", "oneway")

    def __init__(self, node_ids, lat, lon, edges, mode_mask, oneway):
        self.node_ids = node_ids
        self.lat = lat
        self.lon = lon
        self.edges = edges
        self.mode_mask = mode_mask
        self.oneway = oneway

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            node_ids=self.node_ids,
            lat=self.lat,
            lon=self.lon,
            edges=self.edges,
            mode_mask=self.mode_mask,
            oneway=self.oneway,
        )
        logger.info("Cached OSM road layers to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "OSMRoadLayers":
        with np.load(Path(path)) as z:
            return cls(
                z["node_ids"], z["lat"], z["lon"],
                z["edges"], z["mode_mask"], z["oneway"],
            )

    # -- convenience -------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return int(len(self.node_ids))

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))

    def mode_edges(self, mode: str) -> np.ndarray:
        col = {"walk": 0, "bike": 1, "car": 2}[mode]
        return self.edges[self.mode_mask[:, col]]


# --------------------------------------------------------------------------
# Streaming parse
# --------------------------------------------------------------------------

def _classify(tags: Dict[str, str]) -> Tuple[bool, bool, bool]:
    """Return ``(walk, bike, car)`` accessibility flags for a tagged way."""
    hw = tags.get("highway")
    if hw is None or hw not in _ALL_HIGHWAYS:
        return False, False, False

    access = tags.get("access", "")
    if access in ("private", "no"):
        return False, False, False

    walk = hw in WALK_HIGHWAYS and tags.get("foot") != "no"
    if hw in ("motorway", "motorway_link", "trunk", "trunk_link"):
        walk = False

    bike = (hw in BIKE_HIGHWAYS and tags.get("bicycle") != "no") or hw == "cycleway"

    car = hw in CAR_HIGHWAYS and tags.get("motor_vehicle") != "no"

    return walk, bike, car


def parse_osm(
    osm_path: str | Path,
    bbox: Tuple[float, float, float, float] | None = None,
) -> OSMRoadLayers:
    """Stream ``osm_path`` and return the three mode-specific road layers.

    Parameters
    ----------
    osm_path
        Path to an uncompressed OSM XML extract.
    bbox
        Optional ``(min_lat, min_lon, max_lat, max_lon)`` filter applied to
        node coordinates.  Ways whose nodes all fall outside the box are
        dropped.

    Notes
    -----
    OSM XML guarantees that all ``<node>`` elements precede all ``<way>``
    elements, so a single pass suffices.  Coordinates are held in plain Python
    lists during the node phase and converted to ``numpy`` arrays before the
    way phase, which keeps peak memory close to 3 bytes per node.
    """
    osm_path = Path(osm_path)
    logger.info("Streaming OSM XML from %s ...", osm_path)

    node_id_list: List[int] = []
    lat_list: List[float] = []
    lon_list: List[float] = []

    way_refs: List[List[int]] = []
    way_flags: List[Tuple[bool, bool, bool]] = []
    way_oneway: List[bool] = []

    in_way = False
    cur_refs: List[int] = []
    cur_tags: Dict[str, str] = {}

    context = iterparse(str(osm_path), events=("start", "end"))
    for event, elem in context:
        tag = elem.tag

        if event == "start":
            if tag == "way":
                in_way = True
                cur_refs = []
                cur_tags = {}
            elif in_way:
                if tag == "nd":
                    cur_refs.append(int(elem.attrib["ref"]))
                elif tag == "tag":
                    cur_tags[elem.attrib["k"]] = elem.attrib["v"]
            continue

        # event == "end"
        if tag == "node":
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            if bbox is None or (bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]):
                node_id_list.append(int(elem.attrib["id"]))
                lat_list.append(lat)
                lon_list.append(lon)
            elem.clear()

        elif tag == "way":
            in_way = False
            flags = _classify(cur_tags)
            if any(flags) and len(cur_refs) >= 2:
                way_refs.append(cur_refs)
                way_flags.append(flags)
                way_oneway.append(cur_tags.get("oneway") in ("yes", "1", "true"))
            elem.clear()

        elif tag in ("relation", "bounds"):
            elem.clear()

    logger.info("  parsed %d in-box nodes and %d highway ways", len(node_id_list), len(way_refs))

    all_ids = np.asarray(node_id_list, dtype=np.int64)
    all_lat = np.asarray(lat_list, dtype=np.float64)
    all_lon = np.asarray(lon_list, dtype=np.float64)
    del node_id_list, lat_list, lon_list

    order = np.argsort(all_ids)
    all_ids = all_ids[order]
    all_lat = all_lat[order]
    all_lon = all_lon[order]

    # ---- resolve way node references to positions in ``all_ids`` ----------
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_flags: List[Tuple[bool, bool, bool]] = []
    edge_oneway: List[bool] = []

    for refs, flags, ow in zip(way_refs, way_flags, way_oneway):
        pos = np.searchsorted(all_ids, np.asarray(refs, dtype=np.int64))
        pos = np.clip(pos, 0, len(all_ids) - 1)
        valid = all_ids[pos] == np.asarray(refs, dtype=np.int64)
        pos = pos[valid]
        if len(pos) < 2:
            continue
        for a, b in zip(pos[:-1], pos[1:]):
            if a == b:
                continue
            edge_src.append(int(a))
            edge_dst.append(int(b))
            edge_flags.append(flags)
            edge_oneway.append(ow)

    del way_refs, way_flags, way_oneway

    edges = np.column_stack([
        np.asarray(edge_src, dtype=np.int32),
        np.asarray(edge_dst, dtype=np.int32),
    ])
    mode_mask = np.asarray(edge_flags, dtype=bool)
    oneway = np.asarray(edge_oneway, dtype=bool)
    del edge_src, edge_dst, edge_flags, edge_oneway

    # ---- drop nodes that no retained way references -----------------------
    used = np.zeros(len(all_ids), dtype=bool)
    used[edges[:, 0]] = True
    used[edges[:, 1]] = True
    remap = np.full(len(all_ids), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.sum()), dtype=np.int32)

    layers = OSMRoadLayers(
        node_ids=all_ids[used],
        lat=all_lat[used],
        lon=all_lon[used],
        edges=np.column_stack([remap[edges[:, 0]], remap[edges[:, 1]]]),
        mode_mask=mode_mask,
        oneway=oneway,
    )
    logger.info(
        "  retained %d road nodes / %d road segments (walk %d, bike %d, car %d)",
        layers.n_nodes, layers.n_edges,
        int(mode_mask[:, 0].sum()), int(mode_mask[:, 1].sum()), int(mode_mask[:, 2].sum()),
    )
    return layers


def load_or_parse(
    osm_path: str | Path,
    cache_path: str | Path,
    bbox: Tuple[float, float, float, float] | None = None,
) -> OSMRoadLayers:
    """Return cached road layers, parsing ``osm_path`` on a cache miss."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info("Loading cached OSM road layers from %s", cache_path)
        return OSMRoadLayers.load(cache_path)
    layers = parse_osm(osm_path, bbox=bbox)
    layers.save(cache_path)
    return layers
