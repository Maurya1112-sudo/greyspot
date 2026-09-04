"""OS Open Roads ingestion — the dossier's originally-preferred official
road network (`ingest/network.py`'s module docstring named this exact
upgrade path before any of this existed: "Swapping in OS Open Roads later
only requires replacing `build_borough_graph` with a loader for the
downloaded GeoPackage... everything downstream operates on the same edge
GeoDataFrame shape.").

The user downloaded and extracted the OS Data Hub GeoPackage (GB-wide,
~2GB, 3.96M road_link features, `oproad_gpkg_gb/Data/oproad_gb.gpkg`)
2026-09-01. This module builds an **OSMnx-compatible** `nx.MultiDiGraph`
from a bbox-filtered slice of it, so every downstream function
(`graph_to_edges_gdf`, `snap_points_to_graph`, the line-graph/GAT
machinery) works completely unchanged — this was a deliberate design
choice from the very first session, not a retrofit.

Why this is a real upgrade, not just an alternative: OS Open Roads is
Ordnance Survey's official, surveyed road network (Crown copyright,
Open Government Licence), with genuine `start_node`/`end_node` topology
and an official road classification — versus OSMnx/OpenStreetMap's
community-edited geometry, which the dossier itself named as a fallback/
cross-check, not the preferred source (see `docs/research_notes.md`).

Known, documented limitation: OS Open Roads carries no oneway/direction
attribute (unlike OSM's `oneway` tag) — every link is added in both
directions here, which slightly over-connects genuinely one-way streets.
This is a real simplification, not hidden — see the loop below.
"""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import osmnx as ox
import pyogrio
from pyproj import Transformer
from shapely.geometry import Point

logger = logging.getLogger(__name__)

_BNG = "epsg:27700"  # British National Grid - the GeoPackage's native CRS
_WGS84 = "epsg:4326"  # everything else in this project (STATS19, OSMnx) uses this


def borough_bbox_wgs84(osm_place: str) -> tuple[float, float, float, float]:
    """The borough's administrative-boundary bbox via a lightweight
    Nominatim geocode (`ox.geocode_to_gdf`) — deliberately *not* a full
    `ox.graph_from_place` drive-network download, so OS Open Roads
    ingestion never depends on OSMnx's own network fetch (verified
    2026-09-01: within ~0.001 degrees of the equivalent drive-graph
    bbox for Westminster — close enough for a borough-sized bbox filter,
    where the margin of error is a handful of metres against a borough
    several kilometres across).

    **Only a coarse pre-filter** — see `borough_polygon_wgs84` for the
    real administrative-boundary shape, which `build_borough_graph_os_open_roads`
    now also requires. A bbox around an irregularly-shaped borough
    (Westminster's includes a long river frontage) covers a meaningfully
    larger area than the actual polygon - found 2026-09-02, the first
    time this function's output was actually used end-to-end against
    the real GeoPackage (rather than just verified against OSMnx's own
    bbox in isolation): a bbox-only clip of Westminster pulled in 11,347
    road links, most of them genuinely outside the borough (e.g. across
    the Thames) - polygon-clipping (see `build_borough_graph_os_open_roads`)
    cuts this to ~5,500."""
    gdf = ox.geocode_to_gdf(osm_place)
    return tuple(gdf.total_bounds)


def borough_polygon_wgs84(osm_place: str):
    """The borough's real administrative-boundary polygon (not just its
    bounding box) - added 2026-09-02 after `build_borough_graph_os_open_roads`
    turned out to only ever ask for a bbox-based clip, silently pulling
    in over twice as many road links as the actual borough contains (see
    `borough_bbox_wgs84`'s docstring). Returns a shapely geometry in
    EPSG:4326, matching `ox.geocode_to_gdf`'s own output CRS."""
    gdf = ox.geocode_to_gdf(osm_place)
    return gdf.geometry.iloc[0]


def _map_highway(row) -> str:
    """OS Open Roads' own classification vocabulary mapped to OSM-style
    `highway` values, since every downstream feature/UI (build_features.py's
    road-type aggregation, the frontend's "Primary/Residential/Trunk"
    labels) expects OSM's vocabulary. This is Greyspot's own heuristic
    crosswalk — no official OS-to-OSM standard exists — verified 2026-09-01
    against a real sample of Westminster's OS Open Roads data (see
    docs/decision_log.md for the field-value distributions this was built
    from, not guessed)."""
    if row["road_classification"] == "Motorway":
        return "motorway"
    if row.get("trunk_road") is True:
        return "trunk"
    if row["road_classification"] == "A Road":
        return "primary"
    if row["road_classification"] == "B Road":
        return "secondary"
    if row["road_classification"] == "Classified Unnumbered":
        return "tertiary"
    if row["road_function"] == "Local Road":
        return "residential"
    if row["road_function"] in ("Restricted Local Access Road", "Local Access Road", "Secondary Access Road"):
        return "service"
    return "unclassified"


def build_borough_graph_os_open_roads(
    bbox_wgs84: tuple[float, float, float, float],
    gpkg_path: Path,
    cache_path: Path | None = None,
    polygon_wgs84=None,
) -> nx.MultiDiGraph:
    """Build an OSMnx-compatible graph from a local OS Open Roads
    GeoPackage, clipped to `bbox_wgs84` (minx, miny, maxx, maxy in
    EPSG:4326). Pass the *same* bbox an existing OSMnx borough graph
    covers (e.g. its edges' `.total_bounds`) so the two sources are
    directly comparable over the same area.

    Bbox filtering uses pyogrio's native GeoPackage spatial-index support
    (`bbox=`) — reads a borough-sized slice (~11k features for
    Westminster, verified in well under a second) without ever loading
    the full 2GB/3.96M-feature national file into memory.

    `polygon_wgs84` (added 2026-09-02, see `borough_polygon_wgs84`):
    when given, links are further filtered to those with at least one
    endpoint inside the real administrative-boundary polygon, not just
    the bbox - matching OSMnx's own node-based inclusion convention for
    `graph_from_place`. Without this, a bbox-only clip of an irregularly
    shaped borough (e.g. Westminster's long Thames frontage) pulls in
    over twice as many links as the borough actually contains (11,347
    vs ~5,500 for Westminster - found the first time this function was
    run end-to-end against the real GeoPackage). Kept optional (not
    required) for backwards compatibility with any caller that only has
    a bbox on hand - but every caller SHOULD pass it when the real
    borough polygon is available, which is always (`borough_polygon_wgs84`
    costs one extra lightweight geocode call).
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached OS Open Roads graph from %s", cache_path)
        # node_dtypes/edge_dtypes={"osmid": str}: OS Open Roads node AND
        # edge ids are UUID-style TOIDs (e.g.
        # "5216C3AC-92A7-486A-A3B4-20B29C6F5982"), not OSMnx's own
        # integer ids - ox.load_graphml's hardcoded default (osmid: int,
        # applied to both nodes and edges) crashes on the round trip
        # through a cached file (found 2026-09-02, the first time this
        # cache path was ever actually re-loaded rather than freshly
        # built - it took two attempts to find both places the default
        # int coercion applies, node ids then edge ids).
        return ox.load_graphml(cache_path, node_dtypes={"osmid": str}, edge_dtypes={"osmid": str})

    transformer = Transformer.from_crs(_WGS84, _BNG, always_xy=True)
    minx, miny = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    maxx, maxy = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])

    logger.info("Reading OS Open Roads road_link within bbox from %s", gpkg_path)
    links = pyogrio.read_dataframe(str(gpkg_path), layer="road_link", bbox=(minx, miny, maxx, maxy))
    n_total = len(links)
    links = links[links["fictitious"] != True].copy()  # noqa: E712 - explicit bool compare, NA-safe
    n_fictitious = n_total - len(links)
    if n_fictitious:
        logger.info("Dropped %d/%d fictitious (non-real) links", n_fictitious, n_total)
    links = links.to_crs(_WGS84)

    if polygon_wgs84 is not None:
        n_before_polygon = len(links)

        def _endpoint_inside(geom) -> bool:
            coords = list(geom.coords)
            return polygon_wgs84.contains(Point(coords[0])) or polygon_wgs84.contains(Point(coords[-1]))

        links = links[links.geometry.apply(_endpoint_inside)].copy()
        logger.info(
            "Polygon-clipped %d bbox-matched links down to %d (at least one endpoint inside the real borough boundary)",
            n_before_polygon, len(links),
        )

    links["highway"] = links.apply(_map_highway, axis=1)

    graph = nx.MultiDiGraph(crs=_WGS84)
    seen_nodes: set[str] = set()
    edge_counts: dict[tuple[str, str], int] = {}

    def _ensure_node(node_id: str, point: Point) -> None:
        if node_id not in seen_nodes:
            seen_nodes.add(node_id)
            graph.add_node(node_id, x=point.x, y=point.y)

    n_skipped_geom = 0
    for row in links.itertuples(index=False):
        geom = row.geometry
        if geom is None or geom.is_empty:
            n_skipped_geom += 1
            continue
        u, v = row.start_node, row.end_node
        coords = list(geom.coords)
        _ensure_node(u, Point(coords[0]))
        _ensure_node(v, Point(coords[-1]))
        edge_attrs = dict(
            geometry=geom, highway=row.highway, name=row.name_1,
            length=row.length, osmid=row.id, road_classification=row.road_classification,
        )
        # OS Open Roads has no oneway/direction attribute (see module
        # docstring) - add both directions, matching how a two-way OSM
        # street (the common case) is represented in an OSMnx graph.
        for a, b in ((u, v), (v, u)):
            key = edge_counts.get((a, b), 0)
            edge_counts[(a, b)] = key + 1
            graph.add_edge(a, b, key=key, **edge_attrs)

    if n_skipped_geom:
        logger.warning("Skipped %d links with missing/empty geometry", n_skipped_geom)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache_path)
        logger.info("Cached OS Open Roads graph to %s", cache_path)

    logger.info(
        "Built OS Open Roads graph: %d nodes, %d edges (from %d links, bidirectional)",
        graph.number_of_nodes(), graph.number_of_edges(), len(links),
    )
    return graph
