"""Road-network construction (OSMnx) and collision-to-segment snapping.

OS Open Roads is the dossier's preferred network source (official
link-and-node topology) but requires a manual OS Data Hub account/download.
OSMnx/OpenStreetMap is used here instead for this first pipeline pass — it
needs no registration and the dossier itself names it as an accepted
fallback/cross-check network. Swapping in OS Open Roads later only requires
replacing `build_borough_graph` with a loader for the downloaded
GeoPackage/shapefile; everything downstream (feature building, snapping,
models) operates on the same edge GeoDataFrame shape.

`build_borough_graph` takes any OSMnx-geocodable place name, not just
Westminster - see `ingest.boroughs` for the registry this project uses to
run against more than one borough (the "Westminster now, London later"
plan in `docs/project_management.md`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

logger = logging.getLogger(__name__)

WESTMINSTER_PLACE_NAME = "City of Westminster, London, United Kingdom"


def build_borough_graph(place_name: str, cache_path: Path | None = None) -> nx.MultiDiGraph:
    """Fetch (or load cached) the drivable road network graph for any
    OSMnx-geocodable place."""
    if cache_path and cache_path.exists():
        logger.info("Loading cached graph from %s", cache_path)
        return ox.load_graphml(cache_path)

    logger.info("Downloading OSM drive network for %s", place_name)
    graph = ox.graph_from_place(place_name, network_type="drive")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache_path)
        logger.info("Cached graph to %s", cache_path)

    return graph


def build_westminster_graph(cache_path: Path | None = None) -> nx.MultiDiGraph:
    """Convenience wrapper kept for backwards compatibility with earlier
    callers/tests - equivalent to `build_borough_graph(WESTMINSTER_PLACE_NAME, ...)`."""
    return build_borough_graph(WESTMINSTER_PLACE_NAME, cache_path)


def consolidate_borough_graph(
    graph: nx.MultiDiGraph, tolerance_m: float, cache_path: Path | None = None,
) -> nx.MultiDiGraph:
    """Merges intersection nodes within `tolerance_m` metres of each other
    into single nodes (`osmnx.simplification.consolidate_intersections`),
    reducing the number of distinct road segments - added 2026-09-02 after
    this project's own re-reading of the paper's thesis (Table 7.2) found
    the paper's OS Open Roads network is consistently ~1.5x COARSER than
    this project's OSMnx network (Westminster: their 4,822 roads vs this
    project's 7,552 segments) - the leading remaining hypothesis for the
    AccHR@20 gap, previously untestable without the paper's own OS Open
    Roads source file (still not available - see `ingest.os_open_roads`'s
    own docstring).

    **A disclosed, self-directed approximation, NOT a claim of
    replicating OS Open Roads' actual topology**: this merges nearby OSM
    intersection nodes by DISTANCE alone (e.g. offset crossings, small
    roundabouts, closely-spaced signal-controlled junctions that OSM
    contributors mapped as separate nodes) - a real, principled way to
    reduce a network's granularity, but not the same thing as OS's own
    survey-based road link/node convention, which may group segments on
    entirely different criteria (e.g. by named-street continuity, not
    physical node proximity). Tested here as a genuinely testable version
    of the "does denser/finer network granularity hurt AccHR@20"
    hypothesis, using data already on hand, while the real OS Open Roads
    swap remains blocked on a source file only the project owner can
    download (requires a free OS Data Hub account).

    Found empirically (not derived): a Westminster tolerance around 2m
    produces ~4,324 segments, the closest this project could get to the
    paper's own reported 4,822 without overshooting (the segment count
    drops sharply even at very small tolerances - 0.5m already yields
    ~4,365 - because `consolidate_intersections`'s own graph rebuild step
    also merges directionally-duplicate edges, not just literally-
    coincident nodes). Each borough's own true "best" tolerance was not
    separately searched for - the same tolerance is used across boroughs
    for a consistent, disclosed methodology rather than one hand-tuned
    per borough to hit its own exact target count (which would risk
    looking like curve-fitting to the paper's numbers rather than testing
    the actual hypothesis).

    Returns a WGS84 `MultiDiGraph` compatible with every existing
    downstream function (`graph_to_edges_gdf`, `snap_points_to_graph`,
    the line-graph/GAT machinery) unchanged - consolidation happens in a
    projected CRS internally (required by `consolidate_intersections`
    itself, which operates on real distances) and reprojects back before
    returning, so no caller needs to know this happened.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached consolidated graph from %s", cache_path)
        return ox.load_graphml(cache_path)

    undirected = ox.convert.to_undirected(graph)
    projected = ox.projection.project_graph(undirected)
    consolidated = ox.simplification.consolidate_intersections(
        projected, tolerance=tolerance_m, rebuild_graph=True, dead_ends=False,
    )
    result = ox.projection.project_graph(consolidated, to_crs="epsg:4326")
    logger.info(
        "Consolidated graph at tolerance=%.1fm: %d nodes, %d edges (from %d nodes, %d edges)",
        tolerance_m, result.number_of_nodes(), result.number_of_edges(),
        graph.number_of_nodes(), graph.number_of_edges(),
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(result, cache_path)
        logger.info("Cached consolidated graph to %s", cache_path)

    return result


def graph_to_edges_gdf(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Convert an OSMnx graph to an edge GeoDataFrame with a stable segment_id."""
    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    edges = edges.reset_index()  # brings u, v, key back as columns
    edges["segment_id"] = (
        edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)
    )
    # OSMnx sometimes stores list-valued attributes (multiple OSM ways merged
    # into one edge); collapse to a single representative value for modelling.
    for col in ("highway", "name", "maxspeed", "lanes"):
        if col in edges.columns:
            edges[col] = edges[col].apply(
                lambda v: v[0] if isinstance(v, list) and v else v
            )
    return edges


DEGREES_TO_METRES = 111_320  # rough constant (WGS84 latitude); an approximation,
# not a proper projected-CRS distance - fine for a coarse "is this count
# point even near this road" cutoff, not for anything requiring precision.


def snap_points_to_graph(
    points: pd.DataFrame,
    graph: nx.MultiDiGraph,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    max_distance_m: float | None = None,
) -> pd.DataFrame:
    """Snap each point to its nearest graph edge (u, v, key) -> segment_id.

    Generic version used for both collisions (`snap_collisions_to_graph`,
    kept as a thin wrapper for backwards compatibility) and other point data
    such as AADF traffic-count locations, which are sparser than collisions
    and can legitimately sit far from any road (e.g. a count point on a
    trunk road just outside the study area) - `max_distance_m` lets the
    caller refuse a match rather than silently attaching a point to a
    segment hundreds of metres away.

    Rows with missing/invalid coordinates are dropped and the drop count is
    logged, matching the dossier's data-quality-gate expectation of visible
    join diagnostics rather than a silent failure.
    """
    valid = points.dropna(subset=[lon_col, lat_col]).copy()
    n_dropped = len(points) - len(valid)
    if n_dropped:
        logger.warning(
            "Dropping %d/%d points with missing coordinates before snapping",
            n_dropped, len(points),
        )

    if valid.empty:
        valid["segment_id"] = pd.Series(dtype="object")
        return valid

    result = ox.distance.nearest_edges(
        graph, X=valid[lon_col].to_numpy(), Y=valid[lat_col].to_numpy(),
        return_dist=max_distance_m is not None,
    )
    if max_distance_m is not None:
        nearest, dist_degrees = result
        dist_m = np.asarray(dist_degrees) * DEGREES_TO_METRES
    else:
        nearest, dist_m = result, None

    # osmnx>=2.0 returns, for multiple points, a 1-D object array of
    # (u, v, key) tuples (not three parallel arrays and not a clean 2-D
    # array) - normalise via list() so this works regardless of exact
    # array shape/dtype.
    edge_ids = [nearest] if len(valid) == 1 and isinstance(nearest, tuple) else list(nearest)
    valid["segment_id"] = [f"{u}_{v}_{key}" for u, v, key in edge_ids]

    if dist_m is not None:
        too_far = dist_m > max_distance_m
        n_too_far = int(too_far.sum())
        if n_too_far:
            logger.warning(
                "%d/%d points snapped further than max_distance_m=%.0fm; setting segment_id to NA for those",
                n_too_far, len(valid), max_distance_m,
            )
        valid.loc[too_far, "segment_id"] = pd.NA

    return valid


def snap_collisions_to_graph(
    collisions: pd.DataFrame,
    graph: nx.MultiDiGraph,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
) -> pd.DataFrame:
    """Thin wrapper over `snap_points_to_graph` (collisions never used a
    distance cutoff - kept for backwards compatibility with existing
    callers/tests)."""
    return snap_points_to_graph(collisions, graph, lon_col=lon_col, lat_col=lat_col, max_distance_m=None)


def redistribute_junction_crashes(
    snapped: pd.DataFrame,
    graph: nx.MultiDiGraph,
    junction_col: str = "junction_detail",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    weight_col: str = "crash_weight",
    per_physical_road: bool = False,
    home_share: float = 0.0,
) -> pd.DataFrame:
    """Spread each junction crash's risk equally across every road segment
    meeting at that junction, instead of attributing it wholly to one
    arbitrary nearest segment.

    `per_physical_road` (added 2026-09-02, then found to be a NO-OP -
    kept only so the finding is not silently lost): the intent was to
    dilute less by splitting across undirected physical roads rather
    than directed edges. It is **mathematically identical** to the
    default whenever roads are bidirectional: 1/n_roads shared between a
    road's two directions is exactly 1/(2*n_roads) per directed edge,
    which is what per-edge splitting already produces. Verified
    empirically (identical weight vectors, identical AccHR@20 on the
    first evaluated window). The test originally written for it asserted
    only per-road totals, which both variants satisfy - a genuinely
    insufficient test, since it could not distinguish the two things it
    was meant to compare.

    `home_share` (added 2026-09-02, the actually-different variant):
    keep `home_share` of the crash's weight on its originally-snapped
    segment and distribute only the remaining `1 - home_share` across
    the junction's other incident edges. `home_share=0.0` reproduces the
    full-dilution default; `home_share=1.0` reproduces no redistribution
    at all. This exists because full redistribution tested significantly
    WORSE (p=0.0432) with a collapsed prediction variance (std 2.66% vs
    the baseline's 11.26%) - strong evidence that spreading risk evenly
    across up to eight arms flattens exactly the sharp segment-level
    distinctions a top-20% ranking task depends on. A partial share is
    the principled middle ground between "all risk on one arbitrary arm"
    and "risk smeared uniformly over the whole junction".

    Implements the balanced weighting scheme Gao et al. describe in their
    own methodology (thesis Section 7.2.1): "To further address the
    complexity of crash occurrences at intersections, where multiple road
    segments converge, the methodology implements a balanced weighting
    scheme, whereas the associated risk value is distributed equally among
    all connected road segments." Added 2026-09-02 - the largest
    unimplemented methodological difference remaining against that paper.

    **Why this matters quantitatively, not just in principle**: STATS19's
    own `junction_detail` field (0 = "not at or within 20 metres of a
    junction", >0 = at a junction of some type, <0 = missing) says
    **63-72% of all collisions in the three study boroughs are
    junction-related** (Westminster 71.9%, Lambeth 67.7%, Tower Hamlets
    63.2%, measured over 2022-2024). Attributing all of that risk to a
    single segment concentrates two-thirds of the signal onto whichever
    approach arm happened to be nearest the recorded coordinate - an
    essentially arbitrary choice between several segments that share the
    junction's risk in reality.

    Each junction crash is assigned to the graph node nearest its recorded
    coordinate (the junction itself), then emitted once per edge incident
    to that node with `weight_col = 1/n`. Non-junction crashes (and those
    with a missing `junction_col`, which are left alone rather than
    guessed at) keep their existing single-segment assignment with
    `weight_col = 1.0`. Total weight per crash is 1.0 either way, so this
    redistributes risk without inflating or deflating the total - callers
    aggregating with `sum(weight_col)` instead of `size` get a
    same-magnitude target.

    Returns a new DataFrame (the input is not modified) that may be LONGER
    than the input, since one junction crash becomes several weighted rows.
    Callers must aggregate on `weight_col`, not row counts - see
    `features.daily_features.collision_severity_counts_by_segment_day`'s
    own `weight_col` parameter.

    If `junction_col` is absent entirely, every row is passed through
    unchanged with weight 1.0 - so this is safe to call on collision
    tables that predate the field, and degrades to exactly the previous
    behaviour rather than failing.
    """
    result = snapped.copy()
    if junction_col not in result.columns:
        logger.warning(
            "No '%s' column found - passing every crash through unredistributed (weight 1.0)",
            junction_col,
        )
        result[weight_col] = 1.0
        return result

    junction_flag = pd.to_numeric(result[junction_col], errors="coerce")
    at_junction = (junction_flag > 0) & result[lon_col].notna() & result[lat_col].notna()
    n_junction = int(at_junction.sum())

    plain = result.loc[~at_junction].copy()
    plain[weight_col] = 1.0

    if n_junction == 0:
        logger.info("No at-junction crashes to redistribute; all %d rows kept as-is", len(plain))
        return plain

    junction_rows = result.loc[at_junction].copy()
    nearest_nodes = ox.distance.nearest_nodes(
        graph, X=junction_rows[lon_col].to_numpy(), Y=junction_rows[lat_col].to_numpy(),
    )
    # osmnx returns a scalar for a single point and an array for many.
    node_list = [nearest_nodes] if np.isscalar(nearest_nodes) else list(nearest_nodes)

    expanded_rows = []
    n_orphan = 0
    for (_, row), node in zip(junction_rows.iterrows(), node_list):
        # Incident edges must come from the DIRECTED graph: `segment_id` is
        # built as "u_v_key" from `graph_to_edges_gdf`, which iterates the
        # directed edges - taking them from an undirected copy would emit
        # reversed (u, v) pairs that match no real segment_id, silently
        # dropping the redistributed weight at join time.
        incident = sorted(
            set(graph.out_edges(node, keys=True)) | set(graph.in_edges(node, keys=True))
        )
        if per_physical_road and incident:
            # Group the directed edges by the undirected physical road they
            # represent, so a four-arm junction splits FOUR ways (each road
            # getting 1/4, shared between its two directions) rather than
            # eight. OS Open Roads has no oneway attribute and so stores
            # every street as two opposing directed edges - splitting per
            # directed edge therefore dilutes each crash twice as hard as
            # the paper's own graph (where road segments are nodes) implies.
            by_road: dict[tuple, list[tuple]] = {}
            for u, v, key in incident:
                by_road.setdefault((frozenset((u, v)), key), []).append((u, v, key))
            share_per_road = 1.0 / len(by_road)
            for edges_for_road in by_road.values():
                # Split each physical road's share evenly between whichever
                # of its directions actually exist, so the road's total is
                # `share_per_road` regardless of one-way-ness.
                per_edge = share_per_road / len(edges_for_road)
                for u, v, key in edges_for_road:
                    new_row = row.copy()
                    new_row["segment_id"] = f"{u}_{v}_{key}"
                    new_row[weight_col] = per_edge
                    expanded_rows.append(new_row)
            continue
        if not incident:
            # An isolated node (no incident edges at all) - keep the crash
            # on its originally-snapped segment rather than dropping it.
            n_orphan += 1
            keep = row.copy()
            keep[weight_col] = 1.0
            expanded_rows.append(keep)
            continue
        home_segment = row.get("segment_id")
        others = [e for e in incident if f"{e[0]}_{e[1]}_{e[2]}" != home_segment]
        if home_share > 0.0 and others:
            # Keep `home_share` on the originally-snapped segment and spread
            # only the remainder over the junction's other arms - see the
            # docstring on why full dilution measurably flattens the ranking.
            keep = row.copy()
            keep[weight_col] = home_share
            expanded_rows.append(keep)
            spread = (1.0 - home_share) / len(others)
            for u, v, key in others:
                new_row = row.copy()
                new_row["segment_id"] = f"{u}_{v}_{key}"
                new_row[weight_col] = spread
                expanded_rows.append(new_row)
            continue
        share = 1.0 / len(incident)
        for u, v, key in incident:
            new_row = row.copy()
            new_row["segment_id"] = f"{u}_{v}_{key}"
            new_row[weight_col] = share
            expanded_rows.append(new_row)

    if n_orphan:
        logger.warning("%d at-junction crashes snapped to a node with no incident edges; left on their original segment", n_orphan)

    expanded = pd.DataFrame(expanded_rows)
    combined = pd.concat([plain, expanded], ignore_index=True)
    logger.info(
        "Redistributed %d/%d at-junction crashes across their connected segments (%d rows -> %d weighted rows)",
        n_junction, len(result), len(result), len(combined),
    )
    return combined
