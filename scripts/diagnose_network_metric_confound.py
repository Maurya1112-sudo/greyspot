"""Does the OS Open Roads network make the MODEL better, or the TASK easier?

The project's strongest claim is that switching from an OSMnx-derived
network to the real OS Open Roads survey network is worth +5.86 to
+14.46 AccHR@20 points (p=0.0010 pooled). But that comparison changed
TWO things at once:
  - road GEOMETRY/topology (the intended variable)
  - segment COUNT: ~7,552 directed edges -> ~11,100 (about 50% more)

A finer segmentation changes the ranking task itself: the top-20% bucket
holds ~2,220 segments instead of ~1,510, each covering less road. If
that alone makes crashes easier to capture, part of the "network effect"
is a property of the metric rather than of the model - exactly the
multiple-changes-one-attribution error that produced (and then
un-produced) this project's long-history confusion on 2026-09-04.

**The control, which needs no training at all**: rank segments by a
FIXED, simple feature on BOTH networks and score with the same metric.
A trivial ranker cannot "exploit better topology" - it has no model. So:
  - if the trivial baseline ALSO jumps on OS Open Roads, the network
    change made the TASK easier and the +11.59 is partly a metric artefact
  - if the trivial baseline scores similarly on both, the gain is the
    model genuinely using better road geometry

Rankers used (all parameter-free, all computable on either network):
  random, segment length, node degree, and historical crash count -
  the last being the most informative, since it is the strongest simple
  signal this project has found.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.network import graph_to_edges_gdf, snap_collisions_to_graph  # noqa: E402
from greyspot.ingest.os_open_roads import (  # noqa: E402
    borough_bbox_wgs84,
    borough_polygon_wgs84,
    build_borough_graph_os_open_roads,
)
from greyspot.ingest.stats19 import load_local_authority_collisions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("diagnose_network_metric_confound")

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
OS_GPKG = ROOT / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"
YEARS = [2022, 2023, 2024]
HORIZON = 14
EVAL_STARTS = pd.to_datetime(
    ["2023-07-15", "2023-10-13", "2024-01-11", "2024-04-10", "2024-07-09", "2024-10-07"]
)


def score_rankers(graph, collisions, label):
    segs = [f"{u}_{v}_{k}" for u, v, k in graph.edges(keys=True)]
    sidx = {s: i for i, s in enumerate(segs)}
    n = len(segs)
    edges = graph_to_edges_gdf(graph)
    edges = edges.drop_duplicates(subset="segment_id").set_index("segment_id")

    length = np.zeros(n)
    degree = np.zeros(n)
    for (u, v, k) in graph.edges(keys=True):
        i = sidx[f"{u}_{v}_{k}"]
        sid = f"{u}_{v}_{k}"
        length[i] = float(edges.loc[sid, "length"]) if sid in edges.index else 0.0
        degree[i] = graph.degree(u) + graph.degree(v)

    snapped = snap_collisions_to_graph(collisions, graph).dropna(subset=["segment_id"]).copy()
    snapped["cdate"] = pd.to_datetime(snapped["date"], format="%d/%m/%Y")

    rng = np.random.default_rng(0)
    out = {}
    for name in ["random", "length", "degree", "history_365d"]:
        accs = []
        for s in EVAL_STARTS:
            if name == "random":
                score = rng.random(n)
            elif name == "length":
                score = length
            elif name == "degree":
                score = degree
            else:
                hist = snapped[(snapped.cdate < s) & (snapped.cdate >= s - pd.Timedelta(days=365))]
                score = np.zeros(n)
                for sid, cnt in hist.segment_id.value_counts().items():
                    if sid in sidx:
                        score[sidx[sid]] = cnt
            y = np.zeros((HORIZON, n))
            fut = snapped[(snapped.cdate >= s) & (snapped.cdate < s + pd.Timedelta(days=HORIZON))]
            for sid, d in zip(fut.segment_id.values, fut.cdate.values):
                if sid in sidx:
                    day = (pd.Timestamp(d) - s).days
                    if 0 <= day < HORIZON:
                        y[day, sidx[sid]] += 1
            accs.append(accuracy_hit_rate(y, np.tile(score, (HORIZON, 1)), top_fraction=0.20))
        out[name] = float(np.mean(accs))
    out["_n_segments"] = n
    logger.info("%s: %s", label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def main(borough_name: str = "Lambeth") -> None:
    b = get_borough(borough_name)
    collisions = load_local_authority_collisions(b.ons_code, YEARS, RAW_DIR)

    osm_cache = INTERIM_DIR / f"{slug(b.name)}_graph.graphml"
    if not osm_cache.exists():
        logger.error("OSMnx cache %s missing - cannot compare networks", osm_cache)
        return
    g_osm = ox.load_graphml(osm_cache)
    g_os = build_borough_graph_os_open_roads(
        borough_bbox_wgs84(b.osm_place), OS_GPKG,
        cache_path=INTERIM_DIR / f"{slug(b.name)}_os_open_roads_graph.graphml",
        polygon_wgs84=borough_polygon_wgs84(b.osm_place),
    )

    a = score_rankers(g_osm, collisions, "OSMnx")
    c = score_rankers(g_os, collisions, "OS Open Roads")

    print()
    print("=== TRIVIAL RANKERS ON BOTH NETWORKS (no model, no training) ===")
    print("%-14s %12s %12s %10s" % ("ranker", "OSMnx", "OS Open Rds", "delta"))
    for k in ["random", "length", "degree", "history_365d"]:
        print("%-14s %11.2f%% %11.2f%% %+9.2f" % (k, 100 * a[k], 100 * c[k], 100 * (c[k] - a[k])))
    print()
    print("segments: OSMnx %d -> OS Open Roads %d (%.0f%% more)" % (
        a["_n_segments"], c["_n_segments"], 100 * (c["_n_segments"] / a["_n_segments"] - 1)))
    print()
    print("GNN on the same windows: OSMnx 57.72%% -> OS Open Roads 63.59%% (+5.86)")
    print()
    print("READ: if trivial rankers gain about as much as the GNN, the network")
    print("      change made the TASK easier. If they gain much less, the GNN is")
    print("      genuinely exploiting better road geometry.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Lambeth")
