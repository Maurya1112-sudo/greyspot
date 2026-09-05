"""Is a low POI density a TRUNCATED download, or a genuinely sparser borough?

The density guard added on 2026-09-05 refused Brent at 62.3 adjacencies/km2
against a floor of 75. That floor was calibrated on Camden, Kensington &
Chelsea, Lambeth, Tower Hamlets and Westminster - **all five of which are
inner London**. Brent is outer London, where lower POI density is expected
rather than suspicious, so the refusal may be a false positive of my own
calibration rather than a real data problem.

Guessing either way is not acceptable: accepting a truncated download
silently corrupts a borough's results (that is the whole reason the guard
exists), and rejecting a valid one drops a borough from the generalisation
study for no reason.

**The discriminator is reproducibility.** A truncated Overpass response is
a transport failure - the cut lands at a different point each time, so
repeated downloads disagree. A complete response is deterministic: the same
bbox and tags return the same features. So this script downloads twice and
compares. Stable counts across runs mean the response is complete and the
floor needs recalibrating for outer London; unstable counts mean the
response really is being truncated.

Run: python scripts/check_poi_density_calibration.py Brent
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.network import graph_to_edges_gdf  # noqa: E402
from greyspot.ingest.os_open_roads import borough_bbox_wgs84  # noqa: E402
from greyspot.ingest.poi import (  # noqa: E402
    POI_COUNT_COLUMNS,
    POI_TAG_CATEGORIES,
    bbox_area_km2,
    count_pois_near_segments,
    download_borough_pois,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

INTERIM_DIR = ROOT / "data" / "interim"


def main(borough_name: str, repeats: int = 2) -> None:
    import osmnx as ox

    b = get_borough(borough_name)
    bbox = borough_bbox_wgs84(b.osm_place)
    area = bbox_area_km2(bbox)
    graph = ox.load_graphml(
        INTERIM_DIR / f"{slug(b.name)}_os_open_roads_graph.graphml",
        node_dtypes={"osmid": str}, edge_dtypes={"osmid": str},
    )
    print("=" * 74)
    print("POI DOWNLOAD REPRODUCIBILITY CHECK: %s" % b.name)
    print("=" * 74)
    print("bbox area ~%.1f km2, %d road segments" % (area, graph.number_of_edges()))
    print()

    runs = []
    for i in range(1, repeats + 1):
        # strict=False so a partial download is RETURNED for inspection
        # rather than raising - the whole point here is to measure it.
        pois = download_borough_pois(b.osm_place, strict=False)
        by_cat = pois.poi_category.value_counts().to_dict()
        counts = count_pois_near_segments(pois, graph)
        adj = float(counts[POI_COUNT_COLUMNS].to_numpy().sum())
        runs.append({"run": i, "raw_pois": len(pois), "adjacencies": adj,
                     "adj_per_km2": adj / area,
                     **{c: by_cat.get(c, 0) for c in POI_TAG_CATEGORIES}})
        print("run %d: %6d raw POIs -> %7.0f adjacencies = %6.1f/km2   %s"
              % (i, len(pois), adj, adj / area,
                 " ".join("%s=%d" % (c[:4], by_cat.get(c, 0)) for c in POI_TAG_CATEGORIES)))

    df = pd.DataFrame(runs)
    print()
    spread = int(df.raw_pois.max() - df.raw_pois.min())
    rel = spread / max(df.raw_pois.mean(), 1)
    print("raw POI spread across %d runs: %d (%.2f%% of mean)" % (repeats, spread, 100 * rel))
    print()
    if rel < 0.01:
        print("VERDICT: REPRODUCIBLE -> the response is COMPLETE.")
        print("  A low density here reflects the borough, not a truncated download.")
        print("  The floor was calibrated on five INNER London boroughs and does not")
        print("  transfer to outer London. Recalibrate rather than discard the borough.")
    else:
        print("VERDICT: UNSTABLE -> the response is being TRUNCATED.")
        print("  Repeated downloads disagree, which a complete query cannot do.")
        print("  The guard is correct to refuse this borough.")

    out = ROOT / "reports" / ("poi_density_calibration_%s.csv" % slug(b.name))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Brent")
