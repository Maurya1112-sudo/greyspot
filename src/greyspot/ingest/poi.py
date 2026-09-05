"""Point-of-Interest (POI) density features - added 2026-09-02 after
re-reading Gao et al.'s PhD thesis a second time (Table 7.2, "Data
Characteristics by Region") found their own feature set includes
"PointofInterest" (source: Ordnance Survey, 20 classes) for every
borough studied - a real input this project had never incorporated,
alongside socio-demographic characteristics (see `ingest.socio_demographic`).

**Disclosed substitute, not the literal same source**: OS's own Points
of Interest product is a commercial/premium dataset (unlike OS Open
Roads, it is NOT part of OS OpenData), so it is not freely accessible
the way this project's other Ordnance Survey substitute (OS Open Roads)
is. OpenStreetMap POI tags (`shop`, `amenity`, `leisure`, `tourism`) are
used instead - free, no registration, matching this project's
established "free, key-less" substitution pattern (the same OSM data
source this project's own road network already uses as its OSMnx
substitute for OS Open Roads). Four broad categories are used here
(not the paper's literal 20 OS classes) - a coarser but genuinely
comparable "how commercially/socially active is this area" signal, not
a claimed one-to-one match.

Each of the four tag categories is queried SEPARATELY, not combined
into one Overpass request, and by BOUNDING BOX rather than the borough's
full administrative polygon (found 2026-09-02: a single combined
`{'amenity': True, 'shop': True, 'leisure': True, 'tourism': True}`
polygon query for a whole borough timed out after 180s on the public
Overpass API; `amenity` alone by polygon still hadn't returned after 6+
minutes and was killed, while the SAME query by bounding box completed
in ~20s for 21,505 features - Overpass's server-side polygon-clipping
against a complex administrative boundary is evidently far more
expensive than a simple bbox filter). This is the same bbox-not-polygon
approach `ingest.os_open_roads.borough_bbox_wgs84` already uses for
exactly this reason, reused here directly rather than reimplemented.
The real, disclosed cost: a bbox is a rectangle, so a handful of POIs
just outside the borough's actual (non-rectangular) boundary but inside
its bounding box get included - an acceptable, disclosed imprecision
for a density feature, not treated as exact administrative attribution.
"""
from __future__ import annotations

import logging
import time

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

from .network import snap_points_to_graph
from .os_open_roads import borough_bbox_wgs84

logger = logging.getLogger(__name__)

POI_TAG_CATEGORIES = ["shop", "amenity", "leisure", "tourism"]
POI_COUNT_COLUMNS = [f"poi_{category}_count" for category in POI_TAG_CATEGORIES]

_OVERPASS_TIMEOUT_S = 300  # generous, though the bbox queries this module actually uses complete in well under a minute

# Minimum plausible POI-adjacency density (sum of per-segment POI counts
# per km2 of borough bbox). CALIBRATED, not guessed:
#
#   COMPLETE downloads          | TRUNCATED download
#   Westminster           357.5 | Wandsworth (amenity dropped)  23.6
#   Tower Hamlets         280.9 |
#   Kensington & Chelsea  232.1 |
#   Camden                227.6 |
#   Lambeth               151.7 |
#   Brent (OUTER London)   62.3 |
#
# **Brent is why this is 40 and not 75.** The first version of this guard
# used 75, calibrated on the five verified boroughs - every one of them
# INNER London. It then refused Brent at 62.3/km2. That refusal was a false
# positive: two independent downloads of Brent returned byte-identical
# results (9,905 raw POIs, identical per-category counts), and a truncated
# Overpass response cannot be reproducible, because the cut falls in a
# different place each time. Outer London is genuinely less POI-dense than
# inner London, and a floor calibrated on inner boroughs does not transfer.
# See scripts/check_poi_density_calibration.py, which performs that test.
#
# 40 sits 1.7x above the known-truncated download and 1.6x below the
# lowest verified one. That is a thinner margin than the original, and it
# is deliberate: this check is only the BACKSTOP. The primary guard is the
# per-category failure check, which catches the dominant real failure mode
# directly (a dropped category) rather than inferring it from volume - it
# is what actually caught Wandsworth on both occasions. This floor exists
# for the residual case where every category returns but each is truncated.
#
# A borough measuring below this should be run through
# check_poi_density_calibration.py before being either trusted or
# discarded, rather than assumed one way.
_MIN_POI_ADJACENCY_PER_KM2 = 40.0


def _download_category_with_retry(bbox, category: str, osm_place: str, attempts: int = 4):
    """One Overpass category query, retried with exponential backoff.

    The observed failure is not an outage but a truncated response
    (`ChunkedEncodingError: Response ended prematurely`) on the heaviest
    query - `amenity` over a large borough bbox. Wandsworth failed this way
    on 2026-09-05, twice, while every other category succeeded and while
    Overpass itself reported free slots. A transient truncation deserves a
    retry; what it must never get is silent acceptance, which is what
    produced the cached 1,637-POI Wandsworth file.

    Returns the GeoDataFrame, or None if every attempt failed.
    """
    for attempt in range(1, attempts + 1):
        try:
            return ox.features_from_bbox(bbox, {category: True})
        except Exception as exc:
            if attempt == attempts:
                logger.warning("POI category '%s' failed for %s after %d attempts: %s",
                               category, osm_place, attempts, exc)
                return None
            delay = 5 * 2 ** (attempt - 1)  # 5s, 10s, 20s
            logger.warning("POI category '%s' attempt %d/%d failed for %s (%s) - retrying in %ds",
                           category, attempt, attempts, osm_place, type(exc).__name__, delay)
            time.sleep(delay)
    return None


class PoiDownloadError(RuntimeError):
    """An Overpass response that must NOT be cached as if it were complete.

    Raised rather than warned because the failure mode is silent: a partial
    download produces a plausible-looking POI file, gets cached, and every
    subsequent run reuses it. That happened twice on 2026-09-05 (Wandsworth
    cached 1,637 adjacencies against an expected ~10,000; Brent cached 0)
    and both borough results had to be quarantined and discarded. Rule R14
    in docs/MASTER_PLAN.md exists because of it - this is that rule
    enforced in code rather than by memory.
    """


def download_borough_pois(osm_place: str, *, strict: bool = True) -> gpd.GeoDataFrame:
    """Fetches OSM POI features (points, lines, or polygons - a park's
    `leisure=park` boundary is as valid a POI here as a shop's point) for
    one OSMnx-geocodable place's bounding box, one tag category at a
    time (see this module's own docstring for why bbox-not-polygon and
    why one-category-at-a-time), concatenated into a single GeoDataFrame
    with a `poi_category` column recording which of `POI_TAG_CATEGORIES`
    each row came from. A category that returns no features (or whose
    Overpass query fails) is logged and skipped, not treated as a fatal
    error - POI coverage varying by borough/category is expected, not a
    bug."""
    ox.settings.requests_timeout = _OVERPASS_TIMEOUT_S
    bbox = borough_bbox_wgs84(osm_place)
    frames = []
    failed: list[str] = []
    for category in POI_TAG_CATEGORIES:
        gdf = _download_category_with_retry(bbox, category, osm_place)
        if gdf is None:
            failed.append(category)
            continue
        if gdf.empty:
            logger.info("POI category '%s' returned 0 features for %s", category, osm_place)
            continue
        # Keep the tag's own VALUE (e.g. amenity="school" vs
        # amenity="pub"), not just which tag family matched. The original
        # version selected `[["geometry"]]` only, which was sufficient for
        # the 4 coarse family counts but silently made the 20-class
        # taxonomy (`POI_FINE_CLASSES`) impossible - every POI matched no
        # class, because the column it classifies on had been discarded
        # upstream. Found 2026-09-03 when `count_pois_near_segments_fine`
        # reported "20040/20040 POIs matched none of the 20 fine classes".
        gdf = gdf.reset_index()
        keep = ["geometry"] + ([category] if category in gdf.columns else [])
        gdf = gdf[keep].copy()
        gdf["poi_category"] = category
        # Normalise to a single `poi_tag_value` column so downstream code
        # does not need to know which of the four tag columns is populated
        # for a given row (they are mutually exclusive by construction -
        # one query per category).
        gdf["poi_tag_value"] = gdf[category] if category in gdf.columns else pd.NA
        logger.info("Downloaded %d '%s' POIs for %s", len(gdf), category, osm_place)
        frames.append(gdf)
    if strict and failed:
        raise PoiDownloadError(
            f"{osm_place}: Overpass failed for POI categor{'y' if len(failed) == 1 else 'ies'} "
            f"{failed}. The other categories downloaded fine, so the result LOOKS valid while "
            f"missing whole feature classes. Refusing to return a partial download - retry when "
            f"Overpass recovers (https://overpass-api.de/api/status)."
        )
    if not frames:
        if strict:
            raise PoiDownloadError(
                f"{osm_place}: every POI category returned zero features. No London borough has no "
                f"shops, amenities, leisure or tourism POIs - this is an Overpass failure, not an "
                f"empty borough."
            )
        return gpd.GeoDataFrame({"geometry": [], "poi_category": []}, geometry="geometry", crs="epsg:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def bbox_area_km2(bbox) -> float:
    """Approximate area of a `borough_bbox_wgs84` bbox.

    That function returns `GeoDataFrame.total_bounds`, documented and fixed
    as (minx, miny, maxx, maxy) = (west, south, east, north), so the values
    are unpacked positionally. An earlier version tried to identify which
    pair were latitudes by numeric range and silently picked wrong in
    London, where latitude (~51.5) and longitude (~-0.1) both sit inside
    +/-90.

    Equirectangular approximation: precision is irrelevant against a
    threshold the two populations straddle by 3x or more. The bbox exceeds
    the true borough polygon, so this under-states density and the guard
    errs toward caution.
    """
    west, south, east, north = (float(v) for v in bbox)
    height_km = abs(north - south) * 110.574
    width_km = abs(east - west) * 111.320 * float(np.cos(np.deg2rad((north + south) / 2.0)))
    return max(height_km * width_km, 1e-6)


def assert_poi_counts_plausible(poi_counts: pd.DataFrame, osm_place: str, bbox=None) -> float:
    """Guard the SNAPPED counts before they are cached.

    Catches the case the download-time checks cannot: every category
    returned data and nothing raised, but each response was truncated. This
    runs on the quantity the model actually consumes, and its threshold is
    calibrated on real verified and known-bad downloads (see
    `_MIN_POI_ADJACENCY_PER_KM2`). Returns the measured density.
    """
    total = float(poi_counts[POI_COUNT_COLUMNS].to_numpy().sum())
    # bbox is optional so the ~50 run scripts can call this with one line,
    # regardless of whether they happen to hold a bbox in scope. Geocoding
    # costs one lightweight Nominatim lookup against an Overpass POI
    # download measured in minutes.
    area_km2 = bbox_area_km2(bbox if bbox is not None else borough_bbox_wgs84(osm_place))
    density = total / area_km2
    if density < _MIN_POI_ADJACENCY_PER_KM2:
        raise PoiDownloadError(
            f"{osm_place}: {total:.0f} POI adjacencies over ~{area_km2:.1f} km2 = {density:.1f}/km2, "
            f"below the calibrated floor of {_MIN_POI_ADJACENCY_PER_KM2:.0f}/km2 (complete "
            f"downloads measured 62.3-357.5; the truncated Wandsworth one measured 23.6). Nothing "
            f"errored, but the volume suggests the Overpass response was incomplete. Refusing to "
            f"cache. If this borough may be genuinely low-density, confirm with "
            f"scripts/check_poi_density_calibration.py before trusting or discarding it."
        )
    logger.info("POI density check PASSED for %s: %.0f adjacencies over ~%.1f km2 = %.1f/km2",
                osm_place, total, area_km2, density)
    return density


def count_pois_near_segments(pois: gpd.GeoDataFrame, graph, max_distance_m: float = 50.0) -> pd.DataFrame:
    """Snaps each POI's centroid (a Point, LineString, or Polygon
    geometry all reduce to one representative point the same way) to its
    nearest road segment via `ingest.network.snap_points_to_graph` (the
    same snapping function already used for collisions and AADF count
    points - one consistent "attach a point to the road network"
    convention across this whole project, not a POI-specific
    reimplementation), then counts POIs per segment per category.

    `max_distance_m=50` (tighter than AADF's typical 100m) - a POI more
    than 50m from any road is unlikely to meaningfully affect that
    road's crash risk exposure; a disclosed, reasonable default, not
    empirically tuned.

    Returns one row per `segment_id` with one count column per
    `POI_TAG_CATEGORIES` entry (`POI_COUNT_COLUMNS`) - segments with no
    nearby POI at all are simply absent from the result (filled to 0 by
    `features.daily_features.attach_poi_features` downstream, matching
    this project's "no data != a real zero" convention).
    """
    # geopandas warns that `.centroid` on a geographic (WGS84) CRS is
    # "likely incorrect" - true in general, but negligible here: POI
    # geometries are either points already (centroid = itself, exact) or
    # small building/park footprints at most a few hundred metres
    # across, where the WGS84-degree-based centroid and a properly-
    # projected one differ by a distance far smaller than
    # `max_distance_m`'s own snapping tolerance. Not worth reprojecting.
    centroids = pois.geometry.centroid
    points_df = pd.DataFrame({
        "longitude": centroids.x.values,
        "latitude": centroids.y.values,
        "poi_category": pois["poi_category"].values,
    })
    snapped = snap_points_to_graph(points_df, graph, max_distance_m=max_distance_m)
    snapped = snapped.dropna(subset=["segment_id"])
    counts = (
        snapped.groupby(["segment_id", "poi_category"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(columns=POI_TAG_CATEGORIES, fill_value=0)
    counts.columns = POI_COUNT_COLUMNS
    return counts.reset_index()


# ---------------------------------------------------------------------------
# Fine-grained (20-class) POI taxonomy
# ---------------------------------------------------------------------------
# Gao et al.'s Table 7.2 lists "Point of Interest | Ordnance Survey | 20
# [classes]" as a model input; this project used FOUR coarse OSM tag
# families (shop/amenity/leisure/tourism), so a pub, a school and a
# hospital were all summed into one `poi_amenity_count`. Those land uses
# have very different crash-risk profiles (a school at 15:00, a pub at
# 23:00, a hospital with ambulance traffic), so collapsing them discards
# most of the signal the paper's 20 classes carry.
#
# **A disclosed substitution, not a replication**: OS's own Points of
# Interest product is commercial and its exact 20-class scheme is not
# public here. These 20 classes are derived from OSM tag VALUES, chosen
# to span the land-use types with a plausible mechanism for affecting
# road risk (pedestrian generators, night-time economy, vulnerable-user
# concentrations, vehicle-attracting destinations). The count is 20 to
# match the paper's stated granularity; the membership is this project's
# own, and is stated here rather than buried.
POI_FINE_CLASSES: dict[str, dict[str, set[str]]] = {
    "education":     {"amenity": {"school", "college", "university", "kindergarten", "childcare"}},
    "nightlife":     {"amenity": {"pub", "bar", "nightclub", "biergarten", "casino"}},
    "food":          {"amenity": {"restaurant", "cafe", "fast_food", "food_court", "ice_cream"}},
    "healthcare":    {"amenity": {"hospital", "clinic", "doctors", "pharmacy", "dentist", "veterinary"}},
    "worship":       {"amenity": {"place_of_worship"}},
    "transport":     {"amenity": {"bus_station", "taxi", "ferry_terminal", "car_rental", "bicycle_rental"},
                      "public_transport": {"station", "stop_position", "platform"}},
    "parking":       {"amenity": {"parking", "parking_entrance", "motorcycle_parking", "bicycle_parking"}},
    "fuel":          {"amenity": {"fuel", "charging_station", "car_wash"}},
    "finance":       {"amenity": {"bank", "atm", "bureau_de_change"}},
    "civic":         {"amenity": {"police", "fire_station", "townhall", "courthouse", "post_office", "library", "community_centre"}},
    "entertainment": {"amenity": {"cinema", "theatre", "arts_centre", "events_venue", "social_facility"}},
    "retail_food":   {"shop": {"supermarket", "convenience", "greengrocer", "butcher", "bakery", "alcohol", "deli"}},
    "retail_general":{"shop": {"clothes", "department_store", "variety_store", "shoes", "furniture", "electronics",
                               "hardware", "doityourself", "books", "gift", "jewelry", "florist", "sports", "toys"}},
    "retail_vehicle":{"shop": {"car", "car_repair", "car_parts", "motorcycle", "bicycle", "tyres"}},
    "services":      {"shop": {"hairdresser", "beauty", "laundry", "dry_cleaning", "travel_agency", "estate_agent", "optician"}},
    "accommodation": {"tourism": {"hotel", "hostel", "guest_house", "motel", "apartment"}},
    "attraction":    {"tourism": {"attraction", "museum", "gallery", "artwork", "viewpoint", "zoo", "theme_park"}},
    "park":          {"leisure": {"park", "garden", "nature_reserve", "common", "dog_park"}},
    "sports":        {"leisure": {"pitch", "sports_centre", "stadium", "fitness_centre", "swimming_pool", "track", "golf_course"}},
    "playground":    {"leisure": {"playground", "recreation_ground", "water_park"}},
}
POI_FINE_COLUMNS = [f"poi_{name}_count" for name in POI_FINE_CLASSES]


def classify_poi_fine(pois: gpd.GeoDataFrame) -> pd.Series:
    """Map each POI row to one of `POI_FINE_CLASSES` (or NA if it matches
    none), by looking up its OSM tag VALUE within its tag family.

    Reads the normalised `poi_category` (which tag family) +
    `poi_tag_value` (the value within it) pair that
    `download_borough_pois` emits, falling back to per-tag columns
    (`amenity`, `shop`, ...) when a caller passes a raw OSMnx frame
    instead. Supporting both matters: the fine taxonomy was originally
    written against the raw-frame shape and silently matched NOTHING on
    real pipeline data, because `download_borough_pois` had already
    collapsed those columns away.

    First match wins, in `POI_FINE_CLASSES` insertion order - a feature
    tagged both `amenity=pub` and `tourism=attraction` counts once, as
    nightlife, rather than being double-counted into two classes and
    inflating the total POI density.
    """
    result = pd.Series(pd.NA, index=pois.index, dtype="object")
    normalised = "poi_category" in pois.columns and "poi_tag_value" in pois.columns
    for class_name, tag_values in POI_FINE_CLASSES.items():
        for tag, values in tag_values.items():
            if normalised:
                candidate = (pois["poi_category"] == tag) & pois["poi_tag_value"].isin(values)
            elif tag in pois.columns:
                candidate = pois[tag].isin(values)
            else:
                continue
            result[candidate & result.isna()] = class_name
    return result


def count_pois_near_segments_fine(pois, graph, max_distance_m: float = 50.0) -> pd.DataFrame:
    """The 20-class analogue of `count_pois_near_segments` - identical
    snapping (same `snap_points_to_graph`, same 50m cutoff), but counting
    into `POI_FINE_COLUMNS` instead of the four coarse tag families.

    POIs matching none of the 20 classes are dropped rather than pooled
    into an "other" bucket: an undifferentiated catch-all would be
    dominated by whatever OSM happens to tag densely in a given borough
    and carries no consistent land-use meaning across boroughs.
    """
    centroids = pois.geometry.centroid
    fine = classify_poi_fine(pois)
    points_df = pd.DataFrame({
        "longitude": centroids.x.values,
        "latitude": centroids.y.values,
        "poi_fine_class": fine.values,
    }).dropna(subset=["poi_fine_class"])
    n_dropped = len(pois) - len(points_df)
    if n_dropped:
        logger.info("%d/%d POIs matched none of the %d fine classes and were dropped",
                    n_dropped, len(pois), len(POI_FINE_CLASSES))
    snapped = snap_points_to_graph(points_df, graph, max_distance_m=max_distance_m)
    snapped = snapped.dropna(subset=["segment_id"])
    counts = snapped.groupby(["segment_id", "poi_fine_class"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(POI_FINE_CLASSES), fill_value=0)
    counts.columns = POI_FINE_COLUMNS
    return counts.reset_index()
