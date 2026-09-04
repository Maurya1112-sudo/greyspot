"""Build a segment-year feature/target table from snapped collisions + graph edges.

Each row is one (segment_id, year) with:
  - target: collision count in that segment-year (what models predict)
  - static features: road class, length, speed limit, node degree
  - historical features: prior-year count(s), so the historical-rate baseline
    and XGBoost can be compared on the same table.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def collision_counts_by_segment_year(snapped: pd.DataFrame) -> pd.DataFrame:
    """Aggregate snapped collisions into segment-year counts."""
    counts = (
        snapped.assign(year=snapped["collision_year"].astype(int))
        .groupby(["segment_id", "year"])
        .size()
        .rename("collision_count")
        .reset_index()
    )
    return counts


# Columns produced by aggregate_enriched_features that are pure counts
# (fill missing segment-years with 0, like collision_count).
ENRICHED_COUNT_COLUMNS = [
    "n_casualties", "n_fatal_casualties", "n_serious_casualties", "n_slight_casualties",
    "n_pedestrian_casualties", "n_cyclist_casualties", "n_vehicles_involved",
]
# Non-count columns (context, not sums) - missing segment-years stay NaN
# rather than being filled with a misleading 0.
ENRICHED_CONTEXT_COLUMNS = ["avg_imd_decile"]


def aggregate_enriched_features(snapped_enriched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-collision severity/vulnerable-user/vehicle/IMD context
    (already merged onto `snapped_enriched` at the collision level) up to
    segment-year sums/averages.

    IMPORTANT: these describe the *same* collisions that make up
    `collision_count` for that segment-year, so they must only ever be used
    as **lagged** (prior-year) features in the model - see
    `build_segment_year_table`'s `prior_year_*` columns - otherwise the
    model would effectively see the answer restated in more detail.
    """
    grouped = snapped_enriched.assign(year=snapped_enriched["collision_year"].astype(int)).groupby(
        ["segment_id", "year"]
    )
    agg = grouped.agg(
        n_casualties=("n_casualties", "sum"),
        n_fatal_casualties=("n_fatal_casualties", "sum"),
        n_serious_casualties=("n_serious_casualties", "sum"),
        n_slight_casualties=("n_slight_casualties", "sum"),
        n_pedestrian_casualties=("n_pedestrian_casualties", "sum"),
        n_cyclist_casualties=("n_cyclist_casualties", "sum"),
        n_vehicles_involved=("n_vehicles", "sum"),
        avg_imd_decile=("imd_decile", "mean"),
    ).reset_index()
    return agg


EXPOSURE_COLUMNS = ["aadf_all_motor_vehicles", "aadf_pedal_cycles"]


def aggregate_exposure_features(snapped_aadf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate snapped AADF traffic-count points to segment-year exposure.

    Unlike `aggregate_enriched_features`, this is **not** lagged when merged
    into the feature table - AADF is measured independently of collision
    outcomes (it is not derived from the crashes it will help explain), so
    using the same year's traffic volume as a feature for that year's
    collision count is not leakage. A segment with no nearby count point
    gets NaN here, not 0 - "no data" and "no traffic" are different things
    and must not be conflated (dossier Section 5.2).
    """
    grouped = snapped_aadf.dropna(subset=["segment_id"]).groupby(["segment_id", "year"])
    agg = grouped.agg(
        aadf_all_motor_vehicles=("all_motor_vehicles", "mean"),
        aadf_pedal_cycles=("pedal_cycles", "mean"),
    ).reset_index()
    return agg


def node_degrees(graph: nx.MultiDiGraph) -> dict:
    """Undirected degree per node, used as a simple network-importance feature."""
    undirected = graph.to_undirected()
    return dict(undirected.degree())


def build_segment_year_table(
    edges: pd.DataFrame,
    counts: pd.DataFrame,
    graph: nx.MultiDiGraph,
    years: list[int],
    enriched: pd.DataFrame | None = None,
    exposure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-join every segment x every year, fill in counts (0 where none),
    static edge features and lag-1 historical features.

    `enriched` (optional): output of `aggregate_enriched_features` - severity,
    vulnerable-user, vehicle and IMD context per segment-year. Only its
    **lagged** (prior-year) versions are added as model features, since the
    same-year values would leak the target (see that function's docstring).

    `exposure` (optional): output of `aggregate_exposure_features` - AADF
    traffic volume per segment-year. Merged **unlagged** (same year), since
    it is not derived from the collisions it will help explain - see that
    function's docstring.
    """
    degrees = node_degrees(graph)

    static_cols = [c for c in ("segment_id", "highway", "length", "maxspeed", "oneway") if c in edges.columns]
    seg_static = edges[static_cols].drop_duplicates(subset="segment_id").copy()
    seg_static["u_degree"] = edges["u"].map(degrees).values if "u" in edges.columns else None
    seg_static["v_degree"] = edges["v"].map(degrees).values if "v" in edges.columns else None

    segments = seg_static["segment_id"].unique()
    scaffold = pd.MultiIndex.from_product([segments, years], names=["segment_id", "year"]).to_frame(index=False)

    table = scaffold.merge(seg_static, on="segment_id", how="left")
    table = table.merge(counts, on=["segment_id", "year"], how="left")
    table["collision_count"] = table["collision_count"].fillna(0).astype(int)

    if exposure is not None:
        table = table.merge(exposure, on=["segment_id", "year"], how="left")
        # An explicit "do we even have exposure data here" flag, captured
        # BEFORE any fill - so a model can tell "no data" apart from
        # "zero traffic" even after the exposure columns themselves are
        # eventually filled/handled downstream (dossier Section 5.2: never
        # silently substitute a denominator).
        table["has_aadf"] = table["aadf_all_motor_vehicles"].notna().astype(int)

    if enriched is not None:
        table = table.merge(enriched, on=["segment_id", "year"], how="left")
        for col in ENRICHED_COUNT_COLUMNS:
            if col in table.columns:
                table[col] = table[col].fillna(0)
        # ENRICHED_CONTEXT_COLUMNS (avg_imd_decile) intentionally left as NaN
        # where there is no collision that segment-year - a segment with no
        # history has no known IMD context, not an IMD of zero.

    table = table.sort_values(["segment_id", "year"])
    table["prior_year_count"] = table.groupby("segment_id")["collision_count"].shift(1)
    table["prior_2yr_avg"] = (
        table.groupby("segment_id")["collision_count"]
        .rolling(window=2, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    if enriched is not None:
        for col in ENRICHED_COUNT_COLUMNS + ENRICHED_CONTEXT_COLUMNS:
            if col in table.columns:
                table[f"prior_year_{col}"] = table.groupby("segment_id")[col].shift(1)
        # Drop the same-year enriched columns themselves - only their lagged
        # (prior_year_*) versions are safe to use as model features.
        table = table.drop(columns=[c for c in ENRICHED_COUNT_COLUMNS + ENRICHED_CONTEXT_COLUMNS if c in table.columns])

    return table.reset_index(drop=True)
