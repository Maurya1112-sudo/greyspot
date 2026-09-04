"""Daily-granularity segment-day feature/target tables.

Built specifically to run a genuinely comparable evaluation against Gao
et al. (2024)'s STZITD-GNN (see `docs/publication_readiness.md`), whose
task is fundamentally daily-count, 14-day-multi-step forecasting - not
this project's own annual segment-year grain (`build_features.py`),
which cannot be converted into daily numbers after the fact. This module
builds the daily table from scratch, from the same raw STATS19 collision
rows and the same snapped-to-segment join already used everywhere else in
this project - no new data source, just a different aggregation grain.

Deliberately *not* an extension of `build_features.py`'s
`build_segment_year_table`: that function's lag features are single
scalars (`prior_year_count`), which makes sense once per year but not
once per day - a sequence model at daily grain needs a genuine input
*window* of raw daily counts (see `features/daily_temporal.py`), not a
hand-engineered lag column. Static edge attributes are still shared - see
`build_segment_day_scaffold`, which reuses `build_features.node_degrees`.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .build_features import node_degrees
from ..ingest.stats19 import collision_severity_and_vulnerable_user_features

logger = logging.getLogger(__name__)

# Traffic Crash Risk (TCR) severity weight, Gao et al. (2024)'s exact
# target definition (Eq. 7.1 in the PhD thesis, discovery.ucl.ac.uk/id/
# eprint/10210801, "Definition 1"): y_it = sum_k C^t_{i,k} * l_k, where
# l_k in {1, 2, 3} weights minor/serious/fatal crashes respectively (their
# own words: "l is assigned the values 1, 2, and 3, representing minor
# injury, serious injury, and fatal crash severities, respectively").
# STATS19's own `collision_severity` field codes the OPPOSITE numeric
# direction (1=Fatal, 2=Serious, 3=Slight - verified directly against
# data/raw/collision-2024.csv's value distribution, 2026-09-01: 1,502
# fatal / 23,567 serious / 75,858 slight, the expected real-world shape),
# so the weight applied is `4 - collision_severity`, giving fatal->3,
# serious->2, slight->1 - matching the paper's stated weights by MEANING,
# not by reusing STATS19's own numeric code as if it were already the
# weight.
TCR_SEVERITY_WEIGHT = {1: 3, 2: 2, 3: 1}

# Severity/vulnerable-user columns this module can add per segment-day -
# named identically to `ingest.stats19.collision_severity_and_vulnerable_user_features`'s
# own output columns (minus `collision_index`) so annual and daily grain
# never silently disagree on how e.g. "fatal casualty" is counted.
SEVERITY_COLUMNS = [
    "n_casualties", "n_fatal_casualties", "n_serious_casualties", "n_slight_casualties",
    "n_pedestrian_casualties", "n_cyclist_casualties",
]

# Road classes, one-hot. Gao et al.'s Table 7.2 lists "Roads | Ordnance
# Survey | 8 [classes]" as one of their six model inputs; this project
# had the data on every graph edge (OS Open Roads'
# `road_classification`/`road_function`, mapped to an OSM-style
# `highway` value by `ingest.os_open_roads._map_highway`) but never
# passed it to the model - found 2026-09-02 by auditing their table
# class-by-class against this project's own FEATURE_COLUMNS. Road type
# is among the strongest priors for crash risk (exposure differs by
# orders of magnitude between a motorway and a cul-de-sac), so its
# absence was a real, unforced gap rather than a scoping decision.
#
# These eight values are `_map_highway`'s complete output vocabulary,
# listed explicitly rather than discovered per-borough with
# `pd.get_dummies`: a borough containing no motorway would otherwise
# silently produce a DIFFERENT feature-matrix width than one that does,
# breaking cross-borough comparability and the standardiser alike.
ROAD_CLASS_VALUES = [
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "residential", "service", "unclassified",
]
ROAD_CLASS_COLUMNS = [f"road_class_{v}" for v in ROAD_CLASS_VALUES]


# Date features. Gao et al.'s Table 7.2 lists "Date | / | 4 [classes]"
# as a model input; this project used exactly one (`day_of_week`).
# These three bring it to four in total. `month_sin`/`month_cos` encode
# seasonal position CYCLICALLY rather than as a raw 1-12 integer, so
# December and January sit adjacent as they physically are, instead of
# being 11 apart - the standard treatment for a periodic variable, and
# the reason this is a sin/cos pair rather than one column.
DATE_FEATURE_COLUMNS = ["is_weekend", "month_sin", "month_cos"]


def attach_date_features(table: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add `DATE_FEATURE_COLUMNS` alongside the existing `day_of_week`,
    bringing this project's date inputs to the four the paper lists."""
    result = table.copy()
    dates = pd.to_datetime(result[date_col])
    result["is_weekend"] = (dates.dt.dayofweek >= 5).astype("float32")
    month_angle = 2.0 * np.pi * (dates.dt.month - 1) / 12.0
    result["month_sin"] = np.sin(month_angle).astype("float32")
    result["month_cos"] = np.cos(month_angle).astype("float32")
    return result


def attach_road_class_features(table: pd.DataFrame, road_class_col: str = "highway") -> pd.DataFrame:
    """One-hot encode each segment's road class into `ROAD_CLASS_COLUMNS`.

    Static per segment (like `length`), so the value simply broadcasts
    across every day of that segment's rows. Any class outside
    `ROAD_CLASS_VALUES` (or a missing value) yields an all-zero row
    rather than a new column, keeping the feature width fixed at 8
    across every borough and both network sources - see
    `ROAD_CLASS_VALUES`'s own comment on why the vocabulary is fixed
    rather than inferred.
    """
    result = table.copy()
    if road_class_col not in result.columns:
        logger.warning(
            "No '%s' column found - road-class features will be all-zero (feature width preserved)",
            road_class_col,
        )
        for col in ROAD_CLASS_COLUMNS:
            result[col] = 0.0
        return result

    values = result[road_class_col].astype("object").where(result[road_class_col].notna(), None)
    for value, col in zip(ROAD_CLASS_VALUES, ROAD_CLASS_COLUMNS):
        result[col] = (values == value).astype("float32")

    matched = result[ROAD_CLASS_COLUMNS].to_numpy().sum(axis=1)
    n_unmatched = int((matched == 0).sum())
    if n_unmatched:
        unknown = sorted(set(values[matched == 0].dropna().unique()))[:10]
        logger.warning(
            "%d/%d segment-day rows have a road class outside the fixed vocabulary (e.g. %s) - encoded as all-zero",
            n_unmatched, len(result), unknown,
        )
    return result


def parse_stats19_date(date_col: pd.Series) -> pd.Series:
    """STATS19's `date` field is `DD/MM/YYYY` strings (verified directly
    against `data/raw/collision-2024.csv`, 2026-09-01) - not ISO 8601, and
    not something `pd.to_datetime`'s dayfirst-guessing should be trusted
    with silently (`day_of_week` is also in the raw file and could
    cross-check a wrong parse, but stating the format explicitly removes
    the ambiguity outright)."""
    return pd.to_datetime(date_col, format="%d/%m/%Y")


def collision_counts_by_segment_day(snapped: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Aggregate snapped collisions into segment-day counts - the daily
    analogue of `build_features.collision_counts_by_segment_year`.

    Prefers the `datetime` column `ingest.stats19.load_collision_years`
    already adds (`date` + `time` combined, same `%d/%m/%Y %H:%M` format
    parsed the same way - see that module) over re-parsing `date_col`
    itself, so this never duplicates/risks disagreeing with that parsing
    logic. Falls back to parsing `date_col` directly (via
    `parse_stats19_date`) only when no `datetime` column is present -
    e.g. a caller passing in a raw STATS19 slice that hasn't gone through
    `load_collision_years`, or this module's own unit tests' synthetic
    fixtures.
    """
    if "datetime" in snapped.columns:
        dates = snapped["datetime"].dt.normalize()
    else:
        dates = parse_stats19_date(snapped[date_col])
    counts = (
        snapped.assign(date=dates)
        .groupby(["segment_id", "date"])
        .size()
        .rename("collision_count")
        .reset_index()
    )
    return counts


def collision_severity_counts_by_segment_day(
    snapped: pd.DataFrame, casualties: pd.DataFrame, date_col: str = "date",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """The daily analogue of `build_features.aggregate_enriched_features`:
    segment-day counts of collisions **and** their severity/vulnerable-user
    breakdown (fatal/serious/slight casualties, pedestrian/cyclist
    casualties), not just a bare collision count. Also computes
    `tcr_score` - Gao et al.'s exact Traffic Crash Risk target (see
    `TCR_SEVERITY_WEIGHT`'s docstring) - so a caller can use the plain
    `collision_count` as an input FEATURE (this project's own, disclosed,
    target definition for every other pipeline) while using `tcr_score` as
    the TARGET for a genuinely paper-matched comparison run, without
    running two separate aggregation passes over the same collision data.

    Reuses `ingest.stats19.collision_severity_and_vulnerable_user_features`
    (the exact same per-collision aggregation the annual pipeline already
    uses) rather than re-deriving severity counting logic here - annual and
    daily grain must never be able to silently disagree on what counts as a
    "fatal casualty". `casualties` is the raw STATS19 casualty table for the
    relevant years (e.g. `ingest.stats19.load_casualty_years(...)`), joined
    onto `snapped` via `collision_index`, the same key the annual pipeline
    joins on (`ingest.stats19.py`).

    Filters `casualties` down to only the collision_index values present in
    `snapped` before the per-collision aggregation - callers commonly pass
    in a *national* casualty table (`load_casualty_years` has no
    borough-filtering of its own, matching the annual pipeline's existing
    convention), and `collision_severity_and_vulnerable_user_features`'s
    per-group `.apply()` calls scale with the number of *national*
    collision groups if not pre-filtered - several minutes on the full GB
    casualty table versus a few seconds once cut down to one borough's
    ~5-30k collisions (measured 2026-09-01 building this).
    """
    relevant = casualties[casualties["collision_index"].isin(snapped["collision_index"])]
    severity = collision_severity_and_vulnerable_user_features(relevant)
    merged = snapped.merge(severity, on="collision_index", how="left")

    if "datetime" in merged.columns:
        dates = merged["datetime"].dt.normalize()
    else:
        dates = parse_stats19_date(merged[date_col])

    assign_kwargs = {"date": dates}
    agg_kwargs: dict[str, tuple[str, str]] = {}

    if weight_col is not None:
        # Weighted mode (see `ingest.network.redistribute_junction_crashes`):
        # one junction crash arrives as several rows summing to weight 1.0,
        # so every count must be a weighted SUM, never a row count - a plain
        # `size` here would multiply each junction crash by its number of
        # connected segments instead of dividing it among them.
        if weight_col not in merged.columns:
            raise KeyError(
                f"weight_col='{weight_col}' not present in the snapped collisions table - "
                "pass the output of `redistribute_junction_crashes` (which adds it), or leave weight_col=None."
            )
        weights = merged[weight_col].astype(float)
        agg_kwargs["collision_count"] = ("_row_weight", "sum")
        assign_kwargs["_row_weight"] = weights
        for col in SEVERITY_COLUMNS:
            if col in merged.columns:
                assign_kwargs[f"_w_{col}"] = merged[col].fillna(0) * weights
                agg_kwargs[col] = (f"_w_{col}", "sum")
        if "collision_severity" in merged.columns:
            assign_kwargs["_tcr_weight"] = merged["collision_severity"].map(TCR_SEVERITY_WEIGHT).fillna(0) * weights
            agg_kwargs["tcr_score"] = ("_tcr_weight", "sum")
    else:
        agg_kwargs["collision_count"] = ("collision_index", "size")
        for col in SEVERITY_COLUMNS:
            if col in merged.columns:
                agg_kwargs[col] = (col, "sum")
        if "collision_severity" in merged.columns:
            assign_kwargs["_tcr_weight"] = merged["collision_severity"].map(TCR_SEVERITY_WEIGHT).fillna(0)
            agg_kwargs["tcr_score"] = ("_tcr_weight", "sum")

    counts = merged.assign(**assign_kwargs).groupby(["segment_id", "date"]).agg(**agg_kwargs).reset_index()
    return counts


def attach_long_history_features(
    table: pd.DataFrame,
    snapped: pd.DataFrame,
    lookbacks: tuple[int, ...] = (730, 1095, 1825),
    date_col: str = "date",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """Multi-year crash-history counts, computed WITHOUT extending the
    segment-day table back over the whole history period.

    **Why this exists.** `attach_rolling_collision_features` computes its
    windows by rolling over the segment-day table itself, so a 1,825-day
    (5-year) window would require the table to span five extra years:
    ~11.6k segments x ~2,557 extra days = ~30M additional rows before any
    feature columns, which does not fit in this machine's memory.

    This computes the same quantity from the *sparse* collision list
    instead. A daily count matrix [n_segments, n_days] over
    (table_start - max_lookback, table_end) is ~135 MB for five years of
    history, and every lookback is then a difference of two columns of
    its cumulative sum - so arbitrarily long horizons cost no extra
    memory beyond the one matrix.

    **Why long horizons matter here** (measured on real Lambeth data,
    ranking segments by history alone on the project's six evaluation
    windows - see docs/decision_log.md):

    | Lookback | AccHR@20 | Segments with non-zero history |
    |---|---|---|
    | 30d | 21.86% | 78 / 11,596 |
    | 365d | 50.50% | 672 / 11,596 |
    | 730d | 60.26% | 1,177 / 11,596 |
    | 1095d | 70.15% | 1,565 / 11,596 |

    **There is no plateau** - the curve is still climbing steeply at
    three years. The reason is mechanical: the top-20% bucket needs
    2,319 segments, but a 365-day window leaves 94.2% of segments tied
    at exactly zero, so the ranking among them is arbitrary. Longer
    horizons break those ties with real signal. (An earlier LSOA-level
    proxy measurement suggested a plateau at 365d; that was misleading,
    because Lambeth's 201 LSOAs are ~58x denser than its segments.)

    Convention matches `attach_rolling_collision_features`: the window is
    backward-looking and INCLUSIVE of the current day, i.e. the value at
    date d for lookback L counts crashes in [d - L + 1, d]. No leakage is
    introduced - every target window in
    `daily_temporal.build_daily_multistep_instances` begins strictly
    after its input window ends.

    `snapped` is a snapped collision table (one row per collision, with
    `segment_col` and a datetime `date_col`); it should span further back
    than `table` does, otherwise the long windows are silently truncated
    and a warning is logged.
    """
    result = table.copy()
    if snapped.empty:
        for lb in lookbacks:
            result[f"collision_count_{lb}d"] = 0.0
        logger.warning("No snapped collisions supplied - long-history features zero-filled")
        return result

    table_dates = pd.to_datetime(pd.Series(result[date_col].unique())).sort_values()
    min_date, max_date = table_dates.iloc[0], table_dates.iloc[-1]
    max_lb = max(lookbacks)

    snapped_dates = pd.to_datetime(snapped[date_col])
    history_available = (min_date - snapped_dates.min()).days
    if history_available < max_lb:
        logger.warning(
            "Only %d days of collision history before the table's start, but the longest "
            "lookback is %d days - the longest windows are truncated (partial sums)",
            history_available, max_lb,
        )

    # Day axis spans the deepest lookback so every table date has a full window.
    axis_start = min_date - pd.Timedelta(days=max_lb)
    all_days = pd.date_range(axis_start, max_date, freq="D")
    day_pos = {d: i for i, d in enumerate(all_days)}

    segments = result[segment_col].astype(str).unique()
    seg_pos = {s: i for i, s in enumerate(segments)}

    counts = np.zeros((len(segments), len(all_days)), dtype=np.int32)
    for sid, d in zip(snapped[segment_col].astype(str).to_numpy(), snapped_dates.to_numpy()):
        si = seg_pos.get(sid)
        if si is None:
            continue
        di = day_pos.get(pd.Timestamp(d))
        if di is not None:
            counts[si, di] += 1

    # Prepend a zero column so a lookback reaching before the axis start
    # reads 0 rather than wrapping around to the end of the array.
    cum = np.concatenate([np.zeros((len(segments), 1), dtype=np.int64),
                          np.cumsum(counts, axis=1, dtype=np.int64)], axis=1)

    row_seg = result[segment_col].astype(str).map(seg_pos).to_numpy()
    row_day = pd.to_datetime(result[date_col]).map(day_pos).to_numpy()
    valid = ~(pd.isna(row_seg) | pd.isna(row_day))
    row_seg_i = np.where(valid, row_seg, 0).astype(np.int64)
    row_day_i = np.where(valid, row_day, 0).astype(np.int64)

    for lb in lookbacks:
        # inclusive of the current day: cum[d+1] - cum[d+1-lb]
        hi = row_day_i + 1
        lo = np.maximum(hi - lb, 0)
        vals = (cum[row_seg_i, hi] - cum[row_seg_i, lo]).astype("float32")
        result[f"collision_count_{lb}d"] = np.where(valid, vals, 0.0).astype("float32")

    logger.info(
        "Long-history features %s computed over %d segments x %d days (history depth %d days)",
        [f"collision_count_{lb}d" for lb in lookbacks], len(segments), len(all_days), history_available,
    )
    return result


def attach_rolling_collision_features(
    table: pd.DataFrame, windows: tuple[int, ...] = (7, 14, 30), count_col: str = "collision_count"
) -> pd.DataFrame:
    """Adds causal trailing-window sums of `count_col`, e.g. `collision_count_7d`
    = the count on this segment summed over the 7 days up to and including
    this row's own date.

    The single highest-priority gap this closes: before this function
    existed, nothing in the daily/UCL-comparable pipeline (see
    `docs/publication_readiness.md`) ever gave the model *any* view of a
    segment's own collision history - `FEATURE_COLUMNS` in
    `scripts/run_ucl_comparison.py` was `["length", "day_of_week",
    "u_degree", "v_degree"]` only, meaning the model was ranking 7,500+
    roads by risk using nothing but static topology and calendar day - not
    even the raw `collision_count` sequence its own GRU encoder was built
    to consume. Real-world collision risk is dominated by collision
    history (a location that has crashed before is disproportionately
    likely to crash again); a model with zero access to that signal cannot
    rank well regardless of architecture.

    Safe to use as a model INPUT feature at any date, including dates
    inside a training window: this only ever aggregates the row's own date
    and earlier dates within the same segment (`rolling(window=w)` looks
    backward, never forward), and every training/held-out target window in
    `daily_temporal.build_daily_multistep_instances` starts strictly after
    its own input window ends - so a rolling feature computed here can
    never see into a target window, regardless of which instance it's used
    in.
    """
    sorted_table = table.sort_values(["segment_id", "date"]).copy()
    grouped = sorted_table.groupby("segment_id")[count_col]
    for w in windows:
        sorted_table[f"{count_col}_{w}d"] = grouped.transform(
            lambda s, w=w: s.rolling(window=w, min_periods=1).sum()
        )
    return sorted_table.sort_index()


def attach_static_exposure_features(table: pd.DataFrame, exposure_per_segment_year: pd.DataFrame) -> pd.DataFrame:
    """Broadcasts AADF traffic-exposure onto every day for a segment.

    AADF (`ingest.exposure.load_local_authority_aadf`, aggregated per
    segment-year by `build_features.aggregate_exposure_features`) is
    itself an *annual average* by construction - there is no real daily
    exposure series to attach, so the per-year values are averaged across
    whichever years are available into one static per-segment number. This
    is a documented simplification specific to the daily table (the annual
    table instead keeps exposure per-year, since it has a year column to
    key on) - traded for tractability, not because day-to-day exposure
    variation is unmeasurable in principle, just unmeasured by this data
    source. `has_aadf` follows the same "no data ≠ no traffic" convention
    as the annual pipeline (`build_features.py`): a segment with no nearby
    AADF count point gets `has_aadf=0` and `aadf_*=0`, never a silent NaN
    that a model would otherwise treat as an arbitrary missing-value code.
    """
    per_segment = (
        exposure_per_segment_year.groupby("segment_id")[["aadf_all_motor_vehicles", "aadf_pedal_cycles"]]
        .mean()
        .reset_index()
    )
    merged = table.merge(per_segment, on="segment_id", how="left")
    merged["has_aadf"] = merged["aadf_all_motor_vehicles"].notna().astype("int8")
    merged["aadf_all_motor_vehicles"] = merged["aadf_all_motor_vehicles"].fillna(0.0)
    merged["aadf_pedal_cycles"] = merged["aadf_pedal_cycles"].fillna(0.0)
    return merged


def propagate_aadf_by_road_name(table: pd.DataFrame, segment_road_names: pd.DataFrame) -> pd.DataFrame:
    """Extend AADF coverage from measured segments to unmeasured segments
    of the SAME physical named road, added 2026-09-03 after a real-data
    audit found `has_aadf` covers only 1.56% of Westminster's segments
    (173/11,098) - DfT's AADF count points are genuine official traffic
    monitors, but they are sparse by design (placed on a representative
    subset of roads, not installed on every street), so the vast
    majority of segments get `aadf_*=0`/`has_aadf=0` even when they sit
    on an objectively busy, well-monitored road.

    **The assumption, stated plainly**: traffic volume is roughly
    homogeneous along one continuous named road between major junctions
    - a standard simplification in transport engineering when direct
    counts are sparse, not an invented number. A segment named "Harrow
    Road" with no count point of its own is filled with the MEAN AADF of
    every OTHER "Harrow Road" segment that does have one. Segments whose
    name is shared by no measured segment (a genuinely unmonitored road,
    e.g. most residential streets) are left untouched at 0 - this
    extends real signal, it does not manufacture data for roads with no
    nearby measurement at all.

    Verified on real Westminster data before use: 2,973/10,925
    (27.2%) of AADF-missing segments share a name with a measured
    segment, raising `has_aadf` coverage from 1.56% to 28.35%.

    `segment_road_names` must have columns `segment_id`, `name` (the OS
    Open Roads / OSMnx `name` edge attribute - unrelated to and read
    independently of `road_name_toid`, which this project's cached
    graphs do not retain). Only rows where `has_aadf == 0` are touched;
    already-measured segments and their real values are never
    overwritten.
    """
    result = table.copy()
    named = result.merge(segment_road_names, on="segment_id", how="left")
    # mean REAL AADF per road name, computed only from rows that already
    # have a genuine measurement (has_aadf == 1) - never from the 0-filled
    # rows this function itself is about to fill in.
    measured = named[named["has_aadf"] == 1]
    name_means = measured.groupby("name")[["aadf_all_motor_vehicles", "aadf_pedal_cycles"]].mean()

    missing_mask = named["has_aadf"] == 0
    fillable = named.loc[missing_mask, "name"].map(name_means["aadf_all_motor_vehicles"])
    fillable_cycles = named.loc[missing_mask, "name"].map(name_means["aadf_pedal_cycles"])
    n_recovered = int(fillable.notna().sum())

    result.loc[missing_mask.to_numpy(), "aadf_all_motor_vehicles"] = fillable.fillna(0.0).to_numpy()
    result.loc[missing_mask.to_numpy(), "aadf_pedal_cycles"] = fillable_cycles.fillna(0.0).to_numpy()
    # has_aadf now marks "has a real OR name-propagated estimate" - a
    # separate, more conservative flag could be added if a caller ever
    # needs to distinguish the two; not needed by anything today.
    # `fillable` is indexed only over the missing rows (from `.loc[missing_mask, ...]`
    # above) - reindex it back to the full table's index before combining
    # with `missing_mask`, which is full-length, to avoid a length mismatch.
    recovered = fillable.notna().reindex(result.index, fill_value=False)
    result.loc[(missing_mask & recovered).to_numpy(), "has_aadf"] = 1

    logger.info(
        "AADF name-propagation: recovered %d/%d missing segments (has_aadf coverage %.2f%% -> %.2f%%)",
        n_recovered, int(missing_mask.sum()),
        100 * (table["has_aadf"] == 1).mean(), 100 * (result["has_aadf"] == 1).mean(),
    )
    return result


def attach_socio_demographic_features(table: pd.DataFrame, segment_socio_demographics: pd.DataFrame) -> pd.DataFrame:
    """Broadcasts LSOA-level socio-demographic characteristics
    (`ingest.socio_demographic.snap_segments_to_lsoa` - IMD 2019
    deprivation-domain scores + population density, a disclosed
    substitute for Gao et al.'s "Census 2011 socio-demographic
    characteristics" input, see that module's own docstring) onto every
    day for a segment - static per-segment values, same "no data != a
    real zero" convention as `attach_static_exposure_features`
    (`has_socio_demographic=0` and every score column filled to 0.0 for
    a segment whose midpoint fell outside every LSOA polygon, rather
    than a silent NaN)."""
    from greyspot.ingest.socio_demographic import SOCIO_DEMOGRAPHIC_COLUMNS

    merged = table.merge(segment_socio_demographics, on="segment_id", how="left")
    merged["has_socio_demographic"] = merged[SOCIO_DEMOGRAPHIC_COLUMNS[0]].notna().astype("int8")
    for col in SOCIO_DEMOGRAPHIC_COLUMNS:
        merged[col] = merged[col].fillna(0.0)
    return merged


def attach_poi_features(
    table: pd.DataFrame, segment_poi_counts: pd.DataFrame,
    count_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Broadcasts Point-of-Interest counts per segment
    (`ingest.poi.count_pois_near_segments` - a disclosed OpenStreetMap
    substitute for Gao et al.'s Ordnance Survey Points of Interest
    input, see that module's own docstring) onto every day for a
    segment - static, same "no data != a real zero" convention as the
    other static-feature attachers (`has_poi=0` and every count column
    filled to 0.0 for a segment with no POI within the search radius,
    not a silent NaN)."""
    from greyspot.ingest.poi import POI_COUNT_COLUMNS

    # `count_columns` (added 2026-09-03) lets the same attacher serve the
    # 20-class fine taxonomy (`ingest.poi.POI_FINE_COLUMNS`) as well as
    # the original 4 coarse families; defaults to the coarse set so every
    # existing caller is unaffected.
    columns = list(count_columns) if count_columns is not None else list(POI_COUNT_COLUMNS)

    merged = table.merge(segment_poi_counts, on="segment_id", how="left")
    merged["has_poi"] = merged[columns[0]].notna().astype("int8")
    for col in columns:
        merged[col] = merged[col].fillna(0.0)
    return merged


def attach_daily_weather_features(table: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Broadcasts one London-wide daily weather reading onto every segment
    for that day (`ingest.weather.load_london_daily_weather`) - added
    2026-09-01 after re-reading Gao et al.'s PhD thesis (Table 7.2)
    confirmed their own feature set includes Met Office meteorological
    data, which this project's daily pipeline had none of before now.

    Unlike AADF (per-segment, static across days), weather is per-DAY,
    shared across every segment - a plain merge on `date` alone, no
    segment-level join needed. Missing days (a gap in the cached weather
    file, or a date range this table covers that the download didn't) get
    0 for every weather column plus `has_weather=0`, the same "no data ≠
    a real zero reading" convention `attach_static_exposure_features` and
    `build_features.py`'s AADF handling already use - a 0.0mm rainfall
    reading and "we don't know" must never be conflated.
    """
    weather_cols = [c for c in weather.columns if c != "date"]
    merged = table.merge(weather, on="date", how="left")
    merged["has_weather"] = merged[weather_cols[0]].notna().astype("int8") if weather_cols else 0
    for col in weather_cols:
        merged[col] = merged[col].fillna(0.0)
    return merged


def attach_spillover_target(
    table: pd.DataFrame,
    edge_index: np.ndarray,
    segment_order: list[str],
    source_col: str = "collision_count",
    first_order_weight: float = 0.5,
    second_order_weight: float = 0.25,
) -> pd.DataFrame:
    """Reconstructs a "spillover" target inspired by Gao et al.'s
    predecessor paper (STZINB-GNN, arXiv:2307.13816), whose Data
    Description states their crash-risk target incorporates "spillover
    effects on first and second-order neighbouring roads" - a road with
    no crash of its own but adjacent to one gets a nonzero score too.
    **Neither that paper nor the journal STZITD-GNN paper states the
    exact decay function or aggregation rule** (found and disclosed,
    not silently assumed, in docs/decision_log.md's "think, think,
    think" entry) - this is a disclosed, explicit, defensible choice for
    experimentation, not a claimed replication of their precise formula,
    exactly like this module's own `tcr_score` reconstruction of
    Definition 1 elsewhere in this file.

    The explicit choice made here: for each (segment, day),
    `spillover_score = own + first_order_weight * sum(1st-order neighbour
    values) + second_order_weight * sum(2nd-order neighbour values)`,
    using the SAME line-graph adjacency (`edge_index`) already built for
    the GAT itself - the model's own notion of "neighbouring road" is
    reused rather than inventing a second one. Decay weights 0.5/0.25
    are a simple, monotonically-decreasing, disclosed default (not
    fitted or searched) - halving influence per hop is the most
    common convention in spatial-smoothing literature when no paper-
    specified value is available.

    Implementation note: "2nd-order neighbour" here is computed as
    `A @ (A @ own)` (`A` = the adjacency matrix), which is a standard
    but impure two-hop count - it does NOT exclude paths that return to
    a first-order neighbour or to the segment itself (e.g. i-j-i), so it
    mildly double-counts influence already captured by the first-order
    term for well-connected segments. A more careful de-duplicated
    two-hop count was judged not worth the added complexity given the
    formula itself is already an explicit, disclosed approximation, not
    a precise replication - flagged here rather than silently glossed
    over.

    `edge_index` must be built from the same `segment_order` (e.g. via
    `features.graph_temporal.edge_index_from_line_graph`) or neighbour
    lookups will silently misalign. `table` must have exactly one row
    per (segment_id, date) pair, as `build_segment_day_table` produces.
    """
    if source_col not in table.columns:
        raise ValueError(f"attach_spillover_target: '{source_col}' not found in table columns.")

    n_segments = len(segment_order)
    seg_to_idx = {seg: i for i, seg in enumerate(segment_order)}

    adjacency = sparse.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
        shape=(n_segments, n_segments),
    ).tocsr()

    pivot = table.pivot(index="date", columns="segment_id", values=source_col)
    # Reorder columns to segment_order, filling any segment absent from
    # this particular table (shouldn't happen given build_segment_day_table's
    # cross-join, but defended rather than silently misaligning columns).
    pivot = pivot.reindex(columns=segment_order, fill_value=0.0).fillna(0.0)
    own = pivot.to_numpy(dtype=np.float32)  # [n_dates, n_segments]

    first_order = own @ adjacency.T
    second_order = first_order @ adjacency.T
    spillover = own + first_order_weight * first_order + second_order_weight * second_order

    spillover_long = pd.DataFrame(spillover, index=pivot.index, columns=segment_order)
    spillover_long = spillover_long.stack().rename(f"{source_col}_spillover").reset_index()
    spillover_long.columns = ["date", "segment_id", f"{source_col}_spillover"]

    return table.merge(spillover_long, on=["segment_id", "date"], how="left")


def build_segment_day_table(
    edges: pd.DataFrame,
    counts: pd.DataFrame,
    graph: nx.MultiDiGraph,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Cross-join every segment x every calendar day in
    [start_date, end_date] (inclusive), filling zero counts.

    Scale note: unlike the annual table (~segments x 5 rows), this is
    segments x days - e.g. ~7,500 segments x ~1,095 days (3 years) is
    ~8.2M rows for Westminster alone. Kept to essential static columns
    (no per-day enrichment merge here, unlike the annual table's severity/
    IMD/exposure joins) specifically to keep this tractable; day-of-week
    is added directly since STATS19 crash risk is known to vary
    systematically by weekday (rush-hour exposure) and costs nothing to
    include.
    """
    degrees = node_degrees(graph)
    static_cols = [c for c in ("segment_id", "highway", "length", "maxspeed") if c in edges.columns]
    seg_static = edges[static_cols].drop_duplicates(subset="segment_id").copy()
    seg_static["u_degree"] = edges["u"].map(degrees).values if "u" in edges.columns else None
    seg_static["v_degree"] = edges["v"].map(degrees).values if "v" in edges.columns else None

    segments = seg_static["segment_id"].unique()
    all_dates = pd.date_range(start_date, end_date, freq="D")
    scaffold = pd.MultiIndex.from_product([segments, all_dates], names=["segment_id", "date"]).to_frame(index=False)

    table = scaffold.merge(seg_static, on="segment_id", how="left")
    table = table.merge(counts, on=["segment_id", "date"], how="left")
    # Every column `counts` contributes (collision_count, and - when built
    # via `collision_severity_counts_by_segment_day` instead of the plain
    # `collision_counts_by_segment_day` - the severity/vulnerable-user
    # columns too) is a real count that a missing (segment, day) combination
    # means zero for, not unknown - fill all of them the same way, not just
    # `collision_count`, so richer callers get the same zero-filling
    # guarantee without this function needing to know their column names.
    count_like_cols = [c for c in counts.columns if c not in ("segment_id", "date")]
    for col in count_like_cols:
        table[col] = table[col].fillna(0)
    if "collision_count" in table.columns:
        # int16 is a real memory win on a segments x days table (millions of
        # rows), but is only valid when counts are whole numbers. Junction
        # redistribution (`ingest.network.redistribute_junction_crashes`)
        # produces genuinely FRACTIONAL counts - a crash split four ways is
        # 0.25 on each arm - and casting those to int16 would silently
        # truncate every one of them to 0, deleting two-thirds of the signal
        # without any error. Cast only when it is actually lossless.
        counts_are_integral = np.isclose(table["collision_count"] % 1, 0).all()
        if counts_are_integral:
            table["collision_count"] = table["collision_count"].astype("int16")
        else:
            table["collision_count"] = table["collision_count"].astype("float32")
    table["day_of_week"] = table["date"].dt.dayofweek.astype("int8")
    table = table.sort_values(["segment_id", "date"]).reset_index(drop=True)
    return table
