"""Tests for daily-granularity feature building (`features/daily_features.py`).

Built for the UCL-comparable daily/14-day-multi-step evaluation
(`docs/publication_readiness.md`) - synthetic fixtures throughout, no
dependency on the real multi-gigabyte STATS19 files.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from greyspot.features import daily_features as daily_features_mod
from greyspot.features.daily_features import (
    attach_daily_weather_features,
    attach_poi_features,
    attach_rolling_collision_features,
    attach_socio_demographic_features,
    attach_spillover_target,
    attach_static_exposure_features,
    build_segment_day_table,
    collision_counts_by_segment_day,
    collision_severity_counts_by_segment_day,
    parse_stats19_date,
    propagate_aadf_by_road_name,
    attach_long_history_features,
)


def test_parse_stats19_date_is_day_first():
    parsed = parse_stats19_date(pd.Series(["31/01/2024", "01/12/2024"]))
    assert parsed.iloc[0] == pd.Timestamp("2024-01-31")
    assert parsed.iloc[1] == pd.Timestamp("2024-12-01")


def test_collision_counts_by_segment_day_aggregates_correctly():
    snapped = pd.DataFrame(
        {
            "segment_id": ["a", "a", "a", "b"],
            "date": ["01/01/2024", "01/01/2024", "02/01/2024", "01/01/2024"],
        }
    )
    counts = collision_counts_by_segment_day(snapped)
    counts = counts.set_index(["segment_id", "date"])["collision_count"]
    assert counts.loc[("a", pd.Timestamp("2024-01-01"))] == 2
    assert counts.loc[("a", pd.Timestamp("2024-01-02"))] == 1
    assert counts.loc[("b", pd.Timestamp("2024-01-01"))] == 1


def _tiny_graph_and_edges():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0)
    graph.add_edge(2, 3, key=0)
    edges = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0"],
            "u": [1, 2],
            "v": [2, 3],
            "highway": ["primary", "residential"],
            "length": [100.0, 50.0],
        }
    )
    return graph, edges


def test_build_segment_day_table_fills_zero_and_cross_joins_every_day():
    graph, edges = _tiny_graph_and_edges()
    counts = pd.DataFrame(
        {"segment_id": ["1_2_0"], "date": [pd.Timestamp("2024-01-02")], "collision_count": [3]}
    )
    table = build_segment_day_table(edges, counts, graph, start_date="2024-01-01", end_date="2024-01-03")

    # 2 segments x 3 days = 6 rows, every combination present.
    assert len(table) == 6
    assert set(table["segment_id"].unique()) == {"1_2_0", "2_3_0"}
    assert set(table["date"].unique()) == {
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
    }

    row = table[(table["segment_id"] == "1_2_0") & (table["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert row["collision_count"] == 3
    # every other (segment, day) combination not in `counts` is a real
    # zero, not a missing row - the whole point of the cross-join.
    zero_row = table[(table["segment_id"] == "2_3_0") & (table["date"] == pd.Timestamp("2024-01-01"))].iloc[0]
    assert zero_row["collision_count"] == 0


def test_build_segment_day_table_adds_day_of_week():
    graph, edges = _tiny_graph_and_edges()
    counts = pd.DataFrame({"segment_id": [], "date": [], "collision_count": []})
    table = build_segment_day_table(edges, counts, graph, start_date="2024-01-01", end_date="2024-01-01")
    # 2024-01-01 was a Monday - pandas dayofweek: Monday=0.
    assert (table["day_of_week"] == 0).all()


def test_build_segment_day_table_carries_static_edge_features():
    graph, edges = _tiny_graph_and_edges()
    counts = pd.DataFrame({"segment_id": [], "date": [], "collision_count": []})
    table = build_segment_day_table(edges, counts, graph, start_date="2024-01-01", end_date="2024-01-01")
    row = table[table["segment_id"] == "1_2_0"].iloc[0]
    assert row["highway"] == "primary"
    assert row["length"] == 100.0


def test_collision_severity_counts_by_segment_day_sums_severity_and_vulnerable_columns():
    snapped = pd.DataFrame(
        {
            "segment_id": ["a", "a", "b"],
            "date": ["01/01/2024", "01/01/2024", "01/01/2024"],
            "collision_index": ["c1", "c2", "c3"],
        }
    )
    casualties = pd.DataFrame(
        {
            "collision_index": ["c1", "c1", "c2", "c3"],
            "casualty_severity": [1, 3, 2, 3],  # 1=fatal, 2=serious, 3=slight
            "casualty_type": [0, 1, 3, 0],  # 0=pedestrian, 1=cyclist, 3=other
        }
    )
    counts = collision_severity_counts_by_segment_day(snapped, casualties)
    row_a = counts[counts["segment_id"] == "a"].iloc[0]
    # Segment "a": collision c1 (1 fatal pedestrian + 1 slight cyclist) + c2 (1 serious, no vulnerable user).
    assert row_a["collision_count"] == 2
    assert row_a["n_fatal_casualties"] == 1
    assert row_a["n_serious_casualties"] == 1
    assert row_a["n_slight_casualties"] == 1
    assert row_a["n_pedestrian_casualties"] == 1
    assert row_a["n_cyclist_casualties"] == 1

    row_b = counts[counts["segment_id"] == "b"].iloc[0]
    assert row_b["collision_count"] == 1
    assert row_b["n_fatal_casualties"] == 0
    assert row_b["n_slight_casualties"] == 1


def test_collision_severity_counts_by_segment_day_computes_tcr_score():
    # Gao et al.'s Eq. 7.1: y = sum_k C_k * l_k, l_k=1/2/3 for minor/
    # serious/fatal. STATS19's own collision_severity codes the opposite
    # direction (1=Fatal, 2=Serious, 3=Slight), so the applied weight is
    # (4 - collision_severity): fatal->3, serious->2, slight->1.
    snapped = pd.DataFrame(
        {
            "segment_id": ["a", "a", "a", "b"],
            "date": ["01/01/2024", "01/01/2024", "01/01/2024", "01/01/2024"],
            "collision_index": ["c1", "c2", "c3", "c4"],
            "collision_severity": [1, 2, 3, 3],  # fatal, serious, slight, slight
        }
    )
    casualties = pd.DataFrame({"collision_index": [], "casualty_severity": [], "casualty_type": []})
    counts = collision_severity_counts_by_segment_day(snapped, casualties)

    row_a = counts[counts["segment_id"] == "a"].iloc[0]
    # segment a: 1 fatal (weight 3) + 1 serious (weight 2) + 1 slight (weight 1) = 6
    assert row_a["tcr_score"] == 6

    row_b = counts[counts["segment_id"] == "b"].iloc[0]
    # segment b: 1 slight (weight 1)
    assert row_b["tcr_score"] == 1


def test_tcr_score_defaults_to_zero_when_collision_severity_column_is_absent():
    # Older/synthetic callers without a collision_severity column (e.g.
    # this file's other fixtures above) must not crash - tcr_score simply
    # isn't computed rather than raising a KeyError.
    snapped = pd.DataFrame(
        {"segment_id": ["a"], "date": ["01/01/2024"], "collision_index": ["c1"]}
    )
    casualties = pd.DataFrame({"collision_index": [], "casualty_severity": [], "casualty_type": []})
    counts = collision_severity_counts_by_segment_day(snapped, casualties)
    assert "tcr_score" not in counts.columns


def test_build_segment_day_table_zero_fills_every_count_like_column_from_counts():
    graph, edges = _tiny_graph_and_edges()
    counts = pd.DataFrame(
        {
            "segment_id": ["1_2_0"],
            "date": [pd.Timestamp("2024-01-02")],
            "collision_count": [2],
            "n_fatal_casualties": [1],
        }
    )
    table = build_segment_day_table(edges, counts, graph, start_date="2024-01-01", end_date="2024-01-02")
    zero_row = table[(table["segment_id"] == "2_3_0") & (table["date"] == pd.Timestamp("2024-01-01"))].iloc[0]
    # A severity column the caller passed in (not just collision_count)
    # must also be a real zero for a (segment, day) not present in `counts`.
    assert zero_row["n_fatal_casualties"] == 0
    present_row = table[(table["segment_id"] == "1_2_0") & (table["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert present_row["n_fatal_casualties"] == 1


def test_attach_rolling_collision_features_sums_trailing_window_causally():
    table = pd.DataFrame(
        {
            "segment_id": ["a", "a", "a", "a"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "collision_count": [1, 0, 2, 0],
        }
    )
    out = attach_rolling_collision_features(table, windows=(2,))
    out = out.set_index("date")["collision_count_2d"]
    # Day 1: only day 1 itself in a 2-day trailing window -> 1.
    assert out.loc[pd.Timestamp("2024-01-01")] == 1
    # Day 2: days 1+2 -> 1 + 0 = 1.
    assert out.loc[pd.Timestamp("2024-01-02")] == 1
    # Day 3: days 2+3 -> 0 + 2 = 2 (day 1's collision has rolled out of the window).
    assert out.loc[pd.Timestamp("2024-01-03")] == 2
    # Day 4: days 3+4 -> 2 + 0 = 2.
    assert out.loc[pd.Timestamp("2024-01-04")] == 2


def test_attach_rolling_collision_features_does_not_leak_across_segments():
    table = pd.DataFrame(
        {
            "segment_id": ["a", "b"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "collision_count": [5, 0],
        }
    )
    out = attach_rolling_collision_features(table, windows=(7,))
    row_b = out[out["segment_id"] == "b"].iloc[0]
    # Segment b's rolling window must never pick up segment a's collisions.
    assert row_b["collision_count_7d"] == 0


def test_attach_static_exposure_features_broadcasts_and_flags_missing():
    table = pd.DataFrame({"segment_id": ["a", "a", "b"], "date": [1, 2, 1]})
    exposure = pd.DataFrame(
        {
            "segment_id": ["a", "a"],
            "year": [2023, 2024],
            "aadf_all_motor_vehicles": [1000.0, 2000.0],
            "aadf_pedal_cycles": [10.0, 30.0],
        }
    )
    out = attach_static_exposure_features(table, exposure)
    row_a = out[out["segment_id"] == "a"].iloc[0]
    assert row_a["has_aadf"] == 1
    assert row_a["aadf_all_motor_vehicles"] == 1500.0  # averaged across the two available years

    row_b = out[out["segment_id"] == "b"].iloc[0]
    assert row_b["has_aadf"] == 0
    assert row_b["aadf_all_motor_vehicles"] == 0.0  # "no data" filled as 0, not left NaN


def test_attach_socio_demographic_features_broadcasts_and_flags_missing():
    table = pd.DataFrame({"segment_id": ["a", "a", "b"], "date": [1, 2, 1]})
    socio_demo = pd.DataFrame({
        "segment_id": ["a"],
        "imd_score": [25.0],
        "imd_income_score": [0.2],
        "imd_employment_score": [0.15],
        "imd_education_score": [0.3],
        "imd_health_score": [0.5],
        "imd_crime_score": [0.4],
        "imd_living_environment_score": [0.6],
        "population_density": [120.0],
    })
    out = attach_socio_demographic_features(table, socio_demo)
    row_a = out[out["segment_id"] == "a"].iloc[0]
    assert row_a["has_socio_demographic"] == 1
    assert row_a["imd_score"] == 25.0
    assert row_a["population_density"] == 120.0

    row_b = out[out["segment_id"] == "b"].iloc[0]
    assert row_b["has_socio_demographic"] == 0
    assert row_b["imd_score"] == 0.0  # "no data" filled as 0, not left NaN
    assert not pd.isna(row_b["imd_score"])


def test_attach_poi_features_broadcasts_and_flags_missing():
    table = pd.DataFrame({"segment_id": ["a", "a", "b"], "date": [1, 2, 1]})
    poi_counts = pd.DataFrame({
        "segment_id": ["a"],
        "poi_shop_count": [5.0],
        "poi_amenity_count": [3.0],
        "poi_leisure_count": [1.0],
        "poi_tourism_count": [0.0],
    })
    out = attach_poi_features(table, poi_counts)
    row_a = out[out["segment_id"] == "a"].iloc[0]
    assert row_a["has_poi"] == 1
    assert row_a["poi_shop_count"] == 5.0

    row_b = out[out["segment_id"] == "b"].iloc[0]
    assert row_b["has_poi"] == 0
    assert row_b["poi_shop_count"] == 0.0  # "no data" filled as 0, not left NaN
    assert not pd.isna(row_b["poi_shop_count"])


def test_attach_daily_weather_features_broadcasts_by_date_not_segment():
    table = pd.DataFrame(
        {
            "segment_id": ["a", "b", "a"],
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
        }
    )
    weather = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "temperature_2m_mean": [5.0, 8.0],
            "precipitation_sum": [1.2, 0.0],
        }
    )
    out = attach_daily_weather_features(table, weather)

    # Both segments on 2024-01-01 get the SAME reading - weather is
    # per-day, not per-segment.
    day1 = out[out["date"] == pd.Timestamp("2024-01-01")]
    assert (day1["temperature_2m_mean"] == 5.0).all()
    assert (day1["has_weather"] == 1).all()

    day2 = out[out["date"] == pd.Timestamp("2024-01-02")]
    assert (day2["temperature_2m_mean"] == 8.0).all()


def test_attach_daily_weather_features_flags_missing_days_as_zero_not_nan():
    table = pd.DataFrame({"segment_id": ["a"], "date": [pd.Timestamp("2024-06-01")]})
    weather = pd.DataFrame(
        {"date": [pd.Timestamp("2024-01-01")], "temperature_2m_mean": [5.0], "precipitation_sum": [1.2]}
    )
    out = attach_daily_weather_features(table, weather)
    row = out.iloc[0]
    assert row["has_weather"] == 0
    assert row["temperature_2m_mean"] == 0.0
    assert not pd.isna(row["temperature_2m_mean"])


def test_attach_spillover_target_hand_computed_on_a_three_segment_chain():
    # Chain A-B-C (both directions, matching edge_index_from_line_graph's
    # convention). One day: A=2 crashes, B=0, C=0.
    # 1st-order neighbour sums: A's only neighbour is B (own=0) -> 0;
    # B's neighbours are A,C (2+0) -> 2; C's only neighbour is B (0) -> 0.
    # 2nd-order (first_order @ A again, deliberately impure - see
    # docstring): A's neighbour B has first_order=2 -> 2; B's neighbours
    # A,C have first_order 0,0 -> 0; C's neighbour B has first_order=2 -> 2.
    # spillover = own + 0.5*first_order + 0.25*second_order:
    #   A = 2 + 0.5*0 + 0.25*2 = 2.5
    #   B = 0 + 0.5*2 + 0.25*0 = 1.0
    #   C = 0 + 0.5*0 + 0.25*2 = 0.5
    segment_order = ["A", "B", "C"]
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    table = pd.DataFrame({
        "segment_id": ["A", "B", "C"],
        "date": [pd.Timestamp("2024-01-01")] * 3,
        "collision_count": [2, 0, 0],
    })
    out = attach_spillover_target(table, edge_index, segment_order)
    by_seg = out.set_index("segment_id")["collision_count_spillover"]
    assert by_seg["A"] == pytest.approx(2.5)
    assert by_seg["B"] == pytest.approx(1.0)
    assert by_seg["C"] == pytest.approx(0.5)


def test_attach_spillover_target_zero_weights_reduces_to_the_own_count():
    segment_order = ["A", "B", "C"]
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    table = pd.DataFrame({
        "segment_id": ["A", "B", "C"],
        "date": [pd.Timestamp("2024-01-01")] * 3,
        "collision_count": [3, 1, 0],
    })
    out = attach_spillover_target(
        table, edge_index, segment_order, first_order_weight=0.0, second_order_weight=0.0,
    )
    assert list(out.sort_values("segment_id")["collision_count_spillover"]) == [3.0, 1.0, 0.0]


def test_attach_spillover_target_isolated_segment_gets_no_spillover():
    # D has no edges at all - it must keep exactly its own count, not
    # crash or silently pick up neighbours it doesn't have.
    segment_order = ["A", "B", "D"]
    edge_index = np.array([[0], [1]], dtype=np.int64)  # only A->B (one direction given)
    table = pd.DataFrame({
        "segment_id": ["A", "B", "D"],
        "date": [pd.Timestamp("2024-01-01")] * 3,
        "collision_count": [5, 5, 5],
    })
    out = attach_spillover_target(table, edge_index, segment_order)
    d_value = out.set_index("segment_id")["collision_count_spillover"]["D"]
    assert d_value == pytest.approx(5.0)


def test_attach_spillover_target_raises_a_clear_error_for_a_missing_source_column():
    table = pd.DataFrame({
        "segment_id": ["A", "B"], "date": [pd.Timestamp("2024-01-01")] * 2, "collision_count": [1, 0],
    })
    edge_index = np.array([[0], [1]], dtype=np.int64)
    with pytest.raises(ValueError, match="tcr_score"):
        attach_spillover_target(table, edge_index, ["A", "B"], source_col="tcr_score")


def test_collision_severity_counts_weighted_mode_sums_weights_not_rows():
    # Junction redistribution emits one crash as several fractional rows;
    # aggregating with `size` would MULTIPLY the crash by its number of
    # connected segments instead of dividing it among them.
    snapped = pd.DataFrame({
        "collision_index": ["c1", "c1", "c1", "c1"],
        "segment_id": ["a_b_0", "b_a_0", "a_c_0", "c_a_0"],
        "date": ["05/01/2023"] * 4,
        "collision_severity": [3, 3, 3, 3],
        "crash_weight": [0.25, 0.25, 0.25, 0.25],
    })
    casualties = pd.DataFrame({
        "collision_index": ["c1"], "casualty_severity": [3], "casualty_class": [1],
        "casualty_type": [0],
    })
    out = collision_severity_counts_by_segment_day(
        snapped, casualties, weight_col="crash_weight",
    )
    # one crash, split four ways: each segment gets 0.25, total stays 1.0
    assert out["collision_count"].sum() == pytest.approx(1.0)
    assert out["collision_count"].to_numpy() == pytest.approx(0.25)


def test_collision_severity_counts_weighted_mode_requires_the_column():
    snapped = pd.DataFrame({
        "collision_index": ["c1"], "segment_id": ["a_b_0"],
        "date": ["05/01/2023"], "collision_severity": [3],
    })
    casualties = pd.DataFrame({
        "collision_index": ["c1"], "casualty_severity": [3], "casualty_class": [1],
        "casualty_type": [0],
    })
    # Failing loudly beats silently falling back to unweighted counts,
    # which would look plausible but quietly undo the redistribution.
    with pytest.raises(KeyError, match="crash_weight"):
        collision_severity_counts_by_segment_day(
            snapped, casualties, weight_col="crash_weight",
        )


def test_build_segment_day_table_preserves_fractional_counts():
    # The int16 cast that makes the (segments x days) table affordable is
    # only lossless for whole numbers - on redistributed fractional counts
    # it would truncate every 0.25 to 0, deleting the signal silently.
    edges = pd.DataFrame({
        "segment_id": ["a_b_0"], "u": ["a"], "v": ["b"],
        "length": [100.0], "highway": ["residential"],
    })
    counts = pd.DataFrame({
        "segment_id": ["a_b_0"], "date": pd.to_datetime(["2023-01-02"]),
        "collision_count": [0.25],
    })
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", key=0)
    table = build_segment_day_table(
        edges, counts, graph, start_date="2023-01-01", end_date="2023-01-03",
    )
    assert table["collision_count"].sum() == pytest.approx(0.25)


def test_attach_road_class_features_one_hot_encodes_fixed_vocabulary():
    table = pd.DataFrame({
        "segment_id": ["a", "b", "c"],
        "highway": ["primary", "residential", "service"],
        "date": pd.to_datetime(["2023-01-01"] * 3),
    })
    out = daily_features_mod.attach_road_class_features(table)
    # exactly 8 columns, always, regardless of which classes appear
    assert daily_features_mod.ROAD_CLASS_COLUMNS == [c for c in out.columns if c.startswith("road_class_")]
    assert len(daily_features_mod.ROAD_CLASS_COLUMNS) == 8
    assert out.loc[0, "road_class_primary"] == 1.0
    assert out.loc[0, "road_class_residential"] == 0.0
    assert out.loc[1, "road_class_residential"] == 1.0
    # each row is one-hot: exactly one 1.0
    assert out[daily_features_mod.ROAD_CLASS_COLUMNS].sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_attach_road_class_features_keeps_width_fixed_for_unseen_classes():
    # A borough with no motorway must still produce 8 columns - otherwise
    # the feature matrix width differs per borough and cross-borough
    # comparison (and the fitted standardiser) silently breaks.
    table = pd.DataFrame({
        "segment_id": ["a"], "highway": ["some_unmapped_value"],
        "date": pd.to_datetime(["2023-01-01"]),
    })
    out = daily_features_mod.attach_road_class_features(table)
    assert len(daily_features_mod.ROAD_CLASS_COLUMNS) == 8
    # unknown class -> all-zero row, not a new column
    assert out[daily_features_mod.ROAD_CLASS_COLUMNS].sum(axis=1).iloc[0] == 0.0


def test_attach_road_class_features_missing_column_is_safe():
    table = pd.DataFrame({"segment_id": ["a"], "date": pd.to_datetime(["2023-01-01"])})
    out = daily_features_mod.attach_road_class_features(table)
    assert out[daily_features_mod.ROAD_CLASS_COLUMNS].sum(axis=1).iloc[0] == 0.0


def test_attach_date_features_encodes_weekend_and_cyclical_month():
    table = pd.DataFrame({
        "segment_id": ["a", "b", "c"],
        # 2023-01-07 is a Saturday, 2023-01-09 a Monday, 2023-12-15 a Friday
        "date": pd.to_datetime(["2023-01-07", "2023-01-09", "2023-12-15"]),
    })
    out = daily_features_mod.attach_date_features(table)
    assert out.loc[0, "is_weekend"] == 1.0
    assert out.loc[1, "is_weekend"] == 0.0
    assert out.loc[2, "is_weekend"] == 0.0
    # December and January must be ADJACENT in cyclical space - the whole
    # point of sin/cos over a raw 1-12 integer (which would put them 11 apart)
    import math
    jan = (out.loc[0, "month_sin"], out.loc[0, "month_cos"])
    dec = (out.loc[2, "month_sin"], out.loc[2, "month_cos"])
    dist = math.dist(jan, dec)
    assert dist < 0.6  # one month's arc on the unit circle, not eleven


def test_propagate_aadf_by_road_name_fills_only_same_named_missing_segments():
    # seg_a and seg_b share a name and both start with has_aadf=0; only
    # seg_a has a real measurement. seg_c has a DIFFERENT name and no
    # measured neighbour anywhere - it must stay at 0.
    table = pd.DataFrame({
        "segment_id": ["seg_a", "seg_b", "seg_c"],
        "has_aadf": [1, 0, 0],
        "aadf_all_motor_vehicles": [1000.0, 0.0, 0.0],
        "aadf_pedal_cycles": [50.0, 0.0, 0.0],
    })
    names = pd.DataFrame({
        "segment_id": ["seg_a", "seg_b", "seg_c"],
        "name": ["Harrow Road", "Harrow Road", "Some Cul-de-sac"],
    })
    out = propagate_aadf_by_road_name(table, names)

    # seg_b recovers seg_a's value (the only real "Harrow Road" measurement)
    assert out.loc[out.segment_id == "seg_b", "aadf_all_motor_vehicles"].iloc[0] == 1000.0
    assert out.loc[out.segment_id == "seg_b", "has_aadf"].iloc[0] == 1

    # seg_c has no same-named measured segment anywhere - stays at 0
    assert out.loc[out.segment_id == "seg_c", "aadf_all_motor_vehicles"].iloc[0] == 0.0
    assert out.loc[out.segment_id == "seg_c", "has_aadf"].iloc[0] == 0

    # the ALREADY-measured segment's real value is never touched
    assert out.loc[out.segment_id == "seg_a", "aadf_all_motor_vehicles"].iloc[0] == 1000.0


def test_propagate_aadf_by_road_name_never_uses_zero_filled_rows_as_a_source():
    # A pathological case: if TWO segments of the same name both start
    # unmeasured (has_aadf=0), propagation must NOT let one borrow from
    # the other's already-zero-filled value - the source of truth is only
    # rows where has_aadf==1 at the time this function runs.
    table = pd.DataFrame({
        "segment_id": ["seg_a", "seg_b"],
        "has_aadf": [0, 0],
        "aadf_all_motor_vehicles": [0.0, 0.0],
        "aadf_pedal_cycles": [0.0, 0.0],
    })
    names = pd.DataFrame({
        "segment_id": ["seg_a", "seg_b"],
        "name": ["Unmeasured Street", "Unmeasured Street"],
    })
    out = propagate_aadf_by_road_name(table, names)
    assert (out["aadf_all_motor_vehicles"] == 0.0).all()
    assert (out["has_aadf"] == 0).all()


def _history_fixture():
    """A 2-segment table over 2022, with collisions reaching back into 2021."""
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    table = pd.DataFrame({
        "segment_id": np.repeat(["a", "b"], len(dates)),
        "date": np.tile(dates, 2),
    })
    snapped = pd.DataFrame({
        "segment_id": ["a", "a", "a", "b"],
        # two crashes inside 2021 (before the table starts), one inside 2022
        "date": pd.to_datetime(["2021-03-01", "2021-09-01", "2022-02-01", "2021-06-15"]),
    })
    return table, snapped


def test_attach_long_history_counts_crashes_from_before_the_table_starts():
    table, snapped = _history_fixture()
    out = attach_long_history_features(table, snapped, lookbacks=(730,))
    # On 2022-06-01, segment "a"'s 730-day window reaches back to 2020-06 and
    # must see all three of its crashes (two in 2021, one in 2022).
    row = out[(out.segment_id == "a") & (out.date == pd.Timestamp("2022-06-01"))]
    assert row["collision_count_730d"].iloc[0] == 3.0
    # segment "b" has exactly one, in 2021 - still inside the window
    rb = out[(out.segment_id == "b") & (out.date == pd.Timestamp("2022-06-01"))]
    assert rb["collision_count_730d"].iloc[0] == 1.0


def test_attach_long_history_window_is_backward_looking_only():
    table, snapped = _history_fixture()
    out = attach_long_history_features(table, snapped, lookbacks=(730,))
    # On 2022-01-15, segment "a"'s 2022-02-01 crash is in the FUTURE and must
    # not be counted - only the two 2021 crashes are.
    row = out[(out.segment_id == "a") & (out.date == pd.Timestamp("2022-01-15"))]
    assert row["collision_count_730d"].iloc[0] == 2.0


def test_attach_long_history_matches_the_existing_rolling_implementation():
    """The load-bearing correctness check: for a lookback the OLD code can
    also compute, both must agree exactly. Guards against off-by-one errors
    in the cumsum window, which would be invisible in the model's output."""
    dates = pd.date_range("2022-01-01", "2022-06-30", freq="D")
    rng = np.random.default_rng(0)
    seg_ids = np.repeat(["a", "b", "c"], len(dates))
    table = pd.DataFrame({
        "segment_id": seg_ids,
        "date": np.tile(dates, 3),
        "collision_count": rng.poisson(0.05, size=len(seg_ids)).astype("float32"),
    })
    # Reconstruct the equivalent sparse collision list from the counts
    rows = []
    for _, r in table.iterrows():
        for _ in range(int(r.collision_count)):
            rows.append({"segment_id": r.segment_id, "date": r.date})
    snapped = pd.DataFrame(rows)

    old = attach_rolling_collision_features(table, windows=(30,))
    new = attach_long_history_features(table, snapped, lookbacks=(30,))
    merged = old[["segment_id", "date", "collision_count_30d"]].merge(
        new[["segment_id", "date", "collision_count_30d"]],
        on=["segment_id", "date"], suffixes=("_old", "_new"),
    )
    assert len(merged) == len(table)
    np.testing.assert_allclose(
        merged["collision_count_30d_old"].to_numpy(),
        merged["collision_count_30d_new"].to_numpy(),
        err_msg="cumsum-based long-history disagrees with the rolling implementation",
    )


def test_attach_long_history_warns_and_truncates_when_history_is_short():
    table, snapped = _history_fixture()
    # Longest lookback far exceeds the ~10 months of pre-table history present
    out = attach_long_history_features(table, snapped, lookbacks=(3650,))
    # Values are partial sums, not errors - and never exceed the true total
    assert out["collision_count_3650d"].max() <= 3.0
