import networkx as nx
import pandas as pd

from greyspot.features.build_features import (
    aggregate_enriched_features,
    aggregate_exposure_features,
    build_segment_year_table,
)
from greyspot.ingest.imd import attach_imd_to_collisions
from greyspot.ingest.stats19 import (
    collision_severity_and_vulnerable_user_features,
    vehicle_mix_features,
)


def _toy_graph():
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, key=0)
    g.add_edge(2, 3, key=0)
    return g


def _toy_edges():
    return pd.DataFrame(
        {
            "segment_id": ["1_2_0", "2_3_0"],
            "u": [1, 2],
            "v": [2, 3],
            "highway": ["residential", "primary"],
            "length": [100.0, 250.0],
            "maxspeed": ["30", "40"],
            "oneway": [False, True],
        }
    )


def test_collision_severity_and_vulnerable_user_features_counts_correctly():
    casualties = pd.DataFrame(
        {
            "collision_index": ["c1", "c1", "c2"],
            "casualty_severity": [1, 3, 2],  # fatal, slight, serious
            "casualty_type": [0, 1, 9],  # pedestrian, cyclist, car occupant
        }
    )
    out = collision_severity_and_vulnerable_user_features(casualties).set_index("collision_index")

    assert out.loc["c1", "n_casualties"] == 2
    assert out.loc["c1", "n_fatal_casualties"] == 1
    assert out.loc["c1", "n_slight_casualties"] == 1
    assert out.loc["c1", "n_pedestrian_casualties"] == 1
    assert out.loc["c1", "n_cyclist_casualties"] == 1
    assert out.loc["c2", "n_serious_casualties"] == 1
    assert out.loc["c2", "n_pedestrian_casualties"] == 0


def test_vehicle_mix_features_counts_and_distinct_types():
    vehicles = pd.DataFrame(
        {
            "collision_index": ["c1", "c1", "c2"],
            "vehicle_type": [9, 9, 1],  # two cars in c1, one motorcycle in c2
        }
    )
    out = vehicle_mix_features(vehicles).set_index("collision_index")
    assert out.loc["c1", "n_vehicles"] == 2
    assert out.loc["c1", "n_distinct_vehicle_types"] == 1
    assert out.loc["c2", "n_vehicles"] == 1


def test_attach_imd_to_collisions_matches_and_flags_missing():
    collisions = pd.DataFrame({"lsoa_of_accident_location": ["E01000001", "E01999999"]})
    lookup = pd.DataFrame({"lsoa_code": ["E01000001"], "imd_decile": [3], "imd_score": [15.2]})
    out = attach_imd_to_collisions(collisions, lookup)
    assert out.loc[0, "imd_decile"] == 3
    assert pd.isna(out.loc[1, "imd_decile"])  # unmatched LSOA -> NaN, not silently dropped


def test_aggregate_enriched_features_and_lagging_avoids_same_year_leakage():
    snapped_enriched = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "1_2_0"],
            "collision_year": [2022, 2023],
            "n_casualties": [2, 0],
            "n_fatal_casualties": [1, 0],
            "n_serious_casualties": [0, 0],
            "n_slight_casualties": [1, 0],
            "n_pedestrian_casualties": [1, 0],
            "n_cyclist_casualties": [0, 0],
            "n_vehicles": [2, 0],
            "imd_decile": [4, None],
        }
    )
    enriched = aggregate_enriched_features(snapped_enriched)

    graph = _toy_graph()
    edges = _toy_edges()
    counts = pd.DataFrame({"segment_id": ["1_2_0"], "year": [2022], "collision_count": [2]})
    table = build_segment_year_table(edges, counts, graph, years=[2022, 2023], enriched=enriched)

    # the 2023 row's prior_year_n_fatal_casualties should reflect 2022's value (1)
    row_2023 = table[(table.segment_id == "1_2_0") & (table.year == 2023)]
    assert row_2023["prior_year_n_fatal_casualties"].iloc[0] == 1
    assert row_2023["prior_year_avg_imd_decile"].iloc[0] == 4

    # same-year enriched columns must NOT survive into the final table (leakage guard)
    assert "n_fatal_casualties" not in table.columns
    assert "avg_imd_decile" not in table.columns


def test_aggregate_exposure_features_averages_per_segment_year_and_drops_unmatched():
    snapped_aadf = pd.DataFrame(
        {
            "segment_id": ["1_2_0", "1_2_0", pd.NA],  # third row never matched a segment
            "year": [2023, 2023, 2023],
            "all_motor_vehicles": [1000, 2000, 5000],
            "pedal_cycles": [10, 20, 999],
        }
    )
    agg = aggregate_exposure_features(snapped_aadf)

    assert len(agg) == 1  # the unmatched row is excluded entirely, not averaged in
    row = agg.iloc[0]
    assert row["aadf_all_motor_vehicles"] == 1500  # mean of the two real count points
    assert row["aadf_pedal_cycles"] == 15


def test_build_segment_year_table_merges_exposure_unlagged_and_leaves_missing_as_nan():
    graph = _toy_graph()
    edges = _toy_edges()
    counts = pd.DataFrame({"segment_id": ["1_2_0"], "year": [2023], "collision_count": [1]})
    exposure = pd.DataFrame(
        {"segment_id": ["1_2_0"], "year": [2023], "aadf_all_motor_vehicles": [12000.0], "aadf_pedal_cycles": [80.0]}
    )

    table = build_segment_year_table(edges, counts, graph, years=[2022, 2023], exposure=exposure)

    row_2023 = table[(table.segment_id == "1_2_0") & (table.year == 2023)].iloc[0]
    # SAME year's AADF is used directly (not lagged) - it isn't derived from the target
    assert row_2023["aadf_all_motor_vehicles"] == 12000.0

    # a segment/year with no matching count point is NaN, not 0
    row_2023_other_segment = table[(table.segment_id == "2_3_0") & (table.year == 2023)].iloc[0]
    assert pd.isna(row_2023_other_segment["aadf_all_motor_vehicles"])
