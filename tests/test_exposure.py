import pandas as pd
import pytest

from greyspot.ingest.exposure import load_local_authority_aadf


def test_load_local_authority_aadf_missing_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="dft_traffic_counts_aadf"):
        load_local_authority_aadf(missing)


def test_load_local_authority_aadf_filters_and_selects_columns(tmp_path):
    csv_path = tmp_path / "aadf.csv"
    pd.DataFrame(
        {
            "count_point_id": [1, 2, 3],
            "year": [2023, 2023, 2023],
            "local_authority_name": ["Westminster", "Camden", "Westminster"],
            "road_name": ["A1", "A2", "A3"],
            "road_category": ["PA", "PA", "PA"],
            "latitude": [51.5, 51.5, 51.5],
            "longitude": [-0.1, -0.1, -0.1],
            "pedal_cycles": [10, 20, 30],
            "all_motor_vehicles": [1000, 2000, 3000],
            "extra_unused_column": ["x", "y", "z"],
        }
    ).to_csv(csv_path, index=False)

    out = load_local_authority_aadf(csv_path, local_authority_name="Westminster")

    assert len(out) == 2
    assert set(out["count_point_id"]) == {1, 3}
    assert "extra_unused_column" not in out.columns
