import numpy as np
import pandas as pd

from greyspot.models.xgboost_model import prepare_features


def test_prepare_features_leaves_aadf_columns_as_nan_but_fills_others():
    table = pd.DataFrame(
        {
            "length": [100.0, np.nan],
            "prior_year_count": [np.nan, 2.0],
            "aadf_all_motor_vehicles": [np.nan, 5000.0],
            "aadf_pedal_cycles": [np.nan, 40.0],
            "has_aadf": [0, 1],
        }
    )
    X = prepare_features(table, feature_columns=["length", "prior_year_count", "aadf_all_motor_vehicles", "aadf_pedal_cycles", "has_aadf"])

    # ordinary lagged/count-like features get filled with 0
    assert X["length"].iloc[1] == 0.0
    assert X["prior_year_count"].iloc[0] == 0.0

    # AADF columns keep NaN for "no data" rows - never silently 0
    assert np.isnan(X["aadf_all_motor_vehicles"].iloc[0])
    assert np.isnan(X["aadf_pedal_cycles"].iloc[0])
    assert X["aadf_all_motor_vehicles"].iloc[1] == 5000.0

    # the indicator itself is never NaN
    assert list(X["has_aadf"]) == [0, 1]


def test_prepare_features_handles_missing_columns_gracefully():
    table = pd.DataFrame({"length": [10.0]})
    X = prepare_features(table, feature_columns=["length", "prior_year_count"])
    assert X["prior_year_count"].iloc[0] == 0.0  # a wholly-missing non-AADF column still defaults to 0
