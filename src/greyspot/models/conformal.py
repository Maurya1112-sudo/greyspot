"""Conformal prediction uncertainty layer (MAPIE v1 SplitConformalRegressor).

Per the dossier (Section 10), a prediction interval is only meaningful if it
is calibrated on a genuinely held-out set - reusing training data for
calibration would silently overstate the empirical coverage this module
exists to test. `fit_split_conformal` therefore takes two disjoint tables:
one to fit the point model, another (from a *later, untouched* period) to
calibrate the interval width.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor

from greyspot.models.xgboost_model import FEATURE_COLUMNS, prepare_features, train_xgboost


def fit_split_conformal(
    fit_table: pd.DataFrame,
    calib_table: pd.DataFrame,
    confidence_level: float = 0.9,
    feature_columns: list[str] = FEATURE_COLUMNS,
):
    """Fit the XGBoost point model on `fit_table`, calibrate intervals on
    the disjoint `calib_table`. Returns (conformal_wrapper, point_model)."""
    point_model = train_xgboost(fit_table, feature_columns=feature_columns)
    conformal = SplitConformalRegressor(
        estimator=point_model, confidence_level=confidence_level, prefit=True
    )
    X_calib = prepare_features(calib_table, feature_columns)
    y_calib = calib_table["collision_count"].to_numpy()
    conformal.conformalize(X_calib, y_calib)
    return conformal, point_model


def predict_with_interval(
    conformal: SplitConformalRegressor, table: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = prepare_features(table, feature_columns)
    point, intervals = conformal.predict_interval(X)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    # Collision counts can't be negative. Clip only the *displayed* lower
    # bound; empirical_coverage below is checked against the raw (unclipped)
    # bound so clipping can't be used to inflate the coverage number.
    lower_clipped = np.clip(lower, 0, None)
    return point, lower_clipped, upper


def manual_split_conformal_interval(
    calib_true: np.ndarray, calib_pred: np.ndarray, test_pred: np.ndarray, confidence_level: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal prediction for any point-prediction array - the same
    underlying method as MAPIE's `SplitConformalRegressor` with its default
    'absolute' conformity score (nonconformity = |y - prediction|, interval
    = prediction +/- the target quantile of calibration residuals), but
    implemented directly rather than through MAPIE's sklearn-estimator
    wrapper. This is needed for the GAT+GRU model, which is a transductive,
    whole-graph model and does not fit the row-wise `fit(X, y)` /
    `predict(X)` interface MAPIE expects.

    `calib_true`/`calib_pred` must come from a genuinely held-out
    calibration period the point model was never trained to predict (see
    how the caller builds the calibration GAT in scripts/run_pipeline.py).
    """
    residuals = np.abs(calib_true - calib_pred)
    q = np.quantile(residuals, confidence_level, method="higher")
    lower = np.clip(test_pred - q, 0, None)
    upper = test_pred + q
    return lower, upper


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values that actually fall inside [lower, upper]."""
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))
