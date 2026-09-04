"""XGBoost comparator: the strong non-graph baseline the dossier requires
before any graph-neural-network claim is credible (Section 9's "model ladder").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEATURE_COLUMNS = [
    "length",
    "u_degree",
    "v_degree",
    "prior_year_count",
    "prior_2yr_avg",
    # Lagged severity / vulnerable-user / vehicle / equity context (see
    # features.build_features.aggregate_enriched_features) - only ever the
    # prior-year versions, never same-year, to avoid leaking the target.
    "prior_year_n_casualties",
    "prior_year_n_fatal_casualties",
    "prior_year_n_serious_casualties",
    "prior_year_n_pedestrian_casualties",
    "prior_year_n_cyclist_casualties",
    "prior_year_n_vehicles_involved",
    "prior_year_avg_imd_decile",
    # Exposure (AADF traffic counts, same year - see
    # features.build_features.aggregate_exposure_features's docstring for
    # why this one is NOT lagged). Only ~1.6% of Westminster segment-years
    # have a nearby count point (see docs/decision_log.md), so these are
    # deliberately left as NaN in `prepare_features` (XGBoost natively
    # learns a default split direction for missing values) rather than
    # filled with 0, which would misrepresent "no data" as "no traffic".
    "aadf_all_motor_vehicles",
    "aadf_pedal_cycles",
    "has_aadf",
]

# Columns whose real missingness must reach XGBoost as NaN, not 0 - see the
# comment above. Every other feature is filled with 0 (a reasonable "no
# history" default for lagged counts).
_LEAVE_AS_NAN_COLUMNS = {"aadf_all_motor_vehicles", "aadf_pedal_cycles"}


def prepare_features(table: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    X = table.reindex(columns=feature_columns).copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if col not in _LEAVE_AS_NAN_COLUMNS:
            X[col] = X[col].fillna(0.0)
    return X


def train_xgboost(
    train_table: pd.DataFrame,
    target_col: str = "collision_count",
    feature_columns: list[str] = FEATURE_COLUMNS,
    **xgb_kwargs,
) -> XGBRegressor:
    X = prepare_features(train_table, feature_columns)
    y = train_table[target_col].to_numpy()
    model = XGBRegressor(
        n_estimators=xgb_kwargs.pop("n_estimators", 300),
        max_depth=xgb_kwargs.pop("max_depth", 4),
        learning_rate=xgb_kwargs.pop("learning_rate", 0.05),
        subsample=xgb_kwargs.pop("subsample", 0.8),
        colsample_bytree=xgb_kwargs.pop("colsample_bytree", 0.8),
        objective="count:poisson",  # appropriate for sparse, non-negative collision counts
        random_state=42,
        **xgb_kwargs,
    )
    model.fit(X, y)
    return model


def predict_xgboost(model: XGBRegressor, table: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> np.ndarray:
    X = prepare_features(table, feature_columns)
    return model.predict(X)
