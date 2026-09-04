"""Historical-rate baseline: the transparent comparator every other model must beat.

Per the dossier's model ladder (Section 9), the historical-rate baseline
simply reuses the prior period's observed count (or short rolling average)
as this period's risk score. If XGBoost/GNN models cannot beat this, the
added complexity is not earning its place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def historical_rate_score(table: pd.DataFrame, feature: str = "prior_year_count") -> np.ndarray:
    """Use the prior-year count as the predicted risk score. Missing history -> 0."""
    return table[feature].fillna(0.0).to_numpy()
