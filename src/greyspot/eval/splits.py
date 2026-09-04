"""Spatial and temporal held-out splits — no random shuffling.

Random splits on segment-year rows leak information (the same road segment
appears in both train and test across years), which is exactly the mistake
flagged in the project dossier's evaluation discipline (Section 14). Two
splits are provided:

- temporal_split: train on early years, test on the latest year.
- spatial_split: within the training years, hold out a random subset of
  *segments* entirely, to test generalisation to unseen roads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(table: pd.DataFrame, test_year: int, min_train_year: int | None = None):
    train = table[table["year"] < test_year]
    if min_train_year is not None:
        train = train[train["year"] >= min_train_year]
    test = table[table["year"] == test_year]
    return train.copy(), test.copy()


def spatial_split(table: pd.DataFrame, holdout_frac: float = 0.2, seed: int = 42):
    """Hold out a random subset of segment_ids (not rows) for spatial generalisation testing."""
    rng = np.random.default_rng(seed)
    segments = table["segment_id"].unique()
    n_holdout = max(1, int(len(segments) * holdout_frac))
    holdout_segments = set(rng.choice(segments, size=n_holdout, replace=False))

    is_holdout = table["segment_id"].isin(holdout_segments)
    return table.loc[~is_holdout].copy(), table.loc[is_holdout].copy()
