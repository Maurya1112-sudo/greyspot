"""Daily [T, N, F] tensor construction and multi-step walk-forward windows.

`features/graph_temporal.py`'s tensor/walk-forward builders iterate the
segment-year table row by row (`.iterrows()`) - fine at ~7,500 rows/year,
but a segment-day table can be millions of rows (see
`daily_features.py`'s scale note), where a Python-level loop over every
row would be minutes-to-hours slow. Everything here is built with
`pivot`/vectorised NumPy operations instead - the same conceptual
tensors, built to actually finish running.

This also implements genuine **multi-step** (14-day-ahead) targets, which
the annual pipeline never needed - matching Gao et al. (2024)'s `p = 14`
forecast horizon (see docs/publication_readiness.md) via "direct"
multi-horizon output (the model predicts all 14 future days from one
input window in a single forward pass, rather than recursively feeding
its own day-1 prediction back in to predict day 2) - a standard,
legitimate multi-step forecasting strategy, not a shortcut approximation
of the paper's own (undisclosed) decoder mechanics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_feature_tensor(
    segment_day_table: pd.DataFrame,
    segment_order: list[str],
    feature_columns: list[str],
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """[T, N, F] array via `pivot` (genuinely vectorised - no Python-level
    loop over rows or cells) - the daily analogue of
    `graph_temporal.build_temporal_feature_tensor`. Missing (segment, date)
    combinations, and non-numeric values (e.g. `highway`'s road-type
    string), are filled with 0, matching that function's convention -
    categorical columns should be one-hot/numerically encoded by the
    caller before being passed in here, the same expectation the annual
    tensor builder already places on its callers.
    """
    n_dates, n_segments, n_features = len(dates), len(segment_order), len(feature_columns)
    tensor = np.zeros((n_dates, n_segments, n_features), dtype=np.float32)

    subset = segment_day_table[
        segment_day_table["segment_id"].isin(segment_order) & segment_day_table["date"].isin(dates)
    ]
    for f, col in enumerate(feature_columns):
        if col not in subset.columns:
            continue
        values = pd.to_numeric(subset[col], errors="coerce")
        pivot = subset.assign(**{col: values}).pivot_table(
            index="date", columns="segment_id", values=col, aggfunc="mean"
        )
        # Reindex to the exact requested dates/segments (in that order) so
        # a plain `.to_numpy()` assignment lines up with `tensor`'s axes -
        # no per-cell lookup loop.
        pivot = pivot.reindex(index=dates, columns=segment_order).fillna(0.0)
        tensor[:, :, f] = pivot.to_numpy(dtype=np.float32)
    return tensor


def build_daily_target_matrix(
    segment_day_table: pd.DataFrame,
    segment_order: list[str],
    dates: pd.DatetimeIndex,
    target_col: str = "collision_count",
) -> np.ndarray:
    """[T, N] matrix of the target count, one row per date - vectorised via
    `pivot_table`, not a per-row Python loop (this table can be millions
    of rows; see `daily_features.py`'s scale note)."""
    subset = segment_day_table[
        segment_day_table["segment_id"].isin(segment_order) & segment_day_table["date"].isin(dates)
    ]
    pivot = subset.pivot_table(index="date", columns="segment_id", values=target_col, aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=dates, columns=segment_order, fill_value=0)
    return pivot.to_numpy(dtype=np.float32)


def fit_feature_standardizer(train_x_seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) computed from TRAINING instances' `x_seq`
    tensors only - never the held-out instance - matching this project's
    walk-forward discipline of never letting test-period statistics leak
    into anything used at training time (`docs/decision_log.md`'s
    2026-08-31 correction entry is the same principle applied to targets
    instead of feature scale).

    Added 2026-09-01 after the data-richness pass's enriched feature set
    (`scripts/run_ucl_comparison.py`) made a real, previously-latent gap
    acute: nothing in this pipeline ever normalised features, which was
    survivable when every feature was a small integer (`day_of_week`,
    node degree) or a low-hundreds distance (`length`), but AADF traffic
    volume ranges into the tens of thousands (measured on Westminster:
    mean ~22,939, max ~125,289) - four to five orders of magnitude larger
    than collision-count/severity features, which would dominate a GAT's
    linear layers and a GRU's gates numerically regardless of how
    informative the smaller-scale features actually are. Returns
    `std=1.0` (not division by ~0) for any feature with near-zero
    variance in the training data (e.g. a borough with no matched AADF
    count points at all, or a constant `has_aadf` column), so a
    zero-variance column becomes a harmless all-zero input rather than
    an inf/nan.
    """
    stacked = np.concatenate([x.reshape(-1, x.shape[-1]) for x in train_x_seqs], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_standardizer(x_seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Applies a `(mean, std)` pair from `fit_feature_standardizer` to any
    `[T, N, F]` tensor - training, calibration, or held-out - with the
    *same* fitted statistics every time, never refit per-instance (refitting
    per instance would itself leak each instance's own held-out-adjacent
    distribution into its own inputs)."""
    return ((x_seq - mean) / std).astype(np.float32)


def build_daily_multistep_instances(
    segment_day_table: pd.DataFrame,
    segment_order: list[str],
    feature_columns: list[str],
    all_dates: pd.DatetimeIndex,
    input_window: int,
    horizon: int,
    stride: int,
    target_col: str = "collision_count",
) -> list[tuple[np.ndarray, np.ndarray, pd.Timestamp]]:
    """Sliding-window (input_window days -> horizon days ahead) instances,
    stepped every `stride` days - the daily, multi-step analogue of
    `graph_temporal.build_walkforward_instances`.

    Each instance is `(x_seq [input_window, N, F], y_multistep [horizon, N],
    first_target_date)` - genuinely disjoint target windows for different
    instances when `stride >= horizon` (no two instances share a target
    day), matching this project's walk-forward discipline
    (`docs/decision_log.md`'s 2026-08-31 correction entry) rather than
    Gao et al.'s own within-2019 6:2:2 split (their PhD thesis, page 232,
    quoted exactly 2026-09-02: "the ratio is 6:2:2" - corrected from an
    earlier, less careful reading of this project's own that had cited
    "8:2:2"), whose chronological disjointness is not stated in the
    paper. See `scripts/run_2019_replication.py` for a genuine attempt
    at replicating that exact protocol, rather than this project's own
    (harder, multi-year) walk-forward.

    The **last** instance in the returned list is the one this project
    treats as the genuinely held-out test window - callers should train
    only on the instances before it, mirroring
    `build_walkforward_instances`'s own convention exactly.
    """
    n_dates = len(all_dates)
    instances = []
    start = 0
    while start + input_window + horizon <= n_dates:
        input_dates = all_dates[start : start + input_window]
        target_dates = all_dates[start + input_window : start + input_window + horizon]
        x_seq = build_daily_feature_tensor(segment_day_table, segment_order, feature_columns, input_dates)
        y_multistep = build_daily_target_matrix(segment_day_table, segment_order, target_dates, target_col)
        instances.append((x_seq, y_multistep, target_dates[0]))
        start += stride
    return instances


def fit_feature_maxscaler(train_x_seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature max-value scaling: x / max(|x|) over the TRAINING
    instances only, returning `(zeros, maxes)` so it is drop-in
    compatible with `apply_feature_standardizer`'s `(mean, std)` contract.

    Added 2026-09-03 after reading the reference implementation's own
    code (github.com/ZhuangDingyi/STZINB - the public predecessor to
    Gao et al.'s STZITD-GNN, by the same group): it scales inputs by
    `E_maxvalue` (the training-set maximum) and leaves TARGETS raw,
    rather than z-scoring anything. This project has always z-scored.

    **Why this is likely to matter, measured not assumed**: on real
    Lambeth data this project's z-scored features reach an absolute
    maximum of ~172 (measured 2026-09-03 while diagnosing a NaN). That
    is a direct consequence of z-scoring 99.98%-zero count columns - a
    feature that is zero almost everywhere has a tiny standard
    deviation, so its rare non-zero values become enormous z-scores.
    Feeding values of that magnitude into GAT attention logits is
    exactly the setup that produced attention-parameter NaNs at
    lr=0.01. Max-scaling instead maps every feature into [-1, 1],
    which is what the reference code does and what attention softmax
    is far better behaved on.

    Near-zero-max columns get a divisor of 1.0 (not ~0), by the same
    reasoning `fit_feature_standardizer` uses for zero-variance columns:
    a constant column should become a harmless zero input, never an
    inf/nan.
    """
    stacked = np.concatenate([x.reshape(-1, x.shape[-1]) for x in train_x_seqs], axis=0)
    maxes = np.abs(stacked).max(axis=0)
    maxes = np.where(maxes < 1e-8, 1.0, maxes)
    zeros = np.zeros_like(maxes)
    return zeros, maxes


def apply_clipped_standardizer(x_seq: np.ndarray, mean, std, clip: float = 5.0) -> np.ndarray:
    """z-score, then CLIP to +/- `clip` standard deviations - the fix that
    actually addresses the measured defect, added 2026-09-03 after two
    other candidates were tested and rejected on evidence:

      - **plain z-scoring** reaches |1179| on real Lambeth data. A
        99.98%-zero count column has a near-zero standard deviation, so
        each rare non-zero becomes an enormous z-score. This is the
        condition behind the GAT attention NaNs and the 10x lower
        learning rate the paper's encoder order needed.
      - **max-value scaling** (what the reference implementation itself
        uses) bounds everything to [-1, 1] but CRUSHES the variance of
        those same sparse columns - almost every value becomes exactly
        0, so the feature stops carrying signal. Measured on Lambeth:
        34.70% on window 1 against a 75.76% baseline.
      - **log1p-then-z-score**, the textbook count-data transform, does
        NOT help here either, and the reason is worth recording: for a
        column that is one non-zero in ~500 cells, both the mean and the
        standard deviation scale with that value, so any monotone
        transform leaves the z-score essentially unchanged (verified:
        identical absmax before and after log1p). **The extremeness
        comes from the sparsity, not from the magnitude** - so no
        rescaling of the values can fix it.

    Clipping addresses it directly: typical values keep their
    z-scored spread (unit variance, good gradient flow), while the rare
    extreme values are capped at a magnitude attention softmax can
    handle. `clip=5.0` is a standard choice (5 sigma), picked once on
    that basis and not tuned against the evaluation windows.
    """
    z = (x_seq - mean) / std
    return np.clip(z, -clip, clip).astype("float32")
