"""Tests for the vectorised daily tensor/multi-step walk-forward builders
(`features/daily_temporal.py`)."""
import numpy as np
import pytest
import pandas as pd

from greyspot.features.daily_temporal import (
    apply_feature_standardizer,
    build_daily_feature_tensor,
    build_daily_multistep_instances,
    build_daily_target_matrix,
    fit_feature_standardizer,
)


def _tiny_table():
    dates = pd.date_range("2024-01-01", "2024-01-06", freq="D")
    rows = []
    for seg in ["a", "b"]:
        for i, d in enumerate(dates):
            rows.append({"segment_id": seg, "date": d, "collision_count": i if seg == "a" else i * 2, "length": 10.0})
    return pd.DataFrame(rows), dates


def test_build_daily_feature_tensor_shape_and_values():
    table, dates = _tiny_table()
    segment_order = ["a", "b"]
    tensor = build_daily_feature_tensor(table, segment_order, ["length", "collision_count"], dates[:3])
    assert tensor.shape == (3, 2, 2)
    # length is constant 10.0 everywhere.
    assert np.allclose(tensor[:, :, 0], 10.0)
    # collision_count for segment 'a' on day i is i; segment 'b' is i*2.
    assert tensor[2, 0, 1] == 2  # segment a, day index 2
    assert tensor[2, 1, 1] == 4  # segment b, day index 2


def test_build_daily_feature_tensor_fills_missing_with_zero():
    table, dates = _tiny_table()
    # request a date range that includes a day not in the table
    extra_dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
    tensor = build_daily_feature_tensor(table, ["a", "b"], ["collision_count"], extra_dates)
    assert tensor.shape == (10, 2, 1)
    # day index 6 onward (2024-01-07+) wasn't in the source table -> 0
    assert tensor[6:, :, 0].sum() == 0


def test_build_daily_target_matrix_shape_and_values():
    table, dates = _tiny_table()
    y = build_daily_target_matrix(table, ["a", "b"], dates, target_col="collision_count")
    assert y.shape == (6, 2)
    assert y[3, 0] == 3  # segment a, day index 3
    assert y[3, 1] == 6  # segment b, day index 3


def test_build_daily_multistep_instances_produces_disjoint_target_windows():
    table, dates = _tiny_table()
    instances = build_daily_multistep_instances(
        table, ["a", "b"], ["collision_count"], dates, input_window=2, horizon=2, stride=2,
    )
    # 6 days, input_window=2 + horizon=2 = 4 per instance, stride=2 -> 2 instances fit
    assert len(instances) == 2
    (x0, y0, d0), (x1, y1, d1) = instances
    assert x0.shape == (2, 2, 1)
    assert y0.shape == (2, 2)
    # genuinely disjoint target windows - the whole point of stride>=horizon.
    assert d1 > d0
    assert d1 - d0 == pd.Timedelta(days=2)


def test_build_daily_multistep_instances_last_instance_is_the_latest():
    table, dates = _tiny_table()
    instances = build_daily_multistep_instances(
        table, ["a", "b"], ["collision_count"], dates, input_window=2, horizon=2, stride=1,
    )
    target_dates = [d for _, _, d in instances]
    assert target_dates == sorted(target_dates)


def test_fit_feature_standardizer_gives_zero_mean_unit_std_on_training_data():
    rng = np.random.default_rng(0)
    # Two features on wildly different scales - feature 0 like a collision
    # count (0-3), feature 1 like AADF traffic volume (tens of thousands) -
    # the exact mismatch found 2026-09-01 (docs/decision_log.md).
    train_seqs = [
        np.stack(
            [rng.integers(0, 4, size=(5, 10)).astype(np.float32), rng.normal(20000, 5000, size=(5, 10)).astype(np.float32)],
            axis=-1,
        )
        for _ in range(3)
    ]
    mean, std = fit_feature_standardizer(train_seqs)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    # Feature 1's mean should be near 20000, not near feature 0's ~1.5 -
    # confirms per-feature (not global) statistics.
    assert 15000 < mean[1] < 25000
    assert 0 < mean[0] < 4

    stacked = np.concatenate([x.reshape(-1, 2) for x in train_seqs], axis=0)
    standardized = apply_feature_standardizer(stacked, mean, std)
    # After standardizing on its own fitting data, each feature column
    # should have ~zero mean and ~unit std - the whole point.
    np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(standardized.std(axis=0), [1.0, 1.0], atol=1e-4)


def test_apply_feature_standardizer_uses_the_same_fitted_stats_on_new_data():
    # The held-out/calibration data must be transformed with the TRAINING
    # data's mean/std, never refit on its own - refitting per-instance
    # would leak each instance's own distribution into its own inputs.
    train_seqs = [np.full((2, 3, 1), 10.0, dtype=np.float32)]  # mean=10, std=0 (constant)
    mean, std = fit_feature_standardizer(train_seqs)
    assert std[0] == 1.0  # zero-variance training feature -> std floored to 1.0, not division by ~0

    held_out = np.array([[[20.0], [10.0], [0.0]]], dtype=np.float32)
    scaled = apply_feature_standardizer(held_out, mean, std)
    # (20-10)/1=10, (10-10)/1=0, (0-10)/1=-10 - the training mean/std, not
    # anything derived from held_out's own values.
    np.testing.assert_allclose(scaled.flatten(), [10.0, 0.0, -10.0])


def test_fit_feature_standardizer_never_divides_by_a_near_zero_std():
    train_seqs = [np.zeros((2, 4, 2), dtype=np.float32)]  # both features constant (e.g. has_aadf all-0)
    mean, std = fit_feature_standardizer(train_seqs)
    assert np.all(std == 1.0)
    assert not np.any(np.isnan(apply_feature_standardizer(train_seqs[0], mean, std)))
    assert not np.any(np.isinf(apply_feature_standardizer(train_seqs[0], mean, std)))


def test_fit_feature_maxscaler_bounds_everything_within_unit_range():
    from greyspot.features.daily_temporal import fit_feature_maxscaler, apply_feature_standardizer
    # a sparse count-like column (the 99.98%-zero case that z-scoring
    # blows up into |z| ~ 172 on real data), plus a constant column
    x = np.zeros((5, 10, 3), dtype="float32")
    x[0, 0, 0] = 50.0      # rare large value in an otherwise-zero column
    x[:, :, 1] = 7.0       # constant column
    x[2, 3, 2] = -4.0      # negative value
    mean, scale = fit_feature_maxscaler([x])
    out = apply_feature_standardizer(x, mean, scale)
    assert np.abs(out).max() <= 1.0 + 1e-6


def test_fit_feature_maxscaler_never_divides_by_near_zero():
    from greyspot.features.daily_temporal import fit_feature_maxscaler
    all_zero = np.zeros((2, 4, 2), dtype="float32")
    _, scale = fit_feature_maxscaler([all_zero])
    assert (scale == 1.0).all()  # not 0 -> no inf/nan downstream


def test_maxscaler_keeps_sparse_columns_far_smaller_than_zscoring_would():
    # The actual motivation, as a test: on a 99%-zero column, z-scoring
    # sends the rare non-zero to a huge value while max-scaling sends it
    # to exactly 1.0.
    from greyspot.features.daily_temporal import (
        fit_feature_maxscaler, fit_feature_standardizer, apply_feature_standardizer,
    )
    x = np.zeros((10, 10, 1), dtype="float32")
    x[0, 0, 0] = 1.0  # one non-zero in 100 cells

    zm, zs = fit_feature_standardizer([x])
    z_out = apply_feature_standardizer(x, zm, zs)
    mm, ms = fit_feature_maxscaler([x])
    m_out = apply_feature_standardizer(x, mm, ms)

    assert np.abs(z_out).max() > 5.0        # z-score blows the rare value up
    assert np.abs(m_out).max() == pytest.approx(1.0)  # max-scaling does not
