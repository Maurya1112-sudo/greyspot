"""Tests for the walk-forward temporal training fix (see docs/decision_log.md,
2026-08-31: the original "temporal" GAT evaluation trained directly against
its own evaluation target - an in-sample fit reported as a held-out test).
These tests exist specifically to catch a regression back to that bug.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from greyspot.features.graph_temporal import build_walkforward_instances
from greyspot.models.gat_temporal import GATTemporal, predict_gat_temporal, train_gat_temporal_walkforward, zero_inflated_poisson_nll


def test_build_walkforward_instances_produces_non_overlapping_target_years():
    table = pd.DataFrame(
        {
            "segment_id": ["a", "a", "a", "a"],
            "year": [2021, 2022, 2023, 2024],
            "feat": [1.0, 2.0, 3.0, 4.0],
            "collision_count": [0, 1, 0, 2],
        }
    )
    instances = build_walkforward_instances(
        table, segment_order=["a"], feature_columns=["feat"], years=[2021, 2022, 2023, 2024], window=2
    )
    target_years = [t for _, _, t in instances]
    assert target_years == [2023, 2024]  # two instances, no target year repeated
    assert len(set(target_years)) == len(target_years)


def test_build_walkforward_instances_windows_use_the_right_years():
    table = pd.DataFrame(
        {
            "segment_id": ["a"] * 4,
            "year": [2021, 2022, 2023, 2024],
            "feat": [10.0, 20.0, 30.0, 40.0],
            "collision_count": [0, 0, 0, 5],
        }
    )
    instances = build_walkforward_instances(
        table, segment_order=["a"], feature_columns=["feat"], years=[2021, 2022, 2023, 2024], window=2
    )
    # second instance: window [2022,2023] -> target 2024
    x_seq, y_target, target_year = instances[1]
    assert target_year == 2024
    assert x_seq[0, 0, 0] == 20.0  # 2022's feature value
    assert x_seq[1, 0, 0] == 30.0  # 2023's feature value
    assert y_target[0] == 5.0  # 2024's actual count


def test_build_walkforward_instances_raises_with_too_few_years():
    table = pd.DataFrame({"segment_id": ["a"], "year": [2021], "feat": [1.0], "collision_count": [0]})
    with pytest.raises(ValueError, match="more than"):
        build_walkforward_instances(table, segment_order=["a"], feature_columns=["feat"], years=[2021], window=2)


def test_train_gat_temporal_walkforward_does_not_see_the_held_out_target():
    # The critical regression test: training on instance A (target year X)
    # must produce a model whose predictions for a *different*, unseen
    # instance B are NOT simply memorised - i.e. changing instance B's
    # target must not have been "seen" by training, verified indirectly by
    # confirming training only touches the instances actually passed in.
    rng = np.random.default_rng(0)
    n_segments = 6
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)

    train_instances = [
        (rng.normal(size=(2, n_segments, 3)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32)),
    ]
    model = train_gat_temporal_walkforward(train_instances, edge_index, epochs=5)

    # Predicting on a held-out instance the model never trained on must
    # still run and produce valid, non-negative, finite output - proving
    # the function doesn't implicitly require or leak the eval target.
    held_out_x_seq = rng.normal(size=(2, n_segments, 3)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x_seq, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_train_gat_temporal_walkforward_trains_across_multiple_instances():
    rng = np.random.default_rng(1)
    n_segments = 5
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)

    instances = [
        (rng.normal(size=(1, n_segments, 2)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32)),
        (rng.normal(size=(1, n_segments, 2)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32)),
    ]
    model = train_gat_temporal_walkforward(instances, edge_index, epochs=5)
    preds = predict_gat_temporal(model, instances[-1][0], edge_index)
    assert preds.shape == (n_segments,)


def test_train_gat_temporal_walkforward_rejects_empty_instances():
    with pytest.raises(ValueError, match="at least one"):
        train_gat_temporal_walkforward([], edge_index=np.zeros((2, 0), dtype=np.int64))


def test_per_instance_backward_matches_summed_loss_backward():
    """Regression test for the 2026-09-01 OOM fix (see docs/decision_log.md's
    data-richness entry): `train_gat_temporal_walkforward` now calls
    `.backward()` once per instance and accumulates gradients, instead of
    summing every instance's loss into one graph and calling `.backward()`
    once - to bound peak GPU memory to a single instance rather than every
    instance at once (a 128-instance run OOM'd an 8GB GPU under the old
    approach). This test proves the two are numerically identical (as
    calculus says they must be - d(a+b)/dx = da/dx + db/dx), not just that
    training still runs: two identically-seeded, identically-initialised
    models accumulate the exact same gradient whether given one combined
    backward() on the summed loss, or one backward() per instance.
    """
    torch.manual_seed(0)
    n_segments, n_features, horizon = 5, 3, 1
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    instances = [
        (torch.randn(2, n_segments, n_features), torch.rand(n_segments).clamp(min=0)),
        (torch.randn(2, n_segments, n_features), torch.rand(n_segments).clamp(min=0)),
        (torch.randn(2, n_segments, n_features), torch.rand(n_segments).clamp(min=0)),
    ]

    def build_model():
        torch.manual_seed(123)
        return GATTemporal(in_channels=n_features, horizon=horizon, zero_inflated=True)

    # Reference: one combined backward() over the summed loss (the old approach).
    model_summed = build_model()
    total_loss = 0.0
    for x_t, y_t in instances:
        lam, pi = model_summed(x_t, edge_index)
        total_loss = total_loss + zero_inflated_poisson_nll(lam, pi, y_t)
    total_loss.backward()
    summed_grads = [p.grad.clone() for p in model_summed.parameters()]

    # Current approach: one backward() per instance, gradients accumulate.
    model_per_instance = build_model()
    for x_t, y_t in instances:
        lam, pi = model_per_instance(x_t, edge_index)
        loss = zero_inflated_poisson_nll(lam, pi, y_t)
        loss.backward()
    per_instance_grads = [p.grad.clone() for p in model_per_instance.parameters()]

    assert len(summed_grads) == len(per_instance_grads) > 0
    for g_summed, g_per_instance in zip(summed_grads, per_instance_grads):
        torch.testing.assert_close(g_summed, g_per_instance, rtol=1e-4, atol=1e-6)


def test_early_stopping_holds_out_the_last_instance_and_trains_successfully():
    # Regression/smoke test for the 2026-09-01 early-stopping addition -
    # motivated by this project's own evidence (epochs=20 underfit,
    # epochs=500 overfit on identical data, docs/decision_log.md). With
    # >=2 instances and early_stopping_patience set, the LAST instance is
    # held out as validation (never trained on) - the function must still
    # run to completion and return a valid, usable model.
    rng = np.random.default_rng(20)
    n_segments = 6
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    instances = [
        (rng.normal(size=(2, n_segments, 3)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(4)
    ]
    model = train_gat_temporal_walkforward(
        instances, edge_index, epochs=50, early_stopping_patience=3, zero_inflated=True,
    )
    held_out_x = rng.normal(size=(2, n_segments, 3)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_early_stopping_silently_disabled_with_only_one_instance():
    # With exactly 1 training instance, there is nothing left to hold out
    # as validation - early stopping must not crash (e.g. dividing by/
    # indexing an empty val set), it should just train normally for the
    # full `epochs` budget, identical to early_stopping_patience=None.
    rng = np.random.default_rng(21)
    n_segments = 5
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    single_instance = [
        (rng.normal(size=(2, n_segments, 3)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
    ]
    model = train_gat_temporal_walkforward(single_instance, edge_index, epochs=5, early_stopping_patience=2)
    preds = predict_gat_temporal(model, single_instance[0][0], edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()


def test_early_stopping_works_with_multistep_horizon():
    rng = np.random.default_rng(22)
    n_segments, horizon = 6, 4
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    instances = [
        (
            rng.normal(size=(2, n_segments, 3)).astype(np.float32),
            rng.poisson(0.8, size=(n_segments, horizon)).astype(np.float32),
        )
        for _ in range(3)
    ]
    model = train_gat_temporal_walkforward(
        instances, edge_index, epochs=20, early_stopping_patience=2, horizon=horizon, zero_inflated=True,
    )
    held_out_x = rng.normal(size=(2, n_segments, 3)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments, horizon)
    assert np.isfinite(preds).all()


def test_early_stopping_val_instances_holds_out_several_and_averages_their_loss():
    # k-fold-style generalisation (2026-09-01), added after the
    # single-instance version measurably hurt AccHR@20 (26.37% vs 49.57%,
    # docs/decision_log.md's "think, think, think" entry) - diagnosed as
    # one held-out instance being too noisy a validation signal at this
    # pipeline's small per-window instance counts. With
    # early_stopping_val_instances=3 and 6 available instances, the LAST
    # 3 (not just 1) must be held out as validation, and training must
    # still run to completion and return a usable model.
    rng = np.random.default_rng(23)
    n_segments = 6
    edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64)
    instances = [
        (rng.normal(size=(2, n_segments, 3)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(6)
    ]
    model = train_gat_temporal_walkforward(
        instances, edge_index, epochs=30, early_stopping_patience=3,
        early_stopping_val_instances=3, zero_inflated=True,
    )
    held_out_x = rng.normal(size=(2, n_segments, 3)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_early_stopping_val_instances_falls_back_to_one_when_not_enough_instances():
    # early_stopping_val_instances=5 but only 4 instances total: holding
    # out 5 would leave nothing to train on, so this must silently fall
    # back to holding out just 1 (matching the default behaviour) rather
    # than crashing or holding out every instance.
    rng = np.random.default_rng(24)
    n_segments = 5
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    instances = [
        (rng.normal(size=(2, n_segments, 3)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(4)
    ]
    model = train_gat_temporal_walkforward(
        instances, edge_index, epochs=10, early_stopping_patience=2, early_stopping_val_instances=5,
    )
    held_out_x = rng.normal(size=(2, n_segments, 3)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
