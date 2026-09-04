import numpy as np
import pytest
import torch

from greyspot.models.gat_temporal import (
    GATTemporal,
    _resolve_device,
    pairwise_rank_hinge_loss,
    predict_gat_temporal,
    train_gat_temporal,
    zero_inflated_negative_binomial_nll,
    zero_inflated_poisson_nll,
    zero_inflated_tweedie_nll,
)


def test_train_and_predict_gat_temporal_shapes_and_validity():
    rng = np.random.default_rng(0)
    n_timesteps, n_segments, n_features = 3, 6, 4

    x_seq = rng.normal(size=(n_timesteps, n_segments, n_features)).astype(np.float32)
    edge_index = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64)
    y_target = rng.poisson(1.0, size=n_segments).astype(np.float32)
    train_mask = np.array([True, True, True, True, False, False])  # last two "held out"

    model = train_gat_temporal(x_seq, edge_index, y_target, train_mask=train_mask, epochs=5)
    preds = predict_gat_temporal(model, x_seq, edge_index)

    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()  # softplus output - collision counts can't be negative


def test_train_gat_temporal_with_no_mask_uses_all_segments():
    rng = np.random.default_rng(1)
    x_seq = rng.normal(size=(2, 4, 3)).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    y_target = rng.poisson(1.0, size=4).astype(np.float32)

    model = train_gat_temporal(x_seq, edge_index, y_target, epochs=3)
    preds = predict_gat_temporal(model, x_seq, edge_index)
    assert preds.shape == (4,)


def test_ablation_flags_produce_valid_output_with_no_edges():
    # use_graph=False must work even with an empty edge_index (it never
    # touches it), proving the "no graph" ablation truly ignores structure.
    rng = np.random.default_rng(2)
    x_seq = rng.normal(size=(3, 5, 4)).astype(np.float32)
    empty_edge_index = np.zeros((2, 0), dtype=np.int64)
    y_target = rng.poisson(1.0, size=5).astype(np.float32)

    model = train_gat_temporal(x_seq, empty_edge_index, y_target, epochs=3, use_graph=False, use_temporal=True)
    preds = predict_gat_temporal(model, x_seq, empty_edge_index)
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()


def test_ablation_no_temporal_ignores_earlier_timesteps():
    # With use_temporal=False, only the *last* timestep should affect the
    # prediction - perturbing an earlier timestep must leave predictions
    # bit-for-bit identical.
    rng = np.random.default_rng(3)
    x_seq = rng.normal(size=(3, 6, 4)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    y_target = rng.poisson(1.0, size=6).astype(np.float32)

    model = train_gat_temporal(x_seq, edge_index, y_target, epochs=1, use_graph=True, use_temporal=False)
    preds_original = predict_gat_temporal(model, x_seq, edge_index)

    x_seq_perturbed = x_seq.copy()
    x_seq_perturbed[0] += 100.0  # wildly change the *first* (non-final) timestep only
    preds_perturbed = predict_gat_temporal(model, x_seq_perturbed, edge_index)

    assert np.allclose(preds_original, preds_perturbed)


def test_use_residual_requires_use_graph_to_have_any_effect():
    # use_residual=True with use_graph=False should behave identically to
    # use_residual=False - the residual projection is meaningless without a
    # graph branch to supplement, so GATTemporal disables it silently
    # rather than building an unused parameter.
    from greyspot.models.gat_temporal import GATTemporal

    model_with = GATTemporal(in_channels=4, use_graph=False, use_residual=True)
    model_without = GATTemporal(in_channels=4, use_graph=False, use_residual=False)
    assert not hasattr(model_with, "residual_proj")
    assert not hasattr(model_without, "residual_proj")
    assert model_with.use_residual is False


def test_use_residual_with_graph_adds_a_projection_and_runs():
    rng = np.random.default_rng(4)
    x_seq = rng.normal(size=(2, 5, 3)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    y_target = rng.poisson(1.0, size=5).astype(np.float32)

    model = train_gat_temporal(x_seq, edge_index, y_target, epochs=3, use_graph=True, use_residual=True)
    assert hasattr(model, "residual_proj")
    preds = predict_gat_temporal(model, x_seq, edge_index)
    assert preds.shape == (5,)
    assert np.isfinite(preds).all()


def test_custom_heads_and_hidden_size_change_output_width_but_not_final_shape():
    rng = np.random.default_rng(5)
    x_seq = rng.normal(size=(2, 4, 3)).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    y_target = rng.poisson(1.0, size=4).astype(np.float32)

    model = train_gat_temporal(x_seq, edge_index, y_target, epochs=2, heads=1, gat_hidden=8)
    preds = predict_gat_temporal(model, x_seq, edge_index)
    assert preds.shape == (4,)
    # model.spatial became model.spatial_layers (a ModuleList) 2026-09-01,
    # when a real 2-layer GAT option was added (`gat_layers`, matching the
    # paper's own "GNNs are all two-layered") - single layer by default,
    # so index [0] is still the (only) GATConv.
    assert model.spatial_layers[0].heads == 1


def test_resolve_device_defaults_to_cuda_when_available_else_cpu():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert _resolve_device(None).type == expected


def test_resolve_device_respects_explicit_override():
    assert _resolve_device("cpu").type == "cpu"


def test_zero_inflated_poisson_nll_matches_hand_computed_values():
    # A single observation with lam=2.0, pi=0.3, y=0:
    # -log(0.3 + 0.7*exp(-2.0))
    lam = torch.tensor([2.0])
    pi = torch.tensor([0.3])
    y_zero = torch.tensor([0.0])
    expected_zero = -np.log(0.3 + 0.7 * np.exp(-2.0))
    assert zero_inflated_poisson_nll(lam, pi, y_zero).item() == pytest.approx(expected_zero, abs=1e-4)

    # y=3: -log(0.7) + lam - y*log(lam) + lgamma(y+1)  [lgamma(4)=log(3!)=log(6)]
    y_pos = torch.tensor([3.0])
    expected_pos = -np.log(0.7) + 2.0 - 3.0 * np.log(2.0) + np.log(6.0)
    assert zero_inflated_poisson_nll(lam, pi, y_pos).item() == pytest.approx(expected_pos, abs=1e-4)


def test_zero_inflated_poisson_nll_prefers_correct_pi_over_wrong_pi():
    # For all-zero data, a model with high pi (mostly structural zeros)
    # should get a lower (better) NLL than one with low pi.
    lam = torch.full((20,), 1.5)
    y = torch.zeros(20)
    loss_high_pi = zero_inflated_poisson_nll(lam, torch.full((20,), 0.9), y)
    loss_low_pi = zero_inflated_poisson_nll(lam, torch.full((20,), 0.1), y)
    assert loss_high_pi.item() < loss_low_pi.item()


def test_train_and_predict_zero_inflated_gat_temporal():
    rng = np.random.default_rng(6)
    x_seq = rng.normal(size=(2, 8, 3)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
    y_target = rng.poisson(0.5, size=8).astype(np.float32)

    model = train_gat_temporal(x_seq, edge_index, y_target, epochs=5, zero_inflated=True)
    assert model.zero_inflated is True
    assert hasattr(model, "zero_gate")

    preds = predict_gat_temporal(model, x_seq, edge_index)
    assert preds.shape == (8,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()  # (1-pi)*lam is still non-negative


def test_zero_inflated_forward_returns_tuple_non_zero_inflated_returns_tensor():
    from greyspot.models.gat_temporal import GATTemporal

    rng = np.random.default_rng(7)
    x_seq = torch.tensor(rng.normal(size=(2, 4, 3)).astype(np.float32))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    plain_model = GATTemporal(in_channels=3, zero_inflated=False)
    out = plain_model(x_seq, edge_index)
    assert isinstance(out, torch.Tensor)

    zip_model = GATTemporal(in_channels=3, zero_inflated=True)
    lam, pi = zip_model(x_seq, edge_index)
    assert lam.shape == (4,) and pi.shape == (4,)
    assert (pi >= 0).all() and (pi <= 1).all()


# --- horizon > 1 (multi-step, added 2026-09-01 for the UCL-comparable
# 14-day-forecast evaluation - docs/publication_readiness.md) ---------


def test_horizon_greater_than_one_keeps_trailing_dimension():
    from greyspot.models.gat_temporal import GATTemporal

    rng = np.random.default_rng(11)
    x_seq = torch.tensor(rng.normal(size=(3, 5, 4)).astype(np.float32))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    model = GATTemporal(in_channels=4, horizon=14, zero_inflated=False)
    out = model(x_seq, edge_index)
    assert out.shape == (5, 14)  # [N, horizon] - not squeezed, unlike horizon=1


def test_horizon_greater_than_one_zero_inflated_returns_matching_shapes():
    from greyspot.models.gat_temporal import GATTemporal

    rng = np.random.default_rng(12)
    x_seq = torch.tensor(rng.normal(size=(3, 5, 4)).astype(np.float32))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    model = GATTemporal(in_channels=4, horizon=14, zero_inflated=True)
    lam, pi = model(x_seq, edge_index)
    assert lam.shape == (5, 14)
    assert pi.shape == (5, 14)


def test_horizon_one_is_the_unchanged_default():
    from greyspot.models.gat_temporal import GATTemporal

    rng = np.random.default_rng(13)
    x_seq = torch.tensor(rng.normal(size=(3, 5, 4)).astype(np.float32))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    model = GATTemporal(in_channels=4)  # horizon defaults to 1
    out = model(x_seq, edge_index)
    assert out.shape == (5,)  # squeezed, exactly the pre-existing contract


def test_multistep_walkforward_training_and_prediction_end_to_end():
    """The generic walk-forward trainer/predictor needs zero code changes
    for horizon > 1 - it operates element-wise on whatever shape `lam`
    (and the matching `y_target`) come in as. This is the smoke test that
    confirms that claim, not just asserts it in a comment."""
    from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward

    rng = np.random.default_rng(14)
    n_segments, n_features, horizon = 6, 3, 14
    edge_index = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64)

    train_instances = [
        (
            rng.normal(size=(3, n_segments, n_features)).astype(np.float32),
            rng.poisson(0.5, size=(n_segments, horizon)).astype(np.float32),  # [N, horizon]
        )
        for _ in range(2)
    ]
    model = train_gat_temporal_walkforward(train_instances, edge_index, epochs=3, horizon=horizon)

    held_out_x = rng.normal(size=(3, n_segments, n_features)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments, horizon)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_zero_inflated_negative_binomial_nll_matches_hand_computed_values():
    # mu=2.0, pi=0.3, r=5.0, y=0: -log(pi + (1-pi)*(r/(r+mu))^r)
    mu = torch.tensor([2.0])
    pi = torch.tensor([0.3])
    r = torch.tensor([5.0])
    y_zero = torch.tensor([0.0])
    expected_zero = -np.log(0.3 + 0.7 * (5.0 / 7.0) ** 5.0)
    assert zero_inflated_negative_binomial_nll(mu, pi, r, y_zero).item() == pytest.approx(expected_zero, abs=1e-4)

    # y=3: -[log(1-pi) + lgamma(y+r) - lgamma(r) - lgamma(y+1)
    #        + r*log(r/(r+mu)) + y*log(mu/(r+mu))]
    y_pos = torch.tensor([3.0])
    from math import lgamma, log

    expected_pos = -(
        log(0.7)
        + lgamma(3.0 + 5.0) - lgamma(5.0) - lgamma(4.0)
        + 5.0 * log(5.0 / 7.0) + 3.0 * log(2.0 / 7.0)
    )
    assert zero_inflated_negative_binomial_nll(mu, pi, r, y_pos).item() == pytest.approx(expected_pos, abs=1e-4)


def test_zero_inflated_negative_binomial_nll_prefers_correct_pi_over_wrong_pi():
    mu = torch.full((20,), 1.5)
    r = torch.full((20,), 3.0)
    y = torch.zeros(20)
    loss_high_pi = zero_inflated_negative_binomial_nll(mu, torch.full((20,), 0.9), r, y)
    loss_low_pi = zero_inflated_negative_binomial_nll(mu, torch.full((20,), 0.1), r, y)
    assert loss_high_pi.item() < loss_low_pi.item()


def test_zero_inflated_negative_binomial_nll_larger_dispersion_r_approaches_poisson():
    # NB2 -> Poisson as r -> infinity (variance = mu + mu^2/r -> mu). A
    # very large r should give a near-identical NLL to the Poisson-based
    # zero_inflated_poisson_nll at the same (mu, pi, y) - confirms the NB2
    # parameterisation reduces to the right limit, not just "some other
    # distribution that happens to also have a pi gate."
    #
    # float64 throughout: at r=1e6, `lgamma(y+r)` and `lgamma(r)` are each
    # individually ~1.28e7, and their ~1-unit *difference* is exactly the
    # signal being tested - float32 (this project's usual training dtype)
    # only carries ~7 significant digits, which is nowhere near enough to
    # resolve a difference that small against a base that large. That's a
    # precision artifact of the test's own extreme r value, not a defect
    # in the loss function (verified: the two losses agree to 9 decimal
    # places in float64) - so this specific check needs float64, not a
    # looser tolerance in float32.
    mu = torch.tensor([2.0], dtype=torch.float64)
    pi = torch.tensor([0.3], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=torch.float64)
    r_huge = torch.tensor([1e6], dtype=torch.float64)
    nb_loss = zero_inflated_negative_binomial_nll(mu, pi, r_huge, y).item()
    poisson_loss = zero_inflated_poisson_nll(mu, pi, y).item()
    assert nb_loss == pytest.approx(poisson_loss, abs=1e-6)


def test_negative_binomial_gat_temporal_forward_returns_mu_pi_r_triple():
    rng = np.random.default_rng(7)
    x_seq = rng.normal(size=(2, 5, 3)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)

    model = GATTemporal(in_channels=3, negative_binomial=True)
    assert model.zero_inflated is True  # negative_binomial implies zero-inflation
    assert hasattr(model, "dispersion_gate")

    x_t = torch.tensor(x_seq)
    edge_t = torch.tensor(edge_index)
    mu, pi, r = model(x_t, edge_t)
    for tensor, name in [(mu, "mu"), (pi, "pi"), (r, "r")]:
        assert tensor.shape == (5,), name
        assert torch.isfinite(tensor).all(), name
    assert (mu >= 0).all()
    assert ((pi >= 0) & (pi <= 1)).all()
    assert (r > 0).all()


def test_negative_binomial_gat_temporal_trains_and_predicts_end_to_end():
    from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward

    rng = np.random.default_rng(8)
    n_segments, n_features = 6, 3
    edge_index = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64)
    train_instances = [
        (rng.normal(size=(2, n_segments, n_features)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(2)
    ]
    model = train_gat_temporal_walkforward(train_instances, edge_index, epochs=3, negative_binomial=True)
    assert model.negative_binomial is True

    held_out_x = rng.normal(size=(2, n_segments, n_features)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_zero_inflated_tweedie_nll_matches_hand_computed_values():
    # Gao et al. (2024)'s own ZITD decoder (Eq. 5-6, transcribed directly
    # from the paper's PDF text, docs/decision_log.md) - mu=1.0, pi=0.3,
    # phi=1.0, rho=1.5.
    #
    # y=0 (exact, Eq. 6): -log(pi + (1-pi)*exp(-mu^(2-rho)/(phi*(2-rho))))
    mu = torch.tensor([1.0])
    pi = torch.tensor([0.3])
    phi = torch.tensor([1.0])
    rho = 1.5
    y_zero = torch.tensor([0.0])
    p1, p2 = 1 - rho, 2 - rho
    log_p_zero_td = -(1.0**p2) / (1.0 * p2)
    expected_zero = -np.log(0.3 + 0.7 * np.exp(log_p_zero_td))
    assert zero_inflated_tweedie_nll(mu, pi, phi, y_zero, rho=rho).item() == pytest.approx(expected_zero, abs=1e-4)

    # y=2 (reference-code approximation ported from the paper's own
    # linked repo, see the function's own docstring): alpha=(2-rho)/(1-rho),
    # j_max=y^(2-rho)/((2-rho)*phi), log_a_approx=-log(y)+j_max*(alpha-1)
    # -log(j_max)-0.5*log(-alpha), log_f_td_pos=y*mu^p1/(phi*p1)-mu^p2/(phi*p2)
    # -log_a_approx, positive_term=-log(1-pi)-log_f_td_pos.
    y_pos = torch.tensor([2.0])
    alpha = p2 / p1
    j_max = 2.0**p2 / (p2 * 1.0)
    log_a_approx = -np.log(2.0) + j_max * (alpha - 1) - np.log(j_max) - 0.5 * np.log(-alpha)
    log_f_td_pos = 2.0 * 1.0**p1 / (1.0 * p1) - 1.0**p2 / (1.0 * p2) - log_a_approx
    expected_pos = -np.log(0.7) - log_f_td_pos
    assert zero_inflated_tweedie_nll(mu, pi, phi, y_pos, rho=rho).item() == pytest.approx(expected_pos, abs=1e-4)


def test_zero_inflated_tweedie_nll_prefers_correct_pi_over_wrong_pi():
    mu = torch.full((20,), 1.5)
    phi = torch.full((20,), 1.0)
    y = torch.zeros(20)
    loss_high_pi = zero_inflated_tweedie_nll(mu, torch.full((20,), 0.9), phi, y)
    loss_low_pi = zero_inflated_tweedie_nll(mu, torch.full((20,), 0.1), phi, y)
    assert loss_high_pi.item() < loss_low_pi.item()


def test_zero_inflated_tweedie_nll_rejects_rho_outside_the_valid_compound_poisson_gamma_range():
    mu = torch.tensor([1.0])
    pi = torch.tensor([0.3])
    phi = torch.tensor([1.0])
    y = torch.tensor([0.0])
    with pytest.raises(ValueError, match="rho"):
        zero_inflated_tweedie_nll(mu, pi, phi, y, rho=1.0)
    with pytest.raises(ValueError, match="rho"):
        zero_inflated_tweedie_nll(mu, pi, phi, y, rho=2.0)


def test_tweedie_gat_temporal_forward_returns_mu_pi_phi_triple():
    rng = np.random.default_rng(10)
    x_seq = rng.normal(size=(2, 5, 3)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)

    model = GATTemporal(in_channels=3, tweedie=True)
    assert model.zero_inflated is True  # tweedie implies zero-inflation
    assert model.negative_binomial is False
    assert hasattr(model, "phi_gate")

    x_t = torch.tensor(x_seq)
    edge_t = torch.tensor(edge_index)
    mu, pi, phi = model(x_t, edge_t)
    for tensor, name in [(mu, "mu"), (pi, "pi"), (phi, "phi")]:
        assert tensor.shape == (5,), name
        assert torch.isfinite(tensor).all(), name
    assert (mu >= 0).all()
    assert ((pi >= 0) & (pi <= 1)).all()
    assert (phi > 0).all()


def test_tweedie_gat_temporal_trains_and_predicts_end_to_end():
    from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward

    rng = np.random.default_rng(11)
    n_segments, n_features = 6, 3
    edge_index = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64)
    train_instances = [
        (rng.normal(size=(2, n_segments, n_features)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(2)
    ]
    model = train_gat_temporal_walkforward(train_instances, edge_index, epochs=3, tweedie=True)
    assert model.tweedie is True

    held_out_x = rng.normal(size=(2, n_segments, n_features)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_tweedie_and_negative_binomial_together_raises_value_error():
    # The two are alternative third-decoder-head choices for the same
    # slot - setting both is a caller mistake, not a valid combination.
    with pytest.raises(ValueError, match="negative_binomial and tweedie"):
        GATTemporal(in_channels=3, negative_binomial=True, tweedie=True)


def test_rate_link_rejects_unknown_value():
    with pytest.raises(ValueError, match="rate_link"):
        GATTemporal(in_channels=3, rate_link="sigmoid")


def test_rate_link_exp_matches_hand_computed_value():
    # A degenerate single-layer check: with use_graph=False and
    # use_temporal=False, the decoder is just a plain nn.Linear applied
    # to the last timestep's features - zero the weights and set a known
    # bias so the raw pre-activation is an exact, predictable constant,
    # then confirm rate == exp(bias), not softplus(bias).
    torch.manual_seed(0)
    model = GATTemporal(in_channels=2, gat_hidden=4, heads=1, use_graph=False, use_temporal=False, rate_link="exp")
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.bias.fill_(1.234)
        # Zero every other layer's weights so the decoder's INPUT is a
        # known constant too (irrelevant here since weight=0 makes the
        # input multiplier vanish regardless, but zeroing keeps the
        # whole forward pass deterministic and easy to reason about).
        for layer in model.spatial_layers:
            layer.weight.zero_()
            layer.bias.zero_()

    x_seq = torch.zeros(1, 3, 2)  # [T=1, N=3, F=2]
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    rate = model(x_seq, edge_index)
    assert torch.allclose(rate, torch.full_like(rate, float(np.exp(1.234))), atol=1e-5)


def test_rate_link_exp_clamps_large_inputs_to_avoid_inf():
    torch.manual_seed(0)
    model = GATTemporal(in_channels=2, gat_hidden=4, heads=1, use_graph=False, use_temporal=False, rate_link="exp")
    with torch.no_grad():
        model.decoder.weight.zero_()
        model.decoder.bias.fill_(1000.0)  # would overflow exp() unclamped
        for layer in model.spatial_layers:
            layer.weight.zero_()
            layer.bias.zero_()

    x_seq = torch.zeros(1, 3, 2)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    rate = model(x_seq, edge_index)
    assert torch.isfinite(rate).all()


def test_gat_layers_two_stacks_a_second_spatial_layer():
    model = GATTemporal(in_channels=4, gat_hidden=8, heads=2, gat_layers=2)
    assert len(model.spatial_layers) == 2
    # First layer maps in_channels(4) -> gat_hidden*heads(16); second layer
    # must consume THAT width as its input, not the original in_channels -
    # this is the concrete "stacked", not "duplicated", check.
    assert model.spatial_layers[0].in_channels == 4
    assert model.spatial_layers[1].in_channels == 16


def test_gat_layers_two_trains_and_predicts_end_to_end():
    from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward

    rng = np.random.default_rng(9)
    n_segments, n_features = 6, 3
    edge_index = np.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=np.int64)
    train_instances = [
        (rng.normal(size=(2, n_segments, n_features)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(2)
    ]
    model = train_gat_temporal_walkforward(train_instances, edge_index, epochs=3, gat_layers=2, heads=2, gat_hidden=8)
    held_out_x = rng.normal(size=(2, n_segments, n_features)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()


def test_pairwise_rank_hinge_loss_hand_computed():
    # 1 positive (score=1.0), 2 negatives (scores 0.5, 2.0), margin=1.0.
    # violation vs neg1 (0.5): margin - (1.0-0.5) = 0.5
    # violation vs neg2 (2.0): margin - (1.0-2.0) = 2.0 (badly ranked below a negative)
    # mean = (0.5 + 2.0) / 2 = 1.25
    scores = torch.tensor([1.0, 0.5, 2.0])
    y = torch.tensor([1.0, 0.0, 0.0])
    loss = pairwise_rank_hinge_loss(scores, y, margin=1.0)
    assert loss.item() == pytest.approx(1.25, abs=1e-6)


def test_pairwise_rank_hinge_loss_zero_when_perfectly_ranked_beyond_margin():
    # Positive scores 5.0 above every negative - margin=1.0 fully satisfied.
    scores = torch.tensor([10.0, 1.0, 2.0])
    y = torch.tensor([1.0, 0.0, 0.0])
    loss = pairwise_rank_hinge_loss(scores, y, margin=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pairwise_rank_hinge_loss_returns_zero_with_no_positives_or_no_negatives():
    all_zero_y = torch.zeros(5)
    scores = torch.rand(5)
    assert pairwise_rank_hinge_loss(scores, all_zero_y).item() == 0.0

    all_positive_y = torch.ones(5)
    assert pairwise_rank_hinge_loss(scores, all_positive_y).item() == 0.0


def test_pairwise_rank_hinge_loss_prefers_correctly_ranked_scores():
    y = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    good_scores = torch.tensor([5.0, 4.0, 0.1, 0.2, 0.3])  # positives clearly on top
    bad_scores = torch.tensor([0.1, 0.2, 5.0, 4.0, 0.3])  # positives clearly at the bottom
    assert pairwise_rank_hinge_loss(good_scores, y).item() < pairwise_rank_hinge_loss(bad_scores, y).item()


def test_train_gat_temporal_walkforward_with_rank_loss_runs_end_to_end():
    from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward

    rng = np.random.default_rng(10)
    n_segments, n_features = 8, 3
    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    train_instances = [
        (rng.normal(size=(2, n_segments, n_features)).astype(np.float32), rng.poisson(1.0, size=n_segments).astype(np.float32))
        for _ in range(3)
    ]
    model = train_gat_temporal_walkforward(
        train_instances, edge_index, epochs=3, zero_inflated=True, rank_loss_weight=0.5, rank_margin=1.0,
    )
    held_out_x = rng.normal(size=(2, n_segments, n_features)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments,)
    assert np.isfinite(preds).all()


def test_train_gat_temporal_walkforward_with_rank_loss_and_multistep_horizon_runs():
    from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward

    rng = np.random.default_rng(11)
    n_segments, n_features, horizon = 8, 3, 5
    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    train_instances = [
        (
            rng.normal(size=(2, n_segments, n_features)).astype(np.float32),
            rng.poisson(0.8, size=(n_segments, horizon)).astype(np.float32),
        )
        for _ in range(3)
    ]
    model = train_gat_temporal_walkforward(
        train_instances, edge_index, epochs=3, horizon=horizon, zero_inflated=True, rank_loss_weight=0.5,
    )
    held_out_x = rng.normal(size=(2, n_segments, n_features)).astype(np.float32)
    preds = predict_gat_temporal(model, held_out_x, edge_index)
    assert preds.shape == (n_segments, horizon)
    assert np.isfinite(preds).all()


def _tiny_graph_inputs(n=12, t=4, f=5):
    x = torch.randn(t, n, f)
    edge_index = torch.tensor(
        np.array([[i, (i + 1) % n] for i in range(n)]).T, dtype=torch.long
    )
    return x, edge_index


def test_temporal_first_encoder_order_matches_the_papers_gru_then_gat_topology():
    # The paper's appendix A.3/A.4 states ZT = GRU(X_1:t, Y_1:t) and that
    # those temporal embeddings are then "directed to graph neural
    # encoders" - i.e. GRU FIRST, then GAT. This project's original order
    # is the reverse. Both must produce identical output shapes so they
    # are drop-in comparable in a sweep.
    x, edge_index = _tiny_graph_inputs()
    spatial = GATTemporal(in_channels=5, heads=3, horizon=7, zero_inflated=True, encoder_order="spatial_first")
    temporal = GATTemporal(in_channels=5, heads=3, horizon=7, zero_inflated=True, encoder_order="temporal_first")

    lam_s, pi_s = spatial(x, edge_index)
    lam_t, pi_t = temporal(x, edge_index)
    assert lam_s.shape == lam_t.shape == (12, 7)
    assert pi_s.shape == pi_t.shape == (12, 7)


def test_temporal_first_actually_uses_a_different_computation_path():
    # Guard against the encoder_order flag being silently inert (the class
    # of mistake that produced a bit-identical "new" experiment earlier in
    # this project - see tests/test_network.py's per_physical_road test).
    x, edge_index = _tiny_graph_inputs()
    torch.manual_seed(0)
    spatial = GATTemporal(in_channels=5, heads=3, horizon=7, zero_inflated=True, encoder_order="spatial_first")
    torch.manual_seed(0)
    temporal = GATTemporal(in_channels=5, heads=3, horizon=7, zero_inflated=True, encoder_order="temporal_first")

    # Different topology => different parameter counts (the GRU consumes
    # raw features rather than GAT output, and the GAT consumes the GRU's
    # hidden state rather than raw features).
    assert sum(p.numel() for p in spatial.parameters()) != sum(p.numel() for p in temporal.parameters())
    assert spatial.gru is not None and spatial.gru_first is None
    assert temporal.gru is None and temporal.gru_first is not None


def test_temporal_first_requires_the_temporal_encoder():
    with pytest.raises(ValueError, match="requires use_temporal=True"):
        model = GATTemporal(in_channels=5, encoder_order="temporal_first", use_temporal=False)
        model(*_tiny_graph_inputs())


def test_invalid_encoder_order_is_rejected_loudly():
    with pytest.raises(ValueError, match="encoder_order must be"):
        GATTemporal(in_channels=5, encoder_order="sideways")
