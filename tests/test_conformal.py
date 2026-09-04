import numpy as np
import pandas as pd
import pytest

from greyspot.models.conformal import (
    empirical_coverage,
    fit_split_conformal,
    manual_split_conformal_interval,
    mean_interval_width,
    predict_with_interval,
)


def test_empirical_coverage_basic_cases():
    y_true = np.array([1.0, 2.0, 3.0, 10.0])
    lower = np.array([0.0, 1.0, 2.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0, 5.0])
    # first three are inside their interval, the last (10) is not
    assert empirical_coverage(y_true, lower, upper) == 0.75


def test_mean_interval_width():
    lower = np.array([0.0, 1.0])
    upper = np.array([2.0, 5.0])
    assert mean_interval_width(lower, upper) == pytest.approx(3.0)


def _synthetic_table(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prior = rng.poisson(1.5, size=n).astype(float)
    return pd.DataFrame(
        {
            "length": rng.uniform(50, 300, size=n),
            "u_degree": rng.integers(1, 5, size=n),
            "v_degree": rng.integers(1, 5, size=n),
            "prior_year_count": prior,
            "prior_2yr_avg": prior,
            "prior_year_n_casualties": prior,
            "prior_year_n_fatal_casualties": np.zeros(n),
            "prior_year_n_serious_casualties": np.zeros(n),
            "prior_year_n_pedestrian_casualties": np.zeros(n),
            "prior_year_n_cyclist_casualties": np.zeros(n),
            "prior_year_n_vehicles_involved": prior,
            "prior_year_avg_imd_decile": rng.integers(1, 10, size=n).astype(float),
            # target correlates loosely with prior_year_count plus noise
            "collision_count": rng.poisson(prior + 0.5),
        }
    )


def test_fit_split_conformal_produces_valid_intervals_and_measurable_coverage():
    fit_table = _synthetic_table(200, seed=1)
    calib_table = _synthetic_table(100, seed=2)
    test_table = _synthetic_table(100, seed=3)

    conformal, point_model = fit_split_conformal(fit_table, calib_table, confidence_level=0.9)
    point, lower, upper = predict_with_interval(conformal, test_table)

    assert len(point) == len(test_table)
    assert (upper >= lower).all()
    assert (lower >= 0).all()  # clipped at zero - collision counts can't be negative

    coverage = empirical_coverage(test_table["collision_count"].to_numpy(), lower, upper)
    # With a 90% target and n=100 test points, a reasonable-but-not-exact
    # coverage is expected; just check it's in a sane ballpark, not a fixed
    # number - avoids an over-specified, flaky assertion.
    assert 0.5 <= coverage <= 1.0


def test_manual_split_conformal_interval_matches_expected_quantile_logic():
    # Calibration residuals are all exactly 2.0 -> the interval half-width
    # at any confidence level must be exactly 2.0 too (not an approximation).
    calib_true = np.array([10.0, 8.0, 5.0, 3.0])
    calib_pred = np.array([8.0, 10.0, 7.0, 1.0])  # |residual| = 2.0 for every point
    test_pred = np.array([5.0, 0.5])

    lower, upper = manual_split_conformal_interval(calib_true, calib_pred, test_pred, confidence_level=0.9)

    assert np.allclose(upper, test_pred + 2.0)
    assert lower[0] == pytest.approx(3.0)  # 5.0 - 2.0
    assert lower[1] == 0.0  # 0.5 - 2.0 would be negative -> clipped at zero


def test_manual_split_conformal_interval_produces_sane_coverage_on_real_residual_distribution():
    rng = np.random.default_rng(7)
    calib_true = rng.poisson(2.0, size=500).astype(float)
    calib_pred = np.full(500, 2.0)  # a constant "point model" - residuals are just the Poisson noise
    test_true = rng.poisson(2.0, size=500).astype(float)
    test_pred = np.full(500, 2.0)

    lower, upper = manual_split_conformal_interval(calib_true, calib_pred, test_pred, confidence_level=0.9)
    coverage = empirical_coverage(test_true, lower, upper)
    assert 0.8 <= coverage <= 1.0  # should land close to the 90% target, not far off
