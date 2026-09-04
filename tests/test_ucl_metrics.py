"""Tests for the UCL-paper-exact metrics (`eval/ucl_metrics.py`), each
checked against a hand-computed value - these numbers feed directly into
a comparison claim (docs/publication_readiness.md), so they get the same
"trust but verify" treatment as this project's own leakage-critical code.
"""
import numpy as np
import pytest

from greyspot.eval.ucl_metrics import (
    accuracy_hit_rate,
    mae,
    mape_excluding_zeros,
    mpiw,
    picp,
    rmse,
    true_zero_rate,
    ucl_metric_suite,
)


def test_mae_hand_computed():
    y_true = np.array([[1.0, 2.0, 0.0]])
    y_pred = np.array([[1.5, 1.0, 0.5]])
    # |1-1.5| + |2-1| + |0-0.5| = 0.5 + 1.0 + 0.5 = 2.0; /3 = 0.6667
    assert mae(y_true, y_pred) == pytest.approx(2.0 / 3)


def test_rmse_hand_computed():
    y_true = np.array([[0.0, 0.0]])
    y_pred = np.array([[3.0, 4.0]])
    # sqrt((9+16)/2) = sqrt(12.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_mape_excludes_zero_actuals_and_reports_excluded_fraction():
    y_true = np.array([[0.0, 2.0, 4.0, 0.0]])
    y_pred = np.array([[5.0, 1.0, 5.0, 5.0]])
    value, excluded = mape_excluding_zeros(y_true, y_pred)
    # only indices 1,2 have nonzero actuals: |2-1|/2=0.5, |4-5|/4=0.25 -> mean 0.375
    assert value == pytest.approx(0.375)
    assert excluded == pytest.approx(0.5)  # 2 of 4 observations were zero


def test_mape_all_zero_actuals_returns_nan_not_a_crash():
    y_true = np.zeros((1, 3))
    y_pred = np.array([[1.0, 2.0, 3.0]])
    value, excluded = mape_excluding_zeros(y_true, y_pred)
    assert np.isnan(value)
    assert excluded == 1.0


def test_mpiw_hand_computed():
    lower = np.array([[1.0, 2.0]])
    upper = np.array([[3.0, 5.0]])
    # widths 2.0 and 3.0 -> mean 2.5
    assert mpiw(lower, upper) == pytest.approx(2.5)


def test_picp_counts_true_value_inside_interval_not_predicted_mean():
    # Deliberately checks the disclosed departure from the PDF's literal
    # (self-contradictory) formula - PICP must test the ACTUAL value
    # against the interval, per the paper's own prose definition.
    y_true = np.array([[2.0, 10.0]])
    lower = np.array([[1.0, 1.0]])
    upper = np.array([[3.0, 3.0]])
    # first observation (true=2) is inside [1,3]; second (true=10) is not.
    assert picp(y_true, lower, upper) == pytest.approx(0.5)


def test_true_zero_rate_hand_computed():
    y_true = np.array([[0.0, 0.0, 1.0, 0.0]])
    y_pred = np.array([[0.1, 0.9, 0.1, 0.4]])
    # true zero at indices 0,1,3. predicted "zero" (< 0.5) at 0,2,3.
    # both true-zero AND predicted-zero: indices 0 and 3 -> 2/4 = 0.5
    assert true_zero_rate(y_true, y_pred, zero_threshold=0.5) == pytest.approx(0.5)


def test_accuracy_hit_rate_top_20_percent_hand_computed():
    # 5 segments, one day. Actual crashes on segments 0 and 4.
    y_true = np.array([[1.0, 0.0, 0.0, 0.0, 2.0]])
    y_pred = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    # top 20% of 5 segments = top 1 segment = index 0 (highest predicted).
    # of the 2 actual-crash segments (0 and 4), only segment 0 is in the
    # top-1 -> hit rate = 1/2 = 0.5
    assert accuracy_hit_rate(y_true, y_pred, top_fraction=0.20) == pytest.approx(0.5)


def test_accuracy_hit_rate_no_crashes_returns_nan():
    y_true = np.zeros((1, 5))
    y_pred = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    assert np.isnan(accuracy_hit_rate(y_true, y_pred))


def test_accuracy_hit_rate_selects_exactly_k_even_with_boundary_ties():
    # Regression test for the 2026-09-01 fix (docs/decision_log.md): the
    # old `y_pred >= threshold` implementation silently included EVERY
    # segment tied at the boundary value, not just the intended top-k -
    # harmless for continuous model outputs that rarely tie exactly, but
    # a real bug for sparse/integer-valued predictions (found via a
    # trivial-baseline sanity check that scored a suspicious 100%).
    # 10 segments: 2 have a clearly higher score (5.0, 4.0), the other 8
    # are all tied at exactly 0.0 - the true top-20% (k=2) must be exactly
    # those 2 segments, never all 10 just because 8 of them tie at the
    # threshold value.
    y_true = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    y_pred = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0]])
    # Both actual-crash segments (8, 9) are also the top-2 predicted -> hit rate 1.0.
    assert accuracy_hit_rate(y_true, y_pred, top_fraction=0.20) == pytest.approx(1.0)

    # Same shape, but the crash happened on a segment tied at the 0.0
    # boundary (not in the true top-2) - the old buggy implementation
    # would have called this a "hit" too (since the threshold at k=2 in a
    # field of eight 0.0s and two higher values is 4.0... but if MORE
    # segments tie at the boundary than k, the bug manifests instead as
    # over-inclusion). This case isolates that a genuinely out-of-top-k
    # segment is correctly scored as a miss, not swept in by a tie.
    y_true_miss = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert accuracy_hit_rate(y_true_miss, y_pred, top_fraction=0.20) == pytest.approx(0.0)


def test_accuracy_hit_rate_many_tied_zero_predictions_does_not_inflate_score():
    # The exact failure mode found in practice: almost all predictions are
    # 0 (a heavily zero-inflated count target), a handful are genuinely
    # informative, and top_fraction*n is smaller than the number of
    # segments tied at 0. The old `>= threshold` implementation degraded
    # to "threshold == 0.0 -> everyone with a non-negative score passes",
    # inflating the score toward 1.0 regardless of whether the model's
    # nonzero predictions were actually any good. 100 segments, top-20% =
    # 20; only 3 segments have a nonzero (informative) prediction.
    #
    # The actual-crash segment is given a value strictly *below* the large
    # tied-at-0.0 block (-1.0, the array's unique minimum) rather than
    # tied within it - `argpartition`'s tie-breaking among the 96
    # zero-valued segments that fill out the remaining 17 top-20 slots is
    # implementation-defined, so a crash segment placed *inside* that tied
    # block would make this test's outcome depend on numpy's internal
    # partition order, not on the fix being tested. A value below every
    # tied competitor is excluded from the top-20 regardless of how ties
    # among the *other* segments are broken - the only way to test "many
    # boundary ties don't inflate the score" without the test itself being
    # tie-order-dependent.
    n = 100
    y_pred = np.zeros((1, n))
    y_pred[0, [5, 6, 7]] = [3.0, 2.0, 1.0]
    y_pred[0, 50] = -1.0
    y_true = np.zeros((1, n))
    y_true[0, 50] = 1.0
    assert accuracy_hit_rate(y_true, y_pred, top_fraction=0.20) == pytest.approx(0.0)


def test_accuracy_hit_rate_averages_per_day_ratios_not_pooled_totals():
    # Regression test for the 2026-09-02 fix (re-extracted the paper's
    # exact Eq. 20 text: Acc@a = (1/p) * sum_j [hits_j / crashes_j], the
    # MEAN of each day's own ratio) - the old implementation instead
    # pooled hits and crashes across days into one ratio
    # (sum_j hits_j / sum_j crashes_j), which is only equal to the mean
    # when every day has the same crash count. This case deliberately
    # gives the two days very different crash counts (4 vs 1) so the two
    # formulas diverge, distinguishing "was this actually fixed" from
    # "does the fix accidentally not change anything here."
    #
    # 5 segments/day, top_fraction=0.20 -> k=1.
    # Day 1: segments 0-3 have crashes (4 total); segment 0 is also the
    #   top-1 prediction -> hits=1, ratio = 1/4 = 0.25.
    # Day 2: only segment 4 has a crash (1 total); segment 4 is also the
    #   top-1 prediction -> hits=1, ratio = 1/1 = 1.0.
    # Pooled (old, wrong): (1+1)/(4+1) = 0.4
    # Per-day mean (new, matches the paper's Eq. 20): (0.25+1.0)/2 = 0.625
    y_true = np.array([
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    y_pred = np.array([
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    ])
    result = accuracy_hit_rate(y_true, y_pred, top_fraction=0.20)
    assert result == pytest.approx(0.625)
    assert result != pytest.approx(0.4)  # the old pooled-formula value


def test_ucl_metric_suite_returns_all_named_keys():
    y_true = np.array([[0.0, 1.0, 2.0]])
    y_pred = np.array([[0.2, 1.1, 1.8]])
    lower = np.array([[0.0, 0.5, 1.0]])
    upper = np.array([[0.5, 1.5, 2.5]])
    result = ucl_metric_suite(y_true, y_pred, lower, upper)
    for key in ("MAE", "RMSE", "ZR", "AccHR", "MAPE_excl_zeros", "MAPE_excluded_fraction", "MPIW", "PICP"):
        assert key in result


def test_ucl_metric_suite_omits_interval_metrics_when_not_provided():
    y_true = np.array([[0.0, 1.0]])
    y_pred = np.array([[0.1, 0.9]])
    result = ucl_metric_suite(y_true, y_pred)
    assert "MPIW" not in result
    assert "PICP" not in result

