import numpy as np
import pandas as pd

from greyspot.eval.metrics import (
    evaluate_predictions,
    precision_at_k,
    pr_auc_has_collision,
    rank_correlation,
)
from greyspot.eval.splits import spatial_split, temporal_split


def test_precision_at_k_perfect_ranking():
    y_true = np.array([0, 0, 5, 3, 0])
    y_score = np.array([0.1, 0.05, 0.9, 0.8, 0.01])
    # top-2 by score are indices 2 and 3, both have collisions -> precision 1.0
    assert precision_at_k(y_true, y_score, k=2) == 1.0


def test_precision_at_k_worst_ranking():
    y_true = np.array([0, 0, 5, 3, 0])
    y_score = np.array([0.9, 0.8, 0.01, 0.02, 0.7])
    # top-2 by score are indices 0 and 1, neither has a collision -> precision 0.0
    assert precision_at_k(y_true, y_score, k=2) == 0.0


def test_pr_auc_nan_when_no_positive_class():
    y_true = np.zeros(5)
    y_score = np.random.rand(5)
    assert np.isnan(pr_auc_has_collision(y_true, y_score))


def test_rank_correlation_perfect_monotonic():
    y_true = np.array([1, 2, 3, 4])
    y_score = np.array([10, 20, 30, 40])
    assert rank_correlation(y_true, y_score) == 1.0


def test_evaluate_predictions_returns_expected_keys():
    y_true = np.array([0, 1, 2, 0, 3])
    y_score = np.array([0.1, 0.4, 0.6, 0.2, 0.9])
    result = evaluate_predictions(y_true, y_score, k_values=(2,))
    assert set(result.keys()) == {
        "n", "n_with_collision", "pr_auc", "spearman_rank_corr", "precision_at_2",
    }


def test_temporal_split_separates_by_year():
    table = pd.DataFrame({"year": [2021, 2022, 2023, 2023], "x": [1, 2, 3, 4]})
    train, test = temporal_split(table, test_year=2023)
    assert set(train["year"]) == {2021, 2022}
    assert set(test["year"]) == {2023}


def test_spatial_split_holds_out_whole_segments_not_rows():
    table = pd.DataFrame(
        {
            "segment_id": ["a", "a", "b", "b", "c", "c"],
            "year": [2022, 2023] * 3,
        }
    )
    train, holdout = spatial_split(table, holdout_frac=1 / 3, seed=0)
    # no segment should appear in both splits
    assert set(train["segment_id"]).isdisjoint(set(holdout["segment_id"]))
    assert len(train) + len(holdout) == len(table)
