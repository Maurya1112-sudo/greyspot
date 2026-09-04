"""Ranking / count-aware evaluation metrics.

Raw accuracy is misleading on STATS19 segment-year data because it is highly
zero-inflated (most segments have zero collisions in a given year) — a
trivial "always predict zero" model scores well on accuracy/MAE while being
useless for prioritisation. Instead we evaluate whether the model *ranks*
higher-risk segments above lower-risk ones, per the dossier's evaluation
chapter (Section 14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def precision_at_k(y_true_count: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Of the top-k segments by predicted score, what fraction had >=1 actual collision?"""
    k = min(k, len(y_score))
    if k == 0:
        return float("nan")
    top_idx = np.argsort(-y_score)[:k]
    return float((y_true_count[top_idx] > 0).mean())


def pr_auc_has_collision(y_true_count: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC for the binary target 'segment had >=1 collision', scored by y_score."""
    y_binary = (y_true_count > 0).astype(int)
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return float("nan")  # undefined with no positive/negative class
    return float(average_precision_score(y_binary, y_score))


def rank_correlation(y_true_count: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman correlation between predicted score and actual count."""
    if np.all(y_true_count == y_true_count[0]) or np.all(y_score == y_score[0]):
        return float("nan")
    corr, _ = spearmanr(y_score, y_true_count)
    return float(corr)


def evaluate_predictions(
    y_true_count: np.ndarray, y_score: np.ndarray, k_values: tuple[int, ...] = (10, 25, 50)
) -> dict:
    """Bundle the standard Greyspot ranking-metric report for one model on one split."""
    result = {
        "n": len(y_true_count),
        "n_with_collision": int((y_true_count > 0).sum()),
        "pr_auc": pr_auc_has_collision(y_true_count, y_score),
        "spearman_rank_corr": rank_correlation(y_true_count, y_score),
    }
    for k in k_values:
        result[f"precision_at_{k}"] = precision_at_k(y_true_count, y_score, k)
    return result


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model") if rows and "model" in rows[0] else pd.DataFrame(rows)
