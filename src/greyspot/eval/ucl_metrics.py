"""The exact evaluation metrics from Gao et al. (2024) ("Uncertainty-Aware
Probabilistic Graph Neural Networks for Road-Level Traffic Crash
Prediction", arXiv:2309.05072v4, Section 4.3), implemented directly from
their published formulas (Eq. 13-16) - built to make a genuine,
apples-to-apples comparison possible (see docs/publication_readiness.md),
not this project's own PR-AUC/Precision@K, which measures something
different.

All functions take `y_true`/`y_pred` shaped `[p, N]` (p = forecast
horizon in days, N = number of road segments) - matching the paper's own
double sum over `j` (time step) and `i` (road), so a result computed here
is directly the same quantity the paper reports, not a rescaled proxy.

Two deliberate, disclosed departures from a literal reading of the paper,
both because the paper's stated formula is unusable or self-contradictory
as written - not silent "improvements":

1. **MAPE** (Eq. 13) divides by `y_ij`, which is 0 for the vast majority
   of observations in this (and the paper's own) zero-inflated dataset -
   undefined/infinite for every true zero. The paper never states how it
   handles this. Implemented here as MAPE over **non-zero actuals only**
   (a standard, named convention - "MAPE excluding zeros"), with the
   excluded fraction always returned alongside the number so it is never
   silently misleading.
2. **PICP** (Eq. 15 as OCR'd from the PDF) reads `I(U_ij < yhat_ij <
   L_ij)` - comparing the interval bounds against the *predicted* value,
   with the inequality direction reversed (upper bound less than lower
   bound is never true). The paper's own prose defines PICP as "whether
   the ground truth value, denoted y_ij, lies within the predicted
   interval" - implemented exactly per that prose (`L <= y_true <= U`),
   which is also the standard, universal definition of PICP in the
   conformal-prediction literature this project's own conformal work
   already follows (`models/conformal.py`).
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 13. Mean Absolute Error over every (day, segment) pair."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape_excluding_zeros(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """A disclosed departure from Eq. 13 - see module docstring. Returns
    `(mape, excluded_fraction)`; `excluded_fraction` is the share of
    (day, segment) observations with `y_true == 0` that had to be dropped
    to compute this at all - report it alongside the number, since a high
    excluded fraction (expected, given zero-inflation) means this MAPE
    value is describing a small, non-representative subset of the data."""
    nonzero = y_true != 0
    excluded_fraction = float(1.0 - nonzero.mean()) if y_true.size else 0.0
    if not nonzero.any():
        return float("nan"), excluded_fraction
    value = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
    return value, excluded_fraction


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 13. Root Mean Squared Error over every (day, segment) pair."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Eq. 14. Mean Prediction Interval Width."""
    return float(np.mean(upper - lower))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Eq. 15, per the paper's own prose definition (see module
    docstring for why this departs from the PDF's literal, self-
    contradictory formula symbol) - fraction of (day, segment)
    observations where the true value falls within [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def true_zero_rate(y_true: np.ndarray, y_pred: np.ndarray, zero_threshold: float = 0.5) -> float:
    """Eq. 16 (ZR). Fraction of (day, segment) pairs where the actual
    count is zero AND the model's own prediction also rounds to zero
    (`zero_threshold` is the rounding cutoff for a continuous point
    prediction - 0.5 means "predicts fewer than half a crash", the
    natural cutoff for a count; not stated explicitly in the paper, whose
    own decoder has an explicit zero-inflation gate to call this from
    directly, which this project's own ZIP decoder also has - see
    `models/gat_temporal.py`)."""
    true_zero = y_true == 0
    pred_zero = y_pred < zero_threshold
    return float(np.mean(true_zero & pred_zero))


def accuracy_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, top_fraction: float = 0.20) -> float:
    """Eq. 16 (AccHR@a). Of the (day, segment) pairs with an actual crash
    (`y_true > 0`), what fraction fall within that day's own top
    `top_fraction` (default 20%, matching the paper's AccHR@20) of
    predicted-risk segments. Computed per day (per row of the `[p, N]`
    arrays), matching the paper's own per-time-step ranking - a segment
    is only ever compared against other segments on the *same* day, never
    across days, since risk ranking is inherently a same-day comparison.

    Selects **exactly** `k = round(n * top_fraction)` segments via
    `argpartition` + explicit index selection, not a `y_pred >= threshold`
    boolean mask (fixed 2026-09-01: the threshold approach silently
    included every segment tied at the boundary value, not just k of
    them - harmless for a model whose continuous outputs essentially
    never tie exactly, but a real correctness bug for any sparse/integer-
    valued or heavily-regularised prediction array, where dozens or
    hundreds of segments can share the exact same boundary value - found
    while sanity-checking a suspiciously perfect trivial-baseline score
    during the 2026-09-01 data-richness pass, see docs/decision_log.md).
    Ties beyond the k-th position are broken by `argpartition`'s own
    (arbitrary but deterministic) internal order, matching standard
    top-k ranking-evaluation convention - "if tied, which one counts is
    unspecified, but exactly k are counted" - rather than being either
    silently permissive (the old bug) or silently order-dependent on the
    caller's own array layout.

    **Averaging fixed 2026-09-02** (found by re-extracting the paper's
    exact Eq. 20 text, not re-deriving it from memory): the paper's own
    formula is `(1/p) * sum_j [hits_j / crashes_j]` - the MEAN of each
    day's own hit ratio - not a single ratio of totals pooled across all
    `p` days (`sum_j hits_j / sum_j crashes_j`), which is what this
    function computed before this fix. The two are only equal when every
    day has the same crash count; otherwise the pooled version lets
    high-crash days dominate the average, understating a model that does
    well on typical (low-crash) days but poorly on one unusually busy
    one, or vice versa. Days with zero actual crashes are still excluded
    from the average entirely (0/0 is undefined, not 0) - the paper's
    own `(1/p)` literally divides by every one of the `p` days including
    crash-free ones, which is a genuine, disclosed ambiguity in their
    formula (implicitly defining crash-free days as contributing 0, not
    skipping them) that this project does not adopt: for THEIR data
    (4,822-5,659 roads per borough, Table 3), a calendar day with truly
    zero crashes network-wide is plausibly rare-to-nonexistent, so the
    distinction may not matter much for them - but for this project's
    smaller multi-window test slices it very much can, and "skip
    undefined days" is the more defensible choice of the two available
    ambiguous readings, not an attempt to inflate the score (it makes no
    difference in which DIRECTION the choice biases the result without
    knowing the true crash-day distribution in advance).
    """
    p, n = y_true.shape
    k = max(1, int(round(n * top_fraction)))
    daily_ratios = []
    for j in range(p):
        crash_mask = y_true[j] > 0
        n_crashes = int(crash_mask.sum())
        if n_crashes == 0:
            continue
        top_k_mask = np.zeros(n, dtype=bool)
        if n > k:
            top_k_idx = np.argpartition(-y_pred[j], k - 1)[:k]
            top_k_mask[top_k_idx] = True
        else:
            top_k_mask[:] = True
        hits_j = int(np.sum(crash_mask & top_k_mask))
        daily_ratios.append(hits_j / n_crashes)
    if not daily_ratios:
        return float("nan")
    return float(np.mean(daily_ratios))


def ucl_metric_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    top_fraction: float = 0.20,
) -> dict[str, float]:
    """All of Gao et al. (2024)'s reported metrics in one call, keyed
    exactly by their paper's own metric names (Table 4) - the
    single entry point a comparison report should call, so no caller
    accidentally computes only a subset and calls it "the UCL metrics"."""
    result = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "ZR": true_zero_rate(y_true, y_pred),
        "AccHR": accuracy_hit_rate(y_true, y_pred, top_fraction=top_fraction),
    }
    mape_value, excluded = mape_excluding_zeros(y_true, y_pred)
    result["MAPE_excl_zeros"] = mape_value
    result["MAPE_excluded_fraction"] = excluded
    if lower is not None and upper is not None:
        result["MPIW"] = mpiw(lower, upper)
        result["PICP"] = picp(y_true, lower, upper)
    return result
