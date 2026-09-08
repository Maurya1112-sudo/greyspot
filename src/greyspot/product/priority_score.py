"""Transparent Safety Priority Score (project dossier Section 5A).

This is a **policy/decision layer**, deliberately kept separate from the
prediction models in `greyspot.models`: the model estimates evidence (a
relative-risk number); this module makes an explicit, versioned policy
choice about how to combine that evidence with severity, vulnerable-user
relevance, trend, network importance and data confidence into one 0-100
ranking number analysts can actually act on. Per the dossier: "The score
is a decision layer, not the prediction model itself... must be labelled
as a project policy prototype, not discovered truth."

Every score comes with its component breakdown (`components` dict) so an
evidence panel can show *why* a segment ranked where it did - never a bare
number (dossier Section 5.5, "Why this road?" explainability).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Illustrative default weights, matching the dossier's Section 5A table
# exactly. These are a labelled policy choice, not a validated finding -
# a real deployment would need these signed off by the government analyst
# stakeholders the dossier names, not inferred from data.
DEFAULT_WEIGHTS = {
    "risk_signal": 0.35,
    "severity": 0.25,
    "vulnerable_users": 0.15,
    "trend": 0.10,
    "network_importance": 0.10,
    "data_confidence": 0.05,
}

# Below this many prior-period observations, a trend is reported as
# "insufficient data" rather than "worsening"/"improving" - dossier
# Section 5.6: "a small-sample worsening state should become uncertain
# trend rather than an alert."
MIN_OBSERVATIONS_FOR_TREND = 2.0


@dataclass(frozen=True)
class PolicyProfile:
    """A named, versioned set of priority-score weights. Two profiles can
    be compared side by side ("policy profile A vs B") without retraining
    any model - exactly the dossier's recommended governance pattern."""

    name: str
    weights: dict[str, float]

    def __post_init__(self):
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Policy profile {self.name!r} weights sum to {total}, not 1.0")
        missing = set(DEFAULT_WEIGHTS) - set(self.weights)
        if missing:
            raise ValueError(f"Policy profile {self.name!r} is missing weight(s): {sorted(missing)}")


DEFAULT_POLICY_PROFILE = PolicyProfile(name="default-v1", weights=DEFAULT_WEIGHTS)


def _minmax_normalise(series: pd.Series) -> pd.Series:
    """Scale to [0, 1] within the given batch. A documented simplification:
    this is relative to *this* evaluation batch, not a fixed universal
    scale - re-running on a different set of segments will shift what
    "high" means. Fine for a first version; a production system would fix
    reference bounds from a calibration set instead."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def compute_priority_score(
    table: pd.DataFrame,
    risk_score_col: str,
    profile: PolicyProfile = DEFAULT_POLICY_PROFILE,
    interval_width_col: str | None = None,
) -> pd.DataFrame:
    """Compute a transparent 0-100 priority score with a visible
    component breakdown, for one evaluation batch (e.g. one held-out
    test set for one year).

    Required columns in `table`: `risk_score_col`, `prior_year_n_fatal_casualties`,
    `prior_year_n_serious_casualties`, `prior_year_n_slight_casualties`,
    `prior_year_n_pedestrian_casualties`, `prior_year_n_cyclist_casualties`,
    `prior_year_count`, `prior_2yr_avg`, `u_degree`, `v_degree`.
    Missing columns raise `KeyError` rather than silently contributing 0 -
    a missing column contributing 0 would misrepresent "we don't have this
    evidence" as "this component found no risk", which is exactly the kind
    of hidden-feature behaviour the dossier's data-quality principles rule
    out (see `features/build_features.py`'s exposure-handling docstring for
    the same principle applied elsewhere).

    `interval_width_col` (optional): a conformal prediction-interval width
    column: narrower -> higher `data_confidence` component. If omitted,
    `data_confidence` is a neutral 0.5 for every row and flagged as such -
    per the dossier, absent confidence data must never be presented as
    "safe" or silently folded into the risk signal.
    """
    required = [
        risk_score_col,
        "prior_year_n_fatal_casualties", "prior_year_n_serious_casualties", "prior_year_n_slight_casualties",
        "prior_year_n_pedestrian_casualties", "prior_year_n_cyclist_casualties",
        "prior_year_count", "prior_2yr_avg", "u_degree", "v_degree",
    ]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise KeyError(
            f"compute_priority_score is missing required column(s) {missing} - "
            "a real evidence gap must be visible, not silently treated as zero risk."
        )

    out = table.copy()
    w = profile.weights

    # 1. Risk signal: the model's own evidence, normalised.
    component_risk = _minmax_normalise(out[risk_score_col].fillna(0.0))

    # 2. Severity: weighted mix of prior-year casualty severity (fatal
    #    counts most, slight least), normalised - a segment with the same
    #    collision count but worse outcomes ranks higher.
    severity_index = (
        3 * out["prior_year_n_fatal_casualties"].fillna(0.0)
        + 2 * out["prior_year_n_serious_casualties"].fillna(0.0)
        + 1 * out["prior_year_n_slight_casualties"].fillna(0.0)
    )
    component_severity = _minmax_normalise(severity_index)

    # 3. Vulnerable users: share of prior-year CASUALTIES who were
    #    pedestrians/cyclists (0 when there's no prior history, not "safe").
    #
    #    BUG FOUND AND FIXED (2026-09-08, via the evidence panel's own
    #    "what does this number mean" explanation surfacing it): this used
    #    to divide by `prior_year_count` - a COLLISION count, not a
    #    casualty count. One collision can injure several people, so a
    #    segment with 1 collision and 4 pedestrian casualties produced a
    #    "share" of 4.0 (400%), which is not a share of anything. The
    #    correct denominator is the total casualty count (fatal + serious
    #    + slight), the same population `vulnerable_count` is drawn from -
    #    every casualty has exactly one severity band and one road-user
    #    type per STATS19, so this ratio is structurally bounded to [0, 1].
    vulnerable_count = out["prior_year_n_pedestrian_casualties"].fillna(0.0) + out["prior_year_n_cyclist_casualties"].fillna(0.0)
    total_casualties = (
        out["prior_year_n_fatal_casualties"].fillna(0.0)
        + out["prior_year_n_serious_casualties"].fillna(0.0)
        + out["prior_year_n_slight_casualties"].fillna(0.0)
    )
    vulnerable_share = np.where(total_casualties > 0, vulnerable_count / total_casualties.replace(0, np.nan), 0.0)
    # Defensive clip, not load-bearing given the STATS19 guarantee above -
    # cheap insurance against a future denominator change reintroducing
    # an out-of-range "share".
    component_vulnerable = pd.Series(np.clip(np.nan_to_num(vulnerable_share), 0.0, 1.0), index=out.index)

    # 4. Trend: is the latest year higher than the recent average? Gated
    #    by a minimum-data rule (dossier Section 5.6) - too little history
    #    to call a trend gets a neutral 0.5, not a false "improving" signal.
    has_enough_history = out["prior_2yr_avg"].fillna(0.0) >= MIN_OBSERVATIONS_FOR_TREND / 2
    trend_raw = out["prior_year_count"].fillna(0.0) - out["prior_2yr_avg"].fillna(0.0)
    trend_normalised = _minmax_normalise(trend_raw)
    component_trend = pd.Series(np.where(has_enough_history, trend_normalised, 0.5), index=out.index)

    # 5. Network importance: how connected this segment's endpoints are.
    component_network = _minmax_normalise((out["u_degree"].fillna(0.0) + out["v_degree"].fillna(0.0)))

    # 6. Data confidence: narrower conformal interval -> more confident.
    #    Never used to imply safety on its own (dossier: "never used to
    #    imply safety") - it only ever contributes its small, fixed weight.
    if interval_width_col is not None and interval_width_col in out.columns:
        component_confidence = 1 - _minmax_normalise(out[interval_width_col].fillna(out[interval_width_col].max()))
        confidence_is_default = pd.Series(False, index=out.index)
    else:
        component_confidence = pd.Series(0.5, index=out.index)
        confidence_is_default = pd.Series(True, index=out.index)

    out["component_risk_signal"] = component_risk
    out["component_severity"] = component_severity
    out["component_vulnerable_users"] = component_vulnerable
    out["component_trend"] = component_trend
    out["component_network_importance"] = component_network
    out["component_data_confidence"] = component_confidence
    out["data_confidence_is_default"] = confidence_is_default

    out["priority_score"] = 100 * (
        w["risk_signal"] * component_risk
        + w["severity"] * component_severity
        + w["vulnerable_users"] * component_vulnerable
        + w["trend"] * component_trend
        + w["network_importance"] * component_network
        + w["data_confidence"] * component_confidence
    )
    out["policy_profile"] = profile.name

    return out
