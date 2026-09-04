import pandas as pd
import pytest

from greyspot.product.priority_score import (
    DEFAULT_POLICY_PROFILE,
    PolicyProfile,
    compute_priority_score,
)


def _toy_table():
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "model_score": [0.1, 5.0, 2.0],
            "prior_year_n_fatal_casualties": [0, 1, 0],
            "prior_year_n_serious_casualties": [0, 1, 0],
            "prior_year_n_slight_casualties": [1, 0, 3],
            "prior_year_n_pedestrian_casualties": [0, 1, 0],
            "prior_year_n_cyclist_casualties": [0, 0, 0],
            "prior_year_count": [1, 2, 3],
            "prior_2yr_avg": [1.0, 1.0, 3.0],
            "u_degree": [2, 4, 3],
            "v_degree": [2, 4, 3],
        }
    )


def test_policy_profile_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to"):
        PolicyProfile(name="bad", weights={**DEFAULT_POLICY_PROFILE.weights, "risk_signal": 0.99})


def test_policy_profile_rejects_missing_weight_keys():
    with pytest.raises(ValueError, match="missing"):
        PolicyProfile(name="incomplete", weights={"risk_signal": 1.0})


def test_compute_priority_score_raises_on_missing_required_column():
    table = _toy_table().drop(columns=["u_degree"])
    with pytest.raises(KeyError, match="u_degree"):
        compute_priority_score(table, risk_score_col="model_score")


def test_compute_priority_score_produces_score_in_0_100_with_components():
    table = _toy_table()
    out = compute_priority_score(table, risk_score_col="model_score")

    assert (out["priority_score"] >= 0).all() and (out["priority_score"] <= 100).all()
    for col in [
        "component_risk_signal", "component_severity", "component_vulnerable_users",
        "component_trend", "component_network_importance", "component_data_confidence",
    ]:
        assert col in out.columns

    # segment "b" has the highest risk score, worst severity, and a vulnerable-user
    # casualty - it should rank highest overall.
    assert out.loc[out["segment_id"] == "b", "priority_score"].iloc[0] == out["priority_score"].max()


def test_compute_priority_score_flags_default_confidence_when_no_interval_given():
    table = _toy_table()
    out = compute_priority_score(table, risk_score_col="model_score")
    assert out["data_confidence_is_default"].all()
    assert (out["component_data_confidence"] == 0.5).all()


def test_compute_priority_score_uses_real_confidence_when_interval_provided():
    table = _toy_table()
    table["interval_width"] = [5.0, 0.5, 2.0]  # segment "b" has the narrowest (most confident) interval
    out = compute_priority_score(table, risk_score_col="model_score", interval_width_col="interval_width")
    assert not out["data_confidence_is_default"].any()
    assert out.loc[out["segment_id"] == "b", "component_data_confidence"].iloc[0] == out["component_data_confidence"].max()


def test_compute_priority_score_applies_minimum_data_rule_for_trend():
    # segment with almost no prior history (prior_2yr_avg below the
    # minimum-data threshold) must get a neutral trend component (0.5),
    # not a spurious "worsening"/"improving" signal from noise.
    table = _toy_table()
    table.loc[0, "prior_2yr_avg"] = 0.0  # segment "a": too little history
    out = compute_priority_score(table, risk_score_col="model_score")
    assert out.loc[out["segment_id"] == "a", "component_trend"].iloc[0] == 0.5


def test_compute_priority_score_records_policy_profile_name():
    table = _toy_table()
    out = compute_priority_score(table, risk_score_col="model_score")
    assert (out["policy_profile"] == "default-v1").all()

    custom = PolicyProfile(name="equity-weighted-v1", weights={**DEFAULT_POLICY_PROFILE.weights})
    out2 = compute_priority_score(table, risk_score_col="model_score", profile=custom)
    assert (out2["policy_profile"] == "equity-weighted-v1").all()
