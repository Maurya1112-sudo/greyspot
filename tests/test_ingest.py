import pandas as pd

from greyspot.ingest.stats19 import WESTMINSTER_ONS_CODE, filter_to_local_authority


def test_filter_to_local_authority_keeps_only_matching_rows():
    df = pd.DataFrame(
        {
            "local_authority_ons_district": [WESTMINSTER_ONS_CODE, "E09000007", WESTMINSTER_ONS_CODE],
            "collision_severity": [1, 2, 3],
        }
    )
    out = filter_to_local_authority(df, WESTMINSTER_ONS_CODE)
    assert len(out) == 2
    assert set(out["local_authority_ons_district"]) == {WESTMINSTER_ONS_CODE}


def test_filter_to_local_authority_adds_severity_label():
    df = pd.DataFrame(
        {
            "local_authority_ons_district": [WESTMINSTER_ONS_CODE],
            "collision_severity": [1],
        }
    )
    out = filter_to_local_authority(df, WESTMINSTER_ONS_CODE)
    assert out.iloc[0]["severity_label"] == "fatal"


def test_filter_to_local_authority_empty_result_has_no_rows():
    df = pd.DataFrame(
        {
            "local_authority_ons_district": ["E09000007"],
            "collision_severity": [2],
        }
    )
    out = filter_to_local_authority(df, WESTMINSTER_ONS_CODE)
    assert len(out) == 0
