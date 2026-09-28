"""Negative controls for the Delaware-table check in
scripts/verify_arxiv_package.py (usrep_table_problems).

The table once shipped internally inconsistent at its printed precision
(30.10 - 16.16 = 13.94 beside a printed Range of 14.5) with no test able to
notice. Each case here mutates the table one way and asserts it is caught.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_arxiv_package as v  # noqa: E402

TEX = re.sub(r"(?<!\\)%.*$", "", (ROOT / "paper/arxiv/main.tex").read_text(encoding="utf-8"), flags=re.M)
CSV = (ROOT / "reports/horizon_us_de_thresholds.csv").read_text(encoding="utf-8")


def _mutate(old: str, new: str) -> str:
    assert old in TEX, "fixture text %r not in main.tex - test would be vacuous" % old
    return TEX.replace(old, new, 1)


def test_real_table_matches_source_and_is_self_consistent():
    assert v.usrep_table_problems(TEX, CSV) == []


def test_mistyped_cell_is_caught():
    bad = _mutate("16.16 & 30.10", "16.61 & 30.10")
    assert any("1 mo" in p for p in v.usrep_table_problems(bad, CSV))


def test_range_using_ten_year_definition_is_caught():
    # The reviewer's reading of the old table: Range as 10yr - 1mo = 13.94.
    bad = _mutate("30.63 (5 yr) & 14.5", "30.63 (5 yr) & 13.9")
    assert v.usrep_table_problems(bad, CSV) != []


def test_wrong_best_lookback_label_is_caught():
    bad = _mutate("30.63 (5 yr)", "30.63 (10 yr)")
    assert any("best lookback" in p for p in v.usrep_table_problems(bad, CSV))


def test_best_value_not_matching_source_is_caught():
    bad = _mutate("12.62 (3 yr)", "12.82 (3 yr)")
    assert v.usrep_table_problems(bad, CSV) != []


def test_missing_table_is_reported_not_crashed():
    assert v.usrep_table_problems("no table here", CSV) == ["Table tab:usrep not found in main.tex"]


def test_missing_cut_is_caught():
    bad = re.sub(r"Top 0\.2\\% &[^\n]*\n", "", TEX, count=1)
    assert bad != TEX
    assert v.usrep_table_problems(bad, CSV) != []
