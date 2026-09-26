"""Negative controls for the arXiv form-field check in
scripts/verify_arxiv_package.py.

A check that has only ever been seen passing proves nothing: each test below
mutates one thing a human could plausibly get wrong when pasting into
arXiv's web form and asserts the check reports it.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import verify_arxiv_package as v  # noqa: E402

RAW_TEX = (ROOT / "paper" / "arxiv" / "main.tex").read_text(encoding="utf-8")
TEX = re.sub(r"(?<!\\)%.*$", "", RAW_TEX, flags=re.M)
MD = (ROOT / "paper" / "ARXIV_SUBMISSION.md").read_text(encoding="utf-8")


def test_real_files_match():
    assert v.form_field_mismatches(TEX, MD) == []


def test_stale_number_in_abstract_is_caught():
    bad = MD.replace("22.71%", "22.17%", 1)
    assert bad != MD, "fixture did not change - test is vacuous"
    problems = v.form_field_mismatches(TEX, bad)
    assert any(p.startswith("abstract differs") for p in problems)


def test_reworded_title_is_caught():
    bad = MD.replace("Trivial Baselines", "Simple Baselines", 1)
    assert bad != MD
    assert any(p.startswith("title:") for p in v.form_field_mismatches(TEX, bad))


def test_wrong_author_is_caught():
    bad = MD.replace("**Authors**: `Maurya Patel`", "**Authors**: `M. Patel`", 1)
    assert bad != MD
    assert any(p.startswith("authors:") for p in v.form_field_mismatches(TEX, bad))


def test_tex_percent_escape_pasted_into_form_is_caught():
    # \% is right in the .tex and wrong in the plain-text form.
    bad = MD.replace("83.94%", "83.94\\%", 1)
    assert bad != MD
    problems = v.form_field_mismatches(TEX, bad)
    assert any("TeX-only escape" in p for p in problems)


def test_title_linebreak_pasted_into_form_is_caught():
    bad = MD.replace("Baselines in Road-Level", "Baselines in\\\\ Road-Level", 1)
    assert bad != MD
    assert v.form_field_mismatches(TEX, bad) != []


def test_over_length_abstract_is_caught():
    m = re.search(r"```text\n(.*?)\n```", MD, re.S)
    bad = MD.replace(m.group(1), m.group(1) + " x" * 300, 1)
    problems = v.form_field_mismatches(TEX, bad)
    assert any("limit 1,920" in p for p in problems)


def test_missing_block_is_reported_not_crashed():
    assert v.form_field_mismatches(TEX, "# nothing here") != []
