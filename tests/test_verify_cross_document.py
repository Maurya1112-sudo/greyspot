"""Negative controls for scripts/verify_cross_document.py (rule R17: a check
that has only ever been seen passing proves nothing).

The checker pinned a STALE matched-horizon figure (-0.85) and did not know
the reference architecture's single-seed 64.45% had been withdrawn, so both
survived in public documents after the paper moved on (found 2026-09-28).
Each test below injects one stale claim into a scratch document and asserts
the checker reports exactly that claim.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_cross_document.py"


def _load():
    spec = importlib.util.spec_from_file_location("vcd_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vcd_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_on(tmp_path, text):
    """Run the checker's main() on one scratch document; return {label: ok}."""
    mod = _load()
    doc = tmp_path / "scratch.md"
    doc.write_text(text, encoding="utf-8")
    mod.DOCS = {"scratch.md": doc}
    mod.results.clear()
    with pytest.raises(SystemExit):  # scratch has no canonical figures, so it exits 1
        mod.main()
    return {label: ok for ok, label, _ in mod.results}


def test_matched_horizon_canonical_is_derived_from_the_artefact():
    mod = _load()
    assert mod._matched_horizon_gap() == ["-0.90", "−0.90"]


@pytest.mark.parametrize("stale", ["64.45%", "0.6398", "80.14%"])
def test_stale_value_in_prose_is_caught(tmp_path, stale):
    seen = _run_on(tmp_path, "A plain claim that the model scored %s here.\n" % stale)
    assert seen["superseded %s not asserted" % stale] is False


@pytest.mark.parametrize("stale", ["64.45%", "0.6398", "80.14%"])
def test_same_value_in_a_correction_note_is_allowed(tmp_path, stale):
    seen = _run_on(tmp_path, "Earlier figure %s was superseded by the 5-seed value.\n" % stale)
    assert seen["superseded %s not asserted" % stale] is True


def test_clean_document_passes_the_superseded_checks(tmp_path):
    seen = _run_on(tmp_path, "The reference architecture scores 66.57% over five seeds.\n")
    assert all(ok for label, ok in seen.items() if label.startswith("superseded"))


def test_real_repository_documents_pass():
    import subprocess
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-1500:]
