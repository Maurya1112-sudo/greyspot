"""Tests for resumable walk-forward checkpointing.

The behaviour that matters is the REFUSAL path: a checkpoint written by a
different configuration must never be silently adopted, because the
failure mode is a plausible-looking average silently mixing two
experiments.
"""
from __future__ import annotations

import pandas as pd
import pytest

from greyspot.eval.checkpoint import (
    FINGERPRINT_COLUMN,
    append_checkpoint,
    compute_fingerprint,
    load_checkpoint,
)


def _row(date: str, acchr: float, fingerprint: str, candidate: str = "cand") -> dict:
    return {
        "candidate": candidate,
        "held_out_start": date,
        "AccHR": acchr,
        "n_train_instances": 10,
        FINGERPRINT_COLUMN: fingerprint,
    }


def test_missing_file_returns_empty(tmp_path):
    assert load_checkpoint(tmp_path / "nope.csv", "cand", "abc") == {}


def test_append_then_load_roundtrip(tmp_path):
    path = tmp_path / "ckpt.csv"
    fp = compute_fingerprint(borough="Westminster", stride=14)
    append_checkpoint(path, _row("2023-07-07", 0.7667, fp))
    append_checkpoint(path, _row("2023-07-21", 0.6838, fp))

    done = load_checkpoint(path, "cand", fp)
    assert set(done) == {pd.Timestamp("2023-07-07"), pd.Timestamp("2023-07-21")}
    assert done[pd.Timestamp("2023-07-07")]["AccHR"] == pytest.approx(0.7667)


def test_header_written_exactly_once(tmp_path):
    path = tmp_path / "ckpt.csv"
    fp = compute_fingerprint(a=1)
    for i, date in enumerate(["2023-07-07", "2023-07-21", "2023-08-04"]):
        append_checkpoint(path, _row(date, 0.5 + i / 100, fp))
    assert path.read_text().count("candidate,held_out_start") == 1
    assert len(pd.read_csv(path)) == 3


def test_different_fingerprint_refuses_to_resume(tmp_path):
    """The load-bearing guard: a checkpoint from another config must NOT be
    reused, or two experiments get averaged into one number."""
    path = tmp_path / "ckpt.csv"
    append_checkpoint(path, _row("2023-07-07", 0.7667, compute_fingerprint(stride=14)))
    with pytest.raises(ValueError, match="DIFFERENT configuration"):
        load_checkpoint(path, "cand", compute_fingerprint(stride=90))


def test_fingerprint_is_order_independent_but_value_sensitive():
    assert compute_fingerprint(a=1, b=2) == compute_fingerprint(b=2, a=1)
    assert compute_fingerprint(a=1, b=2) != compute_fingerprint(a=1, b=3)
    assert compute_fingerprint(features=["x", "y"]) != compute_fingerprint(features=["y", "x"])


def test_other_candidates_are_not_returned(tmp_path):
    path = tmp_path / "ckpt.csv"
    fp = compute_fingerprint(a=1)
    append_checkpoint(path, _row("2023-07-07", 0.70, fp, candidate="A"))
    append_checkpoint(path, _row("2023-07-07", 0.80, fp, candidate="B"))
    assert load_checkpoint(path, "A", fp)[pd.Timestamp("2023-07-07")]["AccHR"] == pytest.approx(0.70)
    assert load_checkpoint(path, "B", fp)[pd.Timestamp("2023-07-07")]["AccHR"] == pytest.approx(0.80)


def test_legacy_csv_without_fingerprint_is_rejected(tmp_path):
    """A pre-checkpointing per-window CSV has no fingerprint column. It must
    raise rather than be treated as 'nothing done yet' (which would double
    the work) or as valid (which would skip windows it cannot vouch for)."""
    path = tmp_path / "legacy.csv"
    pd.DataFrame([{"candidate": "cand", "held_out_start": "2023-07-07", "AccHR": 0.7}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_checkpoint(path, "cand", compute_fingerprint(a=1))


def test_empty_file_is_treated_as_nothing_done(tmp_path):
    """A shutdown during the very first append leaves a zero-byte file. That
    is 'no windows done', not an error - the run must be able to start."""
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert load_checkpoint(path, "cand", compute_fingerprint(a=1)) == {}


def test_torn_final_line_is_dropped_not_counted(tmp_path):
    """A hard shutdown can truncate the last row mid-write. That window did
    NOT complete, so it must be re-run, not skipped."""
    path = tmp_path / "torn.csv"
    fp = compute_fingerprint(a=1)
    append_checkpoint(path, _row("2023-07-07", 0.7667, fp))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"cand,2023-07-21,")  # truncated mid-row: no AccHR
    done = load_checkpoint(path, "cand", fp)
    assert set(done) == {pd.Timestamp("2023-07-07")}
