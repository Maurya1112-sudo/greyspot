"""Per-window checkpointing for the walk-forward evaluation scripts.

Every multi-window script in this project holds its per-window results in
a list in memory and writes its CSV only after the LAST window finishes.
That is fine for the 6-window runs (minutes) but not for the 38-window
dense evaluations, which take 4-6 hours: a machine shutdown, a CUDA OOM,
or an accidental Ctrl-C anywhere in that span destroys every completed
window. This already happened once (dense eval interrupted at 31/38; the
numbers had to be scraped back out of the log - see
`reports/lambeth/dense_eval_31windows_recovered.csv` and
`docs/decision_log.md`).

This module makes those runs RESUMABLE: each window is appended to a CSV
and fsync'd the moment it completes, and a restart skips windows already
present. Worst case a shutdown costs the one window currently training.

**The fingerprint guard is the load-bearing part.** Resuming is only
valid if the restarted run is computing the SAME quantity - same
borough, same instance grid, same stride/horizon, same model config. A
checkpoint keyed on `held_out_start` alone would happily let a run with
different features silently adopt another config's windows and report
the average as one number. That is exactly the class of bug that
produced this project's false-retraction incident (see rule R3 in
`docs/MASTER_PLAN.md`), so a fingerprint mismatch REFUSES to resume
rather than guessing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

FINGERPRINT_COLUMN = "run_fingerprint"


def compute_fingerprint(**parts: Any) -> str:
    """Stable short hash of everything that must match for a resume to be
    valid. Sorted keys so argument order never changes the result."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_checkpoint(path: Path, candidate: str, fingerprint: str) -> dict[pd.Timestamp, dict]:
    """Completed windows for `candidate`, keyed by held-out start date.

    Raises if the file was written by a run with a different fingerprint -
    a stale checkpoint must be deleted deliberately, never silently reused.
    """
    if not path.exists() or path.stat().st_size == 0:
        return {}
    done = pd.read_csv(path)
    if done.empty:
        return {}
    missing = {"candidate", "held_out_start", FINGERPRINT_COLUMN} - set(done.columns)
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing column(s) {sorted(missing)} - it predates "
            f"checkpointing or is corrupt. Delete it to start this run from scratch."
        )
    # Drop torn rows BEFORE the fingerprint check. A hard shutdown mid-append
    # can truncate the final line, which pandas parses into a row of NaNs -
    # including a NaN fingerprint, which would otherwise read as "written by a
    # different configuration" and refuse to resume the very run this module
    # exists to rescue. Such a row is not a completed window either way, so it
    # is discarded and that window is re-run.
    torn = done["held_out_start"].isna() | done["AccHR"].isna() | done[FINGERPRINT_COLUMN].isna()
    if torn.any():
        logger.warning(
            "Dropping %d incomplete row(s) from %s (torn write from an interrupted run)",
            int(torn.sum()), path,
        )
        done = done[~torn]
    if done.empty:
        return {}
    stale = set(done[FINGERPRINT_COLUMN].unique()) - {fingerprint}
    if stale:
        raise ValueError(
            f"Checkpoint {path} was written by a DIFFERENT configuration "
            f"(fingerprint(s) {sorted(stale)}, this run is {fingerprint}). Resuming would "
            f"mix results from two different experiments into one average. "
            f"Delete the file to re-run, or point this run at a different checkpoint path."
        )
    done = done[done["candidate"] == candidate].copy()
    if done.empty:
        return {}
    done["held_out_start"] = pd.to_datetime(done["held_out_start"])
    # Later duplicates win: a window re-run after a partial write should
    # supersede the earlier row rather than be silently ignored.
    return {row["held_out_start"]: row.to_dict() for _, row in done.iterrows()}


def append_checkpoint(path: Path, row: dict) -> None:
    """Append ONE completed window and force it to disk.

    The fsync matters: without it the row can sit in the OS write cache
    and be lost by exactly the hard shutdown this module exists to
    survive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        pd.DataFrame([row]).to_csv(handle, header=header, index=False)
        handle.flush()
        os.fsync(handle.fileno())
