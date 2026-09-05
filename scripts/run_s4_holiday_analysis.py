"""S4 - is the Christmas/New Year period systematically harder to predict?

The dense 38-window evaluation (2026-09-04) found that three of its five
WORST windows fell between 8 December and 19 January:

  2023-12-08  65.22%   |   2023-12-22  61.82%   |   2024-01-19  61.11%

against a run mean of 78.84%. That is either a real seasonal limitation
worth reporting, or coincidence in a noisy metric.

**Why it would be real**: holiday traffic diverges sharply from the rest
of the year - commuting collapses, retail and night-time activity spike,
and the spatial distribution of exposure shifts. A model trained
predominantly on normal weeks would rank that period poorly.

**Why it might not be**: the dense grid has 38 windows, so some cluster
of low scores is expected by chance, and this project has repeatedly
found apparent patterns in single-borough single-seed data that did not
survive checking.

This tests it directly on the dense-eval results already computed, with
no new training: compare holiday-period windows against the rest, and
check whether the effect tracks crash density (a confound - if holiday
windows simply contain fewer crashes, the metric is noisier there for
reasons unrelated to season).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from greyspot.ingest.boroughs import get_borough  # noqa: E402
from greyspot.ingest.stats19 import load_local_authority_collisions  # noqa: E402

HORIZON = 14


def is_holiday_window(start: pd.Timestamp) -> bool:
    """A window overlapping 8 Dec - 19 Jan, the stretch identified in the
    dense evaluation. Defined once, from the observed cluster, and not
    tuned afterwards."""
    end = start + pd.Timedelta(days=HORIZON)
    for yr in {start.year, end.year}:
        if start <= pd.Timestamp(f"{yr}-01-19") and end >= pd.Timestamp(f"{yr - 1}-12-08"):
            return True
    return False


def main() -> None:
    # The dense run was interrupted at 31/38 windows and never wrote its CSV;
    # its log was later lost when /tmp was cleared. The per-window values were
    # preserved and are recovered here, with dates reconstructed from the known
    # stride-14 grid starting 2023-07-07 (verified against the per-window table
    # recorded on 2026-09-04). Documented rather than silently re-derived.
    src = ROOT / "reports" / "lambeth" / "dense_eval_31windows_recovered.csv"
    if not src.exists():
        print("recovered dense-eval file not found; S4 needs its per-window results")
        return
    df = pd.read_csv(src)
    df = df.rename(columns={"held_out_start": "start", "AccHR": "acchr"})
    df["start"] = pd.to_datetime(df["start"])
    df["holiday"] = df.start.map(is_holiday_window)

    b = get_borough("Lambeth")
    col = load_local_authority_collisions(b.ons_code, [2023, 2024], ROOT / "data" / "raw")
    col["cdate"] = pd.to_datetime(col["date"], format="%d/%m/%Y")
    df["n_crashes"] = [
        int(((col.cdate >= s) & (col.cdate < s + pd.Timedelta(days=HORIZON))).sum()) for s in df.start
    ]

    hol, norm = df[df.holiday], df[~df.holiday]
    print()
    print("=== S4: HOLIDAY-PERIOD WINDOWS (Lambeth dense eval, n=%d) ===" % len(df))
    print("  holiday windows (8 Dec - 19 Jan): n=%d, mean AccHR %.4f, mean crashes %.1f"
          % (len(hol), hol.acchr.mean(), hol.n_crashes.mean()))
    print("  all other windows:                n=%d, mean AccHR %.4f, mean crashes %.1f"
          % (len(norm), norm.acchr.mean(), norm.n_crashes.mean()))
    if len(hol) >= 2 and len(norm) >= 2:
        t, p = stats.ttest_ind(hol.acchr, norm.acchr, equal_var=False)
        print("  difference: %+.2f points, Welch t-test p=%.4f" % (100 * (hol.acchr.mean() - norm.acchr.mean()), p))
        tc, pc = stats.ttest_ind(hol.n_crashes, norm.n_crashes, equal_var=False)
        print("  crash-count difference: %+.1f, p=%.4f  <- CONFOUND CHECK" % (hol.n_crashes.mean() - norm.n_crashes.mean(), pc))
        r, pr = stats.pearsonr(df.n_crashes, df.acchr)
        print("  AccHR vs crash count across ALL windows: r=%.3f, p=%.4f" % (r, pr))
    print()
    print("  windows sorted by score (worst 8):")
    for _, row in df.nsmallest(8, "acchr").iterrows():
        print("    %s  %.4f  crashes=%3d  %s" % (row.start.date(), row.acchr, row.n_crashes,
                                                  "HOLIDAY" if row.holiday else ""))


if __name__ == "__main__":
    main()
