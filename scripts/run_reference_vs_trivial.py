"""Does the REFERENCE architecture also lose to a crash-count sort?

Section 3 of the preprint shows our GNN does not beat sorting road
segments by past crash count. The obvious objection is that this says
something about our implementation rather than about the model class.

This tests the reference architecture (Gao et al.'s STZITD-GNN as
reproduced in `run_headtohead_stzitd.py`) against the same trivial
baselines, on the same held-out windows.

**Scope.** This is their ARCHITECTURE on OUR data and protocol, at the
learning rate we found by sweeping on their behalf - not a re-evaluation
of their published numbers, which cannot be reproduced without their
per-window outputs. The claim it supports is narrow and specific: the
trivial baseline is not a bar that our model uniquely falls under.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BOROUGHS = {"Lambeth": "lambeth", "Westminster": "westminster", "Tower Hamlets": "tower_hamlets"}
THEIRS = "Gao et al. architecture, LONG history (the key test)"
OURS = "This project's architecture, LONG history (final model)"
BASELINES = {
    "crash-count sort (~8yr)": "raw count (no shrinkage)",
    "sort capped to 5yr": "raw count CAPPED at 1825d (matched to GNN)",
    "Empirical Bayes (HSM)": "EB (HSM method)",
}


def main() -> None:
    arms: dict[str, list[np.ndarray]] = {"theirs": [], "ours": []}
    base: dict[str, list[np.ndarray]] = {k: [] for k in BASELINES}
    for borough, sl in BOROUGHS.items():
        h = pd.read_csv(ROOT / "reports" / sl / "headtohead_per_window.csv")
        s2 = pd.read_csv(ROOT / "reports" / sl / "s2_empirical_bayes_per_window.csv")
        for key, name in [("theirs", THEIRS), ("ours", OURS)]:
            arms[key].append(h[h.candidate == name].sort_values("held_out_start").AccHR.to_numpy())
        for label, name in BASELINES.items():
            base[label].append(s2[s2.baseline == name].sort_values("held_out_start").AccHR.to_numpy())

    theirs = np.concatenate(arms["theirs"])
    ours = np.concatenate(arms["ours"])
    print("=" * 78)
    print("Reference architecture vs trivial baselines (pooled, n=%d windows)" % len(theirs))
    print("=" * 78)
    print()
    print("  reference architecture (swept optimum, long history)  %.2f" % (100 * theirs.mean()))
    print("  our architecture (long history, seed 42)              %.2f" % (100 * ours.mean()))
    print()
    print("%-28s %8s %9s %12s %9s" % ("baseline", "score", "ref diff", "paired p", "their wins"))
    rows = []
    for label in BASELINES:
        b = np.concatenate(base[label])
        d = theirs - b
        p = float(stats.ttest_rel(theirs, b).pvalue)
        wins = int((d > 0).sum())
        print("%-28s %8.2f %+9.2f %12.6f %6d/%-2d" % (label, 100 * b.mean(), 100 * d.mean(), p, wins, len(d)))
        rows.append({"baseline": label, "baseline_score": 100 * b.mean(),
                     "reference_score": 100 * theirs.mean(), "diff": 100 * d.mean(),
                     "p": p, "reference_wins": wins, "n": len(d)})
    print()
    print("Per borough (reference vs the uncapped sort):")
    for (borough, _), t, b in zip(BOROUGHS.items(), arms["theirs"], base["crash-count sort (~8yr)"]):
        print("  %-15s reference %.2f   sort %.2f   %+.2f" % (borough, 100 * t.mean(), 100 * b.mean(),
                                                              100 * (t - b).mean()))
    out = ROOT / "reports" / "reference_vs_trivial.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
