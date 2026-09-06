"""Two claims settled at 5 seeds, from one regenerated arm.

Supersedes the single-seed figures behind:

1. **"The reference architecture loses to a crash-count sort on 18 of 18
   windows"** — the paper's strongest claim, and until now single-seed.
2. **"Our architecture beats theirs at matched history depth"** — C6's
   surviving half, also single-seed.

Both are now paired against a 5-seed opponent:

| arm | source | seeds |
|---|---|---|
| theirs LONG | `headtohead_multiseed_per_window.csv` | 5 |
| ours LONG | `headline_multiseed_per_window.csv` (same config, verified) | 5 |
| crash-count sort | `s2_empirical_bayes_per_window.csv` | deterministic |

**How the pairing is done, and why it matters.** For claim 2 both sides
are model runs, so they pair on **seed as well as window**: a seed's bias
is then shared between the arms and cancels. For claim 1 the sort has no
seed, so the model side is seed-averaged per window first and the pairing
is on windows alone. Mixing these would understate or overstate the
variance depending on which way round it was done.

**A single-seed result is also reported alongside**, not to be quoted but
to show what the original evidence looked like — the point of the exercise
is whether 5 seeds change the answer.

Run: python scripts/run_headtohead_multiseed_analysis.py
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
SORT = "raw count (no shrinkage)"
CAPPED = "raw count CAPPED at 1825d (matched to GNN)"
EB = "EB (HSM method)"
NOISE_BAND = 4.0


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")
    return 100 * float(d.mean()), t_p, w_p, int((d > 0).sum())


def main() -> None:
    theirs, ours, base, missing = [], [], [], []
    for borough, sl in BOROUGHS.items():
        t_p = ROOT / "reports" / sl / "headtohead_multiseed_per_window.csv"
        o_p = ROOT / "reports" / sl / "headline_multiseed_per_window.csv"
        b_p = ROOT / "reports" / sl / "s2_empirical_bayes_per_window.csv"
        if not t_p.exists():
            missing.append(borough)
            continue
        t = pd.read_csv(t_p)
        t = t.groupby("seed").filter(lambda g: len(g) == 6)
        if t.seed.nunique() < 5:
            missing.append("%s (%d/5 seeds)" % (borough, t.seed.nunique()))
            if t.seed.nunique() == 0:
                continue
        o = pd.read_csv(o_p)
        b = pd.read_csv(b_p)
        for df in (t, o, b):
            df["held_out_start"] = pd.to_datetime(df["held_out_start"])
        t["borough"] = o["borough"] = borough
        b["borough"] = borough
        theirs.append(t); ours.append(o); base.append(b)

    if missing:
        print("INCOMPLETE: %s - figures below are provisional" % "; ".join(missing))
        print()
    if not theirs:
        print("No data yet.")
        return

    T = pd.concat(theirs, ignore_index=True)
    O = pd.concat(ours, ignore_index=True)
    B = pd.concat(base, ignore_index=True)

    print("=" * 82)
    print("CLAIM 1: reference architecture vs the trivial sort (5 seeds vs deterministic)")
    print("=" * 82)
    tw = T.groupby(["borough", "held_out_start"], as_index=False).AccHR.mean()
    print("%-32s %8s %8s %9s %7s %10s %10s" %
          ("baseline", "theirs", "base", "diff", "wins", "t-test p", "Wilcoxon p"))
    rows = []
    for label, name in [("crash-count sort (~8yr)", SORT),
                        ("sort capped to 5yr", CAPPED),
                        ("Empirical Bayes (HSM)", EB)]:
        bb = B[B.baseline == name][["borough", "held_out_start", "AccHR"]]
        m = tw.merge(bb, on=["borough", "held_out_start"], suffixes=("_t", "_b"))
        if m.empty:
            continue
        diff, tp, wp, wins = paired(m.AccHR_t.to_numpy(), m.AccHR_b.to_numpy())
        print("%-32s %8.2f %8.2f %+9.2f %4d/%-2d %10.6f %10.6f" %
              (label, 100 * m.AccHR_t.mean(), 100 * m.AccHR_b.mean(), diff, wins, len(m), tp, wp))
        rows.append({"claim": "reference vs " + label, "n": len(m), "diff": diff,
                     "wins": wins, "t_p": tp, "w_p": wp})

    print()
    print("=" * 82)
    print("CLAIM 2: our architecture vs theirs, matched long history (both 5 seeds)")
    print("=" * 82)
    # paired on seed AND window - a seed's bias is shared and cancels
    m2 = O.merge(T, on=["borough", "seed", "held_out_start"], suffixes=("_o", "_t"))
    print("%-16s %5s %8s %8s %9s %7s %10s" %
          ("borough", "n", "ours", "theirs", "diff", "wins", "t-test p"))
    for borough, g in m2.groupby("borough"):
        diff, tp, wp, wins = paired(g.AccHR_o.to_numpy(), g.AccHR_t.to_numpy())
        print("%-16s %5d %8.2f %8.2f %+9.2f %4d/%-2d %10.6f" %
              (borough, len(g), 100 * g.AccHR_o.mean(), 100 * g.AccHR_t.mean(), diff, wins, len(g), tp))
    diff, tp, wp, wins = paired(m2.AccHR_o.to_numpy(), m2.AccHR_t.to_numpy())
    print("%-16s %5d %8.2f %8.2f %+9.2f %4d/%-2d %10.6f  <<<" %
          ("POOLED", len(m2), 100 * m2.AccHR_o.mean(), 100 * m2.AccHR_t.mean(), diff, wins, len(m2), tp))
    rows.append({"claim": "ours vs theirs (paired on seed+window)", "n": len(m2),
                 "diff": diff, "wins": wins, "t_p": tp, "w_p": wp})

    print()
    print("-" * 82)
    print("PER-SEED SPREAD of the reference arm (is a single seed interpretable?)")
    print("-" * 82)
    ps = T.groupby(["borough", "seed"]).AccHR.mean().mul(100)
    for borough, g in ps.groupby("borough"):
        print("  %-16s %s   (spread %.2f pts)" %
              (borough, "  ".join("%.2f" % v for v in g), g.max() - g.min()))
    print()
    print("  Largest per-borough spread: %.2f points, against a ~%.0f-point noise band."
          % (ps.groupby("borough").apply(lambda g: g.max() - g.min()).max(), NOISE_BAND))

    out = ROOT / "reports" / "headtohead_multiseed_analysis.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
