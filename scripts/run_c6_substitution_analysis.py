"""C6: does the ARCHITECTURE x HISTORY substitution effect replicate?

Two separate claims live in the head-to-head experiment and they must not
be conflated:

1. **"Our architecture beats theirs"** - compares the two architectures at
   the same history depth. Already recorded as replicating.

2. **"The advantage is largely substitutable for history"** - the claim
   that the gap between the two architectures COLLAPSES once both are
   given long crash history, i.e. that much of the architectural advantage
   is really a proxy for information the features can carry directly. This
   was measured on Westminster (+24.42 -> +9.08) and recorded as needing a
   second borough.

This script tests claim 2 across all three boroughs. **No GPU run is
needed**: `headtohead_per_window.csv` already exists for Lambeth,
Westminster and Tower Hamlets - the MASTER_PLAN note saying Tower Hamlets
was missing was stale.

Replication criterion, fixed before computing: the substitution effect is
the change in the (ours - theirs) gap when both move from short to long
history. It replicates only if that change has the SAME SIGN on all three
boroughs. A sign reversal means the effect is borough-specific, which is
what failure to generalise means here - the same standard applied to road
class and rank scaling.
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
THEIRS_SHORT = "Gao et al. architecture (ZI-Tweedie, GRU->GAT, 2-layer/42), SHORT history"
THEIRS_LONG = "Gao et al. architecture, LONG history (the key test)"
OURS_SHORT = "This project's architecture, SHORT history"
OURS_LONG = "This project's architecture, LONG history (final model)"


def arm(d: pd.DataFrame, name: str) -> np.ndarray:
    a = d[d.candidate == name].sort_values("held_out_start")
    if a.empty:
        raise KeyError(name)
    return a.AccHR.to_numpy()


def main() -> None:
    rows, gaps = [], []
    for borough, sl in BOROUGHS.items():
        p = ROOT / "reports" / sl / "headtohead_per_window.csv"
        if not p.exists():
            print("%s: no head-to-head data - skipped" % borough)
            continue
        d = pd.read_csv(p)
        ts, tl, os_, ol = (arm(d, THEIRS_SHORT), arm(d, THEIRS_LONG),
                           arm(d, OURS_SHORT), arm(d, OURS_LONG))
        gap_short, gap_long = os_ - ts, ol - tl
        subst = gap_long - gap_short  # negative => the gap collapsed
        p_sub = float(stats.ttest_rel(gap_long, gap_short).pvalue)
        rows.append({
            "borough": borough, "n": len(ts),
            "theirs_short": 100 * ts.mean(), "theirs_long": 100 * tl.mean(),
            "ours_short": 100 * os_.mean(), "ours_long": 100 * ol.mean(),
            "gap_short": 100 * gap_short.mean(), "gap_long": 100 * gap_long.mean(),
            "substitution": 100 * subst.mean(), "p": p_sub,
            "theirs_gains": 100 * (tl - ts).mean(), "ours_gains": 100 * (ol - os_).mean(),
        })
        gaps.append((gap_short, gap_long))

    if len(rows) < 2:
        print("Need at least two boroughs.")
        return
    res = pd.DataFrame(rows)

    print("=" * 84)
    print("C6: does the architecture x history SUBSTITUTION effect replicate?")
    print("=" * 84)
    print()
    print("%-15s %9s %9s %9s %9s %9s" %
          ("borough", "gap SHORT", "gap LONG", "change", "theirs+", "ours+"))
    for _, r in res.iterrows():
        print("%-15s %+9.2f %+9.2f %+9.2f %+9.2f %+9.2f" %
              (r.borough, r.gap_short, r.gap_long, r.substitution, r.theirs_gains, r.ours_gains))

    gs = np.concatenate([g[0] for g in gaps]); gl = np.concatenate([g[1] for g in gaps])
    pooled = 100 * float((gl - gs).mean())
    p_pool = float(stats.ttest_rel(gl, gs).pvalue)
    print("%-15s %+9.2f %+9.2f %+9.2f   (pooled n=%d, p=%.4f)" %
          ("POOLED", 100 * gs.mean(), 100 * gl.mean(), pooled, len(gs), p_pool))

    print()
    print("-" * 84)
    print("VERDICT")
    print("-" * 84)
    signs = set(np.sign(res.substitution.round(2)))
    print("substitution effect per borough: %s" %
          ", ".join("%s %+.2f" % (r.borough, r.substitution) for _, r in res.iterrows()))
    if len(signs - {0.0}) > 1:
        print()
        print("SIGN REVERSES between boroughs -> the substitution effect does NOT replicate.")
        widen = res[res.substitution > 0].borough.tolist()
        shrink = res[res.substitution < 0].borough.tolist()
        print("  gap COLLAPSES on: %s" % ", ".join(shrink))
        print("  gap WIDENS on:    %s" % ", ".join(widen))
        print()
        print("  The pooled figure (%+.2f) averages effects pointing opposite ways and" % pooled)
        print("  must not be reported as the effect.")
        verdict = "FAILS - sign reverses between boroughs"
    else:
        print()
        print("Consistent sign across boroughs; pooled p=%.4f" % p_pool)
        verdict = "REPLICATES" if p_pool < 0.05 else "consistent sign but not significant"

    print()
    print("Note: 'our architecture beats theirs at long history' is a DIFFERENT")
    print("claim and is unaffected - it holds on all boroughs measured:")
    for _, r in res.iterrows():
        print("    %-15s ours %.2f vs theirs %.2f  (+%.2f)"
              % (r.borough, r.ours_long, r.theirs_long, r.gap_long))

    out = ROOT / "reports" / "c6_substitution.csv"
    res.to_csv(out, index=False)
    print()
    print("VERDICT: %s" % verdict)
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
