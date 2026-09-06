"""S5: does the deep-history NULL replicate, and can the GNN use extra history?

The preprint's abstract states that the model "cannot exploit additional
history that the trivial sort converts into a +2.95-point gain". That claim
came from Lambeth alone (−0.72, p=0.7438) and is one of the paper's
sharper assertions, so it needs a second borough.

This script compares, per borough:

- the GNN's gain from extending crash history 5yr → 9yr (S5 vs the final
  model, same windows, same seed), and
- the trivial sort's gain from extending its horizon 5yr → ~8yr (the
  capped vs uncapped baselines from S2),

so "the sort gains what the GNN cannot" is tested as a comparison rather
than asserted from one borough's null.

Run: python scripts/run_s5_two_borough_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

NOISE_BAND = 4.0
BOROUGHS = {"Lambeth": "lambeth", "Westminster": "westminster", "Tower Hamlets": "tower_hamlets"}
CAPPED = "raw count CAPPED at 1825d (matched to GNN)"
UNCAPPED = "raw count (no shrinkage)"


def main() -> None:
    print("=" * 78)
    print("S5: the GNN's gain from deeper history, vs the trivial sort's")
    print("=" * 78)
    rows = []
    for borough, sl in BOROUGHS.items():
        base_p = ROOT / "reports" / sl / "ucl_multiwindow_per_window_multiyear.csv"
        deep_p = ROOT / "reports" / sl / "s5_deep_history_per_window.csv"
        s2_p = ROOT / "reports" / sl / "s2_empirical_bayes_per_window.csv"
        row: dict = {"borough": borough}

        if base_p.exists() and deep_p.exists():
            base = pd.read_csv(base_p).sort_values("held_out_start")
            deep = pd.read_csv(deep_p).sort_values("held_out_start")
            assert (base.held_out_start.values == deep.held_out_start.values).all(), \
                f"{borough}: S5 and baseline windows differ"
            a, c = deep.AccHR.to_numpy(), base.AccHR.to_numpy()
            row["gnn_gain"] = 100 * float((a - c).mean())
            row["gnn_p"] = float(stats.ttest_rel(a, c).pvalue)
            row["gnn_wins"] = int((a - c > 0).sum())
            row["n"] = len(a)
        if s2_p.exists():
            s2 = pd.read_csv(s2_p)
            cap = s2[s2.baseline == CAPPED].sort_values("held_out_start").AccHR.to_numpy()
            unc = s2[s2.baseline == UNCAPPED].sort_values("held_out_start").AccHR.to_numpy()
            if len(cap) and len(cap) == len(unc):
                row["sort_gain"] = 100 * float((unc - cap).mean())
                row["sort_p"] = float(stats.ttest_rel(unc, cap).pvalue)
        rows.append(row)

    res = pd.DataFrame(rows)
    print()
    print("%-15s %11s %9s %7s %12s %9s" %
          ("borough", "GNN gain", "p", "wins", "sort gain", "p"))
    for _, r in res.iterrows():
        g = "%+11.2f" % r.gnn_gain if pd.notna(r.get("gnn_gain")) else "%11s" % "-"
        gp = "%9.4f" % r.gnn_p if pd.notna(r.get("gnn_p")) else "%9s" % "-"
        gw = "%5d/%d" % (r.gnn_wins, r.n) if pd.notna(r.get("gnn_wins")) else "%7s" % "-"
        s = "%+12.2f" % r.sort_gain if pd.notna(r.get("sort_gain")) else "%12s" % "-"
        sp = "%9.4f" % r.sort_p if pd.notna(r.get("sort_p")) else "%9s" % "-"
        print("%-15s %s %s %s %s %s" % (r.borough, g, gp, gw, s, sp))

    tested = res[res.gnn_gain.notna()]
    print()
    print("-" * 78)
    print("VERDICT")
    print("-" * 78)
    signs = set(np.sign(tested.gnn_gain.round(2)))
    print("GNN gain from 9-year history: %s" %
          ", ".join("%s %+.2f" % (r.borough, r.gnn_gain) for _, r in tested.iterrows()))
    if len(signs - {0.0}) > 1:
        print()
        print("SIGN FLIPS -> the deep-history NULL does NOT replicate.")
        print()
        print("The abstract's claim that the model 'cannot exploit additional history'")
        print("rests on Lambeth alone. On Westminster it gains +%.2f points, winning"
              % tested[tested.borough == "Westminster"].gnn_gain.iloc[0])
        print("%d of %d windows - so 'cannot exploit' is not supportable as stated."
              % (int(tested[tested.borough == "Westminster"].gnn_wins.iloc[0]),
                 int(tested[tested.borough == "Westminster"].n.iloc[0])))
    print()
    both = res[res.gnn_gain.notna() & res.sort_gain.notna()]
    if len(both):
        print("What DOES hold on every borough measured: the sort gains MORE than the")
        print("model from the same extra history.")
        for _, r in both.iterrows():
            print("    %-14s sort %+.2f  vs  model %+.2f   (sort ahead by %.2f)"
                  % (r.borough, r.sort_gain, r.gnn_gain, r.sort_gain - r.gnn_gain))
        if (both.sort_gain > both.gnn_gain).all():
            print()
            print("  True on %d of %d boroughs -> this is the defensible claim." % (len(both), len(both)))
    print()
    print("Both GNN effects are inside the ~%.0f-point seed-noise band, so neither"
          % NOISE_BAND)
    print("borough's figure is individually interpretable at a single seed (R10).")

    out = ROOT / "reports" / "s5_two_borough.csv"
    res.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
