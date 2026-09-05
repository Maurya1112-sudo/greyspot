"""C2 across three boroughs: does "fewer features == the full 35" replicate?

**Why a third borough.** C2 was recorded as PARTIALLY replicating: Lambeth
showed -0.99 points (p=0.35, a tie) but Westminster showed -4.08 (p=0.092),
so the claim was downgraded pending more evidence. This project has already
seen three claims survive one borough and die on the second (road class
reversed sign; weight_decay shrank monotonically to nothing; rank scaling
went from best-ever to worst-ever), so a two-borough disagreement is not
something to average away.

**What replication means here**, decided before looking at the third
borough's numbers so the criterion cannot be fitted to them:

- REPLICATES as a null ("fewer features are as good") only if the smallest
  feature set is statistically indistinguishable from 35 on ALL THREE
  boroughs, and the pooled paired test is also non-significant.
- FAILS if the effect reverses sign between boroughs, regardless of
  significance - a sign flip means the effect is borough-specific, which is
  exactly what "does not generalise" means.
- Effects smaller than the measured ~4-point seed-noise band are reported
  as indistinguishable from noise whatever their p-value, per R10.

Run: python scripts/run_c2_three_borough_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# Lambeth's ladder was run before the script was renamed for the
# replication; the file differs in name only, same candidates and windows.
SOURCES = {
    "Lambeth": ["reports/lambeth/feature_ladder_per_window.csv",
                "reports/lambeth/c2_ladder_repl_per_window.csv"],
    "Westminster": ["reports/westminster/c2_ladder_repl_per_window.csv"],
    "Tower Hamlets": ["reports/tower_hamlets/c2_ladder_repl_per_window.csv"],
}
SMALLEST = "12 features (geometry + crash history only)"
FULL = "35 features (+ socio-demographic = FINAL)"
NOISE_BAND = 4.0  # points; V8/V11 measured seed spread of 3.2-4.7


def load(borough: str, paths: list[str]) -> pd.DataFrame | None:
    for rel in paths:
        p = ROOT / rel
        if p.exists():
            d = pd.read_csv(p)
            d["borough"] = borough
            return d
    return None


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")
    return 100 * float(d.mean()), t_p, w_p, int((d > 0).sum())


def main() -> None:
    frames, missing = [], []
    for borough, paths in SOURCES.items():
        d = load(borough, paths)
        (frames if d is not None else missing).append(d if d is not None else borough)
    if missing:
        print("Not yet available: %s" % ", ".join(missing))
    if len(frames) < 2:
        print("Need at least two boroughs; stopping.")
        return

    all_d = pd.concat(frames, ignore_index=True)
    print("=" * 76)
    print("C2 ACROSS %d BOROUGHS: smallest feature set vs the full 35" % len(frames))
    print("=" * 76)

    rows = []
    for borough, grp in all_d.groupby("borough"):
        s = grp[grp.candidate == SMALLEST].sort_values("held_out_start")
        f = grp[grp.candidate == FULL].sort_values("held_out_start")
        if len(s) != len(f) or len(s) == 0:
            print("%s: candidate rows do not pair (%d vs %d) - skipped" % (borough, len(s), len(f)))
            continue
        assert (s.held_out_start.values == f.held_out_start.values).all(), \
            f"{borough}: windows differ between candidates"
        diff, t_p, w_p, wins = paired(s.AccHR.to_numpy(), f.AccHR.to_numpy())
        rows.append({"borough": borough, "n": len(s),
                     "small": 100 * s.AccHR.mean(), "full": 100 * f.AccHR.mean(),
                     "diff": diff, "wins": wins, "t_p": t_p, "w_p": w_p})

    pooled_s, pooled_f = [], []
    for borough, grp in all_d.groupby("borough"):
        s = grp[grp.candidate == SMALLEST].sort_values("held_out_start")
        f = grp[grp.candidate == FULL].sort_values("held_out_start")
        if len(s) == len(f) and len(s):
            pooled_s.append(s.AccHR.to_numpy()); pooled_f.append(f.AccHR.to_numpy())
    ps, pf = np.concatenate(pooled_s), np.concatenate(pooled_f)
    diff, t_p, w_p, wins = paired(ps, pf)
    rows.append({"borough": "POOLED", "n": len(ps), "small": 100 * ps.mean(),
                 "full": 100 * pf.mean(), "diff": diff, "wins": wins, "t_p": t_p, "w_p": w_p})

    res = pd.DataFrame(rows)
    print()
    print("%-16s %3s %8s %8s %9s %7s %10s %10s" %
          ("borough", "n", "12 feat", "35 feat", "diff", "wins", "t-test p", "Wilcoxon p"))
    for _, r in res.iterrows():
        mark = "  <<<" if r.borough == "POOLED" else ""
        print("%-16s %3d %8.2f %8.2f %+9.2f %4d/%-2d %10.4f %10.4f%s" %
              (r.borough, r.n, r.small, r["full"], r["diff"], r.wins, r.n, r.t_p, r.w_p, mark))

    per_borough = res[res.borough != "POOLED"]
    signs = set(np.sign(per_borough["diff"].round(2)))
    pooled = res[res.borough == "POOLED"].iloc[0]

    print()
    print("-" * 76)
    print("VERDICT")
    print("-" * 76)
    print("per-borough differences: %s" %
          ", ".join("%s %+.2f" % (r.borough, r["diff"]) for _, r in per_borough.iterrows()))
    if len(signs - {0.0}) > 1:
        print()
        print("SIGN FLIPS between boroughs -> the effect is BOROUGH-SPECIFIC.")
        print("  The claim does NOT replicate, whatever the pooled p-value says:")
        print("  a pooled average over effects pointing opposite ways is not an effect.")
        verdict = "FAILS - sign flips between boroughs"
    elif (per_borough.t_p > 0.05).all() and pooled.t_p > 0.05:
        print()
        print("No borough differs significantly and the pooled test is null.")
        print("  The claim REPLICATES as a null: the extra features are droppable.")
        verdict = "REPLICATES as a null"
    else:
        print()
        print("Consistent sign but at least one borough (or the pool) is significant.")
        print("  The extra features MATTER; the original null does not replicate.")
        verdict = "FAILS - extra features are significant"
    if abs(pooled["diff"]) < NOISE_BAND:
        print()
        print("NOTE: the pooled effect (%.2f points) is inside the ~%.0f-point seed-noise"
              % (abs(pooled["diff"]), NOISE_BAND))
        print("      band, so it must be reported as indistinguishable from noise (R10)")
        print("      regardless of its p-value.")

    out = ROOT / "reports" / "c2_three_borough.csv"
    res.to_csv(out, index=False)
    print()
    print("VERDICT: %s" % verdict)
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
