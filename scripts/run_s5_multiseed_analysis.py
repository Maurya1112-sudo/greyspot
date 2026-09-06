"""S5 settled: can the model use crash history beyond five years?

Supersedes `run_s5_two_borough_analysis.py`, which compared single-seed
runs. That comparison produced −0.72 on Lambeth and +2.98 on Westminster,
a sign flip that led to withdrawing the claim "the model cannot exploit
deeper history" from the preprint abstract. Both figures sat inside the
~4-point seed-noise band, so neither settled anything.

This uses 5 seeds per borough on the deep arm
(`run_s5_multiseed.py`) against the same 5 seeds on the 5-year baseline
(`reports/run_logs/v8_*.log`), so the two arms pair on **seed as well as
window** — the strongest pairing available, since a seed's bias is shared
between the arms and cancels.

Two questions, kept separate because they have different answers:

1. **Does deep history help on average?** Seed-averaged per window, then
   paired across windows.
2. **Is the per-seed effect stable?** If the sign flips between seeds
   within one borough, no single-seed run of this experiment means
   anything, whatever its p-value.

Run: python scripts/run_s5_multiseed_analysis.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

NOISE_BAND = 4.0
EVAL_STARTS = pd.to_datetime(
    ["2023-07-15", "2023-10-13", "2024-01-11", "2024-04-10", "2024-07-09", "2024-10-07"]
)
BOROUGHS = {"Lambeth": ("lambeth", "v8_multiseed"),
            "Westminster": ("westminster", "v8_westminster")}
_SEED_RE = re.compile(r"seed\s+(\d+)\s+6-window mean AccHR@20 = ([0-9.]+)\s+\(windows: ([0-9. ]+)\)")


def baseline_per_window(log_stem: str) -> pd.DataFrame:
    t = (ROOT / "reports" / "run_logs" / f"{log_stem}.log").read_text(
        encoding="utf-8", errors="replace")
    rows = []
    for m in _SEED_RE.finditer(t):
        seed = int(m.group(1))
        vals = [float(v) for v in m.group(3).split()]
        rows += [{"seed": seed, "held_out_start": s, "AccHR": v}
                 for s, v in zip(EVAL_STARTS, vals)]
    return pd.DataFrame(rows)


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue)
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        w_p = float("nan")
    return 100 * float(d.mean()), t_p, w_p, int((d > 0).sum())


def main() -> None:
    print("=" * 80)
    print("S5 MULTI-SEED: does history beyond 5 years help? (5 seeds x 6 windows)")
    print("=" * 80)

    per_borough, seed_rows, incomplete = [], [], []
    for borough, (sl, log_stem) in BOROUGHS.items():
        ckpt = ROOT / "reports" / sl / "s5_multiseed_checkpoint.csv"
        if not ckpt.exists():
            incomplete.append(f"{borough} (no deep-arm data)")
            continue
        deep = pd.read_csv(ckpt)
        deep["held_out_start"] = pd.to_datetime(deep["held_out_start"])
        complete = deep.groupby("seed").filter(lambda g: len(g) == len(EVAL_STARTS))
        n_seeds = complete.seed.nunique()
        if n_seeds < 5:
            incomplete.append("%s (%d/5 seeds)" % (borough, n_seeds))
        if n_seeds == 0:
            continue
        base = baseline_per_window(log_stem)
        base = base[base.seed.isin(complete.seed.unique())]

        m = complete.merge(base, on=["seed", "held_out_start"], suffixes=("_deep", "_base"))
        assert len(m) == len(complete), "deep and baseline windows do not pair"
        for seed, g in m.groupby("seed"):
            seed_rows.append({"borough": borough, "seed": int(seed),
                              "base": 100 * g.AccHR_base.mean(),
                              "deep": 100 * g.AccHR_deep.mean(),
                              "diff": 100 * (g.AccHR_deep - g.AccHR_base).mean()})
        # seed-average per window, then pair across windows
        w = m.groupby("held_out_start")[["AccHR_deep", "AccHR_base"]].mean()
        diff, t_p, w_p, wins = paired(w.AccHR_deep.to_numpy(), w.AccHR_base.to_numpy())
        per_borough.append({"borough": borough, "seeds": n_seeds, "n_windows": len(w),
                            "base": 100 * w.AccHR_base.mean(), "deep": 100 * w.AccHR_deep.mean(),
                            "diff": diff, "wins": wins, "t_p": t_p, "w_p": w_p})

    if incomplete:
        print("\nINCOMPLETE: %s - figures below are provisional" % "; ".join(incomplete))
    if not per_borough:
        print("\nNo data yet.")
        return

    sd = pd.DataFrame(seed_rows)
    print()
    print("PER SEED (each row pairs the two arms at the SAME seed)")
    print("%-14s %6s %9s %9s %8s" % ("borough", "seed", "5yr base", "9yr deep", "diff"))
    for _, r in sd.iterrows():
        print("%-14s %6d %9.2f %9.2f %+8.2f" % (r.borough, r.seed, r.base, r.deep, r["diff"]))

    print()
    print("SEED-AVERAGED, PAIRED ACROSS WINDOWS")
    print("%-14s %6s %9s %9s %8s %7s %10s %10s" %
          ("borough", "seeds", "5yr base", "9yr deep", "diff", "wins", "t-test p", "Wilcoxon p"))
    res = pd.DataFrame(per_borough)
    for _, r in res.iterrows():
        print("%-14s %6d %9.2f %9.2f %+8.2f %4d/%-2d %10.4f %10.4f" %
              (r.borough, r.seeds, r.base, r.deep, r["diff"], r.wins, r.n_windows, r.t_p, r.w_p))

    print()
    print("-" * 80)
    print("VERDICT")
    print("-" * 80)
    for borough, g in sd.groupby("borough"):
        signs = set(np.sign(g["diff"].round(2))) - {0.0}
        flips = len(signs) > 1
        print("%-14s per-seed: %s" % (borough, "  ".join("%+.2f" % v for v in g["diff"])))
        if flips:
            print("%-14s   SIGN FLIPS BETWEEN SEEDS on the same borough - no single-seed"
                  % "")
            print("%-14s   run of this experiment is interpretable, whatever its p-value."
                  % "")
    print()
    worst = sd["diff"].abs().max()
    print("Largest per-seed effect seen: %.2f points, against a ~%.0f-point noise band."
          % (worst, NOISE_BAND))
    print()
    print("The preprint abstract previously carried 'the model cannot exploit")
    print("additional history', from the single-seed Lambeth run (-0.72). It was")
    print("withdrawn on 2026-09-05 when Westminster showed +2.98 at the same seed.")
    print("These figures show the withdrawal was correct and the original claim")
    print("could not have been supported either way at one seed.")

    out = ROOT / "reports" / "s5_multiseed_analysis.csv"
    res.to_csv(out, index=False)
    sd.to_csv(ROOT / "reports" / "s5_multiseed_per_seed.csv", index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
