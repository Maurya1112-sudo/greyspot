"""S2b - the GNN vs trivial-baseline comparison, done properly.

**What was wrong with the original comparison.** MASTER_PLAN reported the
GNN as +0.85 ahead of a matched-horizon crash-count sort. That number had
two defects:

1. It used the SINGLE-SEED GNN score (0.7944 on Lambeth). V8/V11 later
   established that seed 42 is +1.69 points optimistic on Lambeth and that
   the seed spread is ~4 points - larger than the effect being measured.
2. It compared MEANS. With the same six held-out windows on both sides a
   paired test is available and is strictly more powerful; comparing means
   throws that away and cannot produce a p-value at all.

Rule R10 requires any published number to be multi-seeded per borough, and
this is one of the paper's central claims - arguably its most
uncomfortable one - so it has to meet that bar.

**What this script does.** Pairs, window by window, the seed-averaged GNN
score against each baseline, then runs both a paired t-test and a Wilcoxon
signed-rank test per borough and pooled. The baselines are deterministic
(no training, no seed), so all sampling variation sits on the GNN side.

**Inputs.** GNN per-seed per-window scores are parsed from the V8 run logs,
which record them to **3 decimal places** (the mean on the same line is
4dp, which is easy to misread as the per-window precision - an earlier
version of this docstring did). Rounding is therefore up to +/-0.0005 per
window, which propagates to at most +/-0.0005 on a 6-window mean: about
1.3% of the ~0.037 effects measured here, and immaterial to every
conclusion drawn. The seed-42 row is cross-checked against the
full-precision per-window CSV at exactly that 0.0005 tolerance.

**This input is superseded.** `scripts/run_headline_multiseed.py`
regenerates the same quantity at full precision into a committed CSV.
Once that run completes this script should read it instead of parsing
logs, removing the rounding entirely. Baseline per-window scores come from
`reports/<borough>/s2_empirical_bayes_per_window.csv`.

Run: python scripts/run_s2_paired_comparison.py
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

EVAL_STARTS = pd.to_datetime(
    ["2023-07-15", "2023-10-13", "2024-01-11", "2024-04-10", "2024-07-09", "2024-10-07"]
)
# The V8 run logs are COMMITTED under reports/run_logs/ rather than read
# from /tmp. Two reasons: a temp directory does not survive a reboot (this
# project has already lost one run that way), and Python on Windows does
# not resolve Git Bash's /tmp at all, so the original paths silently
# resolved to a non-existent path and every borough was skipped.
V8_LOGS = {
    "Lambeth": ROOT / "reports" / "run_logs" / "v8_multiseed.log",
    "Westminster": ROOT / "reports" / "run_logs" / "v8_westminster.log",
    "Tower Hamlets": ROOT / "reports" / "run_logs" / "v8_th.log",
}
DISCREPANT: list = []
SLUGS = {"Lambeth": "lambeth", "Westminster": "westminster", "Tower Hamlets": "tower_hamlets"}
_SEED_RE = re.compile(r"seed\s+(\d+)\s+6-window mean AccHR@20 = ([0-9.]+)\s+\(windows: ([0-9. ]+)\)")


def parse_v8_log(path: Path, borough: str) -> pd.DataFrame:
    """Per-seed, per-window GNN scores from a V8 multi-seed run log."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _SEED_RE.search(line)
        if not m:
            continue
        seed = int(m.group(1))
        reported_mean = float(m.group(2))
        windows = [float(v) for v in m.group(3).split()]
        if len(windows) != len(EVAL_STARTS):
            raise ValueError(f"{path}: seed {seed} has {len(windows)} windows, expected {len(EVAL_STARTS)}")
        # The logged mean is computed from full-precision values while the
        # per-window figures are rounded, so they agree only to rounding.
        # A larger disagreement means the line is not what it appears.
        if abs(np.mean(windows) - reported_mean) > 0.001:
            raise ValueError(
                f"{path}: seed {seed} windows average {np.mean(windows):.4f} but the line reports "
                f"{reported_mean:.4f} - these are not the same quantity."
            )
        rows += [{"borough": borough, "seed": seed, "held_out_start": s, "AccHR": v}
                 for s, v in zip(EVAL_STARTS, windows)]
    if not rows:
        raise ValueError(f"{path}: no per-seed lines found.")
    return pd.DataFrame(rows)


def crosscheck_seed42(gnn: pd.DataFrame, borough: str) -> None:
    """Confirm the log describes the same run as the committed CSV."""
    csv = ROOT / "reports" / SLUGS[borough] / "ucl_multiwindow_per_window_multiyear.csv"
    if not csv.exists():
        print("    (no per-window CSV for %s - cross-check skipped)" % borough)
        return
    ref = pd.read_csv(csv)
    ref["held_out_start"] = pd.to_datetime(ref["held_out_start"])
    got = gnn[gnn.seed == 42].set_index("held_out_start")["AccHR"]
    exp = ref.set_index("held_out_start")["AccHR"].reindex(got.index)
    if exp.isna().any():
        print("    (CSV windows do not cover the log's - cross-check skipped)")
        return
    diffs = (got - exp).abs()
    worst = float(diffs.max())
    n_off = int((diffs > 0.0005).sum())
    status = "OK" if n_off == 0 else "%d/%d WINDOW(S) DIFFER" % (n_off, len(diffs))
    print("    seed-42 cross-check vs committed CSV: max |diff| = %.5f  [%s]" % (worst, status))
    if n_off:
        # NOT fatal, but never silent.
        #
        # MECHANISM UNRESOLVED - corrected 2026-09-06. This comment
        # previously asserted that AccHR@20's step-function behaviour over
        # tied segments makes same-seed reruns non-identical. That was a
        # plausible story, not a measurement, and direct evidence now
        # contradicts it: re-running the S5 config at seed 42 through a
        # different script reproduced the original per-window score exactly
        # (0.8106 vs 0.8106, Lambeth 2023-07-15).
        #
        # The likelier explanation is provenance. The per-window CSVs were
        # committed when version control was initialised (775fe87), so they
        # capture whatever was on disk at that moment - which may predate
        # the code the V8 logs were produced with. A single differing
        # window is consistent with an older artefact, not with
        # non-determinism, which would perturb every window.
        #
        # Either way the discrepancy is reported and the sensitivity check
        # below confirms no conclusion rests on it.
        for d, v_log, v_csv in zip(diffs.index, got, exp):
            if abs(v_log - v_csv) > 0.0005:
                print("        %s: log %.4f vs CSV %.4f (%+.4f)"
                      % (d.date(), v_log, v_csv, v_log - v_csv))
        DISCREPANT.append((borough, [d for d, x in diffs.items() if x > 0.0005]))


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    """Mean difference (a-b) in points, t-test p, Wilcoxon p, wins for a."""
    d = a - b
    t_p = float(stats.ttest_rel(a, b).pvalue) if len(d) > 1 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:  # all-zero differences
        w_p = float("nan")
    return 100 * float(d.mean()), t_p, w_p, int((d > 0).sum())


def main() -> None:
    print("=" * 78)
    print("S2b: SEED-AVERAGED, PAIRED comparison - GNN vs trivial baselines")
    print("=" * 78)

    gnn_all, base_all = [], []
    for borough, log in V8_LOGS.items():
        if not log.exists():
            print("\n%s: V8 log missing (%s) - SKIPPED" % (borough, log))
            continue
        print("\n%s" % borough)
        g = parse_v8_log(log, borough)
        print("    seeds: %s" % sorted(g.seed.unique()))
        crosscheck_seed42(g, borough)
        b_path = ROOT / "reports" / SLUGS[borough] / "s2_empirical_bayes_per_window.csv"
        if not b_path.exists():
            print("    baseline per-window CSV not written yet - SKIPPED")
            continue
        b = pd.read_csv(b_path)
        b["held_out_start"] = pd.to_datetime(b["held_out_start"])
        gnn_all.append(g)
        base_all.append(b)

    if not gnn_all:
        print("\nNothing to compare yet.")
        return

    gnn = pd.concat(gnn_all, ignore_index=True)
    base = pd.concat(base_all, ignore_index=True)
    # Seed-average per (borough, window) - the GNN score for that window.
    gnn_mean = gnn.groupby(["borough", "held_out_start"], as_index=False)["AccHR"].mean()
    gnn_sd = gnn.groupby(["borough", "held_out_start"], as_index=False)["AccHR"].std(ddof=1)

    rows = []
    for name in base.baseline.unique():
        bb = base[base.baseline == name]
        m = gnn_mean.merge(bb, on=["borough", "held_out_start"], suffixes=("_gnn", "_base"))
        assert len(m) == len(bb), f"{name}: paired on {len(m)} of {len(bb)} windows"
        for borough, grp in m.groupby("borough"):
            diff, t_p, w_p, wins = paired(grp.AccHR_gnn.to_numpy(), grp.AccHR_base.to_numpy())
            rows.append({"baseline": name, "borough": borough, "n": len(grp),
                         "GNN": 100 * grp.AccHR_gnn.mean(), "base": 100 * grp.AccHR_base.mean(),
                         "diff": diff, "wins": wins, "t_p": t_p, "w_p": w_p})
        diff, t_p, w_p, wins = paired(m.AccHR_gnn.to_numpy(), m.AccHR_base.to_numpy())
        rows.append({"baseline": name, "borough": "POOLED", "n": len(m),
                     "GNN": 100 * m.AccHR_gnn.mean(), "base": 100 * m.AccHR_base.mean(),
                     "diff": diff, "wins": wins, "t_p": t_p, "w_p": w_p})

    res = pd.DataFrame(rows)
    for name, grp in res.groupby("baseline", sort=False):
        print("\n" + "-" * 78)
        print("GNN (5-seed mean)  vs  %s" % name)
        print("-" * 78)
        print("%-16s %3s %8s %8s %8s %7s %10s %10s" %
              ("borough", "n", "GNN", "base", "diff", "wins", "t-test p", "Wilcoxon p"))
        for _, r in grp.iterrows():
            mark = "  <<<" if r.borough == "POOLED" else ""
            print("%-16s %3d %8.2f %8.2f %+8.2f %4d/%-2d %10.4f %10.4f%s" %
                  (r.borough, r.n, r.GNN, r["base"], r["diff"], r.wins, r.n, r.t_p, r.w_p, mark))

    # SENSITIVITY: drop any window whose two same-seed runs disagreed, and
    # confirm the pooled conclusion does not rest on it.
    if DISCREPANT:
        print()
        print("-" * 78)
        print("SENSITIVITY: pooled result excluding the window(s) that differed between")
        print("two same-seed runs (see cross-check above)")
        print("-" * 78)
        drop = {(b, pd.Timestamp(d)) for b, ds in DISCREPANT for d in ds}
        for name in base.baseline.unique():
            bb = base[base.baseline == name]
            m = gnn_mean.merge(bb, on=["borough", "held_out_start"], suffixes=("_gnn", "_base"))
            keep = ~m.apply(lambda r: (r.borough, pd.Timestamp(r.held_out_start)) in drop, axis=1)
            full = paired(m.AccHR_gnn.to_numpy(), m.AccHR_base.to_numpy())
            sub = paired(m[keep].AccHR_gnn.to_numpy(), m[keep].AccHR_base.to_numpy())
            print("%-44s all n=%d: %+.2f (p=%.4f) | dropped n=%d: %+.2f (p=%.4f)"
                  % (name[:44], len(m), full[0], full[1], int(keep.sum()), sub[0], sub[1]))

    out = ROOT / "reports" / "s2_paired_comparison.csv"
    res.to_csv(out, index=False)
    print("\nWritten to %s" % out)
    print("\nPer-window GNN seed spread (std across 5 seeds), for scale:")
    print("    median %.4f  max %.4f" % (gnn_sd.AccHR.median(), gnn_sd.AccHR.max()))


if __name__ == "__main__":
    main()
