"""Paired significance testing between two per-window AccHR@20 result sets.

Reusable helper matching the paired t-test / Wilcoxon signed-rank standard
this project has applied to every architecture/feature claim this session
(heads=3-vs-heads=1 at p=0.036; POI+socio-demographic at p=0.0094/0.0120).
Takes two `ucl_multiwindow_per_window*.csv` files (as produced by the
`run_ucl_comparison_multiwindow_*.py` scripts), matches rows by
`held_out_start` (so window order/count differences can't silently
misalign the pairing), and reports both tests plus the win count.

Usage:
    python scripts/paired_significance.py <baseline.csv> <candidate.csv> \
        [--baseline-label "..."] [--candidate-label "..."]
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_csv")
    parser.add_argument("candidate_csv")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_csv)
    candidate = pd.read_csv(args.candidate_csv)

    merged = baseline[["held_out_start", "AccHR"]].merge(
        candidate[["held_out_start", "AccHR"]],
        on="held_out_start", suffixes=("_baseline", "_candidate"),
    )
    if len(merged) < 2:
        raise ValueError(
            f"Only {len(merged)} matching windows (by held_out_start) between "
            f"{args.baseline_csv} and {args.candidate_csv} - cannot run a paired test."
        )
    if len(merged) < min(len(baseline), len(candidate)):
        print(
            f"WARNING: {min(len(baseline), len(candidate)) - len(merged)} window(s) "
            "did not match by held_out_start and were dropped from the pairing."
        )

    diffs = merged["AccHR_candidate"] - merged["AccHR_baseline"]
    t_stat, t_p = stats.ttest_rel(merged["AccHR_candidate"], merged["AccHR_baseline"])
    try:
        w_stat, w_p = stats.wilcoxon(merged["AccHR_candidate"], merged["AccHR_baseline"])
    except ValueError as exc:  # e.g. all diffs identical/zero
        w_stat, w_p = float("nan"), float("nan")
        print(f"(Wilcoxon not computable: {exc})")

    n_wins = int((diffs > 0).sum())
    n = len(merged)

    print(f"n windows paired: {n}")
    print(f"{args.baseline_label} mean AccHR@20: {merged['AccHR_baseline'].mean():.4f}")
    print(f"{args.candidate_label} mean AccHR@20: {merged['AccHR_candidate'].mean():.4f}")
    print(f"mean diff (candidate - baseline): {diffs.mean():+.4f}")
    print(f"candidate wins: {n_wins}/{n}")
    print(f"paired t-test: t={t_stat:.4f}, p={t_p:.4f}")
    print(f"Wilcoxon signed-rank: W={w_stat:.4f}, p={w_p:.4f}")
    print()
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
