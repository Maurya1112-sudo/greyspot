"""The paper's headline table, computed from the regenerated per-window data.

Until now the headline (per-borough means, SDs, CIs and the pooled figure)
was computed ad hoc from run logs. That satisfied R12 only loosely: a
script existed that *produced* the runs, but nothing regenerated the
TABLE, so its arithmetic — which confidence interval, which standard
deviation convention, which significance test — lived only in whatever
command last happened to compute it.

This script owns that arithmetic, reading
`reports/<borough>/headline_multiseed_per_window.csv`.

Two conventions are load-bearing and easy to get wrong; both are fixed
here rather than chosen per invocation:

- **Sample SD (ddof=1)**, not population. At n=5 the two differ by
  sqrt(5/4) = 1.12. `docs/final_model.md` once mixed both inside a single
  table row, making three boroughs non-comparable.
- **t distribution for the CI**, not the normal approximation. With n=5,
  t(4)=2.776 against z=1.96 — the interval is ~40% wider, and using z
  understates it by roughly half a point.

The comparison against the reference paper is a **one-sample** t-test of
our 5 seed means against their published point estimate. It cannot be
paired: they publish three point estimates, not per-window results. That
is a real limitation of the comparison and is stated in the paper.

Run: python scripts/make_headline_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BOROUGHS = {"Westminster": "westminster", "Tower Hamlets": "tower_hamlets", "Lambeth": "lambeth"}
GAO = {"Westminster": 68.98, "Tower Hamlets": 72.24, "Lambeth": 76.59}
GAO_POOLED = 72.60


def main() -> None:
    rows, all_seed_means, missing = [], [], []
    for borough, sl in BOROUGHS.items():
        p = ROOT / "reports" / sl / "headline_multiseed_per_window.csv"
        if not p.exists():
            missing.append(borough)
            continue
        d = pd.read_csv(p)
        complete = d.groupby("seed").filter(lambda g: len(g) == 6)
        if complete.seed.nunique() < 5:
            missing.append("%s (%d/5 seeds)" % (borough, complete.seed.nunique()))
            continue
        seed_means = complete.groupby("seed").AccHR.mean().to_numpy() * 100
        sd = seed_means.std(ddof=1)
        half = stats.t.ppf(0.975, len(seed_means) - 1) * sd / np.sqrt(len(seed_means))
        p_val = float(stats.ttest_1samp(seed_means, GAO[borough]).pvalue)
        rows.append({"borough": borough, "n_seeds": len(seed_means),
                     "mean": seed_means.mean(), "sd": sd,
                     "ci_lo": seed_means.mean() - half, "ci_hi": seed_means.mean() + half,
                     "gao": GAO[borough], "p": p_val,
                     "verdict": "better" if (p_val < 0.05 and seed_means.mean() > GAO[borough])
                                else ("worse" if p_val < 0.05 else "tie")})
        all_seed_means.append(seed_means)

    if missing:
        print("NOT YET REGENERATED: %s" % ", ".join(missing))
        print("(figures below cover only the boroughs that are complete)")
        print()
    if not rows:
        print("No regenerated data yet.")
        return

    if len(rows) == len(BOROUGHS):
        pooled = np.concatenate(all_seed_means)
        sd = pooled.std(ddof=1)
        half = stats.t.ppf(0.975, len(pooled) - 1) * sd / np.sqrt(len(pooled))
        p_val = float(stats.ttest_1samp(pooled, GAO_POOLED).pvalue)
        rows.append({"borough": "POOLED", "n_seeds": len(pooled),
                     "mean": pooled.mean(), "sd": sd,
                     "ci_lo": pooled.mean() - half, "ci_hi": pooled.mean() + half,
                     "gao": GAO_POOLED, "p": p_val,
                     "verdict": "better" if (p_val < 0.05 and pooled.mean() > GAO_POOLED) else "tie"})

    res = pd.DataFrame(rows)
    print("=" * 84)
    print("HEADLINE TABLE (sample SD, t-distribution CI, one-sample test vs Gao et al.)")
    print("=" * 84)
    print("%-15s %5s %8s %7s %18s %8s %10s %8s" %
          ("borough", "n", "mean", "sd", "95% CI", "Gao", "p", "verdict"))
    for _, r in res.iterrows():
        print("%-15s %5d %8.2f %7.2f  [%6.2f, %6.2f] %8.2f %10.4f %8s" %
              (r.borough, r.n_seeds, r["mean"], r.sd, r.ci_lo, r.ci_hi, r.gao, r.p, r.verdict))

    print()
    print("Markdown for the paper:")
    print()
    print("| Borough | This work | 95% CI | Gao et al. | |")
    print("|---|---|---|---|---|")
    for _, r in res.iterrows():
        bold = "**" if r.verdict == "better" else ""
        name = "**Pooled**" if r.borough == "POOLED" else r.borough
        # "p=0.0000" is not a p-value a paper should print - it reads as zero
        # when it means "smaller than the precision shown".
        pstr = "p<0.0001" if r.p < 0.0001 else "p=%.4f" % r.p
        note = pstr if r.verdict != "tie" else "tie, " + pstr
        print("| %s | %s%.2f%% ± %.2f%s | [%.2f, %.2f] | %.2f%% | %s |"
              % (name, bold, r["mean"], r.sd, bold, r.ci_lo, r.ci_hi, r.gao, note))

    out = ROOT / "reports" / "headline_table.csv"
    res.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
