"""Does effect size actually predict replication?

The preprint's abstract claims: "effect size predicted which - every effect
above the measured ~4-point seed-noise band replicated, every effect below
it failed", with interactions named as the sole exception.

That is a headline methodological claim, so it needs checking against the
table rather than being asserted from memory. This script encodes every
replication attempt with its borough-1 effect size and its outcome, and
tests the rule mechanically.

Effect sizes are the borough-1 figures as recorded in
`docs/MASTER_PLAN.md` and `docs/decision_log.md`; where a finding has two
(e.g. road class was measured at two feature-set sizes) the LARGER
magnitude is used, which is the most favourable reading for the heuristic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

NOISE_BAND = 4.0  # points, from V8/V11: measured seed spread 3.2-4.7

# finding, |effect| on borough 1, replicated?, is it an interaction?
FINDINGS = [
    ("Architecture topology (layers)",      26.46, True,  False),
    ("Architecture topology (encoder)",     13.61, True,  False),
    ("Long-horizon crash history",          15.85, True,  False),
    ("Our architecture vs theirs",          18.53, True,  False),
    ("Architecture x history interaction",  25.04, False, True),
    ("Road class harmful",                   6.87, False, False),
    ("Rank-transform scaling",                5.80, False, False),
    ("weight_decay=0.01",                     2.53, False, False),
    ("Architecture ensembling",               1.79, False, False),
    ("hidden=42/42",                          1.46, False, False),
    ("13 features == 35 features",            0.99, False, False),  # settled 2026-09-05: sign flips
]


def main() -> None:
    df = pd.DataFrame(FINDINGS, columns=["finding", "effect", "replicated", "interaction"])
    df["above_band"] = df.effect > NOISE_BAND
    resolved = df[df.replicated.notna()].copy()

    print("=" * 74)
    print("Does effect size predict replication?  (band = %.0f points)" % NOISE_BAND)
    print("=" * 74)
    print("%-38s %8s %10s %9s" % ("finding", "|effect|", "above band", "replicated"))
    for _, r in df.iterrows():
        rep = "unresolved" if pd.isna(r.replicated) else ("yes" if r.replicated else "NO")
        print("%-38s %8.2f %10s %9s%s" % (r.finding, r.effect, "yes" if r.above_band else "no",
                                          rep, "   (interaction)" if r.interaction else ""))

    # The rule as stated, with interactions excluded as the paper allows.
    main_effects = resolved[~resolved.interaction]
    predicted = main_effects.above_band
    actual = main_effects.replicated.astype(bool)
    wrong = main_effects[predicted != actual]

    print()
    print("-" * 74)
    print("TESTING THE RULE ON MAIN EFFECTS ONLY (interactions excluded, as claimed)")
    print("-" * 74)
    print("agreement: %d of %d" % (int((predicted == actual).sum()), len(main_effects)))
    if len(wrong):
        print()
        print("COUNTEREXAMPLES - the rule is WRONG for these:")
        for _, r in wrong.iterrows():
            print("  %-36s |effect|=%.2f is ABOVE the band but did NOT replicate"
                  % (r.finding, r.effect))
        print()
        print("=> The abstract's claim that EVERY effect above the band replicated is FALSE.")
        print("   The true pattern is one-directional: every effect BELOW the band failed")
        print("   (%d of %d), but being above it does not guarantee replication (%d of %d"
              % (int((~main_effects.above_band & ~actual).sum()),
                 int((~main_effects.above_band).sum()),
                 int((main_effects.above_band & actual).sum()),
                 int(main_effects.above_band.sum())))
        print("   above-band effects replicated).")
    else:
        print("no counterexamples - the rule holds as stated")

    below = main_effects[~main_effects.above_band]
    print()
    print("-" * 74)
    print("WHAT DOES HOLD")
    print("-" * 74)
    print("Below the band: %d of %d failed (%.0f%%)"
          % (int((~below.replicated.astype(bool)).sum()), len(below),
             100 * float((~below.replicated.astype(bool)).mean())))
    above = main_effects[main_effects.above_band]
    print("Above the band: %d of %d replicated (%.0f%%)"
          % (int(above.replicated.astype(bool).sum()), len(above),
             100 * float(above.replicated.astype(bool).mean())))
    print()
    print("So a small effect is reliable evidence of NON-replication, while a large")
    print("effect is only weak evidence FOR it. That is a necessary-not-sufficient")
    print("relationship, not the biconditional the abstract states.")

    out = ROOT / "reports" / "effect_size_heuristic.csv"
    df.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
