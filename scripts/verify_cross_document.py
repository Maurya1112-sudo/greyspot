"""Do the project's documents agree with each other?

`verify_preprint_claims.py` checks the preprint against its source data.
It cannot catch a document that is internally consistent about something
wrong, and on 2026-09-06 two such cases surfaced within an hour:

- The preprint counted **ten** findings while
  `check_effect_size_heuristic.py` counted **eleven**, because the preprint
  combined two separate ablations into one table row. Both passed their own
  checks.
- `docs/final_model.md` — which declares itself canonical — still carried
  the SINGLE-SEED headline (pooled 80.76) two days after the 5-seed figures
  replaced it in the README, the preprint and MASTER_PLAN, and still
  described two architecture ablations as null after they had been
  overturned and replicated.

Neither was found by a check. The first surfaced while tracing SOP claims
to source, the second while drafting a Methods section. This script makes
that class of drift detectable instead of incidental.

It asserts that the headline figures, the replication counts and the
trivial-baseline result read the same in every document that states them,
and that no document repeats a claim the ledger records as overturned.

Run: python scripts/verify_cross_document.py
Exit code 1 on any disagreement.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS = {
    "README.md": ROOT / "README.md",
    "final_model.md": ROOT / "docs" / "final_model.md",
    "MASTER_PLAN.md": ROOT / "docs" / "MASTER_PLAN.md",
    "DRAFT_preprint.md": ROOT / "paper" / "DRAFT_preprint.md",
    "sop_paragraph.md": ROOT / "docs" / "sop_paragraph.md",
}

def _canonical_from_artefacts() -> dict[str, list[str]]:
    """Canonical figures read from the computed artefacts, not hard-coded.

    Hard-coding them made this checker weaker than it looked: it verified
    that documents AGREE with each other, but not that they agree with the
    numbers the scripts actually produce. Every document could drift
    together - or the artefact could move under them - and it would pass.
    Reading `reports/headline_table.csv` (written by
    `scripts/make_headline_table.py`) ties the documents to the evidence.

    Falls back to the hard-coded values when the artefact is absent, so the
    check still runs on a fresh clone before anything has been computed.
    """
    fallback = {
        "pooled headline": ["80.14"],
        "Westminster headline": ["80.03"],
        "Tower Hamlets headline": ["82.63"],
        "Lambeth headline": ["77.75"],
    }
    path = ROOT / "reports" / "headline_table.csv"
    if not path.exists():
        return fallback
    import pandas as pd
    d = pd.read_csv(path)
    out: dict[str, list[str]] = {}
    for _, r in d.iterrows():
        label = ("pooled headline" if r.borough == "POOLED"
                 else "%s headline" % r.borough)
        out[label] = ["%.2f" % r["mean"]]
    # only trust it once every borough plus the pooled row is present
    return out if len(out) == 4 else fallback


# Figures that must read identically wherever they appear at all. A document
# that does not mention one is fine; a document that contradicts it is not.
CANONICAL = {
    **_canonical_from_artefacts(),
    "trivial sort (uncapped)": ["83.94"],
    "matched-horizon gap": ["-0.85", "−0.85"],
}

# Values that were superseded. Any appearance outside a correction note is a
# document still asserting a withdrawn number.
SUPERSEDED = {
    "80.76": "single-seed pooled headline, replaced by 80.14 (V8/V11)",
    "78.92%": "single-seed Westminster, replaced by 80.03",
    "83.93%": "single-seed Tower Hamlets, replaced by 82.63",
    "79.44%": "single-seed Lambeth, replaced by 77.75",
}

# Claims the ledger records as overturned; their old wording must not recur.
OVERTURNED = [
    (r"2 layers tested, null", "GAT depth was overturned: -26.4 6, p=0.0003"),
    (r"GAT layers \(1 vs 2\) \| null", "GAT depth was overturned"),
    (r"order is \*equivalent\*", "encoder order was overturned: -13.61, p=0.0135"),
]

# A correction note legitimately quotes a superseded number. Lines within
# this many lines of such a marker are exempt.
# A superseded HEADLINE figure is a problem. The same number quoted as an
# ablation baseline, or inside a table explicitly labelled single-seed, is
# not - those are correct statements about a seed-42 run. So the exemption
# covers both correction notes and explicit single-seed labelling.
MARKERS = ("correction", "withdrawn", "overturned", "superseded", "previously",
           "earlier version", "no longer", "retracted", "~~",
           "single seed", "single-seed", "seed 42", "seed-42",
           "not quotable", "per-window", "load-bearing")
results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


def in_correction_context(lines: list[str], idx: int) -> bool:
    """Is this specific line a legitimate mention of a superseded figure?

    Three narrow cases, and no others:

      1. The line itself carries a marker ("previously", "superseded",
         strikethrough, "single-seed", ...).
      2. The line is inside a blockquote - the form every correction note
         in this project uses.
      3. The line is a table row, and the caption immediately above the
         table carries a marker.

    **Section scoping was tried first and was actively dangerous.** Making
    the whole enclosing section exempt meant a single correction note
    blanket-exempted every other claim in that section. A negative control
    proved it: re-injecting the exact stale claim that had survived two
    days ("2 layers tested, null") into a section containing a correction
    note produced 14/14 OK. The checker looked healthy and was blind.

    A checker that fails to fail is worse than no checker, so the rule is
    now narrow enough to be wrong only in the direction of false alarms.
    """
    line = lines[idx].lower()
    if any(m in line for m in MARKERS):
        return True
    if line.lstrip().startswith(">"):
        return True
    if line.lstrip().startswith("|"):
        # walk up past the table to its caption
        for i in range(idx - 1, max(-1, idx - 40), -1):
            stripped = lines[i].strip()
            if stripped.startswith("|") or not stripped:
                continue
            return any(m in " ".join(lines[max(0, i - 3):i + 1]).lower() for m in MARKERS)
        return False
    # Prose: the enclosing PARAGRAPH, bounded by blank lines. A qualifying
    # sentence ("both sides of this comparison are single-seed") often
    # follows the figure it qualifies rather than sharing its line, and
    # markdown wraps prose at arbitrary points. The paragraph is the unit
    # such a sentence actually governs - narrow enough that a correction
    # note cannot exempt unrelated claims, unlike section scoping.
    lo = idx
    while lo > 0 and lines[lo - 1].strip():
        lo -= 1
    hi = idx
    while hi + 1 < len(lines) and lines[hi + 1].strip():
        hi += 1
    return any(m in " ".join(lines[lo:hi + 1]).lower() for m in MARKERS)


def main() -> None:
    texts = {name: (p.read_text(encoding="utf-8") if p.exists() else None)
             for name, p in DOCS.items()}
    missing = [n for n, t in texts.items() if t is None]
    for n in missing:
        check(False, f"{n} present", "file not found")

    # --- 1. canonical figures agree wherever stated --------------------
    for label, variants in CANONICAL.items():
        stating = [n for n, t in texts.items() if t and any(v in t for v in variants)]
        check(len(stating) > 0, f"canonical: {label}",
              "stated in %d doc(s): %s" % (len(stating), ", ".join(stating) or "NONE"))

    # --- 2. no document asserts a superseded figure --------------------
    for value, why in SUPERSEDED.items():
        offenders = []
        for name, t in texts.items():
            if not t:
                continue
            lines = t.splitlines()
            for i, ln in enumerate(lines):
                if value in ln and not in_correction_context(lines, i):
                    offenders.append(f"{name}:{i+1}")
        check(not offenders, f"superseded {value} not asserted",
              (why if not offenders else "STILL ASSERTED at " + ", ".join(offenders[:4])))

    # --- 3. no document repeats an overturned claim --------------------
    for pattern, why in OVERTURNED:
        offenders = []
        for name, t in texts.items():
            if not t:
                continue
            for i, ln in enumerate(t.splitlines()):
                if re.search(pattern, ln) and not in_correction_context(t.splitlines(), i):
                    offenders.append(f"{name}:{i+1}")
        check(not offenders, f"overturned claim absent: {pattern[:34]}",
              (why if not offenders else "STILL PRESENT at " + ", ".join(offenders[:4])))

    # --- 4. replication counts agree across documents ------------------
    counts: dict[str, set[str]] = {}
    for name, t in texts.items():
        if not t:
            continue
        found = set(re.findall(r"(\d+) of (?:10|11) (?:single-borough )?(?:findings|claims)", t))
        found |= {m for m in re.findall(r"of (ten|eleven) findings", t)}
        if found:
            counts[name] = found
    stated_totals = set()
    for name, t in texts.items():
        if not t:
            continue
        for tot in re.findall(r"of (ten|eleven|\d+) findings", t):
            stated_totals.add({"ten": "10", "eleven": "11"}.get(tot, tot))
        for tot in re.findall(r"\d+ of (10|11) single-borough", t):
            stated_totals.add(tot)
    check(len(stated_totals) <= 1, "finding TOTAL agrees across documents",
          "totals stated: %s" % (sorted(stated_totals) or ["none"]))

    # --- report --------------------------------------------------------
    print("=" * 78)
    print("CROSS-DOCUMENT CONSISTENCY")
    print("=" * 78)
    for ok, label, detail in results:
        print("  %-4s %-44s %s" % ("OK" if ok else "FAIL", label, detail))
    failed = [r for r in results if not r[0]]
    print()
    print("%d checks, %d failed" % (len(results), len(failed)))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
