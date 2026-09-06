"""Check every quantitative claim in the preprint against its source file.

Rule R12 says any number quoted in a doc needs a script in the repo that
regenerates it. This closes the loop: it re-derives each headline number
from raw sources and compares it to what the preprint actually says, so
drift between the docs and the evidence is caught mechanically rather than
by re-reading.

Two real errors had already been found by hand on 2026-09-05 - a
title/abstract/table contradiction over how many findings replicated, and
an abstract claiming effect size predicted replication when it predicts
only NON-replication. Both were the kind of thing a check like this catches
in a second.

Run: python scripts/verify_preprint_claims.py
Exit code 1 if any claim fails, so it can gate a commit.
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

PREPRINT = ROOT / "paper" / "DRAFT_preprint.md"
V8_LOGS = {
    "Lambeth": "v8_multiseed",
    "Westminster": "v8_westminster",
    "Tower Hamlets": "v8_th",
}
_SEED_MEAN = re.compile(r"seed\s+(\d+)\s+6-window mean AccHR@20 = ([0-9.]+)")

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"]

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


def seed_stats(borough: str):
    """Mean, sd and 95% CI over the 5 seeds. The CI uses the t
    distribution, not the normal approximation: with n=5 the difference is
    material (t(4)=2.776 vs z=1.96, ~40% wider), and using z here would
    understate the interval by roughly half a point."""
    t = (ROOT / "reports" / "run_logs" / f"{V8_LOGS[borough]}.log").read_text(
        encoding="utf-8", errors="replace")
    v = np.array([float(m.group(2)) for m in _SEED_MEAN.finditer(t)])
    if len(v) == 0:
        return None
    sd = v.std(ddof=1)
    half = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return 100 * v.mean(), 100 * sd, 100 * (v.mean() - half), 100 * (v.mean() + half), len(v)


def main() -> None:
    text = PREPRINT.read_text(encoding="utf-8")
    # Markdown wraps prose, so "4 of 6" can straddle a newline. Search a
    # whitespace-collapsed copy for phrases; keep `text` for line-based work.
    flat = re.sub(r"\s+", " ", text)

    # --- 1. headline per-borough figures -------------------------------
    for borough in V8_LOGS:
        st = seed_stats(borough)
        if st is None:
            check(False, f"headline {borough}", "no seed data")
            continue
        mean, sd, lo, hi, n = st
        row = re.search(rf"\|\s*{re.escape(borough)}\s*\|(.+?)\|", text)
        if not row:
            check(False, f"headline {borough}", "row not found in preprint")
            continue
        nums = [float(x) for x in re.findall(r"\d+\.\d+", row.group(0))]
        stated_mean, stated_sd = nums[0], nums[1]
        ok = abs(stated_mean - mean) < 0.05 and abs(stated_sd - sd) < 0.05
        check(ok, f"headline {borough}",
              "preprint %.2f±%.2f vs computed %.2f±%.2f (n=%d)" % (stated_mean, stated_sd, mean, sd, n))
        # and the CI on the same line
        ci = re.search(rf"\|\s*{re.escape(borough)}\s*\|[^|]+\|\s*\[([\d.]+),\s*([\d.]+)\]", text)
        if ci:
            slo, shi = float(ci.group(1)), float(ci.group(2))
            ok = abs(slo - lo) < 0.05 and abs(shi - hi) < 0.05
            check(ok, f"  CI {borough}",
                  "preprint [%.2f, %.2f] vs t-dist [%.2f, %.2f]" % (slo, shi, lo, hi))

    # --- 2. pooled figure ---------------------------------------------
    allv = []
    for borough in V8_LOGS:
        t = (ROOT / "reports" / "run_logs" / f"{V8_LOGS[borough]}.log").read_text(
            encoding="utf-8", errors="replace")
        allv += [float(m.group(2)) for m in _SEED_MEAN.finditer(t)]
    pooled = 100 * float(np.mean(allv))
    m = re.search(r"\*\*Pooled\*\*\s*\|\s*\*\*([\d.]+)%", text)
    if m:
        check(abs(float(m.group(1)) - pooled) < 0.05, "pooled headline",
              "preprint %.2f vs computed %.2f (n=%d seed-runs)" % (float(m.group(1)), pooled, len(allv)))

    # --- 3. the trivial-baseline table --------------------------------
    p = ROOT / "reports" / "s2_paired_comparison.csv"
    if p.exists():
        s2 = pd.read_csv(p)
        for label, needle in [("EB (HSM method)", "Empirical Bayes (HSM)"),
                              ("raw count (no shrinkage)", "Cumulative crash count"),
                              ("raw count CAPPED at 1825d (matched to GNN)", "Count capped")]:
            row = s2[(s2.baseline == label) & (s2.borough == "POOLED")]
            if row.empty:
                continue
            diff, t_p = float(row["diff"].iloc[0]), float(row.t_p.iloc[0])
            # A label such as "Empirical Bayes (HSM)" appears in BOTH the
            # ranker table and the paired-test table. Match the paired one
            # by requiring a p-value-shaped 4-decimal number on the line,
            # rather than taking the first hit.
            line = next((ln for ln in text.splitlines()
                         if needle in ln and "|" in ln and re.search(r"0\.\d{4}", ln)), None)
            if not line:
                check(False, f"baseline row '{needle}'", "not found in preprint")
                continue
            nums = re.findall(r"−?-?[\d.]+", line.replace("−", "-"))
            floats = [float(x) for x in nums if x not in ("-", ".")]
            ok = any(abs(f - diff) < 0.02 for f in floats) and any(abs(f - t_p) < 0.001 for f in floats)
            check(ok, f"baseline '{needle}'", "expect diff %.2f, p %.4f" % (diff, t_p))

    # --- 4. effect-size heuristic table -------------------------------
    p = ROOT / "reports" / "effect_size_heuristic.csv"
    if p.exists():
        e = pd.read_csv(p)
        res = e[e.replicated.notna() & (~e.interaction)]
        below = res[~res.above_band]
        above = res[res.above_band]
        b_rep, b_n = int(below.replicated.sum()), len(below)
        a_rep, a_n = int(above.replicated.sum()), len(above)
        check(f"{a_rep} of {a_n}" in flat, "heuristic 'above band' figure",
              "computed %d of %d replicated above the band" % (a_rep, a_n))
        check(b_rep == 0, "heuristic 'below band' figure",
              "computed %d of %d replicated below the band" % (b_rep, b_n))

    # --- 5. internal consistency of the replication counts ------------
    # Locate the replication table by its HEADING TEXT, not its number.
    # Section numbers shift whenever a section is inserted - adding Methods
    # renumbered five headings at once and broke a hard-coded "## 2." split.
    m_sec = re.search(r"^##\s*\d+\.\s*The main result.*?$", text, re.M)
    if not m_sec:
        check(False, "replication table located", "no 'The main result' heading")
        sec = ""
    else:
        rest = text[m_sec.end():]
        nxt = re.search(r"^##\s", rest, re.M)
        sec = rest[:nxt.start()] if nxt else rest
    rows = [ln for ln in sec.splitlines()
            if ln.startswith("|") and "---" not in ln and "Finding" not in ln
            and "Borough-1 effect" not in ln and "Below the" not in ln and "Above it" not in ln]
    rep = sum(1 for ln in rows if "replicated" in ln and "downgraded" not in ln)
    fail = sum(1 for ln in rows if "failed" in ln)
    unres = sum(1 for ln in rows if "downgraded" in ln)
    total = len(rows)
    check((total < len(WORDS) and f"of {WORDS[total]} findings" in text)
          or f"of {total} findings" in text,
          "abstract count matches table", "table has %d rows" % total)
    # Do not hard-code the composition - it changes as replications land.
    # The invariant that matters is that the ABSTRACT's stated counts match
    # what the table actually contains.
    words = {w: i for i, w in enumerate(WORDS)}
    m_rep = re.search(r"only (\w+) survived replication", flat)
    m_fail = re.search(r"\((\w+) failed", flat)
    stated_rep = words.get(m_rep.group(1)) if m_rep else None
    stated_fail = words.get(m_fail.group(1)) if m_fail else None
    check(stated_rep == rep, "abstract 'survived' count",
          "abstract says %s, table has %d" % (stated_rep, rep))
    check(stated_fail == fail, "abstract 'failed' count",
          "abstract says %s, table has %d" % (stated_fail, fail))
    check(rep + fail + unres == total, "counts sum to table size",
          "%d + %d + %d vs %d rows" % (rep, fail, unres, total))

    # --- 6. internal cross-references resolve --------------------------
    # Section numbers shifted twice on 2026-09-06 (inserting Methods, then
    # Related work). Each insertion renumbered every later heading, and a
    # stale reference points confidently at the wrong section without any
    # visible breakage - the reader simply follows it to the wrong place.
    # The trailing dot after a heading number is optional ("### 5.1 The
    # baseline's ..."), so it must not be required when parsing.
    heads = {m.group(1) for m in re.finditer(r"^#{2,3}\s*(\d+(?:\.\d+)?)\.?\s+\S", text, re.M)}
    refs = set(re.findall(r"§(\d+(?:\.\d+)?)", text))
    dangling = sorted(refs - heads)
    check(not dangling, "cross-references resolve",
          "%d reference(s), %s" % (len(refs),
                                   "all resolve" if not dangling
                                   else "DANGLING: " + ", ".join("§" + d for d in dangling)))

    # --- report --------------------------------------------------------
    print("=" * 76)
    print("PREPRINT CLAIM VERIFICATION")
    print("=" * 76)
    for ok, label, detail in results:
        print("  %-4s %-32s %s" % ("OK" if ok else "FAIL", label, detail))
    failed = [r for r in results if not r[0]]
    print()
    print("%d checks, %d failed" % (len(results), len(failed)))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
