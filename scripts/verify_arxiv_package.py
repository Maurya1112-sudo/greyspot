"""Check the arXiv submission package against arXiv's stated rules.

**What this can and cannot do.** No LaTeX toolchain is installed on this
machine, so the source cannot be compiled here. Everything below is a
static check. **A clean run does NOT mean the paper compiles** — it means
the failure modes that can be detected without compiling are absent. The
submission must still be compiled before upload, and arXiv's own
"submission preview" is the authoritative test.

Rules encoded here, from arXiv's format-requirements and TeX-submission
pages (retrieved 2026-09-06):

- abstract under 1,920 characters, and not containing the word "Abstract"
- figures only in JPEG, PNG or PDF (SVG is *not* accepted) — we sidestep
  this by emitting the figure as pgfplots source
- no `\\pdfoutput` (arXiv forbids forcing the output format). Re-checked
  2026-09-26 against the primary source after a secondary summary
  recommended ADDING `\\pdfoutput=1`: the current page
  (info.arxiv.org/help/submit_tex.html) says "You should not use
  `\\pdfoutput` to change the output format" and that the processor is
  chosen in the web form (select PDFLaTeX). The `\\pdfoutput=1`-in-first-
  5-lines advice comes from the *legacy* submission page
  (submit_legacy_differences.html) and is superseded. Do not re-add it.
- no `psfig` (unsupported)
- no custom `.sty` beyond TeX Live
- no line numbers, watermarks or margin notes
- 10--14pt type, 1in margins, single spaced
- title and authorship present (no anonymous submissions)
- if a `.bbl` is shipped it must match the main `.tex` basename
- no hidden files (deleted on announcement)

Run: python scripts/verify_arxiv_package.py
Exit code 1 on any failure.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "paper" / "arxiv"
MAIN = PKG / "main.tex"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _braced_after(text: str, macro: str) -> str | None:
    """Contents of the {...} group following `macro`, balancing nested braces."""
    i = text.find(macro)
    if i < 0:
        return None
    i += len(macro)
    while i < len(text) and text[i] != "{":
        i += 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def form_field_mismatches(tex: str, submission_md: str) -> list[str]:
    """Compare the arXiv web-form fields recorded in ARXIV_SUBMISSION.md with
    what main.tex actually says. `tex` must already have comments stripped.

    The form differs from the source in exactly two sanctioned ways: the
    title's `\\\\` line break becomes a space (a web form has no TeX line
    breaks), and `\\%` becomes `%` in the abstract (the form is plain text
    with optional $...$ math, so an escaped percent would show a stray
    backslash). Anything else - a reworded sentence, a stale number, a
    changed author - is a mismatch. Returns human-readable problems; an
    empty list means the form fields match.
    """
    problems: list[str] = []
    f_title = re.search(r"\*\*Title\*\*: `([^`]+)`", submission_md)
    f_auth = re.search(r"\*\*Authors\*\*: `([^`]+)`", submission_md)
    f_abs = re.search(r"```text\n(.*?)\n```", submission_md, re.S)
    tex_title = _braced_after(tex, "\\title")
    tex_auth = re.search(r"\\IEEEauthorblockN\{([^}]*)\}", tex)
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)

    if not (f_title and f_auth and f_abs):
        return ["ARXIV_SUBMISSION.md is missing a Title, Authors or ```text abstract block"]
    if tex_title is None or tex_auth is None or a is None:
        return ["main.tex is missing \\title, \\IEEEauthorblockN or the abstract"]

    want_title = _collapse(tex_title.replace("\\\\", " "))
    if _collapse(f_title.group(1)) != want_title:
        problems.append("title: form %r != tex %r" % (_collapse(f_title.group(1)), want_title))
    if f_auth.group(1).strip() != _collapse(tex_auth.group(1)):
        problems.append("authors: form %r != tex %r" % (f_auth.group(1).strip(), _collapse(tex_auth.group(1))))

    form_abs = f_abs.group(1).strip()
    want_abs = _collapse(a.group(1)).replace("\\%", "%")
    if form_abs != want_abs:
        n = next((k for k, (x, y) in enumerate(zip(form_abs, want_abs)) if x != y),
                 min(len(form_abs), len(want_abs)))
        problems.append("abstract differs at char %d: form ...%r vs tex ...%r"
                        % (n, form_abs[max(0, n - 20):n + 30], want_abs[max(0, n - 20):n + 30]))
    if len(form_abs) >= 1920:
        problems.append("form abstract is %d chars (limit 1,920)" % len(form_abs))
    if "\\%" in form_abs or "\\\\" in form_abs:
        problems.append("form abstract contains a TeX-only escape (\\% or \\\\)")
    return problems


USREP_CSV = ROOT / "reports" / "horizon_us_de_thresholds.csv"


def usrep_table_problems(tex: str, csv_text: str) -> list[str]:
    """Table IV (tab:usrep, the Delaware horizon sweep) against its source.

    Two separate things are checked, because a reviewer will do both:

    1. every cell reproduces from reports/horizon_us_de_thresholds.csv,
       which scripts/run_horizon_us_replication.py writes (a reviewer who
       recomputes the table from the data must get the same numbers), and
    2. the table is self-consistent at its own displayed precision: Range is
       Best minus the 1-month value. A reader doing that subtraction on the
       printed cells must land on the printed Range. The first version of
       this table printed "1 month", "10 years" and "Range" with Range
       defined as best-minus-first but best != the 10-year value at the 1%
       and 0.2% cuts, so 30.10 - 16.16 = 13.94 sat next to a printed 14.5.
       Every number was right; the column just did not say what it was.

    `tex` must have comments stripped. Returns problems; empty means OK.
    """
    m = re.search(r"\\label\{tab:usrep\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return ["Table tab:usrep not found in main.tex"]
    row_re = re.compile(
        r"Top\s+([\d.]+)\\%\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*"
        r"\((\d+)\s*yr\)\s*&\s*(?:\\textbf\{)?([\d.]+)\}?\s*\\\\")
    rows = {float(r[0]) / 100: r for r in row_re.findall(m.group(1))}

    series: dict[float, dict[int, float]] = {}
    for rec in csv.DictReader(csv_text.splitlines()):
        series.setdefault(round(float(rec["top_fraction"]), 6), {})[int(rec["lookback_months"])] = \
            100 * float(rec["AccHR"])
    if set(round(f, 6) for f in rows) != set(series):
        return ["cuts in table %s != cuts in CSV %s" % (sorted(rows), sorted(series))]

    problems: list[str] = []
    for f, s in series.items():
        key = next(k for k in rows if round(k, 6) == f)
        _, one, ten, best, best_yr, rng = (float(x) for x in rows[key])
        b_lb = max(s, key=s.get)
        want = {"1 mo": (one, s[1]), "10 yr": (ten, s[120]), "best": (best, s[b_lb]),
                "range": (rng, s[b_lb] - s[1])}
        for name, (shown, exact) in want.items():
            dp = 1 if name == "range" else 2
            if abs(shown - round(exact, dp)) > 1e-9:
                problems.append("top-%g%% %s: table %s vs source %.*f" % (f * 100, name, shown, dp, exact))
        if abs(best_yr - b_lb / 12) > 1e-9:
            problems.append("top-%g%% best lookback: table %g yr vs source %g yr" % (f * 100, best_yr, b_lb / 12))
        if abs((best - one) - rng) > 0.06:
            problems.append("top-%g%% Range %s is not Best-1mo (%.2f) at displayed precision"
                            % (f * 100, rng, best - one))
    return problems


def main() -> None:
    if not MAIN.exists():
        print("no package at %s" % PKG)
        sys.exit(1)
    raw = MAIN.read_text(encoding="utf-8")
    # Strip LaTeX comments before scanning for forbidden constructs: the
    # file's own header comment explains WHY \pdfoutput is absent, and
    # matching that produced a false failure on the first run. An
    # unescaped % starts a comment; \% does not.
    t = re.sub(r"(?<!\\)%.*$", "", raw, flags=re.M)
    files = sorted(p for p in PKG.rglob("*") if p.is_file())

    # --- abstract -----------------------------------------------------
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S)
    if not m:
        check(False, "abstract present", "no abstract environment")
    else:
        body = m.group(1).strip()
        check(len(body) < 1920, "abstract under 1920 chars", "%d chars" % len(body))
        check("abstract" not in body.lower()[:40],
              "abstract does not open with the word 'Abstract'", "")

    # --- web-form fields must match the source exactly ----------------
    sub_md = ROOT / "paper" / "ARXIV_SUBMISSION.md"
    if sub_md.exists():
        problems = form_field_mismatches(t, sub_md.read_text(encoding="utf-8"))
        check(not problems, "form title/authors/abstract match main.tex",
              "; ".join(problems))

    # --- Delaware table must trace to its source CSV ------------------
    if USREP_CSV.exists():
        problems = usrep_table_problems(t, USREP_CSV.read_text(encoding="utf-8"))
        check(not problems, "Table IV (Delaware) traces to source CSV, Range self-consistent",
              "; ".join(problems))
    else:
        check(False, "Table IV (Delaware) traces to source CSV, Range self-consistent",
              "missing %s - run scripts/run_horizon_us_replication.py" % USREP_CSV.name)

    # --- forbidden constructs ----------------------------------------
    for pat, label in [
        (r"\\pdfoutput", "no \\pdfoutput"),
        (r"\\usepackage\{psfig\}", "no psfig package"),
        (r"\\usepackage\{lineno\}|\\linenumbers", "no line numbers"),
        (r"\\watermark|\\usepackage\{draftwatermark\}", "no watermark"),
        (r"\\marginpar", "no margin notes"),
        (r"\\includegraphics[^}]*\.svg", "no SVG figures"),
        (r"\\includegraphics[^}]*\.eps", "no EPS figures (pdflatex)"),
    ]:
        hits = re.findall(pat, t)
        check(not hits, label, "" if not hits else "found %d" % len(hits))

    # --- required metadata -------------------------------------------
    for pat, label in [(r"\\title\{", "title present"),
                       (r"\\author\{", "author present"),
                       (r"\\begin\{document\}", "document environment"),
                       (r"\\end\{document\}", "document closed")]:
        check(bool(re.search(pat, t)), label)

    # --- type size and margins ---------------------------------------
    m = re.search(r"\\documentclass\[([^\]]*)\]", t)
    opts = m.group(1) if m else ""
    pt = re.search(r"(\d+)pt", opts)
    size = int(pt.group(1)) if pt else 10
    # IEEEtran sets its own type size and margins to the publisher's spec,
    # which satisfies arXiv's 10-14pt and 1in minimums; geometry is only
    # needed when using the plain article class.
    ieee = "IEEEtran" in t
    check(ieee or (10 <= size <= 14), "type size 10-14pt",
          "10pt (IEEEtran conference default)" if ieee else "%dpt" % size)
    check(ieee or "margin=1in" in t or "margin=1.0in" in t,
          "margins", "set by IEEEtran class" if ieee else "geometry margin=1in")
    check(r"\doublespacing" not in t and r"\onehalfspacing" not in t,
          "single spaced", "")

    # --- packages: TeX Live only, no local .sty ----------------------
    local_sty = [f.name for f in files if f.suffix == ".sty"]
    check(not local_sty, "no custom .sty shipped", ", ".join(local_sty))
    pkgs = set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", t))
    pkgs = {p.strip() for grp in pkgs for p in grp.split(",")}
    KNOWN = {"fontenc", "inputenc", "geometry", "amsmath", "booktabs", "graphicx",
             "pgfplots", "hyperref", "microtype", "tikz", "amssymb", "natbib", "url",
             "cite", "algorithm", "algorithmic", "multirow", "subcaption"}
    unknown = pkgs - KNOWN
    check(not unknown, "all packages are standard TeX Live", ", ".join(sorted(unknown)))

    # --- figures -------------------------------------------------------
    figs = [f for f in files if f.suffix.lower() in
            {".svg", ".eps", ".ps", ".tif", ".tiff", ".gif", ".bmp"}]
    check(not figs, "no unaccepted figure formats in package",
          ", ".join(f.name for f in figs))
    inputs = re.findall(r"\\input\{([^}]*)\}", t)
    missing = [i for i in inputs if not (PKG / (i if i.endswith(".tex") else i + ".tex")).exists()]
    check(not missing, "all \\input files present", ", ".join(missing))

    # --- bbl naming ----------------------------------------------------
    bbls = [f for f in files if f.suffix == ".bbl"]
    check(all(f.stem == MAIN.stem for f in bbls), "any .bbl matches main .tex basename",
          ", ".join(f.name for f in bbls) or "none shipped (bibliography inlined)")

    # --- hidden files ---------------------------------------------------
    hidden = [f.name for f in files if f.name.startswith(".")]
    check(not hidden, "no hidden files", ", ".join(hidden))

    # --- balance --------------------------------------------------------
    for env in ["document", "abstract", "table", "table*", "tabular", "figure",
                "figure*", "itemize", "thebibliography", "tikzpicture", "axis",
                "equation", "IEEEkeywords"]:
        allf = t + "".join((PKG / (i if i.endswith(".tex") else i + ".tex")).read_text(encoding="utf-8")
                           for i in inputs
                           if (PKG / (i if i.endswith(".tex") else i + ".tex")).exists())
        # re.escape: 'table*' contains a regex metacharacter, so an
        # unescaped interpolation would match 'tabl' followed by any number
        # of 'e' - silently counting the wrong environment.
        b = len(re.findall(r"\\begin\{%s\}" % re.escape(env), allf))
        e = len(re.findall(r"\\end\{%s\}" % re.escape(env), allf))
        check(b == e, "environment '%s' balanced" % env, "%d begin / %d end" % (b, e))
    check(t.count("{") == t.count("}"), "braces balanced in main.tex",
          "%d open / %d close" % (t.count("{"), t.count("}")))

    # --- labels and references resolve ---------------------------------
    # A \ref to a missing \label compiles to a bare "??" in the PDF - which
    # is easy to miss on a quick read and looks unmistakably broken to a
    # reviewer. Section numbers also shift whenever a section is inserted,
    # so this is checked rather than assumed.
    body = t + "".join(
        (PKG / (i if i.endswith(".tex") else i + ".tex")).read_text(encoding="utf-8")
        for i in inputs
        if (PKG / (i if i.endswith(".tex") else i + ".tex")).exists())
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\(?:page)?ref\{([^}]+)\}", body))
    dangling = sorted(refs - labels)
    check(not dangling, "all \\ref targets exist",
          "%d label(s), %d ref(s); %s" % (len(labels), len(refs),
                                          "all resolve" if not dangling
                                          else "DANGLING: " + ", ".join(dangling)))
    cites = set(re.findall(r"\\cite\{([^}]+)\}", body))
    cites = {c.strip() for grp in cites for c in grp.split(",")}
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
    missing_cites = sorted(cites - bibitems)
    check(not missing_cites, "all \\cite keys have a \\bibitem",
          "%d cited, %d defined; %s" % (len(cites), len(bibitems),
                                        "all present" if not missing_cites
                                        else "MISSING: " + ", ".join(missing_cites)))
    unused = sorted(bibitems - cites)
    check(not unused, "no uncited bibliography entries", ", ".join(unused) or "none")

    # --- report ---------------------------------------------------------
    print("=" * 76)
    print("ARXIV PACKAGE CHECK  (static only - NOT a compile test)")
    print("=" * 76)
    for ok, label, detail in results:
        print("  %-4s %-46s %s" % ("OK" if ok else "FAIL", label, detail))
    failed = [r for r in results if not r[0]]
    print()
    print("package files: %s" % ", ".join(f.name for f in files))
    print("%d checks, %d failed" % (len(results), len(failed)))
    print()
    print("REMAINING MANUAL STEPS (cannot be automated here):")
    print("  1. Compile with pdflatex twice locally or in Overleaf - no toolchain here.")
    print("  2. Upload and inspect arXiv's own submission preview before announcing.")
    print("  3. Choose categories: cs.LG primary, with stat.AP or cs.CY cross-list.")
    print("  4. First submission to a category may need endorsement.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
