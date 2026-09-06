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
- no `\\pdfoutput` (arXiv forbids forcing the output format)
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

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "paper" / "arxiv"
MAIN = PKG / "main.tex"

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))


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
    check(10 <= size <= 14, "type size 10-14pt", "%dpt" % size)
    check("margin=1in" in t or "margin=1.0in" in t, "1 inch margins", "")
    check(r"\doublespacing" not in t and r"\onehalfspacing" not in t,
          "single spaced", "")

    # --- packages: TeX Live only, no local .sty ----------------------
    local_sty = [f.name for f in files if f.suffix == ".sty"]
    check(not local_sty, "no custom .sty shipped", ", ".join(local_sty))
    pkgs = set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", t))
    pkgs = {p.strip() for grp in pkgs for p in grp.split(",")}
    KNOWN = {"fontenc", "inputenc", "geometry", "amsmath", "booktabs", "graphicx",
             "pgfplots", "hyperref", "microtype", "tikz", "amssymb", "natbib", "url"}
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
    for env in ["document", "abstract", "table", "tabular", "figure", "itemize",
                "thebibliography", "tikzpicture", "axis"]:
        allf = t + "".join((PKG / (i if i.endswith(".tex") else i + ".tex")).read_text(encoding="utf-8")
                           for i in inputs
                           if (PKG / (i if i.endswith(".tex") else i + ".tex")).exists())
        b = len(re.findall(r"\\begin\{%s\}" % env, allf))
        e = len(re.findall(r"\\end\{%s\}" % env, allf))
        check(b == e, "environment '%s' balanced" % env, "%d begin / %d end" % (b, e))
    check(t.count("{") == t.count("}"), "braces balanced in main.tex",
          "%d open / %d close" % (t.count("{"), t.count("}")))

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
