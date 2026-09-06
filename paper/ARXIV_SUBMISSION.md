# arXiv submission package

Contents: `main.tex`, `fig_horizon.tex`. Nothing else is needed — the
bibliography is inlined and the figure is pgfplots source, so there are no
external assets, no `.bib`/`.bbl` name-matching, and no subdirectories.

## Before uploading

1. **Compile it.** No LaTeX toolchain exists on the machine this was
   written on, so `scripts/verify_arxiv_package.py` is a *static* check
   only — a clean run means the detectable-without-compiling failure modes
   are absent, not that the paper builds. Run `pdflatex main` twice (twice
   for cross-references) locally or in Overleaf.
2. **Upload the source, not a PDF.** arXiv detects TeX-produced PDFs and
   rejects them.
3. **Inspect arXiv's own submission preview** before announcing. That is
   the authoritative render.

## Metadata

- **Primary category**: `cs.LG` (machine learning). The contribution is
  methodological — replication behaviour and baseline design — rather than
  a transport-domain result.
- **Cross-list**: `stat.AP` (applied statistics) fits the paired-testing
  and seed-variance content. `cs.CY` is a weaker alternative.
- **Abstract**: 1,609 characters, within arXiv's 1,920 limit. Do not
  paste the word "Abstract" into the field, and indent any line following
  a carriage return.
- **Licence**: arXiv's default non-exclusive licence is fine. Do not add a
  copyright statement to the PDF that restricts redistribution — arXiv
  rejects those.

## Endorsement

First submission to a category may require endorsement from an
established author. An institutional email often qualifies automatically,
but **not always**: arXiv tightened this for `math` in December 2025, and
policy can change per archive. Check current rules at submission time
rather than assuming.

## Known gaps a reviewer will reasonably raise

These are stated in the paper rather than hidden, but expect them:

- Three boroughs for the benchmark comparison, one city.
- No paired test against the reference paper is possible — they publish
  point estimates, not per-window results.
- The horizon-curve mapping of published figures onto our curve is
  suggestive, not controlled: two of those figures are on their data.
- An unexplained target-density discrepancy (99.97% zero vs their
  reported 95.72–96.71%).
