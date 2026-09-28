# arXiv submission package

Contents: `main.tex` — **one file, nothing else.** The bibliography is
inlined and Figure 1 is pgfplots source inside `main.tex`, so there are no
external assets, no `.bib`/`.bbl` name-matching, no `\input`, and no
subdirectories. (arXiv itself *does* accept multi-file submissions via
`\input`; this is a single file purely so nothing can be forgotten on upload.)

## Before uploading

1. **Compiled and checked (2026-09-28).** Fresh empty directory containing
   only `main.tex`, `pdflatex` run twice with MiKTeX 25.12: both passes exit
   0, no errors, no undefined references, no rerun warning, no overfull
   boxes; 6 pages (the last holds only the final two references), US
   Letter; every font embedded Type 1. `scripts/verify_arxiv_package.py`
   (41 static checks) also passes. **Caveat:** arXiv builds with TeX Live,
   not MiKTeX, so arXiv's own preview is still the authoritative render.
2. **Upload the source, not a PDF.** arXiv detects TeX-produced PDFs and
   rejects them.
3. **Choose the `PDFLaTeX` processor in the web form.** Do *not* add
   `\pdfoutput=1` to the source: arXiv's current TeX page says not to use
   `\pdfoutput` to change the output format (the `\pdfoutput=1` advice is
   from the retired legacy page). The verifier enforces its absence and
   records the provenance.
4. **Inspect arXiv's own submission preview** before announcing.

## Form fields

Paste these exactly. `scripts/verify_arxiv_package.py` checks them against
`main.tex` on every run; the only sanctioned differences are that the title's
`\\` line break becomes a space and `\%` becomes `%` in the abstract (a web
form has no TeX line breaks, and an escaped percent would show a stray
backslash). `$...$` math is left as-is; arXiv renders TeX math in abstracts.

- **Title**: `Replication Failure and Trivial Baselines in Road-Level Crash Prediction`
- **Authors**: `Maurya Patel`
- **Abstract** (1493 characters, limit 1,920; do not paste the
  word "Abstract", keep it as one paragraph):

```text
Graph neural networks are increasingly applied to road-level crash prediction, but the stability of their reported gains has received little scrutiny. We independently reconstruct the data pipeline of a recent uncertainty-aware model and evaluate eleven of its design decisions across three London boroughs under an expanding-window protocol. Four survive replication on a second borough; seven do not, and four of those reverse sign rather than attenuate. Multi-seed evaluation is decisive: one effect reverses sign between random seeds within a single borough, and the reference architecture exhibits per-borough seed spreads of up to 35.7 points against 4 points for ours. We further compare both networks against a parameter-free baseline that ranks segments by cumulative past crash count. At matched history depth our model is statistically indistinguishable from that baseline ($-0.90$ points, $p=0.61$), and the reference architecture loses to it on 18 of 18 held-out windows ($-17.37$, $p<10^{-6}$). Sweeping the baseline's lookback horizon shows it spans 22.71% to 83.94% accuracy on that variable alone, and that every published figure in this line of work is matched by the baseline at a horizon of one to five years. We argue that the apparent margin of graph networks over historical baselines in this task is substantially an artefact of the short horizons those baselines were computed over, and recommend horizon-matched baselines and multi-seed reporting as minimum practice.
```

- **Comments**: `6 pages, 4 tables, 1 figure. Code: https://github.com/Maurya1112-sudo/greyspot`
- **Primary category**: `cs.LG` (machine learning). The contribution is
  methodological — replication behaviour and baseline design — rather than
  a transport-domain result.
- **Cross-list**: `stat.AP` (applied statistics) fits the paired-testing
  and seed-variance content. `cs.CY` is a weaker alternative.
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
