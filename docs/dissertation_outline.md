# Dissertation outline — how the docs/ suite assembles into the final report

*A map from the working documents in this repository to a standard UK
Computer Science final-year dissertation structure. Check this against
your actual course handbook's required structure/word count/referencing
style the moment you have it — chapter names and expectations vary by
department even within the same university.*

| Dissertation chapter | Fed by | Status |
|---|---|---|
| Abstract | Write last, summarising `docs/results.md` + `docs/proposal.md` | Not started |
| 1. Introduction | `docs/proposal.md`, `docs/module_context.md` | Draft exists (proposal) |
| 2. Literature review | `docs/literature_review.md` | Draft exists — needs a systematic search protocol added |
| 3. Requirements & governance | `docs/requirements_and_ethics.md` | Draft exists |
| 4. Methodology | `docs/methodology.md` | Draft exists |
| 5. Design & implementation | `src/greyspot/` + `README.md` + architecture notes in the original dossier | Partial — needs a dedicated design-decisions writeup once the product layer exists |
| 6. Results | `docs/results.md` | Draft exists, regenerate after every pipeline change |
| 7. Discussion | Not yet written — synthesise `docs/results.md` + `docs/literature_review.md` | Not started |
| 8. Testing (software, separate from ML evaluation) | `tests/` (33 unit tests) + this note | Partial — needs an explicit testing chapter distinguishing software correctness from model quality |
| 9. Conclusion & reflection | Not yet written | Not started |
| References | `docs/literature_review.md`'s reference list | Draft exists, needs your course's required style confirmed |
| Appendices | `docs/decision_log.md`, `docs/supervision_log.md`, full source code | Ongoing |

## Living documents vs. one-time drafts

Some files in `docs/` are meant to be **regenerated or appended to
continuously** through the project, not written once:

- `docs/decision_log.md` — append an entry every time you make a design
  decision, find a bug, or get a surprising result. This is the raw
  material the Results and Discussion chapters will be built from.
- `docs/supervision_log.md` — append after every supervisor meeting and
  significant solo session.
- `docs/results.md` — regenerate after every pipeline run that changes the
  model or data.
- `docs/project_management.md` — update the risk register and decision
  gates as they resolve or change.

Others are **drafts to revise**, not append-only logs:

- `docs/proposal.md`, `docs/literature_review.md`,
  `docs/requirements_and_ethics.md`, `docs/methodology.md`,
  `docs/module_context.md` — these should converge toward their final
  dissertation-chapter form as the project matures, rather than
  accumulate history.

## Immediate gaps before this can be called a complete draft

1. No systematic literature search protocol yet (just targeted
   verification reading) — needed for L6.07's methodology credit.
2. ~~No statistical significance testing on model comparisons.~~
   **CLOSED (2026-09-02)** — paired t-tests and Wilcoxon signed-rank
   tests are now the project's standard for every architecture/feature
   claim, applied consistently across all UCL-benchmark experiments.
   See `docs/ucl_benchmark_results.md` §5 for the full tested ledger
   (confirmed positives with p-values, confirmed negatives, nulls).
   `scripts/paired_significance.py` is the reusable helper.
3. No equity-stratified (IMD decile / road-user) evaluation yet, despite
   IMD data already being integrated.
4. No design/implementation chapter — there's no product layer to describe
   yet.
5. No discussion or conclusion chapter — these need the above gaps closed
   first, or at least acknowledged as scoped limitations.

## The UCL benchmark strand (added 2026-09-02)

A substantial body of work now sits alongside the original
GAT-vs-XGBoost research questions: a direct, statistically-validated
benchmark against Gao et al. (2024)'s STZITD-GNN, the closest academic
precedent to this project. This is dissertation-grade material in its
own right and should have its own results section, drawing on
`docs/ucl_benchmark_results.md`:

- **Contribution**: identified the road-network *source* (Ordnance
  Survey vs OpenStreetMap) as the dominant factor in benchmark
  performance — a +11.59-point AccHR@20 improvement, p=0.0010 across
  18 paired walk-forward windows and all three of the reference
  paper's own case-study boroughs. Matches/exceeds the paper's own
  Westminster figure (70.03% vs 68.98%).
- **Negative results worth reporting**: a naive "coarser network"
  proxy (distance-based node consolidation) is significantly *worse*
  (p=0.0277), demonstrating that the benefit comes from the real
  survey-based topology rather than segment count — a genuinely
  informative negative that constrains the explanation.
- **Methodological honesty material**: six independent hypotheses
  tested and closed before the real answer was found; two real bugs
  found the first time a long-built-but-never-exercised code path was
  run; a disclosed CUDA non-determinism caveat. Good raw material for
  a reflection chapter.

---

## Update 2026-09-03 — the results chapter now has a clear spine

**Central claim** (evidenced, not asserted): *for road-level crash
prediction at extreme sparsity, data representation dominates model
architecture.* Two independent significant findings support it, and
both are data-side:

1. **Road network source** — real OS Open Roads vs an OSMnx
   approximation: **+11.59 points, p=0.0010**, all three boroughs.
2. **Feature horizon** — 90/365-day crash history vs a 30-day cap:
   **+6.03 points pooled, p=0.0303**, 10/12 windows.

Against this, **every model-side lever tested was null**: GAT heads,
layers, hidden size, decoder family (ZIP/ZINB/Tweedie), encoder order,
learning rate, loss function (two ranking losses), ensembling (three
kinds), joint multi-borough training, and training-set density. That
asymmetry *is* the contribution.

### A mechanism chapter worth writing

Four experiments converge on one principle: **on a within-day top-k
ranking metric at 99.97% sparsity, per-segment sharpness dominates
coverage.**

| Transformation | Effect on sharpness | Result |
|---|---|---|
| Junction redistribution (1 crash → 8 arms) | destroys | −, p=0.0432 |
| Rank-blend with near-random history | destroys | 75.8% → 29.9% |
| AADF name-propagation (coverage 1.6%→28%) | destroys | −10.42 pts |
| Long-horizon history (30d → 365d) | **preserves + improves** | **+6.03, p=0.0303** |

### Reflection chapter material (strengthened)

- **Two false positives caught by cross-borough replication** —
  hidden=42/42 (won Lambeth +1.46, lost Westminster −9.97) and the
  architecture ensemble (won two boroughs, died on the third,
  p=0.5535 at n=18). Both would have been reported as findings under a
  single-borough protocol.
- **A self-corrected overclaim** — an r=0.731 crash-density correlation
  computed on six Lambeth windows fell to r=0.247 (p=0.32) at n=18. The
  correction is logged in `docs/decision_log.md` rather than quietly
  dropped.
- **Two bugs caught before launch, not after** — the stride-90 grid
  shift (365 mod 90 = 5 days) and an incomplete `YEARS` list that would
  have silently zeroed the new 365-day sums.
- **A disproven hypothesis costing minutes rather than hours** —
  directed/undirected segment consolidation was predicted to halve
  zero-inflation and moved it 99.97% → 99.94%; checked on real data
  before any GPU time was spent.


> Results chapter source of truth: [`docs/final_model.md`](final_model.md).
> §4 gives the two findings that produced the result, §5 the full null
> ledger, §7 the limitations.
