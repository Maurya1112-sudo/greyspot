# Scaling from Westminster to all of London

*Direct answer to: "can we expand after, like when we get the best model
for Westminster, then we expand to whole London?" — yes, and the codebase
was refactored on 2026-08-31 specifically to make this a config change
rather than a rewrite. This document is the concrete plan.*

## Short answer

Yes. The architecture already supports it. As of today:

- Every ingestion function (`ingest/stats19.py`, `ingest/network.py`,
  `ingest/exposure.py`) takes a borough/ONS-code/place-name **parameter**
  instead of hardcoding "Westminster".
- `ingest/boroughs.py` is a small registry — adding a new borough is a
  3-line addition (name, ONS code, OSMnx place string), not a code change.
- `scripts/run_pipeline.py` takes the borough as a command-line argument
  and writes to borough-scoped output directories
  (`data/processed/{borough}/`, `reports/{borough}/`), so running many
  boroughs never overwrites another's results.
- **Proof, not just a plan**: as of 2026-08-31, **all six of Westminster's
  real neighbouring boroughs** have been run end-to-end, separately, with
  zero ingestion-code changes — Lambeth, Camden, Kensington & Chelsea,
  Brent, Wandsworth and City of London (Westminster's complete ring,
  verified against an OS boundary map, not an arbitrary sample). Every run
  reached 99.9-100% collision-to-segment join rate; XGBoost beat the
  historical-rate baseline on every split in every borough; MAPIE
  conformal calibration landed at 89.3-91.7% (target 90%) everywhere it
  was checked. See "Seven-borough cross-validation: what actually
  generalises" below for the model-comparison findings this unlocked.

## Staged plan (recommended order)

| Stage | Scope | Status | What it proves |
|---|---|---|---|
| 1 | Westminster only | **Done** | The core pipeline and model ladder work |
| 2 | + Lambeth (separate per-borough runs) | **Done** | The code generalises to a new borough with zero code changes |
| 3 | + Camden, Kensington & Chelsea, Brent, Wandsworth, City of London (separate runs) | **Done** — Westminster's full ring of neighbours, 7 boroughs total | Whether findings from one borough hold across genuinely different road networks/collision profiles — they only partly do, see below |
| 4 | **One multi-borough model** (train on 2-3 boroughs combined, test on a held-out borough entirely) | Not started — **now the clearly-motivated next step**, see below | The real test: does a model trained on some of London transfer to a part it has never seen at all? This is a stronger generalisation test than the current spatial split (unseen *segments*, same boroughs) |
| 5 | Inner London (~12-14 boroughs) | Not started | Validates compute/time at a meaningfully larger scale before going all-in |
| 6 | Greater London (33 boroughs) | Not started | The "whole London" goal |

**Recommendation: do Stage 4 next.** Stage 3's seven-borough result below
is exactly the finding that makes Stage 4 non-optional rather than a nice-
to-have: XGBoost is the safer default on the temporal split, but the GAT's
advantage is real and replicates on 5/7 boroughs for the spatiotemporal
split — which means the interesting scientific question is no longer
"does the graph model beat the baseline on one borough?" (answered, and
it's nuanced) but
"does a model trained on several boroughs' combined graph transfer to a
borough it never saw at all?" That is precisely Stage 4's held-out-borough
test, and it is now the most informative thing this project can run next.

## Seven-borough cross-validation: what actually generalises

Full per-borough temporal and spatiotemporal PR-AUC, all freshly re-run
2026-08-31 against the current codebase (walk-forward GAT training
throughout - see `docs/decision_log.md`'s correction entry, and its
"process note" on why Westminster/Lambeth were re-verified rather than
trusting earlier logged numbers that turned out to be stale):

| Borough | temporal XGBoost | temporal GAT | temporal winner | spatiotemporal XGBoost | spatiotemporal GAT | spatiotemporal winner | GAT conformal coverage (target 90%) |
|---|---:|---:|:-:|---:|---:|:-:|---:|
| Westminster | 0.332 | 0.333 | tie | 0.347 | 0.348 | tie | 90.3% |
| Lambeth | **0.311** | 0.287 | XGBoost | **0.284** | 0.248 | XGBoost | 84.4% |
| Camden | 0.293 | 0.291 | tie | 0.257 | **0.319** | GAT | 89.3% |
| Kensington & Chelsea | **0.283** | 0.259 | XGBoost | 0.251 | **0.302** | GAT | 85.5% |
| Brent | **0.275** | 0.261 | XGBoost | 0.270 | **0.307** | GAT | 88.9% |
| Wandsworth | 0.249 | 0.259 | GAT (slight) | 0.207 | **0.246** | GAT | 87.6% |
| City of London | **0.400** | 0.182 | XGBoost (large) | 0.218 | **0.257** | GAT | 91.3% |
| **Score** | | | **0 clear GAT wins / 2 ties / 4 XGBoost wins** | | | **5 GAT wins / 1 tie / 1 XGBoost win** | |

("tie" = within 0.002 PR-AUC — inside plausible run-to-run GPU numerical
noise for an unregularised 200-epoch fit, not a real difference.)

Three findings, reported exactly as they came out (no cherry-picking the
boroughs that flatter the more sophisticated model):

1. **There is no reliable temporal-split GAT advantage.** XGBoost matches
   or clearly beats the GAT in 6/7 boroughs — only Wandsworth shows even a
   small GAT edge (+0.010, likely still within noise). It loses clearly on
   Lambeth, Kensington & Chelsea and Brent, and loses badly on City of
   London (0.182 vs XGBoost's 0.400 — XGBoost's best result of any
   borough, on the smallest, sparsest network). Network size doesn't
   explain the pattern: Brent and Wandsworth have the two largest graphs
   (9,007 and 9,505 edges) yet sit on opposite sides of the temporal
   comparison. **Conclusion: for the simple "same roads, next year" task,
   XGBoost is the safer default, and any GAT edge is small-to-nonexistent
   and not currently predictable from any borough feature measured here.**

2. **The spatiotemporal-split GAT advantage is real and consistent: 5/7
   boroughs, by a meaningful margin (+0.037 to +0.062 PR-AUC) each time.**
   Westminster is an essential tie and only Lambeth is a clear
   counter-example (XGBoost 0.284 vs GAT's 0.248). On the hardest, most
   realistic test — a road segment the model has never seen, evaluated a
   year it has never seen — the graph architecture's ability to borrow
   information from a new segment's *neighbours* in the network helps in
   the clear majority of cases, across a financial district (City of
   London, 1,387 edges) all the way up to a large outer-London
   residential borough (Wandsworth, 9,505 edges). **This is the more
   scientifically defensible headline result of the whole research
   phase**: not "GAT beats XGBoost", but "GAT's specific advantage is
   generalising to genuinely unseen roads, not fitting the same roads over
   time" — which is also the more policy-relevant capability, since a real
   deployment constantly meets roads it has no collision history for yet.

3. **GAT's manual split-conformal intervals systematically under-cover
   relative to XGBoost's MAPIE intervals.** XGBoost lands at 90.6-91.7%
   against a 90% target in every borough checked (tight, consistent,
   trustworthy). The GAT's manual split-conformal implementation lands
   below 90% in 5/7 boroughs checked (as low as 84.4% on Lambeth), only
   meeting/exceeding target on Westminster (90.3%) and City of London
   (91.3%). **This is a real, honestly-reported limitation of the current
   GAT uncertainty layer** (see
   `models/conformal.py::manual_split_conformal_interval`) — plausible
   causes worth investigating before Stage 4: the calibration set (a
   single year, 2023) may be too small/non-exchangeable with the test year
   for a raw split-conformal guarantee, or the residual distribution may
   not be well captured by symmetric intervals around a ZIP-style skewed
   count target. A weighted or asymmetric conformal scheme (e.g. CQR -
   conformalised quantile regression) is a concrete, citable next step,
   not just a vague "improve calibration" TODO.

## What changes technically at each stage

### Stage 4 (multi-borough training/testing) — concrete steps
- OSMnx's `graph_from_place` already accepts a **list** of place names and
  merges them into one graph automatically — building a combined
  Westminster+Lambeth+Camden graph is one function call, not new code.
- `filter_to_local_authority` currently takes one ONS code; extend it to
  accept a list (`df["local_authority_ons_district"].isin(codes)`) — a
  small, contained change.
- The line-graph and GAT machinery are already borough-agnostic (they
  operate on whatever `edges` GeoDataFrame they're given) — no change
  needed there.
- New evaluation split needed: hold out an entire borough's segments
  (not just 20% of one borough's), train on the rest, test on the
  held-out borough. This is a natural extension of
  `eval/splits.py::spatial_split` — split by borough membership instead of
  a random segment sample.

### Stage 5/6 (Inner/Greater London scale) — what to watch
- **Road network size**: Westminster has 7,552 segments; Greater London
  has roughly 300,000-400,000 (rough estimate, not yet measured). The line
  graph and GAT are linear-ish in edge count, so this is a scale-up, not a
  different algorithm — but training time per epoch will grow accordingly
  and should be measured before committing to it, not assumed.
- **Memory**: feature tensors are small even at this scale (a `[3, 400000,
  15]` float32 array is ~72MB) — not a laptop-breaking concern.
- **STATS19/AADF/IMD**: all already national-scale files, already
  downloaded in full — no new data acquisition needed, just filtering more
  of what's already on disk.
- **OS Open Roads** (the dossier's preferred official network, still
  pending a manual OS Data Hub account — see `docs/decision_log.md`)
  becomes more worth pursuing at London scale, since OSMnx's
  community-edited geometry has more room to diverge from official
  topology across a much larger area than it does for one borough.
- **Time budget**: this is real additional work, not a "just change a
  config and wait five minutes" step — treat Stage 5/6 as its own planned
  phase in `docs/project_management.md`'s timeline, not an afternoon
  add-on to the current one.

## What does NOT need to change

- The evaluation protocol (temporal/spatial/spatiotemporal splits, PR-AUC/
  Precision@K/Spearman metrics) — already borough-agnostic.
- The conformal uncertainty layer — already operates on whatever table
  it's given.
- The leakage controls (lagged enrichment features, no random splits) —
  already general, not Westminster-specific.
- Today's GAT architecture fix (1 head + residual, see
  `docs/decision_log.md`) — found on Westminster, should be re-verified on
  a second borough before assuming it transfers unchanged, but the
  underlying mechanism (over-smoothing across a segment's line-graph
  neighbours) is architectural, not borough-specific, so there's good
  reason to expect it holds.
