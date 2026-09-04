# Methodology (draft)

*Addresses L6.07 (research methodology, critical thinking). Describes what
was actually built and why, at the level of detail a dissertation
methodology chapter needs — cross-referenced to the actual code rather than
restated in the abstract.*

## 1. Research question and hypotheses

**Research question** (from the project dossier, unchanged): does
connected-road-network modelling, with calibrated uncertainty, improve
reliable prioritisation on unseen places and later periods compared with
strong non-graph baselines?

**H1**: A GAT+temporal model ranks elevated-risk segments better than a
strong XGBoost baseline on spatially/temporally held-out data.
**H2**: Conformal prediction intervals attain their target empirical
coverage on held-out data.

Both hypotheses are stated so that they can be *rejected* — H1 in
particular has already returned a nuanced, non-dominant first result (see
`docs/results.md`), which is treated as valid evidence, not a failed
experiment.

## 2. Data sources and provenance

| Source | Role | Verification |
|---|---|---|
| STATS19 collision/casualty/vehicle (DfT), 2021–2025 | Historical outcomes, severity, vulnerable-user and vehicle context | `docs/research_notes.md` |
| OSMnx / OpenStreetMap | Road-network graph (nodes = intersections, edges = segments) — the annual pipeline's network | ″ |
| **OS Open Roads (Ordnance Survey, OGL)** | **Road-network graph for the UCL-benchmark pipeline** — the reference paper's own network source | `ingest/os_open_roads.py`, `docs/ucl_benchmark_results.md` |
| IMD 2019 (London Datastore, LSOA level) | Equity context, lagged; also the socio-demographic substitute in the UCL-benchmark pipeline | ″ |
| OSM POI tags (shop/amenity/leisure/tourism) | Point-of-interest density (disclosed substitute for OS's commercial POI product) | `ingest/poi.py` |
| Open-Meteo historical reanalysis | Weather (disclosed substitute for Met Office) | `ingest/weather.py` |

Scope: City of Westminster only (ONS code `E09000033`) for the annual
pipeline, matching the dossier's committed primary geography. The
UCL-benchmark pipeline covers all three of the reference paper's own
case-study boroughs (Westminster, Lambeth, Tower Hamlets).

**Network-source note (2026-09-02)**: two network sources are in
active use, deliberately. The annual pipeline uses OSMnx; the
UCL-benchmark pipeline uses **real OS Open Roads**, since matching the
reference paper's own network source turned out to be the single
largest driver of benchmark performance (a statistically significant
+11.59-point AccHR@20 improvement, p=0.0010 across 18 paired windows —
see `docs/ucl_benchmark_results.md`). `ingest/os_open_roads.py` builds
an OSMnx-compatible `MultiDiGraph` from the national GeoPackage, so
every downstream function works unchanged regardless of which source
produced the graph. Two real bugs were found and fixed the first time
this loader was exercised end-to-end: bbox-only clipping (not clipping
to the real administrative boundary, which over-included road links
from outside the borough) and integer coercion of OS's UUID-style node
and edge IDs when re-loading a cached graph. Both are covered by
regression tests.

## 3. Preprocessing pipeline

Implemented in `src/greyspot/` and orchestrated by `scripts/run_pipeline.py`:

1. **Ingest** (`ingest/stats19.py`, `ingest/imd.py`): download/filter
   STATS19 to Westminster; load casualty/vehicle tables; load the IMD
   lookup.
2. **Network construction** (`ingest/network.py`): fetch the Westminster
   drivable road graph; convert to an edge table with a stable `segment_id`.
3. **Spatial join** (`ingest/network.py::snap_collisions_to_graph`): snap
   each collision to its nearest road segment (100% join rate achieved on
   the current data — see `docs/results.md`).
4. **Feature/target table** (`features/build_features.py`): cross-join
   every segment × every year; aggregate collisions into segment-year
   counts (the target); attach lagged (prior-year only) static, network and
   enrichment features.
5. **Graph-temporal tensors** (`features/graph_temporal.py`): a
   segment-level line graph (segment = node, shared intersection = edge)
   and `[T, N, F]` feature tensors for the GAT+GRU model.

## 4. Leakage controls

Three controls are treated as non-negotiable, not best-effort — the third
was added 2026-08-31 after a real leakage bug slipped past the first two:

- **No random splits.** `eval/splits.py` implements only `temporal_split`
  (train on earlier years, test on a later year) and `spatial_split` (hold
  out a random *segment*, never a random *row*). A row-level random split
  would let the same road appear in both train and test across different
  years, overstating performance.
- **No same-year enrichment features.** `features/build_features.py`
  computes severity/vulnerable-user/vehicle/IMD aggregates for a
  segment-year, then immediately drops the same-year columns and keeps only
  their prior-year lagged versions (regression-tested in
  `tests/test_enrichment.py`) — because the same-year values are computed
  from the very collisions that make up that segment-year's target count.
- **The GAT's "temporal" evaluation must train on a genuinely different
  transition than the one it is evaluated on.** A real bug (found
  2026-08-31, see `docs/decision_log.md`) trained the GAT's loss directly
  against the exact year it was later scored against, with nothing held
  out — an in-sample fit reported as a held-out temporal test. Fixed with
  `features/graph_temporal.py::build_walkforward_instances` +
  `models/gat_temporal.py::train_gat_temporal_walkforward`: train on an
  earlier (window, target) transition, evaluate on a later, disjoint one.
  Regression-tested in `tests/test_walkforward.py`. XGBoost's temporal
  split never had this problem — it genuinely trains only on early-year
  rows and tests only on the final year's rows, which is what made the bug
  specific to the GAT's differently-structured (single shared input
  sequence) evaluation.

## 5. Evaluation protocol

Three held-out regimes, run on the same underlying feature table:

| Regime | What's held out | Question it answers |
|---|---|---|
| Temporal | Later year (2024), same segments | Does last year predict next year? |
| Spatial | 20% of segments (never seen in training), earlier years | Does this generalise to unseen roads? |
| Spatiotemporal | Held-out segments **and** the held-out year | The hardest, most realistic test: a genuinely new road, next year |

The GAT+GRU model is transductive (trained on the whole graph at once) and
predicts a *fixed* target year from a fixed input sequence, so it only has
temporal and spatiotemporal analogues, not a spatial-only one — an
architectural consequence, documented in `scripts/run_pipeline.py`'s
comments, not an evaluation gap.

**Metrics** (`eval/metrics.py`): PR-AUC (for "did this segment have ≥1
collision"), Precision@{10,25,50}, and Spearman rank correlation between
predicted score and actual count. Raw accuracy/MAE is deliberately not the
headline metric because the target is extremely zero-inflated — a model
that always predicts zero would look good on those metrics while being
useless for prioritisation (the same point the dossier makes in Section
14, referencing the academic precedent's reported zero-inflation rate).

## 6. Models

| Model | Role | Implementation |
|---|---|---|
| Historical rate | Transparent baseline | `models/baseline.py` — last year's count as this year's score |
| XGBoost | Strong non-graph comparator | `models/xgboost_model.py` — `count:poisson` objective, matched feature set |
| GAT + GRU | Research model | `models/gat_temporal.py` — one GATConv layer over the segment line graph, one GRU layer over the year sequence, softplus decoder, Poisson NLL loss |
| GAT (no graph) | Ablation | Same model, GATConv replaced by a per-node Linear layer |
| GAT (no temporal) | Ablation | Same model, GRU skipped — decodes directly from the last timestep |

## 7. Uncertainty quantification

Two conformal implementations, deliberately kept methodologically
comparable (same fit/calibrate/test year split: fit on 2021–2022,
calibrate on 2023, evaluate on 2024):

- **XGBoost**: MAPIE v1's `SplitConformalRegressor` (`models/conformal.py`).
- **GAT+GRU**: a manually implemented split-conformal (absolute-residual
  quantile method — the same underlying statistics MAPIE uses), because a
  transductive whole-graph model does not fit MAPIE's row-wise
  `fit(X, y)`/`predict(X)` estimator interface.

**Known limitation** (see `docs/literature_review.md` §2): plain
split-conformal assumes exchangeable residuals. On a graph, a node's
residual is correlated with its neighbours', which the graph-conformal
literature (Huang et al. 2023; Zargarbashi et al. 2023) treats as a
distinct problem requiring graph-aware calibration. The current
implementation does not yet account for this — the empirical coverage
result (`docs/results.md`) should be read as a first-pass sanity check,
not a validated theoretical guarantee for the GAT model specifically.

## 8. Experiment discipline

- Fixed random seeds (`seed=42` throughout).
- Every pipeline run's numbers, decisions and surprises are logged
  contemporaneously in `docs/decision_log.md` — including negative/null
  results (the GAT vs XGBoost comparison; the ~5.2% IMD join-failure rate).
- Data-quality gates checked at each stage (join rate, missingness,
  specification-transition awareness) rather than assumed.

## 9. What this methodology does not yet cover

- Statistical significance testing (bootstrap confidence intervals, paired
  tests) on the model comparisons — the current results are point
  estimates only.
- IMD-decile-stratified or road-user-stratified evaluation (equity
  slicing) — currently only aggregate metrics are reported.
- A systematic literature search protocol (see
  `docs/literature_review.md`'s closing note).

These are explicit, scoped gaps for the next phase, not omissions the
dissertation should gloss over.

---

## 10. Feature horizon (added 2026-09-03)

Crash-history features are computed as backward-looking rolling sums per
segment over **7, 14, 30, 90 and 365 days**. The 90- and 365-day windows
were added on 2026-09-03 after analysis of the target established that
30 days is too sparse to estimate per-segment risk (at ~2–3 crashes/day
across ~11.6k segments), and produced this project's first
statistically significant model-side improvement (+6.03 points pooled,
p=0.0303 — see `docs/results.md`).

Two implementation details are load-bearing and worth stating in the
write-up:

1. **The feature table starts a year earlier than the evaluation grid.**
   A 365-day rolling window needs a full year of prior data, so the
   segment-day table is built from 2021-01-01 while the instance grid
   remains anchored at 2022-01-01. Moving the grid start instead would
   silently shift the stride-90 evaluation dates by 5 days
   (365 mod 90 = 5), and every paired significance test — which merges
   on `held_out_start` — would have matched zero windows while still
   producing plausible-looking output. An assertion now enforces the
   grid start.

2. **The collision years loaded must cover the extended table.** The
   2021 rows are only meaningful if 2021 collisions are loaded;
   otherwise the new rows read as crash-free and every 365-day sum is
   silently undercounted. Both issues were caught before the experiment
   ran, not after.

No leakage is introduced: `rolling(window=w, min_periods=1)` looks
strictly backward, and every target window begins strictly after its
input window ends.

## 11. Statistical significance testing (gap in §9 now closed)

Section 9 previously listed significance testing as an open gap. It is
now the project's mandatory standard: every claim is evaluated with a
paired t-test **and** a Wilcoxon signed-rank test over matched held-out
windows (`scripts/paired_significance.py`), and cross-borough
replication is required before any result is reported as real. This
standard has caught two false positives that a single-borough point
estimate would have accepted (hidden=42/42 and the architecture
ensemble; see `docs/ucl_benchmark_results.md`).


> The exact final configuration (architecture, hyperparameters, features,
> protocol) is specified in [`docs/final_model.md`](final_model.md).
