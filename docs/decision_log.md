# Decision log

Dated record of scope/design decisions, per the dossier's evaluation
discipline (Section 14: "keep a decision log for changes to graph
construction, target definition and policy weights").

## 2026-08-31 — Kickoff session

- **Decision**: Build the Python-only research core first (ingestion,
  network graph, baselines, evaluation, minimal map); defer
  FastAPI/PostGIS/React to a later phase.
  **Why**: Matches the dossier's own scope-control rule ("model and
  evaluation come first", Section 18) and gets to a first testable research
  result fastest.

- **Decision**: Use OSMnx/OpenStreetMap as the road-network source instead
  of OS Open Roads for this phase.
  **Why**: OS Open Roads requires a free OS Data Hub account; account
  creation is out of scope for an assistant to do on the user's behalf.
  OSMnx needs no registration and is explicitly sanctioned by the dossier as
  a fallback/cross-check network (Section 9). Swapping later is isolated to
  `ingest/network.py`.

- **Decision**: Use STATS19 collision years 2021–2025 only for this phase
  (not 2018–2020).
  **Why**: DfT does not publish 2018–2020 under the per-year CSV naming
  pattern used for recent years (confirmed via a live 404); those years are
  bundled into large multi-year consolidated files instead. Five years is
  enough for a temporal train/test split (train 2021–2023/2024, test 2025).
  Revisit if more history is needed once the GAT/temporal model is built.

- **Decision**: Filter STATS19 to Westminster using
  `local_authority_ons_district == "E09000033"`, not
  `local_authority_district`.
  **Why**: Inspecting the live 2025 file showed `local_authority_district`
  is no longer populated (constant -1) in current releases.

- **Note (data quality, not yet resolved)**: 2025 is a provisional/partial
  year (smaller file size than prior years). Using it as the final held-out
  test year means the "actual" 2025 collision count is an undercount, not a
  full year. This should be revisited before drawing evaluation conclusions
  from the 2025 test split — either wait for the final annual release, or
  treat 2024 as the primary test year and 2025 as an early-look sanity
  check only.

- **Bug found and fixed**: `osmnx.distance.nearest_edges` (v2.1.1) returns a
  1-D object array of `(u, v, key)` tuples for multiple points, not three
  parallel arrays as older osmnx versions/docs examples suggest. Fixed in
  `ingest/network.py::snap_collisions_to_graph`; regression test added in
  `tests/test_network.py`.

- **First real result** (pipeline run against live data, test year 2024,
  trained on 2021–2023): 5,887 Westminster collisions loaded, 100% snapped
  to a road segment, 7,552 road segments modelled. XGBoost beat the
  historical-rate baseline on PR-AUC (0.327 vs 0.227) and on
  Precision@25/50 (0.76/0.70 vs 0.72/0.60), but was *worse* at
  Precision@10 (0.80 vs 0.90) — with only 10 segments compared, this is
  likely small-sample noise rather than a real reversal, but it should be
  rechecked once more years of history / the GAT model are added rather
  than quietly dropped. Full table: `reports/baseline_vs_xgboost_results.csv`.

## 2026-08-31 — Dataset enrichment (casualty/vehicle/IMD)

- **Added**: STATS19 casualty + vehicle tables (2021–2025) and IMD 2019
  London LSOA-level deprivation data (downloaded directly from London
  Datastore — no account needed, unlike OS Open Roads). New lagged
  (prior-year only) features per segment: casualty counts by severity,
  pedestrian/cyclist casualty counts, vehicles involved, average IMD
  decile of collisions in that segment.

- **Leakage guard**: these enrichment features describe the *same*
  collisions that make up `collision_count` for a segment-year, so only
  their **prior-year (lagged)** versions are exposed to the model — the
  same-year values are computed then deliberately dropped in
  `build_segment_year_table`. Regression-tested in
  `tests/test_enrichment.py`.

- **Real data-quality finding (STATS19 specification transition)**:
  306/5,887 Westminster collisions (~5.2%) did not match any LSOA in the
  IMD 2019 lookup. Likely cause: STATS19's `lsoa_of_accident_location`
  appears to use 2021-census LSOA codes for at least some records, while
  the IMD 2019 file is keyed on 2011-census LSOA codes — boundaries
  changed for a subset of areas between the two censuses. This is exactly
  the kind of specification-transition risk flagged in the dossier's data
  quality gates (Section 8); logged rather than silently dropped, and
  worth a proper 2011↔2021 LSOA lookup table if this feature is kept in
  the Strong Submission tier.

- **Modelling choice, not a hidden feature**: missing `avg_imd_decile`
  (~92% of segment-years, since most have no prior-year collision at all)
  is filled with 0 before XGBoost sees it — an out-of-range sentinel
  ("no history"), not a claim that the location is in decile 0. This is a
  simplification appropriate for a first pass; revisit with an explicit
  missing-indicator column if this feature is carried into later phases.

- **First real finding from the enriched model** (same train/test split as
  above): `prior_year_n_casualties` is the model's **second most
  important feature** (importance 0.234), ranking above the raw
  `prior_year_count` itself (0.199) and just below `prior_2yr_avg`
  (0.270). In other words, *how severe* prior collisions were carries more
  predictive signal for next-year risk than *how many* there were. Overall
  ranking metrics barely moved (PR-AUC 0.326 vs 0.327 before enrichment) —
  so enrichment didn't obviously improve the model in aggregate, but it
  did surface a specific, interpretable, citable signal worth carrying
  into the dissertation's feature-importance discussion.

- **Added spatial + spatiotemporal held-out evaluation** (the dossier's
  `eval/splits.py::spatial_split` existed but was unused until now). The
  pipeline now reports three regimes: **temporal** (same segments, unseen
  year), **spatial** (unseen segments, seen years), and
  **spatiotemporal** (unseen segments *and* unseen year — the hardest,
  most realistic test). Result: XGBoost beats the historical-rate baseline
  on PR-AUC in **all three** regimes (temporal 0.326 vs 0.227; spatial
  0.317 vs 0.212; spatiotemporal 0.339 vs 0.258) — i.e. it isn't just
  memorising known roads, it transfers to unseen ones too. This is the
  strongest result so far and the one worth leading with in the proposal.
  Precision@10 is noisier and sometimes favours the baseline (small N=10
  per split) — reported honestly rather than only showing the metrics that
  favour XGBoost. Full table: `reports/baseline_vs_xgboost_results.csv`.

## 2026-08-31 — GAT+GRU model and MAPIE conformal layer

- **Added**: `features/graph_temporal.py` (segment-level line graph — nodes
  are road segments, edges connect segments sharing an intersection — built
  directly rather than via `networkx.line_graph` to avoid edge-orientation
  relabelling silently breaking the segment_id mapping); `models/gat_temporal.py`
  (one GATConv layer + one GRU layer + softplus decoder, trained with
  Poisson NLL loss, matched to XGBoost's `count:poisson` objective for a
  fair comparison); `models/conformal.py` (MAPIE v1 `SplitConformalRegressor`
  wrapping the XGBoost point model, calibrated on a genuinely disjoint year).
  PyTorch 2.13 (CPU) + PyTorch Geometric 2.8 installed — GATConv works with
  no torch-scatter/torch-sparse extension needed on this PyG version.

- **GNN evaluation shape differs from XGBoost's, by design, not oversight**:
  XGBoost treats each segment-year as an independent row, so it naturally
  supports temporal/spatial/spatiotemporal splits. The GAT is trained
  transductively over the whole line graph to predict one fixed target year
  from a 3-year input sequence (2021–2023 → predict 2024), so it only has a
  **temporal** analogue (loss on all segments) and a **spatiotemporal**
  analogue (loss restricted to the same spatial-holdout segments used
  elsewhere, evaluated only on those). There is no GAT "spatial-only,
  same-year" row — the target year is fixed by construction.

- **First real GAT vs XGBoost result** (line graph: 7,552 nodes, 56,258
  directed edges from 3,349 intersections; 3 input years → predict 2024):

  | split | model | PR-AUC | Spearman | P@10 | P@25 | P@50 |
  |---|---|---|---|---|---|---|
  | temporal | xgboost | 0.326 | 0.270 | 0.7 | 0.72 | 0.66 |
  | temporal | gat_temporal | 0.319 | **0.313** | 0.5 | 0.56 | 0.46 |
  | spatiotemporal | xgboost | 0.339 | 0.272 | 0.5 | 0.64 | 0.60 |
  | spatiotemporal | gat_temporal | 0.308 | **0.307** | 0.5 | 0.40 | 0.44 |

  **Honest reading**: the graph model does not clearly win. It has a
  *better* Spearman rank correlation in both regimes (predicting the overall
  ordering of risk more faithfully) but a *worse* Precision@25/50 (worse at
  getting the specific top-N segments right) and comparable PR-AUC. This is
  exactly the kind of clean, reportable, non-dominant result the dossier
  anticipates (Section 9: "a graph model that does not win is not a failed
  dissertation if the test is clean") — it should be written up as a genuine
  finding, not hidden or spun as a win. Candidate explanations worth testing
  next: only one GAT layer (segments two hops away never influence each
  other); only 3 input years (a GRU has very little temporal signal to
  learn from); no ablation yet isolating the GRU's contribution vs GAT alone.

- **Conformal calibration result** (90% target, calibrated on 2023,
  evaluated on 2024): **empirical coverage = 90.8%**, mean interval width =
  0.88 collisions, n=7,552. This is a strong, clean calibration result —
  almost exactly hits the target — and is genuinely usable evidence for
  the dossier's "does the model know what it doesn't know?" question
  (Section 10). Full table: `reports/conformal_calibration.csv`.

## 2026-08-31 — Ablations, GAT conformal, and a real answer to "does the graph help?"

- **Ablation result (temporal split, same data/training budget as the full
  model, only one architectural switch changed per row)**:

  | model | PR-AUC | Spearman | P@10 | P@25 | P@50 |
  |---|---|---|---|---|---|
  | xgboost | 0.326 | 0.270 | 0.7 | 0.72 | 0.66 |
  | gat_temporal (full) | 0.319 | **0.313** | 0.5 | 0.56 | 0.46 |
  | **gat_no_graph** (GATConv → plain Linear) | **0.347** | 0.280 | **0.8** | **0.72** | **0.72** |
  | gat_no_temporal (GRU skipped) | 0.303 | 0.271 | 0.8 | 0.68 | 0.66 |

  **This is a clear, specific, well-isolated negative result for the graph
  attention component**: removing it (`gat_no_graph`) *improves* every
  precision metric over the full GAT+GRU model, and beats XGBoost outright
  on PR-AUC and Precision@10/50. The full model's one advantage is the best
  Spearman rank correlation of any model (0.313) - it orders risk more
  faithfully overall even while missing more of the exact top-K segments.
  Candidate explanations for the graph *hurting* top-K precision here (to
  investigate further, not yet confirmed): (a) a single GAT layer with 4
  heads may over-smooth a segment's own signal by averaging in ~7-8
  line-graph neighbours' features (56,258 directed edges / 7,552 nodes ≈
  7.5 average degree), diluting a genuinely high-risk segment's own
  distinct signal; (b) only 12 input features may not give attention enough
  to differentiate neighbours meaningfully, so it behaves close to
  unweighted averaging; (c) 3 input years may be too short a sequence for
  the GRU to learn much beyond what the graph step already blurred. This is
  exactly the kind of "graph model does not win, and here is a clean
  ablation-based explanation of why" outcome the dossier explicitly
  accepts as a strong dissertation finding (Section 9) - it should be
  written up prominently, not softened.

- **GAT conformal result**: manual split-conformal (absolute-residual
  quantile, same method MAPIE uses internally) on the GAT+GRU model, fit on
  2021, calibrated on 2022→2023, evaluated on 2024: **90.6% empirical
  coverage** against a 90% target (mean interval width 0.84) - essentially
  matching XGBoost's 90.8%. Both conformal implementations calibrate well;
  see `docs/literature_review.md` for the caveat that plain split-conformal
  on a graph model doesn't carry the same theoretical guarantee a graph-aware
  method would (residuals aren't exchangeable when neighbours' predictions
  are correlated).

- **Open item**: the `gat_no_graph` result suggests the current single-GAT-layer
  design may be actively counterproductive for this task/feature set - worth
  testing a shallower/differently-configured attention mechanism (e.g. fewer
  heads, or attention only over same-road-class neighbours) before concluding
  graphs don't help at all, rather than stopping at this first architecture.
  A proper 2011↔2021 LSOA crosswalk and stratified (equity-sliced) evaluation
  remain pending. Frontend (FastAPI/PostGIS/React) remains out of scope for
  this phase per `docs/proposal.md`.

## 2026-08-31 — Deeper research pass: UCL precedent details, exposure data, regularisation experiment

- **Research**: confirmed the academic precedent (Gao et al. 2024) is
  genuinely UCL-affiliated (James Haworth, SpaceTimeLab; Huanfa Chen and
  Stephen Law, Bartlett CASA). Fetched the paper's full HTML text and found
  concrete training details previously unknown: 2-layer GAT (3 heads, 42
  hidden units — deeper/narrower-per-head than Greyspot's 1-layer/4-head/16-
  hidden setup), dropout 0.2, weight decay 0.01, Adam lr 0.01, 20 epochs
  with early-stopping patience 10. Also confirmed: **the paper has no
  traffic-exposure feature and no graph-vs-no-graph ablation** — both real
  gaps this project's next two changes address. A related UCL PhD thesis
  (Xiaowei Gao, UCL Discovery eprint 10210801) exists but repeatedly timed
  out downloading (43MB) — left for a later session.

- **Added real exposure data (DfT AADF traffic counts)** — freely
  downloadable, no account needed
  ([roadtraffic.dft.gov.uk/downloads](https://roadtraffic.dft.gov.uk/downloads)),
  something the dossier flagged as valuable but possibly infeasible.
  2,975/3,098 Westminster count-point-years snapped to a road segment
  within 100m (`ingest/exposure.py`, `network.py::snap_points_to_graph`
  generalised with a distance cutoff for this). An explicit `has_aadf`
  indicator feature keeps "no data" distinct from "zero traffic" (only
  ~120 count points exist for ~7,500 segments — sparse by nature, not a
  bug). XGBoost's `prepare_features` was changed to leave the AADF columns
  as genuine NaN rather than filling with 0, so XGBoost's native
  missing-value handling manages the sparsity rather than a fabricated
  zero implying "no traffic".

  **Result: a clean, unambiguous win.** XGBoost improved on every single
  split after adding exposure + `has_aadf`:

  | split | PR-AUC (before → after) | P@25 (before → after) | P@50 (before → after) |
  |---|---|---|---|
  | temporal | 0.326 → 0.332 | 0.72 → 0.80 | 0.66 → 0.70 |
  | spatial | 0.317 → 0.328 | 0.60 → 0.64 | 0.62 → 0.64 |
  | spatiotemporal | 0.339 → 0.347 | 0.64 → 0.64 | 0.60 → 0.60 |

- **Tested the overfitting hypothesis from the previous entry by adding
  regularisation (dropout 0.2, weight decay 0.01) to the GAT, matching the
  precedent's values exactly.** **Result: refuted, clearly.** Every GAT
  variant got *worse*, not better:

  | model | PR-AUC (unregularised → regularised) | Spearman (unregularised → regularised) |
  |---|---|---|
  | gat_temporal (full) | 0.319 → 0.252 | 0.313 → 0.257 |
  | gat_no_graph | 0.347 → 0.289 | 0.280 → 0.228 |
  | gat_no_temporal | 0.303 → 0.209 | 0.271 → **0.088** (near-random ranking) |

  **Honest interpretation**: the precedent's regularisation strength was
  tuned for *their* architecture (2-layer, 42 hidden units, presumably a
  larger dataset and more training signal to spare) — applying the same
  numbers unmodified to a much smaller, shallower model was too strong a
  perturbation, not a matched intervention. This is a textbook case of why
  hyperparameters don't transfer between papers without re-tuning for the
  new setting. Testing a lighter regularisation (dropout 0.1, weight decay
  0.001) next, as a genuine follow-up rather than abandoning the idea after
  one failed setting — result to follow in this log.

- **Light-regularisation follow-up (dropout 0.1, weight decay 0.001), then
  a same-data zero-regularisation rerun to complete the comparison.**
  Holding the exposure-enriched feature set constant and varying only
  regularisation strength (all three now genuinely comparable, unlike the
  first heavy-regularisation run which changed exposure and regularisation
  at once):

  | model (temporal split) | PR-AUC: none | PR-AUC: light (0.1/0.001) | PR-AUC: heavy (0.2/0.01) |
  |---|---:|---:|---:|
  | gat_temporal (full) | 0.261 | **0.276** | 0.252 |
  | gat_no_graph | **0.360** | 0.319 | 0.289 |
  | gat_no_temporal | **0.227** | 0.220 | 0.209 |

  **Honest reading, corrected from an earlier overclaim in this log**:
  `gat_no_graph` and `gat_no_temporal` *do* show a clean monotonic
  dose-response (less regularisation → better, all the way to none). The
  full `gat_temporal` model does **not** — light regularisation edges out
  both no-regularisation and heavy regularisation, but the three values
  (0.252/0.261/0.276) sit close enough together, from single training runs
  with no repeated seeds, that this is plausibly noise rather than a real
  effect. Rather than force a clean story onto an inconclusive result: the
  regularisation hypothesis is **refuted for the ablation variants** and
  **inconclusive for the full model**, and no configuration tested ever
  made the full model dramatically better, so reverting to no
  regularisation (`GAT_DROPOUT=0.0`, `GAT_WEIGHT_DECAY=0.0`) is still the
  right call - it's never the worst choice and is simplest to explain.

- **The one fully robust finding across every configuration tested**
  (pre-exposure, post-exposure, and all three regularisation settings —
  six separate training runs in total): **`gat_no_graph` outperforms the
  full `gat_temporal` model on PR-AUC every single time.** This is now the
  headline, evidence-backed answer to H1 for this architecture: a plain
  per-node encoder plus a GRU consistently beats adding graph attention on
  top of it, regardless of what else changes. The real explanation remains
  open (over-smoothing, feature richness, sequence length — see the
  original ablation entry) but the *fact* of it is now solid across six
  independent runs, not a one-off.

- **Final configuration locked in from this research pass**: DfT AADF
  exposure data (kept — a clean, unambiguous win for XGBoost and for
  `gat_no_graph`) + unregularised GAT (kept — never the worst setting
  tested, simplest to explain, and the ablation-variant evidence points
  this way even where the full model is ambiguous). This is the
  configuration `scripts/run_pipeline.py` runs by default going forward;
  `docs/results.md` reflects it.

  > **⚠️ SUPERSEDED.** Every "temporal"-split GAT number in this and the
  > next entry (0.424, 0.421, the whole architecture sweep, the ZIP
  > experiment) was computed with a data-leakage bug — see the correction
  > entry below, dated the same day, found a few hours after this one. The
  > "spatiotemporal"-split GAT numbers were **not** affected (they used a
  > different, already-correct training method) and stand as reported. Kept
  > here, struck through in spirit rather than deleted, because the
  > diagnosis-fix-validate-replicate story that led to this bug being found
  > is itself the point — see `docs/methodology.md` §4 for why this stays
  > in the record rather than being quietly edited away.

## 2026-08-31 — CORRECTION: the "temporal" GAT evaluation had a real data-leakage bug

**What happened**: while running a follow-up experiment (does a
Zero-Inflated Poisson decoder help further?), a 2000-epoch training run
produced a suspiciously perfect result — PR-AUC 0.850, Precision@25 = 1.00,
Precision@50 = 0.98, on a real-world crash-count prediction task where even
strong published models don't get near that. That "too good to be true"
instinct was correct, and following it up found the actual cause.

**The bug**: `scripts/run_pipeline.py`'s "temporal" GAT evaluation called
`train_gat_temporal(x_seq, edge_index, y_test_year, train_mask=None, ...)`
— i.e. it trained the model's loss **directly against `y_test_year`**
(2024's real, actual collision counts) for every single segment, with
nothing held out (`train_mask=None` → all segments included). It then
evaluated the same model's predictions **against those exact same 2024
labels**. This is not a held-out temporal test at any epoch count — it is
training accuracy, reported as if it were generalisation to an unseen
year. More epochs didn't mean "more generalisation," it meant "more
overfitting to the label it was directly optimised against," which is
exactly why 2000 epochs looked dramatically better than 200: the model was
just memorising harder, not predicting better.

**Why XGBoost's temporal numbers were never affected**: XGBoost's temporal
split genuinely trains only on rows from years 2021–2023 and evaluates
only on 2024 rows it has never seen the label (or the row) for — real,
disjoint train/test data. The GAT's "temporal" design was structurally
different (one shared multi-year *input sequence* predicting one target
year, trained and evaluated on the identical instance) and that structural
difference is exactly where the bug hid.

**Why the "spatiotemporal" GAT numbers were fine and needed no fix**: that
evaluation already correctly used `train_mask` to exclude held-out
segments' labels from the loss — a standard *transductive* GNN setup (the
held-out segments' pre-2024 *features* still flow through the graph via
message passing, which is legitimate, but their 2024 *labels* never enter
the loss). This is the same paradigm used in standard GNN benchmarks
(e.g. citation-network node classification) and is genuinely valid.

**The fix**: `features/graph_temporal.py::build_walkforward_instances` and
`models/gat_temporal.py::train_gat_temporal_walkforward` — train the model
on an *earlier*, genuinely different (window, target) transition (here,
with only 4 usable years, the sole option is [2021,2022]→2023), then
evaluate it, unmodified, on the held-out [2022,2023]→2024 transition it
was never trained to predict. This mirrors exactly how XGBoost's
already-correct temporal split works, adapted to a sequence model.
Regression tests added in `tests/test_walkforward.py` specifically to
catch a reversion to this bug.

**The corrected numbers** (Westminster, full pipeline re-run after the
fix — `reports/westminster/baseline_vs_xgboost_results.csv`):

| split | model | PR-AUC (before fix) | PR-AUC (after fix) |
|---|---|---:|---:|
| temporal | gat_temporal (full) | 0.424 *(invalid)* | **0.331** |
| temporal | gat_no_graph | 0.360 *(invalid)* | **0.307** |
| temporal | gat_no_temporal | 0.227 *(invalid)* | **0.249** |
| temporal | xgboost (unaffected) | 0.332 | 0.332 |
| spatiotemporal | gat_temporal (unaffected methodology, GPU-vs-CPU numerical drift) | 0.350 | 0.337 |
| spatiotemporal | xgboost (unaffected) | 0.347 | 0.347 |

**Honest new headline**: the architecture fix (1 head + residual) is
**still real** — the full model still beats both of its own ablations by
a real, if much more modest, margin (0.331 vs 0.307 vs 0.249). But it does
**not** decisively beat XGBoost as the earlier (invalid) numbers claimed —
the two models are now essentially **on par** on the temporal split
(0.331 vs 0.332) and XGBoost is slightly ahead on spatiotemporal (0.347 vs
0.337). This is a much less dramatic, much more credible finding, and it
is the one that should be reported in the dissertation - not the earlier
one.

**Why this is being written up this prominently rather than quietly fixed
and moved on**: finding your own data-leakage bug via a "this result is
too good to be true" instinct, diagnosing the exact mechanism, building a
proper fix with regression tests, and re-running everything honestly is
precisely the kind of critical, rigorous research practice Level 6 work is
supposed to demonstrate (see `docs/module_context.md`'s L6.07 mapping).
Hiding this and quietly editing the numbers would have been a worse
outcome for the project than having made the mistake in the first place.

**Lambeth re-run with the corrected method** (same day):

| split | model | PR-AUC |
|---|---|---:|
| temporal | gat_temporal (full) | 0.291 |
| temporal | gat_no_graph | 0.291 *(statistically indistinguishable from the full model)* |
| temporal | gat_no_temporal | 0.265 |
| temporal | xgboost | **0.311** |
| spatiotemporal | gat_temporal | 0.252 |
| spatiotemporal | xgboost | **0.284** |

On Lambeth, the architecture fix's benefit over `gat_no_graph` **does not
replicate at all** (0.2909 vs 0.2909 - a genuine tie, not a rounding
artefact) and XGBoost is now ahead on both splits. Combined with
Westminster's more modest-than-originally-claimed result, the honest,
final headline for this research pass is: **the graph-attention fix is a
real, small, Westminster-specific improvement over its own ablations, not
a reliable win over XGBoost, and not confirmed to generalise to a second
borough.** This is a substantially more modest claim than either the
original (buggy) result or even the first corrected Westminster-only
result, and it is the one that should stand in the dissertation.

**Additional finding from the Lambeth re-run**: the GAT's conformal
coverage dropped to **83.4%** against the 90% target (vs 88.6% on
Westminster) - a real, meaningful under-coverage, not just noise. The
walk-forward model is trained on a single transition with far less
supervision than the old (invalid) approach, which plausibly explains why
its uncertainty calibration is now visibly weaker on Lambeth. This is an
honest new limitation, not previously visible before the fix.

**Open item**: the ZIP decoder experiment (see the earlier entry showing a
promising 0.458 PR-AUC) was run with the old, invalid training method and
must be re-run with `train_gat_temporal_walkforward` before its result can
be trusted either way - not yet done, flagged rather than assumed to still
hold. More years of history (recovering 2018–2020, or waiting for a full
2026 STATS19 release) would directly address the thinness of the
walk-forward training set (currently just one transition) and is now the
single highest-value data improvement for this specific research question.

## 2026-08-31 — The over-smoothing fix: found, validated, and confirmed across boroughs

- **Root-caused and fixed the graph-hurts-precision problem.**
  `scripts/experiment_gat_architecture.py` swept two candidate fixes for
  the over-smoothing hypothesis (a segment's own signal being diluted by
  averaging with ~7-8 line-graph neighbours): (1) a residual/skip
  connection around the GAT layer (the standard GNN-literature fix, e.g.
  GCNII, JKNet - lets the model see the segment's own raw features
  *alongside* the graph-aggregated ones rather than being forced to
  replace one with the other), and (2) fewer attention heads (1 instead of
  4 - less redundant averaging). Tested individually and combined:

  | variant | PR-AUC (temporal) |
  |---|---:|
  | baseline (4 heads, no residual) | 0.261 |
  | + residual only | 0.318 |
  | 1 head only | 0.314 |
  | larger hidden (64) only | 0.272 |
  | **1 head + residual (combined)** | **0.424** |

  The combination is dramatically better than either fix alone -
  synergistic, not additive. This is the first configuration all session
  where the full GAT+GRU model beats its own `gat_no_graph` ablation.

- **Validated on the harder spatiotemporal split and against fresh
  ablations using the new config**
  (`scripts/experiment_gat_best_config_validation.py`), on Westminster:

  | split | model | PR-AUC | notes |
  |---|---|---:|---|
  | temporal | **gat_best_config** (1 head + residual) | **0.424** *(invalid, see below)* | beats xgboost (0.332) and gat_no_graph (0.336, re-run with 1 head) |
  | spatiotemporal | **gat_best_config** | **0.350** | beats xgboost (0.347) - the hardest test, and still wins |
  | temporal | gat_no_temporal (1 head + residual) | 0.277 *(invalid, see below)* | full model still beats this ablation |

  H1 is now genuinely **supported** on Westminster: the graph-temporal
  model beats the strong non-graph baseline on both the temporal and
  spatiotemporal splits, and beats its own ablations - the complete
  reversal of every result logged earlier today, achieved through
  diagnosis-driven architecture changes, not hyperparameter luck.

  > **⚠️ SUPERSEDED (all three "temporal" rows above, and both tables
  > below).** These numbers were produced with the same data-leakage bug
  > documented in the "CORRECTION" entry earlier in this file (the
  > forward-reference at that entry's end - "0.424, 0.421, the whole
  > architecture sweep" - refers to exactly this section, but the mark
  > never landed directly on these two tables until this pass; fixed now
  > for anyone reading top-to-bottom). **Verified, current numbers**
  > (fresh re-run 2026-08-31, current codebase, walk-forward training):
  > Westminster temporal gat_temporal=**0.333** (xgboost 0.332, essentially
  > tied), spatiotemporal gat_temporal=**0.348** (xgboost 0.347, also
  > essentially tied). The "temporal" *(invalid)* figures above are kept,
  > not deleted, for the same reason as the CORRECTION entry gives - the
  > diagnosis-fix-validate story is part of the record. The
  > "spatiotemporal" figures in this section (0.350/0.347) were **not**
  > affected by the leakage bug and remain valid as originally reported.

- **Promoted to the default pipeline config** (`GAT_HEADS=1`,
  `GAT_USE_RESIDUAL=True` in `scripts/run_pipeline.py`). *(The "reproduces
  exactly: 0.424 temporal" claim originally logged here inherited the same
  leakage bug - see the SUPERSEDED note above. The residual+1-head
  architecture choice itself is unaffected and still stands: it is
  evaluated identically to its ablations regardless of which training
  method computes the loss.)* GAT conformal coverage improved slightly
  (89.2% → 91.8% against the 90% target, Westminster) - this coverage
  figure predates the walk-forward fix too; see the seven-borough table
  below for current values.

- **Cross-borough validation on Lambeth and, later, five further
  boroughs** - superseded by the full seven-borough re-run below (see
  "Expanded to Westminster's full ring of neighbours"), which uses the
  corrected walk-forward training throughout and is the version that
  should be cited.

## 2026-08-31 — Frontend build: rolldown-vite/MapLibre incompatibility

- **Problem**: `apps/web` (scaffolded via `npm create vite@latest . --
  template react-ts`) threw `SyntaxError: The requested module
  '.../maplibre-gl...' does not provide an export named 'default'` in the
  browser, and the dev server separately logged `The file does not exist at
  ".../maplibre-gl-worker.mjs" ... dependency might be incompatible with the
  dep optimizer`. Both persisted across a source-level fix (`import *  as
  maplibregl` instead of a default import - maplibre-gl v6 genuinely has no
  default export, confirmed by inspecting its `.mjs` build), an
  `optimizeDeps.exclude: ['maplibre-gl']` config change, and multiple full
  `node_modules/.vite` cache clears + server restarts.
- **Root cause**: the current `npm create vite@latest` React-TS template
  defaults to **Vite 8 with the experimental Rolldown-based bundler**
  (evidenced by a `rolldown-runtime-*.js` chunk appearing in
  `node_modules/.vite/deps/`), not classic esbuild-based Vite. Rolldown's
  dependency pre-bundler kept re-generating `.vite/deps/maplibre-gl.js`
  despite the `exclude` flag and mishandled maplibre-gl's internal
  Web-Worker entry point, producing a module-resolution failure that no
  amount of cache-clearing fixed because the bundler itself, not the cache,
  was misbehaving.
- **Decision**: pin `apps/web/package.json` to the stable, long-established
  esbuild pipeline - `vite: ^5.4.11` (resolved 5.4.21) and
  `@vitejs/plugin-react: ^4.3.4` (resolved 4.7.0) - instead of chasing
  Rolldown-specific workarounds, and remove the now-unneeded
  `optimizeDeps.exclude` from `vite.config.ts`.
  **Why**: Vite 5 + esbuild is the combination maplibre-gl's own examples
  and the wider React+MapLibre ecosystem are built against and documented
  for; a bleeding-edge, still-experimental bundler is the wrong foundation
  for a project whose deliverable includes reproducibility for academic
  assessment. After a clean `npm install`, the error was gone on the first
  load with no other source changes required.
- **Verified end-to-end in the browser** (not just "server starts
  cleanly"): the map renders (zoom controls + MapLibre attribution present
  in the accessibility tree), the priority queue populates from real
  `GET http://127.0.0.1:8000/boroughs/Westminster/...` responses (200 OK),
  and clicking a queue row loads that segment's full evidence panel (score
  breakdown, observed history, exposure, model prediction, limitations)
  from the live FastAPI backend.

## 2026-08-31 — Expanded to Westminster's full ring of neighbours (7 boroughs)

- **Decision**: run the full pipeline (GPU-trained, walk-forward-corrected
  GAT) separately on Camden, Kensington & Chelsea, Brent, Wandsworth and
  City of London - Westminster's complete real neighbour ring, verified
  against an OS boundary map earlier this project (`ingest/boroughs.py`),
  not an arbitrary convenience sample. Combined with Westminster and
  Lambeth (both **re-run fresh today** rather than trusting old logged
  numbers - see the process note below), this gives seven independently
  and consistently run boroughs. All runs used CUDA (RTX 4060 Laptop GPU,
  confirmed via a new one-time device log line in `_resolve_device`,
  `src/greyspot/models/gat_temporal.py`) - each borough's full model
  ladder (baseline + XGBoost x3 splits + GAT+GRU x3 splits + 2 ablations +
  2 conformal layers) trained in well under two minutes end-to-end, GPU
  time itself only a few seconds per model.

- **Process note - a second stale-number bug caught by re-verifying rather
  than trusting the log**: before writing this entry, `docs/results.md`
  and an earlier table in this file disagreed about Westminster's GAT
  temporal PR-AUC (0.331 vs 0.424 - a huge gap). Rather than pick whichever
  number looked more convenient, both Westminster and Lambeth were
  re-run from scratch against today's code. The result confirmed
  `docs/results.md`'s numbers (0.331-family) were correct and the 0.424
  figure was the stale, pre-leakage-fix number that a routine document
  cross-check in the "over-smoothing fix" entry above had already
  flagged as superseded but never removed from an active-looking table.
  Worth recording as its own small lesson: **numbers copied between
  documents drift out of sync with the code; a live re-run beats trusting
  either document when two disagree.**

- **Verified seven-borough table** (all fresh runs, current codebase,
  walk-forward GAT training throughout -
  `reports/{borough}/baseline_vs_xgboost_results.csv` and
  `conformal_calibration.csv`, 2026-08-31):

  | Borough | temporal XGBoost | temporal GAT | temporal winner | spatiotemporal XGBoost | spatiotemporal GAT | spatiotemporal winner | GAT conformal coverage |
  |---|---:|---:|:-:|---:|---:|:-:|---:|
  | Westminster | 0.332 | 0.333 | tie | 0.347 | 0.348 | tie | 90.3% |
  | Lambeth | **0.311** | 0.287 | XGBoost | **0.284** | 0.248 | XGBoost | 84.4% |
  | Camden | 0.293 | 0.291 | tie | 0.257 | **0.319** | GAT | 89.3% |
  | Kensington & Chelsea | **0.283** | 0.259 | XGBoost | 0.251 | **0.302** | GAT | 85.5% |
  | Brent | **0.275** | 0.261 | XGBoost | 0.270 | **0.307** | GAT | 88.9% |
  | Wandsworth | 0.249 | 0.259 | GAT (slight) | 0.207 | **0.246** | GAT | 87.6% |
  | City of London | **0.400** | 0.182 | XGBoost (large) | 0.218 | **0.257** | GAT | 91.3% |

  ("tie" = within 0.002 PR-AUC, i.e. inside plausible run-to-run GPU
  numerical noise for an unregularised 200-epoch fit, not a real
  difference either way.)

- **Headline finding (full detail in `docs/scaling_to_london.md`'s
  "Seven-borough cross-validation" section)**: on the **temporal** split,
  XGBoost matches or clearly beats the GAT in **6/7** boroughs (only
  Wandsworth shows even a small GAT edge, +0.010) - there is no reliable
  temporal-split GAT advantage, and on City of London it loses badly
  (0.182 vs 0.400). On the **spatiotemporal** (hardest, unseen-segment)
  split the picture reverses: GAT shows a real, moderate improvement
  (+0.037 to +0.062 PR-AUC) in **5/7** boroughs, is essentially tied on
  Westminster, and loses clearly only on Lambeth. **This is the project's
  central, honestly-earned empirical finding**: the graph-attention
  architecture's real advantage is generalising to a road segment it has
  never seen, not fitting the same segments over time - and it is a
  moderate, majority-of-boroughs effect, not a universal law.
- **Secondary finding**: XGBoost's MAPIE conformal intervals are
  consistently well-calibrated (90.6-91.7% against a 90% target, every
  borough checked - Westminster 90.9%, Lambeth 91.2%, Camden 90.6%,
  Kensington & Chelsea 91.0%, Brent 90.9%, Wandsworth 90.9%, City of
  London 91.7%); the GAT's manual split-conformal intervals under-cover in
  5/7 boroughs (as low as 84.4% on Lambeth - see the table above),
  meeting/exceeding target only on Westminster (90.3%) and City of London
  (91.3%). Reported as a real limitation of the current manual conformal
  implementation, with concrete next steps (larger/rolling calibration
  window, or a CQR-style asymmetric interval) rather than left as an
  unexplained anomaly.
- **This directly motivates Stage 4** of `docs/scaling_to_london.md` (a
  single model trained on several boroughs' combined graph, tested on one
  held out entirely) as the next concrete piece of work: seven separate
  single-borough runs now show *which* generalisation claim holds
  (spatiotemporal, mostly) and which doesn't (temporal, mostly not), so
  the natural next experiment is whether a multi-borough-trained model
  changes either picture - particularly whether it closes XGBoost's
  temporal-split lead or strengthens the GAT's spatiotemporal one further.

## 2026-08-31 — CARTO basemap regression: free tiles started requiring an API key

- **Problem**: the user reported the live map showing "API KEY REQUIRED"
  watermarked directly on the map tiles, in both the React frontend and
  (unverified until this entry, then confirmed) the standalone Python map
  export. Verified by fetching a CARTO tile URL directly with `curl` -
  the returned PNG itself carries the watermark and a link to
  `carto.com/basemaps/apikey`, baked into the image server-side. Not a
  CORS/referrer issue (the raw fetch needed no browser context at all) -
  CARTO's free anonymous "Positron" raster endpoint, previously used
  since the very first map prototype this project built, appears to have
  started requiring registration at some point before this date.
- **Fix**: switched both map surfaces (`apps/web/src/components/MapView.tsx`
  and `src/greyspot/viz/maplibre_map.py`) to
  [OpenFreeMap](https://openfreemap.org/quick_start/)'s "positron" vector
  style (`https://tiles.openfreemap.org/styles/positron`) - free forever,
  no API key, no rate limit, no registration, explicitly built as a
  permanent alternative for exactly this kind of provider lock-out.
  **Why not just get a CARTO API key**: OpenFreeMap needs no account to
  create or maintain, which matters for a project other people (a
  supervisor, an examiner) need to run without first registering for a
  third-party service; it also serves real vector tiles (crisper at every
  zoom, with labels/place names the old raw raster tiles never rendered)
  rather than a fixed-resolution raster image.
- **Verified**: the style URL returns a valid MapLibre style JSON (checked
  directly, both via `curl`/WebFetch and via `fetch()` executed inside the
  running page); zero console errors on a fresh page load after the
  change; `pytest tests/ -k maplibre` still passes (3/3, unaffected by the
  basemap change since those tests don't inspect basemap tiles).

## 2026-08-31 — Frontend redesign: GOV.UK Design System, played straight

- **Trigger**: the user's own screenshot showed the map (before the CARTO
  fix above) plus a genuinely plain, undifferentiated UI, and asked
  explicitly for the Impeccable design skill to be used properly for a
  full visual pass - "make ui better whole frontend with impeccable."
- **Process**: Impeccable's `impeccable` skill wasn't registered as a
  dispatchable skill in this session (`Skill` tool returned "Unknown
  skill" for it) despite being installed at the user level
  (`~/.claude/skills/impeccable`) - worked around by following its own
  `SKILL.md`-documented procedure directly via the Bash-permitted script
  paths it declares (`node .claude/skills/impeccable/scripts/*.mjs`),
  which is the same effective sequence a proper dispatch would have run.
  Ran `context.mjs` (found no `PRODUCT.md`/`DESIGN.md` - a genuine
  from-scratch redesign, not an extension), then `init.md`'s interview:
  asked the user two focused questions (primary audience; design
  standard) via `AskUserQuestion` rather than inventing answers - user
  delegated the audience question ("do whatever you like") and explicitly
  confirmed **GOV.UK Design System + WCAG 2.2 AA** for the design
  standard. Wrote `PRODUCT.md` from that answer plus the project's own
  existing docs (`docs/proposal.md`, `docs/requirements_and_ethics.md`),
  minimising redundant questions per `init.md`'s own instruction to
  "explore before asking."
- **Skipped the full `new-work.md` concept-tournament machinery
  (concept-seed dice rolls, a decision-page server, image-generation
  comps) as genuinely inapplicable, not as a shortcut**: that apparatus
  exists for open-ended identity exploration when no direction is pinned;
  here the user had already pinned the direction explicitly (GOV.UK
  Design System), which the skill's own rules treat as the "standing
  exit" / canon path - "ask once for two or three products this should
  sit alongside, make their craft level the bar, and execute the canon at
  full fidelity" (`new-work.md`). No image-generation tool exists in this
  environment either, which independently rules out the comp-led round.
  Proceeded straight to grounding the real GOV.UK spec (verified via
  WebFetch/WebSearch against design-system.service.gov.uk and
  govuk-frontend's own source - colours, the static spacing scale, the
  tag and inset-text and warning-text component conventions) and
  `craft-floor.md`'s quality checks, then built.
- **What changed** (`apps/web/src/App.css`, `Header.tsx`,
  `PriorityQueue.tsx`, `EvidencePanel.tsx`, `MapView.tsx`, `index.html`):
  - Full GOV.UK token system: black `#0b0c0c` header, GOV.UK's static
    5/10/15/20/30/40px spacing scale, square corners throughout (GOV.UK
    components are never rounded), the signature solid-yellow `#ffdd00`
    focus outline with a black inset border on every interactive element.
  - **Switched the risk colour scale from an arbitrary blue-yellow-red
    diverging gradient to RAG (Red-Amber-Green)** - the standard UK
    civil-service convention for a priority/risk rating, and a far more
    contextually legible choice for this product's confirmed audience
    (government analysts) than an invented palette. Applied identically
    across the map line colours, the priority-queue tags, and the
    evidence-panel score tag - one colour language, defined once.
  - Added a GOV.UK `tag` component for risk bands (sentence case, light
    background + dark text - GOV.UK's own research found uppercase harder
    to read and dark-background tags got misread as clickable buttons).
  - Converted the evidence panel's "Limitations" section to GOV.UK's real
    `warning-text` component (a bordered "!" circle beside bold text) -
    the actual component for exactly this kind of stated caveat, not a
    generic alert box.
  - Added a floating RAG legend to the live map - the standalone Python
    map export already had one; the React dashboard never did, a real UX
    gap this closes.
  - Thickened the map's risk-line width (1.5-5px → 2-7px across the zoom
    range) - at a whole-borough zoom the original width was easy to miss
    entirely, which is very plausibly a real contributor to the user's
    "it does not have those dots and lines on the street" observation
    (verified the underlying GeoJSON data itself was always correct - all
    7,552 Westminster segments carry `has_score:1` and a real
    `priority_score` - the visual weight was the actual issue).
- **Two real defects found and fixed via the mechanical detector and a
  manual WCAG audit, not left in**:
  1. `node detect.mjs` flagged an invented decorative `border-left` accent
     on the evidence panel's score-hero block as a "side-tab" pattern -
     the most recognisable AI-generated-UI tell. Removed entirely (no
     real GOV.UK component uses that device for a numeric readout). A
     second flag, a 10px left border on the model-limitations caveat, was
     verified against govuk-frontend's actual source
     (`$govuk-border-width-wide` = 10px, confirmed via GitHub) as a
     legitimate real `inset-text` component match, not slop - kept, with
     the verified value corrected from an approximated 5px to the real
     10px.
  2. **Manually computed WCAG 2.2 contrast ratios for every text/
     background colour pair** (relative-luminance formula, not eyeballed)
     and found two real AA failures: the raw GOV.UK "orange" token
     (`#f47738`) is only 2.78:1 as text on white (needs 4.5:1), and the
     raw risk-high red only 3.82-3.90:1 as tag text (AA-large, not the
     4.5:1 a 12px bold tag actually needs). Both were about to ship as
     genuine accessibility defects in a product explicitly committed to
     WCAG 2.2 AA. Fixed with darkened, verified `-text` variants
     (`--color-risk-mid-text: #8a4c00` at 6.74:1; tag red `#ab2d2d` at
     4.95:1) reserved for text use, while the original values stay
     correct for their non-text (map line, tag background) uses -
     recorded as a Named Rule in `DESIGN.md` so it doesn't regress.
- **Wrote `PRODUCT.md` and `DESIGN.md`** at the project root, per
  Impeccable's own convention - durable product truth and the visual
  system respectively, so a future redesign pass (or a different session)
  has real, checkable authority to work from instead of re-deriving
  everything from scratch or drifting from what shipped.
- **Verified**: `tsc -b` clean; `node detect.mjs` clean (one remaining
  flag is the verified-correct real GOV.UK component, not a defect);
  `pytest` 81/81 still passing (this was a frontend-only change); browser
  verification via the accessibility tree and `get_page_text` confirmed
  the RAG tags, map legend, header phase-tag/wordmark, and the
  warning-text markup structure all render as designed (no visual
  screenshot was available in this session's browser tool - see the
  honest limitation noted below).
- **Honest limitation of this verification pass**: this session's browser
  tool could not produce an actual pixel screenshot ("the Browser pane is
  not displayed, so the page is not compositing frames" on every attempt,
  independent of anything in this project's own code). Verification
  instead relied on: the accessibility tree (confirms structure and
  text), `get_page_text` (confirms rendered content), computed WCAG
  contrast math (confirms colour accessibility independent of rendering),
  and a direct in-page `fetch()` check of the new basemap URL (confirms
  the map tile source itself is reachable and valid). This is solid
  evidence the redesign is structurally and functionally correct, but it
  is not the same as a human (or the user, who could and did see a real
  screenshot in this conversation) confirming the final visual result
  looks as intended - flagged honestly rather than claimed as a completed
  visual QA pass. **This honest limitation turned out to matter**: the
  very next user screenshot showed the map canvas rendering completely
  blank (no basemap, no roads) - see the next entry.

## 2026-09-01 — The real blank-map bug: maplibre-gl's Web Worker 404s under Vite

**What happened**: the user's screenshot after the redesign pass showed
the header, priority queue, RAG tags, and floating legend all rendering
exactly as designed - but the map canvas itself was empty (just the page
background colour visible behind the zoom controls). This is a genuine
functional regression, not a cosmetic one, and it had been present since
the Vite-5 pin fixed the *previous* MapLibre bug earlier in this session
- it was never actually caught, because every check run at the time
(console errors, the accessibility tree, `get_page_text`, a direct
`fetch()` of the style URL) passed, and none of them actually verified
that map *tiles* painted onto the canvas.

**Root cause, found by direct inspection of MapLibre's internal state**
(`map.loaded()`, `map.isStyleLoaded()`, `performance.getEntriesByType
('resource')`, and eventually a forced `preserveDrawingBuffer` canvas
pixel readback - see the diagnostic method below): maplibre-gl spawns a
Web Worker to parse vector tiles off the main thread. Under this
project's Vite setup, the worker's script URL 404'd
(`node_modules/.vite/deps/maplibre-gl-worker.mjs` did not exist - the
exact same warning Vite had been printing since the Rolldown-vite era of
this session, which was dismissed at the time as a "harmless,
esbuild-specific quirk" once the console stopped showing a *different*
error - that dismissal was wrong). With the worker never available, the
main thread's style/sprite/font/TileJSON requests all succeeded (none of
those need the worker), `map.getStyle()` returned a fully-populated style
object, and every earlier "it looks fine" check passed - but the actual
vector *tile* fetch-and-parse step, which is dispatched to the worker,
silently hung forever. `map.loaded()` and `map.isStyleLoaded()` both
stayed `false` indefinitely, with zero error events ever firing.

**Why this was hard to catch, and the actual diagnostic method that
worked** (worth recording as a reusable technique, not just this fix):
console-error-absence, DOM/accessibility-tree structure, and even a
direct `fetch()` of the style JSON are all necessary but **not
sufficient** evidence that a WebGL map is rendering real content - none
of them observe the canvas's actual painted pixels. The check that
actually distinguished "looks fine" from "is fine" was: (1) querying
`map.loaded()`/`map.isStyleLoaded()`/`map.isSourceLoaded('openmaptiles')`
directly on the live map instance (temporarily exposed via
`window.__debugMap` for diagnosis, removed once fixed), (2)
`performance.getEntriesByType('resource')` filtered for the tile
domain, which showed the browser had fetched a style, a TileJSON, and
sprite assets, but **never a single `.pbf` vector-tile request** - the
one request type that requires the worker, and (3), for final,
unambiguous confirmation, monkey-patching `HTMLCanvasElement.prototype
.getContext` to force `preserveDrawingBuffer: true` *before* MapLibre's
own script ran (injected as a temporary inline `<script>` at the top of
`index.html`'s `<body>`, since retrofitting the flag onto an
already-created WebGL context is not possible), then reading the
canvas's actual pixels via `gl.readPixels()` and counting unique colours
- 1 uniform colour (background only) before the fix, 5,008 distinct
colours (real roads/buildings/water/parks) after it. A cross-check
against a bare, non-Vite HTML page loading the exact same OpenFreeMap
style via a CDN `<script>` tag confirmed OpenFreeMap/MapLibre themselves
were never the problem - that page rendered correctly throughout,
isolating the bug to this project's Vite bundling specifically.

**The fix**: MapLibre's own documented Vite integration path
(https://maplibre.org/maplibre-gl-js/docs/) - import the worker file
through Vite's `?worker&url` query (which routes it through Vite's own
worker bundler, producing a self-contained chunk with its dependencies
inlined, instead of serving the file verbatim and missing its sibling
`maplibre-gl-shared.mjs` chunk), then call `maplibregl.setWorkerUrl()`
with the resulting URL before constructing any map
(`apps/web/src/components/MapView.tsx`). Verified working in both dev
(fresh `.vite` cache, a fresh browser tab, zero errors, the worker
request now returns 200) and in a production build (`npm run build` now
emits a separate `maplibre-gl-worker-*.js` chunk, ~478KB, that didn't
exist in any earlier build).

**Honest accounting of the mistake**: this bug was introduced (or at
least went uncaught) during the Vite-8-to-Vite-5 migration earlier in
this session, and every verification pass run since then - including
ones explicitly framed as "let's be rigorous about this" - accepted
console-silence and structural checks as sufficient proof. It took the
user's own screenshot, twice, to surface a defect that a canvas-content
check from the very first verification pass would have caught
immediately. The lesson recorded for future sessions on this project (and
now baked into the diagnostic method above): **for any canvas/WebGL-based
UI, "no console errors and the right DOM elements exist" is not
verification that content actually rendered - check the canvas's
internal load-state APIs and/or its actual pixels.**

## 2026-09-01 — Ran `/impeccable audit` on `apps/web` (the user's explicit ask)

The user pointed out, correctly, that the redesign pass earlier this
session used Impeccable's underlying scripts directly rather than its
actual dispatchable commands - a fair distinction, and one worth
following properly once the skill became invocable in this session.
Ran `impeccable audit apps/web` per its own `reference/audit.md`
protocol: the mechanical detector plus a systematic 5-dimension scan
(Accessibility, Performance, Theming, Responsive Design, Implementation
Integrity), each scored 0-4.

**Result: 15/20 ("Good")** - Accessibility 3/4, Performance 3/4, Theming
3/4, Responsive Design 2/4, Implementation Integrity 4/4 (pass - the
detector's one finding is the already-verified real `inset-text`
component, not slop). Three P2 findings, fixed immediately (the user has
given standing permission to act, not just report):

1. **Missing `<h1>`** - the page had no top-level heading at all (jumped
   straight to `<h2>` "Priority queue"), a real semantic-HTML/screen-
   reader-navigation gap any axe/Lighthouse scan would flag. Fixed by
   making the "Greyspot" wordmark an actual `<h1>` (`Header.tsx`), with a
   CSS reset so it still reads as a header wordmark, not a giant page
   title (`App.css`).
2. **Mobile header ate ~22% of viewport height** (178px of 812px) because
   the four-item audit-trail readout wrapped into stacked full-width
   lines below the 1100px breakpoint. Changed to a single horizontally-
   scrollable row (`flex-wrap: nowrap; overflow-x: auto`) - verified down
   to 127px, a real improvement, though not claimed as fully minimal.
3. **MapLibre's RAG line colours were hardcoded hex literals**,
   independent of the CSS custom properties everywhere else in the
   system - MapLibre's paint-expression spec can't read `var(--...)`
   directly, so this had silently drifted from being a genuine single
   source of truth (exactly the risk DESIGN.md's "One Risk Language Rule"
   is meant to prevent). Fixed with a `cssColor()` helper
   (`MapView.tsx`) that reads the real computed CSS custom property value
   at map-construction time, with a hardcoded fallback only for the
   unlikely case the property is missing. Verified the resolved paint
   expression afterward (`map.getPaintProperty('roads-layer',
   'line-color')`) matches `App.css`'s literal values exactly.

Two P3s were noted but deliberately not fixed (touch-target sizes below
the 44px "enhanced" guidance but above WCAG 2.2 AA's 24px minimum; no
dark theme) - both are reasonable, low-priority choices for a desktop-
first analyst tool, not defects, and are recorded as such rather than
silently ignored.

**Verified after fixing**: `tsc -b` and `npm run build` both clean; the
mechanical detector re-run shows no new findings; the map's internal
state (`loaded()`, `isStyleLoaded()`, `isSourceLoaded()`, and the actual
resolved paint-expression colours) confirmed the `cssColor()` refactor
didn't regress the map fix from the entry above; `pytest` 81/81 (frontend-
only change, unaffected).

## 2026-09-01 — OS Open Roads wired in: the dossier's originally-preferred network, finally used

The user downloaded and extracted OS Data Hub's OS Open Roads GeoPackage
(GB-wide, ~2GB, 3.96M `road_link` features, GeoPackage format on my
recommendation - a single modern file `geopandas`/`pyogrio` read natively,
versus Shapefile's fragile multi-file bundle or GML's slower parsing).
This is the network source `ingest/network.py`'s very first version named
as the dossier's preferred upgrade path, deferred all along only because
account creation isn't something an assistant can do on a user's behalf.

**Built `ingest/os_open_roads.py`** - produces an **OSMnx-compatible**
`nx.MultiDiGraph` from a bbox-filtered slice of the GeoPackage, so every
downstream function (`graph_to_edges_gdf`, `snap_points_to_graph`, the
line-graph/GAT machinery, the FastAPI backend) works completely
unchanged - exactly the swap-in this project always intended, not a
parallel code path. Key implementation facts:
- **Bbox filtering via `pyogrio`'s native GeoPackage spatial index**
  (`bbox=`, in the dataset's own EPSG:27700 CRS) reads a borough-sized
  slice (~11,347 features for Westminster) in well under a second,
  without ever loading the 2GB/3.96M-feature national file into memory.
- **The borough bbox itself comes from a lightweight `ox.geocode_to_gdf`
  Nominatim lookup**, not a full `ox.graph_from_place` drive-network
  download - verified within ~0.001° of the equivalent OSMnx drive-graph
  bbox, so OS Open Roads ingestion never actually depends on OSMnx.
- **Highway-type crosswalk** (`_map_highway`): OS's own classification
  vocabulary (`road_classification`, `road_function`, `trunk_road`) mapped
  to OSM-style values every downstream feature/UI already expects -
  Greyspot's own heuristic, verified against a real sample of
  Westminster's data (no official OS-to-OSM crosswalk exists), documented
  as such rather than presented as authoritative.
- **Known, disclosed limitation**: OS Open Roads carries no oneway/
  direction attribute (unlike OSM's `oneway` tag) - every link is added
  in both directions, matching how a two-way OSM street is normally
  represented, but this will over-connect genuinely one-way streets. Not
  hidden - stated in the module docstring and here.
- 13 new tests (`tests/test_os_open_roads.py`), synthetic fixtures only -
  the real 2GB file is never required for the test suite to pass, matching
  this project's existing ingestion-testing policy.
- `scripts/run_pipeline.py` now takes an optional second CLI argument
  (`python scripts/run_pipeline.py Westminster os_open_roads`), writing to
  a separate `{borough}_os_open_roads/` output directory so it never
  overwrites the OSMnx-based run for the same borough - both stay
  available side by side.

**Data-quality check (this project's own established gate)**: 100%
collision-to-segment join rate on Westminster - identical to OSMnx's own
100%, and the first real validation that the new source is at least as
usable as the old one, not just theoretically preferable.

**The graph itself is substantially different, not just a drop-in
upgrade of the same shape**: 22,694 edges vs OSMnx's 7,552 (a real ~3x
increase - OS Open Roads segments more finely, includes access/service
roads OSMnx's `network_type="drive"` filter excludes, and every link is
duplicated for both directions per the limitation above). This matters
for interpreting the model comparison below: **the unit of analysis is
different, not just the data source** - a segment on OS Open Roads
typically covers a shorter stretch of road than the equivalent OSMnx
segment, which changes the base collision rate per segment (more,
shorter segments -> more zero-inflation) independent of which network is
more "accurate." This is a real confound, disclosed rather than glossed
over, in the comparison below.

**Full pipeline re-run, Westminster, OS Open Roads vs OSMnx (same
years, same models, same walk-forward-corrected training)**:

| split | model | OSMnx | OS Open Roads | delta |
|---|---|---:|---:|---:|
| temporal | xgboost | 0.332 | **0.347** | +0.015 |
| temporal | gat_temporal | 0.333 | 0.297 | **-0.036** |
| spatiotemporal | xgboost | 0.347 | **0.435** | +0.088 |
| spatiotemporal | gat_temporal | 0.348 | 0.416 | +0.068 |

XGBoost conformal coverage: 90.1% (OS Open Roads) vs 90.9% (OSMnx) -
both essentially on target. **GAT conformal coverage: 85.3% (OS Open
Roads) vs 90.3% (OSMnx)** - notably worse, consistent with (and now
extending) the seven-borough finding that the GAT's manual conformal
implementation under-covers more often than not.

**Honest interpretation, not a "which network is better" verdict**:

1. **Both models improve on the spatiotemporal split with the official
   network** (+0.088 XGBoost, +0.068 GAT) - real geometry and an official
   road classification plausibly gives both models cleaner signal than
   OSM's community-edited attributes.
2. **But XGBoost improves more than the GAT does**, so on Westminster's
   flagship spatiotemporal result - previously an near-exact tie
   (0.347 vs 0.348, OSMnx) and one of the two clearest "GAT wins" data
   points in the whole seven-borough study - **XGBoost is now clearly
   ahead** (0.435 vs 0.416) on the official network. This is a genuine,
   not-cherry-picked finding that somewhat weakens the project's own
   spatiotemporal headline result on its single most-studied borough,
   and it is reported here exactly as found, not smoothed over because
   it complicates a result already written up favourably elsewhere.
3. **The GAT's temporal-split performance gets worse in absolute terms**
   on the denser official network (0.333 -> 0.297) while XGBoost's
   improves - consistent with, and further reinforcing, the existing
   "no reliable temporal-split GAT advantage" finding across the seven
   OSMnx boroughs, now replicated on a second, independent network source
   for the same borough.
4. **Open question, not yet investigated**: is the GAT's relative
   weakening here because the "1 head + residual" over-smoothing fix
   (tuned against OSMnx's coarser ~7-8-neighbour line-graph topology) no
   longer suits a ~3x denser graph with a different average node degree?
   A architecture re-sweep specifically on the OS Open Roads topology
   would answer this, and is a good candidate for Phase 04 (Model rigor
   hardening) of the project's delivery plan, not assumed either way here.

**Not yet done**: OS Open Roads has only been run for Westminster so far
(one borough, one network source, for a first validation) - extending to
the other six boroughs, and re-running the multi-borough Stage 4
experiment on OS Open Roads instead of OSMnx once that exists, are
natural next steps but not completed in this pass.

## 2026-09-01 — Map UX pass: hover, Street View, 3D buildings, label z-order fix

The user reported three concrete map problems and asked generally for
"the best UI UX for map": street names were unreadable (hidden under the
priority-score overlay), every interaction required a click ("now I have
to click to see"), and asked for 3D buildings and a street-view feature.

**Root cause of the hidden street names**: `MapView.tsx`'s
`map.addLayer({...})` calls never passed a `beforeId`, so the risk
overlay was appended to the very end of OpenFreeMap's own layer stack -
literally on top of every label the basemap draws, including street
names, place names, everything. Inspected the live style's layer order
(`map.getStyle().layers`, 58 layers) to find the boundary between
fill/line layers and the label block, which starts at
`"waterway_line_label"`. Fixed by passing that layer's id as `beforeId`
to both `roads-layer` and `roads-selected-outline` - the overlay now
sits above every basemap fill/line but below every label. Verified
directly (`layers.indexOf('roads-layer') < layers.indexOf('highway-name-major')`),
not just by eye.

**Hover, replacing click-only discovery**: `mousemove`/`mouseleave` on
`roads-layer` drive a small React-rendered card (`.map-hover-card`,
styled from the same GOV.UK tokens as everything else, not a separate
MapLibre popup) that follows the cursor and shows the road's name, type,
RAG tag, and score with no click needed. Click still drives the full
evidence panel - unchanged. Verified by dispatching a synthetic
`mousemove` DOM event at a canvas point returned by
`map.queryRenderedFeatures` (this session's Browser tool cannot produce
real screenshots or a real cursor position - see the note in the
2026-09-01 blank-map entry above; the same honest limitation applies
here, worked around the same way).

**Street View, free, no API key**: the hover card's "Street View ↗" link
opens `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lng}`
in a new tab, using the exact cursor location (`e.lngLat` from the
`mousemove` event, not just the segment's midpoint). This is a plain
navigable URL, not an embedded/keyed API - Google's Street View *embed*
API needs a key and a billing account, which doesn't fit this project's
free-forever architecture; linking out needs neither. Verified the
generated href carries real coordinates matching the hovered road.

**3D buildings, off by default**: a `fill-extrusion` layer
(`buildings-3d`) reads the `openmaptiles` vector source's `building`
source-layer directly - the base "positron" style already includes a
*flat* building fill layer on that same source-layer, so the data was
already there, just never extruded. Height comes from OpenMapTiles'
`render_height` property with a 6m fallback (roughly one storey) for
buildings that don't carry it, following MapLibre's own documented
pattern - extruding to a real height beats collapsing ungeocoded
buildings to 0. A visible `<button aria-pressed>` (not MapLibre's own
control chrome, styled to match the legend card) toggles the layer's
visibility and eases the camera to `pitch: 55` - a 3D extrusion is
visually indistinguishable from its own flat footprint at 0 pitch, so
the toggle needs to move the camera, not just the geometry. Verified
directly: `map.getPitch()` 0 -> 55 and `buildings-3d` layout visibility
`none` -> `visible` after the click, both checked programmatically.

**Verified overall**: `tsc -b` and `npm run build` clean (worker chunk
still emits correctly - this pass didn't touch that fix); the mechanical
detector shows no new findings; `pytest` 94/94 (backend untouched,
frontend-only change).

## 2026-09-01 — Map UX pass, round 2: the Street View link was unclickable, and the basemap was "boring"

The user tried the round-1 map UX changes above and reported two more
concrete problems: the hover card's "Street View ↗" link was impossible
to click ("tooltip moves with cursor as well so I cant"), and the
basemap looked "boring" - an explicit ask for something "more google maps
like" with visible road names and richer 3D buildings.

**The unclickable link, root-caused**: a card that *tracks the cursor*
cannot reliably host a target the cursor has to *travel to*. The hover
card was positioned 14px offset from the pointer; moving the mouse from
the road (where `mousemove` fires and keeps the card open) toward the
link inevitably crosses off the road's hit area first, firing
`mouseleave` and unmounting the card before the pointer arrives - a
structural problem, not a CSS tweak. **Fix**: removed all interactive
content from the hover card (it is now read-only: name, type, RAG tag,
score) and moved Street View to a new `.map-streetview-chip` - a
fixed-position link (bottom-right of the map) set on click, that never
moves. Verified by dispatching a synthetic click via
`canvas.dispatchEvent` at a real road pixel (`map.queryRenderedFeatures`
found the pixel first) and confirming the chip renders with the correct
name and coordinates in the DOM - a static element, trivially clickable
by construction, not just by claim.

**The "boring" basemap**: switched from OpenFreeMap's `positron` style
to its `liberty` style - same free, key-less, no-rate-limit
infrastructure, but liberty is OpenFreeMap's flagship, most detailed
style (111 layers vs positron's 58): full road-type colour hierarchy
with casings, POI icons, transit lines, one-way arrows - much closer to
Google Maps' density than positron's deliberately minimal, near-
monochrome treatment. This is a **deliberate, scoped exception** to the
"GOV.UK restrained palette" strategy the rest of the product follows
(recorded as such in `DESIGN.md`) - the map is the one surface where the
user explicitly asked for visual richness over restraint; the header,
queue, and evidence panel are unchanged.

**3D buildings, better and simpler**: discovered `liberty` already ships
its own `building-3d` fill-extrusion layer (verified by fetching the
style JSON directly) - visible by default past zoom 14, using a flat
`hsl(35,8%,85%)` colour. Deleted the custom `buildings-3d` layer added in
round 1 (which would have z-fought with this one, both extruding the
same source/source-layer) and instead: (1) hide the style's own layer by
default, so the existing toggle button is a deliberate reveal rather
than an automatic one nobody asked for, and (2) re-themed its paint for
real visual depth ("make 3d models more graphics") - a height-based
colour ramp (warm `#e8dcc8` low-rise to cool `#5b6b78` tall buildings)
plus `fill-extrusion-vertical-gradient: true`, which the base style
leaves unset. Verified directly: `getPaintProperty('building-3d',
'fill-extrusion-color')` returns the new ramp, `fill-extrusion-vertical-
gradient` returns `true`, and the toggle still correctly flips
visibility `none`<->`visible` and eases `pitch` `0`<->`55` on the
*style's* layer id (renamed from the round-1 custom layer's id).

**Verified overall**: `tsc -b` and `npm run build` clean (worker chunk
still emits, this pass didn't touch it); mechanical detector shows no
new findings (same single pre-verified `inset-text` flag as every prior
pass); `map.loaded()`/`isStyleLoaded()` true after the style swap, 113
layers loaded; `pytest` 94/94 (backend untouched, frontend-only change).

## 2026-09-01 — First real UCL-comparable run: a genuine result, and a bigger incomparability than first thought

The user, having read the full paper, asked directly to make this
project's model "on par or even better" than Gao et al. (2024). Built
the actual infrastructure to run a genuinely comparable evaluation
(rather than continue reasoning about it in the abstract) and ran it for
real on Westminster:

**New, tested infrastructure** (25 new tests, all passing):
- `features/daily_features.py` - daily (segment, day) collision
  aggregation and the segment-day feature table, reusing STATS19's
  already-parsed `datetime` column (`ingest/stats19.py`) rather than
  re-parsing dates a second time.
- `features/daily_temporal.py` - genuinely vectorised `[T, N, F]`/`[T, N]`
  tensor builders (the annual pipeline's per-row `.iterrows()` loops
  don't scale to a segment-day table - 8.28M rows for Westminster alone,
  confirmed by the real run below) and `build_daily_multistep_instances`,
  the walk-forward analogue for a 14-day forecast horizon with
  genuinely disjoint target windows per instance.
- `models/gat_temporal.py`: `GATTemporal` gained a `horizon` parameter
  (default 1, exactly the prior behaviour) - a "direct multi-horizon"
  decoder predicting all 14 future days in one forward pass. The existing
  `train_gat_temporal_walkforward`/`predict_gat_temporal` needed **zero
  further changes** - both already operate element-wise on whatever
  shape the target/prediction tensors carry.
- `eval/ucl_metrics.py` - MAE, MAPE (a disclosed departure: computed
  excluding zero actuals, since the paper's own formula divides by
  `y_ij` and is undefined at the 99.96%+ zero rate this data actually
  has), RMSE, MPIW, PICP (a second disclosed departure: implemented per
  the paper's own prose definition, not the PDF-extracted formula, whose
  inequality direction and symbol are self-contradictory as written - see
  the module docstring), ZR, and AccHR@a - all matched to Gao et al.
  (2024)'s exact published formulas (Eq. 13-16), unit-tested against
  hand-computed values.
- `scripts/run_ucl_comparison.py` - end-to-end runner: 2022-2024 daily
  data (the paper uses 2019; see below for why year-matching turned out
  not to be the main obstacle), 20-day input window (reading the paper's
  stated "N = 20" as input sequence length - genuinely ambiguous, since
  the paper's own metric formulas reuse `N` for road count; disclosed,
  not asserted as certain), 14-day horizon, quarterly walk-forward stride
  (12 instances, 11 trained on, the latest held out).

**One real bug caught before any result was trusted**: the first run
crashed with a shape-mismatch `ValueError` - `y_true`/`y_pred` had been
transposed in opposite directions at the model/metrics integration
boundary (`held_out_y` is natively `[horizon, N]`, `predict_gat_temporal`'s
output is natively `[N, horizon]` - only one of the two needed a `.T`,
and the script had it backwards on both). Fixed, documented in the
script itself, re-run.

**The result** (Westminster, held-out 14-day window starting
2024-10-07): MAE=0.0009, RMSE=0.0209, ZR=99.96%, MPIW=0.0012, PICP=90.00%,
**AccHR@20=41.3%** - versus STZITD-GNN's own reported MAE=0.0357,
RMSE=0.1015, ZR=73.28%, MPIW=0.0259, PICP=98.93%, AccHR@20=68.98%.

**This is not a win, and treating it as one would repeat exactly the
mistake this project has already caught itself making twice before**
(the leakage bug and the stale-number entries earlier in this log).
Two honest reasons, found by checking rather than assuming:

1. **99.96% of the held-out (segment, day) observations are true
   zeros** (`MAPE_excluded_fraction` in the saved CSV) - at single-borough,
   per-segment, daily grain, actual collisions are extraordinarily rare
   events. MAE/RMSE/ZR computed over a target that is almost entirely
   zero are dominated by how well a model predicts the *trivial* 99.96%,
   not the *informative* 0.04% - a model that simply predicts values
   closer to zero than STZITD-GNN's will score better on these specific
   metrics for that reason alone, which is not the same thing as ranking
   risk better.
2. **A deeper, newly-found incomparability**: re-reading the paper's own
   Section 3.1 while investigating why the numbers looked "too good to
   be true" (the same instinct that caught the original leakage bug)
   surfaced this sentence: *"the crash value applied to both crash counts
   **and the associated severity**"* - their `y_it` ("crash risk score")
   is a severity-weighted composite quantity, not a raw collision count.
   This project's own target (`collision_count`) is a plain count. MAE
   and RMSE computed on two differently-DEFINED target variables are not
   comparable at all, independent of every other difference (year, grain,
   split protocol) already logged in `docs/publication_readiness.md`.
   The exact formula for their composite score could not be recovered
   from the extracted PDF text (the equation itself appears to be an
   image/special-font element that didn't survive text extraction) -
   this is disclosed as an open gap, not glossed over.

**The one metric that *is* still meaningful here, and where this
project currently, honestly, underperforms**: AccHR@20, a pure ranking
metric (does the top-20%-predicted-risk set actually contain the day's
real crashes) that is far less sensitive to the absolute scale/definition
of the target than MAE/RMSE are. **This project's first, simple
"direct multi-horizon" GAT+ZIP model scores 41.3% against STZITD-GNN's
68.98%** - a real, honest gap, not explained away by sparsity or target
scale. Candidate reasons worth investigating next (not yet done): the
"direct" multi-horizon decoder (14 outputs from one forward pass) versus
STZITD-GNN's own ZITD distributional decoder may simply be a weaker
architecture for this specific ranking task; only 11 training instances
(quarterly stride) is far less supervision than a truly continuous
14-day-rolling training regime would give; and the over-smoothing fix
(1 head + residual) was tuned on the *annual* line-graph topology, never
validated at daily grain.

**Updated recommendation for the publication ambition**
(`docs/publication_readiness.md`, updated alongside this entry): the
defensible calibration-precision claim from the earlier pass still
stands untouched (PICP 90.00% here, again close to the 90% target, this
project's conformal approach continues to calibrate tightly against a
stated target where STZITD-GNN's own reported PICP over-covers by ~9
points) - but "on par or better" is not yet true on the metric that
actually reflects the product's real job (ranking which roads are
actually risky), and the honest path to closing that gap runs through
architecture/training changes, not a friendlier metric choice.

## 2026-09-01 - `/impeccable` redesign pass: flow-verification, not a visual overhaul

**User request** (`/impeccable`, args "re design whole ui/ux in this make
sure every flow works as well think what suits better"). PRODUCT.md and
DESIGN.md already record a deliberately-confirmed GOV.UK identity
(`context.mjs` reported no `CONTEXT_STALE` drift), and the map basemap/
3D-buildings/hover-card work earlier the same day was itself already a
direct response to explicit user visual feedback ("this maps looks so
boring", "make 3d models more graphic"). Per the skill's own rule -
"a pinned brief... redirecting a clear brief toward your taste is
failure" - this was treated as a **critique + flow-verification + polish
pass that preserves the existing, already-confirmed visual identity**,
not a license to replace it, with "think what suits better" applied to
interaction gaps rather than to re-skinning a system the user has
already approved twice today.

**Method**: re-read `App.tsx`, `Header.tsx`, `PriorityQueue.tsx`,
`EvidencePanel.tsx`, `MapView.tsx` fresh, then `grep`'d for the
interaction primitives a map/list pairing like this needs
(`flyTo`/`panTo`/`easeTo`, `scrollIntoView`, a loading indicator on the
map itself) rather than assuming they existed. Found three real,
previously-unfixed gaps, not cosmetic ones:

1. **Selecting a road from the priority queue never moved the map.**
   The map only *highlighted* the selected segment's outline
   (`roads-selected-outline` filter) - on a borough-wide view at zoom
   ~11, a single highlighted segment is often too small to see at all,
   so the highlight had close to zero visible effect for exactly the
   rows a user is most likely to click (the top of the list, spread
   across the whole borough). Fixed in `MapView.tsx`: the existing
   segment-highlight `useEffect` now also looks up the selected
   feature's midpoint coordinate and calls
   `map.easeTo({ center: mid, zoom: Math.max(current, 15), duration: 700 })`.
2. **Selecting a road on the map never scrolled the priority queue.**
   The reverse direction of the same gap - clicking a road far down the
   ranking (e.g. rank 26 of 30) highlighted it in the list with
   `queue-row--selected`, but the list gave no indication *where*, since
   nothing scrolled it into view. Fixed in `PriorityQueue.tsx`: a
   `selectedRowRef` attached to the currently-selected row, plus a
   `useEffect` calling `selectedRowRef.current?.scrollIntoView({ block:
   "nearest", behavior: "smooth" })` on `[selectedSegmentId]` change.
   `"nearest"` (not `"center"`/`"start"`) specifically to avoid yanking
   the list when the row is already visible.
3. **Switching boroughs gave no feedback on the map itself.** `App.tsx`
   already tracked a `loadingBorough` boolean and the priority queue
   already rendered a plain "Loading…" line, but the map - the largest
   element on the page, and the one the eye is on right after picking a
   borough from the header - just sat static and unchanged until the
   new GeoJSON arrived, indistinguishable from a stalled or broken
   request. Added a `loading` prop to `MapView`, wired straight from
   `App.tsx`'s existing state (`<MapView ... loading={loadingBorough} />`
   - no new state introduced), and a `.map-loading-overlay` (translucent
   scrim, spinner, "Loading borough…" text, `role="status"
   aria-live="polite"` for screen readers) in `App.css`, respecting
   `prefers-reduced-motion` by freezing the spinner instead of animating
   it.

**Verification** (live in the Browser pane, not just `tsc`/lint):
- Confirmed `npx tsc -b --force` clean and the Impeccable mechanical
  detector showed only the same single, already-verified-legitimate
  `inset-text` border finding on the four touched files
  (`App.tsx`, `App.css`, `MapView.tsx`, `PriorityQueue.tsx`) - nothing
  new.
- Attached a temporary `window.__debugMap` reference (removed before
  finishing, per this project's standing practice of never shipping
  debug hooks) to assert on real MapLibre state rather than trusting
  visual impression alone: clicking priority-queue row 1 moved the map
  from `{lng: -0.1640, lat: 51.5455, zoom: 11.07}` to `{lng: -0.1566,
  lat: 51.5370, zoom: 15}` and set `aria-current="true"` on that row -
  the pan-to-selection fix genuinely fires, not just compiles.
- Shrank the viewport until `.priority-queue`'s scroll container was
  genuinely clipped (`scrollHeight: 1685` vs `clientHeight: 585`),
  clicked row 26 of 30 (off-screen at `scrollTop: 0`), and confirmed
  `scrollTop` moved to `873` with the row's `getBoundingClientRect()`
  now fully inside the panel's visible bounds - the scroll-to-selection
  fix genuinely fires too, in both directions of the map/queue
  relationship the flow review set out to check.
- Switched boroughs via a real `<select>` `change` event (not just
  editing React state directly) and asserted `.map-loading-overlay`
  exists one animation frame after dispatch, then confirmed it is gone
  and the correct borough's queue rows have replaced the old ones one
  second later.
- Re-ran the full suite after removing the debug hook:
  `npx tsc -b --force` clean, `pytest tests/ services/api/tests/` -
  119 passed, 0 changed (frontend-only change set, run anyway per this
  project's established verification habit rather than assumed-safe).

**What this pass deliberately did not do**: repaint the GOV.UK palette,
typography, or layout grid, or touch `Header.tsx`/`EvidencePanel.tsx`
beyond reading them for context - both were re-inspected and found to
already have their own loading/error states and no equivalent
cross-component sync gap, so redesigning them without a concrete finding
would have been change for its own sake, not a fix. If a future session
finds a specific, concrete flow or accessibility gap in either, it
should log it the same way this entry does - a named gap, a fix, and a
live-browser assertion that the fix actually fires - rather than a
general "polish" pass with no falsifiable claim attached.

## 2026-09-01 - "Fieldscope": full creative departure from GOV.UK (OpenDesign)

**User request** (`/opendesign`, verbatim): "redesign my whole project with
your plugin make best ui ux as possible be as much creative you want
animation all the stuff it shoud not look like ai slop and also keep in
mind the flow and also the backend works as well." A structured intake
(per the OpenDesign skill's questioning protocol) confirmed: ground in the
real codebase, but go for a **full creative departure** from the GOV.UK
identity built earlier the same day, covering visuals/motion/layout/copy,
applied directly to the live app (not just a mockup).

**Why a full departure was the right call here, unlike the `/impeccable`
pass earlier today**: that pass explicitly preserved GOV.UK because the
brief was "make sure every flow works" against an identity the user had
already confirmed twice. This request is a different, later, and more
specific instruction from the same user - a named plugin, a named
methodology (OpenDesign's design-system-first workflow), and an explicit
ask to depart from the existing look ("should not look like AI slop"
implies dissatisfaction with genericness, not with GOV.UK specifically,
but the intake question asked directly and the user chose "Full creative
departure" over "Elevate GOV.UK, don't replace it"). Honoring this
instruction over the earlier one is not a contradiction - it's doing what
was actually, currently asked, the same way the map-UX feedback earlier
today was allowed to override an even earlier "keep it plain" instinct.

**The design system**: `opendesign/design-systems/fieldscope/` (tokens,
brand voice/style notes, sources-consulted README with confidence levels
disclosed) - built by actually reading `apps/web/src/{App.tsx,api.ts,
components/*.tsx}` and the previous GOV.UK `App.css`/`DESIGN.md`, not
invented from a blank brief. Concept: Greyspot as a **precision survey
instrument** - a warm "paper" analysis surface (priority queue, evidence
panel) framed by a dark "chassis" bezel (header, map), rather than one
flat theme. Full rationale, non-negotiables, and the "why" behind every
token in `opendesign/design-systems/fieldscope/README.md` and `DESIGN.md`
(root) - not re-derived here.

**Two real technical risks found and fixed during application, not just
during design:**

1. **MapLibre GL's paint-property colour parser cannot parse `oklch()`
   strings.** The system's colour tokens are authored in oklch (for the
   "share chroma/lightness, vary hue" accent discipline the OpenDesign
   default aesthetic calls for) - fine for ordinary DOM CSS (every
   evergreen browser resolves `oklch()` natively), but MapLibre's paint
   properties are compiled to GPU uniforms through MapLibre's own colour
   string parser, which never goes through the browser's CSS engine and
   does not understand oklch syntax. `MapView.tsx`'s existing `cssColor()`
   helper (built during the earlier GOV.UK pass specifically so map
   colours never hardcode-drift from the CSS tokens) would have silently
   handed MapLibre an unparseable string. Fixed by adding a small,
   explicitly-documented set of hex mirrors (`--risk-low-hex`,
   `--risk-mid-hex`, `--risk-high-hex`, `--risk-none-hex`,
   `--selection-outline-hex`) computed directly from the oklch tokens via
   the same OKLab->sRGB math a browser performs (script inlined in this
   session, values checked into `App.css`'s comment) - verified live by
   reading `map.getPaintProperty('roads-layer', 'line-color')` in the
   browser console and confirming it resolved to real hex strings
   (`#067e3f` / `#dd7b2b` / `#c92f33`), not an oklch literal.
2. **OpenFreeMap's "liberty" style has no dark variant.** Rather than fork
   the style JSON (large effort for a student prototype) or drop the
   chassis-zone concept for the map, applied a CSS `filter: brightness(0.8)
   saturate(0.9) contrast(1.04)` to `.map-view` only (not its DOM overlay
   siblings - hover card, buttons, legend - which sit in `.map-view-wrapper`
   alongside it, not inside it) - a well-known, cheap technique for
   tinting a light vector basemap toward a dark theme. Checked that the
   RAG line hues stay visually distinguishable after the filter (hue
   itself is unaffected by brightness/saturate/contrast, only perceived
   vividness) rather than assuming it would look fine.

**A real accessibility regression caught before shipping, the same way
the GOV.UK pass caught its amber/red tag-text failures**: `--ink-3` and
`--fog-3` (the "faint" tertiary text tokens) were first authored at 60%
and 55% lightness respectively, chosen by eye for looking appropriately
de-emphasised. Computing actual WCAG contrast (relative-luminance script,
not eyeballing - same method as 2026-08-31's fix) found `--ink-3` on
`--paper-0` was 3.68:1 and `--fog-3` on `--chassis-0` was 3.81:1 - both
below AA's 4.5:1, and both tokens are used on genuinely small text
(segment IDs, audit micro-labels), not large text where 3:1 would be
enough. Adjusted `--ink-3` 60%→55% lightness and `--fog-3` 55%→60%
lightness (moving each *away* from its background) - now 4.52:1 and
4.68:1 respectively. Every other text/background pair in the system was
checked the same way before shipping (see `DESIGN.md`'s "Verified, not
eyeballed, contrast" rule).

**Impeccable detector findings, triaged during the build (not deferred to
a follow-up pass)**: the mechanical detector flagged (a) Fraunces as an
overused "safe AI choice" display serif - swapped for STIX Two Text, an
academic/scientific-typesetting face that ties directly to the project's
publication ambition instead of just looking distinctive; (b) a bounce/
spring easing on the queue-selection sweep - replaced with a faster
exponential decelerate, no overshoot, per "real objects decelerate
smoothly"; (c) a `width`-based CSS transition on the score-breakdown bars
- switched to `transform: scaleX()` (GPU-composited, no layout thrash);
(d) a few literal colours/radii that had drifted outside the documented
token set - replaced with `color-mix()` derivations of real tokens or
existing tokens outright. One finding (`flat-type-hierarchy`) was left
standing: the detector runs in degraded/no-parser mode in this
environment and cannot resolve custom properties, so it measured a subset
of literal pixel values rather than the system's real 11px-60px scale;
flagged here rather than suppressed, in case a parser-capable run later
disagrees.

**Verified live in the browser, not just visually inspected**: re-used
the project's established `window.__debugMap` pattern (added, checked,
removed - never shipped) to confirm `roads-layer`'s and
`roads-selected-outline`'s actual compiled paint-property values, not just
that the map looked right. Confirmed the previously-fixed cross-panel
flows (pan-on-queue-select, scroll-on-map-select, borough-switch loading
overlay) still fire correctly under the new styling - a redesign that
silently broke a flow fixed hours earlier would be a real regression, not
a style choice. `npx tsc -b --force` clean; `pytest tests/
services/api/tests/` - 119 passed, 0 changed (frontend-only change set,
run anyway per this project's established verification habit).

**What stayed unchanged on purpose**: RAG risk semantics and the "no
estimate ≠ low risk" boundary, the product's information architecture
(three-column map/queue/evidence layout, all real data flows in
`App.tsx`), and every piece of existing copy voice (see
`opendesign/design-systems/fieldscope/brand/voice-and-tone.md`) - "full
creative departure" was scoped to visuals/motion/layout/copy-register per
the intake's own dimension question, not to product truth or working
flows, matching the user's explicit "keep in mind the flow and also the
backend works" instruction.

## 2026-09-01 - Data-richness pass on the UCL comparison: two real bugs found, one real gap closed, the headline number NOT moved

**User request** (verbatim): "lets get into the model stuff to get better
results then ucl reserch paper... make mine more data rich as well like as
much data possible" and a clarifying question about what the model
actually predicts. This entry is the full, honest account of that session
- it does NOT end with "we beat UCL," and says so plainly, because that
is not what happened. What it does contain is real: two genuine
correctness bugs found and fixed (with regression tests proving each
fix), one real methodological gap closed (feature normalisation), and a
properly-isolated measurement of what data enrichment alone is worth.

### What "predict" means here, for the record

Answered directly to the user first: the priority score (0-100) is a
*policy* layer (`product/priority_score.py`), not a prediction. The
prediction happens one layer down - XGBoost (annual grain, live in the
API) or the GAT+GRU (this document's subject, daily/14-day-ahead grain,
research-only) genuinely forecasts future collision counts from a
held-out future period the model never trained on. Improving "the model"
for the UCL comparison means improving that layer, not the scoring
formula - this is the layer everything below is about.

### The starting problem, confirmed by reading the code

`scripts/run_ucl_comparison.py`'s `FEATURE_COLUMNS` was `["length",
"day_of_week", "u_degree", "v_degree"]` - the daily/UCL-comparable model
had **zero access to any segment's own collision history**, despite the
segment-day table already containing it (used only as the *target*, never
as an input). Real-world crash risk is dominated by collision history
(the same locations tend to crash repeatedly); a model with no access to
it cannot rank well regardless of architecture. This was almost certainly
the single biggest driver of the previously-recorded AccHR@20 gap
(41.3% vs STZITD-GNN's 68.98%, `docs/publication_readiness.md`).

### What was built (all with new tests, run before trusting any number)

- `features/daily_features.py`: `collision_severity_counts_by_segment_day`
  (reuses the same per-collision severity aggregation the annual pipeline
  already trusts - `ingest/stats19.py`'s `collision_severity_and_
  vulnerable_user_features` - so annual and daily grain can never silently
  disagree on what counts as a "fatal casualty"), `attach_rolling_
  collision_features` (causal 7/14/30-day trailing sums, verified not to
  leak across segments or look forward in time), `attach_static_
  exposure_features` (AADF, averaged across available years per segment,
  broadcast across every day - a documented simplification, not a real
  daily exposure series, since AADF is annual-average by construction).
  5 new tests.
- `scripts/run_ucl_comparison.py`: `FEATURE_COLUMNS` grew from 4 to 16
  (raw + rolling collision counts, severity/vulnerable-user breakdown,
  AADF exposure) - every one of these was already-downloaded STATS19/DfT
  data, not a new external source. Extended `YEARS`/`START_DATE`/
  `END_DATE` from 3 years (2022-2024) to 5 (2021-2025) and reduced
  `STRIDE_DAYS` from 90 to 14 (== `HORIZON`, the densest non-overlapping
  stride), taking walk-forward training instances from 11 to 128 - one of
  this project's own previously-named candidate reasons for the AccHR@20
  gap.
- **Deliberately not done, and why**: STATS19 years before 2021 are only
  published bundled in DfT's full 1979-present file, not the clean
  per-year files this project's downloader targets - a bigger, riskier
  download (unknown size, unknown schema drift across decades) deferred
  rather than rushed. True per-segment IMD (via a real spatial join to
  LSOA boundaries) doesn't exist anywhere in this codebase yet - the
  annual pipeline's own IMD feature is a collision-history-anchored proxy,
  not a geographic one, and porting that same proxy to daily grain would
  be extremely sparse; flagged as a real future item, not silently
  skipped.

### Bug #1: CUDA OOM from the wrong place to batch gradients

128 training instances, each producing a full computation graph, were
being summed into one big loss before a single `.backward()` call
(`gat_temporal.py`'s `train_gat_temporal_walkforward`) - fine at the old
scale (~10 instances) but held every instance's graph in memory
simultaneously at the new scale, OOM-ing an 8GB GPU. **Fixed** by calling
`.backward()` once per instance and letting gradients accumulate on
`model.parameters()` across calls (`d(a+b)/dx = da/dx + db/dx` - PyTorch
frees each instance's graph immediately after its own backward call,
bounding peak memory to one instance instead of all of them). Proved
identical, not just "still runs": a new test
(`test_per_instance_backward_matches_summed_loss_backward`,
`tests/test_walkforward.py`) constructs two identically-seeded models,
trains one epoch each way, and asserts the accumulated gradients match to
1e-4 - calculus, not assumption.

### Bug #2: AccHR@20's own tie-handling was wrong

The first enriched run reported AccHR@20 = 89.13% (vs the recorded 41.3%
baseline) - large enough to interrogate before trusting, per this
project's own established "too good to be true" discipline. A quick
trivial-baseline sanity check (rank segments purely by raw historical
collision count, no model at all) scored a **vacuous 100%** - impossible
for a genuinely trivial rule to be perfect, which is what exposed the
bug: `accuracy_hit_rate`'s top-k selection used `y_pred[j] >= threshold`
(`eval/ucl_metrics.py`), where `threshold` is the k-th largest value.
With heavily zero-inflated integer/near-collapsed predictions, huge
blocks of segments tie exactly at the boundary value (often 0), and `>=`
lets *every* tied segment through - not just k of them. In the trivial-
baseline check, the boundary value was 0 and every score was
non-negative, so literally every segment "passed," 100% by construction,
proving nothing. **Fixed** by selecting exactly k segments via
`np.argpartition` + explicit index assignment, with ties beyond the k-th
position broken by argpartition's own arbitrary-but-deterministic order
(standard top-k evaluation convention) rather than silently admitting an
unbounded number of tied entries. 3 new regression tests, including one
that reproduces the exact large-tied-block failure mode.

This bug affects **every** AccHR@20 number this project has ever
reported, including the original 41.3% baseline - it had to be
recomputed, not just the new enriched result.

### Bug-adjacent gap #3: no feature normalisation anywhere in the GAT pipeline

Recomputing the enriched result under the fixed metric gave AccHR@20 =
21.74% - *lower* than the (also recomputed) 4-feature baseline's 43.48%,
which would mean richer data made the model worse. Investigated rather
than accepted: `aadf_all_motor_vehicles` measured 95-125,289 on
Westminster (mean ~22,939) versus collision-count/severity features in
the single digits and `length` in the low hundreds - a 4-5 order-of-
magnitude scale mismatch fed unnormalised into a neural network
(`GATConv` + `GRUCell` + linear layers, all scale-sensitive), something
that was survivable with the old 4 roughly-comparable-scale features but
became acute the moment AADF was added. **Nothing in this pipeline, old
or new, annual or daily, ever normalised features** - a real, previously-
latent methodological gap, not something the data-richness pass
introduced. Fixed with `features/daily_temporal.py`'s `fit_feature_
standardizer` (per-feature z-score, fit on TRAINING instances only -
never the held-out one, matching this project's walk-forward
no-leakage discipline) and `apply_feature_standardizer`. 3 new tests,
including one on the exact scale-mismatch scenario found here.

### The honest, final, fully-corrected numbers (Westminster, 2024-10-07 held-out window - the original comparison protocol, for direct comparability with the previously-recorded 41.3%)

| Configuration | AccHR@20 |
|---|---:|
| Original 4-feature baseline (metric bug fixed) | 43.48% |
| Enriched 16 features, **unnormalised**, metric bug fixed | 21.74% |
| Enriched 16 features, **normalised**, metric bug fixed | **39.13%** |

Normalisation recovered most of the gap the scale mismatch had opened
(21.74% → 39.13%), but the fully-corrected, properly-normalised enriched
model still does not clearly beat the simple 4-feature baseline on this
single held-out window (39.13% vs 43.48%) - a ~4-point difference on a
denominator of 46 total (day, segment) crash events across the 14-day
window is well within what a single noisy window can produce either way,
not a result either configuration should claim confidently. **This is
not the "we now beat UCL" result the earlier (bugged) 89.13% briefly
looked like, and reporting it as one would repeat exactly the mistake
this project has already caught itself making twice before with other
numbers.**

### What this session actually accomplished, stated plainly

1. Answered the user's real question (what does the model predict) and
   found the actual root cause of the previously-recorded gap (zero
   historical signal in the daily model's inputs) rather than guessing.
2. Closed that specific gap (features are now genuinely rich - severity,
   vulnerable users, exposure, rolling history, 5 years, dense walk-
   forward instances) with real, tested code.
3. Found and fixed two genuine correctness bugs (a GPU memory bug, and a
   metric implementation bug that had been silently affecting every
   AccHR@20 number this project has ever computed) and one real
   methodological gap (no feature normalisation) - each with tests, not
   just an assertion that it's fixed.
4. Measured the isolated effect of each change properly (a 2x2-plus
   ablation across window/features/normalisation/metric-correctness)
   instead of reporting one confounded before/after number.
5. **Did not move the headline AccHR@20 comparison against STZITD-GNN in
   this project's favour.** The gap this project needs to close to
   credibly claim "on par or better" than the UCL paper is still open.

### Real next steps (none started yet)

- **Evaluate across multiple held-out windows, not one.** 46 crash events
  in a single 14-day window is a genuinely small, high-variance sample -
  a single window's AccHR@20 (in either direction) is not a solid basis
  for a strong claim. The 128-instance dense walk-forward setup already
  built makes a rolling multi-window evaluation (average AccHR@20 across
  many held-out windows, with a spread/CI, not one point estimate)
  straightforward to add.
- **Extend to Lambeth** (the paper's second directly-relevant borough) to
  see if this pattern (rich features roughly matching a simple baseline)
  replicates, or is Westminster/window-specific.
- **Reconsider the "direct multi-horizon" decoder** against a real
  distributional (Zero-Inflated Tweedie-style) decoder now that the
  feature/normalisation confounds are out of the way - the original
  hypothesis that architecture (not data) was the limiting factor has
  neither been confirmed nor ruled out by this session's work.
- Statistical significance testing across whatever multi-window/multi-
  borough evaluation comes next (already a standing open item,
  `docs/methodology.md` §9).

## 2026-09-01 - "The research paper approach": reading Gao et al. (2024) in full, a real architecture win, and a real single-window measurement bug fixed

**User request** (verbatim, across several turns): "do reserch paper
aprroch if you want to i want to get better results then them at any
cost... read reserch paper first and get every minute insights", then
later "keep going dont stop till you get better or same results as ucl
one... you have whole internet access there is nothing stoping you." This
entry covers that whole investigation: re-reading the paper's full
Methodology section (not just its results table, already covered by the
earlier entries), a real bug found in how architecture candidates were
being compared, one genuine, replicated improvement, and an honest
accounting of everything else tried that didn't work.

### What re-reading the paper's Methodology (not just results) actually found

Extracted the full text of arXiv:2309.05072v4's Sections 3-4 and
Appendix A-D (previously only Section 4's results/dataset tables had been
read in depth). Concrete, checkable findings:

1. **The ZITD decoder's exact math** (Eq. 3-12): a genuine 4-parameter
   compound Poisson-Gamma (Tweedie) distribution - crash *count*
   `C_k ~ Poisson(lambda)`, per-crash *severity* `l_k ~ Gamma(alpha,gamma)`
   i.i.d., and the target `y_k` is the Poisson-stopped sum of those Gamma
   draws (Appendix B, Eq. B.1) - mathematically confirming, independently
   of the earlier "the paper's own words say severity-weighted" finding,
   that their target structurally cannot be a plain count: it's built by
   construction as count-times-severity.
2. **"N=20" almost certainly means training epochs, not input window
   length.** The paper defines `N = |V|` (number of roads, in the
   thousands per their own Table 3) earlier in the SAME paper (Section
   3.1) - "N=20" cannot coherently be a road count. Reading it as
   `N_epoch=20` (a garbled/OCR'd subscript) is far more consistent, and
   matches the predecessor paper's own convention (next point). This
   project's `INPUT_WINDOW=20` was accordingly never actually a matched
   value to the paper's own setup, contrary to this pipeline's original
   docstring assumption.
3. **A predecessor paper by the same core authors, found via web search**:
   Gao, Haworth, Zhuang et al., "Uncertainty Quantification... by
   STZINB-GNN" (arXiv:2307.13816, one generation earlier, same GRU->GAT
   architecture, ZINB instead of ZITD decoder). Two critical findings from
   it:
   - **Its own input/horizon convention is SYMMETRIC**: "Long(14-14)" and
     "Short(7-7)" are the only two configurations it reports - never an
     asymmetric 20-in/14-out split. Tested directly (see below):
     `input_window=14` performed *worse* than this project's original
     `input_window=20` guess, so the guess was kept, but the test was run
     rather than assumed.
   - **Its "Hit Rate (HR20%)" is a DIFFERENT metric from the later paper's
     "AccHR@20"**, despite the near-identical name: HR20% is entropy/
     uncertainty-threshold-based ("select the top 20% by predicted risk,
     then further filter to those whose predicted-risk entropy is below
     the network-wide mean"), not "does an actual crash fall in the top
     20%" (the later paper's actual, prose-stated AccHR@20 definition,
     which this project's `ucl_metrics.accuracy_hit_rate` correctly
     implements). Their reported 61.8%/57.5% HR20% figures are **not**
     comparable to this project's AccHR@20 numbers or to the later paper's
     own 68.98%/76.59% - a real, easy-to-miss trap from two similarly-named
     metrics across two papers by overlapping authors, now on record so no
     future session conflates them.
4. **A public repo (`github.com/STTDAnonymous/STTD`) turned out to be a
   dead end for the crash-specific architecture**: it implements the
   Tweedie/NB/Gaussian *decoder* framework (confirming `nhid=42`,
   revealing `lr=1e-3`/`weight_decay=1e-4` in its actual code - more
   trustworthy than the possibly-OCR-garbled "0.01"/"0.01" read from the
   PDF text) but its spatial/temporal encoder is `D_GCN`+`B_TCN`
   (diffusion graph convolution + bidirectional temporal conv), **not**
   GAT+GRU, and its `Accident_risk` folder (where crash-specific code
   would live) is an empty 1-byte stub. The paper's claim "our code is
   available on GitHub" does not appear to include a working release of
   the actual GAT+GRU crash model - disclosed here so a future session
   doesn't waste time hunting for it again.

### A real bug found: single-window AccHR@20 is too coarse to compare architectures

Four meaningfully different configurations (ZIP heads=1, ZINB heads=1,
ZIP heads=3, ZIP 2-layer) all produced the *identical* AccHR@20 (18/46)
on the one held-out window `run_ucl_comparison.py` evaluates - suspicious
enough, per this project's "too consistent to be right" discipline, to
check directly rather than shrug it off. `scripts/diagnose_architecture_
sensitivity.py` confirmed the four models' raw predictions are genuinely
different (pairwise correlation as low as 0.67, standard deviations
differing by up to 65%) - not a wiring bug. The real explanation: with
only 46 actual crash events in the denominator, `accuracy_hit_rate` only
has 47 possible values (0/46...46/46) - far too coarse a ruler to resolve
real differences between models that broadly agree on which roads are
"busy" but disagree on the finer ranking.

**Fixed** by building `scripts/run_ucl_comparison_multiwindow.py`: a
genuine expanding-window walk-forward evaluation (train on every instance
strictly before each of the last 6 held-out windows, evaluate on it,
report mean+std across all 6) instead of one point estimate. This is now
the standard way to compare architecture candidates on this pipeline -
every number in the table below comes from it, on the original 2022-2024/
quarterly-stride protocol (12 total instances) specifically because that
config's fast (~1-2min/window) training makes 6-window sweeps of many
candidates tractable, unlike the dense 128-instance 2021-2025 config.

### The full sweep (Westminster, 6 expanding-window held-out periods each)

| Configuration | Mean AccHR@20 | Std | Verdict |
|---|---:|---:|---|
| baseline (heads=1, 1 layer, ZIP - prior config) | 42.25% | 3.76% | reference |
| **heads=3 (Gao et al.'s own value)** | **49.57%** | 6.60% | **best - adopted as new default** |
| 2-layer GAT (heads=1) | 46.96% | 6.01% | modest win alone |
| heads=3 + 2-layer (combined) | 41.67% | 12.89% | worse than heads=3 alone, high variance |
| heads=3 + hidden=42 + lr=1e-3 + wd=1e-4 | 20.55% | 11.52% | badly undertrained at 10x lower lr, unchanged epochs |
| heads=3 + input_window=14 (symmetric, predecessor paper) | 42.31% | 7.89% | worse than input_window=20 |
| heads=3 + epochs=500 (vs 200) | 43.93% | 5.33% | overfits with no early stopping |
| heads=3 + weight_decay=1e-4 (isolated from the lr/hidden confound above) | 47.09% | 5.32% | neutral, slightly worse |
| heads=3 + `pairwise_rank_hinge_loss`(w=0.1) | 48.29% | 7.27% | neutral (within noise) |
| heads=3 + `pairwise_rank_hinge_loss`(w=1.0) | 40.76% | 3.31% | hurts - noisy gradient from too few positives/window |
| **heads=3, validated on Lambeth (2nd borough)** | **47.31%** | 5.29% | **confirms the heads=3 win generalises, not a Westminster fluke** |

**heads=3 alone is the only lever that produced a real, replicated
improvement.** Every other paper-matched value tried, once actually
measured this way rather than assumed, either did nothing (weight decay,
light ranking loss) or made things measurably worse (more capacity
combined with a lower LR at the same epoch budget, more epochs with no
early stopping, a shorter symmetric window, a heavier ranking loss).
`GAT_HEADS`'s default in `scripts/run_ucl_comparison.py` is updated
3 (from 1) to reflect this - this is now the project's actual best-known
configuration, not a one-off experiment result sitting in a log file.

### New reusable code from this pass (all tested)

- `models/gat_temporal.py`: `zero_inflated_negative_binomial_nll` (a
  genuine second decoder distribution - see its docstring for why ZINB,
  not the paper's own Tweedie, is the architecturally honest choice for
  this project's plain-count target), `pairwise_rank_hinge_loss` + a
  `rank_loss_weight`/`rank_margin` option on `train_gat_temporal_
  walkforward` (a learning-to-rank auxiliary loss, tested and found not
  to help - see table above, but now available for a future session to
  revisit with a different weight/margin without re-implementing it), a
  real 2-layer GAT option (`gat_layers`), `gru_hidden` finally exposed as
  a `train_gat_temporal_walkforward` parameter (was hardcoded).
- `eval/ucl_metrics.py`: `accuracy_hit_rate`'s top-k selection fixed from
  a tie-inclusive `>=threshold` mask to an exact-k `argpartition` -
  affects every AccHR@20 number this project has ever computed, including
  the "41.3%"/"89.13%" figures now superseded (see this file's own prior
  entry). 3 new regression tests, including the exact large-tied-block
  failure mode that exposed the bug.
- `scripts/run_ucl_comparison_multiwindow.py`: the new standard multi-
  window evaluation harness - use this, not the single-window
  `run_ucl_comparison.py`, for any future architecture comparison.
- `scripts/diagnose_architecture_sensitivity.py`,
  `scripts/diagnose_ucl_accHR.py`: kept as real, reusable diagnostic
  tools (not deleted after use) - both encode a specific "is this result
  actually real" check this project has now needed twice.

### What this does and doesn't mean for the publication ambition

The gap to STZITD-GNN's AccHR@20 (68.98% Westminster, 76.59% Lambeth) has
genuinely narrowed - from ~41-43% to ~48-50%, closing roughly a third of
it, via one specific, correctly-sourced, twice-replicated architecture
correction. It has **not closed**. The honest remaining candidates,
in rough order of expected effort-to-value, none started:
1. **A true early-stopping/validation-split training loop.** This
   pipeline still trains a fixed epoch count with no held-out validation
   signal during training (only at final evaluation) - the epochs=500
   result above is exactly the failure mode early stopping exists to
   prevent, and it's plausible 200 is also not the right number, just the
   better of two arbitrarily-chosen options.
2. **Actually reconstructing a severity-weighted target** (matching the
   paper's real task, not a disclosed-different one) - a bigger, riskier
   change (redefines what the model predicts) but would remove the
   single largest remaining "is this even the same task" objection this
   project's own documentation has maintained throughout.
3. **A full compound Poisson-Gamma (true Tweedie) decoder** - now that
   ZINB has been tried and roughly matched ZIP (see the earlier entry in
   this file), the paper's own decoder family hasn't actually been
   implemented, only approximated.
4. Statistical significance testing across the 6-window (or more)
   distribution now available, rather than reporting mean/std informally
   as done in the table above.

## 2026-09-01 - "Think, think, think": the full PhD thesis, the exact TCR target, and early stopping - three real levers tried, one honest negative-that-clarifies

**User request** (verbatim): "do more research you can do better like find
deep like read books more reserch paper like do eevrything to make it
best this is very imp please dont hold back you have all freedom all gpu
and everthing think think think." This entry covers the deeper research
pass that followed: reading Gao's full 339-page UCL PhD thesis (not just
the journal paper), reconstructing the paper's *exact* target definition
from it, and building a real early-stopping mechanism - all tested via
the multi-window methodology, all numbers on the record whether they
helped or not.

### The PhD thesis (discovery.ucl.ac.uk/id/eprint/10210801) resolved the remaining hyperparameter ambiguity

The journal PDF's text extraction had mangled a subscript into "N=20";
the thesis PDF preserved it: **"N_epoch=20... GAT head is set to 3...
The GNNs in STZITD-GNNs and baselines are all two-layered. The hidden
units are set to 42."** Confirms, unambiguously: 20 training epochs (not
an input window - `N` is defined elsewhere in the same document as the
road count, in the thousands, so "N=20" could never have meant that), a
genuine 2-layer GNN, hidden dim 42, lr=0.01, weight_decay=0.01 - the
project's OWN prior "corrected lr/wd" test (20.55% mean, docs/
decision_log.md's earlier entry) had used lr=1e-3/wd=1e-4 from a
*different* model in a sibling repo, a real confound now resolved.

Retested properly, isolating each variable:
- `epochs=20` (the paper's literal value) on this project's own feature
  set: **14.79% mean** - badly underfits. The paper's epoch count was
  tuned for their own (different, smaller) feature/architecture
  combination and does not transfer blindly.
- `epochs=200` (this project's usual budget) with the full paper-matched
  capacity (2-layer, hidden=42, lr=0.01, wd=0.01, heads=3): **46.44%
  mean** - properly trained this time, but still slightly *below*
  `heads=3` alone (49.57%, 1-layer, hidden=16, no weight decay). The
  additional capacity/regularisation does not help this pipeline's data
  regime even once correctly tuned. `heads=3` alone remains the single
  best, most parsimonious configuration found - the architecture
  question is now closed with real evidence on both sides, not left
  ambiguous.

### The exact TCR target, reconstructed and tested - a real fix that didn't move the number it was expected to

The thesis's Definition 1 (Eq. 7.1) gives Gao et al.'s target formula
exactly: `y_it = sum_k C^t_{i,k} * l_k`, l_k in {1,2,3} weighting
minor/serious/fatal crashes. This is fully reconstructible from data
already downloaded - STATS19's own `collision_severity` field (1=Fatal,
2=Serious, 3=Slight; weight applied = `4 - collision_severity`, verified
against `data/raw/collision-2024.csv`'s real value distribution) - no new
source needed. Implemented as `daily_features.py`'s `tcr_score` column
(computed alongside the existing severity/vulnerable-user aggregation,
2 new tests), used as the TARGET while `collision_count` and the
severity breakdown stay available as INPUT features.

**Result** (`scripts/run_ucl_comparison_multiwindow_tcr.py`, heads=3,
6 windows): AccHR@20 mean **50.17%** (std 7.10%) - statistically
indistinguishable from the plain-count target's 49.57%. The
target-definition mismatch, real and worth fixing (MAE/MAPE/RMSE/PICP
are now genuinely comparable to the paper's Table 4 for the first time,
where every earlier entry in this file had to disclaim them as
incommensurable), turns out NOT to explain the AccHR@20 gap. This is
useful to know precisely because it rules out a plausible-sounding
hypothesis with an actual measurement, rather than leaving it as an
unresolved asterisk.

**A related discovery, disclosed but not implemented**: this project's
measured zero-inflation (99.96%) is far higher than the paper's own
(95.72% Westminster). Re-reading the predecessor STZINB-GNN paper's Data
Description found why: their target incorporates "spillover effects on
first and second-order neighbouring roads" - a road with no crash of its
own but adjacent to a crash gets a nonzero score too, mechanically
lowering zero-inflation. Neither paper states the exact spillover
formula (decay function, aggregation rule), so implementing a guessed
version risks misrepresenting their method rather than replicating it -
flagged here as a real, disclosed, unresolved structural difference
rather than silently worked around or invented.

### Early stopping - implemented properly, hurts in this specific data regime, and the reason why is itself informative

Motivated directly by this project's own within-session evidence
(`epochs=20` underfits, `epochs=200` on richer capacity is fine,
`epochs=500` overfits - three data points on the SAME architecture/data
strongly suggesting a real sweet spot a fixed epoch count can only
guess at). Implemented in `train_gat_temporal_walkforward` via
`early_stopping_patience`: holds out the LAST training instance as a
validation set, tracks validation loss per epoch, stops after `patience`
epochs without improvement, restores the best-validation-loss weights
(not whatever epoch happened to run last). 4 new tests (multi-step,
single-instance fallback, standard smoke tests).

**Result** (heads=3 + `early_stopping_patience=10`, `epochs=300` ceiling,
6 windows): AccHR@20 mean **26.37%** (std 8.59%) - clearly *worse* than
the fixed-200-epoch heads=3 (49.57%), and each window trained in ~8-10
seconds instead of ~1-2 minutes, meaning patience was exhausted almost
immediately every time. The likely cause, not just an unexplained
regression: at this pipeline's own small scale (5-10 instances *after*
one is carved out for validation), a single held-out instance's loss is
an extremely noisy estimate of "true" generalisation - noisy enough that
it fails to improve by chance well before the model has actually
converged, triggering premature stopping. Early stopping is a real,
correctly-implemented, generally-sound technique that is simply the
wrong tool for a dataset this small with only a single-instance
validation split - a k-fold-style validation signal (average loss across
several held-out instances, not one) would be a more appropriate next
attempt if this lever is revisited, not evidence that early stopping
itself was a bad idea to try.

### What today's deeper pass changed, and what it didn't

**Confirmed, not changed**: `heads=3` (1-layer, hidden=16, no weight
decay, 200 fixed epochs, `collision_count` target) remains this
project's best-known configuration - AccHR@20 49.57% Westminster / 47.31%
Lambeth vs the paper's 68.98%/76.59%. Three independent, well-motivated
attempts to beat it today (paper-matched capacity, the paper's exact
target, and early stopping) each failed on their own honest terms, not
from lack of trying - each is now closed off with real evidence rather
than left as a plausible-sounding untried idea.

**Genuinely added, independent of whether it moved AccHR@20**: `tcr_score`
makes this project's MAE/MAPE/RMSE/PICP directly comparable to the
paper's Table 4 for future work, and `early_stopping_patience` is
available (and correctly implemented) for any future pipeline with
enough training instances per window for it to have a fair chance -
likely the dense 128-instance 2021-2025 configuration, not the 12-instance
original-window one every test in this entry ran on.

**Real remaining candidates, in order, none started**: (1) the spillover/
spatial-smoothing target reconstruction (needs an explicit, disclosed
methodological choice about the exact diffusion rule, since neither
paper states one precisely), (2) a genuine k-fold-style validation split
for early stopping instead of a single held-out instance, (3) the full
compound Poisson-Gamma (true Tweedie) decoder, (4) statistical
significance testing on the multi-window distributions now available,
(5) extending every one of today's tests to Lambeth and to the dense
128-instance configuration, neither of which has been re-checked against
today's specific findings yet.

## 2026-09-01 - Weather features (a real gap vs. the paper's own inputs, closed) and statistical significance testing on the multi-window distributions

**User request** (verbatim, continuing the same "think, think, think"
session): "continue what you are doing rn now i wnat better results
then ucl reserch paper dont stop till you get better results like find
every way iaginable reserch lot of stuff if you wanna change pipeline
or tweak it you have all fredom no holding back i want best." Two of
the "real remaining candidates" from the previous entry were picked up:
a genuinely new data source (weather), and statistical rigor on the
comparisons already made.

### Weather: closes a real, disclosed gap vs. the paper's own feature set - result is neutral, not a win

Gao's thesis (Table 7.2, "Data Characteristics by Region") lists
"Meteorological Characteristics" (Met Office, 8 classes) as part of
STZITD-GNN's own input features for every region studied. This
project's pipeline had *zero* weather signal before today - not an
assumption that weather doesn't matter, a genuine hole. Closed via
`greyspot.ingest.weather` (new module): real daily London weather
(temperature, precipitation, snowfall, windspeed, humidity, sunshine
duration) from Open-Meteo's free historical archive (no API key,
matching this project's established free/key-less source pattern -
Met Office's own API needs registration) - downloaded and verified
(1826 days, 2021-01-01 to 2025-12-31, zero nulls) to
`data/raw/weather/london_daily_weather_2021_2025.json`. One London-wide
series is broadcast onto every segment via
`attach_daily_weather_features` (day-level join, not segment-level -
weather does not vary meaningfully across boroughs a few km apart,
unlike genuinely local features like AADF).

Result (`heads=3` + 6 weather columns + `has_weather` flag, same
multi-window protocol, Westminster): **AccHR@20 47.81%** (std 7.95%,
range 34.78%-61.54%) vs. `heads=3` alone's 49.57%. A small decrease,
well within noise - weather is neutral-to-slightly-negative here, not
a lever that moves this project's result. Kept in the codebase (real,
tested, disclosed) as a completed, honest negative rather than reverted,
since it closes a real gap against the paper's own reported inputs
regardless of whether it helped this project's specific numbers.

### Statistical significance testing on `heads=3` vs `heads=1` and vs the 2-layer/hidden=42 config

The 6-window paired AccHR@20 series behind the "heads=3 wins" and
"2-layer underperforms" claims (both reported in earlier entries as
plain means) had never been tested for whether the difference is
distinguishable from noise at n=6 windows. Ran three standard paired
tests (`scipy.stats`) on the actual per-window values:

**`heads=3` (mean 49.57%) vs `heads=1` (mean 41.59%)** - per-window
values: `[0.596,0.492,0.517,0.441,0.537,0.391]` vs.
`[0.423,0.373,0.448,0.412,0.488,0.391]`. `heads=3` wins 5 of 6 windows,
ties exactly on the 6th (both 0.391304 - the same window, same held-out
day count, coincidentally identical hit count).
- Paired t-test: t=2.838, **p=0.0363** - significant at α=0.05.
- Wilcoxon signed-rank: p=0.0625.
- Sign test (5 favour heads=3, 0 favour heads=1, 1 tie): p=0.0625.

Reported honestly, not cherry-picked: the two nonparametric tests land
just above the conventional 0.05 line, but 0.0625 is the *minimum
p-value either test can produce* with only 5 non-tied pairs all
pointing the same direction (2^-5 = 0.03125, doubled for a two-sided
test) - a small-sample floor, not weak or ambiguous evidence. All three
tests agree on direction and magnitude of confidence; the parametric
test crosses 0.05 because it also uses the *size* of each window's
margin (heads=3 wins some windows by a wide margin, e.g. 0.596 vs 0.423),
information the sign/rank tests discard. Conclusion: `heads=3`'s
improvement over `heads=1` is real and consistent across windows, not
a fluke of one lucky split - the strongest statistical statement this
project can currently make about any architecture choice.

**`heads=3` (mean 49.57%) vs the thesis-matched 2-layer/hidden=42/lr=0.01/
wd=0.01 config (mean 46.44%)** - per-window diffs
`[0.019,0.000,0.035,0.029,0.073,0.000]` (heads=3 never loses a window,
ties twice). Paired t-test: t=2.344, **p=0.0661** - just short of
significance at α=0.05. Weaker evidence than the heads=1 comparison
(two exact ties pull the effect size down, and the wins are individually
smaller), but the direction is unanimous (0 losses in 6 windows) and
consistent with the effect being real but harder to resolve at this
sample size - not a reason to doubt the "heads=3 alone beats the
paper's exact capacity settings on this project's data" conclusion,
just a note that it does not clear the conventional bar as cleanly as
the heads=1 comparison does.

**Practical upshot**: no change to the project's best-known configuration
(`heads=3`, 1-layer, hidden=16, no weight decay, 200 epochs, plain-count
target, 49.57%/47.31% Westminster/Lambeth) - this pass adds statistical
backing to a choice already made, and closes the weather-gap candidate
as a tested, honest negative. Both `docs/publication_readiness.md` and
the project memory file now cite these significance results as the
standard to hold any future architecture claim to.

### k-fold-style early stopping: the diagnosed noise-reduction fix, tried, and it does NOT recover the gap either

The previous entry's diagnosis for why plain early stopping hurt (26.37%
vs 49.57%) was that ONE held-out validation instance is too noisy a
signal at 5-10 training instances per window. The natural fix -
`early_stopping_val_instances` (new parameter, `gat_temporal.py`) holds
out the LAST N instances instead of one and validates against their
MEAN loss, k-fold-style, without the expense of training N separate
models - was implemented (4 new unit tests: multi-instance holdout,
graceful fallback when too few instances exist, both passing) and
tested at N=2 and N=3 on the same original 12-instance protocol used to
find the original failure:

- `val_instances=2`: mean **25.09%** (std 5.77%)
- `val_instances=3`: mean **27.80%** (std 3.84%)

Both remain far below the fixed-200-epoch baseline (49.57%), and are
roughly in line with (not meaningfully better than) the original
single-instance result (26.37%) - `val_instances=3` is marginally
higher than both, a weak hint of the diagnosed direction, but nowhere
close to closing the gap. **Conclusion, now closed rather than left
open**: the noise-reduction benefit of averaging more validation
instances does not outweigh the cost of removing them from an
already-tiny training set at this protocol's scale (6-11 instances per
window total) - holding out 2-3 of that for validation starves training
more than it stabilises the stopping signal. Early stopping in any
form tried so far is not viable at this data scale; it remains a
plausible candidate specifically for the dense 128-instance
configuration (30-127 training instances per window), where holding out
even several instances for validation leaves a training set an order
of magnitude larger than this protocol's total - untried, and now the
correct next place to test it if this lever is revisited.

### The spillover/spatial-smoothing target: implemented as a disclosed, explicit reconstruction (not yet evaluated)

The last-remaining "real candidate" from the previous entry -
STZINB-GNN's target incorporating "spillover effects on first and
second-order neighbouring roads" (neither paper states the exact decay
function or aggregation rule) - now has a concrete, disclosed
implementation: `attach_spillover_target` (`daily_features.py`), which
reuses the SAME line-graph adjacency already built for the GAT itself
(not a second, invented neighbour notion) and computes
`spillover = own + 0.5 * sum(1st-order neighbours) + 0.25 * sum(2nd-order
neighbours)` - explicit, disclosed default decay weights (halving per
hop, the common spatial-smoothing convention when no paper-specified
value exists), not a claimed replication of Gao's precise formula. The
"2nd-order" term is a standard but impure two-hop count (`A @ (A @ own)`,
does not deduplicate paths that revisit a 1st-order neighbour or the
segment itself) - flagged explicitly in the function's own docstring
rather than glossed over. 4 new unit tests (hand-computed 3-segment
chain, zero-weights-reduces-to-own-count, isolated-segment-unaffected,
missing-column error), all passing.

**A real, directional confirmation, computed before any AccHR training
finished**: applying spillover to `tcr_score` (Westminster, 2022-2024
protocol) moves zero-inflation from **99.96% to 99.08%** - real
movement toward the paper's own reported 95.72%, in the correct
direction, though not all the way there. This confirms the spillover
mechanism does what the predecessor paper's Data Description implies it
should (a road with no crash of its own but adjacent to one stops
counting as a true zero) - the remaining gap to 95.72% is plausibly
explained by some combination of a different decay function, a
genuinely sparser road network (OSMnx's line graph vs. whatever network
Gao's team used), or third-order-and-beyond spillover this
implementation doesn't attempt - not investigated further here, flagged
as the natural next question if this lever is revisited.

`run_ucl_comparison_multiwindow_spillover.py` (new sweep script, applies
spillover on top of `tcr_score` as `TARGET_COL="tcr_score_spillover"`,
same disclosed 0.5/0.25 decay) launched against the original 12-instance
protocol - running at the time of this entry, result in the next
entry.

### Dense (2021-2025, stride=14, 128 walk-forward instances) + `heads=3`: completed, and a real methodological caveat about what this number can and can't tell us

Result: **AccHR@20 = 31.25%** on the single held-out window (MAE 0.0007,
RMSE 0.0174, PICP 0.9000 - exactly hitting the 90% nominal conformal
target again). Took ~25 minutes for this one window alone (128 training
instances, heads=3, 200 epochs) - the retry ran cleanly this time
(60-98% GPU util, peak ~4.5GB/8GB VRAM including the concurrent
spillover sweep below, no OOM).

**This number is NOT directly comparable to the 49.57% multi-window
mean** that is this project's current headline figure, for exactly the
reason this project's own multi-window methodology exists in the first
place (see the "research paper approach" entry): 49.57% is a MEAN
across 6 held-out windows on the light (2022-2024, 12-instance)
protocol; 31.25% here is a SINGLE window on the dense (2021-2025,
128-instance) protocol. Two different things vary at once (protocol
density AND single-vs-multi-window noise) - a lower single number does
not mean "more data hurts," it means this specific comparison is
confounded and cannot support that conclusion either way. The
methodologically correct next step - a genuine 6-window expanding
multi-window sweep at the DENSE data density - would require roughly
6 separate trainings at up to ~128 instances each (very roughly
1-2 hours total, extrapolating from this single 25-minute window), a
real, disclosed compute commitment, not yet run. Recorded here as an
honest single data point, not as a conclusion about data density.

### Spillover-on-TCR sweep result: closes zero-inflation gap toward the paper, but clearly HURTS AccHR@20

`run_ucl_comparison_multiwindow_spillover.py` (heads=3, TCR+spillover
target, 6-window protocol) result: **AccHR@20 mean = 34.57%** (std
4.87%, MAE 0.0184, PICP 0.9025) - clearly below both the plain-count
target's 49.57% AND the plain-TCR target's 50.17%. Every one of the 6
windows scored lower than its plain-TCR counterpart on the identical
held-out period (e.g. 2023-07-15: 35.54% spillover vs 61.54% plain-TCR;
2024-04-10: 28.73% vs 44.12%) - a consistent, not noisy, direction.

**A real tension, disclosed rather than smoothed over**: spillover
succeeds at its OWN stated goal (moving zero-inflation from 99.96% to
99.08%, toward the paper's reported 95.72% - see the earlier entry
above) while clearly making the ranking task *harder*, not easier. A
plausible, disclosed explanation (not verified further here): diffusing
each crash's severity score onto its 1st/2nd-order neighbours blurs
what were sharp, spatially concentrated risk signals into wider "hot
blobs" spanning several neighbouring segments - AccHR@20 needs the
model to pick out the EXACT top-20% highest-risk segments, so smoothing
the target makes many previously-distinguishable segments look nearly
identical, working directly against precise top-k ranking even while it
helps the target's zero-inflation "look more like the paper's."

**Conclusion**: the spillover reconstruction is closed as a real,
tested, disclosed negative for this project's specific ranking metric -
not reverted from the codebase (`attach_spillover_target` stays, tested,
documented, and available for future use, e.g. if MAE/RMSE-style metrics
rather than AccHR@20 become the priority). The project's best-known
configuration for AccHR@20 remains unchanged: `heads=3`, plain-count
target, 49.57%/47.31% (Westminster/Lambeth).

### The full compound Poisson-Gamma (Tweedie) decoder: implemented and unit-tested, grounded in the paper's own Eq. 3-6 (not yet run on real data)

The paper's PDF text (pages 9-12) was extracted directly (not
reconstructed from memory) to get its EXACT Zero-Inflated Tweedie (ZITD)
formulation before implementing anything - `zero_inflated_tweedie_nll`
(new, `gat_temporal.py`) uses the paper's own closed-form y=0 mass
(Eq. 6: `P(y=0|not zero-inflated) = exp(-mu^(2-rho)/(phi*(2-rho)))`,
exact) combined with the standard Tweedie GLM training convention (unit
deviance) for y>0, since the exact positive-value density has no closed
form (an infinite series, Dunn & Smyth 2005) - disclosed explicitly in
the function's own docstring, not glossed over. `rho` is a fixed
hyperparameter (default 1.5) rather than the paper's own 4th learned
parameter - a genuinely differentiable learned-rho version was judged a
substantially larger, riskier undertaking than this pass attempts.

Wired into `GATTemporal` (`tweedie`/`tweedie_rho` params, a third
`phi_gate` decoder head, mutually exclusive with `negative_binomial`)
and `train_gat_temporal_walkforward` exactly like the existing ZIP/ZINB
decoders. 6 new unit tests (hand-computed NLL for y=0 and y>0, prefers-
correct-pi, rejects rho outside (1,2), forward-returns-triple,
end-to-end train+predict, negative_binomial+tweedie-together raises)
- all passing (164 tests total).

**Run through the multi-window AccHR@20 evaluation immediately after**
(`run_ucl_comparison_multiwindow_tweedie.py`, TCR target, same 6-window
protocol): **AccHR@20 mean = 45.76%** (std 6.55%, MAE 0.0009, RMSE
0.0248, PICP 0.8967 - again close to the 90% nominal conformal target,
a decoder-independent calibration property, not specific to Tweedie).

**The full comparison, same target family, same architecture
(heads=3), only the decoder distribution changed:**

| Decoder | Target | AccHR@20 mean |
|---|---|---|
| Zero-Inflated Poisson (ZIP) | `collision_count` | 49.57% |
| Zero-Inflated Poisson (ZIP) | `tcr_score` | 50.17% |
| **Zero-Inflated Tweedie (ZITD, paper's own decoder)** | `tcr_score` | **45.76%** |
| Zero-Inflated Tweedie + spillover | `tcr_score_spillover` | 34.57% |

**Conclusion, stated plainly**: implementing the paper's OWN reported
decoder distribution, as faithfully as this project could without
guessing at an untractable series expansion, does NOT beat this
project's simpler ZIP decoder on either target - it lands a few points
below both. This is a genuinely informative negative result for the
publication angle: the AccHR@20 gap between this project and Gao et
al.'s reported 68.98%/76.59% is evidently NOT explained by "using the
wrong decoder distribution" - four different, reasonable decoder/target
combinations now cluster in the low-to-mid 40s-50% range regardless,
while the paper's own reported number sits nearly 20 points higher.
Whatever explains that gap, it is something more structural than the
loss function alone (candidates still on the table: their spillover
formula being different from this project's disclosed guess in some
consequential way, their road network/feature set being richer or
different in kind, or a genuine methodological difference in how their
own 68.98%/76.59% was measured that this project's careful walk-forward
replication does not share).

The project's best-known configuration for AccHR@20 remains unchanged:
`heads=3`, ZIP decoder, plain-count target, 49.57%/47.31%
(Westminster/Lambeth).

### Lambeth validation: the "Tweedie doesn't beat ZIP" finding replicates on a second borough

`run_ucl_comparison_multiwindow_lambeth_tcr.py` re-ran both the ZIP+TCR
and ZITD/Tweedie+TCR candidates on Lambeth (same 6-window light
protocol) - the same generalisation check this project's own
methodology already applied to the heads=3 finding.

| Decoder | Target | Westminster AccHR@20 | Lambeth AccHR@20 |
|---|---|---|---|
| ZIP | TCR | 50.17% | **48.72%** (std 5.03%) |
| ZITD/Tweedie | TCR | 45.76% | **44.36%** (std 4.62%) |

Same direction, both boroughs: ZIP beats Tweedie by ~4-5 points on
each. This is a genuine, cross-borough-confirmed finding, not a
Westminster-specific fluke - "the paper's own decoder distribution does
not explain the AccHR@20 gap" now rests on evidence from two
independent boroughs, not one. PICP also held steady (0.8877/0.8951 on
Lambeth, both close to the 90% nominal target) - the good conformal
calibration is a general property of this pipeline, not a Westminster
artifact either.

### The dense config's real multi-window result, finally: mean is LOWER than the light protocol, but with a real, disclosed confound this specific sweep design has

`run_ucl_comparison_multiwindow_dense.py` (heads=3, dense 2021-2025/
stride=14 protocol, 3 held-out windows, ~126-128 training instances
each - see the earlier "still running" entry for the compute-time
disclosure) result:

| Window (held-out start) | Training instances | AccHR@20 |
|---|---|---|
| 2025-11-20 | 126 | 53.85% |
| 2025-12-04 | 127 | 45.24% |
| 2025-12-18 | 128 | 31.25% (matches the earlier single-window run exactly - a good internal-consistency check, same config/seed/held-out window) |

**Mean = 43.44%, std = 9.31%** - lower than the light protocol's 49.57%
(std 6.60%), and with noticeably higher variance despite roughly 15x
more training data per window.

**A real, disclosed limitation of this specific sweep design, not
just "more data doesn't help"**: the light protocol's 6 windows are
spread across 2023-2024 at 90-day (roughly quarterly) intervals -
genuinely diverse points in the calendar and across two different
years. The dense protocol's stride is fixed at 14 days (matching the
paper's own horizon, and `run_ucl_comparison.py`'s own default), so
holding out the LAST `N_WINDOWS=3` instances necessarily means 3
windows clustered within a single 4-week span at the very end of the
dataset (2025-11-20 to 2025-12-18) - late autumn/early winter of one
specific year, not 3 independent, temporally-diverse samples. The
53.85%-to-31.25% spread across barely a month could easily reflect a
real seasonal or short-term effect (holiday-period traffic patterns,
a specific weather event, end-of-year data completeness) rather than
genuine model instability or a real density effect - this sweep cannot
distinguish those explanations from each other, and does not claim to.

**What can and can't be concluded**: PICP stayed stable and close to
90% across all three dense windows (0.9028/0.8959/0.9000), same as
every other configuration tested this session - the good conformal
calibration is confirmed as a general pipeline property, independent of
data density. On AccHR@20 specifically, this result does NOT show
denser data clearly helps, and its own within-window variance (9.31%
std on just 3 points) means it does not clearly show it hurts either -
an honest "inconclusive, with a real confound identified" rather than a
clean result either way. A genuinely conclusive answer would need
either more held-out windows at this density (expensive - each costs
~30 minutes) spread across more of the calendar, or a stride large
enough to decorrelate held-out windows in time while still using the
dense feature-richness - neither attempted here.

## 2026-09-02 - Found and used the paper's own linked reference code: `zero_inflated_tweedie_nll`'s y>0 approximation upgraded from a generic GLM substitute to the authors' own choice

**User request** (verbatim): "you can use ucl paper pipeline amd all all
permission." Read as explicit permission to look for and use the
paper's own reference implementation, if one exists, rather than
relying solely on the PDF text and this project's own from-scratch
derivations.

**Found**: the ScienceDirect version of the paper states plainly, in
Section 3.4.1 (ZITD Loss Function): "Our code is available on Github" -
linking to github.com/STTDAnonymous/STTD (verified by fetching the
live page, not assumed). This is the SAME repository already
investigated and disclosed as a dead end earlier this session (its
`Accident_risk` folder is a literal 1-byte empty stub, confirmed
unchanged - `pushed_at: 2023-09-17`, no new commits since) - the repo is
for a different, sibling paper by the same author group (travel-demand
prediction, Chicago/NY taxi data, D_GCN+B_TCN architecture), not the
crash-prediction paper itself. That conclusion still stands.

**What's newly useful**: the repo's `utils.py` contains a genuine,
MIT-licensed, differentiable Tweedie NLL (`tweedie_nll_loss`) that the
same authors wrote for that sibling paper - a legitimate mathematical
reference for the identical Zero-Inflated Tweedie loss family, even
though the specific crash-prediction glue code was never published.
Cross-checking it against this project's own `zero_inflated_tweedie_nll`
(added the previous day, derived independently from the paper's PDF
text) found: **the mu-dependent terms match EXACTLY**
(`-y*mu^(1-rho)/(phi*(1-rho)) + mu^(2-rho)/(phi*(2-rho))`) - independent
confirmation this project's from-scratch derivation was correct. The
one real difference: this project's original version used the standard
GLM "Tweedie unit deviance" convention (McCullagh & Nelder), which
DROPS the intractable, mu-independent normalising term entirely (a
disclosed simplification, since it doesn't affect the gradient w.r.t.
mu); the reference code instead APPROXIMATES that term with the single
dominant series item (a saddlepoint-style approximation evaluated at
the series' own peak index) rather than dropping it - a more faithful
choice, since it gives `phi` a real, if still approximate, gradient
signal from y>0 rows too (the dropped-term version could not).

**Change made**: `zero_inflated_tweedie_nll`'s y>0 branch now ports this
exact reference formula (`alpha`, `j_max`, `log_a_approx` - see the
function's own updated docstring for the full derivation) instead of
the unit-deviance substitute. A genuine sign bug was caught and fixed
during this port (an early draft had `log_a_approx` subtracted where it
needed to be added - caught by re-deriving the formula independently in
a standalone script and checking it reproduces the reference code's own
`ll[mask]` value bit-for-bit before touching the project's actual
implementation, not just by re-reading the algebra). All 6 Tweedie unit
tests updated with newly hand-computed expected values and re-verified;
164 tests passing.

**Re-run immediately after, both boroughs**: the earlier concern (phi's
different gradient could plausibly perturb results, even though mu's
own gradient equation is unchanged) was checked, not assumed.

| Decoder/target | Old (dropped-term) | New (reference-code) |
|---|---|---|
| Tweedie+TCR, Westminster | 45.76% (std 6.55%) | **46.60%** (std 5.17%) |
| Tweedie+TCR, Lambeth | 44.36% (std 4.62%) | **44.58%** (std 4.18%) |

Both essentially unchanged (well within one std of each other) - some
individual windows shifted by a few points in either direction (e.g.
Westminster's 2023-10-13 window: 45.76%→42.37%; its 2024-07-09 window:
48.78%→51.22%), confirming phi's more correct gradient DOES perturb
training somewhat, but the net effect across windows washes out. The
qualitative finding is unchanged and now doubly confirmed: Tweedie does
not beat the simpler ZIP decoder on either the old or the corrected
loss, on either borough. The ZIP+TCR candidates in both re-runs
reproduced their exact prior values (48.72% Lambeth, bit-for-bit
identical per-window) - expected and a useful internal-consistency
check, since that candidate's loss function was untouched by this fix.

## 2026-09-02, "think deeper" pass - re-reading the full PhD thesis a second time found two real structural differences this project had not fully internalised, one of which is now ruled OUT by a clean controlled experiment

**User request** (verbatim): "make model better think deeper more deep."
Re-read the thesis's Data Description and Experiment Setup sections
(pages 229-232) line by line rather than relying on earlier, partial
extractions, and separately re-extracted the journal paper's exact
AccHR@20 formula (Eq. 20 in the arXiv HTML version) to check this
project's own metric implementation against it directly.

### A real metric-fidelity bug found and fixed: `accuracy_hit_rate` pooled days instead of averaging them

The paper's own Eq. 20, quoted exactly: `Acc@a = (1/p) * sum_j
[hits_j / crashes_j]` - the MEAN of each day's own hit ratio. This
project's `accuracy_hit_rate` instead computed `sum_j hits_j / sum_j
crashes_j` - a single ratio pooled across all days, only equal to the
paper's mean-of-ratios when every day happens to have the same crash
count. Fixed to average per-day ratios (days with zero actual crashes
still excluded from the average - the paper's own literal `1/p` divides
by every day including crash-free ones, a real, disclosed ambiguity in
their formula this project does not adopt, since a network-wide
crash-free day is plausible at this project's smaller test-window scale
even if implausible at the paper's own ~5,000-road scale). New
regression test (`test_accuracy_hit_rate_averages_per_day_ratios_not_pooled_totals`)
constructs a 2-day case where the two formulas diverge (0.625 vs 0.4)
to prove the fix actually changed behaviour, not just its wording - 165
tests passing. **Not yet re-run against any of this session's existing
sweep results** - every prior AccHR@20 number in this project's docs was
computed with the OLD (pooled) formula; the two formulas can only
diverge when a multi-step (`horizon > 1`) evaluation has uneven crash
counts across its `p` days, which is true for every sweep this session
has run, so this is a real, disclosed asterisk on every number quoted
above this entry, not assumed to be negligible without checking.

Two genuine findings from the thesis itself, both disclosed
immediately, not discovered-then-buried:

### Finding 1: their road network is Ordnance Survey, not OpenStreetMap - and theirs is consistently ~1.5x COARSER

Table 7.2 ("Data Characteristics by Region"), quoted exactly: "Roads |
Ordnance Survey | 8 [classes] | 4,822" (Westminster), "5,659" (Lambeth),
"4,688" (Tower Hamlets). This project's own cached OSMnx `drive`-network
graphs: **Westminster 7,552 segments, Lambeth 8,238 segments** - a
consistent ~1.5x (1.57x Westminster, 1.46x Lambeth) more granular
network on BOTH boroughs, not a one-off. `network_type="drive"` already
excludes footpaths/cycleways, so the gap isn't about including non-
driveable ways - it's a genuine segmentation-convention difference
(OSMnx/OSM splits ways at more points than an official surveyed network
typically does). This project's own `ingest/os_open_roads.py` (built in
an earlier session) can already load a real OS Open Roads GeoPackage as
an OSMnx-compatible graph, but the underlying 2GB source file is no
longer present locally and requires the user's own OS Data Hub account
to re-download - **not attempted in this pass**, disclosed as a real,
actionable, un-taken next step rather than silently skipped. A finer
network changes the top-20% cutoff's absolute size and the ranking
task's inherent difficulty in ways this project cannot yet quantify.

### Finding 2, tested and RULED OUT: their evaluation is a same-year (2019) 6:2:2 split, not a multi-year walk-forward - replicating it does NOT close the gap

Thesis, Section 7.4.1, page 232, quoted exactly: "The training,
validation and test data are all from 2019, and the ratio is 6:2:2...
early stopping strategy is employed with patience equal to 10." This
project's OWN prior citation of this detail (`daily_temporal.py`'s
docstring) had said "8:2:2" - a less careful earlier reading, now
corrected to "6:2:2" with the exact page/quote on record. Every prior
comparison in this project used a genuinely harder multi-year expanding-
window walk-forward instead (train on whole years, test on a later,
unseen year) - a real, previously-untested candidate explanation for
the AccHR@20 gap, since a same-year split has far less true distribution
shift.

**Built `scripts/run_2019_replication.py`** to test this directly:
downloaded DfT's full historical STATS19 archive (2019 has rolled out of
the "last 5 years" per-year download window; 1.53GB file, filtered to
1,521 Westminster / 2019 collisions - close to, not identical to, the
thesis's own reported 1,745, plausibly a data-revision or minor
definitional difference, disclosed not resolved), split 2019
chronologically 60/20/20 (Jan-Jul train, Jul-Oct val, Oct-Dec test - this
project's own reasonable reading of "6:2:2," since the thesis states
neither chronological-vs-random nor exact boundaries), trained `heads=3`
on the train+val instances.

**First attempt, WITH early stopping (patience=10, matching the thesis
literally)**: finished in ~18 seconds and scored a suspiciously low
**29.07%** (std 3.50%) - the exact same premature-stopping failure mode
already diagnosed at small instance counts (this session's earlier
"think, think, think" entry), even with 3 validation instances instead
of 1. Recognised as confounding the temporal-split question with an
already-known-unreliable stopping mechanism, not a clean test - re-run
instead with a fixed 200-epoch budget (`GREYSPOT_2019_EARLY_STOPPING=0`,
the script's new default) to isolate the actual variable of interest.

**Second attempt, fixed 200 epochs, the real test**: **AccHR@20 =
46.64%** (std only 1.33% - unusually tight, all 5 test windows landing
within 4 points of each other), Westminster. **Confirmed on Lambeth
too: AccHR@20 = 50.04%** (std 5.06%). Both essentially match this
project's own multi-year walk-forward results (49.57% Westminster /
47.31% Lambeth), not the paper's reported 68.98%/76.59% - on BOTH
boroughs, not a one-off.

**Also re-ran the flagship multi-year walk-forward config with the
corrected (per-day-averaged) `accuracy_hit_rate` from the fix above**,
since every number quoted anywhere in this project's docs before this
entry used the old, pooled formula: **50.14%** (std 7.32%, corrected)
vs **49.57%** (std 6.60%, old/pooled) - essentially unchanged. The
metric-averaging bug was real and worth fixing, but was not hiding a
materially different headline number; every comparison in this entry
uses the corrected metric on both sides, so the 2019-vs-multi-year
comparison above is a fair, consistent one.

**Conclusion**: the same-year-vs-multi-year temporal split is now RULED
OUT as an explanation for the AccHR@20 gap, with a real, clean,
controlled experiment on TWO boroughs - not just reasoned about.
Replicating the paper's own literal train/val/test arrangement, on this
project's own network and features, produces a number in the same
45-50% range every other honest configuration this session has landed
in, not their 69-77%. Combined with the earlier decoder-family finding
(ZIP vs the paper's own Tweedie, ruled out on two boroughs) and the
target-definition finding (plain count vs TCR, ruled out), this leaves
the road-network granularity/source difference (Finding 1 above) as the
single most concrete, evidenced, and still-untested remaining
hypothesis for the gap - not because everything else was tried
carelessly, but because it has now been tried carefully and
specifically ruled out.

**Housekeeping**: the 1.53GB historical STATS19 archive downloaded for
this test has been deleted (extraction complete, both boroughs already
run) - re-running `scripts/run_2019_replication.py` requires
re-downloading it first (the script's own `load_2019_collisions` gives
the exact URL if the file is missing). The thesis PDF (`thesis.pdf`,
43MB) is kept in the project root as a local reference, matching this
project's existing convention for the journal paper PDF already there.

## 2026-09-02, later the same day - "why don't we get their results if we copy the whole pipeline": found and closed the two remaining Table 7.2 feature gaps (POI, socio-demographic), result is real but not yet statistically significant

**User request** (verbatim): "why dont we get the same results as ucl
one if we copied the saeme exact pipeline as tjem like copy all from
them copy whole." Answered directly first (there is no complete code
release to copy - `github.com/STTDAnonymous/STTD`'s `Accident_risk`
folder is a genuine empty stub, confirmed a dead end twice now), then
re-checked Table 7.2 for anything still unreconstructed: two real
inputs remained unimplemented - "PointofInterest" (Ordnance Survey, 20
classes) and "Socio-demographicCharacteristics" (Census 2011, 8
classes). Weather (the third Table 7.2 gap) was already closed in an
earlier pass and found neutral.

**Built two new ingest modules, both with disclosed free substitutes for
the paper's literal sources** (full reasoning in each module's own
docstring):
- `ingest.socio_demographic` - 2019 IMD deprivation-domain scores +
  population density (already at 2011 LSOA geography, matching Census
  2011's own boundaries) in place of a literal Census 2011 pull. Needed
  LSOA boundary polygons (not previously in this project) - downloaded
  London's 2011 boundary files from London Datastore (271MB zip, OGL
  v2, free), extracted just the Westminster/Lambeth LSOA shapefiles,
  deleted the zip. 8 features used (IMD score + 6 domain scores +
  population density), matching the paper's own "8 classes" count in
  breadth, not claimed as the same 8 variables.
- `ingest.poi` - OSM POI tags (`shop`/`amenity`/`leisure`/`tourism`) in
  place of OS's own Points of Interest product, which (unlike OS Open
  Roads) is a commercial dataset, not part of OS OpenData, so not
  freely substitutable with an official OS source at all. **A real
  reliability finding, disclosed in the module's own docstring**: a
  single combined-tag polygon query timed out at 180s; the SAME query
  by bounding box (reusing `ingest.os_open_roads.borough_bbox_wgs84`)
  completed in ~20s for 21,505 features - Overpass's polygon-clipping
  is evidently far more expensive than a bbox filter for a complex
  administrative boundary. Even bbox-based, the public Overpass
  endpoints proved flaky the SAME day (`overpass-api.de` connection
  timeouts, then `overpass.kumi.systems` 500/502 errors) - Westminster's
  fetch succeeded (33,363 POIs, 74.6s) before this started happening;
  Lambeth's did not, on either endpoint, and is disclosed as an
  incomplete cross-borough check, not silently skipped.

Both modules tested with synthetic fixtures (9 new tests: LSOA join
correctness, midpoint-outside-every-LSOA handling, POI category
counting, max-distance rejection) - 175 tests passing.

**Result, Westminster only** (`run_ucl_comparison_multiwindow_poi_socio.py`,
heads=3, same 6-window light protocol, POI + socio-demographic columns
added to the existing feature set): **AccHR@20 mean = 55.58%** (std
15.17%, range 37.82%-84.73%) vs the corrected-metric flagship's 50.14%
(std 7.32%) - a real ~5.4-point improvement in the mean, but with more
than double the standard deviation, driven almost entirely by ONE
outlier window (2024-07-09: 84.73% vs the flagship's 54.07% on the same
window, a +30.7-point swing). **Checked with the same paired
significance testing used for the heads=3-vs-heads=1 finding, not just
eyeballed**: paired t-test t=0.84, **p=0.44** - NOT statistically
significant at this sample size. Wilcoxon p=0.56, also not significant.
4 of 6 windows favour POI+socio, 2 favour the flagship - a real,
disclosed mixed picture, not a clean win.

**Honest conclusion**: this is a promising but NOT yet a confirmed
result. The direction is right (positive mean improvement, majority of
windows favour it) and the underlying features are genuinely new signal
this project had never used before, but the evidence at n=6 windows
with this much variance cannot distinguish it from noise the way the
heads=3 finding could.

**The outlier-window question, checked, not left open**: the
2024-07-09 window has 13 of 14 days with at least one actual crash (41
total crash cells) - directly comparable to every other window (11-14
crash-days, 29-59 crash cells), NOT an unusually sparse one. This rules
out the plausible-sounding "recently-fixed per-day-averaged metric is
just noisier on a thin-data window" explanation - the 84.73% score
reflects the model's actual predictions on a normally-populated window,
not a small-sample artifact. Whether that reflects genuine skill (POI/
socio-demographic features happening to align unusually well with that
specific 14-day period's risk pattern) or ordinary run-to-run variance
in what is still, at bottom, a single training run per window remains
open - but it is not explained away as easily as a sparse-data fluke.

**Lambeth cross-borough check, completed once Overpass recovered**:
POI fetch succeeded on retry (2,540/8,238 segments matched within 50m -
lower coverage than Westminster's 3,725/7,552, a real, disclosed
borough difference in POI density, not a bug). Result: **AccHR@20 mean
= 57.72%** (std only **5.13%** - notably tighter than Westminster's
15.17%, no outlier window this time) vs the corrected-metric flagship's
**46.60%** (std 10.13%, re-run fresh on Lambeth for a fair paired
comparison - `run_ucl_comparison_multiwindow.py Lambeth`). Paired
significance: t=1.71, **p=0.148** - still not conventionally
significant alone, but the direction is much stronger and cleaner than
Westminster's (5 of 6 windows favour POI+socio, only one loses, and
that loss is smaller in magnitude than Westminster's own outlier win).

**Combined across both boroughs (n=12 paired windows)**: paired
t-test t=1.86, **p=0.090** - borderline, trending toward the
conventional 0.05 threshold without quite reaching it. **9 of 12
windows favour POI+socio features.** This is now the strongest,
most cross-borough-consistent evidence for any single lever tried in
this whole "think deeper" pass, second only to the original
heads=3-vs-heads=1 finding (p=0.036) - genuinely promising, honestly
short of "confirmed," and reported at exactly that level of confidence,
not rounded up or down.

### A second reference-code detail found: the paper's own rate/mean parameter uses a log-link (exp), not softplus

Continuing to re-read the reference repo (github.com/STTDAnonymous/STTD)
line by line per the user's explicit "connect every dot" instruction -
`model.py`'s `NBNorm_ZeroInflated` class leaves its `n`/`p`/`pi`
(phi/rho/mu) outputs with NO activation at all when `four=True` (the
actual zero-inflated 4-parameter branch this project's own decoders are
modelled on) - the positivity/range constraints seen in `ST_TWEEDIE.forward()`
(`rou_res = sigmoid(rou_res) + 1`, `phi_res = relu(phi_res)`) are applied
one level up, and **mu itself gets no constraint anywhere in the model** -
it is only made positive later, inside the loss function itself
(`tweedie_nll_loss`/`nb_tcn_nll` in `utils.py`): `mu = torch.exp(mu)`.
This is a genuine, different design choice from this project's own
original one (`rate = softplus(decoder_output)`, applied inside the
model, always non-negative even before any loss is computed) - a
log-link (mean = exp(raw)) is the standard convention for Poisson/
Tweedie GLMs, with different growth behaviour than softplus (exponential
vs. near-linear for large inputs).

**Added as a genuine, disclosed, testable alternative**: `GATTemporal`/
`train_gat_temporal_walkforward` gained a `rate_link: str = "softplus"`
parameter (`"exp"` the new option, clamped at +-15 pre-exponent for
numerical safety - a standard log-link precaution, not paper-specific).
Default unchanged, every existing test and prior result unaffected. 3
new unit tests (rejects invalid values, hand-verified `exp(bias)` output
with zeroed weights, confirms the clamp prevents `inf` on an extreme
input) - 178 tests passing.

**Run immediately after, the real test**: `rate_link="exp"` vs the
flagship (`heads=3`, otherwise identical), same 6-window protocol,
Westminster. Result: **AccHR@20 mean = 47.65%** (std 7.98%) vs the
flagship's 50.14% (std 7.32%) - LOSES on every single one of the 6
windows (0/6 wins), a small-to-moderate margin each time (-0.3 to -4.5
points). Paired t-test: t=-3.93, **p=0.011** - statistically
significant, even more confidently than the heads=3-vs-heads=1 finding
(p=0.036). **A clean, decisive negative**: despite `exp` being the
paper's own reference code's actual choice, `softplus` is reliably
better for this project's own architecture/data/hyperparameters. A
plausible reason (not verified further): `exp`'s unbounded exponential
growth for large pre-activation values is a less stable optimisation
target than `softplus`'s gentler near-linear growth, particularly at
this project's lr=0.01 (itself tuned around the softplus behaviour, not
re-tuned for exp - a possible confound worth flagging, though the
result's consistency across all 6 windows suggests a real effect, not
noise). Default (`rate_link="softplus"`) unchanged - this closes the
question rather than opening a new default.

### The single most important finding of this "connect every dot" pass: their OWN baseline comparison table shows almost every model they tried outperforms every configuration this project has tried

Continuing to re-read the thesis for anything missed, Table 7.3 (page
239, "Comparison of multi-step prediction performance (14 days) in
three boroughs") gives AccHR@20 for SEVEN OTHER models the paper itself
trained on the identical data/network as STZITD-GNN, not just their
headline model:

| Model | Lambeth | Tower Hamlets | Westminster |
|---|---|---|---|
| HA (historical average, no neural net) | 45.20% | 47.52% | 42.17% |
| STGCN | 61.13% | 58.69% | 50.20% |
| STGAT (GAT+LSTM) | 64.22% | 69.50% | 48.08% |
| STG-GNN (Gaussian decoder) | 26.66% | 26.47% | 29.33% |
| STNB-GNN | 44.71% | 50.22% | 45.03% |
| STTD-GNN (plain Tweedie, NOT zero-inflated) | 71.23% | 63.68% | 60.75% |
| STZINB-GNN | 61.84% | 58.27% | 51.39% |
| **STZITD-GNN (their full model)** | **76.59%** | **72.24%** | **68.98%** |

**This is decisive, structural evidence, not decoder-choice evidence.**
Every one of their models except the ill-suited Gaussian one (STG-GNN,
which actively underperforms even the trivial historical-average
baseline - a Gaussian's negative-probability mass makes it a poor fit
for zero-inflated crash counts, consistent with why this project never
tried a plain-Gaussian decoder either) scores HIGHER than this
project's own best configuration (49.57%/47.31% Westminster/Lambeth)
- including STNB-GNN (44.71%-50.22%, the closest architectural analogue
to this project's own `heads=3`+ZIP/ZINB setup) and even the trivial HA
baseline, which needs no neural network, no graph, no temporal model at
all, and STILL scores in the same range as this project's best deep
learning configuration. **Whatever explains the gap, it demonstrably
is not which decoder distribution is used** - their own STTD-GNN
(Tweedie, no zero-inflation at all) scores 60.75%-71.23%, far above
this project's own from-scratch Tweedie implementation (45.76%-46.60%
Westminster, verified against their actual reference code) on the
identical decoder family.

**The paper's own text (page 240) explicitly names graph structure as
a factor**: "It performs particularly well in Westminster, ... aided by
the area's lower zero inflation rate and **denser graph structure**."
This is the paper's OWN authors attributing cross-borough performance
differences partly to network density - independent confirmation, from
their own words, of this project's road-network-granularity hypothesis
(Westminster: their 4,822 roads vs this project's 7,552; a "denser"
official network in their case plausibly behaves differently from a
finer-grained OSM one in ways beyond simple node count).

**Updated conclusion**: this table converts the road-network hypothesis
from "the single remaining untested candidate, by elimination" to
"independently corroborated by the paper's own baseline results and its
own authors' stated reasoning" - the strongest evidence yet, in this
entire investigation, for where the real gap comes from. It is not
fully confirmed (still blocked on the missing OS Open Roads source
file), but it is no longer merely the last hypothesis standing - it is
now positively evidenced by data this project did not generate.

### CONFIRMED: extended to Tower Hamlets (the paper's own third borough), POI+socio-demographic is now statistically significant across all three boroughs

Registered Tower Hamlets in `ingest.boroughs` (ONS code E09000030,
verified directly from this project's own cached IMD 2019 data, not
assumed) - Gao et al.'s own third case-study region, chosen specifically
to extend this check to every borough the paper itself reports numbers
for, not an arbitrary third data point. Re-downloaded the London
boundary zip (deleted after the first extraction) just for Tower
Hamlets' LSOA shapefile; found and fixed a real naming-convention bug
along the way (`LSOA_2011_BGC_Tower_Hamlets.shp` uses an underscore for
the multi-word borough name - Westminster/Lambeth are single words, so
this never surfaced before). POI fetch mostly succeeded (26,106 raw
features across 3 of 4 categories; `tourism` failed once on a
connection reset, disclosed - `has_poi` and the other 3 categories are
still real for every matched segment).

**Result: AccHR@20 mean = 52.88%** (std 7.98%) vs a freshly-run
corrected-metric Tower Hamlets flagship of **42.54%** (std 9.44%).
Paired significance: t=3.70, **p=0.014 - wins on ALL 6 of 6 windows**,
the cleanest single-borough result of the three.

**Combined across all three boroughs the paper itself studies (n=18
paired windows)**: paired t-test t=2.93, **p=0.0094**; Wilcoxon
W=29.0, **p=0.0120** - BOTH significant at the conventional α=0.05
threshold, by a comfortable margin, not a borderline call. **15 of 18
windows favour POI+socio-demographic features.** Mean AccHR@20 rises
from 46.43% (flagship, pooled across all 18 windows) to 55.39% - a real
~9-point improvement, now confirmed with the same statistical rigour as
every other claim in this project, not asserted from a good-looking
average.

**This is now the single strongest, most confirmed positive finding of
this entire session - stronger than the original heads=3-vs-heads=1
result (p=0.036).** Two genuinely new data sources (POI density,
socio-demographic characteristics), both real inputs from the paper's
own Table 7.2 that this project had never used before this "think
deeper" pass, produce a significant, reproducible, cross-borough
improvement over the previous best-known configuration. **The project's
best-known configuration for AccHR@20 is hereby updated**: `heads=3` +
POI + socio-demographic features (mean 55.39% pooled across 18 windows,
3 boroughs) supersedes the plain `heads=3` config (46.43% pooled) as
this project's best verified result. The gap to Gao et al.'s own
68.98%/76.59%/72.24% remains real and open, but has now narrowed by a
statistically-defensible margin, not just a reasoned-about one.

## 2026-09-02 - Network-granularity hypothesis: testing a consolidated OSMnx graph

Continuing the "read every minute detail" pass: re-reading the paper's
thesis Table 7.2 (network descriptive statistics, not the results
table) surfaced a fact this project had not previously weighed - the
paper's OS Open Roads network is consistently **~1.5x coarser** than
this project's OSMnx network on every borough it lists: Westminster
4,822 roads (paper) vs 7,552 segments (this project); Lambeth 5,659 vs
8,238. The real OS Open Roads source file remains blocked (needs a
free but manual OS Data Hub account only the project owner can
create - see `ingest.os_open_roads`'s own docstring), so it still
cannot simply be swapped in. Instead this project built a disclosed,
self-directed **approximation** of a coarser network using data
already on hand, to at least test whether granularity itself is a
lever on AccHR@20 - a genuinely new, previously-untried hypothesis,
distinct from the POI/socio-demographic feature work above.

**What was built**: `consolidate_borough_graph()` in
[network.py](../src/greyspot/ingest/network.py) wraps osmnx's own
`simplification.consolidate_intersections()` to merge intersection
nodes within a distance tolerance (offset crossings, small
roundabouts, closely-spaced signal junctions OSM contributors mapped
as separate nodes) into single nodes, reducing the segment count.
**This is explicitly not a claim of replicating OS Open Roads' actual
survey-based link/node convention** - OS may group segments by
named-street continuity or other criteria entirely unrelated to node
proximity. It is one principled, testable way to ask "does a coarser
network help," using only data this project already has.

Tolerance was found **empirically, not derived**: a sweep on the real
Westminster graph gave 10m->3,761 edges, 5m->4,129, 2m->4,324,
1m->4,356, 0.5m->4,365 (original: 7,552 edges). 2.0m was chosen as the
closest approach to the paper's reported 4,822 without overshooting.
The same tolerance is used across every borough deliberately (not
hand-tuned per borough to hit its own target count), to keep the
methodology consistent rather than risk looking like curve-fitting to
the paper's own numbers.

Verified before running any sweep: the consolidated graph is a plain
WGS84 `MultiDiGraph`, fully compatible with every existing downstream
function unchanged (`graph_to_edges_gdf`, `snap_points_to_graph`) -
manually confirmed 100% collision-snapping success (3,610/3,610) on
the consolidated Westminster graph. A subtle bug was caught and fixed
before the first real sweep: consolidation renumbers every
`segment_id`, so the POI-count cache path had to be made
tolerance-specific (`{borough}_poi_counts_consolidated_{tol}m.csv`)
- reusing the raw-network POI cache would have silently produced "no
POIs anywhere" (a segment-id key mismatch) rather than an honest
crash. 5 new unit tests added to `tests/test_network.py`, covering:
correct merging of genuinely-close nodes vs. leaving distant nodes
untouched, an osmnx quirk where an exact `tolerance_m=0.0` degenerates
internally (real library behaviour, not a bug in this project's code -
worked around with `0.001` in tests), pipeline compatibility, and
cache correctness. Full suite: 182 passed.

`scripts/run_ucl_comparison_multiwindow_consolidated.py` (copied from
the POI+socio sweep script, same 6-window walk-forward protocol, same
`heads=3 + POI + socio-demographic` config as the current best-known
setup, now run against the tolerance=2m consolidated Westminster
network instead of the raw OSMnx network) was run to test this
hypothesis.

**Result: a clean, statistically significant NEGATIVE.** Westminster:
mean AccHR@20 drops from the POI+socio baseline's 55.58% to **44.69%**
(std 11.16%), losing 5/6 windows (t=-2.14, p=0.085 - trending but not
significant on its own). Cross-checked on Lambeth (per the "test one
borough first, only expand if promising" policy this now applies
going forward - see project memory - this particular check was already
mid-launch when that policy was agreed, so it was let finish rather
than discarded): mean drops from 57.72% to **52.60%**, losing 4/6
windows (p=0.238). **Combined across both boroughs (n=12 paired
windows): mean AccHR@20 falls from 56.65% to 48.64% (-8.0 points),
candidate wins only 3/12 windows, paired t-test t=-2.5356, p=0.0277,
Wilcoxon W=11.0, p=0.0269 - BOTH significant at α=0.05.** This is now
as confirmed a negative as the `rate_link=exp` finding (p=0.011) - not
a weak or ambiguous result.

**Why this matters beyond "consolidation doesn't help"**: this
sharpens, rather than contradicts, the Table 7.2/7.3-derived hypothesis
above. Matching the paper's raw segment *count* via arbitrary
distance-based node-merging is not equivalent to whatever OS Open
Roads' real survey-based link/node convention actually does - crudely
merging nearby OSM nodes destroys exactly the spatial resolution a
top-20%-segment ranking task depends on (two genuinely distinct,
independently-risky locations get folded into one segment, and
whichever one had more crashes now dominates a merged segment that
under- or over-states the risk of the other). The consistent, roughly
double-digit-point drop across both boroughs is a plausible mechanistic
story, not just a number - it suggests network **granularity by node
proximity alone is actively harmful**, and if OS Open Roads' own
coarser network still contributed to the paper's higher AccHR@20, the
effect must come from something more specific than segment count (e.g.
their link/node convention preserving named-street-level distinctions
this project's proximity-based merge does not, or a different variable
entirely). **This closes the "just make the network coarser" version
of the hypothesis with a real measurement, not a shrug** - the real OS
Open Roads swap (blocked on the missing source file) remains the only
way to test the actual, unapproximated version of this lever.

## 2026-09-02 - Architecture sweep on top of POI+socio features: a clean null result, heads=3 retained

The one combination flagged as genuinely untested in the "third update"
leading-levers list: `heads=3` was tuned BEFORE POI+socio-demographic
features existed. Does the optimal head count shift once those features
are already present? Built
`scripts/run_ucl_comparison_multiwindow_poi_socio_headsweep.py` (same
`build_instances` as the POI+socio script, CANDIDATES swept over
heads=1/2/3/4 at gat_layers=1), run Westminster-only per the "test one
borough first" policy above.

**Result: heads=1 nominally edges out heads=3 (58.49% vs 55.58%, a
+2.9-point mean difference and a noticeably lower std: 8.31% vs
16.62%), but wins only 3 of 6 windows - paired t-test t=0.6245,
p=0.5597; Wilcoxon p=0.625. Not remotely significant.** Full ranking:
heads=1 (58.49%) > heads=4 (56.00%) > heads=3 (55.58%) > heads=2
(53.64%) - no monotonic pattern, consistent with noise around a flat
line rather than a real head-count effect. heads=3's own 6 values
reproduced the original POI+socio run's numbers EXACTLY (0.5925,
0.5910, 0.4333, 0.4923, 0.8473, 0.3782 - all 4 decimal places), a
useful incidental confirmation that training here is fully
deterministic given the same data/config, not a source of the
between-candidate variation.

**Interpretation**: once POI+socio-demographic features are already
present, GAT head count stops being a lever worth tuning further - a
sensible dynamic (the original heads=3-vs-heads=1 advantage, found
WITHOUT these features, may have partly been the model using extra
attention capacity to compensate for a thinner feature set; once richer
features carry more of the signal directly, that capacity stops
mattering as much). This does not overturn the original heads=3
finding (which remains correctly reported as p=0.036 in its own,
features-poorer context) - it just means there is no further, easy win
available on this specific axis on top of the current best
configuration. Per the "test one borough first" policy, no Lambeth/
Tower Hamlets cross-check was run - a p=0.56 result on the very
candidate to be confirmed does not clear the bar for a second borough's
GPU time. `heads=3 + POI + socio-demographic` remains this project's
standing best-known, most rigorously confirmed configuration.

## 2026-09-02 - Stepping back: mapping every remaining gap instead of one more isolated tweak

The user asked to stop testing single features/hyperparameters in
isolation and instead "map out everything... think what can be the best
approach like the UCL did... follow that" - a fair critique after a run
of one-variable-at-a-time experiments (consolidation, heads sweep,
seed ensemble). Built a full comparison table of every dimension the
paper's methodology specifies against this project's current state:

| Dimension | Paper | This project | Status |
|---|---|---|---|
| Target | TCR (severity-weighted) | plain count | Tested standalone (indistinguishable, 50.17% vs 49.57%) - never combined with POI+socio or dense density |
| Network | OS Open Roads (~4822 seg) | OSMnx (~7552 seg) | Blocked; a consolidation approximation tested and found significantly WORSE |
| POI/socio-demographic | OS/Census 2011 | OSM tags/2019 IMD | CONFIRMED positive, p=0.0094 |
| GAT heads | 3 | 3 | Matched |
| GAT layers/hidden | 2-layer, hidden=42 | 1-layer, hidden=16/32 | Tested once, bundled with epochs=20, ONLY at light (6-11 instance) protocol |
| Weight decay | 0.01 | 0.0 | Same confound, never isolated |
| Epochs | 20 | 200 | Tested literally at light protocol - badly underfits (6-11 instances is too little data for 20 epochs regardless of architecture) |
| Decoder | Zero-Inflated Tweedie | Zero-Inflated Poisson | ZIP confirmed better, 2 boroughs, light protocol only |
| Temporal density | N=20/p=14/stride=14 | now running dense (this project's own architecture) | in progress |

**The key insight this surfaced**: the paper's own hyperparameter
bundle (2-layer, hidden=42, epochs=20, wd=0.01) was previously tested
ONLY against light-protocol data (6-11 training instances/window) in
`run_ucl_comparison_multiwindow_faithful.py` - of course 20 epochs
underfits there, independent of whether the architecture itself is
right. The paper's own training data density is much closer to what
the dense protocol provides (20-128 instances/window, launched earlier
today as `run_ucl_comparison_dense_poi_socio.py`). Nobody had tested
the paper's full architecture+training bundle AT that matching density,
combined with the two independently-confirmed-positive features
(POI+socio) and the paper's actual target (TCR) - every previous test
varied one axis while holding the others at this project's own
defaults, never all the paper's real choices simultaneously.

Built `scripts/run_ucl_comparison_dense_faithful_full.py`: DENSE
protocol (calendar-spread windows, same as the POI+socio dense run) +
POI+socio features + TCR target + the paper's full architecture
(heads=3, gat_layers=2, gat_hidden=42, gru_hidden=42, weight_decay=0.01)
at BOTH epochs=20 (literal) and epochs=200 (isolates whether epoch
count alone still limits performance once data density matches the
paper's own - same design `run_ucl_comparison_multiwindow_faithful.py`
already used, just now at the right data scale). Decoder kept as ZIP
(not Tweedie) to keep this run's own comparison isolated to the
epochs=20-vs-200 axis; decoder choice at dense density is a legitimate
separate follow-up. Queued to run immediately after the currently-
running dense POI+socio job finishes (sequential, not concurrent, to
avoid GPU contention on a single 8GB laptop GPU - two heavy jobs
fighting for the same GPU slows both down with no net time saved).

**Superseded mid-run by a much bigger discovery** (see next entry) -
this job was stopped after 4/6 windows of its epochs=20 candidate
(0.3605, 0.3345, 0.4664, 0.4855 - kept here for the record, not
statistically analysed since incomplete) to free the GPU for the real
OS Open Roads test below, which took clear priority.

## 2026-09-02 - The OS Open Roads GeoPackage was NOT missing; a real bbox-vs-polygon clipping bug was found and fixed; the real network is now being tested directly

While asking the user where to download OS Open Roads from (per this
project's own prior note that "the 2GB source GeoPackage is gone from
local disk"), the user pointed out they'd already given it before -
checking, that note was simply WRONG: `oproad_gpkg_gb/Data/oproad_gb.gpkg`
(1.02GB compressed source, downloaded 2026-09-01) has been sitting in
the project root the entire time. Verified valid: 3,961,077 road_link
features nationally, correct CRS/fields, opens in under a second.
**This file had never actually been exercised end-to-end before today**
- built but never run.

**Running it immediately surfaced a real, previously-invisible bug**:
`build_borough_graph_os_open_roads` only ever bbox-clipped, never
clipped to the real administrative-boundary polygon. For Westminster,
whose bbox extends well past the actual borough (e.g. across the
Thames), this pulled in 11,347 road links - polygon-clipping (endpoint-
inside test, matching `ox.graph_from_place`'s own node-based inclusion
convention) cuts this to **5,549 links (11,098 directed edges after
OS Open Roads' own no-oneway-data both-directions convention)**.

**This is HIGHER than this project's own OSMnx graph for the same
borough (7,552 directed edges)** - the OPPOSITE of the "OS Open Roads
is ~1.5x coarser" reading of the paper's thesis Table 7.2 that
motivated the entire network-consolidation experiment (already tested,
found significantly negative - see the 2026-09-02 entry above). Two
honest possibilities, left open rather than resolved by assumption:
either the paper counts "roads" by some different convention (e.g.
aggregating multiple links into one named street, not counting
individual junction-to-junction segments the way both OSMnx and this
project's OS Open Roads loader do), or a different boundary definition
than OSMnx's Nominatim-derived polygon. **Unresolved, and deliberately
not chased further** - it doesn't change what actually matters: this
project can now test the REAL network's effect on AccHR@20 directly,
rather than reasoning from a segment count that may not even be
comparable across sources.

Fixed properly, not just for this one script: `ingest/os_open_roads.py`
gained `borough_polygon_wgs84()` (the real boundary shape, not just its
bbox) and an optional `polygon_wgs84` parameter on
`build_borough_graph_os_open_roads` that filters to links with at least
one endpoint inside the real polygon. Backwards compatible (defaults to
the old bbox-only behaviour when omitted). 4 new regression tests added
(`tests/test_os_open_roads.py`, now 15 tests) - including one
purpose-built to catch exactly this bug (a link genuinely outside the
borough but inside a deliberately loose bbox, real BNG<->WGS84
coordinate correspondence, not synthetic placeholder numbers). Full
suite: 174 passed.

Verified end-to-end before launching anything: 4,026 nodes, 11,098
edges, 100% collision-snapping success (3,610/3,610) - fully pipeline-
compatible, same as every other network source this project has used.

Launched `scripts/run_ucl_comparison_multiwindow_os_open_roads.py`
(light protocol, matching the established fast-iteration convention,
Westminster only per the "test one borough first" policy) - the real
OS Open Roads network + POI+socio-demographic features + heads=3,
isolating the network-source variable alone against the current best-
known configuration (55.58% on Westminster).

**Result: a large, statistically significant POSITIVE - the strongest
finding of the entire investigation.** Mean AccHR@20 on Westminster
rises from 55.58% to **70.03%** (std 6.56%), winning 5 of 6 windows,
paired t-test **t=2.5898, p=0.0488** (significant at α=0.05), Wilcoxon
p=0.0625 (the same mathematical floor for a 5/6-win pattern already
seen with the heads=3 finding - not weaker evidence, a small-sample
artifact). Per-window: 59.25%->65.41%, 59.10%->63.01%,
43.33%->72.42%, 49.23%->67.44%, 84.73%->83.19% (the one loss, both
already very high), 37.82%->68.72%.

**70.03% lands almost exactly on the paper's own reported Westminster
figure of 68.98%** - after six independent attempts to close this gap
this session (data richness, target definition, decoder distribution
x2 boroughs, temporal protocol x2 boroughs, a crude network-
consolidation approximation) each failed or landed inconclusive, the
REAL network swap - the one lever that was blocked all along on a file
that turned out to already be on disk - appears to close it. Bears
repeating why the earlier consolidation experiment gave the opposite
(negative) result: that was an arbitrary distance-based merge of
OSMnx's own nodes, not OS Open Roads' real survey-based link/node
convention - exactly the distinction flagged as the open question when
that experiment was closed. The real topology evidently preserves
distinctions a crude proximity-merge destroys.

**Cross-checked on Lambeth - CONFIRMED, now the strongest finding of
the entire investigation.** Lambeth alone: 57.72%->63.59% (std 10.28%),
4/6 windows, p=0.31 (not significant alone - a real, smaller, noisier
effect than Westminster's). **Combined across both boroughs (n=12
paired windows): mean AccHR@20 rises from 56.65% to 66.81% (+10.16
points), candidate wins 9/12 windows, paired t-test t=2.6317, p=0.0233,
Wilcoxon W=12.0, p=0.0342 - BOTH significant at α=0.05.** This exceeds
the POI+socio finding's own combined significance (p=0.0094/0.0120 on
3 boroughs, n=18) in effect size (+10.16 points vs +8.96) though not
(yet, pending a Tower Hamlets check) in borough count.

**The project's best-known configuration is hereby updated again**:
`heads=3` + POI + socio-demographic + REAL OS Open Roads network
(66.81% pooled across 12 windows, 2 boroughs) supersedes the OSMnx-
network POI+socio config (56.65% pooled) as this project's best
verified result. Westminster's own 70.03% now sits almost exactly on
the paper's reported 68.98% for that borough - the first time any
configuration in this entire investigation has landed within noise of
the paper's own number, not just narrowed the gap. Tower Hamlets check
launched for full 3-borough parity with the POI+socio confirmation
standard.

**CONFIRMED on Tower Hamlets too - this is now, by a clear margin, the
strongest and most significant finding of the entire investigation.**
Tower Hamlets alone: mean AccHR@20 rises from 52.88% to **67.34%** (std
6.81%), winning ALL 6 of 6 windows, paired t-test **t=3.3485, p=0.0204**
- individually significant, the second borough (after Westminster) to
cross significance alone. **Combined across all three of the paper's
own boroughs (n=18 paired windows): mean AccHR@20 rises from 55.39% to
66.99% (+11.59 points), candidate wins 15/18 windows, paired t-test
t=3.9768, p=0.0010, Wilcoxon W=17.0, p=0.0016 - both an order of
magnitude past the conventional α=0.05 bar, more significant than the
POI+socio finding's own p=0.0094/0.0120.**

**Comparison to the paper's own per-borough numbers, for the first time
genuinely close rather than merely narrowed**:

| Borough | This project | Gao et al. | Gap |
|---|---|---|---|
| Westminster | 70.03% | 68.98% | **+1.05 (matched/exceeded)** |
| Tower Hamlets | 67.34% | 72.24% | -4.90 |
| Lambeth | 63.59% | 76.59% | -13.00 |
| Pooled | 66.99% | 72.60% | -5.61 |

Westminster now matches (numerically exceeds) the paper's own reported
figure - the first configuration in this entire investigation to do
so, not merely narrow the gap toward it. Lambeth remains the largest
outstanding gap; whether that reflects a genuine borough-specific
effect (the paper's own text already noted cross-borough differences
partly attributable to network structure) or noise at only 6 windows
per borough is not yet resolved.

**The project's best-known configuration is hereby updated a final
time**: `heads=3` + POI + socio-demographic + REAL OS Open Roads
network (66.99% pooled across 18 windows, 3 boroughs) is now this
project's best verified result, superseding every earlier
configuration this session. Full test suite re-verified: 175 passed
(includes the polygon-clipping and UUID-cache-loading regression tests
added getting here).

## 2026-09-02 - The real-network effect confirmed robust to temporal protocol too

Combined the two independently-confirmed biggest levers for the first
time: the REAL OS Open Roads network + the DENSE protocol (paper's own
exact N=20/p=14/stride=14 density, calendar-spread held-out windows -
`scripts/run_ucl_comparison_dense_os_open_roads.py`). Westminster only.

**Result: mean AccHR@20 rises from 43.35% (dense, OSMnx network) to
62.80% (dense, real network) - +19.46 points, winning ALL 6 of 6
windows, paired t-test t=4.9973, p=0.0041, Wilcoxon p=0.0312.** Even
more significant than the light-protocol Westminster result (p=0.0488)
alone. Per-window: 29.47%->60.96%, 52.86%->61.67%, 48.55%->77.95%,
30.33%->52.07%, 55.95%->69.05%, 42.92%->55.14%.

**This is the important robustness check**: the real network's benefit
holds regardless of which temporal protocol evaluates it - not an
artifact of the light protocol's specific 3-year/quarterly-stride
window choice. Interestingly, the dense protocol's own mean (62.80%)
sits BELOW the light protocol's real-network mean for the same borough
(70.03%) despite far more training data per window - plausibly because
the dense protocol's calendar-spread windows sample genuinely harder-
to-predict periods across all 5 years/seasons, not because more data
hurts. Either way, the headline claim strengthens: the real-network
improvement is not protocol-specific.

## 2026-09-02 - Lambeth architecture recheck on the real network: another null result, plus a genuine reproducibility caveat found

Lambeth remains this project's one open gap (63.59% vs the paper's
76.59%). Since heads=3 was tuned entirely on OSMnx-network data, and
the real network has a meaningfully different topology (denser,
differently-connected), re-swept heads=1/2/3/4 on Lambeth specifically
with the real network (`scripts/run_ucl_comparison_multiwindow_os_open_roads_headsweep.py`).

**Result: another null - heads=3 remains best (or tied-within-noise).**
heads=1: 54.56%, heads=2: 61.64%, heads=3: 62.89% (reproduction of the
63.59% baseline, see caveat below), heads=4: 58.22%. No architecture
change closes Lambeth's gap - this mirrors the earlier OSMnx-network
finding that head count stops being a useful lever once POI+socio
features are present, now confirmed on the real network too.

**A genuine reproducibility caveat found along the way**: re-running
the identical heads=3 config on the identical windows did NOT
reproduce exactly - 5 of 6 windows matched the original run to 4
decimal places, but one window (2024-01-11) gave 53.33% vs the
original 57.50%. Checked the code: `train_gat_temporal_walkforward`
sets `torch.manual_seed(seed)` but never
`torch.use_deterministic_algorithms` - GAT's scatter/gather message-
passing operations on CUDA are not bit-reproducible run-to-run even
with a fixed seed (a well-documented PyTorch/CUDA behaviour, not a bug
in this project's code). Practical implication: individual reported
percentages should be read as "the result of one specific run," not a
perfectly fixed constant - a small amount of run-to-run noise (this
case: ~4 points on one window out of six) is real and expected.
**This does not invalidate any significance test already reported** -
each paired comparison used two specific, actually-executed runs
compared against each other at the time, which remains a valid
one-shot empirical comparison regardless of whether either run would
reproduce identically later. Making training fully deterministic
(`torch.use_deterministic_algorithms(True)` plus the CUDA workspace
config env var) would be a reasonable future hardening step, not
attempted here to avoid derailing the current investigation.

Tower Hamlets dense+real-network run launched next for full 3-borough
parity with Westminster and Lambeth's dense-protocol results.

**Tower Hamlets dense+real-network complete**: mean AccHR@20 = 59.98%
(std 10.87%). **All three boroughs now land in a remarkably tight
60-63% range under the dense protocol**: Westminster 62.80%, Lambeth
60.42%, Tower Hamlets 59.98% (pooled across all 18 dense windows:
mean 61.07%, std 9.47%). This is a strong robustness confirmation -
the real network's benefit generalises consistently across all three
boroughs AND both temporal protocols, even though the light protocol's
specific 3-year window happens to score higher in absolute terms
(pooled light-protocol real-network mean: 66.99%) than the dense
protocol's full 5-year calendar spread (61.07%) - consistent with the
earlier finding that the dense protocol simply samples a wider, harder
range of conditions, not that the network's benefit is protocol-
specific.

## 2026-09-02 - Chasing Lambeth's gap: TCR target null, then a genuine clue from the paper's own text

With architecture (heads sweep) and temporal protocol (dense) both
ruled out as ways to close Lambeth's remaining gap (63.59% vs the
paper's 76.59%), tried the paper's own TCR target combined with the
real network for the first time (`scripts/run_ucl_comparison_multiwindow_os_open_roads_tcr.py`).

**Result: another null, trending slightly negative.** Mean AccHR@20:
63.59% (plain count) -> 61.65% (TCR), candidate wins 0/6 windows (3
exact ties, 3 losses), paired t-test t=-1.4676, p=0.2021 - not
significant, consistent with the earlier OSMnx-network finding that
TCR and plain count are statistically indistinguishable, now confirmed
on the real network too.

**Re-read the thesis's own text specifically on why Lambeth is THEIR
best-performing borough despite being their hardest one - found a
genuine, well-evidenced clue.** Direct quotes: "the spatio-temporal TCR
graph matrix shows zero inflation rates of 95.72% in Westminster,
96.71% in Lambeth, and 96.28% in Tower Hamlets" (Lambeth highest);
"Notably, in Lambeth, the borough exhibiting the highest zero-inflation
and data sparsity, the STZITD-GNN model attains the lowest MAE... it
achieved 76.59% accuracy in identifying actual crashes in the Lambeth
case"; "In extreme zero-inflation scenarios, such as in Lambeth, both
the STTD and STZINB models perform poorly... compared to the STZITD-GNN
model." The paper attributes its OWN strong Lambeth result directly to
its Zero-Inflated Tweedie decoder's handling of extreme sparsity.

**Verified this project's own real-network data shows the identical
ordering** (checked, not assumed): zero-count-day rate 99.98% for
Lambeth vs 99.97% for both Westminster and Tower Hamlets - the same
relative ranking as the paper's own reported rates, at this project's
own (much finer, daily-grain) aggregation scale. This is a real,
corroborated pattern, not a coincidence of framing.

**Launched the one decoder choice never yet combined with the real
network**: ZI-Tweedie + TCR target + real network on Lambeth
specifically (`scripts/run_ucl_comparison_multiwindow_os_open_roads_tweedie.py`)
- this decoder was tested twice before and found worse than ZIP, but
only ever on the OSMnx network; the paper's own text gives a concrete,
specific reason to expect it might behave differently once the network
(and hence the actual sparsity pattern the decoder has to handle)
matches theirs.

**Result: a genuinely interesting partial update to a previous
finding, though still not the lever that closes Lambeth's gap.**
ZI-Tweedie + TCR on the real network: mean AccHR@20 = **63.50%** (std
11.27%) vs the ZIP + plain-count real-network baseline's 63.59% -
**statistically indistinguishable (t=-0.0400, p=0.9696; Wilcoxon
p=1.0000)**, and vs the ZIP + TCR run (61.65%) it is nominally *better*
(+1.85 points, 3/6 wins, p=0.2736, not significant).

**Why this matters despite being a null**: on the OSMnx network, this
same decoder was consistently and repeatably 4-5 points WORSE than ZIP
on both boroughs tested (Westminster 46.60% vs 50.17%; Lambeth 44.58%
vs 48.72%). On the real network that penalty **disappears entirely**.
This is a real, previously-unknown interaction: the decoder's relative
performance depends on the network source. The most plausible reading -
consistent with the paper's own claim that its ZI-Tweedie decoder
matters most in extreme-sparsity settings - is that the real
OS-topology sparsity pattern is the regime the decoder was designed
for, whereas OSMnx's differently-structured graph was not. **It does
not, however, beat ZIP even here** - so this project's ZIP decoder
remains the standing default, now on the stronger footing of "equal on
the paper's own network, better on OSMnx" rather than the previous
"better on OSMnx, untested on the real network."

**Where Lambeth stands after this**: four separate hypotheses have now
been tested and closed against Lambeth's 13-point gap specifically -
architecture (heads sweep, null), temporal protocol (dense, no gain),
target definition (TCR, null), and decoder (ZI-Tweedie, null). None
closes it. The gap is real and remains unexplained.

## 2026-09-02 - Multi-seed ensembling: a clean null, and an informative one

Every AccHR@20 figure this project has ever reported came from a
SINGLE weight initialisation (`seed=42`, the default no sweep script
ever varied), at 6-11 training instances per window - the regime where
one seed is least trustworthy. Independently corroborated by this
session's own CUDA-non-determinism discovery (a re-run of an identical
config moved one window ~4 points). Averaging several independently-
initialised models' predictions before ranking is the standard fix.
Built `scripts/run_ucl_comparison_multiwindow_os_open_roads_ensemble.py`
(5 seeds: 42/123/7/2024/31337, predictions averaged - including on the
calibration instance so the conformal interval stays consistent -
before scoring once, NOT 5 scores averaged after the fact).

**Result: a clean null.** Lambeth, real network: 63.59% (single-seed)
-> **62.49%** (5-seed ensemble), 3/6 windows favour the ensemble,
paired t-test t=-0.4256, **p=0.6881**; Wilcoxon p=1.0000. Per-window
movement was genuinely mixed (two windows up ~2-5 points, one down 12,
three roughly flat) rather than systematically better or worse.

**Why this null is informative rather than merely disappointing**: it
locates where the variance actually lives. If the 6-window spread were
driven by weight-initialisation noise, a 5x ensemble would have
visibly tightened it; it did not. So the spread is a property of the
DATA - genuinely different difficulty across held-out calendar windows
at ~3,000 crashes per borough over three years - not of the training
procedure. Two practical consequences: (1) ensembling is not worth 5x
the compute on this pipeline, closed; (2) more seeds/restarts will not
make these estimates more precise either - only more windows or more
data would, which is a much more expensive proposition and a fair
limitation to state plainly in any write-up.

## 2026-09-02 - Junction risk redistribution: the paper's own method, implemented faithfully, is significantly WORSE here (plus a self-caught evaluation bug)

Re-read the paper's METHODOLOGY section (Section 7.2.1) rather than only
its results/hyperparameter tables, and found a concrete technique this
project had never implemented: "To further address the complexity of
crash occurrences at intersections, where multiple road segments
converge, the methodology implements a balanced weighting scheme,
whereas the associated risk value is distributed equally among all
connected road segments." This project attributed each crash wholly to
its single nearest segment.

**Quantified the potential impact BEFORE building anything** (per the
user's "do heavy research before trying new stuff" instruction), using
STATS19's own `junction_detail` field (0 = "not at or within 20 metres
of a junction"): **63-72% of all collisions in the three study boroughs
are junction-related** - Westminster 71.9%, Lambeth 67.7%, Tower
Hamlets 63.2% over 2022-2024. That made this the largest unimplemented
methodological difference remaining, affecting the majority of the
signal rather than an edge case - a well-justified reason to build it.

Implemented as `ingest.network.redistribute_junction_crashes` (+
`weight_col` support in `collision_severity_counts_by_segment_day`).
Verified end-to-end on real Lambeth data before any sweep: 3,085
crashes in, total weight 3,085.0 out (conserved exactly, no
inflation/deflation), all 3,085 unique crashes preserved, ZERO emitted
segment_ids failing to match a real graph edge. **Two real bugs were
caught in the process**: (1) incident edges taken from an undirected
graph copy would emit reversed (u,v) pairs matching no real
`segment_id`, silently dropping the redistributed weight at join time -
fixed by using the directed graph's own in/out edges; (2)
`build_segment_day_table` cast `collision_count` to `int16`, which
would have silently TRUNCATED every fractional weight (0.25 -> 0),
deleting the entire effect without raising anything - fixed to cast
only when counts are actually integral. 8 new regression tests
(suite: 183 passing).

**A methodology error in this project's OWN first run, caught and
fixed before it produced a false conclusion** - worth recording as
plainly as the result itself: redistribution changes the TARGET, and
`accuracy_hit_rate` defines "actual crashes" as `y_true > 0`. Scoring
against the redistributed target meant one junction crash became 8
fractional entries and the model had to land ALL EIGHT in the top 20%
instead of one - a mechanically harder metric, not a worse model. The
first run's opening window scored 50.83% purely from that artefact.
Had this not been checked, the write-up would have recorded
"redistribution hurts" for entirely the wrong reason. **Fixed by
building two parallel targets**: redistributed for TRAINING, original
un-redistributed for EVALUATION (identical to every other experiment's
ground truth, so AccHR@20 stays comparable), with an assertion that
the two instance lists align one-to-one on held-out start dates so
they cannot silently drift.

**Result, with the corrected and now genuinely fair design: a
statistically significant NEGATIVE.** Lambeth, real network: 63.59%
-> **53.43%** (-10.16 points), winning only 1 of 6 windows, paired
t-test **t=-2.6919, p=0.0432**; Wilcoxon p=0.0625 (the usual 5/6-pattern
floor). Notably the redistributed run's own variance collapsed (std
2.66% vs the baseline's 11.26%) - it produces a much flatter, more
uniform risk ranking, which is precisely the problem: spreading each
junction crash across up to 8 arms blurs exactly the sharp
segment-level distinctions a top-20% ranking task depends on.

**Interpretation, stated carefully**: this does NOT show the paper's
method is wrong on their own setup. It shows that, on this project's
network representation, redistributing junction risk this way is
significantly harmful for AccHR@20. A plausible reason is
representational: OS Open Roads models a two-way street as two
directed edges, so a four-arm junction has EIGHT incident edges here,
and an equal 1/8 split may be a far more aggressive dilution than the
paper's own graph (where road segments are nodes, and a junction
connects fewer of them) intends. Testing a variant that splits across
undirected physical roads rather than directed edges is a legitimate
follow-up. As it stands, this is the third case this session where a
technique taken from the paper's own text (spillover, `rate_link=exp`,
now junction redistribution) measurably hurts on this pipeline - all
disclosed as real negatives, none quietly dropped.

## 2026-09-02 - The 2019 protocol re-tested on the REAL network: confound removed, conclusion holds

Earlier today `run_2019_replication.py` tested Gao et al.'s own
same-year 2019 6:2:2 evaluation protocol and concluded "the
temporal-protocol difference is NOT the explanation for the gap." That
conclusion had a real confound, visible only in hindsight: it ran on
the **OSMnx** network (its own docstring says so, disclosing the OS
Open Roads gap as unresolved because the GeoPackage was believed
missing). Hours later the real network turned out to be the single
largest driver of AccHR@20 in the entire investigation. So the protocol
had been ruled out while the dominant variable was held at the wrong
setting - not a safe conclusion.

Re-ran it properly (`scripts/run_2019_replication_os_open_roads.py`) -
their exact year, their exact protocol, and their exact network source
all at once, the closest like-for-like replication this project can
construct. Required re-downloading DfT's 1.53GB 1979-latest historical
archive (2019 has rolled out of the per-year downloads; verified no
2019-specific file exists - the per-year endpoint 404s and only a
last-5-years file is published). Archive verified before use: 9,116,625
rows total, 117,536 from 2019. Two earlier download attempts died at
~35MB with the server dropping the connection; succeeded with retry and
stall-detection settings.

**Result, Lambeth**: 2019 protocol on OSMnx = 50.04%; 2019 protocol on
the **real OS Open Roads network = 52.26%** (std 4.18%, 5 test
windows). Two things follow, and they point in opposite directions:

1. **The network effect replicates here too** (+2.2 points, same
   direction as every other protocol tested). Further evidence that the
   real-network finding is robust and not protocol-specific.
2. **The 2019 protocol still does not explain the gap - the original
   conclusion survives its confound being removed.** 52.26% is far
   below this project's OWN walk-forward result on the same borough and
   same network (63.59%), and further still from the paper's 76.59%.
   Their same-year split, run on their own network, produces WORSE
   numbers here than this project's harder multi-year walk-forward -
   the opposite of what "their easier protocol explains their higher
   numbers" would predict.

**This closes the last hypothesis that was ever ruled out under a known
confound.** Every protocol-level explanation for the Lambeth gap has
now been tested with the network variable set correctly. The remaining
gap is not explained by evaluation protocol, architecture, head count,
target definition, decoder family, ensembling, or junction weighting -
all measured, all disclosed.

## 2026-09-02 - Deep audit of the paper's Table 7.2 finds real data gaps; road class + date tested (null on mean, halves variance)

Following the user's instruction to plan properly rather than test
levers reactively, audited the paper's own Table 7.2 ("Data
Characteristics by Region") class-by-class against this project's
actual `FEATURE_COLUMNS`. Full plan written to
`docs/gap_closing_plan.md`. The audit's key observation: **both levers
that have ever produced a significant gain this session were DATA
changes (network source, POI+socio); every model-side lever tried
since has been null or negative.** That reframes the search - the
model is not the bottleneck.

The class-by-class comparison:

| Input | Their classes | This project | Gap |
|---|---|---|---|
| Crash Counts | 1 | 1 | matched |
| Socio-demographic | 8 | 8 | matched |
| **Roads (OS)** | **8** | **0** | entirely absent |
| **Point of Interest (OS)** | **20** | **4** | 16 short |
| **Date** | **4** | **1** | 3 short |
| Meteorological | 8 | 0 in best config | absent |

**Road class was entirely absent from the model** despite OS Open
Roads carrying `road_classification`/`road_function` on every edge the
best configuration already loads - `_map_highway` even derives an
OSM-style value from them. The model was ranking roads with no idea
whether a segment is an A road or a cul-de-sac. Implemented
`attach_road_class_features` (8 one-hot columns, **fixed** vocabulary
so a borough lacking motorways cannot silently produce a narrower
feature matrix and break cross-borough comparability) and
`attach_date_features` (is_weekend + cyclical month sin/cos, bringing
date inputs to the paper's four; cyclical so December and January sit
adjacent rather than 11 apart).

**Result (Lambeth, real network): a null on the mean but a real
variance effect.** 63.59% -> 63.74%, 3/6 windows, paired t-test
t=0.0362, **p=0.9725**. However the spread narrows sharply: **std
4.55% vs the baseline's 11.26%**, raising the worst window (57.50% vs
50.71%) while lowering the best (71.15% vs 76.92%). Road class and
calendar position evidently carry real, stabilising signal without
shifting central performance - worth keeping for robustness, not
claimable as an improvement.

## 2026-09-02 - A genuine architectural difference found in the paper's appendix: their encoder order is the REVERSE of this project's

Reading thesis Appendix A.3/A.4 (rather than only the results and
hyperparameter tables) surfaced something no amount of hyperparameter
sweeping would have found. Their spec: "**ZT = GRU(X_1:t, Y_1:t)**...
Once the input features have been processed via the GRU, we obtain
historical temporal embeddings ZT for all roads. **These are
subsequently directed to graph neural encoders**."

That is **GRU first, then GAT**. This project's `GATTemporal.forward`
does the opposite - a GAT pass at every timestep, then a GRU over
those per-timestep embeddings. A real difference in encoder topology,
never tested, taken directly from their specification rather than
guessed at. It is also substantially cheaper (one GAT pass instead of
T).

Implemented as a switchable `encoder_order` parameter
("spatial_first" default = unchanged original behaviour;
"temporal_first" = the paper's order), threaded through
`train_gat_temporal_walkforward`. In the paper's order the GRU
consumes raw features (input width `in_channels`) and the GAT consumes
the GRU's hidden state (input width `gru_hidden`), so the layer
dimensions genuinely differ - verified by differing parameter counts,
not assumed. **A test specifically asserts the two orders take
different computation paths**, guarding against the exact failure mode
that produced a bit-identical "new" experiment earlier this session
(the `per_physical_road` no-op). Suite: 195 passing.

Running both the 1-layer and 2-layer (their stated "GNNs... are all
two-layered") variants on Lambeth - results pending.

## 2026-09-02 - The encoder-order experiment: a self-caught false result, and a learning-rate/architecture interaction

**A result that would have been badly wrong was caught before it was
recorded.** The first run of the paper's GRU->GAT encoder order scored
14.31% - and, revealingly, its 1-layer and 2-layer variants returned
BIT-IDENTICAL per-window numbers. Three tells said "bug", not
"finding": scores below random (~20% for a top-20% selection), a
runtime ~4x faster than normal, and identical results from genuinely
different architectures.

Diagnosis: `temporal_first` was producing **all-NaN predictions**. The
"14.31%" was `accuracy_hit_rate`'s deterministic tie-breaking applied
to a NaN array - which also explains the identical 1-vs-2-layer
numbers (the tie-break returns the same arbitrary indices regardless of
architecture). **Had this been taken at face value the write-up would
have claimed "the paper's own encoder architecture scores 14% vs our
64%" - a dramatic, entirely false claim about their published method,
caused purely by this project's own untuned learning rate.**

Root-causing it took three steps, each ruling out a hypothesis:
1. **Not an implementation error**: on synthetic data with a known
   signal, `temporal_first` recovers it as well as the default order
   (high-risk nodes 2.05 vs 0.05 background). The forward pass is clean.
2. **Not exploding gradients** (the obvious first guess, and wrong):
   epoch-by-epoch instrumentation showed stable, converging training
   with gradient norms around 0.003. Gradient clipping was added, did
   NOT fix the NaN, and was **removed again** - it perturbed the
   default order's own predictions in the 4th significant figure, and
   leaving a baseline-altering change in place would have silently
   invalidated every comparison made this session.
3. **Attention-softmax instability at too high a learning rate**: NaN
   first appears at epoch 12, localised to the GAT's attention
   parameters (`att_src`, `att_dst`, `lin.weight`). At lr=0.001 the
   same configuration trains stably for all 200 epochs.

**The corrected run then produced the genuinely interesting result - an
architecture x learning-rate interaction that a single-lr comparison
would have completely hidden** (Lambeth, real network):

| Encoder order | lr=0.01 | lr=0.001 |
|---|---|---|
| Default (spatial_first, this project's) | **63.59%** | **16.76%** |
| Paper's (temporal_first, GRU->GAT) | **NaN (diverges)** | **50.85%** |

The control run - the default order at the paper-order's learning rate
- is what makes this legible, and was included precisely because the
2019-protocol conclusion had earlier been unsafe for want of one.
**Each encoder order has its own viable learning-rate regime, and each
collapses in the other's.** At lr=0.001 the paper's order beats this
project's by 34 points; at lr=0.01 the reverse holds. Comparing the two
architectures at any single learning rate is therefore not a comparison
of architectures at all - it is a comparison of how well each tolerates
a learning rate chosen for the other.

The 2-layer variant (their stated "GNNs... are all two-layered") scored
50.58%, statistically indistinguishable from the 1-layer 50.85% - and,
importantly, now genuinely DIFFERENT per-window rather than
bit-identical, confirming the earlier duplication really was the NaN
artefact rather than an inert flag.

**Honest status**: lr=0.001 was the first STABLE value tried for the
paper's order, not its optimum, so 50.85% is a lower bound on what that
architecture achieves here, not its ceiling. A bounded sweep
(lr=2e-3/3e-3/5e-3) is running to find its actual best before any
conclusion is drawn about which encoder order is better on this data.

### Resolution: the encoder-order comparison, done fairly

The learning-rate sweep found the paper's encoder order's actual
optimum, and the full stability curve (Lambeth, real network):

| lr | Paper's order (GRU->GAT) |
|---|---|
| 1e-3 | 50.85% |
| **2e-3** | **60.93%** (optimum) |
| 3e-3 | 58.98% |
| 5e-3 | diverges (NaN) |
| 1e-2 | diverges (NaN) |

The 5e-3 and 1e-2 runs both return exactly 0.1431 - the NaN tie-break
signature, now a recognisable fingerprint rather than a mystery.

**The fair comparison - each architecture at its OWN optimum, rather
than both at a learning rate tuned for one of them:**

- This project's order (spatial_first) at its best (lr=1e-2): **63.59%**
- The paper's order (temporal_first) at its best (lr=2e-3): **60.93%**
- Difference: -2.66 points, paper's order wins 2/6 windows
- **Paired t-test t=-1.2019, p=0.2832; Wilcoxon p=0.3125 - a clean
  null.**

**Conclusion**: the two encoder orders perform equivalently on this
data once each is given the learning rate its own topology requires.
The paper's GRU->GAT order is neither better nor worse here - it is
simply more sensitive to learning rate (its stable band is roughly
1e-3 to 3e-3, versus 1e-2 working comfortably for the default order).
The default order is retained, on the grounds that it is already the
tested-and-verified baseline for every other result in this project,
not because it is measurably superior.

**What this episode demonstrates methodologically** (worth keeping for
the write-up's reflection chapter): the same experiment produced three
different "answers" depending on how carefully it was run - 14.31%
(NaN artefact, would have been a spectacular false claim about the
paper), 50.85% (real but under-tuned, would have understated their
architecture by 10 points), and 60.93% (fair, at its own optimum). Only
the last is a legitimate measurement, and reaching it required a
control run, three ruled-out hypotheses, and a bounded hyperparameter
sweep. Reporting either of the first two would have been an honest
mistake, not a fabrication - which is precisely why the checks matter.

## 2026-09-03 - POI at the paper's 20-class granularity: another data-side null (and a total silent failure caught first)

Third and last of the Table 7.2 data gaps (`docs/gap_closing_plan.md`
Step 3). The paper lists "Point of Interest | Ordnance Survey | 20
classes"; this project used four coarse OSM tag families, so a pub, a
school and a hospital all summed into one `poi_amenity_count` despite
having entirely different crash-risk mechanisms. Built
`POI_FINE_CLASSES` - exactly 20 classes (education, nightlife, food,
healthcare, worship, transport, parking, fuel, finance, civic,
entertainment, three retail types, services, accommodation,
attraction, park, sports, playground), derived from OSM tag VALUES.
Disclosed in the module: OS's commercial 20-class scheme is not public,
so the granularity matches the paper's stated count while the
membership is this project's own.

**A total silent failure was caught before it produced a fake result.**
The first run logged "20040/20040 POIs matched none of the 20 fine
classes and were dropped" - every single POI unclassified. Root cause:
`download_borough_pois` did `gdf.reset_index()[["geometry"]]`, throwing
away the tag columns; the pipeline only ever knew a POI was "an
amenity", never *which* amenity. The 20-class taxonomy therefore had
nothing to classify on. **Every unit test passed throughout** - they
built tidy synthetic frames with the columns the classifier expected,
a shape the real pipeline never produces. Had the drop not been logged,
the model would have received all-zero POI features and returned a
plausible null, and this project would have reported "POI granularity
doesn't help" while misrepresenting the paper's most-classes input.

Fixed by preserving the tag value (`poi_tag_value`) through ingestion
and teaching the classifier both shapes (normalised pipeline output AND
raw OSMnx frames). Added a regression test asserting against the shape
`download_borough_pois` ACTUALLY emits, plus one checking tag *family*
is respected (so a shop named "park" cannot count as green space).
After the fix: 14,221/20,040 POIs (71%) classify; the remaining 29% are
genuinely out-of-scope OSM tags (benches, waste baskets, street
furniture), correctly excluded rather than pooled into a meaningless
catch-all. Suite: 202 passing.

**Result: a null.** Lambeth, real network: 63.59% (4 coarse classes)
-> **61.30%** (20 fine classes), 3/6 windows, paired t-test t=-0.8780,
**p=0.4201**.

**All three Table 7.2 data gaps are now closed and individually
tested** - road class (8), date (4), POI (20) - and all three are
nulls. Combined with the fact that every model-side lever is also
exhausted, the gap-closing plan's own hypothesis ("the model is not the
bottleneck, the data is") is now only partly supported: the *network
source* was decisively the bottleneck (+11.59 points, p=0.0010), but
the remaining feature-granularity gaps the same table implies are not.

## 2026-09-03 - Full Table 7.2 parity: every input class at once, and it is the WORST feature configuration

Individually the three closed data gaps were nulls; features can
interact, so the combination was worth one run - every input class the
paper lists, simultaneously: real OS network + POI 20-class + socio 8 +
road class 8 + date 4.

**Result: 63.59% -> 59.76%, 2/6 windows, paired t-test t=-0.9633,
p=0.3796.** Not significant, but it is the LOWEST of every
feature-variant tested, and directionally the worst. The combination
does not rescue the individually-null parts; it compounds them.

**The mechanism is almost certainly sample size, not feature quality.**
The light protocol trains on 6-11 instances per window. Full parity
adds 8 road-class + 3 date + 16 extra POI columns to an already ~40-wide
feature matrix - roughly 27 more parameters to fit per input dimension
from at most 11 examples. That is a textbook overfitting regime, and it
explains a pattern visible across this whole session: **every
feature-ADDING experiment since the network swap has been null or
negative, while the one that changed the DATA ITSELF (the network
source) produced the only large, significant gain.** More columns is
not more information when the training set cannot support them.

**Consequence for the plan**: `docs/gap_closing_plan.md`'s central
hypothesis ("the model is not the bottleneck, the data is") is now
resolved with a sharper answer than the plan anticipated. The data WAS
the bottleneck - but specifically the *fidelity of the road network*,
not the *breadth of the feature set*. Feature-set parity with the
paper's Table 7.2 is achieved and measured, and it does not help at
this data scale.

## 2026-09-03 - Weather re-tested on the real network: null, and the gap-closing plan is now fully executed

Weather was tested once before (-1.76 points, null) but on the OSMnx
network, BEFORE the network was known to be the dominant variable -
the same confound that made the 2019-protocol conclusion unsafe until
it was re-run. It earned one honest retest on the real network.

**Result: 63.59% -> 64.48%, +0.89 points, 3/6 windows, paired t-test
t=0.3260, p=0.7577.** Directionally positive (unlike the OSMnx test's
negative), and it produced the single highest first-window score
recorded on Lambeth (77.27%), but the effect is comfortably inside
noise. The earlier null stands with the confound removed.

**`docs/gap_closing_plan.md` is now fully executed - all four steps:**

| Step | Lever | Result |
|---|---|---|
| 1 | Road class (8 classes, was 0) | null, p=0.9725 (halves variance) |
| 2 | Date (4 classes, was 1) | tested with step 1 |
| 3 | POI 20-class (was 4) | null, p=0.4201 |
| 4 | Weather on the real network | null, p=0.7577 |
| 5 | Full Table 7.2 parity (all combined) | null, p=0.3796 - worst variant |

**Every input class Gao et al. list in Table 7.2 is now implemented and
measured.** Feature-set parity with the reference paper is achieved.
None of it moves AccHR@20 at this data scale.

## 2026-09-03 - Testing the overfitting diagnosis directly: more training data, same evaluation windows

The full-parity result (worst feature variant, ~27 extra columns fitted
from <=11 instances) diagnosed the constraint as SAMPLE SIZE, not
feature quality. That predicts the lever is training DATA, not columns.

**Why the existing dense-protocol run does not already test this**: it
trains on 20-128 instances and scores LOWER than the light protocol
(Lambeth 60.42% vs 63.59%) - but evaluates on entirely different dates
spread across 2021-2025. Training density and evaluation difficulty
move together there, so neither is isolated.

`scripts/run_ucl_comparison_dense_training_same_windows.py` holds
evaluation fixed and varies only training density: the same six
held-out windows, with training drawn from a dense stride-14 pool over
a longer (2021+) history. Built 102 pool instances vs the light
protocol's 6-11 - roughly a 10x increase. Leakage control was verified
explicitly before launching (a pool window is admitted only if its
entire target period ends before the evaluated window begins; the
boundary case was hand-checked, not assumed).

**A design flaw was caught mid-run and corrected.** Extending
START_DATE back to 2021-01-01 for more training history silently
shifted the stride-90 EVAL grid by 5 days (365 % 90 = 5), producing
held-out windows of 2023-07-10 etc. instead of the 2023-07-15 every
other Lambeth run uses. That defeats the experiment's whole purpose -
and, worse, would have been invisible in the output: the run completes
normally and produces plausible numbers, but `paired_significance.py`
merges on `held_out_start` and would have found ZERO matching windows,
turning any "comparison" into an unpaired contrast between different
evaluation periods. Fixed in
`run_ucl_comparison_dense_training_aligned.py` by anchoring the eval
grid at `EVAL_START_DATE = "2022-01-01"` (the light protocol's own
origin) while the training pool still spans 2021+.

## 2026-09-03 - Feature scaling: a real numerical defect, three fixes, and the defect turns out NOT to be the bottleneck

Researched the reference group's own public code
(github.com/ZhuangDingyi/STZINB - the STZITD repo points here, and its
London crash dataset is explicitly "available on request", confirming
exact replication is impossible from public sources). Two concrete
differences from this project: they train at **lr=1e-5** (vs this
project's 1e-2) and they scale model INPUTS by the training-set maximum
while leaving TARGETS raw - **they never z-score.**

**The defect this exposed is real and large.** Measured on real Lambeth
data: this project's z-scored features reach an absolute maximum of
**1179.63**. The cause is structural - a 99.98%-zero count column has a
near-zero standard deviation, so each rare non-zero becomes an enormous
z-score. This is the same condition that produced GAT
attention-parameter NaNs at lr=0.01 and forced the paper's encoder
order down to lr=2e-3.

**Three fixes, tested in order, each ruling out the next:**

1. **Max-value scaling** (their approach): bounds inputs to [-1, 1] but
   CRUSHES the variance of exactly the sparse columns that carry the
   signal - almost every value becomes 0. Measured 34.70% then 12.82%
   (below random) on Lambeth before being stopped. A decisive failure,
   and an instructive one: their scaling works on their data because it
   is not this sparse at their aggregation.
2. **log1p-then-z-score** (the textbook count-data transform):
   rejected WITHOUT a GPU run, by direct computation. With one non-zero
   in ~500 cells, both the mean and the standard deviation scale with
   that value, so any monotone transform leaves the z-score essentially
   unchanged (verified: identical absmax before and after log1p).
   **The extremeness comes from the SPARSITY, not the magnitude** - so
   no rescaling of values can fix it. This is the most transferable
   insight of the three.
3. **Clipped z-scoring (+/-5 sigma)**: the only fix that addresses the
   mechanism - typical values keep unit variance (verified: values
   within +/-5 pass through bit-identical, only ~0.03% are capped)
   while the pathological tail is bounded.

**Result: 63.59% -> 63.61%, 2/6 windows, paired t-test t=0.0133,
p=0.9899.** A textbook null - the two are indistinguishable.

**Conclusion, and it disproves the hypothesis that motivated the
work**: the numerical defect is real, measurable, and explains several
earlier symptoms (the NaN divergence, the encoder order's learning-rate
sensitivity) - but it is **not** the AccHR@20 bottleneck. The model in
this project's own encoder order was already absorbing those extreme
values without harm. Worth recording precisely because the defect
LOOKED like a smoking gun: |1179| inputs into attention softmax is
exactly the kind of thing that should matter, and it does not.

## 2026-09-03 - Where the Lambeth gap ACTUALLY comes from: metric precision, not model weakness

Stopped adding levers and analysed the per-window structure of the gap
instead. Two findings that reframe the entire comparison.

**1. The model already reaches the paper's number - on some windows.**
Lambeth per-window AccHR@20 for the best configuration:

| Window | AccHR@20 | Crashes in the 14-day window |
|---|---|---|
| 2023-07-15 | 75.76% | 42 |
| 2023-10-13 | 53.85% | 27 |
| 2024-01-11 | 57.50% | 27 |
| 2024-04-10 | **76.92%** | 36 |
| 2024-07-09 | 50.71% | 33 |
| 2024-10-07 | 66.79% | 41 |

**76.92% on one window matches Gao et al.'s reported Lambeth AVERAGE of
76.59%.** The mean is dragged to 63.59% by two windows in the low 50s.
Repairing only those two to the model's own demonstrated best would
give 75.04% - essentially their number. The model is not uniformly
weaker; it is inconsistent.

**2. The inconsistency tracks crash density, not anything about the
model.** Correlation between AccHR@20 and crash density (1 - zero rate)
across the six windows: **r = 0.731**. The two worst windows are
precisely the two sparsest (27 crashes each).

**The mechanism is metric precision.** AccHR@20 is computed per DAY and
averaged. At 27 crashes over 14 days (~1.9 crashes/day), a given day's
hit rate can only take the values 0, 1/2 or 1 - the metric has roughly
three attainable values per day. Averaging fourteen such coarse values
produces an estimate with large intrinsic variance, independent of
model quality. This is the same coarse-denominator problem this project
identified on 2026-09-01 (four architecturally distinct models scoring
an identical AccHR@20 on a 46-crash window) - it was solved then by
moving to multi-window evaluation, but multi-window averaging reduces
the variance of the MEAN without making any individual window's
estimate less coarse.

**Why this matters for the comparison, stated carefully.** It does NOT
prove the gap is illusory - the pooled mean is still genuinely below
theirs. What it establishes is that a substantial share of the
remaining difference is attributable to evaluation-window sparsity
rather than to ranking quality, and that the model demonstrably CAN
rank at their level when given a window with enough crashes to measure
it. For a write-up this is the difference between "our model is worse"
and "our model matches theirs on adequately-powered windows, and our
evaluation windows are on average less well-powered than theirs" - the
second is both more accurate and more defensible.

**What it rules out**: further model/feature search aimed at the mean.
Nothing in the architecture explains a 0.73 correlation with how many
crashes happened to occur in the test fortnight.

## 2026-09-03 - Rank blending (GNN x historical rate): a decisive negative, for an informative reason

AccHR@20 is a pure RANKING metric, and the paper's own Table 7.3 reports
a historical-average baseline at 47.99% on Lambeth where this project's
GNN reaches 63.59%. Two predictors that rank roads differently can beat
either alone - the standard result for ranking problems, and a
genuinely different METHOD rather than another hyperparameter. Built
`run_ucl_comparison_rankblend.py`: convert both predictors to per-day
ranks (raw scores are on incomparable scales), average as
`alpha * rank(model) + (1-alpha) * rank(history)`, with the historical
rate computed only from each window's own training instances.
alpha=0.5 was PRE-REGISTERED as the headline (parameter-free equal
weighting) specifically so the alpha curve could be reported without
selecting the best value by looking at the evaluation windows.

**Result on the first window: monotone degradation, so the run was
stopped rather than spending five more windows confirming it.**

| alpha | AccHR@20 |
|---|---|
| 1.00 (model only) | 75.76% |
| 0.75 | 59.55% |
| 0.50 (pre-registered) | 41.21% |
| 0.25 | 38.18% |
| 0.00 (history only) | 29.85% |

**Why it failed, which is the useful part.** The historical predictor
scores 29.85% here - far below the 47.99% the paper reports for its own
HA baseline, and barely above the ~20% a random top-20% selection
gives. The cause is the same extreme sparsity that has shaped this
whole investigation: at 99.98% zeros, the overwhelming majority of
segments have a historical rate of EXACTLY zero, so ranking them is
arbitrary tie-breaking rather than prediction. A historical-average
baseline needs enough non-zero history to discriminate; at this
segment-day granularity it does not have it.

Blending a strong predictor with a near-random one can only dilute, and
the perfectly monotone curve is exactly that. **This also retrospectively
explains why the paper's HA baseline outscores it**: their aggregation
evidently leaves more segments with non-zero history than this
project's does.

**A shape bug was caught and fixed before the real run**: the historical
rate averaged over the wrong axis (`y` is `[horizon, N]`, so a
per-segment mean needs axis=0, not axis=1), producing a
`(14,14)`-vs-`(14,11596)` broadcast failure. Now asserted against
`y_pred`'s own shape so it cannot silently recur.

## 2026-09-03 - hidden=42/42 does NOT replicate: a caught false positive, and why the cross-borough rule earns its cost

The capacity sweep found gat_hidden=42/gru_hidden=42 (the paper's own
stated size) best on Lambeth: 65.05% vs the 16/32 default's 63.59%, and
- matching the independently-diagnosed failure mode - it HALVED the
variance (std 5.22% vs 10.28%) and lifted the worst window from 50.71%
to 57.26%. That was a coherent story: the right mechanism, the paper's
own value, and a plausible mean gain.

**It did not replicate.** Westminster: **70.03% -> 60.06%**, losing
4 of 6 windows, a 10-point drop (p=0.2045).

**Pooled across both boroughs (n=12): 66.81% -> 62.55%, 5/12 windows,
paired t=-1.0910, p=0.2986.** Directionally negative. hidden=42/42 is
NOT an improvement; the Lambeth result was noise, and its accompanying
variance reduction was noise too.

**This is the cross-borough policy doing exactly the job it exists
for.** The Lambeth-only result had everything a false positive needs to
be convincing: a mechanism that matched an independently-derived
diagnosis (AccHR@20 correlates r=0.731 with crash density; 42/42
appeared to fix precisely the sparse windows), provenance from the
reference paper itself, and a favourable variance profile. Reported on
six windows from one borough it would have been a confident, wrong
claim. The rule cost roughly 20 minutes of GPU time and prevented it.

**It also re-confirms that 6 windows cannot resolve differences of this
size.** A +1.46-point Lambeth "gain" and a -9.97-point Westminster
"loss" are both within what this metric's window-to-window variance
produces at ~30 crashes per window. Nothing at this effect size should
be claimed from a single borough.

## 2026-09-03 - THE CHECK THAT SHOULD HAVE COME FIRST: a single sorted feature scores 56.89%

Asked the most basic question in applied machine learning, which this
investigation had never asked: **what does a trivial baseline score on
this metric?** `scripts/diagnose_trivial_baselines.py` sorts road
segments by ONE feature, on the identical six windows, with the
identical metric and leakage discipline. No training, no GPU.

| Ranking | AccHR@20 (Lambeth) |
|---|---|
| random (sanity control) | 18.59% |
| `u_degree` (node degree) | 18.36% |
| `collision_count_30d` (recent crash history) | 21.86% |
| `aadf_all_motor_vehicles` (traffic volume) | 22.17% |
| `length` | 29.29% |
| **`poi_amenity_count` (amenity density)** | **56.89%** |
| GNN, best config (40 features, GAT+GRU+ZIP) | 63.59% |
| Gao et al. reported | 76.59% |

**The random control lands at 18.59%, close to the 20% a top-20%
selection gives by construction - confirming the metric implementation
is sound and these numbers are on a trustworthy scale.**

**Three consequences, and they reframe the whole investigation.**

1. **The GNN's marginal contribution is +6.7 points over sorting by a
   single column.** That is a real improvement and it is not nothing -
   but the honest description of this model is "recovers a strong
   static land-use signal, plus a modest increment", NOT "learns
   spatiotemporal crash dynamics". Any write-up that reports 63.59%
   without this baseline overstates the modelling contribution, and an
   examiner asking "what does a dumb baseline get?" would expose it.

2. **It explains the pattern that has puzzled this entire session.**
   Every feature addition was null or negative, capacity increases hurt,
   more training data did not help. If most of the achievable signal is
   one static feature that the model already has, there is very little
   headroom left for the model to exploit - so additional inputs mostly
   add variance. The apparent "overfitting" diagnosis and this finding
   are the same fact viewed from two directions.

3. **It independently confirms why rank-blending with history failed.**
   `collision_count_30d` scores 21.86%, essentially random. Historical
   crash counts carry almost no rank information at this
   segment-day granularity, exactly as the 99.98%-sparsity argument
   predicted. The paper's own HA baseline reaching 47.99% is therefore
   strong evidence their aggregation differs materially from this
   project's.

**What this opens up.** At 6-11 training instances, a 40-feature GNN is
enormously overparameterised relative to a signal dominated by a
handful of static land-use columns. A deliberately SIMPLE model over
the strongest features may generalise better than the complex one -
which is the direction every negative result this session has been
pointing at without it being recognised.

### Follow-up: can a SIMPLE model beat the GNN? No - and the failure mode is the same one

Given that one sorted column reaches 56.89%, a deliberately simple
model seemed likely to generalise better than a 40-feature GNN trained
on 6-11 instances. Tested with parameter-free rank averages (no fitted
weights, nothing selectable on the evaluation windows):

| Combination | AccHR@20 |
|---|---|
| `poi_amenity_count` alone | **56.68%** |
| POI + length | 46.91% |
| POI + length + AADF | 42.84% |
| all 4 POI classes + length | 37.99% |
| all 4 POI classes | 33.57% |
| **GNN, best config** | **63.59%** |

**Every addition makes the simple model worse** - monotonically. Adding
the other three POI classes to `poi_amenity_count` costs 23 points.
This is the same mechanism seen throughout: diluting one strong signal
with weaker ones destroys rank quality, exactly as rank-blending with
near-random crash history did (75.8% -> 29.9%).

**The GNN beats the best simple baseline by +6.9 points and therefore
earns its complexity** - a defensible conclusion now supported by
evidence rather than assumed. The honest framing for a write-up is:
the model recovers a strong static land-use signal that a single sorted
column already captures, and adds a genuine but modest increment on top
of it. That is a real result, and reporting it alongside the baseline
table is considerably stronger than reporting 63.59% alone.

**Worth noting for the comparison**: Gao et al. report no comparable
single-feature baseline - their weakest comparator is a historical
average at 47.99%. Providing one, with a random control that validates
the metric scale (18.59%, against a theoretical ~20%), is a
methodological contribution beyond what the reference paper offers.

## 2026-09-03 - Architecture ensemble: the most promising result since the network swap (Lambeth), pending replication

Averaging structurally DIFFERENT models - hidden 16/32, hidden 42/42,
and a 2-layer variant - by per-day rank. Distinct from the multi-seed
ensemble that was a clean null (p=0.6881): that averaged five runs of
the SAME architecture, whose errors are highly correlated, and
averaging correlated errors cannot help. These three differ in capacity
and (for the 2-layer member) receptive field, which is the condition
under which ensembling actually reduces error.

**Deliberately NOT per-borough config selection.** The tempting
alternative - pick 42/42 for Lambeth and 16/32 for Westminster because
those scored best - would be selection on the evaluation windows,
optimistically biased and non-replicable. This applies ONE fixed rule,
identically to every borough and window, with no reference to test
performance.

**Lambeth result: 63.59% -> 65.37%, +1.79 points, 4/6 windows, paired
t=1.6197, p=0.1662.**

Not significant. But the per-window profile is the best seen since the
network swap, and differs in character from the session's many nulls:

| Window | Baseline | Ensemble | Delta |
|---|---|---|---|
| 2023-07-15 | 75.76% | 74.24% | -1.52 |
| 2023-10-13 | 53.85% | 55.77% | **+1.92** |
| 2024-01-11 | 57.50% | 63.33% | **+5.83** |
| 2024-04-10 | 76.92% | 76.28% | -0.64 |
| 2024-07-09 | 50.71% | 54.29% | **+3.57** |
| 2024-10-07 | 66.79% | 68.33% | **+1.54** |

The two losses are near-ties; the four wins are larger. That asymmetry
is the signature of a small genuine effect rather than noise, which
tends to produce symmetric wins and losses.

**Held to the same standard that caught the last false positive.**
hidden=42/42 also looked convincing on Lambeth (+1.46, halved variance,
matching an independently-diagnosed mechanism) and then lost 9.97
points on Westminster. Cross-borough replication is running before any
claim is made. If it holds, n=12 may reach significance; if it does
not, it joins the nulls.

### Resolution: the architecture ensemble is a NULL at full power

Tower Hamlets completed the three-borough set and reversed the
promising two-borough picture:

| Borough | Baseline | Ensemble | Delta |
|---|---|---|---|
| Westminster | 70.03% | 72.46% | +2.43 |
| Lambeth | 63.59% | 65.37% | +1.79 |
| **Tower Hamlets** | 67.34% | **65.44%** | **-1.90** |

**Pooled n=18: 66.99% -> 67.76%, +0.77 points, 9/18 windows, paired
t=0.6045, p=0.5535; Wilcoxon p=0.6319.**

**9 wins out of 18 is exactly a coin flip.** The encouraging n=12
result (+2.11, 7/12, p=0.1046) was underpowered optimism, and adding
the third borough resolved it. The ensemble joins the nulls.

**This is the second time in one session that a two-borough positive
failed to survive the third** (hidden=42/42 was the first, winning on
Lambeth and losing 9.97 points on Westminster). The lesson is now
firmly evidenced rather than asserted: **at ~30 crashes per evaluation
window, differences of 2-3 points are indistinguishable from noise
regardless of how coherent the mechanism sounds.** Both candidates had
plausible mechanisms - decorrelated ensemble errors, the paper's own
hidden size - and both evaporated under replication.

**What was nonetheless learned, and is worth keeping**: the ensemble is
not harmful, and it produced this project's single highest borough
figure (Westminster 72.46%). But "highest observed" is not "reliably
better", and reporting it as an improvement would repeat exactly the
error the cross-borough policy exists to prevent.

## 2026-09-03 - Joint multi-borough training: a null, and it disproves the sample-size hypothesis

The most-motivated remaining idea: train ONE model on all three
boroughs simultaneously (block-diagonal graph, 32,736 segments, 281,368
edges) rather than three separate models, then evaluate per borough on
the identical six windows. Gao et al. train per borough, so this was a
genuine methodological departure rather than a replication detail.

**Result: null, trending negative.**

| Borough | Separate models | Joint | Delta | Gao et al. |
|---|---|---|---|---|
| Westminster | 70.03% | 68.36% | -1.67 | 68.98% |
| Tower Hamlets | 67.34% | 64.75% | -2.59 | 72.24% |
| Lambeth | 63.59% | 64.33% | +0.74 | 76.59% |
| **Pooled (n=18)** | **66.99%** | **65.81%** | **-1.17** | 72.60% |

6/18 windows, paired t-test p=0.2384. Joint training also erased
Westminster's margin over the reference paper (68.36% vs their 68.98%).

**Why this matters more than another null: it falsifies the
sample-size explanation.** Nine consecutive nulls this session shared
one apparent signature - too little training data - and joint training
was the cleanest available test of that. It triples the supervised
(node, target) pairs the shared weights see per step, from ~11k nodes
to ~32.7k. If data volume were the binding constraint, this should have
helped. It did not.

**A precision the first draft of the script's own docstring got wrong,
and which explains the result**: joint training triples the NODES per
instance, not the number of time windows. The model still sees only
6-11 distinct points in TIME. Combined with the earlier finding that a
single sorted column (`poi_amenity_count`) already reaches 56.89% of
the model's 63.59%, the picture is now coherent: the model is not
starved of spatial examples - it is starved of temporal variation, and
most of what it can learn is a static land-use ranking that more nodes
do not enrich.

**Consequences.** The remaining shortfall against Gao et al. is not
attributable to model architecture, feature set, feature scaling,
training-set size, capacity, ensembling, evaluation protocol, or
per-borough vs joint training - all now measured. The two factors with
direct evidence behind them are (a) the road-network source, already
exploited for the session's one significant gain (+11.59 points,
p=0.0010), and (b) evaluation-window sparsity, which correlates r=0.731
with the metric and is a property of the evaluation rather than of the
model.

## 2026-09-03 - CORRECTION: the crash-density explanation does not hold at full power

An earlier entry today ("Where the Lambeth gap ACTUALLY comes from")
reported a correlation of **r=0.731** between AccHR@20 and crash
density, and used it to argue that a substantial share of the gap to
Gao et al. is metric precision on sparse windows rather than model
quality. **That correlation was computed on Lambeth alone - six
points.**

Recomputed across all eighteen windows (three boroughs):
**r=0.247, p=0.3234 - not significant.**

**The explanation is therefore not supported**, and the earlier entry
overstated it. This is the third time in one session that a
six-window Lambeth pattern has failed to survive extension to the full
set (after hidden=42/42 and the architecture ensemble), and the first
time the casualty was one of this project's own explanations rather
than a candidate improvement. The lesson generalises: **n=6 is not
adequate to establish anything here, including diagnoses.**

**What remains true and is worth keeping.** Their evaluation windows
genuinely contain more crashes than this project's, computed from their
own Table 7.2 annual counts scaled to a 14-day window:

| Borough | Their ~crashes/14 days | Our observed mean |
|---|---|---|
| Westminster | ~67 | 43.5 |
| Lambeth | ~51 | 34.3 |
| Tower Hamlets | ~47 | 34.7 |

Their windows carry roughly 35-55% more crash events than this
project's. That is a real, documented difference in evaluation
conditions - their 2019 single-year data is denser than this project's
2022-2024 multi-year sampling at the same 14-day window length.

**But this project's own data cannot reliably quantify what that
difference is worth.** The regression of AccHR on crash count
(slope 0.00254/crash) rests on a correlation that is not statistically
significant, so extrapolating "what we would score at their density"
would be building on sand. The honest statement is: *their evaluation
windows are better powered than ours, by a measurable margin; the
effect of that on AccHR@20 is not established by our data.*

Anything stronger would repeat exactly the error this entry exists to
correct.

## 2026-09-03 - Metric-aware (top-k hinge) loss: a well-founded idea that is another clean negative

The training objective and the evaluation metric have never matched:
models are fitted with a distributional NLL (`zero_inflated_poisson_nll`)
but scored with AccHR@20, a pure ranking metric. At 99.98% sparsity the
NLL is dominated by correctly predicting true zeros and is nearly
indifferent to whether the rare crashes outrank them. The
learning-to-rank literature's standard answer is a metric-aware
surrogate loss.

A plain pairwise hinge had already been tried (2026-09-01) and hurt
(48.29% at weight 0.1, 40.76% at weight 1.0). The diagnosis for WHY was
that it weights all ~23,000 daily (positive, negative) pairs equally,
when a day holds ~2 positives against ~11,594 negatives - so nearly
every pair concerns segments deep in the bottom 80%, which AccHR@20 is
completely indifferent to.

`top_k_hinge_loss` was written to fix exactly that: AccHR@20 has one
decision boundary - does a crash segment clear the k-th highest
predicted score, k = 20% of N - so the loss penalises only that
shortfall, `max(0, margin - (score_positive - kth_highest))`. Gradient
reaches the positives and, via `torch.kthvalue`, the segment on the
boundary (the same mechanism as max-pooling). Verified before running:
perfect ranking scores 0, inverted scores 10.1.

**Result: monotone degradation with weight.**

| Weight | AccHR@20 |
|---|---|
| 0.0 (baseline) | 63.59% |
| 0.1 | 63.59% |
| 0.5 | 60.44% |
| 1.0 | 58.64% |

At weight 0.1 the result is identical to baseline to four decimal
places - the term is too small to matter. Above that it actively hurts,
in proportion to how much it is weighted.

**Interpretation.** Two well-motivated ranking losses have now failed,
and the shared explanation is most likely the sparsity again: with
about two positives per day, the top-k objective is estimated from a
handful of examples per training step and is extremely noisy, while the
NLL - however indirect - is estimated from all ~11,596 segments. Trading
a well-conditioned indirect objective for a badly-conditioned direct one
is a poor exchange at this data scale, even though the direct objective
is the "right" one in principle.

This is the same structural finding as everything else this session:
the constraint is the sparsity of the positives, not the choice of
architecture, features, scaling, capacity, training-set size, or now
loss function.

## 2026-09-03 - Re-reading the paper's exact Section 7.4.1 surfaces two real, quantified differences

Responding directly to "what do they have that we don't" by re-reading
the paper's experimental setup section verbatim rather than working
from memory of it.

### 1. Their zero-inflation rate is 95.72%/96.71%/96.28% (Westminster/
Lambeth/Tower Hamlets). Ours, measured on the same real data this
project uses, is **99.97%** (Westminster).

That is not a small gap - their non-zero rate (~4.3%) is roughly 140x
ours (~0.03%). Segment-count differences (their ~4,822-5,659 links vs
our ~11,098) account for at most a 2x dilution, nowhere near 140x. The
remaining, much larger gap most likely comes from TCR being a
severity-weighted continuous score aggregated somewhat differently than
a raw daily crash-occurred indicator, or from a coarser-than-daily
target construction not fully specified in the text available. This is
now a documented, quantified, OPEN discrepancy - not resolved, but no
longer asserted-and-forgotten. It means their model's positives are
much easier to find in expectation, independent of model quality.

### 2. Their exact training bundle - tested as one combined recipe -
diverges on this project's data, and diagnoses WHY every previous
isolated test of its ingredients looked weak.

Full bundle: gat_layers=2, hidden=42, weight_decay=0.01, epochs=20,
early_stopping_patience=10, at this project's winning spatial_first
encoder order and lr=0.01 (never assembled together before - each
ingredient had only been tested alone or confounded with the
temporal_first/lr=0.001 result). Result: predictions collapse to a
near-constant narrow band (0.0003-0.0017, std=0.0002) - the "predict
near-zero for everyone" degenerate optimum - and AccHR@20 drops to
19.54%/10.54% on the first two windows, matching the exact underfitting
signature already documented for `epochs=20` and premature early
stopping in isolation.

**This is not a new negative result - it is the mechanism, finally
isolated, behind why hidden=42 alone looked like a false positive and
why the paper's whole training recipe cannot be transplanted intact.**
Their epochs=20 + patience=10 assumes something this project's
protocol does not have: enough i.i.d.-ish training samples for early
stopping to be a stabiliser rather than a coin flip. Confirmed directly
from their own text: training/validation/test are ALL FROM 2019, ratio
6:2:2 - a single calendar year sliced far more densely in time than
this project's 6-11 quarterly walk-forward instances. If their model is
trained on anything resembling daily-shifted windows within that 219
day training slice, their effective training-instance count is
plausibly 150-200+, against this project's 6-11 - a ~20-30x difference
that no feature or architecture change can substitute for.

**Action taken**: launched
`run_ucl_comparison_dense_training_aligned.py` (built and verified
earlier, never previously run to completion) - same six evaluation
windows as every other Lambeth result, but a stride-14 training POOL
back to 2021, giving 5-8x more training instances per window than the
light protocol while holding evaluation fixed. This is the cleanest
available test of the training-density hypothesis the paper's own
Section 7.4.1 just motivated.

### Dense-training-aligned: completed, and the training-density hypothesis is ALSO a null

`run_ucl_comparison_dense_training_aligned.py` completed on Lambeth -
same six evaluation windows, 64-96 training-pool instances per window
(vs 6-11 in the light protocol), stride-14 back to 2021.

**Result: 63.59% -> 60.21%, -3.38 points, 3/6 wins, p=0.5780. Null,
trending negative.**

| Window | Light (6-11 inst.) | Dense (64-96 inst.) | Delta |
|---|---|---|---|
| 2023-07-15 | 75.76% | 75.76% | 0.00 |
| 2023-10-13 | 53.85% | 64.10% | +10.26 |
| 2024-01-11 | 57.50% | 65.83% | +8.33 |
| 2024-04-10 | 76.92% | 50.64% | **-26.28** |
| 2024-07-09 | 50.71% | 51.31% | +0.60 |
| 2024-10-07 | 66.79% | 53.59% | -13.10 |

An 8-17x increase in training instances neither reliably helps nor
hurts - three real wins (including two of +8-10 points) are cancelled
by two real losses (one of -26 points). This is a coin flip with high
per-window variance, not a trend.

**This closes the training-density hypothesis from BOTH directions
tested this session.** More training examples via spatial breadth
(joint multi-borough training, same 6-11 time instances, 3x the nodes
per instance: null, p=0.2384) and more training examples via temporal
density (this test, same segments, 8-17x the time instances: null,
p=0.5780) both fail. The paper's Section 7.4.1 training regime
(2-layer, hidden=42, wd=0.01, 20 epochs, early-stopping patience=10)
plausibly needs BOTH more instances AND a training distribution shaped
like a single dense calendar year, not either alone - or the
architecture genuinely does not transfer to this project's walk-forward
protocol regardless of instance count. Either way, "just get more
training data" - however it is obtained - is not the missing piece.

**Where this leaves the investigation.** Every mechanism this session
could construct a specific, falsifiable hypothesis for has now been
tested: architecture (heads, layers, hidden, decoder, encoder order),
features (road class, date, POI-20, weather, full Table 7.2 parity),
scaling (three variants), loss function (two ranking losses), ensembling
(three kinds), training data (spatial breadth and temporal density),
and evaluation protocol (light/dense/2019-replication). The two
remaining open items are not fixable by further modelling: the
~140x zero-inflation-rate gap (see previous entry) and the specific
construction of their TCR target, neither of which is resolvable
without access to their actual processed dataset (confirmed
"available on request" via github.com/ZhuangDingyi/STZINB, not public).

## 2026-09-03 - The exact TCR formula, found - it rules out severity weighting as the zero-inflation explanation

Thesis Definition 1 (Section 7.2.1), quoted exactly: "yit = sum_{j=1}^{3}
C^t_{i,k} x l_j", where l_j is a severity weight (slight=1, serious=2,
fatal=3) and "the crash point is allocated to its closest road" - i.e.
TCR is a severity-weighted SUM of crash counts per road per day, built
from the same raw STATS19 source this project uses, via the same
nearest-road snapping approach. Confirms they built their own target
from raw data ("after aggregating and processing the risk data") rather
than using some other pre-cleaned dataset.

**This rules out severity weighting as the explanation for the
zero-inflation gap.** The formula only changes the VALUE of an
already-nonzero cell (1, 2, or 3 instead of a plain count) - it cannot
turn a genuinely crash-free segment-day into a nonzero one. Their
95.72%/96.71%/96.28% zero-inflation vs this project's measured 99.97%
(Westminster) must come from somewhere else.

**Narrowed, not resolved**: their Table 7.2 reports ~4,822-5,659 road
segments per borough; this project's real OS Open Roads network yields
~11,098 for Westminster - about 2.3x more segments. Coarser
segmentation (e.g. consolidated named streets vs individual TOID-level
links) would raise density, but 2.3x cannot explain a ~140x gap in
non-zero rate on its own. The remaining mechanism - most likely
something in their exact segmentation/consolidation method - is not
resolvable from the published text and is exactly the kind of detail
their actual processed dataset (requested via email this session, see
docs/project_management.md) would settle immediately.

Author contact confirmed from the published paper's title page
(Accident Analysis & Prevention, 2024): corresponding author Xiaowei
Gao (xiaowei.gao.20@ucl.ac.uk), co-authors James Haworth
(j.haworth@ucl.ac.uk) and Huanfa Chen (huanfa.chen@ucl.ac.uk), both
UCL. Their own Data Availability statement: "Data will be made
available on request." A data request email was drafted for the user
to send.

## 2026-09-03 - Directed-vs-undirected segment duplication: a well-reasoned hypothesis, disproven on real data before it cost GPU time

Their Westminster road count (4,822) is close to half this project's
directed-edge count (11,098) - suggesting this project might be
splitting every two-way road into two separate "segments" (one per
direction), each independently eligible to be a zero, while they count
each physical road once. If true, merging directions would mechanically
raise the non-zero rate: a road with a crash on one direction and none
on the other goes from "1 nonzero row + 1 zero row" to "1 nonzero row",
shrinking the zero-row count without losing any real information.

**Tested directly on real Westminster data before committing to a full
graph rebuild** (this project's own "verify before launching" rule):
grouped all 11,098 directed edges into 5,460 undirected pairs by shared
endpoints, summed collision counts per day within each pair, and
recomputed zero-inflation.

**Result: 99.97% -> 99.94% - a rounding error, not the hypothesised ~2x
effect.** The reasoning had a real flaw: at this sparsity, the
probability that BOTH directions of the same physical road crash on the
SAME day is astronomically small, so the "1 nonzero + 1 zero -> 1
nonzero" collapse this hypothesis relied on almost never actually
occurs. Merging directions roughly halves both the total cell count AND
the zero-cell count together, preserving nearly the same ratio.

**Disproven cheaply, in minutes, with no GPU time spent** - exactly
what the "verify on real data before launching" rule exists to prevent:
a full undirected-graph rebuild and retrain would have cost hours to
reach the same null. The ~140x zero-inflation gap to Gao et al. remains
unexplained by directed/undirected duplication, segment count, or the
TCR severity-weighting formula (all now checked). The 5,460 vs their
4,822 similarity in raw segment COUNT appears to be coincidental, not
evidence of a shared consolidation method.

**Process note, disclosed rather than hidden**: the diagnostic script
above ran on the GPU/CPU while the learning-rate sweep
(`run_ucl_comparison_lr_sweep.py`) was still mid-run, and crashed it
with a CUDA OOM at the lr=0.015 candidate - the exact mistake
[[greyspot-one-gpu-job-at-a-time]] exists to prevent, repeated despite
having written that rule down. The two lost candidates (lr=0.015,
lr=0.02) were relaunched alone immediately after.

## 2026-09-03 - The lr=0.01 gap, closed: it was never swept, and turns out to already be near-optimal

Flagged earlier: lr=0.01 for spatial_first (this project's winning
encoder) was the untouched original default, never itself swept - only
ever used as a control value while sweeping the OTHER encoder order.
Swept 0.003/0.005/0.01/0.015/0.02 on Lambeth, holding everything else at
the current best config.

| lr | AccHR@20 |
|---|---|
| 0.003 | 25.00% (underfits - same signature as every other too-low-lr collapse this session) |
| 0.005 | 63.64% |
| **0.01 (current default)** | **63.59%** |
| 0.015 | 60.61% |
| 0.02 | 64.40% |

Best candidate (0.02) vs default: **+0.81 points, p=0.6129, 3/6 wins -
null.** The four values from 0.005 to 0.02 all land within about 1
point of each other and none clears significance; only below ~0.005
does performance collapse. **0.01 sits in a broad, flat optimum by
coincidence, not by any prior tuning** - worth stating plainly rather
than implying it was always known to be right. No free improvement
available here; this closes the one remaining "never actually checked"
hyperparameter gap in this project's standing configuration.

## 2026-09-03 - Reading the raw STATS19 file directly (not just via pandas summaries)

Per the user's request to "read its real file to know better" - opened
`data/raw/collision-2023.csv` directly and read genuine rows rather than
only querying aggregate statistics.

**Confirmed clean**: `local_authority_district` is `-1` ("data missing
or out of range") on every row sampled - a deprecated/unpopulated
legacy field. This project correctly never uses it, filtering instead
on `local_authority_ons_district`, which is populated with real ONS
codes throughout. No risk of accidentally filtering on a dead column.

**Confirmed correct (re-checking a past, already-shelved feature)**:
`junction_detail` carries -1 as its own "missing" sentinel on
roughly half the sampled rows. `redistribute_junction_crashes`'s
`at_junction = (junction_flag > 0) & ...` correctly treats BOTH `0`
(genuinely not at a junction) and `-1` (missing) as "leave this crash
alone" - the junction-redistribution feature's earlier negative result
(p=0.0432, tested 2026-09-02) is a real finding, not an artefact of
mishandling this sentinel.

**Two genuine DfT fields this project has never used, found by reading
raw rows rather than a column list**:
- `enhanced_severity_collision` / `collision_injury_based`: a newer,
  separate DfT severity reclassification, `-1` where not (yet)
  reclassified under the newer scheme. Distinct from the standard
  `collision_severity` (1/2/3) this project uses throughout.
- `collision_adjusted_severity_serious` / `_slight`: a PROBABILISTIC
  severity split (e.g. one real row: 0.05522 / 0.94478, summing to
  ~1.0) - plausibly a model-based reclassification addressing known
  inconsistency in how police forces historically coded "serious" vs
  "slight". Not used anywhere in this pipeline.

Neither is pursued as a new lever: severity weighting was already
tested via the exact TCR formula (Gao et al.'s own Eq. 7.1) and found
null on the real network (63.50% vs 63.59%, see earlier entry) - a
different, more granular severity signal is unlikely to move a
mechanism already shown not to matter. Recorded for completeness and
future reference rather than chased further.

## 2026-09-03 - Deep target analysis: the predictability ceiling, and a real gap it exposed in the feature set

A statistical analysis of the TARGET itself (rather than the model),
asking what any model could achieve on this data.

### Finding 1: crashes essentially never repeat at the same place

Lambeth 2022-2024, 3,085 collisions at 3,068 distinct coordinates.
**Only 1.1% of crashes occur at a coordinate that saw any other crash
in three years; not one coordinate saw three.** Crash locations are
near-unrepeating at fine spatial resolution - "learn where crashes
happened and predict them there again" has a hard, low ceiling.

### Finding 2: spatial concentration is real but modest

At LSOA granularity (201 LSOAs with >=1 crash): the top 20% of LSOAs
hold 48.8% of all crashes; the top 50% hold 82.6%. Risk is concentrated,
but not overwhelmingly.

### Finding 3: an honest oracle at LSOA granularity scores ~42%

Ranking LSOAs by 2022-2023 crashes and measuring what fraction of 2024
crashes land in the top 20%: **41.76%**. Using ALL data including the
test period (a cheating oracle) gives only 48.85%. Year-over-year
Spearman rank correlation of LSOA crash counts is 0.80 (2022 vs 2023),
0.71 (2023 vs 2024), 0.60 (2022 vs 2024) - **crash risk rankings
genuinely drift year to year**, which upper-bounds any history-based
model independently of architecture.

(Note the granularity caveat, stated so this is not over-read: these
are LSOA-level numbers, and the project's models operate on ~11.6k road
SEGMENTS. Finer granularity permits sharper concentration - a main road
and a cul-de-sac in the same LSOA differ enormously - which is why the
GNN's 63.59% exceeds these figures rather than contradicting them. A
segment-level oracle is the correct comparison and is queued.)

### Finding 4 (ACTIONABLE): the feature set is capped at 30 days of history

`ROLLING_WINDOWS = (7, 14, 30)` - the longest crash-history feature this
project has ever used is 30 days. At ~2-3 crashes/day over ~11.6k
segments, 30 days is far too sparse to estimate a segment's underlying
risk. Measured directly, walk-forward on the same six evaluation
windows, LSOA granularity as a cheap proxy:

| Lookback | Hit rate | Avg. crashes in history |
|---|---|---|
| 30d | 35.74% | 82 |
| 90d | 42.10% | 242 |
| 180d | 41.07% | 486 |
| **365d** | **46.08%** | 1,007 |
| 730d | 46.00% | 2,150 |

**A 365-day lookback is worth roughly +10 points over the 30-day cap**,
plateauing after that. Long-horizon crash history is also the single
strongest known road-level risk predictor in the safety literature.

**Why this differs from every other feature experiment this session**
(all null): those added a NEW data source - POI classes, weather, road
class, socio-demographics - to a model already short of training data.
This instead extends the HORIZON of the signal already known to be the
most important one present, and is backed by a direct measurement on
real data rather than a plausibility argument.

Built `scripts/run_ucl_comparison_longhistory.py` adding
`collision_count_90d` and `collision_count_365d`. Two correctness
issues were caught and fixed BEFORE launching, per this project's
verify-first rule:
1. A 365d rolling window needs a year of prior data, so the feature
   TABLE now starts 2021-01-01 while the INSTANCE GRID stays anchored at
   2022-01-01 (`TABLE_START_DATE` vs `START_DATE`, with an assertion) -
   otherwise the stride-90 grid would shift 5 days (365 % 90) and every
   paired comparison would silently match zero windows.
2. `YEARS` was still `[2022, 2023, 2024]`, so the new 2021 table rows
   would have contained zero crashes and silently undercounted every
   365-day sum. Now `[2021, 2022, 2023, 2024]`.

### AADF name-propagation: a data-quality fix that measurably HURTS

The audit finding (`has_aadf` real-data coverage only 1.56% of
Westminster segments, 173/11,098) was correct, and the fix worked
exactly as designed - coverage 1.56% -> 28.35%, verified end-to-end on
real data before launching, with sensible recovered values
(5,920-19,933 vehicles/day) and `edge_index` provably unchanged.

**The model got substantially worse: 62.89% -> 52.47%, -10.42 points,
1/6 windows, paired t=-2.2402, p=0.0752** (Wilcoxon p=0.1250).

| Window | Real AADF only | Name-propagated | Delta |
|---|---|---|---|
| 2023-07-15 | 75.76% | 47.12% | **-28.64** |
| 2023-10-13 | 53.85% | 53.85% | 0.00 |
| 2024-01-11 | 53.33% | 47.08% | -6.25 |
| 2024-04-10 | 76.92% | 66.03% | -10.90 |
| 2024-07-09 | 50.71% | 51.90% | +1.19 |
| 2024-10-07 | 66.79% | 48.85% | -17.94 |

**The mechanism, and it is now a recognisable pattern.** A propagated
value is a road-NAME average, identical across every segment sharing
that name - so filling 27% of segments with name-level averages
replaces *sharp, segment-specific* variation with *flat, road-level*
variation. AccHR@20 is a within-day RANKING metric: it depends entirely
on distinctions between segments, and assigning ~2,973 segments a
handful of repeated values destroys exactly those distinctions. The
1.56% of segments carrying real, genuinely segment-specific
measurements were more useful *as a sparse signal* than 28% of segments
carrying a smoothed one.

This is the third independent instance of the same mechanism this
session: junction redistribution (spreading one crash across up to 8
arms, significantly negative, p=0.0432), rank-blending with
near-random crash history (75.8% -> 29.9%), and now AADF smoothing.
**Any transformation that trades sharpness for coverage loses on a
top-k ranking metric at this sparsity.** That generalisation is now
well-evidenced enough to state as a finding in its own right, and it
retro-explains several earlier nulls.

The implementation, tests (207 passing) and diagnostic are kept -
`propagate_aadf_by_road_name` is correct code answering a real question;
the answer is simply "no". Coverage is not the binding constraint.

## 2026-09-03 - LONG-HORIZON CRASH HISTORY: the first real improvement of the session (Lambeth, pending replication)

Adding `collision_count_90d` and `collision_count_365d` to a feature set
that had always been capped at 30 days.

**Lambeth: 63.59% -> 70.29%, +6.70 points, 5/6 windows,
paired t=1.8268 p=0.1273, Wilcoxon p=0.0938.**

| Window | 30d cap (baseline) | +90d/365d | Delta |
|---|---|---|---|
| 2023-07-15 | 75.76% | 71.67% | -4.09 |
| 2023-10-13 | 53.85% | 60.26% | **+6.41** |
| 2024-01-11 | 57.50% | 61.67% | **+4.17** |
| 2024-04-10 | 76.92% | 80.77% | **+3.85** |
| 2024-07-09 | 50.71% | 73.93% | **+23.21** |
| 2024-10-07 | 66.79% | 73.46% | **+6.67** |

Not yet significant at n=6, but the profile is unlike every null this
session: **one small loss (-4.09) against five wins including +23.21**,
and the largest gain lands on the window that had been the WORST of the
six (50.71%), which is where a better risk estimate should help most.

**Why this worked when ~18 other experiments did not.** Every previous
feature experiment added a NEW data source (POI 20-class, weather, road
class, socio-demographics) to a model already short of training data,
and every one was null. This adds no new source at all - it extends the
HORIZON of the signal already known to be most important. And it is the
opposite of the transformation that has failed three times today
(junction redistribution, rank-blending, AADF name-propagation): a
365-day count is still fully **segment-specific**, so it SHARPENS the
per-segment risk estimate rather than smoothing it across neighbours.
On a within-day top-20% ranking metric, sharpness is everything.

It was also predicted quantitatively before being built, not
rationalised afterwards - the LSOA-granularity lookback measurement
(30d 35.74% -> 365d 46.08%) said roughly +10 points were available from
horizon alone.

**Held to the same standard that has already caught two false positives
today** (hidden=42/42 and the architecture ensemble both won on one
borough and died on the third). Westminster replication launched
immediately; Tower Hamlets follows. No claim is made until all three
report.

### REPLICATED on Westminster - and the pooled result is SIGNIFICANT

**Westminster: 70.03% -> 75.39%, +5.36 points, 5/6 windows**
(t=1.5291 p=0.1868, Wilcoxon p=0.1562 - individually underpowered at
n=6, as every borough-level test here is).

**POOLED n=12 (Lambeth + Westminster): 66.81% -> 72.84%, +6.03 points,
10/12 windows, paired t=2.4845 p=0.0303, Wilcoxon p=0.0342.**

This is the **first statistically significant model-side improvement of
the entire investigation**, and only the second significant finding
overall after the OS Open Roads network swap (+11.59, p=0.0010).

It also survives the exact test that killed two candidates earlier
today. hidden=42/42 won Lambeth (+1.46) then lost Westminster (-9.97);
the architecture ensemble won two boroughs then lost Tower Hamlets and
collapsed to p=0.5535 at n=18. This one gains on BOTH boroughs tested
so far, with 10/12 windows.

**Westminster now beats Gao et al. by 6.41 points (75.39% vs 68.98%)** -
up from a 1.05-point margin.

Tower Hamlets is running to complete n=18. It needs 72.24% to match the
paper; its baseline is 67.34%, so a gain comparable to the other two
boroughs would put it at or near parity. No sweep is claimed until it
reports.

### CONFIRMED ON ALL THREE BOROUGHS - the pooled benchmark is beaten

Tower Hamlets: **67.34% -> 75.37%, +8.03 points, 6/6 windows** (its
2024-07-09 window reached 86.03%, the highest single window of the
entire investigation).

**FULL THREE-BOROUGH RESULT:**

| Borough | Baseline | + long history | Delta | Gao et al. | Verdict |
|---|---|---|---|---|---|
| Westminster | 70.03% | **75.39%** | +5.36 | 68.98% | **beats by 6.41** |
| Tower Hamlets | 67.34% | **75.37%** | +8.03 | 72.24% | **beats by 3.13** |
| Lambeth | 63.59% | **70.29%** | +6.70 | 76.59% | below by 6.30 |
| **POOLED** | **66.99%** | **73.69%** | **+6.70** | **72.60%** | **beats by 1.08** |

**Pooled n=18: paired t=3.8313, p=0.001337; Wilcoxon p=0.001579;
16/18 windows won.**

This is the strongest and most robust finding of the investigation
alongside the network swap, and it clears the primary success criterion
set in `docs/gap_closing_plan.md` §4: *"pooled AccHR@20 across all three
boroughs exceeds 72.60%, with paired significance testing on n=18
windows."* Pooled is 73.69% at p=0.0013.

**Honest qualifications, stated up front rather than buried:**

1. **Lambeth still trails by 6.30 points** (70.29% vs 76.59%). The
   pooled win is real but is carried by Westminster and Tower Hamlets;
   this is NOT a clean sweep of all three boroughs and must not be
   described as one.
2. **The comparison is not perfectly like-for-like.** Gao et al.
   evaluate within 2019 on a 6:2:2 split; this project uses multi-year
   expanding-window walk-forward across 2022-2024, which is a stricter
   protocol. Their reported zero-inflation (95.72/96.71/96.28%) also
   differs from this project's measured 99.97%, unexplained after
   checking their TCR formula, segment consolidation and protocol. Both
   caveats are documented in `docs/ucl_benchmark_results.md` and neither
   is resolved by this result.
3. **Individual boroughs are underpowered at n=6** (p=0.13-0.19 each).
   The significance is a pooled n=18 result, which is the correct unit
   given the metric's known per-window noise.

**Why this one replicated when two others today did not.** hidden=42/42
and the architecture ensemble each won on one or two boroughs and died
on the next. This gained on **all three**, with 16/18 windows - the
profile of a mechanism rather than a fluctuation. And the mechanism was
predicted quantitatively before the experiment was built (LSOA-proxy
lookback measurement: 30d 35.74% -> 365d 46.08%), not rationalised
afterwards.

### CORRECTION (same day): "beats the benchmark" OVERSTATES the result - it is a statistical TIE

The entry above reports the pooled figure as "beats by 1.08" and the
per-borough figures as "beats by 6.41 / 3.13". **Those statements are
point-estimate comparisons and should not have been phrased as wins.**
Tested properly against Gao et al.'s published figures:

| Borough | Ours | 95% CI | Gao et al. | one-sample p | Verdict |
|---|---|---|---|---|---|
| Westminster | 75.39% | [66.54%, 84.25%] | 68.98% | 0.1218 | **tie** |
| Tower Hamlets | 75.37% | [67.04%, 83.71%] | 72.24% | 0.3781 | **tie** |
| Lambeth | 70.29% | [62.03%, 78.56%] | 76.59% | 0.1074 | **tie** |
| **Pooled** | **73.69%** | **[69.71%, 77.66%]** | **72.60%** | **0.5730** | **tie** |

**Every confidence interval contains the paper's corresponding figure.**
Bootstrap probability that this project's pooled score genuinely exceeds
theirs: **71.8%** - better than a coin flip, nowhere near a claim.

**The distinction that was being blurred, and it matters:**

- **p=0.001337 is this project's new model vs its OWN previous model.**
  That comparison is paired window-by-window, is highly significant, and
  stands: long-horizon crash history is a real +6.70-point improvement.
- **It is NOT a test against Gao et al.** No such paired test is
  possible - their per-window results are not published, only three
  point estimates. Comparing against a fixed constant with n=6 (or
  n=18) windows at a per-window std of ~8 points has nowhere near the
  power to resolve a 1-6 point difference.

**The correct claim, and it is still a good one**: this project's
results are **statistically indistinguishable from** a peer-reviewed
UCL benchmark published in *Accident Analysis & Prevention*, on a
stricter evaluation protocol (multi-year walk-forward vs their
within-2019 6:2:2 split) and with a measurably sparser target (99.97%
vs their reported 95.72-96.71% zero-inflation). Matching that
benchmark under harder conditions is a legitimate, defensible
dissertation result. **Claiming to beat it is not supported by the
evidence**, and would be exactly the kind of overclaim the
cross-borough replication policy and paired-significance standard exist
to prevent - it would be inconsistent to apply that rigour to internal
comparisons and abandon it for the headline.

## 2026-09-03 - MULTI-YEAR history: the 365d result was not the ceiling, it was the start of a slope

The +6.70-point long-history win prompted the obvious follow-up: had
the horizon saturated? An LSOA-level proxy had suggested a plateau at
365 days. **That proxy was wrong, and checking it properly at SEGMENT
level was the single most valuable diagnostic of the session.**

### The diagnostic: segment-level lookback curve, history alone, no model

Ranking Lambeth's 11,596 segments purely by crash count over the prior
N days, scored with the project's own metric on its own six windows:

| Lookback | AccHR@20 | Segments with non-zero history |
|---|---|---|
| 30d | 21.86% | 78 / 11,596 (99.3% tied at zero) |
| 90d | 30.33% | 195 |
| 365d | 50.50% | 672 (94.2% tied at zero) |
| 730d | 60.26% | 1,177 |
| 1095d | **70.15%** | 1,565 (86.5% tied at zero) |

**No plateau - still climbing ~10 points per extra year at three
years.** And 1095-day history ALONE (70.15%) matched the entire GNN
(70.29%).

**The mechanism is mechanical, not mysterious**: the top-20% bucket
needs 2,319 segments. A 365-day window leaves only 672 with any
non-zero history, so ~1,650 of the selected segments are chosen by
arbitrary tie-breaking among zeros. Every extra year of history
replaces arbitrary ordering with real signal.

**Why the LSOA proxy misled**: Lambeth has 201 LSOAs versus 11,596
segments - roughly 58x denser per unit. LSOA-level history saturates
quickly precisely because it is not sparse. Generalising a sparsity
diagnostic across a 58x granularity change was the error; measuring at
the granularity the model actually uses was the fix.

### Implementation

`features.daily_features.attach_long_history_features` computes any
lookback from the SPARSE collision list via a cumulative-sum matrix
(~150 MB), rather than by rolling over the segment-day table. Rolling
over the table would have required extending it back to 2016 -
~11.6k segments x ~2,557 extra days = ~30M additional rows - which does
not fit in this machine's memory. With the cumsum approach, horizon
length is essentially free.

Data: 2016-2020 extracted from DfT's 1979-2025 historical archive
(schema verified identical to the modern per-year files: 44 columns,
same names). Deep history spans 2016-2024 (10,416 Lambeth collisions);
**the TARGET is untouched at 2021-2024**, so evaluation stays
bit-comparable with every previous run.

Three correctness controls, all applied BEFORE the GPU run:
1. **A test asserts the cumsum implementation matches the existing
   rolling implementation exactly** on a shared 30-day window. An
   off-by-one in the window would otherwise be invisible in model output.
2. Real-data verification of feature population (0.67% -> 17.30%
   non-zero from 30d to 1825d, monotone accumulation asserted).
3. The first verification run emitted `Only 1096 days of collision
   history... longest windows are truncated`, meaning the 1825d feature
   was complete for evaluation windows but PARTIAL for early training
   instances - the same column silently meaning different things across
   the walk-forward. Fixed by extracting 2016-2017 as well; the warning
   is gone and history depth is now 1,827 days against an 1,825-day
   maximum.

### Result (Lambeth)

| Comparison | Delta | Wins | p (t-test) | p (Wilcoxon) |
|---|---|---|---|---|
| vs 365d version | **+9.15** | **6/6** | **0.0092** | 0.0312 |
| vs session-start 30d cap | **+15.85** | **6/6** | **0.0300** | 0.0312 |

**Lambeth: 63.59% -> 70.29% -> 79.44% in one session.**

| Window | 30d cap | +365d | +multi-year |
|---|---|---|---|
| 2023-07-15 | 75.76% | 71.67% | **78.79%** |
| 2023-10-13 | 53.85% | 60.26% | **75.64%** |
| 2024-01-11 | 57.50% | 61.67% | **71.39%** |
| 2024-04-10 | 76.92% | 80.77% | **82.69%** |
| 2024-07-09 | 50.71% | 73.93% | **89.29%** |
| 2024-10-07 | 66.79% | 73.46% | **78.85%** |

Standard deviation also fell from 10.28% to 5.59% - the model is not
merely better on average, it is markedly more stable across windows,
consistent with the tie-breaking explanation above.

**Lambeth now exceeds Gao et al.'s 76.59% on the point estimate
(79.44%). Stated rigorously it remains a TIE**: 95% CI [73.01, 85.87]
contains their figure, one-sample p=0.3059. Lambeth was the project's
worst borough, 13.00 points behind at the start of the day.

Westminster and Tower Hamlets replication running.

### CONFIRMED ON ALL THREE BOROUGHS - and this time it is a genuine WIN, not a tie

Tower Hamlets: 75.37% -> **83.93%**, +8.56 points, 6/6 windows.

| Borough | Ours | 95% CI | Gao et al. | one-sample p | Verdict |
|---|---|---|---|---|---|
| Westminster | 78.92% | [71.20, 86.63] | 68.98% | **0.0212** | **significantly better** |
| Tower Hamlets | 83.93% | [78.51, 89.34] | 72.24% | **0.0026** | **significantly better** |
| Lambeth | 79.44% | [73.01, 85.87] | 76.59% | 0.3059 | tie (higher, not resolvable) |
| **POOLED** | **80.76%** | **[77.61, 83.91]** | **72.60%** | **0.000042** | **significantly better** |

**The pooled 95% CI lies entirely above the paper's figure.** Earlier
today the same comparison was a tie (CI [69.71, 77.66] containing
72.60%); the difference is that the multi-year history raised the mean
by 7 points AND cut per-window variance, so the interval both moved up
and narrowed.

Against this project's own session-start baseline, paired on identical
windows: **66.99% -> 80.76%, +13.78 points, 18/18 windows,
t=6.4610, p=0.0000059** (Wilcoxon p=0.0000076).

**This supersedes the earlier same-day CORRECTION entry** ("beats the
benchmark OVERSTATES the result - it is a statistical TIE"). That
correction was right about the data it described (pooled 73.69%, CI
containing their figure). It is no longer the current state: the
multi-year result clears the bar the correction itself set. Both
entries are kept so the reasoning trail is visible rather than tidied.

**What is claimed, precisely:**
- Pooled and on two of three boroughs, this project's AccHR@20 is
  **significantly higher** than the figures Gao et al. report.
- On Lambeth it is higher (79.44% vs 76.59%) but **not significantly**
  so - stated as a tie.

**What is NOT claimed, and must not be:**
1. **This is not a paired test against them.** Their per-window results
   are unpublished, so the comparison is this project's 6 windows
   against their single fixed number. A paired test would be more
   sensitive; a one-sample test is what the available data supports.
2. **The protocols differ.** They evaluate within 2019 on a 6:2:2
   split; this is multi-year expanding walk-forward over 2022-2024.
   Neither result transfers directly to the other's setup.
3. **The ~140x zero-inflation discrepancy is still unexplained**
   (their 95.72-96.71% vs this project's measured 99.97%), after ruling
   out their TCR formula, segment consolidation, and protocol. If their
   target is constructed differently in some way not stated in the
   paper, the two numbers may not measure quite the same thing.
4. **Six windows per borough remains a small sample.** The dense
   38-window evaluation (`run_ucl_comparison_dense_eval.py`) is built
   and queued specifically to test whether this survives a more precise
   measurement.

The honest headline: **this project's model, evaluated on a stricter
protocol and a sparser target, scores significantly higher than the
published benchmark on the pooled comparison and on two of its three
boroughs.**

## 2026-09-04 - Dense 38-window evaluation: the sparse estimate HOLDS, but Lambeth remains a tie

Motivated by the user's observation that per-window variance, not the
mean, was what kept the Lambeth comparison inconclusive. The fix for an
imprecise estimate is more measurements, not a better model - so the
final model was re-evaluated on a stride-14 grid (38 non-overlapping
14-day target periods) instead of the stride-90 grid's six.

**stride=14 is the densest grid that keeps target periods
NON-OVERLAPPING** (HORIZON=14). A shorter stride would reuse the same
crash days across windows, correlating them - which narrows a
confidence interval without adding information. That would manufacture
significance rather than earn it, and was deliberately not done.

**The run was interrupted by session teardown at 31/38 windows**
(no checkpointing; resuming means redoing all 31, ~6 hours). n=31 is
reported as-is rather than restarted, since it already answers the
question.

**Result (Lambeth, n=31): 78.84%, std 7.56%, 95% CI [76.07, 81.62].**

| Estimate | n | Mean | 95% CI | vs Gao 76.59% |
|---|---|---|---|---|
| Sparse (stride 90) | 6 | 79.44% | [73.01, 85.87] | tie, p=0.3059 |
| **Dense (stride 14)** | **31** | **78.84%** | **[76.07, 81.62]** | **tie, p=0.1073** |

**Two conclusions, one positive and one negative:**

1. **The sparse estimate was not a fluke.** 79.44% (n=6) vs 78.84%
   (n=31) - the six quarterly windows happened to give an almost
   perfectly representative sample. The headline numbers elsewhere in
   this project, all computed on that grid, are trustworthy.

2. **More precision did NOT convert Lambeth into a win.** The CI
   narrowed a lot (+/-6.4 -> +/-2.8 points) exactly as the sqrt(n)
   argument predicted, but the mean also drifted slightly down, so
   76.59% still sits inside the interval. **Lambeth is a tie at n=31,
   and further windows would not obviously change that** - the gap is
   ~2.25 points against a per-window std of 7.56%.

An earlier projection in this session suggested n=38 would "just
clear" 76.59%. That projection was made at n=21 when the running mean
was 79.37%; it was fragile by construction (its lower bound cleared by
0.04 points) and it did not survive more data. Recorded here because
the prediction was stated explicitly and should be marked wrong.

### A real, mechanistic finding from the dense grid

The weakest windows cluster in a specific calendar period:

| Window | AccHR@20 |
|---|---|
| 2023-12-08 | 65.22% |
| 2023-12-22 | 61.82% |
| 2024-01-19 | 61.11% |

**Three of the five worst windows fall in the Christmas / New Year
period.** This is a plausible real effect rather than noise: holiday
traffic patterns (commuting collapses, night-time and retail-area
activity spikes) diverge sharply from the rest of the year, so a model
trained predominantly on normal weeks ranks that period poorly.

This also explains why the dense grid shows higher variance than the
sparse one: stride-90 sampled that period once, stride-14 hits it four
times. **The sparse grid was mildly lucky in its sampling**, and 78.84%
is the more honest figure - though the two agree closely.

Worth stating as a limitation with a mechanism in the write-up, and a
concrete avenue for future work (holiday/seasonal indicator features,
or a separately-calibrated holiday model).

## 2026-09-04 - A CONFIG-REBINDING BUG that invalidated several runs, and the correction of a correction

### The bug

A patch applied to `evaluate_config` in four scripts on 2026-09-03/04
introduced this inside the per-window loop:

```python
for held_out_idx in range(n - n_windows, n):
    ...
    config = dict(config)                      # rebinds the OUTER name
    wanted_cols = config.pop("feature_columns", None)
```

`config = dict(config)` reassigns the variable the loop reads from. On
the first iteration it copies the real config and pops
`feature_columns`; on every SUBSEQUENT iteration `config` is the
already-popped dict, so `wanted_cols` is `None` and **no column
subsetting happens**. Only window 1 used the intended feature subset;
windows 2-6 silently trained on the FULL feature set.

Verified in isolation (no GPU, no data): window 1 gets 3 columns, windows
2-6 get `None`.

Affected: `run_headtohead_stzitd.py`, `run_history_horizon_ablation.py`,
`run_tablestart_control.py`, `run_ucl_comparison_pruned.py`. Fixed by
binding to a new name (`window_config`) and threading that into the
training call.

### What it invalidated

| Result | Verdict |
|---|---|
| Feature-pruning test (null, p=0.3632) | **INVALID** - both arms ran identical features on 5 of 6 windows |
| Head-to-head SHORT-vs-LONG history | **INVALID** - "short" arms were short only on window 1 |
| Table-start control (80.55%) | **INVALID** - arm 1 used 35 features on windows 2-6 |
| Head-to-head OUR-vs-THEIR architecture | **VALID** - both arms carried the same bug, so they remain mutually comparable (+18.53 / +19.01) |
| Baseline 63.59%, longhistory 70.29%, multiyear 79.44% | **VALID** - separate scripts, each with its own FEATURE_COLUMNS, no subsetting anywhere |

### The correction of the correction

Earlier today this log recorded that the "+15.85 points from long crash
history" finding was WRONG and worth only +0.51. **That correction was
itself produced by the buggy comparison** and is hereby withdrawn.

The original three-run chain - 63.59% (30d cap) -> 70.29% (+90/365d)
-> 79.44% (+730/1095/1825d) - came from three independent scripts that
never used column subsetting, so those comparisons were always sound.

**Net: three successive positions on the same question, of which only
the first and third are supported.**
1. "+15.85 from history horizon" (original claim)
2. "no, +0.51, it was the table start" (WITHDRAWN - buggy comparison)
3. "+15.85 stands; the table-start confound is still untested" (current)

### What remains genuinely open

The table-start confound raised in position 2 is a REAL question that
the buggy control failed to answer: extending the table to 2021 to
support a 365-day window also gave the earliest training instances
complete (rather than truncated) rolling features. The fixed control is
now running to separate those two effects.

### Process lessons, recorded because they generalise

1. **Bit-identical outputs across "different" configurations are a bug
   signature, not a coincidence.** Five of six windows matching exactly
   was treated as metric granularity; it was the bug announcing itself.
   This project has now hit that same signature three times (the
   `per_physical_road` no-op, the NaN tie-break, and this).
2. **Reasoning from plausibility failed five times in a row here**
   (history, table start, feature mutation, stale baseline, CUDA
   non-determinism). The bug was found only by mechanically diffing
   exact inputs and by simulating the loop in isolation.
3. **There is no version control on this repo**, so no diff was
   available to find what changed. `git init` before any further work.

### RESOLVED: the table-start confound is NOT the cause - history horizon is

The fixed control (both arms short-features-only, differing ONLY in
feature-table start) settles the question the buggy run could not:

| Configuration | AccHR@20 |
|---|---|
| Original baseline (table 2022, short features) | 63.59% |
| Control arm 1 (table 2022, short features) | **63.32%** |
| Control arm 2 (table 2021, short features) | **63.57%** |
| Multi-year (table 2021, FULL history ladder) | **79.44%** |

**Table start is worth +0.25 points. History horizon is worth +15.87.**

Arm 1 reproducing the original baseline to within 0.27 points also
confirms the bug fix is correct and the original numbers are sound.

**The +15.85 long-history finding is therefore RESTORED**, and is now
better supported than it was before this episode: it previously rested
on a three-run chain with no negative control, and it now has an
explicit control showing that the co-varying change (table start) does
essentially nothing.

### The residual float-accumulation artefact, quantified

Arm 1 (63.32%) differs slightly from the original (63.59%) despite
provably bit-identical instance tensors - all 12 instances, all 30
columns, verified `maxdiff = 0.0`. The cause is that the control derives
its 30 columns via `x[:, :, col_idx]`, whose memory layout differs from
a natively-30-column array, so `np.mean`/`np.std` accumulate in a
different order. Measured effect on the fitted standardiser: `std` max
2.057e+03 vs 2.063e+03 (~0.3%).

**Amplified through 200 training epochs, that ~0.3% input difference
moves single windows by up to ~2.6 points (53.85% vs 56.41%) while
moving the 6-window mean by only 0.27 points.** Worth reporting as a
reproducibility caveat: per-window figures at this sparsity are
sensitive to bit-level numerical details, but window-averaged results
are stable. It is another instance of the same underlying property this
project has met repeatedly - a coarse top-k metric on very few positives
amplifies small numerical differences at the selection boundary.

### Where the day's three positions finally land

1. "+15.85 from history horizon" - **CORRECT** (restored, now with a control)
2. "no, +0.51, it was the table start" - **WRONG** (artefact of the config bug)
3. "the confound is untested" - superseded; it is now tested and negative

## 2026-09-04 - The OS Open Roads network claim, properly controlled: it SURVIVES

The project's strongest claim (+11.59 points pooled, p=0.0010) had the
same structural weakness as the long-history claim: the comparison
changed TWO things at once - road geometry AND segment count
(OSMnx 8,238 directed edges -> OS Open Roads 11,596, about 41% more).
A finer segmentation changes the ranking task itself, so part of the
"network effect" could have been a property of the metric.

**Control: score parameter-free rankers on BOTH networks.** A trivial
ranker has no model and cannot exploit better topology, so any gain it
shows is pure task difficulty.

First attempt, single random seed (Lambeth):

| Ranker | OSMnx | OS Open Roads | Delta |
|---|---|---|---|
| random | 16.36% | 21.29% | **+4.94** |
| length | 32.88% | 29.29% | -3.59 |
| degree | 19.83% | 24.91% | +5.08 |
| history 365d | 47.45% | 50.50% | +3.05 |
| GNN | 57.72% | 63.59% | +5.86 |

That random gain of +4.94 - nearly the whole GNN gain - looked like the
claim collapsing. **It was single-seed noise.** Re-measured across 200
seeds per network:

| | OSMnx | OS Open Roads | Delta |
|---|---|---|---|
| random (200 seeds) | 19.68% +/- 3.63 | 19.96% +/- 3.46 | **+0.28** |
| crashes captured | 206 | **206** | 0 |

**The metric is network-invariant**: random scores ~20% on both, as
theory requires, and both networks capture exactly the same 206 crashes
in the evaluation windows, so the denominators match and the two
evaluations were always directly comparable.

**Conclusion: the +5.86 (Lambeth) / +11.59 (pooled) network effect is
genuine model improvement, not task difficulty.** The claim stands,
and now has an explicit control behind it.

**Process note worth keeping.** A single-seed random baseline has a
per-seed std of ~3.6 points here, so one draw can land 5 points from
the truth. This diagnostic was itself underpowered and nearly produced
a false retraction of the project's best result - the same
single-sample trap that has recurred throughout this work. Baselines
need the same statistical care as the models they are compared against.

**Also worth noting**: `degree` (+5.08) and `history_365d` (+3.05) DO
gain on the finer network, which is expected and not a confound - node
degree is a genuinely different quantity on a different topology, and a
finer segmentation legitimately helps a history-based ranker localise
risk. Only `random` is the invariance test, and it passes.

## 2026-09-04 - Overfitting audit: the model PASSES the label-shuffle control

Three probes on the final 35-feature configuration (Lambeth, window
2023-10-13).

### Probe 3 (the decisive one): label shuffle

Retrain with the crash target randomly permuted ACROSS SEGMENTS,
features untouched. A model exploiting genuine crash signal must
collapse to the random baseline; one exploiting incidental structure
would not.

| | AccHR@20 |
|---|---|
| real targets | **75.64%** |
| shuffled targets | **23.08%** |
| verified random baseline | 19.96% +/- 3.46 |

**PASS.** 23.08% is within one standard deviation of random. The model
is learning crash patterns, not artefacts. This project had never run
this control; it is the strongest single validation of the headline
numbers to date.

**The first version of this control was broken, and the standing rule
caught it.** It permuted `y.shape[0]` - the 14-day horizon axis -
instead of the segment axis, so each segment's crash total was
unchanged and the model learned the identical spatial ranking. It
returned bit-identical scores (0.7564 for both arms), which by this
project's rule R3 (bit-identical across different configs = bug) was
treated as a defect rather than a result. Fixed to permute axis 1, and
an assertion now verifies the shuffle actually changed the target.

### Probe 2: feature-count ladder

| Features | AccHR@20 |
|---|---|
| 10 | 75.64% |
| 20 | 73.72% |
| 26 | **78.21%** |
| 35 (final model) | 73.72% |

**Ten features score as well as thirty-five, and 26 scores best.** On a
single window this is only a hint, but it is consistent with the
overfitting concern that motivated the audit: 35 columns fitted from
6-11 temporal instances, where every static column (POI, IMD, length)
is IDENTICAL across instances, so their effective sample size is closer
to the instance count than the node count.

Promoted to a Phase-1 ledger item rather than a footnote: if a reduced
feature set matches or beats the full one across all six windows, the
published model should be the simpler one.

## 2026-09-04 - weight_decay=0.01: the third false positive caught by cross-borough replication

The architecture ablation's only arm to BEAT the baseline was Gao et
al.'s own weight decay (0.01, vs this project's 0.0): Lambeth
79.44% -> 81.97%, **+2.53 points, 4 wins / 2 exact ties / 0 losses,
p=0.0800**. The "never worse on any window" profile looked stronger
than either previously-caught false positive.

**It does not replicate.**

| Borough | wd=0.0 | wd=0.01 | Delta | p |
|---|---|---|---|---|
| Lambeth | 79.44% | 81.97% | +2.53 | 0.0800 |
| Westminster | 79.12% | 77.65% | **-1.47** | 0.4885 |
| **Pooled (n=12)** | **79.28%** | **79.81%** | **+0.53** | **0.6792** |

7/12 windows - a coin flip. Not adopted.

### This is now a pattern worth reporting in its own right

| Candidate | Single-borough | Cross-borough outcome |
|---|---|---|
| hidden=42/42 | Lambeth +1.46 | Westminster -9.97 -> rejected |
| architecture ensemble | 2 boroughs positive, p=0.10 at n=12 | Tower Hamlets -1.90, p=0.5535 at n=18 -> rejected |
| weight_decay=0.01 | Lambeth +2.53, p=0.0800, 0 losses | Westminster -1.47, p=0.6792 pooled -> rejected |

**Three for three.** Every candidate that reached p ~ 0.05-0.10 on a
single borough has failed replication. That is not a run of bad luck -
it is the expected behaviour of a 6-window evaluation with ~30 crashes
per window, where the noise floor swamps effects below roughly 3
points. It is also a much stronger justification for the replication
protocol than citing convention: this project can show, empirically,
that single-borough results at this scale are uninformative.

### A reproducibility figure worth quoting

Westminster's wd=0.0 arm scored 79.12% here versus 78.92% in the
earlier multi-year run - identical configuration, different script, no
feature subsetting involved. **Independent runs of the same
configuration reproduce the 6-window mean to about +/-0.2-0.3 points.**

That sets a floor on interpretability: the weight-decay effect (+2.53
on Lambeth) is well above it, while the hidden-size (-2.87) and decoder
(-3.50) effects are only marginally so - which is consistent with both
of those testing as null.

### V9 closed at n=18: the effect regressed monotonically to zero

Tower Hamlets completed the set: 83.93% -> 81.06%, **-2.86 points**.

| Borough | Delta | p |
|---|---|---|
| Lambeth | +2.53 | 0.0800 |
| Westminster | -1.47 | 0.4885 |
| Tower Hamlets | -2.86 | 0.2610 |
| **Pooled n=18** | **-0.60** | **0.6061** |

**The estimate shrank monotonically as evidence accumulated: +2.53
(n=6) -> +0.53 (n=12) -> -0.60 (n=18).** That is the textbook signature
of a spurious finding regressing toward zero, and it is a cleaner
illustration than any of the previous two of why a p=0.08 at n=6 here
carries essentially no information. Worth using as the worked example
in any methodological write-up.

## 2026-09-04 - V6 pruning v2: the casualty columns are LOAD-BEARING (v1 said the opposite, and was wrong)

The original pruning test (2026-09-03) was invalidated by the
config-rebinding bug. Re-run against the FINAL 35-feature configuration,
with a per-window assertion that the subset actually applies:

| Config | AccHR@20 |
|---|---|
| FINAL (35 features) | 79.76% |
| pruned (30, casualty breakdown dropped) | **72.87%** |

**-6.89 points, 0/6 windows, paired t p=0.0014, Wilcoxon p=0.0312.**
Every window got worse.

**This reverses v1's conclusion entirely.** v1 reported -0.69, p=0.3632,
"null - the model already ignores these columns", with 5 exact ties. The
ties were the bug (both arms ran identical features); the real effect is
large, consistent, and significant.

**The reasoning that motivated the pruning test was wrong.** The
argument was that `n_fatal/serious/slight/pedestrian/cyclist_casualties`
are a redundant decomposition of `collision_count` - correlations
0.14-0.85 with it, each 2-50x sparser. Measured: they carry substantial
independent signal. Severity and road-user composition evidently
identify risk that the aggregate count does not.

**It also argues against the overfitting hypothesis.** If 35 features
were overfitting on 6-11 temporal instances, dropping 5 collinear
columns should have helped or been neutral. It cost 6.89 points. Read
alongside V5's label-shuffle PASS, the evidence is that this model is
underfit rather than overfit at its current feature count - which makes
V10 (the feature-count ladder) more interesting, not less: the
single-window hint that 26 features beat 35 now looks likely to be noise.

## 2026-09-04 - V10 feature ladder: 13 features match 35, and V6's interpretation was over-read

Full 6-window ladder, groups added in a fixed a-priori order, paired
against the 35-feature final model:

| Config | AccHR@20 | vs 35 | p |
|---|---|---|---|
| 13 (geometry + crash history) | 78.77% | -0.99 | 0.3473 |
| 18 (+ casualty breakdown) | 77.77% | -1.99 | 0.0987 |
| 21 (+ traffic exposure) | 76.15% | -3.61 | 0.1273 |
| 26 (+ POI) | **79.99%** | +0.23 | 0.8875 |
| 35 (+ socio-demographic) | 79.76% | - | - |

**Nothing on the ladder is significantly different from the full model.**

### Finding 1: the model can be 13 features instead of 35

Geometry (length, degrees, day-of-week) plus the full crash-history
ladder reaches 78.77% versus 79.76% - a 0.99-point gap at p=0.3473.
Every other feature group combined (casualty breakdown, traffic
exposure, POI, socio-demographic - 22 columns) is worth approximately
one point, and not significantly.

This is the better model to publish: simpler to describe, cheaper to
reproduce, far less exposed to the overfitting question, and it drops
the dependency on IMD/Census and OSM POI data entirely.

### Finding 2: V6's conclusion was over-read, and is corrected here

V6 found that dropping the 5 casualty columns FROM THE FULL 35 costs
-6.89 points (p=0.0014, 0/6 windows), and this log recorded that as
"the casualty columns are LOAD-BEARING". **That over-states what the
experiment showed.** The ladder finds that ADDING those same columns to
a 13-feature model makes it WORSE (-1.00). Both results cannot mean
"these columns carry ~7 points of signal".

The consistent reading: the specific 30-feature configuration V6 built
is unusually bad, and the -6.89 is a property of that configuration
rather than a measure of the columns' value. The claim is softened
accordingly.

### Finding 3: feature-group attribution is unreliable in this model

The ladder is non-monotonic - 78.77 -> 77.77 -> 76.15 -> 79.99 -> 79.76
- with intermediate configurations WORSE than either endpoint, and every
step inside the noise band. Adding exposure alone costs 2.6 points;
adding POI on top of it recovers 3.8. These groups interact, so no
single group's contribution can be read off in isolation.

**Consequence for the write-up**: report feature groups as a ladder with
confidence intervals, never as "feature X is worth Y points". The
earlier per-feature nulls in this project's ledger (road class, date,
POI-20, weather) should be read the same way - as "no detectable effect
in that configuration", not as evidence about those data sources
generally.

## 2026-09-04 - V7: road class is not null, it is significantly HARMFUL - and a third old null falls

Re-tested under the final configuration, and under the V10 13-feature
model, because V10 showed geometry+history is essentially the whole
model and road class is a geometry-type feature. The stated hypothesis
was that it might now HELP. It does the opposite.

| Config | without | with road class | Delta | wins | t-p | W-p |
|---|---|---|---|---|---|---|
| 13 features | 78.77% | 71.90% | **-6.87** | 0/6 | 0.0556 | 0.0312 |
| 35 features | 79.76% | 75.28% | **-4.49** | 0/6 | **0.0157** | 0.0312 |

The old ledger recorded road class as a clean null (63.59% -> 63.74%,
p=0.9725). **It is significantly harmful in both configurations, on
every window.**

### Three old nulls have now been overturned

| Old finding | Under the final config |
|---|---|
| "2 layers = null" | **-26.46, p=0.0003** |
| "encoder order = equivalent (p=0.2832)" | **-13.61, p=0.0135** |
| "road class = null (p=0.9725)" | **-4.49 to -6.87, p=0.0157** |

**The pre-2026-09-04 null ledger does not transfer to the final
configuration and must not be cited as-is.** Every null quoted in any
output needs re-measurement, or explicit labelling as "measured under
the superseded 30-feature / 30-day-cap configuration".

### The emerging mechanism, stated as a hypothesis not a finding

Three separate additions of sparse columns to the compact model have now
hurt: road class (-6.87), traffic exposure (-2.62 in the V10 ladder),
casualty breakdown (-1.00 in the ladder). The plausible mechanism is one
this project has measured before: z-scoring near-constant sparse columns
produces extreme standardised values (absmax ~1179 measured on real
Lambeth data), and each added sparse column dilutes the standardised
representation of the features that actually carry signal.

This is a HYPOTHESIS. It predicts that a scaling scheme robust to sparse
columns (max-scaling, or clipped z-scores) would reduce or remove the
harm from adding such features. Clipped z-scoring was tested once
(p=0.9899, null) but under the OLD configuration and WITHOUT the added
sparse features - so it is not evidence either way here. A proper test
is a legitimate future experiment, not a claim.

## 2026-09-04 - V8 multi-seed: the headline is ~1.7 points optimistic, and Lambeth is a tie

Every AccHR@20 figure in this project came from a single run at
seed=42. An earlier control had shown CUDA non-determinism does not move
the metric, but that tested repeated runs at the SAME seed - it says
nothing about sensitivity to the initialisation itself.

Final configuration, Lambeth, five seeds, same six windows:

| Seed | 6-window mean |
|---|---|
| **42 (the reported one)** | **79.44%** |
| 1 | 79.44% |
| 7 | 75.59% |
| 123 | 77.10% |
| 2024 | 77.19% |
| **Seed-averaged** | **77.75% +/- 1.67** |

**Spread 3.86 points. Seed 42 is the joint-HIGHEST of the five.**

### Consequences

1. **Lambeth is a tie, not a win.** Seed-averaged 77.75%, 95% CI
   [75.68, 79.82], versus Gao et al.'s 76.59%: one-sample p=0.1941.
   (It was already a tie at n=6 and n=31 single-seed; this confirms it
   on a third basis.)

2. **The seed-42 optimism bias is +1.69 points.** Applying that to the
   other boroughs - an EXTRAPOLATION, since they were single-seed and
   their seed variance is unmeasured - gives Westminster ~77.2%
   (+8.3 vs UCL), Tower Hamlets ~82.2% (+10.0), pooled ~79.1% (+6.5).
   The pooled claim survives comfortably; the margin shrinks.

3. **The "+/-0.2-0.3 reproducibility floor" measured earlier was
   measuring the wrong thing.** It compared same-seed runs of identical
   configs, which is float-accumulation noise only. **The correct floor
   for single-seed comparisons is ~4 points.**

4. **Effects under ~4 points cannot be interpreted from single-seed
   runs.** That includes hidden size (-2.87), decoder family (-3.50),
   and the weight-decay candidate (+2.53, already rejected on other
   grounds). These should be reported as "not distinguishable from
   seed noise", not as measured effects.

5. **Large effects are unaffected** and remain well clear of the noise
   band: layers 1->2 (-26.46), encoder order (-13.61), history horizon
   (+15.87), network source (+11.59), road class (-4.49 to -6.87),
   pruning to 30 features (-6.89).

### What must change in every output

- Report the headline as **seed-averaged with a +/-**, not seed 42's
  single number.
- Attach the ~4-point single-seed noise band to any comparison quoted.
- State explicitly that Westminster and Tower Hamlets seed variance is
  UNMEASURED, and that their adjusted figures are extrapolated.
- Never again quote a single-seed figure as a headline. Multi-seed is
  now mandatory for any number that appears in an output.

## 2026-09-04 - V11: seed bias is BOROUGH-SPECIFIC and flips sign - extrapolation would have been wrong

After V8 found seed 42 to be +1.69 points optimistic on Lambeth, the
obvious shortcut was to apply that correction to the other two boroughs
and adjust the pooled figure downward. **That would have been wrong.**

| Borough | Reported (seed 42) | Seed-averaged (n=5) | Seed-42 bias | vs Gao et al. |
|---|---|---|---|---|
| Lambeth | 79.44% | **77.75% +/- 1.67** | **+1.69 optimistic** | tie, p=0.1941 |
| Westminster | 78.92% | **80.03% +/- 1.40** | **-1.11 CONSERVATIVE** | **better, p=0.0001** |
| Tower Hamlets | 83.93% | running | ? | ? |

**The bias flips sign between boroughs.** Seed 42 was the joint-highest
of five on Lambeth and the LOWEST of five on Westminster. Spread is
comparable (3.86 vs 3.24 points) but direction is not transferable.

### Consequences

1. **Westminster's result strengthens under proper measurement**:
   80.03% +/- 1.40, 95% CI [78.29, 81.77], versus 68.98% -
   **p=0.0001**, up from p=0.0212 single-seed.

2. **The pooled figure cannot be seed-corrected by extrapolation.**
   Any adjusted pooled number requires all three boroughs measured;
   Tower Hamlets is running.

3. **A transferable methodological finding**: single-seed results on
   small held-out sets cannot be corrected by measuring the seed bias
   once and applying it elsewhere. The bias is a property of the
   (borough, seed) pair, not of the seed. Every reported figure needs
   its own seed distribution.

This is the second time today that refusing to extrapolate changed the
answer - the first being the network metric-confound control, where a
single-seed random baseline suggested (falsely) that the network gain
was a metric artefact.

### V11 COMPLETE - all three boroughs seed-averaged; the pooled claim SURVIVES

| Borough | Seed-avg (n=5) | 95% CI | Gao et al. | p | Verdict |
|---|---|---|---|---|---|
| Westminster | **80.03% +/- 1.40** | [78.29, 81.77] | 68.98% | **0.0001** | better |
| Tower Hamlets | **82.63% +/- 2.01** | [80.14, 85.12] | 72.24% | **0.0003** | better |
| Lambeth | 77.75% +/- 1.67 | [75.68, 79.82] | 76.59% | 0.1941 | **tie** |
| **POOLED** | **80.14% +/- 1.12** | **[78.74, 81.53]** | **72.60%** | **0.000115** | **better** |

**Seed-42 bias by borough: Lambeth +1.69 | Westminster -1.11 | Tower
Hamlets +1.30.** Direction is not consistent; magnitude ranges 1.1-1.7
points; per-borough seed spread ranges 3.24-4.74 points.

**The pooled figure was more robust than any individual borough**
(seed-42 pooled bias only +0.63) because the per-borough biases partly
cancel. That is a useful property to state: aggregate figures over
several evaluation regions are less seed-sensitive than any one region,
even when each region is individually noisy.

### Final honest position on the benchmark comparison

- **Two of three boroughs are significantly better** than the published
  figures, with seed-averaged means and proper confidence intervals.
- **Lambeth is a tie** on every basis tested: n=6 single-seed (p=0.3059),
  n=31 dense single-seed (p=0.1073), and n=5 seeds (p=0.1941).
- **Pooled is significantly better**: 80.14% +/- 1.12 vs 72.60%,
  p=0.000115, CI entirely above their figure.

The standing caveats are unchanged and must accompany any statement of
this: no PAIRED test against them is possible (their per-window results
are unpublished); the protocols differ (their within-2019 6:2:2 vs this
project's multi-year walk-forward); and the ~140x target-density
discrepancy remains unexplained.

## 2026-09-04 - C1: the architecture claim REPLICATES on a second borough (the first candidate all day to do so)

The "topology is decisive" claim rested on Lambeth alone at seed 42.
Replicated on Westminster:

| Arm | Lambeth | Westminster | Replicates? |
|---|---|---|---|
| baseline | 79.44% | 79.21% | - |
| **layers 1 -> 2** | **-26.46 (p=0.0003)** | **-43.21 (p=0.0257)** | **YES** |
| **encoder -> temporal_first** | **-13.61 (p=0.0135)** | **-9.42 (p=0.0046)** | **YES** |

Both effects are significant on both boroughs in the same direction.
**This is the first candidate today to survive cross-borough
replication** - three others (hidden=42/42, the architecture ensemble,
weight_decay=0.01) failed it.

### The 2-layer mechanism must be restated: it DIVERGES, it does not merely degrade

Westminster per-window: 0.708, 0.789, **0.142, 0.041**, 0.226, 0.254.
**Two of six windows fall BELOW the 0.20 random baseline**, and 0.041 is
far below anything a merely-worse model produces. That is the NaN
tie-break signature this project has catalogued before. Standard
deviation is 28.39 points versus Lambeth's 10.28.

So the correct claim is **"a 2-layer GAT is numerically unstable at this
data scale"** - degrading smoothly on Lambeth, diverging outright on
Westminster - NOT "2 layers costs 26 points". Reporting a point estimate
would hide the instability, which is the more useful finding: the
configuration is unusable, not just suboptimal.

This also explains why the original ablation's effects exceeded the
total head-to-head gap: a diverged arm's mean is not comparable to a
trained arm's.

### What is now properly supported

| Claim | Boroughs | Status |
|---|---|---|
| Encoder order matters (-9 to -14) | 2 | replicated, significant both |
| 2-layer is unstable/unusable | 2 | replicated, significant both |
| Network source (+11.59) | 3 | replicated |
| History horizon (+15.87) | 1 | **still Lambeth-only** |
| 13 features == 35 | 1 | **still Lambeth-only** |
| Road class harmful | 1 | **still Lambeth-only** |

## 2026-09-04 - C2: the feature ladder on a second borough - one claim survives, one is downgraded

| Config | Lambeth (vs 35) | Westminster (vs 35) | Replicates? |
|---|---|---|---|
| 13 features | -0.99 (p=0.347) | **-4.08 (p=0.092)** | direction yes, MAGNITUDE NO |
| 18 (+casualty) | -1.99 (p=0.099) | -3.74 (p=0.067) | direction yes |
| 21 (+exposure) | -3.61 (p=0.127) | **-5.38 (p=0.012)** | **YES** |
| 26 (+POI) | +0.23 (p=0.887) | -0.03 (p=0.987) | **YES - indistinguishable on both** |

### SURVIVES: socio-demographic features add nothing

The 26-feature model is statistically indistinguishable from the
35-feature model on BOTH boroughs (+0.23 and -0.03, p=0.887/0.987).
**The IMD/Census dependency can be dropped outright** - that is a clean,
replicated simplification, and it removes a data source with licensing
constraints and a 2011-vs-2021 boundary-vintage problem this project
has documented elsewhere.

### DOWNGRADED: "13 features == 35 features"

Lambeth showed -0.99 (p=0.3473) and the recommendation was written as
"prefer the 13-feature model for reproduction". Westminster shows
**-4.08** - four times larger, borderline significant (p=0.0919), and
at the edge of the ~4-point seed-noise band.

**Two boroughs disagreeing by 3 points on the same comparison means the
recommendation was over-confident on one borough's evidence.** It is
downgraded to: *a plausible further simplification requiring a third
borough before it is recommended.* The README's "prefer it" line must
be softened accordingly.

### Also replicated: the exposure/POI interaction is real

Adding traffic exposure alone HURTS on both boroughs (-1.62 Lambeth,
-1.64 Westminster, the latter significant at p=0.012), and adding POI on
top RECOVERS it (+3.84, +5.35). Two independent boroughs showing the
same non-monotonic shape makes this a genuine interaction rather than
noise - and it reinforces that single feature-group ablations cannot be
read in isolation.

## 2026-09-05 - C3: the road-class finding DOES NOT REPLICATE - retracting yesterday's claim

| Config | Lambeth | Westminster |
|---|---|---|
| 13 features + road class | **-6.87 (p=0.056)** | **-0.37 (p=0.861)** |
| 35 features + road class | **-4.49 (p=0.016)** | **+2.25 (p=0.380)** |

**The sign flips.** On Lambeth road class was significantly harmful at
35 features; on Westminster it is mildly helpful. Neither Westminster
result is close to significant.

### Retraction

The 2026-09-04 entry "V7: road class is not null, it is significantly
HARMFUL - and a third old null falls" is **RETRACTED**. What it actually
established is: *road class is harmful on Lambeth and has no detectable
effect on Westminster.* The general claim was made on one borough and
does not survive a second.

**This also weakens the broader statement in that entry** that "the
pre-2026-09-04 null ledger does not transfer". Two of the three
overturned nulls (2-layer, encoder order) DID replicate on Westminster
(C1). The third (road class) did not. So the correct statement is:
*some old nulls are wrong under the final configuration, and each must
be re-tested individually on at least two boroughs - the ledger cannot
be dismissed wholesale any more than it can be trusted wholesale.*

### The scoreboard for single-borough claims is now 1 for 6

| Claim | Second borough |
|---|---|
| hidden=42/42 | FAILED |
| architecture ensemble | FAILED |
| weight_decay=0.01 | FAILED |
| road class harmful | **FAILED** |
| 13 features == 35 | DOWNGRADED (-0.99 vs -4.08) |
| **architecture topology (C1)** | **REPLICATED** |

**One in six single-borough findings survived replication intact.**
That is the single most important methodological result this project
has produced, and it is now supported by six independent attempts
rather than asserted.
