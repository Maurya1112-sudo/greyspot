# Plan: closing the remaining AccHR@20 gap to Gao et al.

**Written 2026-09-02**, after a deep audit of the paper's own Table 7.2
("Data Characteristics by Region") against this project's actual
feature set. Supersedes ad-hoc lever-by-lever testing: this is a
prioritised plan derived from a specific, measured diagnosis.

---

## 1. The diagnosis

### 1.1 The pattern across everything tested so far

Two levers have produced statistically significant improvements this
session. **Both are data/feature changes. Neither is a model change.**

| Lever | Type | Result |
|---|---|---|
| Real OS Open Roads network | **data** | +11.59 pts, p=0.0010 |
| POI + socio-demographic features | **data** | +8.96 pts, p=0.0094 |
| GAT heads=3 vs 1 | model | +7.98 pts, p=0.036 (pre-features) |

Meanwhile **every model-side lever tried since has been null or
negative**: head count on top of features (p=0.56), GAT layers,
hidden dim, decoder family (ZIP/ZINB/Tweedie), target definition
(count vs TCR), rate link, ensembling (p=0.69), early stopping,
epoch budget, rank loss, input window. And every *evaluation-protocol*
hypothesis is now closed too: light vs dense vs the paper's own 2019
6:2:2 split (the last tested with the network variable finally set
correctly - it still does not explain the gap).

**Conclusion: the model is not the bottleneck. The data is.** Further
architecture search is not merely unpromising, it is contraindicated
by the evidence.

### 1.2 The specific, measurable data gap

The paper's Table 7.2 lists every input by source and **number of
classes**. Comparing directly:

| Input | Their classes | This project | Gap |
|---|---|---|---|
| Crash Counts | 1 | 1 | ✅ matched |
| Socio-demographic Characteristics | 8 | 8 | ✅ matched |
| **Roads (Ordnance Survey)** | **8** | **0** | ❌ **entirely absent** |
| **Point of Interest (OS)** | **20** | **4** | ❌ 16 short |
| **Date** | **4** | **1** | ❌ 3 short |
| Meteorological (Met Office) | 8 | 0 in best config | ❌ absent |

**The road-class finding is the headline.** OS Open Roads carries
`road_classification` (Motorway / A Road / B Road / Classified
Unnumbered / Unclassified / Not Classified / Unknown) and
`road_function` (A Road / B Road / Local Road / Minor Road /
Restricted Local Access Road / Local Access Road / Secondary Access
Road) - and `ingest.os_open_roads._map_highway` already derives an
OSM-style `highway` value from them. That column is **present on every
edge of the graph the best configuration already loads, and is simply
not listed in `FEATURE_COLUMNS`.** The model currently ranks roads
without knowing whether a road is a motorway or a cul-de-sac.

Road class is among the strongest known predictors of crash risk
(exposure alone differs by orders of magnitude between an A road and a
residential street), it is one of the paper's own six listed inputs,
and it costs nothing to add.

---

## 2. The plan, in priority order

Priority = (expected effect) x (confidence) / (cost). Each step is a
single isolated variable, tested against the current best
configuration on Lambeth first (the largest remaining gap), then
cross-checked on all three boroughs only if it clears significance -
per the project's established "test one borough first" policy.

### Step 1 - Road class features (8 classes) `[highest priority]`

- **Why**: the paper's own input, entirely absent here, data already in
  memory, strongest prior of any remaining item.
- **How**: one-hot encode the OS `road_classification` /
  `road_function` derived `highway` column into the segment-day table.
  Static per segment (like `length`), so it broadcasts across days.
- **Cost**: near zero - no new data, no new downloads.
- **Risk**: low. Worst case it is uninformative and adds ~8 sparse
  columns.

### Step 2 - Date features (4 classes)

- **Why**: the paper lists 4 date classes; this project uses 1
  (`day_of_week`). Crash risk has known month/season and
  weekend/weekday structure.
- **How**: add `month`, `is_weekend`, and a cyclical
  day-of-year encoding (sin/cos pair counts as the temporal position
  signal) alongside the existing `day_of_week`.
- **Cost**: near zero - computed from the date column already present.
- **Risk**: low.

### Step 3 - POI at full 20-class granularity

- **Why**: the paper uses 20 POI classes; this project aggregates to 4
  coarse OSM tag families. A "pub" and a "school" have very different
  crash-risk profiles and are currently summed into one
  `poi_amenity_count`.
- **How**: split the existing OSM tag values into ~20 semantically
  meaningful classes (school, pub/bar, restaurant, retail, transport
  interchange, hospital, place of worship, park, etc.) rather than
  fetching new data. The raw tag values are already downloaded and
  cached.
- **Cost**: moderate - requires re-running the POI counting step per
  borough, but no new network fetches if the raw POI cache is reusable.
- **Risk**: medium - more columns on a small training set could add
  noise; this is exactly what significance testing is for.

### Step 4 - Weather, re-tested on the real network

- **Why**: weather was tested once (null, -1.76 pts) but **on the
  OSMnx network, before the network was known to be the dominant
  variable** - the same confound that made the 2019-protocol
  conclusion unsafe until it was re-run. It deserves one honest retest.
- **Cost**: low - the ingestion module already exists.
- **Risk**: low.

### Step 5 (conditional) - combine whatever survives

- Run the winners together, then cross-borough confirm on all three
  boroughs with paired significance testing.

---

## 3. What this plan deliberately does NOT do

- **No further architecture/hyperparameter search.** Comprehensively
  exhausted; the evidence says the model is not the bottleneck.
- **No metric redefinition, window cherry-picking, or borough
  dropping.** The goal is to beat the benchmark legitimately or report
  honestly why it could not be beaten.
- **No tuning against the test windows.** Any constant (e.g. the
  `home_share=0.7` value) is chosen once on principle and reported as
  chosen, not swept until something wins.

---

## 4. Success criteria

- **Primary**: pooled AccHR@20 across all three boroughs exceeds
  72.60% (the paper's own pooled figure), with paired significance
  testing on n=18 windows.
- **Secondary**: Lambeth alone exceeds 76.59% and Tower Hamlets
  exceeds 72.24% (Westminster already exceeds at 70.03% vs 68.98%).
- **Reporting standard**: every step's result recorded in
  `docs/ucl_benchmark_results.md` whether positive, null or negative.

---

## 6. OUTCOME (added 2026-09-03 — plan fully executed)

Every step below was implemented, run on Lambeth (the largest gap), and
significance-tested against the current best configuration.

| Step | Lever | Result | p |
|---|---|---|---|
| 1 | Road class (8 classes, was 0) | 63.59% → 63.74% | 0.9725 null |
| 2 | Date (4 classes, was 1) | tested jointly with step 1 | — |
| 3 | POI 20-class (was 4 coarse) | 63.59% → 61.30% | 0.4201 null |
| 4 | Weather, re-tested on the real network | 63.59% → 64.48% | 0.7577 null |
| 5 | All combined (full Table 7.2 parity) | 63.59% → 59.76% | 0.3796 null (worst) |

**Feature-set parity with Gao et al.'s Table 7.2 is achieved.** Every
input class they list is now implemented and measured. None of it moves
AccHR@20 at this data scale.

### What this outcome revises about the plan's own hypothesis

The plan asserted "the model is not the bottleneck, the data is". That
is now resolved more precisely, and the plan was half right:

- **The data WAS the bottleneck — but specifically the FIDELITY of the
  road network**, not the breadth of the feature set. The real OS Open
  Roads network remains the single largest gain of the investigation
  (+11.59 points, p=0.0010, all three boroughs).
- **Feature BREADTH is not a bottleneck at all.** Every feature-adding
  experiment was null, and adding all of them together was the worst
  variant of the entire session.

### The mechanism, and what it points to next

Full parity adds ~27 columns to an already ~40-wide feature matrix,
fitted from at most 11 training instances. That is an overfitting
regime, not a feature-quality problem — which predicted that TRAINING
DATA, not columns, was the lever. Tested directly
(`run_ucl_comparison_dense_training_same_windows.py`, ~10x more
training instances on the same evaluation windows): results were mixed
(74.87 / 58.97 / 60.77 / 56.41 / 51.54), so sample size alone does not
close it either.

**The current leading hypothesis is FEATURE SCALING**, and unlike most
of the above it comes from the reference group's own published code
rather than from reasoning: github.com/ZhuangDingyi/STZINB scales model
inputs by the training-set maximum and leaves targets raw, and never
z-scores. This project has always z-scored — and its z-scored features
reach an absolute maximum of ~172 on real Lambeth data, because
z-scoring a 99.98%-zero count column turns each rare non-zero into an
enormous value. That is the same condition that caused GAT
attention-parameter NaNs earlier in the session. See
`run_ucl_comparison_maxscale.py`.
