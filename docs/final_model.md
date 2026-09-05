# The Greyspot final model — definitive specification

**Status as of 2026-09-03.** This is the canonical reference for the
best-performing configuration produced by this project. Every number
here comes from a run recorded in `reports/<borough>/`, and every
design choice is traceable to a dated entry in `docs/decision_log.md`.

Supersedes scattered configuration notes elsewhere. Where another
document disagrees with this one, this one is correct.

---

## 1. Headline result

| Borough | AccHR@20 | 95% CI | Gao et al. (2024) | Verdict |
|---|---|---|---|---|
| Westminster | **78.92%** | [71.20, 86.63] | 68.98% | ✅ **better, p=0.0212** |
| Tower Hamlets | **83.93%** | [78.51, 89.34] | 72.24% | ✅ **better, p=0.0026** |
| Lambeth | **79.44%** | [73.01, 85.87] | 76.59% | tie (p=0.3059) |
| **Pooled (n=18)** | **80.76%** | **[77.61, 83.91]** | **72.60%** | ✅ **better, p=0.000042** |

**The pooled 95% CI lies entirely above the paper's figure**, so this is
a genuine statistical result rather than a favourable point estimate.
Two of three boroughs are individually significant; Lambeth scores
higher (79.44% vs 76.59%) but not resolvably so at n=6.

**Against this project's own session-start baseline**, paired on
identical windows: 66.99% → 80.76%, **+13.78 points, 18/18 windows,
t=6.4610, p=0.0000059** (Wilcoxon p=0.0000076).

§7 lists four reasons this still must not be overstated — chiefly that
no *paired* test against Gao et al. is possible, and the protocols
differ.

---

## 2. Exact reproduction specification

Run: `python scripts/run_ucl_comparison_multiyear.py <Borough>`

> **A 13-feature model performs statistically the same** (78.77% vs
> 79.76%, p=0.3473 - see §3.4). It needs only OS Open Roads geometry and
> STATS19 crash history: no IMD/Census join, no OSM POI download, no
> AADF. For reproduction, prefer it.

### 2.1 Road network

| Property | Value |
|---|---|
| Source | **OS Open Roads** (Ordnance Survey, OGL), `oproad_gpkg_gb/Data/oproad_gb.gpkg`, layer `road_link` |
| Clipping | Real borough **polygon** (`borough_polygon_wgs84`), endpoint-inside test — *not* a bounding box |
| Filtering | `fictitious != True` links dropped |
| Graph | Directed multigraph; line-graph transform for segment adjacency |
| Scale | Westminster 11,098 directed edges / 4,026 nodes (≈5,549 undirected links) |
| Cache | `data/interim/<borough>_os_open_roads_graph.graphml`, loaded with `node_dtypes={"osmid": str}, edge_dtypes={"osmid": str}` (TOIDs are UUIDs, not ints) |

### 2.2 Data and target

| Property | Value |
|---|---|
| Crash source | STATS19 collision + casualty tables, DfT |
| Years — TARGET | **2021–2024** (`YEARS`); unchanged from every earlier run, so evaluation stays comparable |
| Years — DEEP HISTORY | **2016–2024** (`HISTORY_YEARS`), collisions only. 2016–2020 extracted from DfT's 1979–2025 archive (schema verified identical: 44 columns, same names). Used *solely* to compute the long-horizon features |
| Borough filter | `local_authority_ons_district` (*not* `local_authority_district`, which is `-1` throughout) |
| Snapping | Nearest edge, **no distance cap** (verified benign: median 2.3 m, 0.22% beyond 100 m, none beyond 500 m) |
| Target | Plain daily collision count per segment (**not** severity-weighted TCR — tested, null: 63.50% vs 63.59%) |
| Sparsity | 99.97% zero (Westminster, all segment-days) |

### 2.3 Features — 35 columns

| Group | Columns |
|---|---|
| Geometry / topology | `length`, `u_degree`, `v_degree` |
| Calendar | `day_of_week` |
| **Crash history (8)** | `collision_count`, `_7d`, `_14d`, `_30d`, `_90d`, `_365d`, **`_730d`, `_1095d`, `_1825d`** |
| Casualty breakdown (5) | `n_fatal_casualties`, `n_serious_casualties`, `n_slight_casualties`, `n_pedestrian_casualties`, `n_cyclist_casualties` |
| Traffic exposure (3) | `aadf_all_motor_vehicles`, `aadf_pedal_cycles`, `has_aadf` |
| POI (5) | `poi_shop_count`, `poi_amenity_count`, `poi_leisure_count`, `poi_tourism_count`, `has_poi` |
| Socio-demographic (9) | `imd_score` + 6 IMD domain scores, `population_density`, `has_socio_demographic` |

`ROLLING_WINDOWS = (7, 14, 30, 90, 365)` computed by rolling over the
segment-day table; `LONG_LOOKBACKS = (730, 1095, 1825)` computed by
`attach_long_history_features` from the **sparse collision list** via a
cumulative-sum matrix. The latter exists because rolling a 5-year window
over the table would require ~30M extra rows (out of memory), whereas the
cumsum approach makes horizon length essentially free. **Crash-history
horizon is by far the highest-value lever in the project (§4.2).**

**Standardisation**: z-score, fitted on training instances only,
applied to held-out. (Max-value scaling and log1p+z-score were tested
and rejected; clipped z-score at ±5σ was a clean null, p=0.9899.)

### 2.4 Architecture

`GATTemporal` (`src/greyspot/models/gat_temporal.py`)

| Parameter | Value | Notes |
|---|---|---|
| Encoder order | `spatial_first` | GAT per timestep → GRU over embeddings. The paper's GRU→GAT order is *equivalent* once each is given its own optimal lr (63.59% vs 60.93%, p=0.2832) |
| GAT layers | **1** | 2 layers tested, null |
| GAT hidden | **16** | |
| GRU hidden | **32** | |
| Attention heads | **3** | Matches the paper; heads=1 significantly worse (p=0.036) in the pre-features context |
| Residual | enabled | |
| Decoder | **Zero-Inflated Poisson** | ZINB and Zero-Inflated Tweedie both tested; ZIP best |
| Uncertainty | Split conformal, 90% target, calibrated on the last training instance |

### 2.5 Training

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | **0.01** (swept 0.003–0.02; a broad flat optimum, p=0.61 for the best alternative) |
| Epochs | **200**, fixed |
| Weight decay | **0.0** |
| Early stopping | **none** (patience=10 tested — premature stopping at these instance counts) |
| Device | CUDA (RTX 4060 Laptop, 8 GB) |

### 2.6 Evaluation protocol

| Parameter | Value |
|---|---|
| Scheme | Expanding-window walk-forward |
| Feature table span | 2021-01-01 → 2024-12-31 (deep history reaches to 2016 without extending the table) |
| **Instance grid anchor** | **2022-01-01** (`START_DATE`, assertion-enforced) |
| Input window | 20 days |
| Horizon | 14 days |
| Stride | 90 days |
| Held-out windows | 6 (2023-07-15, 2023-10-13, 2024-01-11, 2024-04-10, 2024-07-09, 2024-10-07) |
| Training instances | 6 → 11, expanding |

> **Load-bearing detail.** The feature table starts a year before the
> instance grid so the 365-day window is complete for every instance.
> Moving `START_DATE` itself would shift the stride-90 grid by 5 days
> (365 mod 90), and paired significance tests — which merge on
> `held_out_start` — would silently match **zero** windows while still
> printing plausible output. An assertion enforces the grid start.

---

## 3. Full results

### 3.1 AccHR@20 per window

| Window | Train inst. | Westminster | Tower Hamlets | Lambeth |
|---|---|---|---|---|
| 2023-07-15 | 6 | 70.41% | 81.60% | 78.79% |
| 2023-10-13 | 7 | 79.93% | 81.90% | 75.64% |
| 2024-01-11 | 8 | **89.85%** | 82.86% | 71.39% |
| 2024-04-10 | 9 | 73.72% | 77.14% | 82.69% |
| 2024-07-09 | 10 | 84.73% | 89.10% | **89.29%** |
| 2024-10-07 | 11 | 74.87% | **90.97%** | 78.85% |
| **Mean** | | **78.92%** | **83.93%** | **79.44%** |
| Std | | 7.35% | 4.71% | 5.59% |

Per-window variance fell substantially alongside the mean improvement
(Lambeth std 10.28% → 5.59%), consistent with the tie-breaking
mechanism in §4.2: fewer segments tied at zero means less of the
top-20% bucket is filled by arbitrary ordering, so less window-to-window
luck.

### 3.2 Full metric suite (mean across 6 windows)

| Borough | MAE | RMSE | ZR | **PICP** | MPIW |
|---|---|---|---|---|---|
| Westminster | 0.0005 | 0.0166 | 0.9997 | **0.9009** | 0.0009 |
| Tower Hamlets | 0.0005 | 0.0157 | 0.9998 | **0.9011** | 0.0009 |
| Lambeth | 0.0005 | 0.0145 | 0.9998 | **0.9012** | 0.0008 |

**Conformal calibration is essentially exact** — PICP lands within
0.1 percentage points of the 90% target on all three boroughs, and
*improved* over the previous best (0.8953–0.8961). This matters for the
government-facing product framing: the uncertainty intervals are
trustworthy, not decorative.

### 3.4 Feature-count ladder (V10, 2026-09-04)

Groups added in a fixed a-priori order, paired against the 35-feature
model over the same six windows:

| Config | AccHR@20 | vs 35 | p |
|---|---|---|---|
| 13 (geometry + crash history) | 78.77% | -0.99 | 0.3473 |
| 18 (+ casualty breakdown) | 77.77% | -1.99 | 0.0987 |
| 21 (+ traffic exposure) | 76.15% | -3.61 | 0.1273 |
| **26 (+ POI)** | **79.99%** | +0.23 | 0.8875 |
| 35 (+ socio-demographic) | 79.76% | - | - |

**Nothing on the ladder differs significantly from the full model**, and
the shape is non-monotonic (down, down, up), with intermediate
configurations worse than either endpoint. Feature groups interact, so
**no single group's contribution can be read off in isolation** - report
ladders, never "feature X is worth Y points".

### 3.4b The model does not beat a trivial baseline (added 2026-09-05)

This belongs at the top of any honest reading of the results.

| Ranker | Lambeth | Westminster | Tower Hamlets | Mean |
|---|---|---|---|---|
| Sort by cumulative crash count (~8 yr) | 83.20% | 81.25% | 87.37% | **83.94%** |
| Empirical Bayes (HSM method) | 82.11% | 81.93% | 87.48% | 83.84% |
| Same count, capped to this model's 5-yr horizon | 80.81% | 76.46% | 85.69% | 80.99% |
| **This model** | 77.75% | 80.03% | 82.63% | **80.14%** |

**At matched history depth the model and a parameter-free sort differ by
+0.85 points with mixed signs** — statistically indistinguishable. Given
three MORE years of history, the sort gains +2.95 while this model gains
−0.72 with rising variance (S5, p=0.7438).

So the graph structure, the 35 features, the conformal intervals and the
200 training epochs do not outperform sorting road segments by how often
they have already crashed. The uncertainty quantification remains
genuinely useful (PICP ≈ 0.901, §3.2) and a sort provides none — but the
*ranking* claim must be stated with this baseline alongside it.

### 3.5 Validation controls (all passed)

| Control | Result |
|---|---|
| **Label shuffle** (targets permuted across segments) | 75.64% -> **23.08%**, i.e. collapses to the random baseline (19.96 +/- 3.46). The model learns crash signal, not artefacts. |
| **Metric sanity** (constant / random / near-constant predictions) | 10.12% / 16.67% / 19.64% - a constant prediction scores BELOW random, so no score here is a tie-break artefact. |
| **Network invariance** (random ranker on both networks, 200 seeds) | 19.68% vs 19.96% (+0.28pp); both networks capture the identical 206 crashes. The network gain is model improvement, not task difficulty. |
| **Reproducibility floor** | Independent runs of an identical config reproduce the 6-window mean to **+/-0.2-0.3 points**. Differences smaller than this are not interpretable. |

### 3.3 Context: baselines on the same windows (Lambeth)

| Ranking | AccHR@20 |
|---|---|
| Random (metric sanity control) | 18.59% |
| Node degree | 18.36% |
| Recent crash history (30d) | 21.86% |
| Traffic volume (AADF) | 22.17% |
| Segment length | 29.29% |
| **Amenity density — best single feature** | **56.89%** |
| Best parameter-free rank-average combination | 56.68% |
| **This model** | **70.29%** |
| Gao et al. | 76.59% |

The random control at 18.59% against a theoretical ~20% validates the
metric implementation. **Gao et al. report no comparable single-feature
baseline** — their weakest comparator is a historical average at
47.99%.

---

## 4. How this configuration was reached

Two findings produced essentially all of the gain, and **both are data
findings, not architecture findings.**

### 4.1 The real road network (+11.59 points, p=0.0010)

Replacing an OSMnx-derived approximation with the genuine OS Open Roads
survey network. Significant on all three boroughs. A naive "coarser
network" proxy (distance-based node consolidation) was significantly
*worse* (p=0.0277), showing the benefit comes from real survey
topology rather than segment count.

### 4.2 Long-horizon crash history (+6.70 points, p=0.001337)

Crash-history features had always been capped at 30 days. Adding 90-
and 365-day rolling counts — no new data source, only a longer horizon
on the signal already present — gained 16/18 windows.

**It was predicted quantitatively before being built.** Analysis of the
target established that crashes almost never repeat spatially (3,085
Lambeth collisions across 3,068 distinct coordinates; 1.1% coordinate
repeat rate over three years) and that risk rankings drift year to year
(LSOA Spearman 0.80 / 0.71 / 0.60). Both imply 30 days is far too
sparse to estimate segment risk. A walk-forward lookback measurement at
LSOA granularity gave 30d → 35.74%, 90d → 42.10%, 365d → **46.08%**,
730d → 46.00% (plateau) — forecasting roughly +10 points from horizon
alone.

### 4.3 The mechanism, evidenced four ways

**On a within-day top-k ranking metric at 99.97% sparsity, per-segment
sharpness dominates coverage.**

| Transformation | Effect on sharpness | Result |
|---|---|---|
| Junction redistribution (1 crash → up to 8 arms) | destroys | negative, p=0.0432 |
| Rank-blend with near-random crash history | destroys | 75.8% → 29.9% |
| AADF name-propagation (coverage 1.6% → 28%) | destroys | **−10.42 points** |
| **Long-horizon history (30d → 365d)** | **preserves & improves** | **+6.70, p=0.0013** |

---

## 5. What does NOT work (do not retry without new reasoning)

Every entry below was tested with paired significance testing on
matched windows and, where positive-looking, cross-borough replication.

| Lever | Result |
|---|---|
| GAT heads on top of features | p=0.56, null |
| GAT layers (1 vs 2) | null |
| Hidden size 42/42 (the paper's own) | **caught false positive** — won Lambeth +1.46, lost Westminster −9.97 |
| Decoder family (ZINB, Zero-Inflated Tweedie) | ZIP best |
| Encoder order (paper's GRU→GAT) | equivalent at each order's own optimal lr, p=0.2832 |
| Learning-rate sweep (0.003–0.02) | null, p=0.6129 — 0.01 already near-optimal |
| Multi-seed ensembling | null, p=0.6881 |
| **Architecture ensembling** | **caught false positive** — won 2 boroughs, died on the 3rd, p=0.5535 at n=18 |
| Rank-blending (GNN × history) | catastrophic, monotone degradation |
| Two metric-aware ranking losses | negative, monotone with weight |
| Joint multi-borough training | null, p=0.2384 |
| Dense training pool (8–17× instances) | null, p=0.5780 |
| Feature pruning (drop 5 redundant columns) | null, p=0.3632 |
| Road class, date features, POI-20, weather | all null |
| **All features combined (full Table 7.2 parity)** | **worst variant of the session** |
| Feature scaling (max-value, log1p, clipped z) | catastrophic / rejected / null p=0.9899 |
| Junction redistribution + variants | negative, p=0.0432 |
| AADF name-propagation | negative, −10.42 points |
| Evaluation protocol (2019 6:2:2 replication) | does not explain the gap |

**The pattern**: every model-side lever is null. Only data-side changes
that preserve per-segment sharpness have ever worked.

---

## 6. Reproducing

```bash
python scripts/run_ucl_comparison_longhistory.py Westminster
```

Requires: `oproad_gpkg_gb/Data/oproad_gb.gpkg`, STATS19
collision/casualty CSVs for 2021–2024 in `data/raw/`, AADF counts,
LSOA boundaries + IMD 2019. Outputs
`reports/<borough>/ucl_multiwindow_{per_window,summary}_longhistory.csv`.

Runtime ≈ 12 min/borough on an RTX 4060 (8 GB). **Run one job at a
time** — concurrent runs caused four CUDA OOM crashes in one session.

Test suite: **207 passing** (`python -m pytest tests/ -q`).

---

## 7. Limitations — read before making any claim

1. **No *paired* test against Gao et al. is possible.** They published
   three point estimates, not per-window results, so the comparison is
   this project's 6 windows against a fixed constant (a one-sample
   test). A paired test would be considerably more sensitive; the
   one-sample form is what the available data supports. The significance
   reported in §1 is real but rests on this weaker design.

2. **Lambeth is a tie, not a win** (79.44% vs 76.59%, p=0.3059). Two of
   three boroughs are individually significant; the third is higher but
   not resolvably so. "Beats them on every borough" would be false.

3. **The comparison is not perfectly like-for-like.** Gao et al.
   evaluate within 2019 on a 6:2:2 split; this project uses multi-year
   expanding walk-forward across 2022–2024, a stricter test. Neither
   result transfers directly to the other's protocol.

4. **A ~140× target-density discrepancy is unexplained.** They report
   95.72 / 96.71 / 96.28% zero-inflation; this project measures 99.97%
   on the same STATS19 source. Their TCR formula (matched exactly from
   their Eq. 7.1), segment consolidation, and evaluation protocol have
   each been checked and ruled out. Their processed dataset is
   "available on request" and has not been obtained.

5. **Six windows per borough is a small sample.** A dense 38-window
   evaluation (`scripts/run_ucl_comparison_dense_eval.py`, stride=14 so
   target periods are adjacent and NON-overlapping) is the honest
   stress-test of whether this survives more precise measurement. Note
   that even there the windows are not fully independent — consecutive
   windows share nearly all their training data — so the effective
   sample size is somewhat below 38.

6. **CUDA training is not bit-deterministic.** Individual percentages
   are one-run results; paired comparisons are unaffected because both
   arms run under identical conditions.

7. **Most single-borough findings do not replicate.** Of nine findings
   that appeared significant on one borough, three survived a second.
   Effect size predicted the outcome for main effects (everything above
   the ~4-point seed-noise band replicated) but NOT for interactions —
   one +25-point interaction reversed sign between boroughs.

8. **Seed bias is borough-specific and flips sign.** +1.69 optimistic on
   Lambeth, −1.11 conservative on Westminster, +1.30 on Tower Hamlets. A
   seed's bias measured on one region cannot correct another.

9. **The evaluation metric is noisy by construction.** Windows contain
   only ~27–59 crashes each; per-window std is ~8 points. Differences
   under ~3 points are not resolvable at n=18.
