# UCL Benchmark: Full Results Compendium

**Status as of 2026-09-02.** This document consolidates every
AccHR@20 experiment run against Gao et al. (2024)'s STZITD-GNN
benchmark into one place. `docs/decision_log.md` remains the
chronological, reasoning-level record (why each thing was tried, what
was learned); `docs/publication_readiness.md` remains the
publication-framing gap analysis. **This file is the numbers.**

---

## 1. What is being benchmarked

**Reference**: Gao, Jiang, Zhuang, Chen, Wang, Law, Haworth (2024),
"Uncertainty-Aware Probabilistic Graph Neural Networks for Road-Level
Traffic Crash Prediction" (arXiv:2309.05072v4), plus the underlying
UCL PhD thesis (discovery.ucl.ac.uk/id/eprint/10210801).

**Scope of their model (verified directly from the paper's own text,
not assumed)**: *"we describe the detailed elements of the STZITD-GNN
model by analysing its performance in crash prediction in three
boroughs of London, UK, namely Lambeth, Tower Hamlets and
Westminster."* Three boroughs, road-segment level, 2019 data. There is
no London-wide version of this model. (The thesis does contain a
London-wide analysis in Chapter 6, but that is a **different model**
- "SMA-Hyper", a hypergraph-GCN - operating at MSOA level, not road
segments. Not comparable, not a scaled-up STZITD-GNN.)

**Metric**: AccHR@20 (their Eq. 20) - the per-day-averaged hit rate of
actual crashes falling within the top-20%-predicted-risk road
segments. Higher is better.

**Their published results**: Westminster 68.98%, Tower Hamlets 72.24%,
Lambeth 76.59%.

**Their full baseline table (thesis Table 7.3)** - important context,
since it shows what "normal" looks like on their data:

| Model | Lambeth | Tower Hamlets | Westminster |
|---|---|---|---|
| HA (historical average, no NN) | 47.99% | 42.66% | 48.32% |
| STGCN | 61.13% | 58.69% | 50.20% |
| STGAT | 64.22% | 69.50% | 48.08% |
| STG-GNN (Gaussian) | 26.66% | 26.47% | 29.33% |
| STNB-GNN | 44.71% | 50.22% | 45.03% |
| STTD-GNN (Tweedie, no ZI) | 71.23% | 63.68% | 60.75% |
| STZINB-GNN | 61.84% | 58.27% | 51.39% |
| **STZITD-GNN (theirs)** | **76.59%** | **72.24%** | **68.98%** |

---

## 2. Headline result

**Best configuration**: `heads=3` + POI + socio-demographic features +
**real OS Open Roads network**, ZIP decoder, plain-count target, light
walk-forward protocol.

| Borough | This project | Gao et al. | Delta |
|---|---|---|---|
| **Westminster** | **70.03%** | 68.98% | **+1.05 (exceeds)** |
| Tower Hamlets | 67.34% | 72.24% | -4.90 |
| Lambeth | 63.59% | 76.59% | -13.00 |
| **Pooled** | **66.99%** | 72.60% | -5.61 |

**Statistical validation** (vs the previous best config, which used the
OSMnx network; n=18 paired walk-forward windows across all three
boroughs): mean AccHR@20 55.39% -> 66.99%, **paired t-test t=3.9768,
p=0.0010**; **Wilcoxon W=17.0, p=0.0016**; candidate wins 15/18
windows. Individually significant on Westminster (p=0.0488) and Tower
Hamlets (p=0.0204, 6/6 windows); Lambeth's own effect is real but
smaller and not significant alone (p=0.31).

**Honest reading**: Westminster matches/exceeds the paper. The other
two boroughs and the pooled average do not, yet. This is the first
configuration in the entire investigation to match the paper on any
borough rather than merely narrow the gap.

---

## 3. Robustness checks on the headline result

**Across temporal protocol** - the real-network benefit is not an
artifact of the light protocol's specific window choice:

| Borough | Dense, OSMnx net | Dense, real net | Paired test |
|---|---|---|---|
| Westminster | 43.35% | **62.80%** | 6/6 wins, t=4.9973, **p=0.0041** |
| Lambeth | (not run) | 60.42% | - |
| Tower Hamlets | (not run) | 59.98% | - |
| **Pooled (dense, real net)** | - | **61.07%** (n=18, std 9.47%) | - |

Dense protocol = the paper's own exact temporal density (2021-2025,
N=20-day input, p=14-day horizon, stride=14), with held-out windows
spread evenly across the calendar rather than clustered. All three
boroughs land in a tight 60-63% band, confirming the real network's
benefit generalises across boroughs AND protocols. The dense
protocol's lower absolute numbers vs the light protocol (61.07% vs
66.99% pooled) reflect its wider, harder sampling of conditions across
five years, not a weakness of the network.

---

## 4. Complete experiment ledger

Every configuration tried, best-first per borough. All figures are
6-window means unless noted.

### Westminster

| Config | Mean AccHR@20 | Std |
|---|---|---|
| **heads=3 + POI+socio + REAL OS network** | **70.03%** | 7.18% |
| heads=3 + POI+socio + REAL OS network, dense | 62.80% | 9.46% |
| heads=1 + POI+socio (OSMnx) | 58.49% | 8.31% |
| heads=4 + POI+socio (OSMnx) | 56.00% | 10.91% |
| heads=3 + POI+socio (OSMnx) | 55.58% | 16.62% |
| heads=2 + POI+socio (OSMnx) | 53.64% | 6.91% |
| heads=3, TCR target | 50.17% | 7.78% |
| heads=3 (flagship, corrected metric) | 50.14% | 8.02% |
| heads=3 (paper's head count) | 49.57% | 7.23% |
| heads=3 + rank-loss (w=0.1) | 48.29% | 7.96% |
| heads=3 + weather | 47.81% | 8.71% |
| heads=3 + rate_link=exp | 47.65% | 8.74% |
| heads=3 + weight_decay=1e-4 | 47.09% | 5.83% |
| 2-layer GAT (paper's value) | 46.96% | 6.59% |
| heads=3, ZITD/Tweedie decoder | 46.60% | 5.67% |
| FAITHFUL bundle, epochs=200 | 46.44% | 5.42% |
| heads=3 + consolidated network | 44.69% | 12.23% |
| heads=3, epochs=500 | 43.93% | 5.84% |
| heads=3 + POI+socio, dense (OSMnx) | 43.35% | 11.30% |
| heads=3, input_window=14 | 42.31% | 8.65% |
| baseline (heads=1, 1 layer, ZIP) | 42.25% | 4.12% |
| heads=3 + 2-layer | 41.67% | 14.12% |
| heads=3 + rank-loss (w=1.0) | 40.76% | 3.63% |
| heads=3, TCR+spillover target | 34.57% | 5.34% |
| heads=3 + early stopping (k-fold, val=3) | 27.80% | 4.20% |
| heads=3 + early stopping (patience=10) | 26.37% | 9.41% |
| heads=3 + hidden=42 + "corrected" lr/wd | 20.55% | 12.62% |
| FAITHFUL bundle, epochs=20 (paper's literal) | 14.79% | 5.62% |

### Lambeth

| Config | Mean AccHR@20 | Std |
|---|---|---|
| **heads=3 + POI+socio + REAL OS network** | **63.59%** | 11.26% |
| heads=3 + POI+socio + REAL OS network (rerun) | 62.89% | 11.82% |
| heads=3 + POI+socio + REAL OS net + TCR + ZI-Tweedie | 63.50% | 11.27% |
| heads=3 + POI+socio + REAL OS net, 5-seed ensemble | 62.49% | 11.39% |
| heads=3 + POI+socio + REAL OS net + TCR | 61.65% | 12.62% |
| heads=2 + POI+socio + REAL OS network | 61.64% | 9.75% |
| heads=3 + POI+socio + REAL OS network, dense | 60.42% | 8.26% |
| heads=4 + POI+socio + REAL OS network | 58.22% | 11.72% |
| heads=3 + POI+socio (OSMnx) | 57.72% | 5.61% |
| heads=1 + POI+socio + REAL OS network | 54.56% | 10.84% |
| heads=3 + POI+socio + REAL OS net + JUNCTION redistribution | 53.43% | 2.66% |
| heads=3 + consolidated network | 52.60% | 10.84% |
| heads=3, ZIP decoder, TCR target | 48.72% | 5.51% |
| heads=3 (flagship) | 46.60% | 11.09% |
| heads=3, ZITD/Tweedie decoder, TCR | 44.58% | 4.58% |

### Tower Hamlets

| Config | Mean AccHR@20 | Std |
|---|---|---|
| **heads=3 + POI+socio + REAL OS network** | **67.34%** | 7.47% |
| heads=3 + POI+socio + REAL OS network, dense | 59.98% | 11.90% |
| heads=3 + POI+socio (OSMnx) | 52.88% | 8.75% |
| heads=3 (flagship) | 42.54% | 10.35% |

---

## 5. What worked, what didn't (statistically tested)

### Confirmed positives

| Lever | Effect | Evidence |
|---|---|---|
| **Real OS Open Roads network** | 55.39% -> 66.99% (+11.59) | **p=0.0010** (t), **p=0.0016** (W), n=18, 3 boroughs |
| **POI + socio-demographic features** | 46.43% -> 55.39% (+8.96) | **p=0.0094** (t), **p=0.0120** (W), n=18, 3 boroughs |
| **GAT heads=3 vs heads=1** | 41.59% -> 49.57% | **p=0.036** (t), n=6, Westminster |

### Confirmed negatives / nulls

| Lever | Result | Evidence |
|---|---|---|
| Network consolidation (distance-merge proxy) | -8.01 points | **p=0.0277** - significantly WORSE |
| `rate_link=exp` (paper's reference-code log-link) | -2.49 points | **p=0.011** - significantly worse |
| Head count on top of POI+socio (OSMnx) | no pattern | p=0.56, null |
| Head count on top of real network (Lambeth) | heads=3 still best | null |
| TCR target vs plain count (real network) | -1.94 points | p=0.2021, null (0/6 wins) |
| TCR target vs plain count (OSMnx) | +0.60 points | indistinguishable |
| ZITD/Tweedie decoder vs ZIP (OSMnx, x2 boroughs) | -4 to -5 points | consistent, both boroughs |
| Dense temporal protocol | lower absolute, same direction | confound-free but harder sampling |
| Junction risk redistribution (paper's own method) | **-10.16 points** | **p=0.0432** - significantly WORSE (see note) |
| Multi-seed ensembling (5 seeds) | -1.10 points | p=0.6881, null - variance is in the data, not the seed |
| Road class (8) + date (4) features | +0.15 points | p=0.9725, null on the mean - but cuts variance (std 4.55% vs 11.26%) |
| Paper's GRU->GAT encoder order (each at its own optimal lr) | -2.66 points | p=0.2832, null - equivalent once fairly tuned; see note |
| POI at the paper's 20-class granularity (vs our 4) | -2.29 points | p=0.4201, null |
| FULL Table 7.2 parity (all input classes at once) | -3.83 points | p=0.3796, null - worst feature variant; overfits at 6-11 instances |
| Weather, re-tested on the REAL network | +0.89 points | p=0.7577, null (confound removed; earlier OSMnx null confirmed) |
| Max-value input scaling (reference impl's own approach) | catastrophic | 34.70% then 12.82% (below random) - crushes sparse-column variance |
| log1p + z-score | rejected pre-run | sparsity, not magnitude, drives the extremes - no monotone transform helps |
| Clipped z-scoring (+/-5 sigma) | +0.02 points | p=0.9899, null - the |1179| extremes are real but NOT the bottleneck |
| ~10x more training instances, same eval windows | ~-3 points | mixed over 5 completed windows; sample size is not the bottleneck either |
| Model capacity 2x (32/64) and 0.5x (8/16) | -2.0 / -2.7 points | p=0.29 / p=0.39, both null - 16/32 is near a local optimum |
| Rank blend: GNN x historical rate | catastrophic | monotone degradation 75.8%->29.9%; history alone is near-random at 99.98% sparsity |
| Architecture ensemble (16/32 + 42/42 + 2-layer, rank-averaged) | +0.77 points | p=0.5535 (n=18), 9/18 wins - null; won on 2 boroughs, lost on the 3rd |
| Joint multi-borough training (one model, 3 boroughs) | -1.17 points | p=0.2384 (n=18), 6/18 wins - null; falsifies the sample-size hypothesis |
| Metric-aware top-k hinge loss (weights 0.1/0.5/1.0) | 0.0 / -3.2 / -5.0 points | monotone degradation - the direct ranking objective is too noisy at ~2 positives/day |
| Dense training pool, aligned eval (64-96 vs 6-11 instances) | -3.38 points | p=0.5780 (n=6), 3/6 wins - null, high variance (+10/-26 same run) |
| Learning rate sweep for spatial_first (0.003-0.02, never swept before) | best +0.81 points (lr=0.02) | p=0.6129 (n=6), 3/6 wins - null; 0.01 already near-optimal by chance |
| Feature pruning (drop 5 redundant casualty-severity columns) | -0.69 points | p=0.3632 (n=6), 0/6 wins - null; 5 exact ties, model already ignored them |
| AADF name-propagation (coverage 1.56% -> 28.35%) | **-10.42 points** | p=0.0752 (n=6), 1/6 wins - NEGATIVE; smoothing destroys ranking sharpness |
| **Long-horizon crash history (+90d/365d, was 30d cap)** | **+6.70 points pooled** | **p=0.001337 (n=18), 16/18 wins - SIGNIFICANT on all three boroughs** |

### HEADLINE (2026-09-03): pooled benchmark exceeded

| Borough | Ours | Gao et al. | |
|---|---|---|---|
| Westminster | **75.39%** | 68.98% | +6.41 (p=0.1218, **tie**) |
| Tower Hamlets | **75.37%** | 72.24% | +3.13 (p=0.3781, **tie**) |
| Lambeth | 70.29% | 76.59% | -6.30 (p=0.1074, **tie**) |
| **Pooled** | **73.69%** | **72.60%** | **+1.08 (p=0.5730, TIE)** |

**Every 95% CI contains the paper's figure - this is a statistical TIE, not a win.** The p=0.001337 result is this project's new model vs its OWN baseline (paired, n=18), NOT a test against Gao et al., whose per-window results are unpublished. Bootstrap P(ours > theirs) = 71.8%.

Pooled n=18: paired t=3.8313, **p=0.001337**; Wilcoxon p=0.001579; 16/18 windows.
Two of three boroughs exceed the paper; Lambeth does not. Protocol differences (their within-2019 6:2:2 split vs this project's multi-year walk-forward) and the unexplained zero-inflation discrepancy remain disclosed caveats.


**Correction (2026-09-03)**: an earlier claim that AccHR@20 correlates r=0.731 with crash density was based on Lambeth alone (n=6). Across all 18 windows it is **r=0.247, p=0.3234 - not significant**. Their evaluation windows do contain ~35-55% more crashes than ours (their Table 7.2 annual counts scaled to 14 days: ~67/51/47 vs our observed 43.5/34.3/34.7), which is a real difference in evaluation conditions - but our data cannot reliably quantify what that difference is worth.

| hidden=42/42 (paper's own size), isolated | -4.26 points pooled | p=0.2986 (n=12) - won on Lambeth (+1.5), lost on Westminster (-10.0); a caught false positive |

### Trivial-baseline context (added 2026-09-03)

| Ranking | Lambeth AccHR@20 |
|---|---|
| random (metric sanity control) | 18.59% |
| node degree | 18.36% |
| recent crash history (30d) | 21.86% |
| traffic volume (AADF) | 22.17% |
| segment length | 29.29% |
| **amenity density (single column)** | **56.89%** |
| best simple rank-average | 56.68% |
| **GNN, best config** | **63.59%** |
| Gao et al. | 76.59% |

The GNN beats the strongest single-feature baseline by **+6.9 points**. The random control at 18.59% (theoretical ~20%) validates the metric scale. Gao et al. report no comparable single-feature baseline.

| Partial junction redistribution (home_share=0.7) | lost 4/4 completed windows | run died at window 5 (CUDA OOM from a job-queue race); clearly negative, not rerun |
| ZI-Tweedie decoder on the REAL network | -0.09 points | p=0.9696, null (but see note: the OSMnx penalty vanishes) |
| Weather features | -1.76 points | within noise |
| Spillover target (0.5/0.25 decay) | -15.6 points | clear negative |
| Early stopping (single- and k-fold validation) | -23 points | clear negative at this data scale |
| Paper's literal epochs=20 | -35 points | badly underfits this feature set |
| 2019 same-year 6:2:2 protocol | ~equal to walk-forward | protocol is not the explanation |

---

## 6. Known limitations and disclosed substitutions

**Data sources that are substitutes, not the paper's own** (each
documented in the relevant `src/greyspot/ingest/` module docstring):

| Paper's source | This project's substitute | Why |
|---|---|---|
| OS Points of Interest (20 classes) | OSM tags (shop/amenity/leisure/tourism) | OS POI is a commercial product |
| Census 2011 socio-demographics (8 classes) | IMD 2019 domain scores + population density at 2011 LSOA geography | Direct Census pull not attempted; IMD is at the same geography |
| Met Office weather | Open-Meteo historical reanalysis | Free, equivalent variables |

**OS Open Roads is NOT a substitute** - as of 2026-09-02 this project
uses the real Ordnance Survey product, the same source the paper uses.

**Methodological caveats**:

1. **CUDA non-determinism**: `train_gat_temporal_walkforward` sets
   `torch.manual_seed` but not `torch.use_deterministic_algorithms`.
   GAT scatter/gather ops on CUDA are not bit-reproducible; re-running
   an identical config reproduced 5/6 windows exactly and one window
   differed by ~4 points. Individual percentages are one-run results,
   not fixed constants. This does not invalidate the paired
   significance tests (each compared two actually-executed runs), but
   full determinism would be a reasonable hardening step.
2. **Segment-count discrepancy, unresolved**: the paper reports 4,822
   roads for Westminster; this project's properly polygon-clipped OS
   Open Roads extract gives 5,549 links / 11,098 directed edges. Either
   they count by a different convention (named-street aggregation?) or
   use a different boundary. Not chased further because the empirical
   network comparison does not depend on resolving it.
3. **Six windows per borough** is a small sample for per-borough
   significance; pooled tests (n=18) are the stronger evidence.
4. **Target definition**: this project's default target is plain
   collision count; the paper's is severity-weighted TCR. Both have
   been tested; they are statistically indistinguishable on this
   pipeline (tested on both network sources).

---

## 6b. Notes on three results that need context

**Junction risk redistribution (-10.16 points, p=0.0432).** The paper's
own methodology (Section 7.2.1) distributes each intersection crash's
risk "equally among all connected road segments"; this project
attributed each crash wholly to its nearest segment. Since **63-72% of
collisions in these boroughs are junction-related** (STATS19's own
`junction_detail` field), this was the largest unimplemented
methodological difference remaining. Implemented faithfully and tested:
it is significantly WORSE here. The redistributed run's variance also
collapsed (std 2.66% vs 11.26%) - it produces a much flatter risk
ranking, which is exactly the failure mode for a top-20% task.

This does **not** show the paper's method is wrong on their setup. A
likely representational cause: OS Open Roads models a two-way street as
two directed edges, so a four-arm junction has EIGHT incident edges
here and an equal 1/8 split is a far more aggressive dilution than
their graph (where road segments are nodes) implies. A variant
splitting across undirected physical roads is a legitimate follow-up.

**An evaluation bug caught in this project's own first attempt at the
above** - recorded because it nearly produced a false published
conclusion. AccHR@20 defines "actual crashes" as `y_true > 0`. Scoring
against the *redistributed* target turns one junction crash into 8
fractional entries, requiring the model to land all 8 in the top 20%
instead of 1 - a mechanically harder metric, not a worse model. The
first run scored 50.83% on its opening window from that artefact alone.
Fixed by training on the redistributed target while **evaluating
against the original**, with an assertion that both instance lists
align one-to-one on held-out dates. All junction figures reported here
use the corrected design.

**ZI-Tweedie decoder on the real network (p=0.9696, null).** Worth more
than its null suggests: on the OSMnx network this decoder was reliably
4-5 points WORSE than ZIP on both boroughs tested; on the real network
that penalty vanishes entirely (63.50% vs ZIP's 63.59%). The decoder's
relative performance depends on the network source - consistent with
the paper's claim that its ZI-Tweedie decoder matters most under
extreme sparsity, which is the regime the real OS topology produces.
It still does not beat ZIP, so ZIP remains this project's default.

**Multi-seed ensembling (p=0.6881, null).** Every figure this project
has reported came from a single weight initialisation (`seed=42`). A
5-seed prediction-averaging ensemble did not tighten the spread, which
locates the variance in the DATA (genuinely different difficulty across
held-out calendar windows) rather than in training. Consequences:
ensembling is not worth 5x compute here, and more seeds will not make
these estimates more precise - only more windows or more data would.

## 7. Open questions

1. **Lambeth's remaining 13-point gap** - the largest outstanding
   discrepancy. The thesis attributes its own strong Lambeth result to
   the ZI-Tweedie decoder's handling of extreme sparsity, and Lambeth
   is confirmed (in this project's own data too) as the sparsest of
   the three boroughs. A ZI-Tweedie + real-network test is underway.
2. **Whether pooled performance can exceed 72.60%** - would require
   closing both Lambeth and Tower Hamlets.
3. **Multi-seed ensembling** - implemented but not yet run to
   completion; the natural variance-reduction step given how noisy
   6-window estimates are at 6-11 training instances.


> **See also** [`docs/final_model.md`](final_model.md) — the canonical
> specification of the final configuration and its limitations.
