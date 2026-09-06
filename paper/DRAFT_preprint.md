# What Actually Matters in Road-Level Crash Prediction: Ten Findings, Three Replications

**Draft 2026-09-06.** Every number below traces to a script in this
repository and an entry in `docs/decision_log.md`. Claims that failed
replication are reported alongside those that survived.

---

## Abstract

We attempt to replicate and extend Gao et al. (2024)'s STZITD-GNN for
road-level crash prediction on three London boroughs, using independently
constructed data. Our model reaches AccHR@20 of 80.14% ± 1.12 (5 seeds)
against their reported 72.60%, significantly better on two of three
boroughs. But the more transferable results are negative and
methodological: of ten findings that appeared significant on a single
borough, **only three survived replication on a second** (seven failed).
Effect size predicts non-replication but not replication: every effect
below the measured ~4-point seed-noise band failed (4/4), while only 4 of
6 effects above it survived — a small effect is reliable evidence against,
a large one is weak evidence for. Failure is characteristically a **sign
reversal** between boroughs, not a shrinkage toward zero: four of the
seven changed direction. We further show that at matched history depth
our graph neural network is **statistically indistinguishable from sorting
road segments by their past crash count**, and that the reference
architecture, given the same data and its own tuned settings, **loses to
that sort on 18 of 18 held-out windows** (−19.49 points, *p* < 10⁻⁵). The
sort also converts three further years of history into a significant
+2.95-point gain (*p* = 0.041) that neither network reliably matches. The
reference paper does include a historical-average baseline, but its data
covers a single year; we argue that the apparent margin of graph networks
over historical baselines in this literature is substantially a function
of how short a horizon those baselines were computed over.

---

## 1. What we set out to do, and what we found instead

The intended contribution was a benchmark improvement. The durable
contribution is an account of how easily results at this evaluation scale
mislead.

**Headline comparison** (AccHR@20, 6 walk-forward windows per borough,
averaged over 5 random seeds, 95% CI):

| Borough | This work | 95% CI | Gao et al. | |
|---|---|---|---|---|
| Westminster | **80.03% ± 1.40** | [78.29, 81.77] | 68.98% | p=0.0001 |
| Tower Hamlets | **82.63% ± 2.01** | [80.14, 85.12] | 72.24% | p=0.0003 |
| Lambeth | 77.75% ± 1.67 | [75.68, 79.82] | 76.59% | tie, p=0.1941 |
| **Pooled** | **80.14% ± 1.12** | [78.74, 81.53] | 72.60% | p=0.000115 |

**This is not a like-for-like comparison** and should not be read as one.
Gao et al. evaluate within 2019 on a 6:2:2 split; we use multi-year
expanding walk-forward over 2022–2024. No paired test is possible: they
publish three point estimates, not per-window results. Our target is also
measurably sparser than theirs (99.97% vs their reported 95.72–96.71%
zero-inflation), a discrepancy we could not resolve after checking their
TCR formula, segment consolidation and evaluation protocol.

## 2. The main result: replication is the exception

| Finding | Effect (borough 1) | Second borough |
|---|---|---|
| Architecture topology (depth, encoder order) | −26.46, −13.61 | **replicated** |
| Long-horizon crash history | +15.85 | **replicated** (+9.70) |
| Our architecture vs theirs | +18.53 | **replicated** (+9.08, +22.28) |
| hidden=42/42 | +1.46 | failed (−9.97) |
| Architecture ensembling | +1.79 | failed (p=0.55 at n=18) |
| weight_decay=0.01 | +2.53, 0 losing windows | failed (−0.60 at n=18) |
| Road class harmful | −6.87, −4.49 (p=0.016) | failed (sign flips, +2.25) |
| 13 features ≈ 35 features | −0.99 (p=0.35) | failed (−4.08, then **+0.77** on a third) |
| Architecture × history interaction | +25.04 vs +9.70 | failed (reverses sign) |
| Rank-transform scaling | +5.80 (best ever) | failed (−39.7, worst ever) |

**Three of ten survived; seven failed.**

Effect size is often proposed as a filter for which single-sample results
to trust. On this evidence it works in **one direction only**
(`scripts/check_effect_size_heuristic.py`):

| Borough-1 effect | Replicated | Failed |
|---|---|---|
| Below the ~4-point seed-noise band | 0 | 4 |
| Above it | 4 | 2 |

Every sub-noise effect failed, so a small effect is reliable evidence
*against* replication. But two comfortably supra-noise main effects failed
anyway — road class (−6.87, p=0.016) reversed sign on the second borough
(+2.25), and rank-transform scaling went from the best result ever
recorded here (+5.80) to the worst (−39.7). A large effect is therefore
only weak evidence *for* replication: necessary, not sufficient.

Interactions are worse still: the architecture×history interaction was +25
points — larger than any surviving main effect — and reversed sign between
boroughs.

**How failures look matters more than how often they happen.** Four of the
seven failures were sign reversals, not attenuations: road class (−6.87 →
+2.25), rank-transform scaling (+5.80 → −39.7), the architecture×history
interaction, and the feature-count claim (−0.99 and −4.08 on two boroughs,
then +0.77 on a third). A practitioner who assumed effects merely shrink
across regions — and who therefore treated a single-region estimate as an
upper bound — would have had the direction wrong, not just the magnitude.

**The practical implication** is asymmetric and cheap to apply: a
sub-noise single-borough result can be discarded without further runs,
while a large one still has to be replicated before it can be believed.

## 3. The model adds little over a trivial baseline

| Ranker | Lambeth | Westminster | Tower Hamlets | Mean |
|---|---|---|---|---|
| Sort by cumulative crash count (~8yr) | 83.20% | 81.25% | 87.37% | **83.94%** |
| Empirical Bayes (HSM) | 82.11% | 81.93% | 87.48% | 83.84% |
| Same count, capped to the GNN's 5yr | 80.81% | 76.46% | 85.69% | 80.99% |
| **Our GNN** | 77.75% | 80.03% | 82.63% | **80.14%** |

Every GNN figure above is a 5-seed mean, and every comparison below is
**paired window-by-window** on the same 18 held-out windows (6 windows ×
3 boroughs); the baselines are deterministic, so all sampling variation
sits on the GNN side.

| GNN vs | Δ (points) | paired *t* | Wilcoxon | GNN wins |
|---|---|---|---|---|
| Empirical Bayes (HSM) | **−3.71** | 0.0146 | 0.0237 | 6/18 |
| Cumulative crash count | **−3.80** | 0.0289 | 0.0342 | 5/18 |
| Count capped to the GNN's 5yr | −0.85 | 0.6398 | 0.7987 | 9/18 |

At **matched history depth** the GNN and a parameter-free sort are
statistically indistinguishable (−0.85, *p* = 0.64, 9 of 18 windows).
Uncapped, both trivial baselines beat it *significantly*: given three
further years of history the sort gains **+2.95 points** (paired *p* =
0.041, 12/18 windows).

Whether the network can use that same extra history is **borough-specific**
(`scripts/run_s5_two_borough_analysis.py`). Extending its history features
from 5 to 9 years costs −0.72 points on Lambeth (*p* = 0.74) but gains
+2.98 on Westminster (*p* = 0.023, 5/6 windows). We previously reported
the Lambeth null alone as evidence the model "cannot exploit" deeper
history; a second borough does not support that. What does hold on both
boroughs measured is weaker and directional: **the sort gains more from
the same extra history than the network does** (+2.38 vs −0.72 on Lambeth;
+4.79 vs +2.98 on Westminster). Both network figures sit inside the
~4-point seed-noise band, so neither is individually interpretable.

A graph network with 35 features, conformal intervals and 200 training
epochs therefore does not beat sorting segments by how often they have
crashed, and extracts less from additional history than the sort does.

**This is not a property of our implementation alone.** Running the
reference architecture on our data — at its own swept optimum, with the
same long history, on the same windows — it loses to the crash-count sort
by **19.49 points, on 18 of 18 windows** (paired *p* = 0.000001), and by
16.54 points against the horizon-matched sort (16 of 18 windows,
*p* = 0.000013) and 19.39 against Empirical Bayes (17 of 18,
*p* = 0.000001) (`scripts/run_reference_vs_trivial.py`).

| Ranker (pooled, 18 windows) | AccHR@20 |
|---|---|
| Crash-count sort (~8yr) | 83.94% |
| Sort capped to 5yr | 80.99% |
| Our GNN | 80.14% |
| Reference architecture (swept optimum, long history) | 64.45% |

We state this carefully: it is the reference *architecture* evaluated on
*our* data and protocol, not a re-evaluation of their published results,
which we cannot reproduce for lack of per-window outputs. What it
establishes is that the trivial baseline is not clearing a bar our model
happens to fall under — neither graph network in this study clears it.

### 3.1 The baseline's strength is almost entirely its horizon

This literature does *not* omit a historical baseline. Gao et al. report a
Historical Average at Acc@20 = 0.4496 (mean of 0.4520 / 0.4752 / 0.4217),
against their model's 0.7260 — a ~28-point margin that reads as clear
evidence for the network. Reconciling that with our sort's 0.8394 requires
only one variable: **their dataset covers 2019 alone**, so their historical
baseline can look back at most one year.

Sweeping the *same* parameter-free ranker across lookback horizons on our
data and windows (`scripts/run_baseline_horizon_curve.py`):

| Lookback | Lambeth | Westminster | Tower Hamlets | Mean |
|---|---|---|---|---|
| 30 days | 21.86% | 23.71% | 22.56% | 22.71% |
| 90 days | 30.33% | 34.78% | 31.44% | 32.19% |
| 180 days | 38.40% | 44.13% | 42.29% | 41.61% |
| 1 year | 50.50% | 54.26% | 57.74% | 54.17% |
| 2 years | 60.26% | 68.13% | 70.66% | 66.35% |
| 3 years | 70.58% | 71.72% | 79.04% | 73.78% |
| 5 years | 80.81% | 76.46% | 85.69% | 80.99% |
| 7 years | 83.62% | 79.29% | 87.67% | 83.53% |
| 9 years | 83.20% | 81.25% | 87.37% | 83.94% |

The same ranker spans **22.71% to 83.94%** — a 61-point range — with no
change but how far back it looks. Laid against the published figures, each
is matched by a sort at a strikingly short horizon:

| Reported system | Score | Matched by a crash-count sort at |
|---|---|---|
| Gao et al., Historical Average | 44.96% | ~1 year |
| Reference architecture (our data) | 64.45% | ~2 years |
| Gao et al., STZITD-GNN | 72.60% | ~3 years |
| Our GNN (5 seeds) | 80.14% | ~5 years |

**This mapping is suggestive, not a controlled comparison** — their two
figures are on their data and segments, ours on ours. It cannot show that
their model would lose to a sort on their own data. What it does show is
that the *quantity* of published improvement over a historical baseline in
this task is of the same order as the improvement obtainable by lengthening
that baseline's horizon by a year or two, on data that is freely available
for both. That is a cheap check, and we could find no paper in this line
that reports it.

The curve also **plateaus after about seven years** (83.53% → 83.94%),
which corrects an earlier reading of our own: at three years it is still
climbing steeply and appears not to plateau at all.

A final internal consistency check. Section 3 reported that the network's
response to history beyond five years is borough-specific — negative on
Lambeth, positive on Westminster. The sort behaves the same way over the
same interval:

| Borough | Sort, 7→9 years | Network, 5→9 years |
|---|---|---|
| Lambeth | −0.43 | −0.72 |
| Westminster | +1.96 | +2.98 |

Both agree in sign on both boroughs. The borough-specific response to deep
history is therefore a property of **the data**, not of the model — Lambeth
has simply exhausted the information in its crash record by seven years,
and Westminster has not.

We note that seed-averaging and paired testing made this result
*stronger*, not weaker. The earlier single-seed comparison of means could
produce no *p*-value at all and recorded the matched-horizon gap with the
sign reversed.

## 4. Measurement properties practitioners should know

- **Seed variance is large and its sign is borough-specific.** Spread
  3.2–4.7 points across 5 seeds. One seed was +1.69 optimistic on one
  borough and −1.11 conservative on another, so a seed's bias measured on
  one region **cannot** correct another.
- **The metric depends on crash volume.** AccHR@20 rises ~0.39 points per
  additional crash in a window (p=0.0316, n=31).
- **Holiday periods are genuinely harder**, −11.14 points independent of
  crash volume (p=0.0078).
- **A constant prediction scores 10.12%**, below random's ~20% — so no
  reported score is a tie-breaking artefact.
- **The label-shuffle control passes**: permuting targets across segments
  collapses the model to 23.08% from 75.64%.

## 5. Reproducibility notes on the reference method

Gao et al.'s published learning rate (0.01) **diverges to NaN** on our
data; their predecessor code's 1e-5 scores 24.74%. The optimum we found
(5e-4) appears in neither source. We report their architecture at its own
swept optimum throughout, not at settings where it fails.

## 6. Limitations

Three boroughs; six windows per borough (31–38 in the dense evaluation);
one city; and a target-density discrepancy against the reference paper
that we could not explain.

Generalisation to five further boroughs is incomplete. The OSM Overpass
API degraded mid-study — reporting free capacity while resetting
connections during larger transfers — and one borough's POI download lost
an entire feature category on three separate attempts while appearing
successful. We quarantined it rather than report it. This is worth stating
plainly because the failure was **silent**: the truncated download
produced a plausible-looking file that would have been cached and reused
indefinitely. We now refuse such responses in code, validating both that
every category returned and that POI density clears a floor calibrated on
verified downloads. That check initially rejected a *valid* outer-London
borough, whose floor we had calibrated only on inner-London ones; we
established the download was complete by confirming two independent
downloads were byte-identical, which a truncated response cannot be.

---

*Data: STATS19 (DfT), OS Open Roads (Ordnance Survey, OGL), IMD 2019,
DfT AADF. Code and full decision log: see repository.*
