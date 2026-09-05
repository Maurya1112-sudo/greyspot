# What Actually Matters in Road-Level Crash Prediction: A Replication Study with Nine Failed Replications

**Draft 2026-09-05.** Every number below traces to a script in this
repository and an entry in `docs/decision_log.md`. Claims that failed
replication are reported alongside those that survived.

---

## Abstract

We attempt to replicate and extend Gao et al. (2024)'s STZITD-GNN for
road-level crash prediction on three London boroughs, using independently
constructed data. Our model reaches AccHR@20 of 80.14% ± 1.12 (5 seeds)
against their reported 72.60%, significantly better on two of three
boroughs. But the more transferable results are negative and
methodological: of nine findings that appeared significant on a single
borough, **only three survived replication on a second**, and effect size
predicted which — every effect above the measured ~4-point seed-noise
band replicated, every effect below it failed. We further show that at
matched history depth our graph neural network is **statistically
indistinguishable from sorting road segments by their past crash count**,
and that it cannot exploit additional history that the trivial sort
converts into a +2.95-point gain.

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
| 13 features ≈ 35 features | −0.99 (p=0.35) | downgraded (−4.08) |
| Architecture × history interaction | +25.04 vs +9.70 | failed (reverses sign) |
| Rank-transform scaling | +5.80 (best ever) | failed (−39.7, worst ever) |

**Three of nine survived.** Effect size predicted the outcome for main
effects: all survivors exceeded the seed-noise band; all small effects
failed. **It did not predict interactions** — the architecture×history
interaction was +25 points and still reversed sign between boroughs.

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
Uncapped, both trivial baselines beat it *significantly*. Given *more*
history the sort gains +2.95; the GNN gains −0.72 with rising variance
(§5). A graph network with 35 features, conformal intervals and 200
training epochs does not beat sorting segments by how often they have
crashed — and given history it cannot use, loses to it outright.

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
one city; a target-density discrepancy against the reference paper that
we could not explain; and generalisation to five further boroughs left
incomplete when the OSM Overpass API failed mid-run — two of those
boroughs produced silently partial data that we quarantined rather than
report.

---

*Data: STATS19 (DfT), OS Open Roads (Ordnance Survey, OGL), IMD 2019,
DfT AADF. Code and full decision log: see repository.*
