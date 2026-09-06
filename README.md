# Greyspot — uncertainty-aware road-level crash risk prediction for London boroughs

A GAT+GRU graph neural network with a zero-inflated Poisson decoder and
conformal prediction intervals, evaluated against
[Gao et al. (2024)](https://doi.org/10.1016/j.aap.2024.107801),
*"Uncertainty-aware probabilistic graph neural networks for road-level
traffic crash prediction"* (Accident Analysis & Prevention), on three
London boroughs.

**Status: research code under active verification.** Numbers below are
current as of 2026-09-04 and are stated with the caveats that apply to
them. See [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) for the live
verification ledger and [`docs/decision_log.md`](docs/decision_log.md)
for every experiment run, including the negative and invalidated ones.

---

## Results

AccHR@20 — the share of crashes falling in the top 20% of
predicted-risk road segments, averaged per day, on six held-out 14-day
windows per borough (expanding-window walk-forward, 2022–2024).

All figures are **averaged over 5 random seeds** with 95% confidence
intervals — single-seed numbers are not reported, for reasons given below.

| Borough | This project | 95% CI | Gao et al. | Verdict |
|---|---|---|---|---|
| Westminster | **80.03% ± 1.40** | [78.29, 81.77] | 68.98% | **better**, p=0.0001 |
| Tower Hamlets | **82.63% ± 2.01** | [80.14, 85.12] | 72.24% | **better**, p=0.0003 |
| Lambeth | **77.75% ± 1.67** | [75.68, 79.82] | 76.59% | **tie**, p=0.1941 |
| **Pooled** | **80.14% ± 1.12** | **[78.74, 81.53]** | **72.60%** | **better**, p=0.000115 |

### Read these caveats before quoting any number

1. **Seed variance is large and its direction is borough-specific.**
   Per-borough spread across 5 seeds is 3.2–4.7 points. The bias of any
   single seed *flips sign between boroughs* — seed 42 is +1.69
   optimistic on Lambeth, −1.11 conservative on Westminster, +1.30 on
   Tower Hamlets — so a seed's bias measured on one region **cannot be
   used to correct another**. Differences under ~4 points from
   single-seed runs are not interpretable. The pooled figure is
   considerably more stable (seed-42 bias +0.63) because per-borough
   biases partly cancel.
2. **The comparison is not like-for-like.** Gao et al. evaluate within
   2019 on a 6:2:2 split; this project uses multi-year expanding
   walk-forward over 2022–2024, a stricter protocol. No *paired* test
   against them is possible — they published three point estimates, not
   per-window results.
3. **A target-density discrepancy is unexplained.** They report
   95.72–96.71% zero-inflation; this project measures 99.97% on the same
   STATS19 source. Their TCR formula, segment consolidation and
   evaluation protocol have each been checked and ruled out as the cause.

---

## The most important caveat

**At matched history depth, this model does not outperform sorting road
segments by their past crash count.**

| Ranker | Mean AccHR@20 (3 boroughs) |
|---|---|
| Sort by cumulative crash count (~8 yr) | **83.94%** |
| Empirical Bayes (Highway Safety Manual) | 83.84% |
| Same count, capped to this model's 5-yr horizon | 80.99% |
| **This GNN** | **80.14%** |

Paired window-by-window over 18 held-out windows, with the GNN
seed-averaged over 5 seeds: at **matched** history depth the two are
statistically indistinguishable (−0.85, p=0.64, 9/18 windows). Uncapped,
both trivial baselines beat the GNN *significantly* (Empirical Bayes
−3.71, p=0.015; raw count −3.80, p=0.029). Given three more years of
history the sort gains **+2.95** (paired p=0.041, 12/18 windows).

Whether the GNN can use that same extra history is **borough-specific**:
−0.72 on Lambeth (p=0.74) but +2.98 on Westminster (p=0.023, 5/6 windows).
An earlier version of this README reported the Lambeth null alone as
evidence the model "cannot exploit" deeper history; a second borough does
not support that. What holds on both is weaker: the sort gains *more* from
the same extra history than the GNN does.

The model does provide calibrated uncertainty intervals (PICP ≈ 0.901)
that a sort cannot, and that has real value for prioritisation. But the
ranking performance should not be presented without this baseline
beside it.

## What was actually learned

**Topology is decisive; capacity is inert.** Changing message-passing
depth (−26.46, p=0.0003) or encoder ordering (−13.61, p=0.0135) is
catastrophic. Changing hidden size (−2.87) or decoder family (−3.50) is
not distinguishable from seed noise.

**Two data levers dominate**: the real OS Open Roads survey network
versus an OSMnx approximation (+11.59 pooled, p=0.0010), and extending
crash-history horizon from 30 days to 5 years (+15.87).

**Socio-demographic features add nothing.** A 26-feature model is
statistically indistinguishable from the 35-feature one on *both* tested
boroughs (+0.23 / −0.03, p=0.887 / 0.987), so the IMD/Census dependency
can be dropped. This sub-claim replicates and still stands.

**But a deeper cut to 12 features does NOT replicate**, and is no longer
recommended. Across three boroughs the effect changes direction: −0.99
(Lambeth), −4.08 (Westminster), **+0.77 (Tower Hamlets)**. The pooled
−1.44 (p=0.229) averages effects pointing opposite ways *and* sits inside
the seed-noise band. Whether the extra features help appears to be
borough-specific, so neither "they are droppable" nor "they matter" is
supportable as a general claim.

### Validation controls (all passed)

| Control | Result |
|---|---|
| Label shuffle (targets permuted across segments) | 75.64% → **23.08%**, collapses to random (19.96 ± 3.46) |
| Metric sanity | Constant predictions score **10.12%**, *below* random — no score is a tie-break artefact |
| Network invariance | Random ranker: 19.68% vs 19.96% across networks; identical 206 crashes captured |

---

## Reproducing

```bash
python -m pytest tests/ -q                                  # 232 tests
python scripts/run_ucl_comparison_multiyear.py Lambeth      # final model
python scripts/run_v8_multiseed.py Lambeth                  # multi-seed check
```

Requires: OS Open Roads GeoPackage (OS Data Hub, free), STATS19
collision/casualty CSVs 2016–2024 (DfT), AADF traffic counts, LSOA
boundaries and IMD 2019. Data directories are gitignored — see
`docs/final_model.md` §2 for the full specification.

Runtime ≈ 12 min/borough on an RTX 4060 (8 GB). **Run one GPU job at a
time.**

---

## Why the decision log is part of the contribution

Of **ten** findings that looked significant on a single borough, only
**three survived replication on a second** (seven failed).

Effect size predicts non-replication, but not replication. Every effect
below the measured ~4-point seed-noise band failed (4 of 4), so a small
single-borough result can be discarded without further runs. But only
**4 of 6** effects above the band survived: road class (−6.87) reversed
sign, and rank-transform scaling went from the best result recorded here
(+5.80) to the worst (−39.7). Interactions are less reliable still — one
+25-point interaction reversed sign between boroughs.

One candidate shrank monotonically from +2.53 (n=6) to −0.60 (n=18) as
evidence accumulated. One configuration scored best-ever on one borough
(85.56%) and worst-ever on another (39.82%). A config bug silently
disabled feature subsetting after the first evaluation window,
invalidating three results and briefly producing a false retraction of a
correct finding.

All of it is recorded in `docs/decision_log.md`, including the mistakes
and the corrections to earlier corrections.

## Licence and data

Code: for academic use. Data is not redistributed here — STATS19 and OS
Open Roads are public (OGL); IMD and Census products carry their own
terms.
