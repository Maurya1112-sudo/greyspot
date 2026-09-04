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
can be dropped. A further cut to 13 features (geometry + crash history
only) is −0.99 on Lambeth but −4.08 on Westminster — plausible, but it
needs a third borough before being recommended.

### Validation controls (all passed)

| Control | Result |
|---|---|
| Label shuffle (targets permuted across segments) | 75.64% → **23.08%**, collapses to random (19.96 ± 3.46) |
| Metric sanity | Constant predictions score **10.12%**, *below* random — no score is a tie-break artefact |
| Network invariance | Random ranker: 19.68% vs 19.96% across networks; identical 206 crashes captured |

---

## Reproducing

```bash
python -m pytest tests/ -q                                  # 211 tests
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

Three candidate improvements reached p≈0.05–0.10 on a single borough and
**all three failed cross-borough replication** — one shrinking
monotonically from +2.53 (n=6) to −0.60 (n=18) as evidence accumulated.
Three previously-recorded null results were **overturned** when
re-measured under the final configuration. A configuration bug silently
disabled feature subsetting after the first evaluation window,
invalidating three results and briefly producing a false retraction of a
correct finding.

All of it is recorded in `docs/decision_log.md`, including the mistakes
and the corrections to earlier corrections.

## Licence and data

Code: for academic use. Data is not redistributed here — STATS19 and OS
Open Roads are public (OGL); IMD and Census products carry their own
terms.
