# Greyspot — uncertainty-aware road-level crash risk prediction for London boroughs

A GAT+GRU graph neural network with a zero-inflated Poisson decoder and
conformal prediction intervals, evaluated against
[Gao et al. (2024)](https://doi.org/10.1016/j.aap.2024.107801),
*"Uncertainty-aware probabilistic graph neural networks for road-level
traffic crash prediction"* (Accident Analysis & Prevention), on three
London boroughs.

**Status: research code under active verification.** Numbers below are
current as of 2026-09-06 and are stated with the caveats that apply to
them. Seven of ten single-borough findings failed replication on a second
borough, four of them by reversing sign; those are reported here alongside
the three that survived. See [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) for the live
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

### This is not specific to our implementation

Run on the same data with its own tuned settings, the **reference
architecture loses to the crash-count sort on 18 of 18 held-out windows**
(64.45% vs 83.94%, paired p=0.000001). Neither graph network in this study
clears the trivial baseline.

### Almost all of the baseline's strength is its horizon

The same parameter-free ranker, swept across lookback windows:

| Lookback | 30d | 90d | 1yr | 2yr | 3yr | 5yr | 7yr | 9yr |
|---|---|---|---|---|---|---|---|---|
| Mean AccHR@20 | 22.71% | 32.19% | 54.17% | 66.35% | 73.78% | 80.99% | 83.53% | 83.94% |

A 61-point range from one ranker with no parameters. Every published
figure in this line of work is matched by that sort at a short horizon —
Gao et al.'s Historical Average at ~1 year, their model at ~3, ours at ~5.
Their dataset covers 2019 alone, so their historical baseline could not
have looked back further. The curve plateaus after ~7 years.

That mapping is suggestive rather than controlled — two of those figures
are on their data, not ours — but the implication is cheap to check and we
found no paper in this line that reports it.

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

### Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate on POSIX
pip install -r requirements.txt
python -m pytest tests/ -q                       # 237 tests, no data or GPU needed
```

The test suite runs on synthetic fixtures, so it passes before any data is
downloaded. It is the fastest check that the environment is sound.

### Data

None of it is redistributable here, and all of it is free. `data/` is
gitignored.

| Source | Where | Goes in |
|---|---|---|
| STATS19 collisions + casualties, 2012–2024 | [data.gov.uk road safety data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) | `data/raw/collision-YYYY.csv`, `casualty-YYYY.csv` |
| OS Open Roads (GeoPackage) | [OS Data Hub](https://osdatahub.os.uk/downloads/open/OpenRoads) — free account | `oproad_gpkg_gb/Data/oproad_gb.gpkg` |
| AADF traffic counts | [DfT road traffic statistics](https://roadtraffic.dft.gov.uk/downloads) | `data/raw/aadf_raw/dft_traffic_counts_aadf.csv` |
| LSOA 2011 boundaries (BGC) | [London Datastore](https://data.london.gov.uk/) | `data/raw/boundaries/lsoa_bgc/` |
| IMD 2019 | [London Datastore](https://data.london.gov.uk/dataset/indices-of-deprivation-2l15g) | `data/raw/imd2019_london_lsoa.xlsx` |

POI features are fetched from the Overpass API on first run and cached to
`data/interim/`. **Overpass is the flakiest dependency in this pipeline** —
see the note below.

`docs/final_model.md` §2 carries the full specification, including the
exact columns used and how segments are constructed.

### Running the model

```bash
python scripts/run_ucl_comparison_multiyear.py Lambeth     # final model, one borough
python scripts/run_headline_multiseed.py                   # 5 seeds x 3 boroughs (~2.5h)
python scripts/run_s5_multiseed.py                         # deep-history arm, 5 seeds
```

≈12 min per borough on an RTX 4060 (8 GB). **Run one GPU job at a time** —
concurrent runs caused four CUDA OOM crashes here. Long runs checkpoint per
window and resume automatically if interrupted.

### Reproducing the paper's claims

Each of these regenerates one result and needs no GPU — they read the
per-window CSVs the model runs produce:

```bash
python scripts/run_s2_paired_comparison.py        # GNN vs trivial baselines, paired
python scripts/run_reference_vs_trivial.py        # reference architecture vs the sort
python scripts/run_baseline_horizon_curve.py      # AccHR@20 vs lookback horizon
python scripts/run_c2_three_borough_analysis.py   # feature-count replication
python scripts/run_c6_substitution_analysis.py    # architecture x history substitution
python scripts/run_s5_two_borough_analysis.py     # deep-history replication
python scripts/check_effect_size_heuristic.py     # does effect size predict replication
python scripts/verify_preprint_claims.py          # checks the paper against its sources
```

`verify_preprint_claims.py` is the one to run first if you only run one: it
re-derives every headline number from its source file and exits non-zero on
any mismatch.

### A note on the Overpass API

POI downloads failed repeatedly during this study, and **silently** — a
truncated response produces a plausible-looking file that gets cached and
reused. One borough's cached file turned out to hold 24% of the real data.
The ingest code now refuses such responses, checking that every tag
category returned and that POI density clears a floor calibrated on
verified downloads (`src/greyspot/ingest/poi.py`). If a borough is refused
and you suspect it is genuinely low-density rather than truncated,
`scripts/check_poi_density_calibration.py` settles it by downloading twice
and comparing — a truncated response cannot be reproducible.

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
