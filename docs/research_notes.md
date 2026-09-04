# Research notes — source verification (2026-08-31)

The `Greyspot: Government Edition` master dossier makes a number of factual
claims about data availability, government policy context and academic
precedent. Before committing engineering time, each was checked against live
sources. Summary: **the dossier is well-grounded — nothing found here
contradicts it**, but a few access-mechanics details needed clarifying for
implementation.

## Data sources

### STATS19 collision data
Confirmed real, free, no API key required. Direct CSV download URLs follow
the pattern:
```
https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-{YEAR}.csv
```
Verified working for 2021–2025 (individual year files). **2018–2020 are not
published under this per-year naming** — DfT bundles older years into
multi-year consolidated files instead (~1–2GB each), which is why this
kickoff pipeline starts from 2021.

The DfT [STATS19 development roadmap](https://assets.publishing.service.gov.uk/media/68373279e11dd1e85b0cbb22/STATS19-future-data-roadmap.odt)
confirms the specification review the dossier describes: the new
specification's rollout across police forces was targeted for completion by
end of 2025, matching the dossier's "since late 2023, rolling out" framing.
This means any model spanning 2021–2025 genuinely does cross a
specification-transition boundary, exactly as the dossier's data-quality
gates (Section 8) anticipate — worth flagging explicitly in evaluation,
not glossing over.

A field-level surprise found while inspecting the live 2025 file: the legacy
`local_authority_district` column is no longer populated (constant -1) in
current releases. Geographic filtering now needs
`local_authority_ons_district` (ONS GSS code; Westminster = `E09000033`).
This is an implementation detail the dossier's data model (Section 8) didn't
anticipate at this level of specificity.

Source: [gov.uk/government/statistics/road-safety-data](https://www.gov.uk/government/statistics/road-safety-data)

### OS Open Roads
Confirmed real, free (Open Government Licence), and genuinely a
link-and-node topological graph (`start_node`/`end_node` fields) — this
substantiates the dossier's core "network-connected modelling" premise.
**Practical caveat**: downloading from the OS Data Hub requires a free
account/registration. Since account creation on the user's behalf is out of
scope for me, this kickoff pipeline uses **OSMnx/OpenStreetMap** instead —
which the dossier itself names as an acceptable fallback/cross-check network
(Section 9) — and defers OS Open Roads to a later swap once the account
exists.

Source: [osdatahub.os.uk/downloads/open/OpenRoads](https://osdatahub.os.uk/downloads/open/OpenRoads)

### IMD 2019 (equity slicing, later phase)
Confirmed real and downloadable at ward/LSOA level.
Source: [London Datastore — Indices of Deprivation](https://data.london.gov.uk/dataset/indices-of-deprivation-2l15g)

### py-stats19 (PyPI `pystats19`)
Exists but is explicitly beta/under-development. Decision: pull STATS19 CSVs
directly with `requests` rather than depending on an unstable library for
core data ingestion.

## Policy context

### TfL Vision Zero Action Plan 2
Confirmed real. Launched jointly by the Mayor, TfL, London councils and the
Met Police in March 2026, with a new interim target of a **65% reduction in
people killed or seriously injured by 2035** against a 2022–24 baseline.
This is a stronger, more current policy hook than the dossier's own citation
implies — worth quoting directly in the proposal's motivation section.

Source: [content.tfl.gov.uk/vision-zero-action-plan-2.pdf](https://content.tfl.gov.uk/vision-zero-action-plan-2.pdf)

## Academic precedent

### Gao, Jiang, Haworth et al. (2024)
"Uncertainty-Aware Probabilistic Graph Neural Networks for Road-Level Traffic
Accident Prediction" — confirmed real, published in *Accident Analysis &
Prevention* (arXiv preprint [2309.05072](https://arxiv.org/abs/2309.05072)).
The model described (STZITD-GNN: GAT + GRU encoder, Tweedie-distribution
decoder for zero-inflated counts) is architecturally very close to what the
dossier proposes (GAT + GRU/LSTM). This is a legitimate, citable precedent —
not a fabricated reference — and the dossier's framing of it as the "closest
academic precedent" holds up.

## Tooling

### MAPIE (conformal prediction)
Confirmed real, actively maintained
([scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)),
with a v1 API refresh released in 2026. Recommendation: use MAPIE for the
uncertainty-quantification layer (Section 10 of the dossier) rather than
implementing conformal calibration from scratch — it directly supports
scikit-learn/PyTorch-wrapped regressors and reports empirical coverage,
which is exactly what the dossier's calibration dashboard needs.

## Additional research (2026-08-31, second pass) — exposure data and the UCL precedent in depth

### DfT AADF (Annual Average Daily Flow) traffic counts
Confirmed real, free, no registration. Bulk GB-wide CSV download:
[storage.googleapis.com/dft-statistics/.../dft_traffic_counts_aadf.zip](https://storage.googleapis.com/dft-statistics/road-traffic/downloads/data-gov-uk/dft_traffic_counts_aadf.zip)
(600,551 records, 2000–2025, via [roadtraffic.dft.gov.uk/downloads](https://roadtraffic.dft.gov.uk/downloads)).
This is the "exposure" data the dossier names as important (Section 5.2)
but flags as possibly infeasible for an undergraduate project - it turned
out to be freely downloadable with no account needed, so it is now
integrated (see `ingest/exposure.py`). Coverage is sparse by design: only
~112–118 count points per year for the whole of Westminster (major roads
only), so most road segments legitimately have no exposure data - handled
via an explicit `has_aadf` indicator feature, not a fabricated value.

### The academic precedent's real training setup
Fetched the full HTML text of Gao, Jiang, Haworth et al. (2024),
arXiv:2309.05072v2 (previously only the abstract/summary had been checked).
Confirmed: James Haworth is at UCL SpaceTimeLab; co-authors Huanfa Chen and
Stephen Law are at UCL's Bartlett Centre for Advanced Spatial Analysis
(CASA) - **this is genuinely "a UCL paper"**, not just an arXiv preprint
with no institutional grounding. New concrete details extracted:
- **2-layer GAT, 3 attention heads, 42 hidden units** (Greyspot's first
  version used 1 layer, 4 heads, 16 hidden units - shallower and narrower).
- **Dropout 0.2, weight decay 0.01, Adam, lr 0.01, 20 epochs with
  early-stopping patience 10** - Greyspot's first version used none of
  the first two and trained for 200 epochs, a plausible explanation for
  why the unregularised GAT branch overfit relative to its own "no graph"
  ablation (see `docs/decision_log.md`).
- Input features are road/census/weather context - **no traffic-volume
  exposure feature**, meaning Greyspot's new AADF integration is a genuine
  addition beyond this precedent, not a replication of it.
- **The paper itself contains no graph-vs-no-graph or temporal-vs-no-temporal
  ablation** - Greyspot's ablation study is real, additional evidence the
  published literature doesn't provide.
- A related UCL PhD thesis (Xiaowei Gao, "Advanced Graph Deep Learning for
  Urban Traffic Crash Research", UCL Discovery eprint 10210801) exists but
  was too large to fetch in full in this session (repeated timeouts on a
  43MB PDF from discovery.ucl.ac.uk) - worth reading properly in a later
  phase for more architectural depth.

## What this changes vs. the dossier

Nothing material. The one real course-correction: **swap OS Open Roads for
OSMnx in the first pipeline pass**, purely for account-registration reasons,
not a data-quality concern — the dossier already sanctions this fallback.
