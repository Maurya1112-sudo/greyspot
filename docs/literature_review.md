# Literature review (draft)

*Addresses L6.06/L6.07. Organised by competing approach, each section ends
with a limitation, and the whole review ends with a gap-to-design
traceability table — following the structure the earlier project-strategy
research recommended for this degree. This is a working draft: citations
are real and checked (see `docs/research_notes.md` for verification notes
and exact URLs), but the critical synthesis will deepen once more of the
model is built and more literature is read during the official project
term.*

## 1. Road-safety prediction: from counts to networks

Classical road-safety analysis treats a crash as an event tied to a site
(a junction, a stretch of road) and models the *rate* of events at that
site using count models — Poisson or negative-binomial regression —
because crash counts are sparse and non-negative. This is transparent and
well-understood, but it treats each site independently: a busy junction and
the residential street two turns away contribute nothing to each other's
estimate, even though risk plausibly propagates along a network (a
dangerous junction upstream changes traffic behaviour downstream).

Graph neural networks (GNNs) relax that independence assumption by letting
a model aggregate information across connected road segments. **Gao,
Jiang, Haworth, Zhuang, Wang, Chen and Law (2024)**, published in
*Accident Analysis & Prevention* (preprint: arXiv:2309.05072), propose
STZITD-GNN — a Graph Attention Network (GAT) encoder combined with a GRU
for temporal dynamics and a Tweedie-distribution decoder specifically
built to handle the extreme zero-inflation of road-level crash counts.
This is the closest direct precedent for Greyspot's own model: same core
idea (GAT + GRU), same problem (sparse, zero-inflated road-segment counts),
same motivation (uncertainty matters as much as the point estimate).

**Limitation**: the Tweedie decoder is a meaningfully more sophisticated
choice than Greyspot's current Poisson-NLL decoder (see
`docs/decision_log.md`, 2026-08-31 entry) — it explicitly models the
probability mass at exactly zero separately from the continuous positive
tail, which a plain Poisson head does not. This is a documented
simplification in the current implementation, not a claim of parity with
the precedent, and is a concrete, well-scoped extension for later in the
project.

**Update (2026-09-02) — this project now benchmarks directly against
STZITD-GNN, and the "limitation" above has been empirically tested
rather than assumed.** The Zero-Inflated Tweedie decoder was
implemented faithfully (verified against the same authors' own public
Tweedie NLL code for a sibling paper) and tested repeatedly: it
consistently performs *worse* than this project's simpler
Zero-Inflated Poisson decoder on the AccHR@20 ranking metric, on two
boroughs and both network sources. Decoder sophistication is therefore
**not** the explanation for the performance gap — the road-network
*source* is (Ordnance Survey vs OpenStreetMap: +11.59 AccHR@20 points,
p=0.0010, n=18 windows). Full evidence: `docs/ucl_benchmark_results.md`.

**Scope of the precedent, verified from the primary sources**: Gao et
al. model **three London boroughs only** — Westminster, Lambeth and
Tower Hamlets — at road-segment level, using 2019 data. The paper
states this directly ("...in three boroughs of London, UK, namely
Lambeth, Tower Hamlets and Westminster"). The underlying PhD thesis
(discovery.ucl.ac.uk/id/eprint/10210801) *does* contain a London-wide
analysis, but of a **different model** (SMA-Hyper, a hypergraph-GCN)
at MSOA rather than road-segment resolution — it is not a scaled-up
STZITD-GNN and is not directly comparable. Worth stating explicitly,
because "did they do it London-wide?" is an obvious question and the
answer materially bounds what a fair comparison looks like.

Related work in the same space — **Huang, Jin, Candès and Leskovec (2023,
NeurIPS)** on conformalised GNNs, and **Zargarbashi, Antonelli and
Bojchevski (2023, ICML)** on conformal prediction sets for GNNs — establish
that conformal calibration is non-trivial on graph-structured data
specifically because a node's prediction is correlated with its
neighbours' (violating the exchangeability assumption plain split-conformal
relies on). Greyspot's current conformal implementation (see
`docs/methodology.md`) does *not* yet account for this — it applies plain
split-conformal to a transductive GNN's output, which is a known
simplification the graph-conformal literature would flag. This is an
important limitation to state explicitly in the dissertation rather than
imply the calibration result carries the same theoretical guarantee as an
i.i.d. setting.

## 2. Uncertainty quantification for count/sparse data

Conformal prediction (Vovk, Gammerman and Shafer's original framework) is
attractive here because it is model-agnostic: it wraps *any* trained point
predictor and, under an exchangeability assumption, produces intervals with
a guaranteed marginal coverage rate — no need to assume a specific error
distribution. **MAPIE** (Cordier et al., arXiv:2207.12274; actively
maintained, v1 API released 2026 — see `docs/research_notes.md`) is the
reference open-source implementation used here for the XGBoost point
model.

**Limitation of the standard method for this project's data**: split
conformal's coverage guarantee is a *marginal* one — averaged across the
whole test set — not a per-segment guarantee. A segment could be
systematically under-covered (e.g. because it is spatially unusual) while
the aggregate number still looks good. Greyspot's evaluation already
reports coverage on a single aggregate test set; a stronger version (flagged
as a next step in `docs/decision_log.md`) would report coverage
stratified by IMD decile and by road-user group, which is exactly what the
dossier's Section 10 calibration dashboard calls for and the current
implementation does not yet do.

## 3. Road-network representation

**OS Open Roads** and **OpenStreetMap/OSMnx** (Boeing, 2017, updated 2025,
*Computers, Environment and Urban Systems*) are the two practical sources
of a usable road-network graph for Great Britain. OS Open Roads is the
official, topologically curated dataset; OSMnx is community-maintained,
free of registration, and widely used in transport research as a
cross-check. Greyspot currently uses OSMnx exclusively (see
`docs/decision_log.md`, 2026-08-31) for a purely practical reason — OS
Open Roads requires a manual OS Data Hub account this project has not yet
obtained — not because OSMnx is considered superior. This is a
methodological limitation worth stating plainly: OSMnx's community-edited
geometry can differ from OS Open Roads' surveyed geometry in ways that
could affect segment boundaries and therefore snapping accuracy, though the
100% collision-to-segment join rate achieved so far (see
`docs/results.md`) suggests this has not been a major issue for
Westminster specifically.

A road graph's *nodes* are naturally intersections and its *edges* are
naturally road segments — but a GNN operating over *segments* (the unit
Greyspot actually wants to rank) needs segments to be first-class graph
nodes. This project's `features/graph_temporal.py` implements the standard
solution — a **line graph** transformation (segment → node, shared
intersection → edge) — directly rather than via `networkx.line_graph`,
for the practical reason documented in that module's docstring (edge
orientation in an undirected line graph can silently relabel node
identity).

## 4. Equity and deprivation in road-safety data

The English Indices of Deprivation (IMD) are the standard UK small-area
deprivation measure and are already used in transport-equity research to
check whether risk, exposure, or service quality differs systematically by
deprivation decile. Greyspot's current IMD integration (2019 release,
LSOA level) surfaced a genuine data-quality problem during implementation:
STATS19's LSOA field and the IMD 2019 lookup use different census
geography vintages (2021 vs 2011), leaving ~5.2% of Westminster collisions
unmatched (see `docs/decision_log.md`). This is not a literature gap so
much as a *known, general* problem with using any pre-2021 deprivation
release against post-2021 administrative geography — the standard fix
(an ONS best-fit lookup between LSOA vintages) is a known, published
technique, not a novel contribution, and remains an open implementation
item for this project (blocked on ONS Open Geography Portal's anti-bot
protection at time of writing — see `docs/decision_log.md`).

## 5. Gap-to-design traceability

| Literature gap / precedent | Design consequence in Greyspot |
|---|---|
| GAT+GRU precedent exists but uses a Tweedie decoder for zero-inflation | Implemented GAT+GRU with a simpler Poisson decoder first; Tweedie decoder flagged as a scoped extension |
| Conformal prediction on graphs violates plain exchangeability assumptions | Used plain split-conformal as a first pass; documented as a simplification, not a validated graph-aware method |
| OS Open Roads is the "official" network but gated behind account signup | Used OSMnx as a sanctioned fallback; architecture keeps network source swappable |
| IMD releases and STATS19 use different LSOA census vintages | Logged the resulting ~5.2% join failure rather than silently dropping or fabricating a match; crosswalk fix scoped but not yet implemented |
| Standard evaluation risk: random splits leak information on spatial/temporal data | Implemented temporal, spatial and spatiotemporal held-out splits from the start (`eval/splits.py`), never a random shuffle |

## Provisional reference list (Harvard-style — confirm against your actual
course's required referencing style in the module handbook)

- Boeing, G. (2017, updated 2025) 'OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks', *Computers, Environment and Urban Systems*.
- Cordier, T. et al. (2022) 'MAPIE: an open-source library for distribution-free uncertainty quantification', arXiv:2207.12274.
- Department for Transport (2025–2026) *Road safety data and statistics: STATS19 review update and future plans*. GOV.UK.
- Gao, X., Jiang, X., Haworth, J., Zhuang, D., Wang, S., Chen, H. and Law, S. (2024) 'Uncertainty-aware probabilistic graph neural networks for road-level traffic accident prediction', *Accident Analysis & Prevention* (preprint arXiv:2309.05072).
- Huang, K., Jin, Y., Candès, E. and Leskovec, J. (2023) 'Uncertainty quantification over graph with conformalized graph neural networks', *NeurIPS 2023*.
- Ministry of Housing, Communities & Local Government (2019) *English Indices of Deprivation 2019*. London Datastore.
- Transport for London (2026) *Vision Zero Action Plan 2*.
- Zargarbashi, S.H., Antonelli, S. and Bojchevski, A. (2023) 'Conformal prediction sets for graph neural networks', *ICML 2023 / PMLR 202*.

*Still to add once the official literature-review phase begins*: a
systematic search protocol (databases searched, search terms, inclusion/
exclusion criteria) per L6.07's "apply appropriate research methodologies"
outcome — the reading above was targeted verification of the project
dossier's claims, not yet a systematic review.
