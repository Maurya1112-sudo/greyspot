# Project management: timeline, risk register, decision gates

*Addresses L6.05 (full lifecycle handling). Current date: 2026-08-31 — this
work is a pre-term head start; the timeline below assumes an October 2026
start to formal supervision, adjust against your actual Blackboard-published
deadlines the moment they're available (see `docs/module_context.md`).*

## 1. Timeline (adapted from the project dossier's roadmap, Section 18)

| Phase | Planned window | Status | Evidence |
|---|---|---|---|
| Pre-term head start | Aug 2026 | **Done** | This repository: real data pipeline, baselines, GAT+GRU, conformal, ablations, full documentation suite |
| Discovery / proposal | Oct–Nov 2026 | Partially done early | `docs/proposal.md`, `docs/module_context.md`; still needed: supervisor sign-off, live rubric cross-check |
| Data pipeline | Nov–Dec 2026 | **Done early** | STATS19 + OSMnx + IMD ingestion, 100% collision-to-segment join rate |
| Baselines | Dec 2026–Jan 2027 | **Done early** | Historical-rate + XGBoost, `docs/results.md` |
| Graph-temporal model | Jan–Feb 2027 | **Done early**, cross-validated on 7 boroughs | GAT+GRU + ablations, `docs/results.md` |
| Uncertainty calibration | Feb–Mar 2027 | **Done early**, cross-validated on 7 boroughs | Conformal (MAPIE + manual), `docs/results.md` |
| Product layer (priority score, map, evidence panel) | Mar–Apr 2027 | **Done early** | FastAPI backend (`services/api/`, 10/10 tests passing) + React/MapLibre frontend (`apps/web/`), verified end-to-end in-browser for multiple boroughs. `docs/requirements_and_ethics.md` FR9–FR10 |
| Scenario/portfolio tools | Apr 2027 | Not started | FR11–FR12 |
| Report generator + AI copilot + hardening | Apr–May 2027 | Not started | FR13–FR14 |
| Dissertation + viva | May 2027 | In progress (this doc suite) | `docs/` |

**Important**: being "done early" on the research components is a genuine
head start, not a reason to treat them as closed. Each has explicit
follow-up items (see `docs/decision_log.md`'s "Open item" entries) that
should be revisited once formal supervision begins.

## 2. Risk register

| Risk | Likelihood (current view) | Status | Mitigation taken |
|---|---|---|---|
| Network join too noisy | Low | Resolved for Westminster | 100% collision-to-segment snap rate achieved |
| GAT too heavy for laptop compute | Very low | Resolved | Full pipeline (5 GAT training runs incl. ablations) completes in well under two minutes end-to-end, GPU-accelerated (CUDA, RTX 4060 Laptop GPU, auto-detected) |
| Conformal under-coverage | Low (XGBoost); **realised (GAT)** | Partially resolved | XGBoost hits 90.6-91.7% across all 7 boroughs tested; GAT's manual split-conformal under-covers in 5/7 boroughs (as low as 84.4%) — a real, documented limitation, see `docs/decision_log.md` and `docs/scaling_to_london.md` |
| Graph adds no value | **Realised, being handled correctly — now precisely characterised across 7 boroughs** | Ongoing, better understood | No reliable temporal-split advantage (XGBoost wins/ties 6/7 boroughs), but a real, replicated spatiotemporal-split advantage (GAT wins 5/7 boroughs) — reported honestly, not averaged away into a single misleading number |
| Exposure data unavailable | N/A | **Resolved** | DfT AADF traffic-count data integrated as an exposure feature across all boroughs |
| Frontend build tooling instability | Realised | **Resolved** | The Vite 8 scaffold default (experimental Rolldown bundler) was incompatible with maplibre-gl's Web Worker; pinned to stable Vite 5 + esbuild, verified end-to-end in-browser |
| OS Open Roads access delayed | Realised, then resolved 2026-09-01 | **Closed — and it turned out to matter enormously** | User registered and downloaded the GB-wide GeoPackage; `ingest/os_open_roads.py` wires it in as a selectable network source. **2026-09-02**: running it end-to-end for the first time revealed the network source is the single largest driver of UCL-benchmark performance (+11.59 AccHR@20 points, p=0.0010) — see `docs/ucl_benchmark_results.md`. Two real bugs were fixed getting there (bbox-vs-polygon clipping; UUID-ID cache reload) |
| Stale project notes causing missed opportunities | **Realised 2026-09-02** | Handled, worth watching | A project note asserted the OS Open Roads file was "gone from local disk" when it had been present since the previous day — this delayed the single most valuable experiment of the investigation by an entire session. **Lesson: verify claims about the environment by checking, not by trusting a prior note.** Applied immediately afterwards (re-verified AADF, LSOA, IMD files are all genuinely present rather than assuming) |
| LSOA vintage mismatch (STATS19 vs IMD 2019) | Realised | Open | ~5.2% of collisions unmatched; ONS crosswalk blocked by anti-bot protection on all three access attempts (API, WebFetch, embedded browser) — needs a manual download |
| Scope creep into the product layer before the research is solid | Actively managed | Ongoing | This document + `docs/decision_log.md` exist specifically to keep research and product work traceable and separable |

## 3. Decision gates (from the dossier, Section 20, tracked against actual status)

| Gate | Criterion | Status |
|---|---|---|
| Gate 1 | Data and graph trustworthy enough to model | **Passed** — 100% join rate, documented data-quality caveats |
| Gate 2 | Baseline evaluation pipeline frozen | **Passed** — temporal/spatial/spatiotemporal splits implemented and tested |
| Gate 3 | Graph model earns its inclusion on held-out data | **Passed, with a precise scope** — earns its place specifically on the spatiotemporal (unseen-segment) task, replicated across 5/7 boroughs; does not earn its place on the simpler temporal task, and this distinction is now the reported finding rather than an open question |
| Gate 4 | Uncertainty calibration passes minimum coverage tests | **Passed for XGBoost** (90.6-91.7% across 7 boroughs); **partially passed for GAT** (at/above target in 2/7 boroughs, under-covering in the rest — a real limitation, not closed) |
| Gate 5 | Core UI is usable before advanced features are added | **Passed** — React/MapLibre frontend (`apps/web/`) verified end-to-end: map render, priority queue, evidence panel, borough switching, all working against the live FastAPI backend |
| Gate 6 | Portfolio/copilot features only included if they don't threaten dissertation evidence | **N/A yet** — not started |

## 4. Immediate next actions (in priority order)

1. Get the live 6COSC023W handbook, rubric and deadlines from Blackboard;
   reconcile against `docs/module_context.md`.
2. Confirm supervisor and get sign-off on the research question and scope
   (`docs/proposal.md`).
3. **Stage 4 of `docs/scaling_to_london.md`**: train one model on several
   boroughs' combined graph, test on a borough held out entirely — the
   natural next experiment now that seven single-borough runs show
   precisely which generalisation claim holds (spatiotemporal) and which
   doesn't (temporal).
4. Investigate the GAT's conformal under-coverage (5/7 boroughs below the
   90% target) — candidate fixes: a larger/rolling calibration window, or
   an asymmetric CQR-style interval instead of raw split-conformal.
5. ~~Manually download the OS Open Roads dataset~~ — **done 2026-09-01**,
   see the risk register. Still open: the ONS LSOA 2011↔2021 crosswalk
   (blocked from automated retrieval by anti-bot protection).
6. Statistical significance testing across the seven boroughs' paired
   PR-AUC differences (currently reported as point estimates only).
7. Run OS Open Roads on the remaining six boroughs and re-run the
   Westminster architecture sweep against its denser (~3x) graph
   topology — the "1 head + residual" over-smoothing fix was tuned
   against OSMnx's coarser line-graph and may not be optimal for OS Open
   Roads' finer segmentation (see `docs/decision_log.md`, 2026-09-01).

---

## Status update 2026-09-03

**Closed:**
- Item 6 (statistical significance testing) — **done and now mandatory**.
  Every claim carries a paired t-test and Wilcoxon test over matched
  windows via `scripts/paired_significance.py`, plus cross-borough
  replication before any result is reported.
- The AccHR@20 gap workstream produced a second significant finding:
  long-horizon crash history, +6.03 points pooled, p=0.0303.

**Current headline:** Lambeth 70.29%, Westminster 75.39% (exceeds Gao
et al.'s 68.98% by 6.41), pooled 72.84% vs their 72.60%. Tower Hamlets
running to complete n=18.

**Still open:**
- Tower Hamlets replication of the long-history result (in progress).
- Lambeth remains 6.30 points below the paper — narrowed from 13.00.
- The ~140× zero-inflation discrepancy vs the paper's Table 7.2:
  unexplained after checking their TCR formula (matched and ruled out),
  segment consolidation (ruled out on real data) and evaluation
  protocol (ruled out). A data request to the authors has been drafted
  (their published Data Availability statement offers data on request;
  corresponding author Xiaowei Gao, xiaowei.gao.20@ucl.ac.uk) — **for
  the student to send, not sent automatically.**
- ONS LSOA 2011↔2021 crosswalk still blocked by anti-bot protection.
