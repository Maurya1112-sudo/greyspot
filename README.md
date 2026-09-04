# Greyspot: Government Edition

Uncertainty-aware, network-connected road-risk prioritisation for London.
Proven end-to-end — research model, FastAPI backend, and a React/MapLibre
government decision-support UI — on all seven of Westminster's real
neighbouring boroughs (Westminster, Lambeth, Camden, Kensington & Chelsea,
Brent, Wandsworth, City of London), with a clear staged plan to expand
across the rest of London (see
[`docs/scaling_to_london.md`](docs/scaling_to_london.md)). This is the
**Greyspot: Government Edition** final-year project — see
[`docs/proposal.md`](docs/proposal.md) for the full scope and
[`docs/research_notes.md`](docs/research_notes.md) for the source verification
behind the design choices below.

Three layers, all working today:
1. **Research core** (`src/greyspot/`) — real data ingestion, a
   borough-parameterised road-network pipeline, the full research model
   ladder (historical-rate -> XGBoost -> GAT+GRU graph-temporal, GPU-
   trained), conformal uncertainty, and interactive risk maps.
2. **Product layer backend** (`services/api/`) — a FastAPI service serving
   per-borough model outputs, a transparent Safety Priority Score
   (`src/greyspot/product/priority_score.py`), and per-segment evidence.
3. **Product layer frontend** (`apps/web/`) — a React + TypeScript +
   MapLibre GL JS government decision-support UI: an interactive risk map,
   a ranked priority queue, and an evidence panel showing exactly why each
   road is ranked where it is (score breakdown, observed history,
   exposure, model prediction, stated limitations).

**Headline research result** (see
[`docs/results.md`](docs/results.md) and
[`docs/scaling_to_london.md`](docs/scaling_to_london.md) for full detail):
across all seven boroughs, **there is no reliable GAT advantage on the
simple temporal (same-segment, next-year) task** — XGBoost matches or
beats it in 6/7 boroughs, sometimes by a wide margin. But on the
**spatiotemporal task (an unseen road segment, in an unseen year)** — the
harder, more policy-relevant generalisation test — the graph-attention
architecture shows a real, consistent improvement (+0.037 to +0.062
PR-AUC) in **5/7 boroughs**. The honest finding is not "the graph wins" or
"the graph loses" but **"the graph architecture specifically helps
generalising to roads without collision history, not forecasting on roads
that already have it."** This finding replaced two earlier, more dramatic
but invalid claims, both caught and corrected in the open rather than
quietly edited away — a data-leakage bug (the model was trained directly
against its own evaluation target, PR-AUC 0.424) and a later stale-number
mistake (the leakage-corrected number, 0.331-0.333, sat undisturbed next
to an uncorrected old 0.424 in a different log entry until a fresh re-run
caught the discrepancy). Both corrections are documented in full in
[`docs/decision_log.md`](docs/decision_log.md) — see the "CORRECTION" and
"process note" entries — because the diagnose-fix-verify story is itself
evidence of rigorous practice, not something to hide.

## What's here

```
src/greyspot/
  ingest/boroughs.py        # borough registry (name, ONS code, OSM place) - add a borough here
  ingest/stats19.py         # STATS19 collision/casualty/vehicle download + local-authority filter
  ingest/network.py         # OSMnx graph (any borough) + collision/point-to-segment snapping
  ingest/imd.py              # IMD 2019 equity context lookup
  ingest/exposure.py         # DfT AADF traffic-volume exposure data
  features/build_features.py    # segment-year feature/target table (leakage-safe lagging)
  features/graph_temporal.py    # segment-level line graph + [T,N,F] tensors for the GAT
  models/baseline.py        # historical-rate baseline
  models/xgboost_model.py   # XGBoost comparator (strong non-graph baseline)
  models/gat_temporal.py    # GAT + GRU graph-temporal research model (GPU auto-detect, walk-forward training)
  models/conformal.py       # MAPIE (XGBoost) + manual (GAT) split-conformal uncertainty
  product/priority_score.py  # transparent Safety Priority Score (policy layer, dossier Section 5A)
  eval/splits.py            # temporal + spatial held-out splits (no random shuffling)
  eval/metrics.py           # Precision@K, PR-AUC, Spearman rank correlation
  viz/map_demo.py            # Folium/Leaflet map (fast sanity-check version)
  viz/maplibre_map.py        # MapLibre GL JS map (standalone risk-map export)
scripts/run_pipeline.py        # end-to-end, takes a borough name as its argument
scripts/experiment_gat_architecture.py            # fast GAT architecture sweep (reuses cached features)
scripts/experiment_gat_best_config_validation.py  # validates a sweep winner on the harder split
services/api/
  app/data_service.py        # per-borough model cache (trains XGBoost + conformal + priority score on demand)
  app/main.py                 # FastAPI endpoints: /boroughs, /roads, /priority-queue, /roads/{id}
  run.py                      # uvicorn entrypoint
apps/web/
  src/App.tsx                 # wires borough selection, map, priority queue, evidence panel
  src/components/MapView.tsx  # MapLibre GL JS map (React + TypeScript, no default export - see note below)
  src/components/PriorityQueue.tsx, EvidencePanel.tsx, Header.tsx
  src/api.ts                  # typed fetch client for the FastAPI backend
docs/                      # full academic documentation suite - see docs/dissertation_outline.md
```

## Documentation (University of Westminster 6COSC023W)

This is a BSc Computer Science final-year project (module **6COSC023W,
Computer Science Final Project**, 40 credits, Level 6 — verified facts in
[`docs/module_context.md`](docs/module_context.md)). Start at
[`docs/dissertation_outline.md`](docs/dissertation_outline.md), which maps
every document below to a standard dissertation chapter:

| Document | Purpose |
|---|---|
| [`docs/module_context.md`](docs/module_context.md) | Verified 6COSC023W facts + learning-outcome mapping |
| [`docs/proposal.md`](docs/proposal.md) | Research question, hypotheses, scope tiers |
| [`docs/research_notes.md`](docs/research_notes.md) | Source verification for every external claim |
| [`docs/literature_review.md`](docs/literature_review.md) | Reviewed by theme, with a gap-to-design table |
| [`docs/requirements_and_ethics.md`](docs/requirements_and_ethics.md) | Functional/non-functional requirements, ethics self-assessment |
| [`docs/methodology.md`](docs/methodology.md) | Data, splits, models, leakage controls, evaluation protocol |
| [`docs/results.md`](docs/results.md) | Regenerated results chapter draft (all seven boroughs) |
| [`docs/decision_log.md`](docs/decision_log.md) | Dated, append-only log of every decision, bug and finding |
| [`docs/project_management.md`](docs/project_management.md) | Timeline, risk register, decision gates |
| [`docs/scaling_to_london.md`](docs/scaling_to_london.md) | Concrete staged plan: Westminster → Lambeth → all of London |
| [`docs/publication_readiness.md`](docs/publication_readiness.md) | Honest gap analysis against the closest academic precedent (UCL's Gao et al. 2024), for the "might publish" ambition |
| [`docs/supervision_log.md`](docs/supervision_log.md) | Supervisor-meeting / work-session log template |

**Note**: the live Blackboard module handbook, assessment brief and marking
rubric for your actual cohort override the publicly-sourced facts in
`docs/module_context.md` — check them against it as soon as you have access.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

## Run the research pipeline

```bash
python scripts/run_pipeline.py                          # Westminster (default)
python scripts/run_pipeline.py Lambeth                   # any registered borough - see ingest/boroughs.py
python scripts/run_pipeline.py "Kensington and Chelsea"  # quote multi-word names
```

This downloads STATS19 collision/casualty/vehicle data for the chosen
borough (cached under `data/raw/`, shared across boroughs) plus IMD 2019
and DfT AADF exposure, builds that borough's road network via OSMnx
(cached under `data/interim/`), snaps collisions to road segments, trains
the historical-rate baseline, XGBoost, the GAT+GRU graph-temporal model
(1 head + residual connection, GPU-accelerated when CUDA is available —
auto-detected, see `models/gat_temporal.py::_resolve_device`) and two
ablations, evaluates all of them on temporal/spatial/spatiotemporal
held-out splits (`reports/{borough}/baseline_vs_xgboost_results.csv`),
fits conformal uncertainty layers on both XGBoost (MAPIE) and the GAT
(manual split-conformal) and reports empirical coverage
(`reports/{borough}/conformal_calibration.csv`), and writes two interactive
risk maps — a MapLibre GL JS one (primary) and a Folium one (fast
fallback) — to `reports/{borough}/figures/`. Full interpretation in
[`docs/results.md`](docs/results.md).

To investigate GAT architecture variants without re-running the ~90-second
ingestion step, use the cached feature table:

```bash
python scripts/experiment_gat_architecture.py
```

## Run the product layer (backend + frontend)

The backend trains/caches each borough's model on first request (no
separate build step) and serves it over HTTP:

```bash
python services/api/run.py
# -> http://127.0.0.1:8000 (docs at /docs)
```

The frontend is a standard Vite dev server:

```bash
cd apps/web
npm install
npm run dev
# -> http://localhost:5173
```

**Note on the Vite version**: `apps/web` is pinned to classic esbuild-based
**Vite 5.4** (not the Vite 8 default from `npm create vite@latest`, which
ships an experimental Rolldown-based dependency optimiser that mishandles
maplibre-gl's internal Web Worker — see `docs/decision_log.md`'s
"rolldown-vite/MapLibre incompatibility" entry, 2026-08-31, before
"upgrading" this dependency). Also note `maplibre-gl` v6 has **no default
export** — `MapView.tsx` uses `import * as maplibregl from "maplibre-gl"`,
not a default import.

Endpoints (`services/api/app/main.py`): `GET /boroughs`, `GET
/boroughs/{name}/model-info`, `GET /boroughs/{name}/roads` (GeoJSON, for
the map), `GET /boroughs/{name}/priority-queue?limit=&min_confidence=`,
`GET /boroughs/{name}/roads/{segment_id}` (full evidence panel: score
breakdown, observed history, exposure, model prediction, stated
limitations). Tested end-to-end against real Westminster data
(`services/api/tests/test_api.py`, 10/10 passing) and verified visually in
the browser (map render, priority-queue click-through, evidence panel
population).

## Design

`apps/web` follows the **GOV.UK Design System**, played straight — a
deliberate, user-confirmed choice (2026-08-31), not an invented aesthetic.
Full rationale, token inventory, and component conventions are in
[`PRODUCT.md`](PRODUCT.md) and [`DESIGN.md`](DESIGN.md) at the project
root. Built with the [Impeccable](https://github.com/pbakaus/impeccable)
design skill (`~/.claude/skills/impeccable`) — every colour, spacing
value, and component was verified against the real
design-system.service.gov.uk spec (not approximated from memory), every
text/background colour pair was checked against a computed WCAG 2.2
relative-luminance contrast ratio (two real AA failures were found this
way and fixed — see `docs/decision_log.md`), and the mechanical design
detector (`node .claude/skills/impeccable/scripts/detect.mjs`) was run
over every changed file. Headline choices: a black GOV.UK-style header,
a RAG (Red-Amber-Green) risk colour scale used consistently across the
map/priority-queue/evidence-panel (the UK civil-service convention for a
priority rating, not an arbitrary gradient), GOV.UK's real `tag` and
`warning-text` components, and the signature solid-yellow focus outline
on every interactive element. Deliberately **not** using the licensed GDS
Transport typeface or the crown/GOV.UK logo — this product borrows the
design *language*, not the *identity*, of a real government service (see
DESIGN.md's Overview for why that distinction matters here).

## Data sources

| Source | Use | Access |
|---|---|---|
| [STATS19 / DfT road safety open data](https://www.gov.uk/government/statistics/road-safety-data) | Collision records | Free, direct CSV download |
| [OSMnx / OpenStreetMap](https://osmnx.readthedocs.io/) | Road network graph (default; no local GeoPackage required) | Free, no registration |
| [OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads) | Preferred official road network — **wired in 2026-09-01** as a selectable source (`ingest/os_open_roads.py`) | Free (Open Government Licence), requires an OS Data Hub account and a manual ~2GB GeoPackage download to `oproad_gpkg_gb/` |
| [IMD 2019 / London Datastore](https://data.london.gov.uk/dataset/indices-of-deprivation-2l15g) | Equity context (average IMD decile per segment, lagged) | Free, direct download — in use |
| STATS19 casualty + vehicle tables | Severity/vulnerable-user/vehicle-mix features (lagged) | Same access as collisions — in use |
| [DfT AADF traffic counts](https://roadtraffic.dft.gov.uk/downloads) | Exposure (same-year, unlagged — see `docs/decision_log.md`) | Free, direct bulk CSV — in use |

**OS Open Roads attribution** (the exact wording OS's own licence file
requires, `oproad_gpkg_gb/Doc/licence.txt`, required whenever results
derived from it are published or submitted): *Contains OS data © Crown
Copyright and database rights 2026.*

## Non-negotiable boundary

Consistent with the project dossier: this is an investigation/prioritisation
research aid. It does not claim that a specific collision will happen, that a
location is "safe" or "unsafe", or that any modelled score is a validated
policy rule. Model outputs are relative-risk research signals only.
