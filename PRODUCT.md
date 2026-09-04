# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Existing codebase: React 19 + TypeScript + Vite 5 (`apps/web/`), MapLibre GL
JS v6 for the map, a FastAPI backend (`services/api/`). Not asked fresh —
already established and working (see `README.md`).

## Users

**Primary user (confirmed 2026-08-31): government road-safety analysts and
policy officers** at a local authority (e.g. a London borough council) or
a body like TfL, using the dashboard to decide where to prioritise road
safety interventions and how to justify that prioritisation with evidence.
This is an internal, expert decision-support tool used repeatedly by a
small, trained user base — not a public-facing transparency site. (The
project dossier separately envisions a possible future public/analyst
dual view — FR10 in `docs/requirements_and_ethics.md` — but that is
explicitly out of scope for this redesign pass; this pass covers the
analyst view only.)

## Product Purpose

Greyspot estimates relative road-safety risk from historical collision
data and road-network structure, quantifies its own uncertainty, and
explains the evidence behind every priority ranking — so an analyst can
defend a prioritisation decision with a specific, auditable reason, not a
black-box score. Success means an analyst can look at any ranked road
segment and immediately see *why* it ranks where it does (score
breakdown, observed collision history, exposure, model prediction and
uncertainty, and stated limitations), and can move between boroughs as
the project scales toward covering all of London.

## Positioning

Unlike a generic "hotspot map" (which just shows raw collision counts) or
an opaque ML risk score, Greyspot's mechanism is: (1) a transparent,
component-weighted Safety Priority Score an analyst can decompose and
audit, (2) calibrated conformal uncertainty intervals on every prediction
(not just a point estimate), and (3) an explicit, enforced refusal to
claim a location is "safe"/"unsafe" or that a specific collision will
happen — the model outputs are labelled relative-risk research signals
throughout, never absolute safety claims. A neighbouring product that
skips the graph-network structure, the uncertainty quantification, or the
transparent scoring breakdown could not truthfully make the same claims.

## Operating Context

An analyst opens the dashboard, selects their borough, and works from a
**priority queue** (ranked road segments) alongside a **map** (the same
segments, colour-coded by risk) and an **evidence panel** (the detail for
whichever segment is selected on either the map or the queue). The
workflow is: scan the queue -> click a segment -> read its full evidence
-> decide whether/how to act -> move to the next segment or switch
borough. This is desk-based analytical work, likely on a standard office
monitor, done in sessions of minutes to tens of minutes, not a glance-and-
go mobile interaction — though it should still work reasonably on a
laptop screen. Real, live data throughout: seven London boroughs
currently modelled (Westminster, Lambeth, Camden, Kensington & Chelsea,
Brent, Wandsworth, City of London), each with a GPU-trained GAT+GRU/
XGBoost model ladder and real STATS19/IMD/AADF data behind it — nothing
in this product is a mockup or placeholder dataset.

## Capabilities and Constraints

- Confirmed working end-to-end today: borough selection, interactive map,
  ranked priority queue, full per-segment evidence panel, all backed by a
  live FastAPI service (`services/api/`) trained on real government open
  data (STATS19, IMD 2019, DfT AADF).
- **Hard constraint, non-negotiable (dossier + `docs/requirements_and_ethics.md`
  NFR5)**: the product must never state or imply that a location is
  "safe" or "unsafe" in absolute terms, or that a specific collision will
  or will not happen. Every risk output is a relative research signal.
  This applies to UI copy as much as to model output — a redesign must
  not introduce confident-sounding copy that violates this.
- Not yet built: scenario/portfolio tools, AI copilot, report generator
  (FR11-FR14) — out of scope for this UI redesign pass.
- Data currency: each borough's evidence panel already states its own
  coverage window (e.g. "Historical evidence only covers 2021-2024 for
  this borough") — this kind of honest data-provenance framing is a
  product commitment, not just a nice-to-have, and should be preserved or
  strengthened, not dropped, in any redesign.
- Known current model limitation, worth reflecting honestly in the UI
  rather than hiding: model reliability varies by borough and by split
  type (see `docs/results.md`) — the product's own "confidence"/interval-
  width display exists specifically so an analyst isn't misled by a
  false sense of precision.

## Brand Commitments

- Name: **Greyspot** (product), full name **Greyspot: Government
  Edition** (project). Existing wordmark/logotype treatment in
  `apps/web/src/components/Header.tsx` is plain text, not a
  fixed/binding asset — free to redesign.
- Tone: precise, evidence-led, calm - a government analytical tool, not a
  marketing product. No hype language, no unfounded certainty claims (see
  the non-negotiable boundary above).

## Evidence on Hand

Real, not placeholder: STATS19 collision/casualty/vehicle records (DfT,
2021-2025), IMD 2019 deprivation deciles (London Datastore), DfT AADF
traffic-count exposure data, OSMnx-derived road network graphs for seven
London boroughs, and trained model outputs (XGBoost + GAT+GRU, MAPIE/
manual conformal intervals) for all seven. No user testimonials, case
studies, or press exist or should be fabricated - this is a final-year
academic project, not a product with real deployment history yet.

## Product Principles

1. **Transparency over polish-that-hides-uncertainty** - every number the
   UI shows must be traceable to a specific model version, data year, and
   (where applicable) confidence interval; never show a bare score
   without a way to see its basis.
2. **Never imply certainty the model doesn't have** - visual design
   choices (colour intensity, iconography, copy) must not make a
   low-confidence prediction look as authoritative as a high-confidence
   one.
3. **Built for repeated expert use, not first impressions** - optimise
   for scanability and low cognitive load across many segments and
   sessions over a flashy first-look "wow", per Impeccable's "Operate"
   mode.
4. **Honest about what's not yet known** - a borough with less historical
   data, a model with wider intervals, or a split where the graph model
   underperforms XGBoost should read as honestly uncertain in the UI, not
   be visually smoothed over.
5. **Government-appropriate, not corporate-SaaS** - the visual language
   should read as a serious public-sector analytical tool (GOV.UK Design
   System conventions, confirmed 2026-08-31), not a startup dashboard.

## Accessibility & Inclusion

**Confirmed 2026-08-31**: this redesign follows **GOV.UK Design System**
conventions and targets **WCAG 2.2 AA** - both because this is genuinely
a public-sector tool in spirit ("Government Edition") and because it
strengthens the dissertation's requirements/ethics chapter
(`docs/requirements_and_ethics.md`) to design and audit against a real,
checkable standard rather than an unstated one. Concretely: sufficient
colour contrast (colour must never be the *only* signal for risk level -
needed anyway for colour-blind analysts reading a risk map), visible
keyboard focus states, semantic HTML landmarks, and accessible names on
all interactive controls (map controls, borough selector, queue rows).
