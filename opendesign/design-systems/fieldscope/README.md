# Fieldscope

The design system for **Greyspot: Government Edition** — a road-risk
prioritisation tool for London borough analysts (BSc final-year project,
Westminster 6COSC023W). Built by request as a **full creative departure**
from the project's earlier GOV.UK-styled interface — see
`../../../DESIGN.md` and `../../../docs/decision_log.md` (2026-08-31 entry)
for that earlier system, which this one replaces as the applied identity
while remaining on record as the project's design history.

## The concept, in one sentence

Greyspot reads as a precision survey instrument: a warm "paper" analysis
surface (priority queue, evidence panel) framed by a dark "chassis" bezel
(header, map) — see `SKILL.md` for the non-negotiables this implies, and
`brand/style-notes.md` for how it plays out per component.

## Sources consulted

- **`apps/web/src/App.tsx`** — the real state machine: borough selection,
  three parallel fetches (model info / roads GeoJSON / priority queue) per
  borough switch, the `boroughChangedRef` guard, evidence fetch on segment
  select. This is the actual flow the redesign must not break.
- **`apps/web/src/api.ts`** — the real data shapes (`ModelInfo`,
  `PriorityQueueRow`, `RoadEvidence`) — every field name used in the UI kit
  and in `brand/style-notes.md` comes from here, not invented.
- **`apps/web/src/components/{Header,MapView,PriorityQueue,EvidencePanel}.tsx`**
  — read in full. `MapView.tsx` in particular carries a long, hard-won
  history (worker-URL fix, `beforeId` label-layer ordering, the hover-card
  →click-chip Street View fix, the pan/scroll selection-sync fix) that this
  redesign must preserve functionally while restyling.
- **`apps/web/src/App.css`** — the previous GOV.UK token set, read as
  evidence of *what content and structure must survive* (RAG three-band
  language, panel structure, real spacing rhythm), not as a source for the
  new palette — per the explicit brief.
- **`DESIGN.md` / `docs/decision_log.md`** (project root) — prior design
  decisions and the reasoning behind them (WCAG contrast fixes, the
  basemap swap, the flow-verification pass), so this redesign doesn't
  silently regress an already-solved problem (e.g. the RAG-amber/red
  text-contrast fix has a direct equivalent in this system's
  `-on-paper` token variants).

## Confidence levels

**Confident:**
- The paper/chassis zoning maps cleanly onto the product's real structure
  (map+header are one "live/spatial" register; queue+evidence are one
  "analysis/reading" register) — this isn't an arbitrary aesthetic split.
- RAG semantics, the non-negotiable "no estimate ≠ low risk" rule, and the
  existing copy voice are preserved exactly — these are product-correctness
  properties, not style, and nothing about a "full creative departure"
  brief should touch them.
- The motion set in `brand/style-notes.md` is scoped to states the app
  already has (selection, loading, borough switch) — no invented feature.

**Less confident — flag for the user to confirm:**
- **STIX Two Text as the display face** is a real departure in *character*
  (an academic/scientific-typesetting serif) for a tool whose primary
  audience is government analysts, not journal readers. It was chosen
  deliberately to serve the project's stated publication ambition (see
  `docs/publication_readiness.md` / the user's memory note on this) — the
  face itself is a "this is a research instrument" signal, and was picked
  specifically over more common display serifs (Fraunces and its peers now
  read as the default "distinctive-looking" AI choice for this role,
  flagged by this project's own design-quality detector during this build)
  so the choice stays genuinely specific rather than generically tasteful.
  If the analyst-tool framing should dominate over the publication framing,
  a technical grotesk display face would be the fix, not a full re-plan.
- **The verdigris accent hue** was chosen to sit outside all three RAG
  hues (avoiding the collision a blue or amber accent would cause) while
  reading as "instrument" rather than "government" or "corporate SaaS." It
  is a genuine aesthetic bet, not derived from any existing brand asset —
  there was no existing non-GOV.UK brand material to anchor it to.
- **No separate JSX UI-kit components were written** (only the single
  `ui-kit-greyspot/index.html` showcase). The scope this pass was
  explicitly chosen as "redesign the live app in place," so the real,
  canonical implementation is `apps/web/src/components/*.tsx` itself —
  writing a second, parallel JSX copy here would create two versions of
  the same components that could drift. The showcase file exists to let
  the design be reviewed/screenshotted independent of the live app's
  build step, not as a second source of truth.

## What's in this folder

- `SKILL.md` — portable summary + non-negotiables (read this first).
- `tokens/colors_and_type.css` — canonical tokens.
- `brand/voice-and-tone.md` — copy voice (mostly "keep exactly as-is," with
  the reasoning for why).
- `brand/style-notes.md` — component conventions, especially the motion
  patterns that answer the "animation, all the stuff" part of the brief.
- `ui-kit-greyspot/index.html` — static interactive showcase of the four
  real product surfaces in this system.
