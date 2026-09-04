---
name: fieldscope
description: Fieldscope — the "precision survey instrument" design system for Greyspot (Government Edition), a road-risk intelligence tool for London borough analysts. Use for any UI work on the Greyspot product surface.
---

# Fieldscope

A full creative departure from Greyspot's earlier GOV.UK-styled interface, built by reading the real product (`apps/web/`, `services/api/`) rather than starting from a blank brief. See `../../../DESIGN.md` (root of the FYP repo) for the applied, up-to-date record of how these tokens are used in the live app, and `README.md` in this folder for the full rationale.

## The one-sentence concept

Greyspot reads as a physical survey instrument — a warm "paper" analysis surface (the priority queue, the evidence panel) framed by a dark "chassis" bezel (the header, the map viewport) — not a government form and not a SaaS dashboard.

## Non-negotiables when applying this system

1. **RAG (red/amber/green) stays the only risk-severity language.** Defined once in `tokens/colors_and_type.css` (`--risk-low` / `--risk-mid` / `--risk-high` / `--risk-none`, each with a `-wash` and `-on-paper` variant for the two surface temperatures) — every component reads from these, never a local hex.
2. **The accent (verdigris teal, `--accent`) is brand/interaction only** — links, focus rings, the wordmark, selection state. It must never appear on risk data, or it will read as a fourth risk band.
3. **Paper vs chassis is a placement rule, not a toggle.** The map and header are chassis (dark); the priority queue and evidence panel are paper (light). A component doesn't get to "pick a theme" — its position in the product decides its surface.
4. **Motion is named by intent** (`--duration-*`, `--ease-*`) and reused, not invented per component. `--duration-map` intentionally matches the MapLibre camera-move duration already in `MapView.tsx` so UI and map motion share one rhythm.
5. **Radius never exceeds `--radius-lg` (6px).** This system is "machined," not "bubbly" — no card gets a large rounded corner as a default aesthetic choice.

## Files

- `tokens/colors_and_type.css` — canonical tokens (colour, type, space, shape, elevation, motion).
- `brand/voice-and-tone.md` — how Greyspot talks (labels, empty states, warnings).
- `brand/style-notes.md` — component-level conventions (cards, tags, focus states, the paper/chassis seam).
- `ui-kit-greyspot/index.html` — a static, interactive showcase of the four real product surfaces (header, map bezel, priority queue, evidence panel) restyled in this system, built from the real component structure and copy in `apps/web/src/components/`.
- `README.md` — sources consulted, confidence levels, open decisions for the user to confirm.
