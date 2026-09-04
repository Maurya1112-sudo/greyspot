---
name: Greyspot
description: A road-risk intelligence tool for London borough analysts, styled as a precision survey instrument — a warm "paper" analysis surface framed by a dark "chassis" bezel.
colors:
  paper-0: "oklch(97.5% 0.006 80)"
  paper-1: "oklch(99% 0.004 80)"
  paper-2: "oklch(93.5% 0.009 80)"
  paper-line: "oklch(87% 0.012 80)"
  ink-1: "oklch(21% 0.012 75)"
  ink-2: "oklch(42% 0.014 75)"
  ink-3: "oklch(55% 0.012 75)"
  chassis-0: "oklch(19% 0.018 250)"
  chassis-1: "oklch(24.5% 0.019 250)"
  chassis-2: "oklch(14.5% 0.017 250)"
  chassis-line: "oklch(33% 0.02 250)"
  fog-1: "oklch(94% 0.006 250)"
  fog-2: "oklch(74% 0.012 250)"
  fog-3: "oklch(60% 0.014 250)"
  accent: "oklch(58% 0.084 196)"
  accent-strong: "oklch(42% 0.094 196)"
  accent-soft: "oklch(88% 0.039 196)"
  risk-low: "oklch(52% 0.135 152)"
  risk-low-on-paper: "oklch(38% 0.11 152)"
  risk-mid: "oklch(68% 0.15 55)"
  risk-mid-on-paper: "oklch(42% 0.13 48)"
  risk-high: "oklch(55% 0.19 25)"
  risk-high-on-paper: "oklch(42% 0.18 25)"
  risk-none: "oklch(62% 0.01 250)"
typography:
  display:
    fontFamily: "STIX Two Text, 'Iowan Old Style', 'Palatino Linotype', Georgia, serif"
    fontWeight: 600
  body:
    fontFamily: "Archivo, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.45
  mono:
    fontFamily: "'IBM Plex Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"
    fontSize: "0.75rem"
rounded:
  sm: "2px"
  md: "4px"
  lg: "6px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "20px"
  6: "28px"
  7: "40px"
  8: "56px"
components:
  tag-green:
    backgroundColor: "{colors.risk-low-wash}"
    textColor: "{colors.risk-low-on-paper}"
    padding: "3px 9px 2px"
  tag-orange:
    backgroundColor: "{colors.risk-mid-wash}"
    textColor: "{colors.risk-mid-on-paper}"
    padding: "3px 9px 2px"
  tag-red:
    backgroundColor: "{colors.risk-high-wash}"
    textColor: "{colors.risk-high-on-paper}"
    padding: "3px 9px 2px"
---

# Design System: Greyspot — "Fieldscope"

## Overview

**Creative North Star: "A Precision Survey Instrument"**

This is Greyspot's second visual identity, and a deliberate, full creative
departure from the first (GOV.UK Design System, "played straight" — see
`docs/decision_log.md`'s 2026-08-31 and 2026-09-01 entries for that system's
full history, kept on record as this project's design lineage, not deleted).
The change was requested explicitly by the user via the OpenDesign plugin
(2026-09-01): "redesign my whole project... be as much creative as you want,
animation all the stuff, it should not look like AI slop... keep in mind the
flow and also the backend works." The canonical design-system source for
this identity lives at `opendesign/design-systems/fieldscope/` (tokens,
brand voice/style notes, sources consulted, confidence levels) — this file
is the applied summary; that folder is the fuller record.

**The concept.** Greyspot reads as one physical object with two zones, not a
single flat theme: a warm "paper" analysis surface (the priority queue, the
evidence panel — where an analyst reads and compares) framed by a dark
"chassis" bezel (the header, the map — the live, spatial, instrument-glass
part of the product). The seam between them is a real design decision
(materials meeting), not an arbitrary colour split — see "Layout" below.

**Why this concept, for this product.** Greyspot is simultaneously (a) a
tool government analysts use repeatedly to make real prioritisation
decisions, and (b) a project with a stated academic-publication ambition
(see `docs/publication_readiness.md`, the user's `greyspot-publication-
ambition` memory). "Precision instrument" serves both: the paper zone reads
as a serious analytical/reading surface (closer to a research instrument's
readout or an editorial data page than a SaaS dashboard), while the chassis
zone gives the live map real presence and depth without either zone
borrowing consumer-app or government-form conventions.

**Key characteristics:**
- Two-material system: warm "paper" (queue, evidence) + cool dark "chassis" (header, map) — never mixed within one component.
- A single brand accent (verdigris teal) reserved for identity/interaction — never used for risk data, so it can never be misread as a fourth RAG band.
- RAG (Red/Amber/Green) risk language preserved exactly from the previous system — this is product truth, not style, and survives the departure unchanged.
- Real, purposeful motion throughout (panel entrance, selection handshake, score count-up, bar fills) — see "Motion" below and `opendesign/design-systems/fieldscope/brand/style-notes.md` for the full rationale per pattern.
- Small, consistent radius (2–6px) — "machined," not rounded-SaaS, not GOV.UK-square.

## Colors

### Paper zone (queue, evidence panel, disclaimer bar)
- **Paper-0/1/2** (warm, chroma ≈0.006–0.009): page background, lifted card surface, recessed well (score hero, bar tracks). Warmed rather than clinical white — a deliberate, chosen neutral, not a default.
- **Ink-1/2/3**: primary / secondary / faint text on paper. Ink-1 is 21% lightness at low chroma — reads as nearly black without being a flat `#000`.

### Chassis zone (header, map)
- **Chassis-0/1/2** (cool graphite, chroma ≈0.017–0.019, hue 250): base / elevated-hover / recessed (map viewport well). Deliberately a *different temperature* from paper, not just a darker version of the same grey — this temperature split is what makes the two zones read as different materials.
- **Fog-1/2/3**: primary / secondary / faint text on chassis.

### Brand accent
- **Accent** (verdigris teal, oklch 58% 0.084 196): links, focus rings, the wordmark's underline sweep, selection state, the 3D-buildings toggle's active state. Chosen specifically to sit outside all three RAG hues (green ≈152°, amber ≈55°, red ≈25°) so it can never be confused with risk data — see the non-negotiable rule below.
- **Accent-strong / accent-soft / accent-wash**: pressed/active, on-dark, and tinted-background variants of the same hue — never a second unrelated hue.

### RAG risk scale (unchanged in meaning, re-tuned for two surfaces)
- **Risk-low / risk-mid / risk-high / risk-none**: the graphic (line, chassis-text) variants.
- **Risk-low/mid/high-on-paper**: WCAG-checked darker variants for tag text on the paper zone's light backgrounds — the direct descendant of the previous system's `-text` variants and the same fix (raw risk-mid/-high fail AA as text on light surfaces; the `-on-paper` variants don't).
- **Risk-*-wash**: light tinted tag backgrounds on paper.

### Named rules
**The One Risk Language Rule** (carried over unchanged): RAG is defined once in `opendesign/design-systems/fieldscope/tokens/colors_and_type.css` and referenced everywhere a risk level is shown.

**The Accent-Never-Means-Risk Rule** (new in this system, direct consequence of choosing a single accent hue outside the RAG range): if a future contributor is tempted to use the accent teal to highlight a "notable" road, don't — that reads as a fourth, undefined risk band. Use a RAG colour (if it's risk-related) or a genuine UI-state token (if it's selection/hover — which the accent already correctly owns).

**The Paper/Chassis Placement Rule**: a component's surface (paper or chassis) is decided by its role in the product (spatial/live vs analytical/reading), not chosen per-component for visual variety.

**Verified, not eyeballed, contrast.** Every text/background pair in this system was checked by computing WCAG relative luminance from the actual oklch values (not judged by eye) — the same discipline the previous GOV.UK system used to catch its amber/red tag-text failures. `--ink-3` and `--fog-3` (the "faint" tertiary text tokens, used on segment IDs and audit micro-labels) were each adjusted once during this build after measuring below 4.5:1 on their real background (`--ink-3` darkened 60%→55% lightness; `--fog-3` lightened 55%→60%) — both now clear AA for the small text they're actually used on.

## Typography

Three faces, each with exactly one job:

- **STIX Two Text** (display) — the wordmark, the evidence panel's score hero numeral, panel `<h2>` headings. A serif with real optical-size personality; chosen deliberately over a technical/grotesk display face to also signal "research instrument," supporting the project's publication ambition alongside its analyst-tool role.
- **Archivo** (body/UI) — every label, body line, button, table cell, the 30-row priority queue. A grotesk with more character than Inter/Roboto defaults, still dense-legible at UI sizes.
- **IBM Plex Mono** (data) — segment IDs, model versions, coordinates, tabular numerals. Unchanged in *role* from the previous system's mono stack, upgraded in *face*.

### Hierarchy
- **Score hero** (STIX Two Text, 600 weight, 3.75rem/60px, tabular numerals, animated count-up on selection): the single largest, most important number in the product.
- **Panel heading** (STIX Two Text, 600 weight, 1.375rem): "Priority queue", "Why this road?"
- **Micro-label** (Archivo, 700 weight, 0.6875rem, uppercase, 0.06em tracking): "SCORE BREAKDOWN", "OBSERVED HISTORY" — a typographic device marking structure, not a voice change (copy stays sentence case everywhere else — see `opendesign/design-systems/fieldscope/brand/voice-and-tone.md`).
- **Body/data** (Archivo, 400–700 weight, 0.8125–0.9375rem): evidence grid values, queue row labels.
- **Tag text** (Archivo, 700 weight, 0.75rem, sentence case): unchanged convention from the previous system.

## Layout

Same three-column "Operate" grid as before on desktop (`1fr / 360px / 400px`: map / priority queue / evidence panel), collapsing to a single stacked column below 1100px — the information architecture is untouched, per the user's explicit "keep in mind the flow" instruction. What changes is the **paper/chassis seam**: the map column (including its bezel, not just the MapLibre canvas) and the header are chassis-dark; the priority queue, evidence panel, and disclaimer bar are paper-light. The seam is a 1px hairline plus a subtle inset shadow where the two meet — drawn once, deliberately, not implied by an accidental colour change.

## Elevation & Depth

- **Paper surfaces** get a soft, short shadow (`--shadow-paper-1/2`) — paper lifted slightly off a desk, not a deep drop shadow.
- **Chassis surfaces** get a 1px hairline highlight (`--shadow-chassis-glow`) simulating backlit instrument glass — never a drop shadow (nothing sits "on" the chassis that would cast one).
- No shadow ever crosses the paper/chassis seam — each material's depth cue stays native to it.

## Shapes

**Machined, not bubbly, not square.** Radius scale is 2/4/6px (`--radius-sm/md/lg`) — a deliberate middle point between the previous system's hard 0px squares (read as a government form) and a rounded-SaaS-card default (would read as consumer app). Never exceeds 6px anywhere.

## Motion

Real, purposeful animation was an explicit, named requirement of this redesign. Every pattern below responds to a state change the previous system left silent — full rationale for each in `opendesign/design-systems/fieldscope/brand/style-notes.md`:

1. **Panel entrance** — queue rows and evidence content stagger in on load/borough-switch (not a block pop-in).
2. **Selection handshake** — selecting a road synchronises three things: the map's `easeTo` (already implemented in `MapView.tsx`), the queue row's wash + accent-edge sweep, and the evidence score hero counting up to its new value rather than snapping.
3. **Hover lift** — interactive rows/buttons rise 1px with a stronger shadow, never a scale-transform.
4. **Score-breakdown bars** — fill 0→value on first paint of a new segment's evidence, staggered per bar.
5. **Focus/press** stay instant (100ms) — only secondary feedback gets slower motion.

All motion respects `prefers-reduced-motion` (reduced to an instant/near-instant equivalent, never silently removed where it carries state information — e.g. loading feedback).

## Components

### Header (chassis)
- Solid dark chassis bar, STIX Two Text wordmark in `--fog-1`, a teal-bordered "Beta" phase tag (still an honest, non-decorative signal — this is a prototype), a chassis-toned borough `<select>`, and the audit-trail readout in `--fog-2/3` on `--fog-1` values.
- **Accent:** a soft teal gradient underline beneath the header (fading right), replacing the previous flat 10px brand-blue bar — same "identity stripe" role, restyled to fit the accent-as-glow language of the chassis zone.

### Map (chassis)
- Basemap, 3D buildings, hover card, Street View chip, legend, and loading overlay are functionally unchanged from the previous system (see `docs/decision_log.md`'s map-UX entries for that history) — restyled to chassis tokens: dark translucent chrome (`oklch(24% 0.019 250 / 0.92)`) instead of opaque white cards, backdrop-blur for a glass-instrument read.
- **Hover card** is the one exception — it stays a paper-toned card (`--paper-1`) even though it floats over the chassis map, because it is momentarily "borrowed" reading surface (text you're meant to read, not instrument chrome) — a deliberate, narrow exception to the placement rule, not drift.

### Priority queue rows (paper)
- Flat rows, hairline divider, hover = paper-1 background + 2px rightward nudge, selected = accent-wash background + an animated accent-edge sweep (the one legitimate accent-coloured edge in the system, reserved for genuine selection state).

### Evidence panel score hero (paper)
- A recessed `--paper-2` well holding the STIX Two Text numeral (animates via count-up on selection) and the RAG tag — no border, no accent stripe, matching the previous system's correct restraint on decorative borders.

### Score-breakdown bars, warning text, evidence grid
- Same components, same data, restyled to the new radius/colour tokens; the warning-text icon moves from a black-ringed circle to a solid `risk-mid-on-paper` disc for better legibility against the new palette, same meaning.

## Do's and Don'ts

### Do:
- **Do** use the RAG scale for every risk-level indicator — one colour language, defined once, unchanged in meaning from the previous system.
- **Do** keep the accent teal out of risk data — it is identity/interaction only.
- **Do** place a component's surface (paper vs chassis) by its role, not per-component taste.
- **Do** keep every animation purposeful and tied to a real state change — see the Motion section.
- **Do** respect `prefers-reduced-motion` on every animation added.

### Don't:
- **Don't** mix paper and chassis tokens within one component ("a dark card inside a paper panel").
- **Don't** use the accent teal to mark a "notable" or "flagged" road — that reads as an undefined fourth risk band.
- **Don't** let any radius exceed 6px, or let a card default to a fully rounded, bubbly shape.
- **Don't** let "no recent modelled estimate" (risk-none) be visually or verbally conflated with "low risk" (risk-low) — unchanged, non-negotiable boundary from the previous system.
- **Don't** add motion that isn't tied to a specific, real state change — decoration-only animation is exactly the "AI slop" the user asked this redesign to avoid.
