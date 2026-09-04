# Style notes — Fieldscope / Greyspot

Component-level conventions that don't fit in raw tokens. Read this before
touching any component; it explains *why*, not just *what*.

## The paper/chassis seam

The single most important layout decision in this system: the product is one
physical object with two zones, and the seam between them is drawn once,
deliberately, not implied by scattered borders.

- **Chassis (dark, cool graphite):** `.app-header`, `.map-column` (the whole
  map viewport including its bezel/frame, not just the MapLibre canvas).
- **Paper (light, warm):** the priority queue, the evidence panel, the
  disclaimer bar under the map.
- The seam itself gets a visible, intentional edge — a 1px `--chassis-line`
  hairline plus a very slight inset shadow where chassis meets paper, so it
  reads as two materials meeting, not a colour that happens to change.
- **Never** put a chassis-toned card inside a paper panel or vice versa as a
  one-off "dark mode section" — that breaks the physical-object metaphor the
  whole system depends on.

## Cards & surfaces

- Paper cards sit on `--paper-1` (lifted) against the `--paper-0` page, with
  `--shadow-paper-1` — a soft, short shadow (paper lifted slightly off a
  desk), not a deep drop shadow.
- Chassis surfaces don't get shadows (nothing sits "on" a chassis panel
  that would cast one) — they get `--shadow-chassis-glow`, a 1px hairline
  highlight simulating backlit instrument glass.
- No card ever gets a coloured left-border accent strip as decoration. The
  one legitimate accent bar in the system is the priority queue's *selected
  row* (a genuine state, not decoration) — same restraint the previous
  GOV.UK system already correctly enforced; carried forward on purpose.

## Tags (risk bands)

- Same three-band language, restyled: `--radius-sm` corners (not GOV.UK's
  hard square, not a pill), sentence case labels unchanged ("Lower",
  "Medium", "Higher"), `-wash` background + `-on-paper` text on paper
  surfaces. On the map itself (chassis surface, no tag chips, just line
  colour) use the graphic `--risk-*` variants directly.
- Never let a risk tag pick up the accent teal — see SKILL.md non-negotiable
  #2.

## Motion patterns (the explicit "animation" ask)

Every motion below is *purposeful* — communicates a state change that was
previously silent — not decoration for its own sake. All respect
`prefers-reduced-motion` (fall back to an instant or near-instant state
change, never removed entirely, since some of these motions carry meaning
a static UI would lose, e.g. the loading state).

1. **Panel entrance.** On first load and on borough switch, the queue rows
   and evidence panel content stagger in (`--duration-fast` per row,
   `--ease-decelerate`, ~24ms stagger) rather than popping in as a block —
   communicates "this is freshly fetched data for a new borough," not a
   static page.
2. **Selection handshake.** Selecting a road (from either the map or the
   queue) triggers three synchronised things, not one: the map's `easeTo`
   (already implemented, `--duration-map`), the queue row's background
   wash fading in (`--duration-base`, `--ease-standard`) plus a brief
   accent-coloured left-edge sweep (`--duration-fast`, `--ease-emphasized`
   — a faster exponential decelerate than the standard curve, earned by
   being a discrete, infrequent event rather than a hover repeated
   hundreds of times; still no overshoot — a bounce/spring easing was
   tried and dropped, since real objects decelerate smoothly and an
   elastic snap reads as dated regardless of frequency), and the
   evidence panel's score hero counting up from its previous value to the
   new one over `--duration-slow` rather than snapping — the score is the
   product's single most important number, and a snap discards the "how
   different is this from what I was looking at" signal a count-up gives
   for free.
3. **Hover lift.** Interactive rows/buttons rise 1px with a stronger shadow
   over `--duration-fast`, `--ease-standard` — small, consistent, never a
   scale-transform (scaling text at UI-density sizes blurs subpixel
   rendering).
4. **Score-breakdown bars.** Fill from 0 → value on first paint of a new
   segment's evidence (`--duration-slow`, `--ease-decelerate`, staggered
   ~40ms per bar) — same "freshly computed, not static" signal as panel
   entrance, applied to the one place in the evidence panel that's
   inherently a magnitude comparison.
5. **Focus and press states** stay `--duration-instant` — a control must
   feel like it responded the moment it's touched; only *secondary*
   feedback (the lift, the sweep) gets slower motion.

## Typography rules specific to this system

- `--font-display` (STIX Two Text) is reserved for: the wordmark, the evidence
  panel's score hero numeral, and panel `<h2>` headings. It never appears in
  body copy, table cells, or button labels — those stay `--font-body`
  (Archivo) so density-heavy areas (the 30-row queue) don't fight a display
  face's larger optical size.
- `--font-mono` (IBM Plex Mono) is reserved for genuinely tabular/identifier
  content: segment IDs, model version strings, coordinates, and any numeral
  column that benefits from fixed-width alignment (already the existing
  product's convention — kept unchanged).
- Micro-labels ("SCORE BREAKDOWN", "OBSERVED HISTORY") use `--font-body` at
  `--text-2xs`, uppercase, `--tracking-label` letter-spacing — a typographic
  device signalling "this is structure, not sentence," not a voice choice
  (see `voice-and-tone.md`).
