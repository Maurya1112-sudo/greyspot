# Voice & tone — Fieldscope / Greyspot

Carried over unchanged from the product's existing, already-good copy discipline
(`apps/web/src/components/*.tsx`) — this redesign is visual and interactive, not
a rewrite of what the product says. Documented here so future UI work doesn't
drift from it.

## What stays exactly as it is

- **Sentence case everywhere**, never Title Case or ALL CAPS, except the
  intentional micro-label treatment (section headers like "SCORE BREAKDOWN")
  which is a typographic device, not a voice choice — see `style-notes.md`.
- **Precise, hedged, evidence-first phrasing.** "No recent modelled estimate,"
  not "No data." "No conformal interval available for this row — confidence
  shown as neutral, not computed," not "N/A." The product never overstates
  certainty it doesn't have — this is a load-bearing property of a risk tool,
  not a style preference.
- **The non-negotiable boundary**: "no recent modelled estimate" (grey) is
  never described or coloured in a way that could be read as "low risk"
  (green). Any new copy must preserve this distinction explicitly.
- **First-person-plural avoided; second-person avoided too.** Copy describes
  the road/segment/model, not "you" or "we." ("Select a road on the map or
  from the priority queue to see its evidence," not "You can select a road…")
- **Numbers presented with their unit and basis stated once**, not repeated
  redundantly ("Motor vehicles/day (AADF)" as a label, then a bare number as
  the value).

## What this redesign is free to change

- Micro-copy for new interactive states this redesign adds (e.g. a segment
  hover affordance, a transition label) should match the existing register:
  plain, declarative, no exclamation points, no marketing verbs ("Unlock",
  "Discover"). "Selected" not "You've selected this road!"
- Loading/empty states may become more specific where the old copy was
  generic ("Loading…" → "Loading Camden's road network…") as long as the
  specificity is real (the borough name is already in state) and not
  invented flavour text.
