# Requirements, ethics and governance (draft)

*Addresses L6.04 (independent requirements development) and the BCS Code of
Conduct / research-ethics expectations flagged in Westminster's programme
specification (BCS accreditation; QAA Computing benchmark). This is a
living document — requirements will be refined as the product layer is
built.*

## 1. Functional requirements (current + planned)

| ID | Requirement | Status |
|---|---|---|
| FR1 | Ingest STATS19 collision/casualty/vehicle data for a chosen local authority | Done |
| FR2 | Build a road-network graph for that authority and snap collisions to segments | Done |
| FR3 | Produce a segment-year feature table with leakage-safe (lagged) historical features | Done |
| FR4 | Train and compare a historical-rate baseline, XGBoost, and a GAT+GRU graph-temporal model | Done |
| FR5 | Evaluate all models on temporal, spatial and spatiotemporal held-out splits | Done |
| FR6 | Ablate the GAT (remove graph structure; remove temporal encoder) | Done |
| FR7 | Quantify prediction uncertainty with calibrated conformal intervals | Done |
| FR8 | Incorporate equity context (IMD) as a lagged, clearly-labelled feature, never a hidden one | Done |
| FR9 | Transparent Safety Priority Score with a visible weight breakdown | Done — `product/priority_score.py` + the evidence panel's "Score breakdown" section, `apps/web/` |
| FR10 | Interactive map + evidence panel (analyst view; public view not started) | Done for the analyst view (`apps/web/`, GOV.UK-styled, WCAG 2.2 AA target) — see PRODUCT.md for why the public/analyst dual view was descoped to analyst-only for this pass |
| FR11 | Scenario Explorer (intervention library, before/after evaluation) | Not started |
| FR12 | Budget-constrained portfolio optimiser | Not started |
| FR13 | Controlled, evidence-grounded AI Investigation Copilot | Not started |
| FR14 | Investigation-brief / report generator | Not started |

## 2. Non-functional requirements

| ID | Requirement | Rationale |
|---|---|---|
| NFR1 | Every model result must be reproducible from a fixed seed and a versioned dataset snapshot | Dissertation defensibility; dossier Section 17 |
| NFR2 | No random train/test splitting on spatial or temporal data | Prevents leakage that would invalidate results (`eval/splits.py`) |
| NFR3 | All enrichment features must be lagged (prior-period only) before reaching a model | Prevents the model from "predicting" a count using data derived from that same count (`features/build_features.py`) |
| NFR4 | Every data-quality issue found must be logged, not silently patched | `docs/decision_log.md` is the running record |
| NFR5 | The system must never output an absolute "safe" / "unsafe" label | Non-negotiable boundary, see below |
| NFR6 | (Future) the web layer must validate input, use parameterised queries, and rate-limit | Standard security baseline once FastAPI exists |

## 3. Non-negotiable safe-use boundary

Carried directly from the project dossier and treated as a hard
requirement, not an ethics footnote:

> Greyspot is an investigation and prioritisation aid. It must not claim
> that a specific collision will happen, that a neighbourhood or group
> causes collisions, or that a scenario will prevent an exact number of
> deaths without a defensible causal design.

Concretely, this constrains the current codebase: model outputs are
labelled "relative risk research signal" everywhere (README, module
docstrings), never "risk of X happening." The eventual product layer must
carry the same wording into the UI.

## 4. Ethics self-assessment

*A self-assessment against common UK university research-ethics
categories, to be transferred into whatever specific ethics form/process
your actual course requires — this is not a substitute for that form.*

| Question | Answer | Notes |
|---|---|---|
| Does the project involve human participants? | Not yet | A future usability study (dossier Section 14) would require a fuller ethics application before recruitment begins |
| Does the project use personal or sensitive data? | No | STATS19, IMD 2019 and OS/OSM road data are public, non-personal, published open datasets. STATS19 records the *event*, not identifiable individuals (no names, addresses, or personal identifiers in the files used) |
| Does the project involve vulnerable groups? | No | No data collection from or about identifiable individuals |
| Could the project cause reputational or psychological harm to a place or community? | Low risk, actively mitigated | "Spatial stigma" (a road/area being labelled "dangerous") is an identified risk in the dossier's equity chapter; mitigated by relative-risk framing and avoiding sensational language in all outputs (map, docs, reports) |
| Does the project involve deception, covert observation, or withholding information from participants? | No | N/A — no participants yet |
| Is there a risk of automation bias (a human uncritically trusting the model)? | Yes, actively mitigated | Every output is framed as a research signal requiring human interpretation; the "non-negotiable boundary" above is a design constraint, not just policy text |
| Overall classification (provisional) | **Low risk** | To be confirmed against the actual Westminster ethics process before any human-subject work begins |

## 5. BCS Code of Conduct alignment

Since this course is BCS-accredited, the BCS Code's themes are treated as
design inputs rather than a final report checklox:

- **Public interest**: the project explicitly targets road-safety
  investigation, a genuine public-safety use case, grounded in real DfT/TfL
  policy documents (see `docs/research_notes.md`), not an invented problem.
- **Professional competence and integrity**: negative/non-dominant results
  (the GAT vs XGBoost comparison) are reported honestly rather than
  reframed as a win — see `docs/results.md`.
- **Duty to relevant authority**: none yet — no client/employer relationship
  exists for this solo academic project.
- **Duty to the profession**: reproducibility (fixed seeds, versioned data,
  a running decision log) is treated as a professional obligation, not
  optional polish.

## 6. Sustainability and EDI

- **Sustainability**: models were deliberately kept small enough to train
  in seconds-to-minutes even on a laptop CPU (see `docs/results.md`); GPU
  training (CUDA, an RTX 4060 Laptop GPU, auto-detected — see
  `models/gat_temporal.py::_resolve_device`) is used when available purely
  for iteration speed across multi-borough experiment batches, not because
  the model needs it — CPU training remains a working fallback. Should
  still be quantified (measured wall-clock/energy proxy) in a later phase
  per the dossier's "GreenModel" framing.
- **EDI**: the vulnerable-road-user features (pedestrian/cyclist casualty
  counts) and planned equity dashboard (IMD-decile-stratified evaluation)
  are direct EDI-relevant design choices, not an afterthought. **The
  accessibility checklist in the original dossier (Section 15 — colour
  never the only encoding, keyboard navigation, screen-reader labels) is
  now partially implemented**, not just planned: the frontend
  (`apps/web/`) targets WCAG 2.2 AA and the GOV.UK Design System
  (confirmed with the user 2026-08-31 — see `PRODUCT.md` and `DESIGN.md`
  at the project root), concretely meaning every risk-level indicator
  pairs colour with a text label (never colour alone — the RAG tags read
  "Lower"/"Medium"/"Higher", not just green/amber/red), every interactive
  element carries GOV.UK's visible yellow focus outline, and every
  text/background colour pair was checked against a computed WCAG
  relative-luminance contrast ratio rather than eyeballed (two real
  failures were found and fixed this way — see `docs/decision_log.md`,
  2026-08-31 redesign entry). Not yet done: a real screen-reader pass
  (only structural/ARIA-role verification so far) and an automated
  accessibility audit tool (axe-core or similar) — flagged as the
  concrete next step, not claimed as complete.

## 7. Data protection

No personal data is processed. STATS19's published open-data extracts
already exclude direct personal identifiers. If a future phase links to
more granular data (e.g. casualty demographics beyond what's in the public
extract), this section must be revisited and a full data-protection
assessment carried out before that data is used.
