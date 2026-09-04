# Module context — 6COSC023W Computer Science Final Project

*Verified against University of Westminster's public Computer Science BSc
Honours Programme Specification (2024–25 version) on 2026-08-31. **The live
Blackboard module handbook, assessment brief and marking rubric for your
actual cohort override everything in this file** — programme specifications
are deliberately generic and course-specific handbooks are not publicly
published. Treat this as a grounding reference to check your handbook
against, not a replacement for it.*

## Confirmed facts

| Item | Value | Source |
|---|---|---|
| Module code | **6COSC023W** | Computer Science BSc Honours Programme Specification 2024–25 |
| Module title | Computer Science Final Project | ″ |
| Level | 6 (final year) | ″ |
| Credit value | **40 UK credits / 20 ECTS** | ″ |
| Status | Core (compulsory) | ″ |
| Assessment type | "Synoptic assessment" — draws together outcomes from across the degree, alongside the Level 5 Software Development Group Project | ″ |
| Accrediting body | British Computer Society (BCS) — course is CITP and partial CEng accredited | ″ |
| External reference points | QAA Subject Benchmark Statement for Computing; BCS accreditation guidelines; SEEC credit-level descriptors | ″ |
| Academic regulations | Handbook of Academic Regulations at westminster.ac.uk/academic-regulations (course-specific regulations may also apply) | ″ |

## Level 6 course learning outcomes (verbatim)

These are the *programme-level* outcomes a student is expected to
demonstrate by the end of Level 6 — the final project is where most of
these converge into one piece of work. Greyspot's mapping to each is in
the table below the list.

- **L6.01** — Identify and appraise the main threats to computer systems and networks security and integrity. *(KU)*
- **L6.02** — Appropriately analyse and design large scale data systems to serve the retrieval and/or decision-making needs of computer systems and their clients. *(PPP)*
- **L6.03** — Implement a comprehensive technical solution to an advanced problem using appropriate programming languages. *(PPP)*
- **L6.04** — Methodically and independently develop requirements to a solution for a large-scale software problem using appropriate languages and tools. *(PPP)*
- **L6.05** — Demonstrate complete handling of the full life-cycle of a computer science project underpinned by an entrepreneurial approach and a focus on the needs of real clients and the wider society. *(KTS)*
- **L6.06** — Following guidance, review literature in Computer Science and present in written and oral form own work and learning, critically comparing, contrasting and evaluating the findings. *(KTS)*
- **L6.07** — Apply appropriate research methodologies in carrying out independent research in computer science and produce a report demonstrating evidence of critical thinking. *(KTS)*

*(KU = Knowledge & Understanding, PPP = Professional/Personal Practice, KTS = Key Transferable Skills)*

## Learning-outcome traceability

| Outcome | How Greyspot addresses it | Evidence |
|---|---|---|
| L6.01 (security threats) | Threat model for the eventual API/web layer (input validation, injection, rate limiting); prompt-injection isolation for the AI Investigation Copilot | `docs/requirements_and_ethics.md`; pending: implemented threat model once the API exists |
| L6.02 (large-scale data systems for decision-making) | PostGIS-shaped data model (Section 8 of the dossier), segment-year feature table at Westminster scale (37,760 rows), decision-support outputs (priority score, portfolio) | `docs/methodology.md`; `data/processed/segment_year_table.parquet` |
| L6.03 (comprehensive technical solution) | Full ML ladder implemented and tested: historical-rate → XGBoost → GAT+GRU, with ablations and conformal uncertainty | `src/greyspot/`, `docs/results.md` |
| L6.04 (independent requirements development) | Requirements independently scoped from the government-edition dossier down to an executable Minimum-Viable tier; scope-control decisions logged as they were made | `docs/requirements_and_ethics.md`; `docs/decision_log.md` |
| L6.05 (full lifecycle, real client/society focus) | Explicit government-analyst user personas, TfL Vision Zero policy grounding, non-causal safe-use boundary as a *requirement* not an afterthought | `docs/requirements_and_ethics.md` |
| L6.06 (literature review) | Structured review of graph-neural crash prediction, conformal uncertainty on graphs, and road-network representation | `docs/literature_review.md` |
| L6.07 (research methodology, critical thinking) | Explicit hypotheses (H1/H2), leakage-safe temporal/spatial/spatiotemporal splits, honestly-reported non-dominant result for the GAT vs XGBoost comparison | `docs/methodology.md`, `docs/results.md` |

## Why this matters for marking

The earlier project-selection research for this degree (see the
`# 100-Mark Strategy...md` document in the parent folder) summarised
Westminster's Level 6 descriptor as reserving the top band for work showing
*"exceptional independent thought and reflection, creative analysis,
critical analysis of sources and methods, extensive research, and confident
communication."* The concrete implication for this project: a result that
doesn't clearly favour the novel model (as seen in the first GAT-vs-XGBoost
comparison) is not a weakness to hide — critically analysing *why* it
happened is exactly the kind of evidence this descriptor rewards. Every
decision in `docs/decision_log.md` is kept for this reason: it is the raw
material for the dissertation's methodology and discussion chapters, not
just a working log.
