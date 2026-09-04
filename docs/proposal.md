# Greyspot: Government Edition — Working Proposal

*Condensed from the project dossier (`Greyspot_Government_Edition_Master_Dossier.pdf`)
for supervisor discussion. Source blueprint holds the full 33-page detail;
this is the version meant to be read in five minutes.*

## Title

**Greyspot: An Uncertainty-Aware, Network-Connected Model and Decision-Support
Platform for Road-Risk Prioritisation in London**

## One-sentence definition

Greyspot estimates relative road-safety risk from historical collisions and
road-network structure, quantifies its own uncertainty, explains the evidence
behind a priority ranking, and helps analysts compare candidate locations —
without claiming to predict individual crashes or establish causality.

## Research question

Does connected-road-network modelling, with calibrated uncertainty, improve
reliable prioritisation on unseen places and later periods compared with
strong non-graph baselines?

## Hypotheses (falsifiable)

- **H1**: A GAT+temporal model ranks elevated-risk segments better than a
  strong XGBoost baseline on spatially/temporally held-out data.
- **H2**: Conformal prediction intervals attain their target empirical
  coverage on held-out data.
- **H3**: Model performance and calibration differ meaningfully across IMD
  deprivation deciles — visible, not hidden.
- **H4**: A graph model that does *not* beat XGBoost is still a valid,
  reportable result if the held-out test is clean.

## Primary geography and data

City of Westminster. STATS19 collisions (DfT, 2021–2025 confirmed available
as per-year CSVs — see [research_notes.md](research_notes.md)), road network
via OSMnx (OS Open Roads as a planned upgrade once an OS Data Hub account is
set up), IMD 2019 for equity slicing.

## Scope tiers (per dossier Section 18)

| Tier | Contains |
|---|---|
| **Minimum viable** | Westminster network + collisions; historical-rate + XGBoost baselines; spatial/temporal held-out evaluation; basic map |
| **Strong submission** | + GAT/temporal model; conformal uncertainty (MAPIE); priority score; evidence panel; IMD equity dashboard |
| **Showcase** | + budget-constrained portfolio optimiser; scenario explorer; controlled evidence-grounded AI copilot |

**This session's build corresponds to the Minimum Viable tier**, using
Python only (no FastAPI/PostGIS/React yet) — see `docs/decision_log.md`.

## Non-negotiable boundary

Greyspot is an investigation and prioritisation aid. It must not claim that a
specific collision will happen, that a location is "safe" or "unsafe" in an
absolute sense, or that a scenario will prevent a specific number of deaths
without a defensible causal design.

## Early evidence (Minimum Viable tier, real data)

Trained on real Westminster STATS19 (2021–2023), tested on 2024, with all
three of the dossier's required generalisation regimes: XGBoost beats the
historical-rate baseline on PR-AUC under **temporal** hold-out (0.33 vs
0.23), **spatial** hold-out — unseen road segments (0.32 vs 0.21), and the
combined **spatiotemporal** hold-out — unseen segments in an unseen year
(0.34 vs 0.26). This doesn't yet test H1 (that requires the GAT model), but
it establishes that even a non-graph learner transfers beyond memorised
roads, which is the right foundation before adding graph structure. Full
numbers: `reports/baseline_vs_xgboost_results.csv`; methodology notes and
caveats (small-N precision@10 noise, LSOA vintage mismatch) in
`docs/decision_log.md`.

## H1/H2 — evidence from a full diagnose-fix-validate-replicate cycle
(and one important self-correction)

**A note before the results**: an earlier version of this section reported
H1 as strongly supported (GAT beating XGBoost 0.424 vs 0.332). That was
found to be invalid — the "temporal" GAT evaluation had a genuine
data-leakage bug (trained directly against the exact label it was later
scored against). It was caught via a "this looks too good to be true"
instinct on a follow-up experiment, diagnosed, and fixed with a proper
walk-forward training scheme. **Full account: `docs/decision_log.md`,
2026-08-31 "CORRECTION" entry.** What follows is the corrected, valid
picture.

**H1** (GAT ranks better than XGBoost): **not supported as a general
finding.** The naive GAT+GRU (4 heads, no residual) was consistently
beaten by its own "remove the graph" ablation — a clean negative result.
Diagnosing over-smoothing led to a fix (1 head + residual connection) that
is **real but modest**: on Westminster, the fixed model clearly beats both
its ablations (temporal PR-AUC 0.331 vs 0.307 vs 0.249) but only ties
XGBoost (0.332) and trails it on spatiotemporal (0.337 vs 0.347). On
**Lambeth, the fix does not replicate even as an ablation-level effect**
(full model and "no graph" are statistically indistinguishable: 0.2909 vs
0.2909), and XGBoost leads on both splits (0.311, 0.284). The honest
conclusion: the graph-attention fix produces a genuine, diagnosed
improvement over its own ablations on one borough, but there is no current
evidence it reliably beats a strong non-graph baseline, or generalises
across boroughs. Full tables: `docs/results.md`.

**H2** (conformal intervals hit their target coverage): **supported for
XGBoost on both boroughs** (90.9%/91.2% against a 90% target). **Partially
supported for the GAT**: close to target on Westminster (88.6%) but a real
under-coverage on Lambeth (83.4%) — plausibly because the corrected
walk-forward method trains on a single transition, giving the model
markedly less supervision than the (invalid) original approach. This
weaker calibration is itself new information the correction surfaced, not
something the original bug allowed to be seen.

**Bonus finding beyond H1/H2**: adding real DfT AADF traffic-exposure data
— free, previously thought possibly infeasible (dossier Section 5.2) —
improved XGBoost's PR-AUC on every held-out split (e.g. temporal 0.326 →
0.332, Precision@25 0.72 → 0.80).

**Scaling proof**: the pipeline was refactored to be borough-parameterised
and run end-to-end on a second borough (Lambeth) with zero ingestion-code
changes — see `docs/scaling_to_london.md` for the concrete staged plan to
expand to all of London.

## Risk register (top items)

| Risk | Fallback |
|---|---|
| Network join (collision-to-road snapping) too noisy | Inspect junction cases; narrow to a smaller sub-area |
| Graph model adds no measurable value over XGBoost | Report the negative result — still a valid dissertation finding |
| STATS19 2024+ specification transition breaks year-over-year comparability | Document the discontinuity explicitly; consider treating pre/post as separate cohorts |
| OS Open Roads access delayed by account setup | Continue on OSMnx; swap later (isolated change to `ingest/network.py`) |

## Immediate next steps (post-kickoff)

1. Get the live 6COSC023W module handbook/rubric from Blackboard and cross-check
   against this scope.
2. Confirm supervisor fit and ethics classification (public/non-personal data
   only at this stage — should be low-risk).
3. Extend the Minimum Viable pipeline (already running against real data) with
   the GAT+temporal model and MAPIE conformal layer.
