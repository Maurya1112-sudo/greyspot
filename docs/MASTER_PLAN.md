# Greyspot — Master Plan and Verification Ledger

**Living document. Re-read before starting any task; update immediately
after finishing one.** Every claim in this project must trace to a row
in §3 marked VERIFIED, or it does not appear in any output.

Last updated: 2026-09-04 11:24 (after architecture-ablation arm 4)

---

## 0. Operating rules (violating these has cost this project hours)

| # | Rule | Evidence for it |
|---|---|---|
| R1 | **One variable per experiment.** If two things change, attribute nothing. | The "+15.85 history" claim was un-attributable for a day because table start co-varied. |
| R2 | **Never conclude from a single sample** — one window, one seed, one run. | Four wrong conclusions today, incl. a near-retraction of the network claim from a single-seed baseline (per-seed std 3.6 pts). |
| R3 | **Bit-identical results across "different" configs = BUG, not coincidence.** | Hit 3x: `per_physical_road` no-op, NaN tie-break, config-rebinding bug. |
| R4 | **Verify on real data before launching GPU work.** | 20-class POI matched 0/20,040 real POIs while all unit tests passed. |
| R5 | **One GPU job at a time.** | 4 CUDA OOM crashes. |
| R6 | **Diff mechanically; do not reason from plausibility.** | 5 consecutive wrong hypotheses today; the bug was found by simulating the loop in isolation. |
| R7 | **Commit after every verified result.** | No git until 2026-09-04; absence of diffs directly caused the longest debugging episode. |

---

## 1. Current headline (all figures Lambeth unless noted)

| Borough | Ours | Gao et al. | Verdict |
|---|---|---|---|
| Westminster | 78.92% | 68.98% | better, p=0.0212 |
| Tower Hamlets | 83.93% | 72.24% | better, p=0.0026 |
| Lambeth | 79.44% | 76.59% | tie, p=0.3059 |
| **Pooled (n=18)** | **80.76%** | **72.60%** | **better, p=0.000042** |

Caveats that must accompany any statement of this: one-sample test only
(their per-window results unpublished); protocols differ; the ~140x
zero-inflation discrepancy is unexplained; Lambeth is a tie at n=6 AND
at n=31.

---

## 2. What is being claimed, and on what evidence

| Claim | Effect | Control | Status |
|---|---|---|---|
| Real OS Open Roads network beats OSMnx | +11.59 pooled, p=0.0010 | Random baseline network-invariant (+0.28pp, 200 seeds); both nets capture the same 206 crashes | **VERIFIED** 2026-09-04 |
| Long crash-history horizon | +15.87 | Table-start control: table start worth only +0.25 | **VERIFIED** 2026-09-04 |
| Architecture choices matter (~+19 vs their config) | +18.5/+19.0 | Ablation decomposing 5 co-varying changes | **IN PROGRESS** |
| Model is not overfitting | — | Label-shuffle + feature ladder | **PENDING** |

---

## 3. Verification ledger — every task, its state, and its cross-check

Legend: ☐ not started · ◐ running · ☑ done+verified · ✗ invalidated

### Phase 1 — Correctness (nothing publishes until all ☑)

| ID | Task | State | Cross-check required | Result |
|---|---|---|---|---|
| V1 | Config-rebinding bug fix | ☑ | Loop simulated in isolation; 4 scripts patched; `window_config` grepped | Fixed, committed 775fe87 |
| V2 | Table-start control | ☑ | Arm 1 reproduces baseline to 0.27pts | table start +0.25, horizon +15.87 |
| V3 | Network metric-confound | ☑ | Random invariant over 200 seeds; identical crash counts | Claim survives |
| V4 | Architecture ablation (7 arms) | ◐ | Each arm differs in exactly 1 key (verified programmatically) | baseline 79.44; **layers 1->2 = 52.98 (-26.46)**; encoder-order @5e-4 = 65.83 (-13.61); @0.01 = 14.31 (NaN, excluded). Effects EXCEED the 19-pt total gap -> non-additive, interactions present |
| V5 | Overfitting audit | ☐ | Label-shuffle must collapse to ~20% | built, queued |
| V6 | Re-run pruning test (was ✗) | ☐ | Must differ from baseline now the bug is fixed | — |
| V7 | Re-verify null ledger under final config | ☐ | Spot-check 3 nulls, not all 20 | — |
| V8 | Multi-seed check on headline | ☐ | 5 seeds on 1 borough; report mean±std | — |

### Phase 2 — Strengthening

| ID | Task | State | Notes |
|---|---|---|---|
| S1 | 5 more boroughs | ☐ | ~5h. Tests generalisation beyond the 3 benchmark boroughs |
| S2 | Empirical Bayes baseline | ☐ | Preempts "you rediscovered EB"; connects to HSM literature |
| S3 | Dense eval of FINAL config | ☐ | ~6h. Previous dense run evaluated a superseded config |
| S4 | Holiday-window analysis | ☐ | 3 of 5 worst dense windows were Dec 8–Jan 19 |

### Phase 3 — Output

| ID | Task | State | Notes |
|---|---|---|---|
| P1 | arXiv preprint | ☐ | Blocked on Phase 1 |
| P2 | Public GitHub repo | ☐ | git initialised; needs README + reproduction instructions |
| P3 | SOP paragraph | ☐ | Must be defensible line-by-line |

---

## 4. What the paper can and cannot claim

**CAN** (once Phase 1 completes):
- Benchmark comparison with paired significance testing and cross-borough replication
- ~20-entry null ledger — unusually complete negative evidence
- Reproducibility finding: their published lr=0.01 diverges on independent data; their predecessor code's lr=1e-5 scores 24.74%; the true optimum (5e-4) appears in neither source
- Methodological cautionary results: a config bug that silently disabled feature subsetting after window 1; a single-seed baseline that nearly triggered a false retraction

**CANNOT**:
- "Beats them on every borough" — Lambeth is a tie
- "Our architecture is better than theirs" — only that these choices matter *in this pipeline*; their published model on their data scores 76.59%
- Anything from an invalidated run (§3 rows marked ✗)

---

## 5. Next action

Always the topmost ☐ or ◐ row in §3 Phase 1. Currently: **V4 (running)**,
then **V5**, then **V6**.

---

## 6. Change log

- 2026-09-04 11:05 — created; V1/V2/V3 marked verified; V4 in progress.
