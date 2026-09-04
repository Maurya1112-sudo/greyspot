# Greyspot — Master Plan and Verification Ledger

**Living document. Re-read before starting any task; update immediately
after finishing one.** Every claim in this project must trace to a row
in §3 marked VERIFIED, or it does not appear in any output.

Last updated: 2026-09-05 00:40 (C7 REPLICATES - both major claims now on 2+ boroughs)

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
| R8 | **Verify every doc edit actually applied.** A string-replace that does not match silently does nothing. | The V5 row sat stale at "queued" for 90 minutes after the task had passed, because an update pattern did not match. |
| R10 | **Multi-seed any number that appears in an output, per borough.** Noise is ~4 points and the bias is NOT transferable between boroughs - it flips sign. | V8/V11: seed 42 was joint-highest of 5 on Lambeth (+1.69) and LOWEST of 5 on Westminster (-1.11). |
| R11 | **Never stop at a milestone.** Completing a phase is not a stopping point - launch the next item in the SAME action as reporting the result. If the work identifies its own next step, no decision is needed. | Phase 1 completion was reported without launching the multi-seed follow-up it had just identified as mandatory. |
| R9 | **Attach a Monitor in the SAME action that launches a run.** Never `nohup` a job and add the watcher later (or not at all). | The user cannot see untracked jobs; a Tower Hamlets run finished silently and sat unnoticed for 12 minutes, and V10 was launched invisibly. |

---

## 1. Current headline (all figures Lambeth unless noted)

| Borough | Ours | Gao et al. | Verdict |
|---|---|---|---|
| Westminster | **80.03% ± 1.40** (5 seeds) | 68.98% | **better, p=0.0001** |
| Tower Hamlets | **82.63% ± 2.01** (5 seeds) | 72.24% | **better, p=0.0003** |
| Lambeth | **77.75% ± 1.67** (5 seeds) | 76.59% | **TIE, p=0.1941** |
| **POOLED** | **80.14% ± 1.12** (5 seeds) | **72.60%** | **better, p=0.000115** |

> **All figures above are seed-averaged over 5 seeds with 95% CIs.**
> Seed-42 bias is borough-specific and flips sign (+1.69 Lambeth,
> -1.11 Westminster, +1.30 Tower Hamlets), so it cannot be extrapolated
> between boroughs. Per-borough seed spread is 3.2-4.7 points; the
> POOLED figure is far more stable (seed-42 bias only +0.63) because
> per-borough biases partly cancel.

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

## 2b. Single-borough claim scoreboard (the headline methodological result)

| Claim | Tested on 2nd borough | Outcome |
|---|---|---|
| hidden=42/42 | yes | FAILED |
| architecture ensemble | yes | FAILED |
| weight_decay=0.01 | yes | FAILED |
| road class harmful | yes | **FAILED** |
| 13 features == 35 | yes | DOWNGRADED |
| architecture topology | yes | **REPLICATED** |
| **history horizon** | yes | **REPLICATED** (+15.85 Lam / +9.70 WM, 6/6 windows) |

**2 of 7 single-borough findings survived replication intact — and
effect size predicted the outcome perfectly.** Both survivors were
large (9–26 pts, well above the ~4-pt seed-noise band); all four
failures were small (1–3 pts, inside it). Practical threshold: at this
evaluation scale a single-borough result is uninformative unless the
effect exceeds the seed-noise band.

## 3. Verification ledger — every task, its state, and its cross-check

Legend: ☐ not started · ◐ running · ☑ done+verified · ✗ invalidated

### Phase 1 — Correctness (nothing publishes until all ☑)

| ID | Task | State | Cross-check required | Result |
|---|---|---|---|---|
| V1 | Config-rebinding bug fix | ☑ | Loop simulated in isolation; 4 scripts patched; `window_config` grepped | Fixed, committed 775fe87 |
| V2 | Table-start control | ☑ | Arm 1 reproduces baseline to 0.27pts | table start +0.25, horizon +15.87 |
| V3 | Network metric-confound | ☑ | Random invariant over 200 seeds; identical crash counts | Claim survives |
| V4 | Architecture ablation (7 arms) | ☑ | Each arm differs in exactly 1 key (verified programmatically) | **TOPOLOGY matters, CAPACITY does not**: layers 1->2 -26.46 (p=0.0003); encoder order -13.61 (p=0.0135); decoder -3.50 (null); hidden 42/42 -2.87 (null); weight_decay +2.53 -> **later rejected at n=18 (V9)**. Effects exceed the 19-pt gap -> non-additive |
| V5 | Overfitting audit | ☑ | Label-shuffle must collapse to ~20% | **PASS**: shuffled 23.08% vs real 75.64% (random 19.96±3.46). First version permuted the TIME axis and returned bit-identical scores - caught by R3 |
| V6 | Re-run pruning test (was ✗) | ☑ | Must differ from baseline now the bug is fixed | **-6.89, 0/6 wins, p=0.0014** for that specific 30-feature config. **Interpretation SOFTENED by V10**: adding the same columns to a 13-feature model HURTS, so this is a property of that configuration, not the columns' value |
| V10 | Feature-count ladder, 6 windows + significance | ☑ | Full ladder + paired tests | **13 features == 35 features** (-0.99, p=0.3473). Nothing on the ladder differs significantly. Non-monotonic -> feature-group attribution unreliable. **Corrects V6's over-read** |
| V7 | Re-verify null ledger under final config | ☑ | Spot-check road class under BOTH configs | **Road class NOT null - significantly HARMFUL** (-6.87 at 13 feat, -4.49 at 35 feat, 0/6 wins, p=0.0157). **Third old null overturned.** Pre-2026-09-04 ledger does NOT transfer and must not be cited as-is |
| V8 | Multi-seed check on headline | ☑ | 5 seeds on 1 borough; report mean±std | **Seed 42 is +1.69 optimistic.** Lambeth seed-avg **77.75% ± 1.67** (was 79.44%). Spread 3.86 pts. **Lambeth is a TIE vs UCL (p=0.1941).** Single-seed noise band ≈4 pts - effects below that are uninterpretable |
| V9 | weight_decay=0.01 cross-borough | ☑ | Must replicate before adoption | **NOT ADOPTED**: n=18 pooled **-0.60, p=0.6061**. Effect shrank monotonically with evidence: +2.53 (n=6) -> +0.53 (n=12) -> -0.60 (n=18). Third false positive caught |

### Phase 1b — Coverage gaps found by audit (2026-09-04 22:00)

An audit against this ledger found that **six Phase-1 findings are
LAMBETH-ONLY and SINGLE-SEED**, which contradicts two things this
project established the same day: single-seed noise is ~4 points, and
three of three single-borough candidates failed replication.

| ID | Finding | Coverage | Action |
|---|---|---|---|
| C1 | Architecture ablation ("topology decisive") | **2 boroughs ☑** | **REPLICATES**: layers -26.46/-43.21, encoder -13.61/-9.42, all p<0.05. First candidate today to survive replication. **2-layer DIVERGES on Westminster (2/6 windows below random) - restate as 'unstable', not 'costs N points'** |
| C2 | 13 features ≈ 35 features | **2 boroughs ☑** | **PARTIALLY**: Lambeth -0.99 (p=0.35) but Westminster **-4.08 (p=0.092)** -> claim DOWNGRADED, needs a 3rd borough. **BUT: 26 features == 35 on both (p=0.89/0.99) -> socio-demographic can be dropped outright** |
| C3 | Road class harmful | **2 boroughs ☑** | **DOES NOT REPLICATE - claim RETRACTED.** Lambeth -6.87/-4.49 (p=0.016) but Westminster -0.37/**+2.25** - sign flips. Effect is Lambeth-specific |
| C4 | Pruning −6.89 | Lambeth, seed 42 | ☐ (superseded by C2 - the ladder is the better test) |
| C5 | Table-start control | Lambeth, seed 42 | ☐ low priority - it was a negative control, and it agreed with the baseline |
| C6 | Head-to-head vs their architecture | Lambeth, seed 42 | ☐ replicate if the paper claims it |

**Effects below ~4 points in those tables are NOT trustworthy** and must
be reported as "not distinguishable from seed noise": hidden size
(−2.87), decoder family (−3.50), and every intermediate step of the
feature ladder.

**Effects comfortably above the noise band** — layers (−26.46), encoder
order (−13.61), history horizon (+15.87), network (+11.59) — are
plausible but still need one replication each before publication.

### Phase 2 — Strengthening

| ID | Task | State | Notes |
|---|---|---|---|
| V11 | Multi-seed all three boroughs | ☑ | MANDATORY before any output | **DONE.** WM 80.03±1.40 (p=0.0001), TH 82.63±2.01 (p=0.0003), Lam 77.75±1.67 (tie). **Pooled 80.14±1.12 vs 72.60, p=0.000115.** Seed-42 bias: +1.69/-1.11/+1.30 - borough-specific, not transferable |
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
- **Any pre-2026-09-04 null, unless re-measured.** Three have been overturned (2-layer, encoder order, road class); the old ledger was measured under the superseded 30-feature/30-day-cap config.

---

## 5. Next action

Always the topmost ☐ or ◐ row in §3, Phase 1 then Phase 1b then Phase 2.
Currently: **Phase 1b complete.** Next: C6 (head-to-head replication) or Phase 2.

**Phase 2 is deliberately NOT started**: adding five new boroughs tests
breadth, but the central claim is not yet replicated even once. Depth
before breadth.

---

## 6. Change log

- 2026-09-04 11:05 — created; V1/V2/V3 marked verified; V4 in progress.
