# Greyspot — Master Plan and Verification Ledger

**Living document. Re-read before starting any task; update immediately
after finishing one.** Every claim in this project must trace to a row
in §3 marked VERIFIED, or it does not appear in any output.

Last updated: 2026-09-05 06:55 (S4 done - holiday effect real; crash-density claim revived at n=31)

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
| R14 | **Validate downloaded data against its peers before caching it.** A partial network response looks identical to a small borough. | Wandsworth's POI download timed out and cached 1,637 POIs vs Camden's 11,367 for a LARGER borough - a silent corruption that would have been reused forever. |
| R13 | **Never claim an INTERACTION between two factors from fewer than 3 boroughs, whatever its size.** Effect size predicts replication for main effects only. | The substitution effect was +25.04 vs +9.70 - far outside the ~4-pt noise band - and still reversed sign between Westminster and Tower Hamlets. |
| R15 | **When you kill a run, kill its Monitor in the same action.** A watcher polling a log that will never complete spins until the session ends. | S5's first (truncated) run was killed at 03:05; its monitor kept polling for 4h43m and was only noticed because the user saw the task chip. |
| R16 | **Verify a kill actually killed it.** `pkill -f` does NOT reliably match full command lines under Git Bash on Windows. On 2026-09-05 a `pkill -f run_s1_retry_after` reported success, left the shell loop alive, and it started a SECOND GPU job two seconds before I launched another - breaking R5 without any error appearing. After any kill, list the processes again and confirm. |
| R12 | **Any number quoted in a doc needs a script in the repo.** An inline diagnostic that prints to console is not a reproducible artefact. | The metric sanity check (a load-bearing claim) existed only as console output until the 2026-09-05 audit. |
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
| **our arch beats theirs** | yes (3 boroughs) | **REPLICATED** (+9.08 to +24.42, p<0.01 both tested) |
| substitution effect (interaction) | yes | **FAILED** - sign reverses (WM gap collapses, TH gap widens) |

**3 of 9 single-borough findings survived replication intact.**

Effect size predicts replication for MAIN EFFECTS: all three survivors
were large (9–26 pts, above the ~4-pt seed-noise band); the four
small failures (1–3 pts) were inside it.

**But it does NOT predict replication for INTERACTIONS.** The
substitution effect was +25.04 vs +9.70 — far outside the noise band —
and still reversed sign between boroughs. **Revised rule: main effects
replicate if they exceed the noise band; interactions between factors
should not be claimed from fewer than three boroughs, whatever their
size.**

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
| C6 | Head-to-head vs their architecture | **2 boroughs ☑** | **REPLICATES**: our arch wins on both, p=0.0007/0.0059. **NEW: substitution effect** - gap collapses +24.42 -> +9.08 once both get long history (their arch gains +25.04 vs our +9.70). Substitution rests on Westminster alone -> needs Tower Hamlets |

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
| S1 | 5 more boroughs | ⚠️ PARTIAL | **RESUMED 2026-09-05 14:34; Wandsworth FAILED AGAIN** - Overpass dropped the `amenity` category mid-download ("Response ended prematurely"). This time the new density guard REFUSED to cache it rather than silently producing a third bad result. Brent + City of London still running. Original note: **HALTED 2026-09-05 03:05 - Overpass API outage.** VALID: Camden 84.66%, K&C 74.54%. INVALID (quarantined): Wandsworth (partial POI), Brent (zero POI). NOT RUN: City of London. Resume when Overpass recovers. Tests whether the final model generalises beyond the 3 benchmark boroughs, which is the first question a reviewer asks of a 3-region study. NOTE: these are NOT in Gao et al.'s study, so they cannot extend the benchmark comparison - they test the MODEL, not the comparison |
| S2 | Empirical Bayes baseline | ☑ | **Major finding + control.** Superseded by S2b below for all reported figures |
| S2b | S2 redone: seed-averaged + PAIRED | ☑ | **R10 compliance for the paper's central claim.** 5 seeds x 6 windows x 3 boroughs, paired t-test + Wilcoxon (`scripts/run_s2_paired_comparison.py`). Pooled n=18: EB **-3.71 (p=0.0146)**, raw count **-3.80 (p=0.0289)**, matched-1825d **-0.85 (p=0.6398, 9/18 wins)**. **SIGN CORRECTED**: the matched-horizon gap was recorded as +0.85 for the GNN; it is -0.85 against. A tie either way. Seed-averaging made the uncapped result SIGNIFICANT where single-seed could produce no p-value at all. Cross-check: pooled GNN 80.13 reproduces V11's 80.14 from a different code path |
| S3 | Dense eval of FINAL config | ☐ | ~6h. Previous dense run evaluated a superseded config |
| S5 | **Extend history ceiling to 9yr** | ☑ | **NULL: -0.72, p=0.7438, variance UP (6.12%->9.83%).** Baseline gains +2.95 from the same extra history; the GNN gains nothing. **The GNN cannot exploit information a trivial sort uses directly.** 5th instance of sparse-column addition degrading this model |
| S6 | Sparse-robust (rank) scaling | ☑ | **REJECTED**: best-ever on Lambeth (85.56%), worst-ever on Westminster (39.82%) - a 46-pt reversal. Refutes the sparse-scaling explanation for the five feature-addition nulls |
| S4 | Holiday-window analysis | ☑ | 3 of 5 worst dense windows were Dec 8–Jan 19 |

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

### Live state — 2026-09-05 15:05 (post-restart)

**Running now** (both visible as Monitors):
- S1 resume: City of London training (Camden ☑ 84.66, K&C ☑ 74.54 already done)
- Chained retry: Wandsworth + Brent, starts automatically when S1 exits
  (waits rather than running concurrently — R5)

**Done since the restart:**
- R14 codified as a real guard (`greyspot.ingest.poi`) and wired into 50
  scripts. One true positive (Wandsworth, 3rd Overpass failure) and one
  false positive (Brent — my floor was calibrated on inner-London boroughs
  only) on day one. Both documented.
- **S2b**: the trivial-baseline comparison redone seed-averaged and paired.
  Sign error corrected; conclusion unchanged and now significance-tested.
  Propagated to README, final_model.md and the preprint.
- Walk-forward evaluation made resumable (per-window checkpointing).

**Queue, highest value first:**
1. **C2 third borough** (Tower Hamlets) — the "13 ≈ 35 features" claim is
   DOWNGRADED pending it. GPU.
2. **C6 substitution effect on Tower Hamlets** — currently rests on
   Westminster alone, and it is in the preprint. GPU.
3. **S5 deep-history null, second borough** — a single-borough null is
   exactly what this project has repeatedly seen fail to replicate. GPU.
4. **S3 dense eval** — resume from the 21 saved windows via
   `--seed-from=reports/westminster/s3_dense_partial_recovered.csv`.
   Lowest value: it narrows a CI on an established number and tests no
   claim. Do it last, or not at all.

---

## 6. Change log

- **2026-09-05** R14 codified in code, not just in this document
  (`greyspot.ingest.poi`): a POI download that loses a category, returns
  nothing, or falls below a calibrated density floor now raises
  `PoiDownloadError` instead of being cached. Floor derived from five
  hand-verified boroughs (151.7-357.5 adjacencies/km2) against the known
  truncated Wandsworth download (23.6); guard wired into 50 run scripts.
  It fired correctly on its first live run within five minutes.
- **2026-09-05** S2b: the trivial-baseline comparison redone seed-averaged
  and paired. Conclusion holds; one sign error corrected. See
  `docs/decision_log.md`.
- **2026-09-05** Walk-forward evaluation made resumable
  (`src/greyspot/eval/checkpoint.py`, wired into the dense-eval script).
  Per-window results are appended and fsync'd as they complete; a restart
  skips windows already done, gated on a config fingerprint that REFUSES
  to resume across differing configurations. Prompted by losing a run to
  an interruption for the second time. Testing found two real bugs in the
  hard-shutdown path (zero-byte file raised; a torn row's NaN fingerprint
  tripped the stale-config guard). 220 tests pass. See `docs/decision_log.md`.

- 2026-09-04 11:05 — created; V1/V2/V3 marked verified; V4 in progress.
