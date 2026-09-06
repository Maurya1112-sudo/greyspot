# Greyspot — Master Plan and Verification Ledger

**Living document. Re-read before starting any task; update immediately
after finishing one.** Every claim in this project must trace to a row
in §3 marked VERIFIED, or it does not appear in any output.

Last updated: 2026-09-06 13:25 (S5 null WITHDRAWN; 4 of 11 claims replicate; Fig 1 + SOP done)

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
| R13 | **Never claim an INTERACTION between two factors from fewer than 3 boroughs, whatever its size.** (Effect size does not rescue main effects either — see §2b: it predicts non-replication only.) | The substitution effect was +25.04 vs +9.70 - far outside the ~4-pt noise band - and still reversed sign between Westminster and Tower Hamlets. |
| R15 | **When you kill a run, kill its Monitor in the same action.** A watcher polling a log that will never complete spins until the session ends. | S5's first (truncated) run was killed at 03:05; its monitor kept polling for 4h43m and was only noticed because the user saw the task chip. |
| R16 | **Verify a kill actually killed it.** `pkill -f` does NOT reliably match full command lines under Git Bash on Windows. On 2026-09-05 a `pkill -f run_s1_retry_after` reported success, left the shell loop alive, and it started a SECOND GPU job two seconds before I launched another - breaking R5 without any error appearing. After any kill, list the processes again and confirm. |
| R12 | **Any number quoted in a doc needs a script in the repo.** An inline diagnostic that prints to console is not a reproducible artefact. | The metric sanity check (a load-bearing claim) existed only as console output until the 2026-09-05 audit. |
| R11 | **Never stop at a milestone.** Completing a phase is not a stopping point - launch the next item in the SAME action as reporting the result. If the work identifies its own next step, no decision is needed. | Phase 1 completion was reported without launching the multi-seed follow-up it had just identified as mandatory. |
| R17 | **Write the negative control before trusting a check.** A verifier that cannot fail is worse than none: it converts an unchecked area into one believed checked. | `verify_cross_document.py` reported 14/14 OK while blind - its section-scoped exemption let one correction note excuse every claim in that section, including the exact stale claim that had survived two days. Only re-injecting that claim revealed it. |
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
| 13 features == 35 | yes (3 boroughs) | **FAILED** - sign flips: -0.99 Lam, -4.08 WM, **+0.77 TH** |
| message-passing depth (1→2 layers) | yes | **REPLICATED** |
| encoder ordering (GAT→GRU vs GRU→GAT) | yes | **REPLICATED** |
| **history horizon** | yes | **REPLICATED** (+15.85 Lam / +9.70 WM, 6/6 windows) |
| **our arch beats theirs** | yes (3 boroughs) | **REPLICATED** (+9.08 to +24.42, p<0.01 both tested) |
| rank-transform scaling | yes | **FAILED** - +5.80 (best ever) became −39.7 (worst ever) |
| substitution effect (interaction) | yes (3 boroughs) | **FAILED** - sign reverses (WM gap collapses −15.33, TH gap widens +5.93) |

**4 of 11 single-borough findings survived replication intact; 7 failed.**

Depth and encoder ordering are counted as separate findings: they are
separate experiments varying one factor each (R1). Earlier revisions of
this document grouped them as one "architecture topology" row and reported
3 of 10, which disagreed with
`scripts/check_effect_size_heuristic.py`. All documents now use 11.

**Effect size predicts NON-replication only** (corrected 2026-09-05 after
checking it mechanically — `scripts/check_effect_size_heuristic.py`):

| Borough-1 effect | Replicated | Failed |
|---|---|---|
| Below the ~4-point seed-noise band | 0 | 4 |
| Above it | 4 | 2 |

Every sub-noise effect failed, so a small single-borough result can be
discarded without spending a replication run on it. But **two supra-noise
MAIN effects failed anyway** — road class (−6.87, sign reversed to +2.25)
and rank-transform scaling (+5.80 → −39.7). The earlier text here claimed
"all three survivors were large … the four small failures were inside the
band", which its own table contradicts: road class is listed as FAILED and
is 6.87 points.

So the relation is **necessary, not sufficient**: large effects still
require replication. Interactions are weaker still — the substitution
effect was +25.04 vs +9.70, far outside the band, and reversed sign (R13).

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
| C2 | 13 features ≈ 35 features | **3 boroughs ☑** | **FAILS - claim RETRACTED** (settled 2026-09-05, `scripts/run_c2_three_borough_analysis.py`). Sign flips: Lambeth -0.99 (p=0.35), Westminster -4.08 (p=0.092), **Tower Hamlets +0.77 (p=0.77)**. Pooled -1.44 (p=0.2293) averages opposite-signed effects AND sits inside the ~4-pt noise band. Criteria were fixed in the script before the third borough ran. **The socio-demographic sub-claim (26 features == 35 on both) is separate and still stands.** |
| C3 | Road class harmful | **2 boroughs ☑** | **DOES NOT REPLICATE - claim RETRACTED.** Lambeth -6.87/-4.49 (p=0.016) but Westminster -0.37/**+2.25** - sign flips. Effect is Lambeth-specific |
| C4 | Pruning −6.89 | Lambeth, seed 42 | **☒ CLOSED, not run** (2026-09-06). Superseded by C2, which tests the same question across a full feature ladder on three boroughs instead of one pruning configuration on one. Running C4 would add a fourth single-borough single-seed datapoint to a question C2 has already answered (the effect flips sign), which is the category of experiment this project has learned least from. |
| C5 | Table-start control | Lambeth, seed 42 | **☒ CLOSED, not run** (2026-09-06). It is a NEGATIVE control that already agreed with the baseline (table start worth +0.25 against the horizon's +15.87). Replicating a negative control on a second borough tests whether a confound is absent twice; the horizon claim it protects has itself replicated. Reopen only if the horizon claim is challenged. |
| C6 | Head-to-head vs their architecture | **3 boroughs ☑** | **ARCHITECTURE CLAIM REPLICATES**: our arch wins on all three at long history (+18.53 Lam / +9.08 WM / +22.28 TH). **SUBSTITUTION CLAIM FAILS** (settled 2026-09-05, `scripts/run_c6_substitution_analysis.py`): the gap collapses on Westminster (−15.33) and barely moves on Lambeth (−0.48) but **WIDENS on Tower Hamlets (+5.93)** - sign reverses, so it is borough-specific. Pooled −3.29 (p=0.2510) averages opposite-signed effects and must not be quoted. NOTE: the earlier 'needs Tower Hamlets' note was stale - the TH head-to-head already existed, so this needed no GPU run |

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
| S1 | 5 more boroughs | **3 of 5 ☑** | **Camden 84.66, K&C 74.54, Brent 77.64** (all seed 42, single-seed - NOT publishable as point estimates per R10, but adequate to answer 'does the model work outside the 3 benchmark boroughs'). Across all six boroughs measured the range is **74.54-84.66**, so it does generalise in the weak sense. **Brent's per-window spread is the widest seen (±11.92 sample SD, one window at 57.69%)** - it is the least dense borough tested and the only outer-London one, consistent with the measured crash-volume dependence of AccHR@20. **Wandsworth** still blocked: Overpass dropped its `amenity` category on 3 attempts. **City of London** killed after 33 min of Overpass retries without reaching training; it is 2.9 km2 and the least informative of the five. |
| S2 | Empirical Bayes baseline | ☑ | **Major finding + control.** Superseded by S2b below for all reported figures |
| S2b | S2 redone: seed-averaged + PAIRED | ☑ | **R10 compliance for the paper's central claim.** 5 seeds x 6 windows x 3 boroughs, paired t-test + Wilcoxon (`scripts/run_s2_paired_comparison.py`). Pooled n=18: EB **-3.71 (p=0.0146)**, raw count **-3.80 (p=0.0289)**, matched-1825d **-0.85 (p=0.6398, 9/18 wins)**. **SIGN CORRECTED**: the matched-horizon gap was recorded as +0.85 for the GNN; it is -0.85 against. A tie either way. Seed-averaging made the uncapped result SIGNIFICANT where single-seed could produce no p-value at all. Cross-check: pooled GNN 80.13 reproduces V11's 80.14 from a different code path |
| S3 | Dense eval of FINAL config | ☐ **DEPRIORITISED** | ~2.5h remaining (21 of 38 windows are saved and it resumes via `--seed-from`). It narrows the CI on a number already established and tests no claim. **It has also been overtaken**: the metric-granularity finding (2026-09-06) shows AccHR@20's step size on these windows is 0.0102-0.0909, so extra windows buy less precision than the arithmetic suggests. Run only if a reviewer asks for a tighter interval. |
| S5 | **Extend history ceiling to 9yr** | **2 boroughs ☑** | **NULL DOES NOT REPLICATE** (2026-09-05, `scripts/run_s5_two_borough_analysis.py`). Lambeth −0.72 (p=0.7438) but **Westminster +2.98 (p=0.0230, 5/6 windows)** - sign flips, so 'the GNN cannot exploit deeper history' is NOT supportable and has been corrected in the preprint abstract. **Both effects are inside the ~4-pt noise band**, so neither is individually interpretable. What DOES hold on both: the trivial sort gains more from the same extra history than the GNN (+2.38 vs −0.72 Lam; +4.79 vs +2.98 WM). The sort's pooled +2.95 gain is significant (p=0.0409, 12/18 windows). |
| S6 | Sparse-robust (rank) scaling | ☑ | **REJECTED**: best-ever on Lambeth (85.56%), worst-ever on Westminster (39.82%) - a 46-pt reversal. Refutes the sparse-scaling explanation for the five feature-addition nulls |
| S4 | Holiday-window analysis | ☑ | 3 of 5 worst dense windows were Dec 8–Jan 19 |

### Phase 3 — Output

| ID | Task | State | Notes |
|---|---|---|---|
| P1 | arXiv preprint | ◐ **substantially drafted** | `paper/DRAFT_preprint.md`: abstract, §1-6, Figure 1 (horizon curve), all 16 claims machine-verified against source. **Methods section added 2026-09-06** (§2: network, target, features, model, evaluation, metric, statistics). Remaining: a short related-work paragraph. Not blocked on Phase 1 - Phase 1 is complete (20 rows ☑). |
| P2 | Public GitHub repo | ☑ **ready** (2026-09-06) | README rewritten for public use: setup, a data-acquisition table with URLs and target paths, the eight GPU-free analysis scripts that regenerate each paper claim, and the Overpass failure mode with its diagnostic. **Blocker found and cleared**: `.git` was 1.1 GB from unreachable objects (a 969 MB archive staged before `.gitignore` covered it), which would have been rejected by GitHub's 100 MB file limit; now 1.35 MiB with all 82 commits intact. |
| P3 | SOP paragraph | ☑ **done** (2026-09-06) | `docs/sop_paragraph.md`: two lengths, a table mapping every claim to the script that regenerates it, and an explicit list of stronger-sounding lines left out because they do not survive the obvious follow-up. Writing it surfaced the 10-vs-11 finding-count disagreement between documents, which no single-document check could have caught. |

---

## 4. What the paper can and cannot claim

**CAN** (once Phase 1 completes):
- Benchmark comparison with paired significance testing and cross-borough replication
- ~20-entry null ledger — unusually complete negative evidence
- Reproducibility finding: their published lr=0.01 diverges on independent data; their predecessor code's lr=1e-5 scores 24.74%; the true optimum (5e-4) appears in neither source
- Methodological cautionary results: a config bug that silently disabled feature subsetting after window 1; a single-seed baseline that nearly triggered a false retraction

**CANNOT**:
- **"Effect size predicts replication."** It predicts NON-replication only:
  below the ~4-point seed-noise band 0 of 4 replicated, above it just 4 of 6.
  Two supra-noise MAIN effects failed — road class (−6.87, sign reversed)
  and rank-transform scaling (+5.80 → −39.7). Corrected 2026-09-05 after
  checking it mechanically (`scripts/check_effect_size_heuristic.py`); the
  abstract, README and final_model.md had all asserted the stronger,
  false version. A large single-borough effect still requires replication.
- **"The model cannot exploit history beyond 5 years."** WITHDRAWN
  2026-09-05: Lambeth −0.72 (p=0.74) but Westminster **+2.98 (p=0.023,
  5/6 windows)** — the sign flips. Both sit inside the noise band, so
  neither borough settles it. The defensible version is comparative: the
  trivial sort gains MORE from the same extra history than the model does,
  on both boroughs measured.
- **"The architectural advantage is substitutable for history."**
  WITHDRAWN 2026-09-05: the gap collapses on Westminster (−15.33) but
  WIDENS on Tower Hamlets (+5.93). The separate claim that our
  architecture beats theirs is unaffected and holds on all three.
- "Beats them on every borough" — Lambeth is a tie
- "Our architecture is better than theirs" — only that these choices matter *in this pipeline*; their published model on their data scores 76.59%
- Anything from an invalidated run (§3 rows marked ✗)
- **Any pre-2026-09-04 null, unless re-measured.** Three have been overturned (2-layer, encoder order, road class); the old ledger was measured under the superseded 30-feature/30-day-cap config.

---

## 5. Next action

Always the topmost ☐ or ◐ row in §3, Phase 1 then Phase 1b then Phase 2.

### Live state — 2026-09-06 12:30

**Running now:** Wandsworth, with the new tiled POI fallback (Monitor attached).

**Settled since yesterday — three claims withdrawn, none by choice:**

| Claim | Was | Now |
|---|---|---|
| 13 features ≈ 35 | downgraded | **FAILS** — sign flips (+0.77 on TH) |
| Architecture × history substitution | needed a 3rd borough | **FAILS** — sign reverses; needed NO GPU, the data was already on disk |
| Model can't use >5yr history | null on Lambeth | **WITHDRAWN** — Westminster +2.98 (p=0.023) |
| Effect size predicts replication | asserted both ways | **one-directional only** — 0/4 below band replicated, 4/6 above |

Scoreboard: **4 of 11 replicated, 7 failed.** (Depth and encoder order are
counted separately: they are separate experiments varying one factor each, per R1.) Four of the
seven failed by SIGN REVERSAL rather than shrinkage toward zero — that is
the characteristic failure mode at this evaluation scale and is a more
useful finding than "small effects are noisy".

**Infrastructure added:** per-window checkpointing (runs survive shutdown);
POI download guards with a calibrated density floor plus tiled fallback;
`verify_preprint_claims.py` (16 mechanical checks, all passing).

**Queue, highest value first:**
1. **Multi-seed S5 on both boroughs.** Its ±3-point effects are inside the
   noise band at a single seed, so neither borough's figure is
   interpretable. This is the only way to settle whether the model can use
   deep history at all. GPU, ~2h.
2. **Wandsworth** — running; completes the 5-borough generalisation.
3. **S3 dense eval** — resume via
   `--seed-from=reports/westminster/s3_dense_partial_recovered.csv`.
   Still lowest value: narrows a CI on an established number, tests no
   claim.

**Not worth doing:** more single-seed single-borough experiments. Today
produced four withdrawals and every one came from adding a borough or a
seed to something already "established". The binding constraint on this
project is evidence per claim, not number of claims.

---

## 6. Change log

- **2026-09-06** Tiled Overpass fallback: once retries are exhausted a
  category is re-fetched as a 3×3 grid of smaller bboxes and de-duplicated
  on OSM identity. Wandsworth's `amenity` query had failed four times over
  two days — always that one category, never the others, while Overpass
  reported free slots — which is a response too large to transfer, not bad
  luck. A tiling with any failed tile is refused rather than returned
  partial, and the duplicates removed matter: left in, they would inflate
  POI density and defeat the density guard.
- **2026-09-06** S5 deep-history null **WITHDRAWN**: Westminster gains
  +2.98 (p=0.0230, 5/6 windows) against Lambeth's −0.72 (p=0.7438). The
  preprint abstract had asserted the model "cannot exploit" history beyond
  five years on one borough's evidence. What survives is comparative and
  holds on both: the trivial sort gains more from the same extra history.
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
