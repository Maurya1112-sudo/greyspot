# Plan: from current results to an arXiv-ready paper

**Written 2026-09-04**, immediately after the head-to-head experiment
overturned this session's central claim. Purpose: get every claim
verified before anything is published, because a preprint is permanent
and will be read by people who know the reference paper.

---

## 0. Why this plan exists — the error that must not recur

Twice in one session, a large effect was attributed to an interesting
variable when **two or more variables had changed at once**:

1. **"Long crash history is worth +15.85 points."** FALSE. When the
   feature table was extended to 2021 to support a 365-day window, that
   simultaneously fixed truncated rolling features in the earliest
   training instances. Controlled test: horizon alone is worth
   **+0.51 points** (5 of 6 windows bit-identical).

2. **"Architecture is worth +19 points."** NOT YET ESTABLISHED — and
   currently has the same defect. "Their architecture" differs from
   this project's in FIVE ways at once: encoder order, layer count,
   hidden size, decoder family, weight decay. The 19-point gap cannot
   be assigned to any one of them without ablation.

**Rule for everything below: one variable per experiment, or it does
not go in the paper.**

---

## 1. Phase 1 — VERIFY (nothing is published until this completes)

### 1.1 Table-start control `[running]`
Isolates feature-table start (2022 vs 2021) with identical SHORT
features. Expected ~+15 points. `run_tablestart_control.py`.

### 1.2 Architecture ablation `[REQUIRED — not yet built]`
Decompose the 19-point gap one variable at a time, starting from this
project's configuration and changing exactly one thing per arm:
- encoder order: `spatial_first` -> `temporal_first`
- layers: 1 -> 2
- hidden: 16/32 -> 42/42
- decoder: ZIP -> Zero-Inflated Tweedie
- weight decay: 0.0 -> 0.01

Each at its own best learning rate where the change destabilises
training (the encoder-order arm demonstrably needs this). Without this,
**no architecture claim can appear in the paper.**

### 1.3 Re-verify the OS Open Roads finding `[REQUIRED]`
The network result (+11.59, p=0.0010) is currently the project's
strongest claim, but it was measured under the OLD table-start regime
with truncated early features. Two questions:
- Does it survive with complete early features?
- Did the OSMnx and OS Open Roads arms differ in anything besides the
  network (segment counts differ, which changes the size of the top-20%
  bucket and hence the task's difficulty)?

### 1.4 Overfitting audit `[REQUIRED]`
35 features against 6-11 training INSTANCES is a legitimate concern,
even though the node-level loss sees ~11.6k segments per instance.
Tests:
- Train vs held-out loss curves per window (is held-out loss rising?)
- Feature-count ablation (does 35 beat 20 beat 10?)
- Does performance degrade as features are added, holding all else
  fixed? (the earlier "full Table 7.2 parity" run WAS the worst variant,
  which hints at overfitting and was never followed up)

### 1.5 Re-verify every surviving claim under the corrected setup
The null ledger (~20 entries) was largely measured with truncated early
features. Nulls are probably robust (a null under a worse configuration
usually stays null), but any claim quoted in the paper must be re-run
under the final configuration.

---

## 2. Phase 2 — STRENGTHEN

### 2.1 Generalisation to more boroughs
Five further boroughs are configured (Camden, Kensington and Chelsea,
City of London, Brent, Wandsworth). They are NOT in Gao et al.'s study,
so they cannot extend the benchmark comparison - but they test whether
findings generalise beyond three hand-picked areas, which is what a
reviewer will actually ask.

### 2.2 Dense evaluation of the FINAL configuration
The existing dense run (n=31, 78.84%) evaluated a configuration whose
causal story has since changed. Re-run once the configuration is final.

### 2.3 Empirical Bayes baseline
The Highway Safety Manual's standard method. Serves two purposes:
positions the work relative to traditional road safety, and preempts
"you have rediscovered EB".

---

## 3. Phase 3 — PUBLISH

### 3.1 What the paper can honestly claim

Pending Phase 1, the defensible contributions are:

1. **A rigorous, reproducible benchmark comparison** against a
   published UCL model on independent London data, with paired
   significance testing and cross-borough replication.
2. **A large null ledger** (~20 interventions) - unusually complete
   negative evidence for what does NOT help at this sparsity.
3. **A documented reproducibility finding**: the reference paper's
   published hyperparameters (lr=0.01) diverge on independent data, and
   its predecessor code's lr=1e-5 underperforms badly (24.74%); the
   architecture's actual optimum here (5e-4) appears in neither source.
4. **A methodological cautionary result**: a plausible, well-motivated
   +15.85-point "finding" that dissolved to +0.51 under control. This
   is publishable in its own right as a warning about confounded
   attribution in walk-forward pipelines.

### 3.2 What the paper must NOT claim
- That long crash history matters (it does not, +0.51).
- That our architecture beats theirs generally (only that these choices
  matter within this pipeline; their published model on their data
  scores 76.59%).
- Any per-borough win on Lambeth (a tie at both n=6 and n=31).
- That the protocols are equivalent (they are not).

### 3.3 Artifacts
- arXiv preprint (workshop-length is appropriate).
- Public GitHub repo with the full decision log - the log IS a
  contribution, not supporting material.

---

## 4. Explicit success criteria

The paper is ready when: every claim in it maps to a single-variable
experiment, replicated on >=3 boroughs, with a p-value; and every claim
NOT meeting that bar has been removed.
