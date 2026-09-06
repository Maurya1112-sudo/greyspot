# Greyspot — Project Dossier for SOP Writing

**Purpose of this file.** Source material for drafting statements of
purpose. It is deliberately over-complete: it contains more angles,
numbers and anecdotes than any single SOP should use, so that a draft can
be tailored to a specific programme by selecting from it rather than
inventing.

**Rule for anyone drafting from this file:** every number below traces to
a named script in the repository. Do not round, soften, or restate a
figure in a way that makes it sound better than it is. The strongest
version of this project's story is the accurate one, and an interviewer
who reads the repository will find the same numbers.

---

## 1. One-paragraph summary

An independent replication and critique of a 2024 *Accident Analysis &
Prevention* paper applying graph neural networks to road-level crash
prediction. Over three days of intensive work (127 commits), I rebuilt the
data pipeline from UK public sources, reproduced the model, and then
subjected it to the kind of scrutiny the original could not report:
replication on additional regions, multi-seed evaluation, and comparison
against a deliberately trivial baseline. Most of the reported design
decisions did not survive. Neither my model nor the reference architecture
beat a parameter-free baseline that ranks road segments by how often they
have already crashed. The output is a preprint, a reproducible repository,
and an unusually complete record of the project's own errors.

---

## 2. Factual profile

| | |
|---|---|
| **Duration** | 2026-09-04 to 2026-09-06 (intensive); project began 2026-08-31 |
| **Scale** | 127 commits, 120 scripts, 31 source modules, 237 tests, 252 result CSVs |
| **Documentation** | 19 markdown documents; a 7,494-line decision log |
| **Compute** | Single RTX 4060 Laptop (8 GB); ~600 model training runs |
| **Data** | STATS19 (DfT), OS Open Roads (Ordnance Survey), IMD 2019, DfT AADF, OpenStreetMap Overpass |
| **Scope** | 7 London boroughs; 2012–2024 crash records; ~11,600 road segments per borough |
| **Reference work** | Gao et al., *Accident Analysis & Prevention* 208:107801, 2024 (STZITD-GNN) |
| **Output** | arXiv preprint (IEEEtran, 2-column), public repository, reproducible verification suite |

**Technical stack.** Python, PyTorch, PyTorch Geometric, GeoPandas,
OSMnx, NetworkX, scipy.stats, pandas. Graph attention networks (GAT),
gated recurrent units (GRU), zero-inflated Poisson/Negative
Binomial/Tweedie decoders, split conformal prediction, line-graph
transforms, Empirical Bayes (Highway Safety Manual method).

---

## 3. What the project actually found

### 3.1 The headline benchmark result

| Borough | This work (5 seeds) | Gao et al. | Verdict |
|---|---|---|---|
| Westminster | 79.75% ± 0.89 | 68.98% | better, *p* < 0.0001 |
| Tower Hamlets | 82.75% ± 2.13 | 72.24% | better, *p* = 0.0004 |
| Lambeth | 77.75% ± 1.67 | 76.59% | tie, *p* = 0.1941 |
| **Pooled** | **80.08% ± 2.62** | 72.60% | better, *p* < 0.0001 |

**Do not lead an SOP with this.** It is true, but the same project shows a
trivial baseline beats it, and leading with the benchmark invites exactly
the question that undoes it. It is also not like-for-like: their protocol
is within-2019 with a 6:2:2 split, ours is multi-year expanding
walk-forward, and no paired test against them is possible because they
publish three point estimates rather than per-window results.

### 3.2 The replication result (the strongest methodological finding)

Of **eleven** design decisions that were statistically significant on one
borough, **four** survived replication on a second. Seven failed, and
**four of those reversed sign** rather than shrinking toward zero.

That last detail matters more than the count. A practitioner who assumes
effects merely attenuate across regions — and therefore treats a
single-region estimate as an upper bound — would have the *direction*
wrong, not just the magnitude.

**Effect size predicts non-replication, but not replication.** Every
effect below the measured ~4-point seed-noise band failed (0 of 4). But
two effects comfortably above it also failed: road-class features (−6.87,
*p* = 0.016) reversed to +2.25, and rank-transform scaling went from the
best result in the project (+5.80) to the worst (−39.7). So a small effect
is reliable evidence *against* replication, while a large one is only weak
evidence *for* it — necessary, not sufficient.

### 3.3 The trivial-baseline result (the most consequential finding)

Ranking road segments by their cumulative count of past crashes — no
parameters, no training, no model — performs as well as the graph network:

| Ranker | Mean AccHR@20 |
|---|---|
| Crash-count sort (~8 years) | **83.94%** |
| Empirical Bayes (Highway Safety Manual) | 83.84% |
| Crash-count sort, capped to 5 years | 80.99% |
| My GNN (5 seeds) | 80.08% |
| Reference architecture (5 seeds) | 66.57% |

At matched history depth the model and the sort are statistically
indistinguishable (−0.90, *p* = 0.61, 9 of 18 windows). Uncapped, both
trivial baselines beat it significantly.

**And it generalises beyond my implementation.** Run on my data at its own
tuned learning rate with the same long history, the *reference*
architecture loses to the sort on **18 of 18 held-out windows** (−17.37,
*p* < 10⁻⁶). Neither graph network in the study clears the trivial
baseline.

### 3.4 The horizon explanation (the most elegant result)

The obvious objection to §3.3 is that the literature *does* use historical
baselines — Gao et al. report a Historical Average at 44.96%. Reconciling
that with my sort's 83.94% needs exactly one variable: **their dataset
covers a single year**, so their baseline can look back at most one year.

Sweeping the same parameter-free ranker across lookback horizons:

| Lookback | 30d | 90d | 1yr | 2yr | 3yr | 5yr | 7yr | 9yr |
|---|---|---|---|---|---|---|---|---|
| AccHR@20 | 22.71% | 32.19% | 54.17% | 66.35% | 73.78% | 80.99% | 83.53% | 83.94% |

**A 61-point range from one ranker with no parameters**, varying only how
far back it looks. Every published figure in this line is matched by that
sort at a short horizon: their Historical Average at ~1 year, the
reference architecture at ~2, their model at ~3, mine at ~5.

The argument this supports is precise and defensible: the quantity of
published improvement over a historical baseline in this task is of the
same order as the improvement obtainable by *lengthening that baseline's
horizon by a year or two*, on data freely available to both. I found no
paper in this line that reports such a comparison.

### 3.5 Measurement findings

- **Training is not deterministic at a fixed seed.** Re-training one
  window at one seed three times gave 0.755, 0.832, 0.845 — a 9-point
  spread — while a control window on another borough was bit-identical
  across three repeats. Cause: scatter-based neighbourhood aggregation in
  the attention layer, with no fixed CUDA accumulation order.
- **The metric is a step function whose resolution is set by data
  sparsity.** AccHR@20 can only change by 1/(crashes that day)/(days in
  window) — 0.0102 to 0.0909 on these windows. So no figure on a sparse
  window should be quoted more precisely than its own step size.
- **The reference architecture fails by divergence, not degradation.**
  Three independent measurements found the same shape: depth-2 message
  passing diverges on 2 of 6 windows; the full architecture spreads 35.7
  points across seeds (one run at 42.58%, barely twice random); their
  encoder ordering costs ~6 points on four seeds and collapses to
  near-random on the fifth — on the *same seed* in both boroughs. A mean
  over seeds misdescribes all three; the failure rate (~1 in 5) is the
  useful quantity.

---

## 4. Errors I made, found, and corrected

**This is the most valuable section for an SOP.** It is what distinguishes
a student who ran experiments from one who did research. Use one or two,
not all.

### 4.1 A silent bug that invalidated three results

`config = dict(config)` inside a per-window loop rebound the outer loop
variable, so feature subsetting applied only to window 1. Every subsequent
window silently used the full feature set. This invalidated three results
and produced a **false retraction** of a correct finding. Found by
simulating the loop in isolation rather than reasoning about it — reading
the code had not revealed it across several attempts.

### 4.2 A control that tested nothing

A label-shuffle control (permuting crash targets across road segments,
which should collapse the model to random) returned *bit-identical* scores
to the real model. The permutation was applied to the 14-day time axis
instead of the 11,596-segment axis, so it permuted nothing meaningful. The
tell was the bit-identical result: two genuinely different configurations
producing identical output is a bug, not a coincidence.

### 4.3 A published uncertainty that could not be reproduced

The project's own headline read "80.14% ± 1.12". Regenerating every figure
through a single script showed the pooled sample SD is **2.60**, not 1.12.
I tested eight candidate definitions — SD, standard error, window-level SD
and SE, SD of borough means, mean of per-borough SDs, raw-window SD and SE
— and none produces 1.12. The uncertainty on the project's most prominent
number had been understated by more than half, and the published
confidence interval was inconsistent with the ± beside it.

### 4.4 A data corruption that looked like valid data

The OpenStreetMap Overpass API degraded mid-study, reporting free capacity
while truncating large transfers. One borough's point-of-interest download
lost an entire feature category on four separate attempts *while appearing
successful*. The cached file looked like a small borough's plausible data.
When eventually retrieved intact, the truncated file proved to contain
**24% of the real data**. I built a guard that validates category
completeness and density against verified downloads — and the guard's
first action was a **false positive**, rejecting a valid outer-London
borough because I had calibrated its threshold on inner-London ones only.
I resolved that by downloading twice and confirming byte-identical
results, which a truncated response cannot produce.

### 4.5 Three wrong explanations for one discrepancy

One evaluation window disagreed between runs. I asserted a mechanism
(metric tie-breaking) without measuring it. Then withdrew it on a single
counter-example — generalising from n=1 while enforcing a rule against
generalising from n=1. The reproducibility check then refuted my
replacement explanation too. It was finally settled by arithmetic: the
observed difference of 0.017857 equals exactly 1/4/14 — one crash crossing
the threshold on a 4-crash day in a 14-day window — matched to six decimal
places between two full-precision runs.

### 4.6 A verifier that could not fail

I wrote a checker to catch inconsistencies between the repository's
documents. It reported 14/14 passing. I then re-injected the exact stale
claim that had survived two days undetected — and it **still** reported
14/14. Its exemption logic scoped to whole document sections, so one
correction note was excusing every other claim in its section. A checker
that cannot fail is worse than none, because it converts an unchecked area
into one believed checked. This became a project rule: **write the
negative control before trusting the check.**

### 4.7 A one-gigabyte repository

Preparing for public release, `.git` was 1.1 GB while the largest tracked
file was 803 KB. A 969 MB road-network archive had been staged before
`.gitignore` covered it, then unstaged with `git reset` — which clears the
index but leaves the object in the store, and `gc`'s default two-week
prune window would not have collected it. This would have blocked
publication outright (GitHub rejects files over 100 MB). Reclaimed to 1.35
MB with all 127 commits intact, after verifying the source files still
existed and were genuinely ignored.

---

## 5. Engineering practices developed

These emerged from failures rather than from planning, which is worth
saying explicitly.

- **A 17-rule operating protocol**, each rule traceable to the incident
  that caused it. Examples: *one variable per experiment*; *never conclude
  from a single sample*; *bit-identical results across different configs
  is a bug, not a coincidence*; *diff mechanically, do not reason from
  plausibility*; *any number quoted in a document needs a script that
  regenerates it*; *write the negative control before trusting the check*.
- **Two automated verifiers.** One re-derives every headline number from
  its source file and compares it against the manuscript (17 checks); the
  other checks that the repository's README, model specification and paper
  do not contradict one another (14 checks) — a class of error
  single-document verification cannot detect, and which caught two real
  discrepancies.
- **Resumable long runs.** Multi-hour evaluations checkpoint per window
  and refuse to resume across differing configurations, gated by a
  configuration fingerprint. Building it surfaced two bugs in the guard
  itself, both only visible at its second real caller.
- **A decision log of 7,494 lines** recording every experiment including
  the failures, the withdrawals, and the corrections-of-corrections.

---

## 6. Angles for different programme types

Pick **one** framing and commit to it.

### For an ML / AI programme
Lead with the replication result and the trivial baseline. The story is:
*I set out to beat a published model, succeeded on the benchmark, and then
discovered the benchmark was the wrong question.* Emphasise multi-seed
evaluation, paired significance testing, and the finding that the
reference architecture loses to a parameter-free sort on every window.

### For a data science / statistics programme
Lead with the measurement work: the metric's step-function granularity,
non-determinism at fixed seed, the unreproducible uncertainty figure, and
the effect-size heuristic being one-directional. The story is about
knowing what a number can and cannot support.

### For a transport / urban analytics programme
Lead with the horizon curve and the Empirical Bayes connection. The story
is that the deep-learning literature encodes "past crashes predict future
crashes" in the *label* while safety engineering encodes it in a
*multi-year feature*, and the gap between those conventions is worth up to
61 points of accuracy.

### For a research-methods-heavy or PhD-track programme
Lead with §4 — the errors. Specifically the verifier that could not fail,
and the three wrong explanations for one discrepancy. The story is about
building the discipline to catch your own mistakes, and about the
difference between a plausible explanation and a measured one.

### For a software engineering / systems programme
Lead with the infrastructure: checkpointed resumable runs, the data-
integrity guards, the 1 GB repository, and the cross-document verifier.
The story is that research code is production code with worse incentives.

---

## 7. Lines that are TRUE and USABLE

Draw from these; do not invent variants.

- "Of eleven findings that were significant on one region, four survived
  replication on a second — and four of the seven failures reversed sign
  rather than shrinking."
- "A parameter-free baseline that ranks road segments by past crash count
  matched my model and beat the reference architecture on 18 of 18
  held-out windows."
- "The same ranker spans 23% to 84% accuracy depending only on how far
  back it looks, which is more than any architectural difference I
  measured."
- "I found that training was not deterministic at a fixed seed, and
  quantified what that means for how precisely any result can be quoted."
- "I wrote a verifier to catch inconsistencies in my own work, then proved
  it was blind by re-injecting an error it had missed."
- "The project's own published uncertainty figure could not be reproduced
  under any of eight candidate definitions; I corrected it."
- "I want to work on evaluation methodology, because this project
  convinced me that is where the errors live."

## 8. Lines that are NOT usable

- ~~"I beat a state-of-the-art model."~~ True on two of three boroughs,
  but the same work shows a trivial baseline beats mine. Saying it invites
  the question that undoes it.
- ~~"I showed GNNs don't work for crash prediction."~~ Not supported. One
  architecture family, one city, one evaluation protocol.
- ~~"I found errors in a published paper."~~ No. Their published learning
  rate diverges on *my* data, which is an observation about transfer, not
  an error on their part. The horizon comparison against their figures is
  suggestive, not controlled — two of those numbers are on their data.
- ~~Any single-seed number.~~ Seed spread is 3.2–4.7 points for my
  architecture and up to 35.7 for theirs, and the direction of the bias is
  region-specific.

---

## 9. Honest weaknesses (anticipate these)

An interviewer or admissions reader may raise these; the paper states them
rather than hiding them.

- **Scale.** Three boroughs for the benchmark comparison, seven for
  generalisation, one city, six held-out windows per borough.
- **No paired test against the reference work** is possible, because they
  publish point estimates rather than per-window results.
- **The horizon mapping is indicative, not controlled** — two of the
  figures plotted against my curve were measured on their data.
- **An unexplained discrepancy**: my target is 99.97% zero against their
  reported 95.72–96.71%, and I could not account for it after checking
  their crash-rate formula, segment consolidation and evaluation protocol.
- **The work is three days of intensive effort**, not a year-long study.
  Its depth is in verification rather than in breadth of experiments.
- **Not peer-reviewed.** A preprint, prepared to arXiv's format
  requirements, not a published paper.

---

## 10. What I would say I learned

Framed as claims that can be defended in an interview:

1. **A single-region, single-seed result is not evidence of much.** I
   started assuming effects would replicate and shrink; the data showed
   they frequently reverse. That changed how I design experiments.
2. **The baseline determines the finding.** The most consequential result
   in this project came from spending an hour on a baseline with no
   parameters, not from months of model development.
3. **A plausible mechanism is not a measured one.** I asserted an
   explanation, withdrew it, proposed another, and was wrong again, before
   settling it with arithmetic that took ten minutes. Measuring first
   would have been cheaper than reasoning three times.
4. **Verification tools need their own tests.** The checker that reported
   everything fine while blind is the clearest lesson in the project.
5. **Recording failures is a research output.** The decision log — with
   every withdrawal and every correction-of-a-correction — is the part of
   this work I would most want a supervisor to read.
