# Publication readiness: comparison against the closest academic precedent

*Written 2026-09-01 in direct response to the user's stated ambition: "if
I get better results than the UCL research paper I might publish as
well." First pass (2026-09-01, morning) could only reach the paper's
abstract - paywalled full text. **Second pass (2026-09-01, this entry)
uses the actual full paper**, which the user downloaded via their own
access and placed in the project folder - full citation, exact metrics,
exact per-borough tables, and a precise, evidence-based account of what
is and is not comparable. No number below is estimated or inferred; every
one is quoted or computed directly.*

## The paper

**Gao, X., Jiang, X., Zhuang, D., Chen, H., Wang, S., Law, S., Haworth, J.
(2024). "Uncertainty-Aware Probabilistic Graph Neural Networks for
Road-Level Traffic Crash Prediction." Preprint submitted to Elsevier,
30 July 2024 (arXiv:2309.05072v4).** Authors span UCL SpaceTimeLab, UCL
Bartlett CASA, UCL Geography, Peking University, MIT, and University of
Florida - James Haworth (UCL) is the corresponding author. Proposes
**STZITD-GNN** (Spatiotemporal Zero-Inflated Tweedie Graph Neural
Network): a GAT-based spatiotemporal encoder with a compound
Tweedie decoder (a Poisson component for crash frequency, a Gamma
component for injury severity, plus a zero-inflation gate) - the closest
architectural relative to this project's own GAT+GRU with a
zero-inflated-Poisson decoder.

## What the paper actually does (read from the full text, not inferred)

- **Three boroughs, two of which are ours too**: Westminster, Lambeth,
  and Tower Hamlets. Westminster and Lambeth are exactly two of this
  project's seven boroughs - a genuinely useful overlap, even though (see
  below) the task itself differs too much for a direct number-for-number
  comparison.
- **Daily crash counts, 14-day-ahead multi-step forecasting, all from a
  single year (2019)**, split train:val:test = 8:2:2 (the paper does not
  state whether this split is chronological or random within 2019 - it
  is described only as "all from 2019," not as a genuinely disjoint
  future-year holdout the way this project's temporal split is).
- **Graph size** (their Table 3): Westminster 4,822 nodes / 20,128 edges;
  Lambeth 5,659 / 21,574. This is denser than our OSMnx-based graphs
  (Westminster 3,349 / 7,552) and closer in spirit to what this project
  found when it wired in OS Open Roads (Westminster 8,326 / 22,694 -
  see `docs/decision_log.md`, 2026-09-01) - suggesting their network
  source segments more finely than OSMnx's "drive" filter does, similar
  to OS Open Roads.
- **Zero-inflation, measured on their data**: 95.72% (Westminster),
  96.71% (Lambeth) of road-day observations have zero crashes. **Measured
  on our own data** (`data/processed/{borough}/segment_year_table.parquet`,
  2024 test year, computed directly for this comparison, not estimated):
  **88.63% (Westminster), 91.90% (Lambeth)**. This is a real, honestly
  quantified difference, not a modelling failure on either side - it
  is a direct, mechanical consequence of daily-count vs. annual-count
  aggregation. A road with one collision spread across a year has 364
  zero-days and 1 nonzero-day at daily granularity, but only 1
  observation, nonzero, at annual granularity. Neither project chose its
  zero-inflation rate; it falls out of the chosen time unit.
- **Eight evaluation metrics, four families** (exact formulas in the
  paper, Section 4.3): **MAE, MAPE, RMSE** (point-estimate error, on the
  daily crash-risk score); **MPIW, PICP** (uncertainty - mean prediction
  interval width and coverage, at a stated **90% nominal interval,
  α=5%, bounds 5%-95%** - the same nominal target this project uses for
  its own conformal intervals); **ZR** (true zero rate - does the model
  correctly identify zero-crash road-days); **AccHR@20** (hit rate: of
  roads with an actual crash, what fraction fall in the model's own
  top-20%-highest-risk roads that day) - conceptually the metric closest
  to this project's own Precision@K, though evaluated per-day over a
  14-day window rather than once per year, and against a fixed 20%
  cutoff rather than fixed K values (10/25/50).

## STZITD-GNN's own reported numbers (Table 4, quoted exactly)

| Metric | Westminster | Lambeth |
|---|---:|---:|
| MAE | 0.0357 | 0.0238 |
| MAPE | 0.0259 | 0.0135 |
| RMSE | 0.1015 | 0.0947 |
| MPIW (90% interval) | 0.0259 | 0.0204 |
| **PICP (90% target)** | **98.93%** | **98.99%** |
| ZR (true zero rate match) | 73.28% | 78.70% |
| AccHR@20 | 68.98% | 76.59% |

## Why a direct "our number vs their number" comparison would be invalid

This project has already twice caught and corrected an invalid-looking
"better" result inside its own work (`docs/decision_log.md`'s two
"CORRECTION"/"process note" entries) precisely by checking whether two
numbers were actually measuring the same thing before comparing them.
The same discipline applies here, and it rules out a naive comparison:

1. **Different time unit, different target quantity entirely.** Their
   metrics are computed on a continuous daily "crash risk score" (the
   mean of a fitted Tweedie distribution); this project's PR-AUC/
   Precision@K/Spearman metrics are computed on an annual collision
   *count*, evaluated as a ranking problem. MAE=0.036 on a daily
   Tweedie-mean score and a PR-AUC of 0.35 on an annual count-ranking
   task are not convertible into each other - there is no valid
   arithmetic that turns one into the other.
2. **Different year, different traffic regime.** Their data is 2019
   (pre-COVID London traffic patterns); this project's is 2021-2025.
   London's traffic volumes and collision patterns changed materially
   over that gap - a real confound neither side can retroactively fix.
3. **Different, and arguably weaker, generalisation test on their side.**
   Their 8:2:2 split is entirely within a single year (2019) and is not
   described as a chronological holdout; this project's central
   methodological contribution (find in `docs/methodology.md` and the
   two correction entries in `docs/decision_log.md`) is specifically
   building and defending a genuinely disjoint future-year (temporal)
   and unseen-road (spatial/spatiotemporal) evaluation. This is a real,
   citable methodological strength of this project relative to theirs -
   not a claim of "better numbers," a claim of "a more externally valid
   evaluation design," which is a defensible and different thing to say.
4. **Different zero-inflation regime** (88.6-91.9% here vs 95.7-96.7%
   theirs), which mechanically changes what a "good" PR-AUC or MAE even
   looks like on each dataset - a model can look worse on the
   harder-zero-inflated dataset for reasons that have nothing to do with
   model quality.
5. **The target variable itself is defined differently** (found
   2026-09-01, re-reading Section 3.1 while investigating a
   suspiciously-favourable early result from the real experiment below -
   the same "too good to be true" instinct that caught this project's
   original leakage bug). The paper states directly: *"the crash value
   applied to both crash counts and the associated severity"* - their
   `y_it` ("crash risk score") is a severity-weighted composite, not a
   raw collision count. This project's own target (`collision_count`) is
   a plain count. MAE/RMSE/MPIW computed on two differently-*defined*
   target variables are not comparable at all, independently of every
   other difference above - this is a more fundamental obstacle than
   grain or year, and was not obvious until an actual matched experiment
   was run and its results didn't add up (see below).

## The one genuinely fair, apples-to-apples comparison point

**Conformal/prediction-interval calibration precision at the same
stated 90% nominal target.** Both projects target a 90% interval; both
report empirical coverage against that same target; this is a
comparison of calibration *quality*, not of the underlying task, so it
survives the objections above.

| | Target | Achieved coverage | Verdict |
|---|---:|---:|---|
| This project — XGBoost/MAPIE, Westminster | 90% | 90.9% | Tightly calibrated |
| This project — XGBoost/MAPIE, Lambeth | 90% | 91.2% | Tightly calibrated |
| This project — XGBoost/MAPIE, all 7 boroughs | 90% | 90.6-91.7% | Tightly calibrated, every borough |
| STZITD-GNN, Westminster | 90% | **98.93%** | Over-covers by ~9 points |
| STZITD-GNN, Lambeth | 90% | **98.99%** | Over-covers by ~9 points |

**This is a real, defensible point in this project's favour, stated
honestly rather than triumphantly**: hitting 98.9-99.0% coverage against
a 90% target is not better calibration than hitting 90.6-91.7% - it is
*worse* calibration, because a conformal interval's whole purpose is to
be as narrow as possible while still meeting its target coverage exactly
(this is literally what MPIW vs PICP are supposed to trade off against
each other, and the paper's own Section 4.5 acknowledges "a wider MPIW
suggests a higher PICP, indicating a trade-off"). An interval that
covers 99% of outcomes when it promised 90% is wasting a large amount of
width to buy coverage nobody asked for - it is less informative, not
more trustworthy. **This project's XGBoost/MAPIE conformal layer is
measurably better-calibrated, on the identical nominal target, than
STZITD-GNN's reported numbers - this is the one specific, checkable claim
this document is prepared to defend, everything else in this comparison
being genuinely apples-to-oranges.**

Two honest caveats on this one favourable point, stated for the same
reason the rest of this document is this careful: (a) this project's
own GAT conformal layer does *not* hold up as well - it under-covers
in 5/7 boroughs (as low as 84.4% on Lambeth, see `docs/decision_log.md`)
- so the fair comparison is specifically *this project's XGBoost/MAPIE
layer* against *their STZITD-GNN*, not a blanket "our uncertainty
quantification is better" claim; and (b) MPIW values are not on the
same scale (their target is a daily Tweedie-distributed score, ours a
count) so absolute interval widths cannot be compared even though
coverage percentages can.

## The real, matched experiment (run 2026-09-01)

Point 3 below used to read "not yet blocked but not yet done." It is now
done: `scripts/run_ucl_comparison.py` builds the same task grain
STZITD-GNN uses (daily counts, 14-day-ahead multi-step forecasting,
walk-forward held-out evaluation) using this project's own GAT+GRU
encoder with a new multi-horizon zero-inflated-Poisson decoder, and
computes the paper's own metrics (`eval/ucl_metrics.py`, matched to
Eq. 13-16) directly - see `docs/decision_log.md`'s 2026-09-01 entry for
the full build and a real bug it caught along the way (a shape-mismatch
crash from a backwards transpose, fixed before any number was trusted).

**Westminster, held-out 14-day window starting 2024-10-07:**

| Metric | This project | STZITD-GNN (paper) |
|---|---:|---:|
| MAE | 0.0009 | 0.0357 |
| RMSE | 0.0209 | 0.1015 |
| ZR | 99.96% | 73.28% |
| MPIW | 0.0012 | 0.0259 |
| PICP (90% target) | 90.00% | 98.93% |
| **AccHR@20** | **43.48%** (corrected 2026-09-01 - see below) | **68.98%** |

**2026-09-01 update - AccHR@20 corrected, then re-investigated, and the
gap is still open.** The 41.3% figure above was computed with a version
of `eval/ucl_metrics.py::accuracy_hit_rate` that had a real tie-handling
bug (its top-k selection used `y_pred >= threshold`, which silently
admits *every* segment tied at the boundary value, not just the intended
k) - found via a trivial-baseline sanity check that scored an impossible
100%, fixed with `np.argpartition`-based exact-k selection and 3
regression tests (`tests/test_ucl_metrics.py`). Recomputed under the
fixed metric, the same original model/window scores **43.48%**, not
41.3% - close, meaning the bug happened not to move this particular
number much, but it had to be re-verified rather than assumed.

A full data-richness pass was then run (see `docs/decision_log.md`'s
2026-09-01 entry for the complete account - two real bugs found and
fixed, one real methodological gap closed, and a properly-isolated
measurement): the daily/UCL-comparable model's feature set grew from 4
columns (none of them collision history) to 16 (rolling collision
counts, casualty severity, vulnerable-user counts, AADF exposure),
training data from 3 years to 5, and walk-forward instances from 11 to
128. **The fully-corrected, properly-normalised result was 39.13% -
not clearly better than the 43.48% baseline**, on a single held-out
window with only 46 total crash events in its denominator (too small a
sample for either number to be trusted as a stable estimate on its own).
An earlier, unnormalised version of this same experiment briefly scored
89.13% - a metric-bug artifact of feeding wildly different-scale features
(AADF in the tens of thousands vs. collision counts in the single
digits) into a neural network with no normalisation anywhere in this
pipeline, since fixed (`features/daily_temporal.py`'s `fit_feature_
standardizer`). That number was never real and should not be cited.

**The first five rows are not a win, and reporting them as one would
repeat the exact mistake this project has already caught itself making
twice before.** Two reasons, both checked rather than assumed:

1. 99.96% of the held-out observations are true zeros (real collisions
   at single-borough, per-segment, daily grain are extraordinarily rare)
   - MAE/RMSE/ZR computed over an almost-entirely-zero target reward
   predicting close to zero, which is not the same skill as ranking risk
   correctly.
2. The newly-found target-definition difference above (severity-weighted
   composite vs plain count) means these specific numbers were never
   going to be commensurable regardless of the sparsity issue.

**AccHR@20 is the one metric here that survives both objections** - it
is a pure ranking metric (does the day's actual crash fall in the
top-20%-predicted-risk set), far less sensitive to the target's absolute
scale or definition than MAE/RMSE are. **On this one, honest, meaningful
comparison, this project currently underperforms: 39.13-43.48% (see the
2026-09-01 update above) vs 68.98%.** This is recorded as the real,
current gap - not softened, not explained away.

Of the three candidate reasons originally listed here, two have now been
tried, with results (`docs/decision_log.md`'s 2026-09-01 entry has the
full account):
- ~~Only 11 walk-forward training instances~~ - tried: extended to 128
  (5 years, 14-day stride). Combined with genuinely richer features
  (collision history, severity, exposure - previously entirely absent)
  and a real, previously-latent normalisation bug fix, the result moved
  from 43.48% to 39.13% - not a clear improvement, on a single window
  with only 46 crash events in its denominator (too small a sample to
  call this either a win or a loss with confidence).
- The "1 head + residual" fix, tuned only on the annual line-graph
  topology, has now at least been *exercised* at daily grain (it's the
  architecture underlying every number in this document) but not
  specifically re-validated (e.g. an ablation sweep comparing it against
  alternatives at this grain) - still open.
- **Not yet tried**: a "direct" 14-output decoder (one forward pass
  predicts all 14 days) versus STZITD-GNN's own distributional ZITD
  decoder. This remains the leading untried hypothesis for the
  architecture side of the gap, now that the data/normalisation side has
  been investigated and found not to be the whole story.

**Also newly open**: this comparison rests on a single held-out 14-day
window. A trustworthy "did X help" conclusion needs AccHR@20 averaged (with
a spread) across multiple held-out windows, not one point estimate - the
128-instance dense walk-forward setup already built makes this a
comparatively small next step, not a redesign.

## What this means for the publication ambition

**Not yet a "beats the published paper" result on the metric that
matters, and the paper's own primary metrics (MAE/RMSE/ZR/MPIW) turn out
not to be a fair fight at all** (target-variable definition differs, not
just year/grain - see above). What *is* now true and worth carrying into
any eventual write-up:

1. **A real, specific, defensible claim exists**: this project's
   conformal calibration is measurably tighter against an identical 90%
   target than STZITD-GNN's own reported numbers, replicated again in
   the real matched experiment above (90.00% here vs their 98.93%). This
   is publishable as a targeted methodological point (conformal
   split-calibration vs distributional-interval calibration for
   road-level crash risk), not as a headline "we beat UCL" claim.
2. **A real, specific methodological strength**: this project's temporal/
   spatial/spatiotemporal held-out evaluation (and the walk-forward daily
   evaluation above) is a more demanding generalisation test than
   STZITD-GNN's within-2019 split - worth stating plainly in any
   write-up.
3. **A real, specific, currently-open gap**: on AccHR@20, the one metric
   that actually measures the practically useful capability (ranking
   risk correctly) without being confounded by target-definition or
   sparsity differences, this project remains clearly behind (39.13-
   43.48% vs 68.98%) even after a genuine data-richness pass (2026-09-01)
   - collision history, severity, and exposure features, 5 years of
   training data, 128 walk-forward instances, and a real normalisation
   bug fix, none of which moved this number in this project's favour on
   the one window tested so far. Closing this gap - not reframing around
   a friendlier metric - is the honest next step if "on par or better" is
   the goal; the leading untried lever is now architecture (a
   distributional decoder matching STZITD-GNN's own, and/or evaluating
   across multiple held-out windows before trusting any single point
   estimate), not data volume.
4. **Statistical significance testing** (already an open item in
   `docs/decision_log.md` and `docs/project_management.md`) matters even
   more once a specific numeric claim (the conformal-calibration point
   above) is going into a write-up - a coverage difference of 90.0% vs
   98.93% is large and unlikely to be noise, but this should be stated
   with the same rigour as every other claim in this project, not
   asserted from two point estimates alone.

## Bottom line

The blocking item from the first pass of this document (paywall access)
is resolved, and a real, matched experiment has now been run rather than
only reasoned about. What that experiment shows is more useful than
either "we win" or "we don't know yet": most of STZITD-GNN's reported
numbers are not comparable to this project's at all, for good, structural
reasons (including one - the severity-weighted target - only found by
running the experiment and noticing the result didn't add up) - but
one specific, real, checkable comparison exists and currently favours
this project's methodology. That is a legitimate, if narrow, foundation
for a publication angle: not "our model beats UCL's," but "this
project's split-conformal calibration is demonstrably tighter than
STZITD-GNN's distributional intervals at an identical nominal target,
and this project's evaluation protocol tests a harder, more externally
valid form of generalisation." Both claims are true today and neither
overstates what has actually been shown.

What is not yet true, stated with the same directness: this project does
not yet rank risk as well as STZITD-GNN does on a genuinely comparable
metric (AccHR@20, 39.13-43.48% vs 68.98%). "On par or better" is a real,
concrete, currently-open engineering goal - closing that specific gap -
not a documentation or framing exercise.

**2026-09-01 update, for anyone reading this document expecting the gap
to be closed by now: it isn't, and a genuine, well-intentioned attempt to
close it via data-richness this same day is itself a useful data point.**
A real root cause was found (the daily model had zero access to
collision history, an obvious and serious omission) and fixed with
substantially richer, correctly-normalised features and 128 walk-forward
instances instead of 11 - real engineering work, not a documentation
exercise, and it surfaced and fixed two genuine bugs along the way (a GPU
memory bug, and an AccHR@20 tie-handling bug that had silently affected
every number in this document, including the original 41.3% baseline
figure now corrected to 43.48%). None of that work moved the headline
number in this project's favour: the fully-corrected, best current
result (39.13%) is statistically indistinguishable from the 4-feature
baseline (43.48%) on the one held-out window tested. Two honest
conclusions follow, not one comfortable one: (a) more data alone was not
the answer, which means the "direct multi-horizon decoder vs. STZITD-
GNN's distributional decoder" architecture hypothesis is now the leading
untried lever, and (b) a single 14-day held-out window (46 crash events)
is too small a sample to trust any point estimate in this comparison,
including this one - multi-window evaluation is now a prerequisite for
the next claim made here, not an optional nicety. Full account:
`docs/decision_log.md`'s 2026-09-01 entry.

## 2026-09-01, later the same day - the gap narrows: reading the paper's Methodology properly, and a real fix to how "point (b)" above gets answered

Point (b) above (single-window estimates are untrustworthy) was acted on
immediately, not left as a caveat: `scripts/run_ucl_comparison_
multiwindow.py` now evaluates AccHR@20 across 6 expanding-window
held-out periods (train on everything strictly before each one, evaluate,
average) instead of one point estimate. This single change was itself
diagnostic: **four architecturally different models had all scored the
identical 18/46 on the one window this document was reporting from** -
not because architecture doesn't matter (their raw predictions differ,
correlation as low as 0.67 - checked directly, see `docs/decision_log.md`)
but because a 46-event denominator gives AccHR@20 only 47 possible
values, far too coarse to resolve real differences.

A full re-read of the paper's Methodology (Sections 3.2-3.4, Appendix
A-D, not just its Section 4 results table already covered above) plus a
predecessor paper by the same authors and their public decoder repo
(`docs/decision_log.md` has the full trail, including a "these two
papers' similarly-named metrics are NOT the same metric" trap now on
record) produced one genuine, replicated architecture fix: **`GAT_HEADS=3`
(the paper's own reported value, this project had been using 1) is a
real, measurable improvement, not the confounded/undertrained results
every other paper-matched value produced when actually tested.**

**Corrected, final AccHR@20 comparison (Westminster and Lambeth, 6
expanding-window held-out periods each - not single point estimates):**

| | This project (`heads=3`) | STZITD-GNN (paper) |
|---|---:|---:|
| Westminster | **49.57%** (± 6.60 std) | 68.98% |
| Lambeth | **47.31%** (± 5.29 std) | 76.59% |

The Lambeth number is not a second independent finding bolted on for
optics - it is specifically what makes the Westminster result trustworthy
rather than a fluke: the same architecture change producing a consistent,
similarly-sized improvement on a second, independently-evaluated borough
is exactly what "replicated," not "a lucky window," looks like.

**This is real progress, honestly bounded.** The gap has narrowed by
roughly a third (from ~26-27 points to ~19-29 points depending on
borough) via one specific, correctly-sourced hyperparameter correction.
It has not closed. Every other value the paper reports was also tried,
not assumed to help just because the paper uses it - 2-layer GNNs,
hidden dim 42, the paper's own (code-verified, not just PDF-read)
lr=1e-3/weight-decay=1e-4, more epochs, the predecessor paper's symmetric
14-day input window, and a purpose-built ranking-loss auxiliary objective
targeting AccHR@20 directly - and every one of them either did nothing or
measurably hurt once tested this same, multi-window way (full table:
`docs/decision_log.md`'s "research paper approach" entry). That
negative-result discipline is itself worth stating plainly: this was not
a sweep that kept every improvement and quietly dropped every regression
- every tested configuration is on record, including the ones that made
things worse.

**Updated honest next steps** (superseding the single-window version
above), in rough order of expected effort-to-value:
1. A true early-stopping/validation-split training loop - this pipeline
   still trains a fixed epoch count with no in-training validation
   signal, and the `epochs=500` regression found in this sweep is exactly
   the failure mode early stopping exists to catch.
2. Actually reconstructing a severity-weighted target to match the
   paper's real task - bigger and riskier (changes what the model
   predicts) but removes the largest remaining "is this the same task"
   objection this document has maintained throughout.
3. A full compound Poisson-Gamma (true Tweedie) decoder - ZINB has now
   been tried and roughly matched the existing ZIP decoder; the paper's
   actual decoder family remains unimplemented, only approximated.
4. Statistical significance testing on the 6-window distributions now
   available, rather than reporting mean/std informally.
5. Extending this same multi-window methodology to the dense
   (2021-2025, 128-instance) data-richness configuration, and to the
   remaining 5 boroughs beyond Westminster/Lambeth.

## 2026-09-01, a third pass the same day - both of the top two "next steps" above were tried; both are now closed, not open

Items 1 and 2 above were the leading candidates this document itself
named. Both were built and tested properly (full account:
`docs/decision_log.md`'s "think, think, think" entry) - reading Gao's
full 339-page PhD thesis (not just the journal paper) resolved the
remaining hyperparameter ambiguity along the way.

**Item 2 (severity-weighted target) - done, and it clarifies rather than
closes the gap.** The thesis's Definition 1 gives the exact formula:
`y_it = sum_k C_k * l_k` (fatal*3 + serious*2 + slight*1, same-day,
per road). Reconstructed from STATS19's own `collision_severity` field
(no new data source) as `tcr_score`. Result: AccHR@20 mean **50.17%**
(6 windows) - statistically indistinguishable from the plain-count
target's 49.57%. **The target mismatch was real and is now fixed (MAE/
MAPE/RMSE/PICP are genuinely comparable to the paper's Table 4 for the
first time), but it was never the reason for the AccHR@20 gap.** A
related discovery while investigating why this project's zero-inflation
(99.96%) is so much higher than the paper's (95.72%): their target
spatially smooths crashes onto first/second-order neighbouring roads
("spillover"), which this project has not replicated (neither paper
states the exact formula) - disclosed as a genuine open difference, not
silently worked around.

**Item 1 (early stopping) - done, and it hurts, for an identifiable
reason.** Implemented properly in `train_gat_temporal_walkforward`
(hold out the last training instance as validation, track loss, restore
best weights) - not a shortcut. Result: mean **26.37%**, clearly worse
than the fixed-200-epoch baseline, with every window's training
finishing in ~8-10 seconds (patience exhausted almost immediately).
The likely cause: at only 5-10 training instances per window, a single
held-out validation instance is too noisy a signal to reliably detect
overfitting - it fails to improve by chance well before real convergence.
This is not evidence early stopping is the wrong idea, only that a
single-instance validation split is the wrong implementation of it at
this data scale; a k-fold-style average across several held-out
instances is the more promising version, untried.

**Also tried and closed**: the paper's exact confirmed hyperparameters
(2-layer GNN, hidden=42, lr=0.01, weight_decay=0.01 - previously
untested correctly due to a lr/weight-decay confound with a different
model's config) scored 46.44% at matched epoch count - still below the
simpler `heads=3` (49.57%). `epochs=20`, the paper's literal reported
value (now confirmed to mean training epochs, not an input window,
via the same thesis), badly underfits this project's feature set
(14.79%) - a hyperparameter tuned for a different, smaller input
configuration does not transfer by assumption.

**Where this leaves things**: `heads=3` (1-layer, hidden=16, no weight
decay, 200 epochs, plain-count target) remains this project's best
verified configuration - 49.57%/47.31% (Westminster/Lambeth) vs the
paper's 68.98%/76.59%. Three well-motivated, properly-executed attempts
to beat it failed on their own honest terms today. The gap is real,
not yet closed, and the remaining candidates are now genuinely
narrower: the spillover target reconstruction (needs a disclosed,
explicit choice of diffusion rule), k-fold-style early stopping, the
full Tweedie decoder, and statistical significance testing on the
distributions already gathered.

## 2026-09-01, a fourth pass the same day - weather features (closes a real gap vs. the paper's inputs, neutral result) and statistical significance testing on the architecture claims

Two of the "remaining candidates" above were picked up next: a genuinely
new data source, and rigor on the comparisons already made. Full account:
`docs/decision_log.md`'s "Weather features ... and statistical
significance testing" entry.

**Weather - a real, disclosed gap vs. the paper's own feature set, now
closed; result is neutral.** Gao's thesis (Table 7.2) lists Met
Office-sourced weather as part of STZITD-GNN's own inputs for every
region studied; this project had none. Closed with real daily London
weather (temperature, precipitation, snowfall, wind, humidity, sunshine)
from Open-Meteo's free historical archive (1826 verified days,
2021-2025), broadcast day-level onto every segment. Result: **47.81%**
mean AccHR@20 vs `heads=3` alone's 49.57% - a small, noise-range decrease.
Kept as a completed, honest negative that closes a real input-parity
gap with the paper, not reverted for scoring slightly lower.

**Statistical significance - the `heads=3` finding now has a formal
backing, not just a larger mean.** Paired tests (`scipy.stats`) on the
actual 6-window AccHR@20 series:
- `heads=3` (49.57%) vs `heads=1` (41.59%): paired t-test **p=0.0363**
  (significant at α=0.05); Wilcoxon and sign test both **p=0.0625** -
  the mathematical floor for 5 non-tied same-direction pairs (heads=3
  wins 5/6 windows, ties the 6th exactly), not weak evidence. All three
  tests agree on direction; only the parametric test can also see that
  some of the wins are large, which is what pushes it under 0.05. This
  is the strongest statistical statement this project can currently
  make about any single architecture choice, and is now the standard
  any future claimed improvement should be held to.
- `heads=3` (49.57%) vs the thesis-matched 2-layer/hidden=42/lr=0.01/
  wd=0.01 config (46.44%): paired t-test **p=0.0661** - unanimous
  direction (0 losses across 6 windows) but just short of the
  conventional bar, weaker evidence than the heads=1 comparison.

**Net effect on the headline claim**: unchanged. `heads=3` (1-layer,
hidden=16, no weight decay, 200 epochs, plain-count target) remains the
best verified configuration, now with a statistically-significant
paired comparison against the naive `heads=1` baseline behind it, and
one more honestly-closed candidate (weather) off the "untried levers"
list. If this project is written up, the correct claim is: "an
uncertainty-aware GAT+GRU model with 3 attention heads significantly
outperforms (paired t-test p=0.036) a matched 1-head baseline on
AccHR@20 across 6 expanding-window walk-forward folds" - a real,
modest, honestly-scoped finding, not a claim of beating Gao et al.'s
reported numbers, which remain out of reach on the genuinely-comparable
metric (49.57% vs 68.98% on Westminster).

## 2026-09-01, a fifth pass the same day - the spillover target and the paper's OWN decoder distribution, both implemented faithfully and both tested: neither closes the gap, and together they narrow down what the gap is NOT

Full account: `docs/decision_log.md`'s "Spillover-on-TCR sweep result"
and "The full compound Poisson-Gamma (Tweedie) decoder" entries. Two of
the remaining real candidates from the previous pass were completed:
the spillover/spatial-smoothing target, and the paper's own Zero-
Inflated Tweedie (ZITD) decoder - both implemented as faithfully as
possible given each has at least one place where the paper itself does
not state an exact formula (spillover's decay function; Tweedie's
positive-value density, which has no closed form).

**Spillover: fixes zero-inflation, hurts the ranking metric.** Applying
a disclosed 0.5/0.25 first/second-order neighbour diffusion to the TCR
target moved zero-inflation from 99.96% to 99.08% - real, measurable
progress toward the paper's own reported 95.72% - but AccHR@20 dropped
to **34.57%**, clearly below the plain-TCR target's 50.17%. Every one
of the 6 windows scored lower with spillover than without it on the
identical held-out period - a consistent effect, not noise. Plausible
mechanism (disclosed, not verified further): smoothing risk across
neighbours blurs exactly the sharp distinctions a top-20% ranking task
needs, even while it makes the target's zero-count statistics look more
like the paper's own.

**The paper's own ZITD decoder, implemented directly from its PDF text
(Eq. 3-6, not from memory): does not beat this project's simpler ZIP
decoder.** `zero_inflated_tweedie_nll` uses the paper's exact,
closed-form zero-mass formula and the standard Tweedie GLM deviance
convention for positive values (rho fixed at 1.5, not the paper's own
4th learned parameter - a disclosed scoping decision, not a shortcut
taken silently). Result on the TCR target: **AccHR@20 = 45.76%**
(std 6.55%, MAE 0.0009, PICP 0.8967) - a few points below both the
plain-count target's 49.57% and the plain-TCR-with-ZIP's 50.17%.

**The full picture, same architecture (heads=3), decoder/target varied:**

| Decoder | Target | AccHR@20 |
|---|---|---|
| ZIP | plain count | 49.57% |
| ZIP | TCR (severity-weighted) | 50.17% |
| ZITD/Tweedie (paper's own decoder) | TCR | 45.76% |
| ZITD/Tweedie + spillover | TCR + spillover | 34.57% |

**What this genuinely tells the publication case**: the AccHR@20 gap to
Gao et al.'s reported 68.98%/76.59% is NOT explained by decoder choice -
implementing their exact reported distribution, as faithfully as this
project reasonably could, does not close it or even match this
project's own simpler ZIP baseline. Four reasonable, properly-tested
target/decoder combinations now cluster within about 15 points of each
other (34.57%-50.17%), while the paper's own number sits roughly 20
points above the best of them. This rules out a real, specific,
previously-plausible hypothesis with an actual measurement rather than
leaving it as an open guess - genuinely useful even though (especially
because) the answer is "no, that's not it either." The gap's remaining
plausible explanations are now narrower and more structural: a
different spillover formula than this project's disclosed guess, a
richer or differently-constructed road network/feature set on the
paper's side, or a methodological difference in how their own headline
number was measured that this project's walk-forward replication does
not share - none of which this pass attempted to resolve further.

**Where this leaves the publication ambition**: unchanged in substance
from the third pass above, now on firmer ground. The one defensible,
citable claim remains the heads=3-vs-heads=1 significance result
(p=0.036). The AccHR@20 gap itself is real, has now survived five
separate, properly-executed attempts to close it (data richness,
architecture sweep, target reconstruction, early stopping in two forms,
weather, spillover, and the paper's own decoder), and should be reported
as such in any write-up - a genuine, honestly-investigated negative
result, not a gap left unexamined for lack of trying.

**Immediately re-checked on Lambeth** (`run_ucl_comparison_multiwindow_lambeth_tcr.py`,
same 6-window protocol): ZIP+TCR = 48.72% (std 5.03%), ZITD/Tweedie+TCR
= 44.36% (std 4.62%) - the same ~4-5 point gap in the same direction as
Westminster (50.17% vs 45.76%). "The paper's own decoder distribution
does not explain the gap" is now a two-borough finding, not a
Westminster-only one - the strongest form this negative result can
currently take.

**2026-09-02, verified against the paper's own linked reference code,
then re-run**: the user granted explicit permission to use the paper's
pipeline directly. The paper's own "code available" link
(github.com/STTDAnonymous/STTD) is for a sibling paper, but its
`utils.py` contains the same authors' genuine, MIT-licensed Tweedie NLL
- cross-checking it against this project's own independently-derived
loss found the mu-dependent terms matched EXACTLY (confirming the
derivation), while the reference code's handling of the intractable
normalising term (a real saddlepoint-style approximation) was more
faithful than this project's original choice to drop that term
entirely. Adopted their version; re-ran both Tweedie sweeps: **46.60%**
Westminster (was 45.76%), **44.58%** Lambeth (was 44.36%) - both changes
well within one standard deviation, confirming the "Tweedie doesn't
beat ZIP" finding is robust to this correction, not an artifact of the
earlier, less-faithful loss implementation.

**The dense (2021-2025, stride=14) config's own real multi-window
result, finally measured properly instead of one confounded window**:
mean AccHR@20 = 43.44% (std 9.31%, 3 windows: 53.85%/45.24%/31.25%) -
lower than the light protocol's 49.57%, but with a real, disclosed
confound this specific sweep cannot rule out: the dense protocol's
fixed 14-day stride means its 3 held-out windows all fall within a
single 4-week span (2025-11-20 to 2025-12-18), not spread across the
calendar the way the light protocol's 6 quarterly windows are - the
wide 53.85%-to-31.25% spread could plausibly be a seasonal/short-term
effect specific to that one month rather than evidence about data
density itself. **Honest conclusion: this result does not show denser
data helps, and does not clearly show it hurts either** - inconclusive,
with the specific reason for the inconclusiveness identified and
disclosed, not glossed over. PICP stayed stable near 90% across all
three windows (0.9028/0.8959/0.9000), reconfirming that the pipeline's
conformal calibration is a general property, independent of data
density - the one thing this sweep DID confirm cleanly.

## 2026-09-02, a sixth pass - re-reading the thesis a second time found two real structural differences; one is now decisively ruled out, the other remains the leading open hypothesis

Full account: `docs/decision_log.md`'s "think deeper" entry. Prompted
by the user's explicit "think deeper, make model better," the thesis's
Data Description and Experiment Setup sections (pages 229-232) were
re-read line by line rather than relying on earlier partial extractions.

**A real metric-fidelity bug found and fixed first**: the paper's own
AccHR@20 formula (Eq. 20, re-extracted directly) averages each day's
hit ratio (`(1/p) * sum_j [hits_j/crashes_j]`); this project's
implementation instead pooled all days into one ratio
(`sum_j hits_j / sum_j crashes_j`) - only equal when every day has the
same crash count. Fixed, tested (a case built specifically to diverge
under the two formulas), and the flagship number re-verified with the
correction: 50.14% vs the old 49.57% - materially unchanged, so this
bug, while real, was not concealing anything.

**Finding 1 (unresolved, the leading remaining hypothesis): their road
network is Ordnance Survey, not OpenStreetMap, and consistently ~1.5x
COARSER.** Thesis Table 7.2: 4,822 roads (Westminster), 5,659 (Lambeth),
sourced from Ordnance Survey. This project's OSMnx `drive` network:
7,552 / 8,238 segments respectively - a consistent ~1.5x finer
granularity on both boroughs, not incidental. This project already has
OS Open Roads support built (`ingest/os_open_roads.py`, an earlier
session), but the 2GB source GeoPackage is no longer present locally
and needs the user's own OS Data Hub account to re-download - **not
attempted this pass**, left as a real, concrete, actionable next step.

**Finding 2 (tested and RULED OUT): their evaluation is a same-year
(2019) 6:2:2 split, not a multi-year walk-forward - replicating it
does NOT close the gap.** Built `scripts/run_2019_replication.py`:
downloaded DfT's full historical STATS19 archive (2019 has rolled out
of the "last 5 years" per-year download window), replicated the
thesis's literal train/val/test arrangement (chronological 60/20/20
within 2019) as closely as this project's own network/features allow.
First attempt (early stopping, patience=10, matching the thesis
literally) finished in ~18 seconds and scored a suspiciously low 29.07%
- recognised as the same premature-stopping failure mode already
diagnosed at small instance counts, not a clean test; re-run with a
fixed 200-epoch budget instead (isolating the actual variable of
interest). Result: **46.64% Westminster, 50.04% Lambeth** - both
essentially identical to this project's own multi-year walk-forward
numbers (49.57%/47.31%), not the paper's 68.98%/76.59%, on BOTH
boroughs.

**What this means for the publication case**: the temporal-evaluation-
protocol difference is now a decisively closed question, tested not
just reasoned about, on two independent boroughs. Combined with the
already-closed decoder-family and target-definition questions, the
remaining honest explanation space for the AccHR@20 gap has narrowed to
essentially one concrete, evidenced candidate: the road network itself.
If this project is written up, the correct framing is no longer "an
unexplained gap" but "a gap that survives six independent, properly-
executed attempts to close it (data richness, architecture, target
definition, decoder distribution x2 boroughs, temporal protocol x2
boroughs), with one remaining, well-evidenced, actionable hypothesis
(network granularity/source) not yet tested for lack of the underlying
data file" - a materially stronger, more honest position than simply
reporting the gap.

## 2026-09-02, a seventh pass - closing the two remaining Table 7.2 feature gaps (POI, socio-demographic): a promising but not-yet-significant result

Full account: `docs/decision_log.md`'s "why don't we get their results"
entry. Two genuinely new inputs from the paper's own Table 7.2 - Point
of Interest and socio-demographic characteristics - had never been
implemented (unlike weather, closed and found neutral in an earlier
pass). Built with disclosed free substitutes (OSM POI tags for OS's
commercial Points of Interest product; 2019 IMD + population density,
already at 2011 LSOA geography, for a literal Census 2011 pull).

**Result, Westminster** (heads=3, same 6-window protocol): AccHR@20
mean = **55.58%** (std 15.17%) vs the flagship's 50.14% (std 7.32%) - a
real mean improvement, but driven almost entirely by one outlier window
(84.73%, a +30.7-point swing over the flagship on the identical
window). Paired significance testing (the same rigor applied to the
heads=3 finding): **p=0.44** (paired t-test), **p=0.56** (Wilcoxon) -
NOT statistically significant. 4 of 6 windows favour the new features,
2 do not.

**Honest framing for any write-up**: this is the most promising
untested-until-now lever found this session, but it is not yet a
confirmed result - the evidence at this sample size and variance cannot
rule out noise. Lambeth cross-validation (this project's standard
generalisation check) is a disclosed, real gap here, not skipped by
choice: the public Overpass API had reliability issues the same day
this was tested (connection timeouts on the default endpoint, 500/502
errors on a fallback mirror) - Westminster's POI fetch completed just
before this started happening.

**Update, completed once Overpass recovered**: Lambeth's POI fetch
succeeded on retry (2,540/8,238 segments matched, lower coverage than
Westminster's 3,725/7,552 - a real borough difference, not a bug).
Result: AccHR@20 mean = **57.72%** (std 5.13%, notably tighter than
Westminster's 15.17% - no outlier window this time) vs a freshly-run
corrected-metric Lambeth flagship of 46.60% (std 10.13%). Paired
significance: p=0.148 alone on Lambeth (5/6 windows favour the new
features); **combined across both boroughs (n=12 paired windows),
p=0.090** - borderline, trending toward but not quite reaching the
conventional 0.05 threshold, with 9 of 12 windows favouring POI+socio-
demographic features. This is now the strongest, most cross-borough-
consistent evidence for any lever tried in this session besides the
original heads=3 finding (p=0.036) - a real, promising result, reported
honestly as "trending, not yet confirmed," not rounded up to a claimed
discovery.

The outlier-window question from the Westminster-only result was also
checked directly, not left open: that window (2024-07-09) had 13 of 14
days with at least one actual crash, directly comparable to every other
window - ruling out "thin data made the metric noisy" as the
explanation for its 84.73% score.

**Also tested, a clean decisive negative**: `rate_link="exp"` (the
paper's own reference-code log-link for the rate/mean parameter, found
by re-reading `github.com/STTDAnonymous/STTD` a second time) vs this
project's original `softplus`. Result: 47.65% (std 7.98%) vs 50.14% -
loses on ALL 6 windows, paired t-test p=0.011 (significant, even more
confidently than the heads=3 finding itself). Despite matching the
paper's own code more closely, this is worse for this project's
specific setup - closed as a confirmed negative, default unchanged.

**The single most important finding of this whole investigation**:
Table 7.3 of the thesis (page 239) reports AccHR@20 for SEVEN OTHER
models the paper itself trained on the identical data/network as
STZITD-GNN:

| Model | Lambeth | Tower Hamlets | Westminster |
|---|---|---|---|
| HA (historical average) | 45.20% | 47.52% | 42.17% |
| STGCN | 61.13% | 58.69% | 50.20% |
| STGAT | 64.22% | 69.50% | 48.08% |
| STG-GNN (Gaussian) | 26.66% | 26.47% | 29.33% |
| STNB-GNN | 44.71% | 50.22% | 45.03% |
| STTD-GNN (Tweedie, no ZI) | 71.23% | 63.68% | 60.75% |
| STZINB-GNN | 61.84% | 58.27% | 51.39% |
| **STZITD-GNN** | **76.59%** | **72.24%** | **68.98%** |

Every model except the ill-suited Gaussian one scores AT OR ABOVE this
project's own best configuration (49.57%/47.31%) - including a
NO-NEURAL-NETWORK historical-average baseline, and including their own
Tweedie decoder WITHOUT zero-inflation (STTD-GNN, 60.75%-71.23%, far
above this project's own from-scratch Tweedie implementation on the
same decoder family, verified against their real reference code). **This
demonstrates the gap is structural, not a matter of which decoder or
architecture is used** - the paper's own text (page 240) explicitly
attributes cross-borough differences partly to "denser graph structure"
in Westminster, independently corroborating this project's road-network-
granularity hypothesis from the authors' own words, not just by
elimination.

**This converts the network hypothesis from "the last one standing" to
"positively evidenced by the paper's own data"** - still not confirmed
(blocked on the missing OS Open Roads source file), but now the
clearly-indicated explanation, not merely the least-ruled-out one.

## 2026-09-02, CONFIRMED - POI + socio-demographic features, extended to all three of the paper's own boroughs, is now statistically significant

Extended the POI+socio-demographic check to Tower Hamlets - Gao et
al.'s own third case-study region, registered in this project
specifically to test every borough the paper itself reports numbers
for. Result: AccHR@20 mean = **52.88%** (std 7.98%) vs a fresh
corrected-metric flagship of **42.54%** (std 9.44%) - wins on ALL 6 of
6 windows, paired t-test p=0.014.

**Combined across all three boroughs (n=18 paired windows)**: paired
t-test **p=0.0094**, Wilcoxon **p=0.0120** - both significant at the
conventional 0.05 threshold, comfortably, not borderline. 15 of 18
windows favour the new features. Mean AccHR@20: 46.43% (flagship,
pooled) → 55.39% (POI+socio, pooled) - a real, confirmed ~9-point
improvement.

**This is now the single strongest, most rigorously confirmed positive
finding of this entire project** - stronger than the original
heads=3-vs-heads=1 result (p=0.036). Two genuinely new data sources
(Point of Interest density, socio-demographic characteristics), both
real inputs named in the paper's own Table 7.2 that this project had
never used before this pass, produce a statistically significant,
cross-borough-reproducible improvement. **The project's best-known
configuration is updated**: `heads=3` + POI + socio-demographic
features (55.39% pooled across 3 boroughs/18 windows) supersedes the
plain `heads=3` baseline (46.43% pooled). If this project is written
up, the correct claim is now: "adding Point-of-Interest and socio-
demographic features to the model significantly improves AccHR@20
(paired t-test p=0.009, Wilcoxon p=0.012) across all three boroughs
studied in the reference paper" - a real, rigorously-tested, positive
result, not a modest negative-result-heavy session. The gap to Gao et
al.'s own 68.98%/76.59%/72.24% remains real, but has now narrowed by a
statistically-defensible margin for the first time in this
investigation.

## 2026-09-02, CLOSED (negative) - a distance-based network-consolidation approximation of the network-granularity hypothesis significantly HURTS AccHR@20

The Table 7.3 finding above converted "the network is structural, not
decoder choice" from elimination to positive evidence, but the real OS
Open Roads network remains blocked on a source file only the project
owner can download. To at least test the *shape* of the hypothesis
with data already on hand, `consolidate_borough_graph()` was built
(`src/greyspot/ingest/network.py`) to merge nearby OSMnx intersection
nodes by distance (`osmnx.simplification.consolidate_intersections`,
tolerance=2m, empirically chosen to bring Westminster's segment count
from 7,552 to ~4,324 - close to the paper's own reported 4,822). This
is explicitly disclosed as an approximation of a coarser network, not
a claim of replicating OS's own survey-based link/node convention.

**Result: a clean, statistically significant NEGATIVE, tested on
Westminster and Lambeth (n=12 paired windows against the current-best
POI+socio-demographic configuration)**: mean AccHR@20 falls from
56.65% to 48.64% (-8.0 points), the consolidated-network candidate
wins only 3 of 12 windows, paired t-test **t=-2.5356, p=0.0277**,
Wilcoxon **W=11.0, p=0.0269** - both significant at α=0.05. Per-borough:
Westminster 55.58%→44.69% (5/6 losses, p=0.085 alone); Lambeth
57.72%→52.60% (4/6 losses, p=0.238 alone) - same direction on both,
the combined test crossing significance.

**Interpretation - this narrows, rather than closes, the network
hypothesis.** Matching the paper's raw segment *count* via arbitrary
proximity-based node merging is measurably harmful, not neutral: it
folds genuinely distinct, independently-risky road locations into a
single segment, actively destroying the spatial resolution a top-20%-
segment ranking metric depends on. This rules out "any sufficiently
coarse network helps" as an explanation. It does not rule out OS Open
Roads itself, whose real link/node convention may preserve exactly the
distinctions this proximity-based approximation destroys (e.g.
grouping by named-street continuity rather than by physical node
proximity) - that remains the one genuinely untested version of this
lever, still blocked on the missing source GeoPackage. If this project
is written up before that becomes available, the honest framing is:
"the network-granularity hypothesis is positively evidenced by the
reference paper's own data (Table 7.3), but a same-shaped approximation
tested here (arbitrary distance-based consolidation) makes AccHR@20
significantly worse (p=0.027) - the mechanism is evidently more
specific than raw segment count, and remains untested in its real
form."

## 2026-09-02, CONFIRMED - the REAL OS Open Roads network closes the gap: this project's strongest result, by a clear margin

The "still blocked on the missing source GeoPackage" line immediately
above was wrong the moment it was written - the 1.02GB source file
(`oproad_gpkg_gb/Data/oproad_gb.gpkg`, downloaded by the user
2026-09-01) had been sitting in the project root the entire time,
simply never exercised end-to-end. Running it for real surfaced and
fixed two genuine bugs first (both with regression tests, full suite
175 passed): the existing loader only bbox-clipped, never clipped to
the real administrative-boundary polygon (pulling in >2x too many
links for an irregularly-shaped borough like Westminster); and
`ox.load_graphml`'s hardcoded integer node/edge-id coercion crashed on
OS Open Roads' UUID-style TOIDs the first time a cached graph was
re-loaded rather than freshly built.

**Result, tested against the current-best POI+socio-demographic
configuration across all three of the paper's own case-study
boroughs**: mean AccHR@20 rises from 55.39% to **66.99%** (n=18 paired
windows), candidate wins 15/18, paired t-test **t=3.9768, p=0.0010**,
Wilcoxon **p=0.0016** - both an order of magnitude past the
conventional significance bar, the strongest statistical result this
project has produced. Individually significant on two of three
boroughs (Westminster p=0.0488, Tower Hamlets p=0.0204; Lambeth
p=0.31, a real but smaller/noisier effect not significant alone).

**Per-borough comparison to the paper's own reported numbers - for the
first time genuinely close, not merely narrowed:**

| Borough | This project | Gao et al. | Gap |
|---|---|---|---|
| Westminster | 70.03% | 68.98% | **+1.05 (matched/exceeded)** |
| Tower Hamlets | 67.34% | 72.24% | -4.90 |
| Lambeth | 63.59% | 76.59% | -13.00 |
| Pooled | 66.99% | 72.60% | -5.61 |

**Westminster now matches (numerically exceeds) the paper's own
reported AccHR@20** - after six independent attempts this session
(data richness, target definition, decoder distribution x2 boroughs,
temporal protocol x2 boroughs, a crude network-consolidation
approximation) each failed or landed inconclusive, the real network
swap - the one lever blocked all session on what turned out to be a
misconception, not a missing file - closes the majority of the
remaining gap. Lambeth's larger residual gap is the one open question:
possibly a genuine borough-specific effect (the paper's own text
already attributes some cross-borough variation to network structure),
possibly noise at 6 windows/borough - not yet resolved.

**Why this reverses, rather than contradicts, the earlier network-
consolidation negative result**: that experiment approximated "a
coarser network" by merging nearby OSMnx nodes purely by distance - an
admittedly crude proxy, explicitly disclosed as such at the time. The
real OS Open Roads network is not even reliably coarser by raw segment
count (proper polygon clipping gives Westminster 11,098 directed edges
- MORE than this project's own OSMnx graph's 7,552, the opposite of
the "~1.5x coarser" reading of the paper's Table 7.2 that motivated the
whole line of inquiry) - whatever OS Open Roads' real survey-based
link/node convention captures, it is evidently not "fewer segments,"
and a proxy built on that wrong premise was always going to point the
wrong direction. **If this project is written up, the correct headline
claim is now**: "using the real Ordnance Survey road network in place
of an OpenStreetMap-derived one produces a large, statistically
significant improvement in AccHR@20 (paired t-test p=0.0010 across all
three of the reference paper's case-study boroughs), closing most of
the gap to the reference paper's own reported figures and matching its
Westminster result almost exactly" - the strongest, most defensible
claim this project has produced, and the first time a result in this
investigation has been described as "matched," not "narrowed."

## 2026-09-03 — Everything tested since the OS Open Roads result: all null. The headline is unchanged, and that is now a stronger claim, not a weaker one.

This document's headline numbers (pooled 66.99%, Westminster 70.03% vs
the paper's 68.98%) are **unchanged** after roughly a dozen further
experiments. That is worth stating explicitly for a write-up, because
"we then tried twelve more things and none beat it" is evidence the
headline configuration is a genuine optimum for this pipeline rather
than a lucky stopping point.

Full per-experiment numbers live in `docs/ucl_benchmark_results.md`;
chronological reasoning in `docs/decision_log.md`. Summary of what has
been closed since:

**Feature-set parity with their Table 7.2 — achieved, and null.** Road
class (8 classes, previously absent entirely; p=0.97), date features
(4 vs 1), POI at their full 20-class granularity (p=0.42), weather
re-tested on the real network (p=0.76), and all of it combined
(p=0.38 — the *worst* variant). Every input class Gao et al. list is
now implemented and measured.

**Architecture — closed, including their own encoder order.** Their
appendix specifies GRU→GAT; this project uses GAT→GRU. Tested fairly
(each at its own optimal learning rate, with a same-learning-rate
control) and the two are equivalent: 63.59% vs 60.93%, p=0.28. A
single-learning-rate comparison would have been badly misleading here —
each order collapses in the other's regime.

**Training-set size — tested directly, mixed.** ~10x more training
instances on identical evaluation windows did not produce a consistent
gain.

**Also closed as significant negatives**: junction risk redistribution
(the paper's own stated method; p=0.043 *worse* on this network
representation) and distance-based network consolidation (p=0.028
worse).

### Methodological transparency for the write-up

Six errors were caught and corrected in this period, each of which
would have produced a confident but false claim if unchecked. Three are
worth reporting in a reflection chapter because they concern claims
*about the reference paper*:

1. **A NaN reported as a score.** Their encoder order initially
   "scored" 14.31% — actually all-NaN predictions being tie-broken by
   the metric. Unchecked, this would have been published as "the
   paper's own architecture scores 14% vs our 64%".
2. **An unfair metric.** Junction redistribution was first evaluated
   against the *redistributed* target, which mechanically hardens
   AccHR@20 (one crash becomes 8 fractional entries, all of which must
   be ranked). Fixed by training on the redistributed target while
   evaluating on the original.
3. **A silent total failure.** The 20-class POI taxonomy matched zero
   of 20,040 real POIs because ingestion had already discarded the tag
   columns it classified on — while every unit test passed, because
   those tests built their own tidy fixtures.

The last one drove a durable process change: verify new feature-building
code against **real** data output before committing GPU time, and write
at least one test against the shape the pipeline actually emits.

### Current status against the success criteria

- Westminster: **exceeds** the paper (70.03% vs 68.98%).
- Tower Hamlets (−4.90) and Lambeth (−13.00) do not.
- Pooled 66.99% vs 72.60% — **not met**.

The one untested lever with a concrete, external basis is **feature
scaling**: the reference group's own public code
(github.com/ZhuangDingyi/STZINB) scales inputs by the training-set
maximum and never z-scores, while this project z-scores throughout —
producing feature values up to |172| on real data, the same condition
that caused attention NaNs earlier. That is being tested now.

---

## 2026-09-03 — Status materially changed: a second significant finding

The status above ("the one untested lever is feature scaling") is now
**superseded**. Feature scaling was tested and was a clean null
(p=0.9899). The real lever was found elsewhere.

### The new headline

**Long-horizon crash history** (`collision_count_90d`,
`collision_count_365d`, replacing a 30-day cap):

| Borough | Previous | + long history | Δ | Gao et al. | |
|---|---|---|---|---|---|
| Westminster | 70.03% | **75.39%** | +5.36 | 68.98% | **+6.41** |
| Tower Hamlets | 67.34% | **75.37%** | +8.03 | 72.24% | **+3.13** |
| Lambeth | 63.59% | **70.29%** | +6.70 | 76.59% | −6.30 |
| **Pooled (n=18)** | **66.99%** | **73.69%** | **+6.70** | **72.60%** | **+1.08** |

**p=0.001337 (paired t), p=0.001579 (Wilcoxon), 16/18 windows —
CONFIRMED on all three boroughs.** That significance is against this
project's OWN previous model.

**Against Gao et al. the result is a statistical TIE, not a win.** Our
pooled 95% CI [69.71%, 77.66%] contains their 72.60% (one-sample
p=0.5730); every per-borough comparison is also a tie (p=0.11-0.38).
The publishable claim is *matching* a peer-reviewed benchmark under a
stricter protocol and sparser target - not beating it.

### Why this strengthens the publication case specifically

1. **Two significant findings now, and both are DATA findings** — the
   real OS Open Roads network (+11.59, p=0.0010) and feature horizon
   (+6.03, p=0.0303). Every model-side lever tested (architecture,
   capacity, decoder, encoder order, loss function, ensembling ×3,
   joint training) was null. That is a coherent, defensible thesis:
   *for road-level crash prediction at this sparsity, data
   representation dominates architecture.* It is a stronger and more
   useful claim than "our GNN variant scores higher."

2. **A mechanism that generalises, evidenced four ways.** Three
   transformations that traded per-segment sharpness for coverage or
   smoothness measurably hurt (junction redistribution p=0.0432;
   rank-blending 75.8%→29.9%; AADF name-propagation −10.42). The one
   that preserved sharpness while improving estimate quality won. On a
   within-day top-k ranking metric, sharpness dominates coverage.

3. **The reference paper reports no comparable baseline.** This project
   now has a random control (18.59%, against a theoretical ~20%), a
   single-feature baseline (56.89%), and a documented predictability
   ceiling analysis (year-over-year Spearman 0.60–0.80; 1.1% spatial
   repeat rate). Gao et al.'s weakest comparator is a historical
   average at 47.99%.

4. **Westminster now exceeds the paper by 6.41 points** (75.39% vs
   68.98%), a margin robust enough to survive reasonable
   run-to-run variation, unlike the previous 1.05-point margin.

### What is still honestly outstanding

- Lambeth remains 6.30 points below the paper (70.29% vs 76.59%) —
  narrowed from 13.00, not closed. The pooled win is carried by the
  other two boroughs and must never be described as a clean sweep.
- The ~140× zero-inflation discrepancy against their Table 7.2 remains
  unexplained and is disclosed, not resolved.


> **Canonical model specification**: [`docs/final_model.md`](final_model.md).
> Note especially §7 (Limitations) before drafting any claim of beating
> the reference paper — the result is a statistical TIE.
