# Supervision & work log

*A dated record of decisions, supervisor discussions and significant work
sessions — evidence of independent, ongoing project management (L6.05) and
useful raw material for the dissertation's reflection chapter. Add an
entry every time you meet your supervisor, and briefly whenever a solo
session changes the project's direction. Keep entries short; the detailed
reasoning belongs in `docs/decision_log.md`.*

## Template

```
## YYYY-MM-DD — [Supervisor meeting / Solo session] — <one-line topic>

**Discussed / done:**
-

**Decisions:**
-

**Actions before next session:**
-
```

---

## 2026-08-31 — Solo session (pre-term) — Project kickoff

**Discussed / done:**
- Read the full 33-page Greyspot: Government Edition project dossier.
- Verified the dossier's factual claims (STATS19, OS Open Roads, TfL Vision
  Zero Action Plan 2, MAPIE, the Gao et al. 2024 academic precedent, IMD
  2019) against live sources — all confirmed accurate.
- Scaffolded a Python research-core repository from scratch.
- Built and tested the full pipeline: STATS19 + OSMnx + IMD ingestion,
  leakage-safe feature engineering, historical-rate/XGBoost/GAT+GRU models,
  temporal/spatial/spatiotemporal evaluation, GAT ablations (no-graph,
  no-temporal), conformal uncertainty (MAPIE for XGBoost, manual
  split-conformal for the GAT).
- Verified University of Westminster's actual public programme
  specification for 6COSC023W (Computer Science Final Project, 40 credits,
  Level 6) and mapped the project against its published learning outcomes.
- Built the initial documentation suite (this file plus proposal,
  literature review, methodology, requirements/ethics, project management,
  results).

**Decisions:**
- Python-only first; defer FastAPI/PostGIS/React until the research model
  is validated (matches the dossier's own scope-control rule).
- OSMnx instead of OS Open Roads for now (account-signup friction);
  swappable later.
- Report the GAT-vs-XGBoost result honestly as non-dominant rather than
  reframing it — full reasoning in `docs/decision_log.md`.

**Actions before next session:**
- Get the live Blackboard 6COSC023W module handbook, rubric and deadlines.
- Confirm supervisor and get sign-off on the proposal and scope.
- Manually obtain OS Open Roads access and the ONS LSOA crosswalk (both
  blocked from automated download).

## 2026-09-02 — Solo session — Real OS Open Roads network closes most of the UCL benchmark gap

**Discussed / done:**
- Continued the UCL/Gao et al. AccHR@20 benchmark investigation. Tested
  and closed several hypotheses: network consolidation (significantly
  negative, p=0.0277), architecture sweep on top of POI+socio features
  (null, p=0.56), dense/paper-matching temporal protocol (robustness
  confirmation, not an improvement), TCR target on the real network
  (null, p=0.20).
- **Discovered a stale, incorrect project note** claiming the OS Open
  Roads GeoPackage was missing from disk. It had been present since
  2026-09-01 and had simply never been exercised end-to-end.
- Running it surfaced and fixed two real bugs (bbox-only clipping
  instead of clipping to the real borough polygon; integer coercion of
  OS's UUID-style node/edge IDs on cache reload). Both now have
  regression tests; suite at 175 passing.
- **Result: the single largest improvement of the whole
  investigation.** Real OS Open Roads network vs OSMnx: pooled AccHR@20
  55.39% -> 66.99% across all three of the paper's own boroughs (n=18
  paired windows), paired t-test p=0.0010, Wilcoxon p=0.0016.
  Westminster reaches 70.03% vs the reference paper's own 68.98% —
  the first configuration in this project to match/exceed the paper on
  any borough.
- Confirmed robust across temporal protocol (dense protocol: 6/6
  windows, p=0.0041 on Westminster; all three boroughs land in a tight
  60–63% band).
- Verified directly from the paper/thesis text that Gao et al. modelled
  **three boroughs only** (Westminster, Lambeth, Tower Hamlets) — there
  is no London-wide version of STZITD-GNN to benchmark against. (The
  thesis's London-wide chapter uses a different model, SMA-Hyper, at
  MSOA rather than road-segment level.)
- Wrote `docs/ucl_benchmark_results.md` — a full results compendium
  consolidating every AccHR@20 experiment, the complete
  what-worked/what-didn't ledger with significance tests, and all
  disclosed data substitutions and caveats.

**Decisions:**
- The UCL-benchmark pipeline now uses **real OS Open Roads**, not
  OSMnx. The annual pipeline keeps OSMnx (unchanged scope).
- Best-known configuration updated to: heads=3 + POI +
  socio-demographic + real OS Open Roads network, ZIP decoder,
  plain-count target.
- Adopted a "test one borough first, expand only if promising" policy
  for new hypotheses, to avoid spending GPU time on three-borough
  sweeps of ideas that fail on the first borough.
- Documented a real reproducibility caveat: training is not
  bit-deterministic on CUDA (no `use_deterministic_algorithms`), so
  individual percentages are one-run results. Paired significance tests
  are unaffected.

**Actions before next session:**
- Close (or honestly report) Lambeth's remaining 13-point gap — the
  thesis attributes its own strong Lambeth result to the ZI-Tweedie
  decoder handling extreme sparsity, and Lambeth is confirmed the
  sparsest borough in this project's data too.
- Consider multi-seed ensembling (implemented, not yet run) as
  variance reduction given noisy 6-window estimates.

---

## Session 2026-09-03 — significant improvement found; ~20 experiments run

**Headline for discussion:** the AccHR@20 gap narrowed substantially,
via a data finding rather than an architecture change.

| Borough | Was | Now | Gao et al. |
|---|---|---|---|
| Lambeth | 63.59% | **70.29%** | 76.59% |
| Westminster | 70.03% | **75.39%** | 68.98% ✅ |
| Pooled (n=12) | 66.81% | **72.84%** (p=0.0303) | 72.60% |

**What changed:** crash-history features were capped at 30 days; adding
90- and 365-day rolling counts gained +6.03 points pooled, 10/12
windows. Found by analysing the target rather than the model (crashes
almost never repeat spatially — 1.1% coordinate repeat rate over three
years; year-over-year rank Spearman only 0.60–0.80), which implied 30
days is far too sparse to estimate segment risk.

**What did NOT work** (all documented with p-values): architecture
ensembling, joint multi-borough training, two metric-aware ranking
losses, 8–17× denser training data, a learning-rate sweep, feature
pruning, and an AADF coverage fix that measurably hurt (−10.42 points).

**Points for supervisor discussion:**
1. Is "data representation dominates architecture" a strong enough
   central claim for the dissertation? Two significant findings support
   it and roughly a dozen model-side nulls corroborate it.
2. The reference paper reports zero-inflation of 95.72%/96.71%/96.28%;
   this project measures 99.97% on the same STATS19 source. Their TCR
   formula, segment consolidation and evaluation protocol have each been
   checked and cannot explain a ~140× difference in positive rate.
   **A data request to the authors has been drafted for me to send** —
   their published Data Availability statement offers data on request.
   Worth advice on whether to send it and whether to cc you.
3. Cross-borough replication has now caught two false positives that a
   single-borough protocol would have accepted. Suggest this becomes an
   explicit methodological contribution in the write-up.

**Actions before next session:**
- Complete Tower Hamlets replication (running) for the n=18 claim.
- If it holds, re-run the remaining boroughs with the new feature set.
- Decide on sending the data request.
