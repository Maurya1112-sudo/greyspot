# SOP paragraph — Greyspot

**Purpose.** A paragraph about this project for a Master's statement of
purpose. Every sentence must be defensible line by line, because an
interviewer may ask about any of it and the honest version is stronger
than the impressive one.

---

## The paragraph (≈180 words)

> I set out to beat a published graph neural network for road-level crash
> prediction (Gao et al., *Accident Analysis & Prevention*, 2024) and
> spent most of the project discovering why that framing was wrong.
> Rebuilding their pipeline independently from UK STATS19 collision
> records and Ordnance Survey road geometry, I found that seven of eleven
> findings that looked significant on one London borough failed to
> replicate on a second — four of them by reversing sign rather than
> shrinking, so a single-region estimate had the direction wrong, not just
> the magnitude. Testing against a deliberately trivial baseline was more
> uncomfortable still: ranking road segments by how often they had already
> crashed matched my model and beat the reference architecture on 18 of 18
> held-out windows. Sweeping that baseline across lookback horizons showed
> why — it spans 23% to 84% accuracy on nothing but how far back it looks,
> and the published results sit at horizons their single-year dataset
> could not exceed. I want to do graduate work on evaluation methodology,
> because this project convinced me that is where the errors live.

---

## Shorter variant (≈95 words)

> Rebuilding a published graph neural network for road-level crash
> prediction, I found that seven of eleven findings significant on one London
> borough failed to replicate on a second — four by reversing sign, not
> merely shrinking. A parameter-free baseline that ranks road segments by
> past crash count matched my model and beat the reference architecture on
> 18 of 18 held-out windows; sweeping its lookback horizon showed it spans
> 23% to 84% accuracy on that variable alone. The negative results proved
> more transferable than the benchmark improvement I set out to make.

---

## What each claim rests on

| Claim in the paragraph | Evidence |
|---|---|
| "seven of eleven … failed to replicate" | `scripts/check_effect_size_heuristic.py`; MASTER_PLAN §2b scoreboard |
| "four … by reversing sign" | road class (−6.87 → +2.25), rank scaling (+5.80 → −39.7), architecture×history, feature count (−0.99/−4.08 → +0.77) |
| "matched my model" | `run_s2_paired_comparison.py`: −0.85, p=0.6398, 9/18 windows at matched horizon |
| "beat the reference architecture on 18 of 18" | `run_reference_vs_trivial.py`: 64.45% vs 83.94%, p=0.000001, 0/18 wins |
| "spans 23% to 84%" | `run_baseline_horizon_curve.py`: 22.71% at 30 days, 83.94% at 9 years |
| "their single-year dataset" | Gao et al. use STATS19 2019 only; their Historical Average scores 0.4496 |

## Lines deliberately NOT used

- **"I beat a published model."** True on two of three boroughs
  (80.14% vs 72.60% pooled) but the protocols differ, no paired test
  against them is possible, and the same work shows a trivial baseline
  beats mine. Leading with it invites exactly the question that undoes it.
- **"I showed GNNs don't work for crash prediction."** Not supported. The
  finding is about one architecture family on one city with this
  evaluation, and the horizon comparison against their data is suggestive,
  not controlled.
- **"I found a bug in their paper."** No. Their published learning rate
  diverges on *our* data, which is a reproducibility observation about
  transfer, not an error on their part.
- **Any single-seed number.** Seed spread is 3.2–4.7 points and its
  direction is borough-specific.

## If asked "so what went wrong?"

The honest answer is a good one: the project's own errors are documented
in `docs/decision_log.md` and several were caught only by checks built
after the fact — a config bug that silently disabled feature subsetting
after the first window, a sign error in a headline comparison, a claim
about the model's limits withdrawn once a second borough was run, and a
data download that silently returned 24% of the real POI data while
looking valid. Each is recorded with what it cost and what check now
prevents it. That record is the part worth talking about.
