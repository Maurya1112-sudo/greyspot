"""Multi-seed the reference architecture — the paper's strongest claim.

**What needs settling.** The abstract states that the reference
architecture, given our data and its own tuned settings, loses to a
crash-count sort on 18 of 18 held-out windows (−19.49 points). That rests
on **single-seed** runs. Rule R10 requires multi-seeding any published
number, and this project has already watched a single-seed claim reverse
sign between seeds within one borough (S5, 2026-09-06), so the same
failure is available here.

**Why only one arm is re-run.** The head-to-head has four arms, and
multi-seeding all of them would be 360 trainings (~9 h). It is not
necessary:

- *ours LONG* is **already** multi-seeded. It is bit-identical to the
  headline configuration — verified structurally (same YEARS,
  HISTORY_YEARS, LONG_LOOKBACKS, ROLLING_WINDOWS, TABLE_START_DATE,
  START_DATE, N_WINDOWS, defaults and feature columns) and empirically
  (Lambeth and Tower Hamlets reproduce to 0.000000 against
  `headline_multiseed_per_window.csv`; Westminster differs only on the
  window already documented as non-deterministic). So
  `reports/<borough>/headline_multiseed_per_window.csv` *is* that arm at
  5 seeds.
- the two SHORT-history arms support the substitution claim, which has
  already **failed** replication (C6, sign reverses between boroughs).
  Multi-seeding a withdrawn claim buys nothing.

That leaves *theirs LONG* — 5 seeds x 6 windows x 3 boroughs = 90
trainings, roughly 2.5 h — which completes **two** claims at once:

1. reference architecture vs the trivial sort (the 18/18 claim), and
2. our architecture vs theirs at matched history (C6's surviving half),

because the other side of each comparison is already multi-seeded or
deterministic.

Seed 42 is re-run rather than reused from `headtohead_per_window.csv`, so
all five seeds come from one code state on one day. Training is not
deterministic here (see `run_determinism_probe.py`), so mixing an old
seed-42 run with four new ones would put a known source of variation
inside the arm rather than between the arms being compared.

Checkpointed per window: a shutdown costs one window.

Run: python scripts/run_headtohead_multiseed.py [Borough ...]
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.checkpoint import (  # noqa: E402
    FINGERPRINT_COLUMN,
    append_checkpoint,
    compute_fingerprint,
    load_checkpoint,
)
from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer, fit_feature_standardizer,
)
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.models.gat_temporal import (  # noqa: E402
    predict_gat_temporal, train_gat_temporal_walkforward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_headtohead_multiseed")

SEEDS = [42, 1, 7, 123, 2024]
N_WINDOWS = 6
ARM = "Gao et al. architecture, LONG history (the key test)"
DEFAULT_BOROUGHS = ["Lambeth", "Westminster", "Tower Hamlets"]


def main(boroughs: list[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "h", ROOT / "scripts" / "run_headtohead_stzitd.py")
    h = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(h)

    arm_cfg = dict(h.CANDIDATES[ARM])
    # This arm uses the FULL feature set, so no subsetting happens. Popping
    # it here (rather than leaving it for the trainer to choke on) keeps the
    # config-rebinding bug class out of reach entirely - that bug silently
    # applied a feature subset to window 1 only and invalidated three
    # results before it was found.
    wanted = arm_cfg.pop("feature_columns", None)
    assert wanted is None or list(wanted) == list(h.FEATURE_COLUMNS), (
        "this arm is expected to use the full feature set; got a %d-column subset"
        % (len(wanted) if wanted else -1))
    logger.info("arm config: %s", arm_cfg)

    for borough in boroughs:
        logger.info("=== HEAD-TO-HEAD MULTI-SEED (%s): %s ===", ARM[:28], borough)
        instances, edge_index = h.build_instances(borough)
        n = len(instances)
        out_dir = ROOT / "reports" / slug(get_borough(borough).name)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "headtohead_multiseed_checkpoint.csv"

        for seed in SEEDS:
            fp = compute_fingerprint(
                borough=borough, seed=seed, arm=ARM, config=arm_cfg,
                years=h.YEARS, history_years=h.HISTORY_YEARS,
                long_lookbacks=h.LONG_LOOKBACKS, rolling=h.ROLLING_WINDOWS,
                n_windows=N_WINDOWS, n_instances=n, features=list(h.FEATURE_COLUMNS),
            )
            done = load_checkpoint(ckpt, str(seed), fp)
            if done:
                logger.info("seed %d: %d window(s) already done - resuming", seed, len(done))
            accs = []
            for idx in range(n - N_WINDOWS, n):
                hx_raw, hy, start = instances[idx]
                if start in done:
                    accs.append(done[start]["AccHR"])
                    continue
                train_slice = instances[:idx]
                mean, std = fit_feature_standardizer([x for x, _, _ in train_slice])
                tri = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
                hx = apply_feature_standardizer(hx_raw, mean, std)
                defaults = dict(epochs=200, horizon=h.HORIZON, zero_inflated=True,
                                use_residual=True, gat_hidden=16, gru_hidden=32)
                model = train_gat_temporal_walkforward(
                    tri, edge_index, **{**defaults, **arm_cfg}, seed=seed,
                )
                acc = float(accuracy_hit_rate(hy, predict_gat_temporal(model, hx, edge_index).T,
                                              top_fraction=0.20))
                accs.append(acc)
                append_checkpoint(ckpt, {
                    "candidate": str(seed), "seed": seed, "borough": borough, "arm": ARM,
                    "held_out_start": start, "AccHR": acc,
                    "n_train_instances": len(train_slice), FINGERPRINT_COLUMN: fp,
                })
                logger.info("  seed %-5d window %s: AccHR@20=%.6f", seed, start.date(), acc)
            logger.info("seed %-5d 6-window mean AccHR@20 = %.6f", seed, float(np.mean(accs)))

        rows = pd.read_csv(ckpt)
        per_seed = rows.groupby("seed").AccHR.mean()
        logger.info("=== %s theirs-LONG: %d seeds, mean=%.4f sd=%.4f range=%.4f-%.4f ===",
                    borough, len(per_seed), per_seed.mean(), per_seed.std(ddof=1),
                    per_seed.min(), per_seed.max())
        rows.to_csv(out_dir / "headtohead_multiseed_per_window.csv", index=False)
        logger.info("Written to %s", out_dir / "headtohead_multiseed_per_window.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_BOROUGHS)
