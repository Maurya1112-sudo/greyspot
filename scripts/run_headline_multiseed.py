"""Regenerate the headline multi-seed figures with per-window output.

**Why this exists.** The paper's headline table (80.14% pooled, 5 seeds,
per-borough CIs) currently traces to `reports/run_logs/v8_*.log` — run
logs, parsed with a regex, recording per-window scores to four decimal
places. Two problems with that as the provenance for a paper's most
prominent number:

1. **A log is not an artefact.** `run_v8_multiseed.py` saved only per-seed
   MEANS; the per-window values exist solely as formatted text inside a
   log line. Rule R12 asks for a script that regenerates any quoted
   number, and this one regenerates only via log-scraping.

2. **One window in the logs cannot be reproduced.** Westminster
   2023-10-13, seed 42, reads 0.8110 in the V8 log. The committed
   per-window CSV says 0.7993, and re-running that exact window on current
   code (`run_reproducibility_check.py`, 2026-09-06) also gives 0.7993.
   The V8 run sits between two runs that agree with each other, so it is
   the outlier. The mechanism was not identified. For the other four seeds
   there is no independent artefact to check against at all.

The sensitivity analysis in `run_s2_paired_comparison.py` shows no
conclusion depends on that window. But "no conclusion depends on it" is a
weaker statement than "the numbers are reproducible", and the headline
table deserves the stronger one.

This re-runs the same configuration on current code, writing every
(seed, window) score to CSV, checkpointed so it survives interruption.
Cost: 3 boroughs × 5 seeds × 6 windows = 90 trainings, ~2.5 h.

Run: python scripts/run_headline_multiseed.py [Borough ...]
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
    apply_feature_standardizer,
    fit_feature_standardizer,
)
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.models.gat_temporal import (  # noqa: E402
    predict_gat_temporal,
    train_gat_temporal_walkforward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_headline_multiseed")

SEEDS = [42, 1, 7, 123, 2024]  # same as V8, so old and new are directly comparable
N_WINDOWS = 6
DEFAULT_BOROUGHS = ["Lambeth", "Westminster", "Tower Hamlets"]


def main(boroughs: list[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(m)

    for borough in boroughs:
        logger.info("=== HEADLINE MULTI-SEED: %s ===", borough)
        instances, edge_index = m.build_instances(borough)
        n = len(instances)
        out_dir = ROOT / "reports" / slug(get_borough(borough).name)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "headline_multiseed_checkpoint.csv"

        for seed in SEEDS:
            fp = compute_fingerprint(
                borough=borough, seed=seed, rolling=m.ROLLING_WINDOWS,
                long_lookbacks=m.LONG_LOOKBACKS, history_years=m.HISTORY_YEARS,
                n_windows=N_WINDOWS, n_instances=n, features=m.FEATURE_COLUMNS,
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
                model = train_gat_temporal_walkforward(
                    tri, edge_index, epochs=200, horizon=m.HORIZON, zero_inflated=True,
                    use_residual=True, gat_hidden=16, gru_hidden=32, heads=3, gat_layers=1,
                    seed=seed,
                )
                acc = float(accuracy_hit_rate(hy, predict_gat_temporal(model, hx, edge_index).T,
                                              top_fraction=0.20))
                accs.append(acc)
                append_checkpoint(ckpt, {
                    "candidate": str(seed), "seed": seed, "borough": borough,
                    "held_out_start": start, "AccHR": acc,
                    "n_train_instances": len(train_slice), FINGERPRINT_COLUMN: fp,
                })
                logger.info("  seed %-5d window %s: AccHR@20=%.6f", seed, start.date(), acc)
            logger.info("seed %-5d 6-window mean AccHR@20 = %.6f", seed, float(np.mean(accs)))

        rows = pd.read_csv(ckpt)
        per_seed = rows.groupby("seed").AccHR.mean()
        logger.info("=== %s: %d seeds, mean=%.4f sd=%.4f (sample) range=%.4f-%.4f ===",
                    borough, len(per_seed), per_seed.mean(), per_seed.std(ddof=1),
                    per_seed.min(), per_seed.max())
        rows.to_csv(out_dir / "headline_multiseed_per_window.csv", index=False)
        logger.info("Written to %s", out_dir / "headline_multiseed_per_window.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_BOROUGHS)
