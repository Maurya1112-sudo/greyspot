"""S5 multi-seeded: can the model actually use history beyond five years?

**Why this is needed.** S5 measured a −0.72 change on Lambeth and +2.98 on
Westminster from extending crash-history features 5yr → 9yr. Both figures
are *inside* the ~4-point seed-noise band established by V8/V11, so at a
single seed neither borough settles anything — the sign flip between them
may be nothing but seed noise. Rule R10 requires any published number to be
multi-seeded per borough, and the claim built on these ("the model cannot
exploit deeper history") was in the preprint abstract until it was
withdrawn on 2026-09-05.

**Design.** Only the DEEP arm is run here. The 5-year baseline is already
multi-seeded — `reports/run_logs/v8_*.log` holds 5 seeds × 6 windows per
borough — so re-running it would burn an hour reproducing numbers that
exist. Same seeds, same windows, same config as V8 apart from the history
depth, which is the single variable (R1).

**Per-window output.** V8 recorded only per-seed MEANS, which made a paired
test against the baseline impossible without re-parsing logs. This writes
every (seed, window) score, so the comparison is a straightforward merge.

Estimated cost: 2 boroughs × 5 seeds × 6 windows ≈ 60 trainings, ~1.5 h.

Run: python scripts/run_s5_multiseed.py [Borough ...]
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
logger = logging.getLogger("run_s5_multiseed")

SEEDS = [42, 1, 7, 123, 2024]  # identical to V8, so the arms are paired on seed too
N_WINDOWS = 6
DEFAULT_BOROUGHS = ["Lambeth", "Westminster"]


def main(boroughs: list[str]) -> None:
    # Load the S5 script as a module and reuse its build_instances, so the
    # deep-history feature construction is literally the same code rather
    # than a copy that can drift.
    spec = importlib.util.spec_from_file_location("s5", ROOT / "scripts" / "run_s5_deep_history.py")
    s5 = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(s5)
    assert 3285 in s5.LONG_LOOKBACKS, "expected the 9-year lookback in the S5 config"

    for borough in boroughs:
        logger.info("=== S5 MULTI-SEED: %s ===", borough)
        instances, edge_index = s5.build_instances(borough)
        n = len(instances)
        out_dir = ROOT / "reports" / slug(get_borough(borough).name)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "s5_multiseed_checkpoint.csv"

        for seed in SEEDS:
            fp = compute_fingerprint(
                borough=borough, seed=seed, lookbacks=s5.LONG_LOOKBACKS,
                history_years=s5.HISTORY_YEARS, rolling=s5.ROLLING_WINDOWS,
                n_windows=N_WINDOWS, n_instances=n, features=s5.FEATURE_COLUMNS,
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
                    tri, edge_index, epochs=200, horizon=s5.HORIZON, zero_inflated=True,
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
                logger.info("  seed %-5d window %s: AccHR@20=%.4f", seed, start.date(), acc)
            logger.info("seed %-5d 6-window mean AccHR@20 = %.4f  (windows: %s)",
                        seed, float(np.mean(accs)), " ".join("%.4f" % a for a in accs))

        rows = pd.read_csv(ckpt)
        per_seed = rows.groupby("seed").AccHR.mean()
        logger.info("=== %s deep-history, %d seeds: mean=%.4f sd=%.4f range=%.4f-%.4f ===",
                    borough, len(per_seed), per_seed.mean(), per_seed.std(ddof=1),
                    per_seed.min(), per_seed.max())
        rows.to_csv(out_dir / "s5_multiseed_per_window.csv", index=False)
        logger.info("Written to %s", out_dir / "s5_multiseed_per_window.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_BOROUGHS)
