"""Multi-seed the encoder-order ablation — the most seed-exposed survivor.

**Why this one.** After the reference-architecture run showed that seed 42
gave the single most favourable value of five (inflating one published
effect by ~9 points), the remaining single-seed claims were ranked by
exposure: effect size against the ~4-point seed band measured for OUR
architecture.

| claim | effect | ratio to band |
|---|---|---|
| layers 1→2 | −26.46 | 6.6x — safe |
| history horizon | +15.87 | 3.9x |
| **encoder order** | **−13.61** | **3.4x — most exposed survivor** |

Encoder order is the narrowest margin among the claims that survived
replication, so it is where a seed-selection effect could still do damage.

**Only one arm is run.** The comparison is our architecture with
`encoder_order="temporal_first"` (their ordering) against our default
`spatial_first`. The default arm is **already** multi-seeded — it is the
headline configuration, so `headline_multiseed_per_window.csv` serves
directly, exactly as it did for the head-to-head run. That halves the
cost: 5 seeds x 6 windows x 2 boroughs = 60 trainings, ~1.5 h.

**Learning rate.** The ablation compared their ordering at its OWN swept
optimum, because at the shared learning rate it collapses to 14.31% and
the comparison would measure a tuning failure rather than an ordering
effect. That choice is preserved here (`lr=5e-4`), so this is the same
comparison with seeds added and nothing else changed (R1).

Run: python scripts/run_encoder_multiseed.py [Borough ...]
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
    FINGERPRINT_COLUMN, append_checkpoint, compute_fingerprint, load_checkpoint,
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
logger = logging.getLogger("run_encoder_multiseed")

SEEDS = [42, 1, 7, 123, 2024]
N_WINDOWS = 6
DEFAULT_BOROUGHS = ["Lambeth", "Westminster"]
# our architecture, their encoder ordering, at the ordering's own swept lr
ARM = dict(heads=3, gat_layers=1, negative_binomial=False,
           encoder_order="temporal_first", lr=5e-4)


def main(boroughs: list[str]) -> None:
    spec = importlib.util.spec_from_file_location(
        "m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(m)
    logger.info("arm: %s", ARM)

    for borough in boroughs:
        logger.info("=== ENCODER-ORDER MULTI-SEED: %s ===", borough)
        instances, edge_index = m.build_instances(borough)
        n = len(instances)
        out_dir = ROOT / "reports" / slug(get_borough(borough).name)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / "encoder_multiseed_checkpoint.csv"

        for seed in SEEDS:
            fp = compute_fingerprint(
                borough=borough, seed=seed, arm=ARM, years=m.YEARS,
                history_years=m.HISTORY_YEARS, long_lookbacks=m.LONG_LOOKBACKS,
                rolling=m.ROLLING_WINDOWS, n_windows=N_WINDOWS,
                n_instances=n, features=m.FEATURE_COLUMNS,
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
                defaults = dict(epochs=200, horizon=m.HORIZON, zero_inflated=True,
                                use_residual=True, gat_hidden=16, gru_hidden=32)
                model = train_gat_temporal_walkforward(
                    tri, edge_index, **{**defaults, **ARM}, seed=seed)
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
        ps = rows.groupby("seed").AccHR.mean()
        logger.info("=== %s temporal_first: %d seeds, mean=%.4f sd=%.4f range=%.4f-%.4f ===",
                    borough, len(ps), ps.mean(), ps.std(ddof=1), ps.min(), ps.max())
        rows.to_csv(out_dir / "encoder_multiseed_per_window.csv", index=False)
        logger.info("Written to %s", out_dir / "encoder_multiseed_per_window.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_BOROUGHS)
