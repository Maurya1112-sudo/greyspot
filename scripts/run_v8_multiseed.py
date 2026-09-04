"""V8 - multi-seed check on the headline number.

Every AccHR@20 figure in this project comes from a SINGLE training run
at seed=42. An earlier control showed CUDA non-determinism does not move
the metric (5 repeats, identical AccHR), but that tested repeated runs at
the SAME seed - it says nothing about how much the result depends on the
particular initialisation.

That matters because three candidate improvements have already been
rejected for failing to replicate across boroughs, and because the
measured reproducibility floor (+/-0.2-0.3 points across identical
configs) was itself derived from same-seed runs.

This trains the final configuration at five different seeds on the same
six Lambeth windows and reports mean +/- std of the 6-window mean. If
seed variance is small, the headline figures stand as reported. If it is
large, every number in this project needs a +/- attached, and small
effects (the hidden-size -2.87, the decoder -3.50) become
uninterpretable.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    fit_feature_standardizer,
)
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_v8_multiseed")

SEEDS = [42, 1, 7, 123, 2024]
N_WINDOWS = 6


def main(borough: str = "Lambeth") -> None:
    spec = importlib.util.spec_from_file_location("m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["x", borough]
    spec.loader.exec_module(m)

    instances, edge_index = m.build_instances(borough)
    n = len(instances)
    per_seed = []
    for seed in SEEDS:
        accs = []
        for held_out_idx in range(n - N_WINDOWS, n):
            train_slice = instances[:held_out_idx]
            hx_raw, hy, start = instances[held_out_idx]
            mean, std = fit_feature_standardizer([x for x, _, _ in train_slice])
            tri = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
            hx = apply_feature_standardizer(hx_raw, mean, std)
            mod = train_gat_temporal_walkforward(
                tri, edge_index, epochs=200, horizon=m.HORIZON, zero_inflated=True,
                use_residual=True, gat_hidden=16, gru_hidden=32, heads=3, gat_layers=1,
                seed=seed,
            )
            p = predict_gat_temporal(mod, hx, edge_index)
            accs.append(accuracy_hit_rate(hy, p.T, top_fraction=0.20))
        per_seed.append(float(np.mean(accs)))
        logger.info("seed %-5d 6-window mean AccHR@20 = %.4f  (windows: %s)",
                    seed, per_seed[-1], " ".join("%.3f" % a for a in accs))

    a = np.array(per_seed)
    print()
    print("=== V8 MULTI-SEED CHECK (%s, final config) ===" % borough)
    for s_, v in zip(SEEDS, per_seed):
        print("  seed %-5d %.4f" % (s_, v))
    print()
    print("  mean %.4f   std %.4f   range %.4f-%.4f   SPREAD %.2f points"
          % (a.mean(), a.std(ddof=1), a.min(), a.max(), 100 * (a.max() - a.min())))
    print()
    print("  headline as reported (seed 42): 0.7944")
    print("  -> if spread is small, headline figures stand; if large, every")
    print("     number in this project needs a +/- and small effects are noise.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Lambeth")
