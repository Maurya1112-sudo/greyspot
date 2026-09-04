"""Is the final model overfitting? Three independent probes.

**Why this is a live concern.** The final configuration uses 35 feature
columns fitted from only 6-11 walk-forward INSTANCES per window. That
ratio looks alarming. The counter-argument is that the loss is computed
per NODE, so each instance supplies ~11,600 supervised targets - but
that argument only covers spatial variety, not temporal: the model still
sees very few distinct points in time, and every static feature (POI,
IMD, length) is IDENTICAL across all of them. Static columns therefore
have an effective sample size closer to the instance count than to the
node count.

There is also an unfollowed hint in this project's own history: the
"full Table 7.2 parity" run, which added every feature class the
reference paper lists, was the WORST variant of that session. That is
what overfitting looks like, and it was recorded as a null rather than
investigated.

Three probes, each answering a different question:

1. **Train-vs-held-out loss gap.** Train the final config and report
   training loss alongside held-out AccHR per window. A widening gap as
   instances accumulate would indicate memorisation.

2. **Feature-count ladder.** Score the model at 10, 20, 30 and 35
   features (dropping least-important groups first, by a fixed a-priori
   ordering - NOT by test performance, which would be selection on the
   evaluation set). If fewer features score as well or better, the extra
   columns are noise.

3. **Label-shuffle control.** Retrain with the TARGET randomly permuted
   across segments, keeping features intact. A model that cannot overfit
   should collapse to the ~20% random baseline. Anything materially
   above that means the model is fitting structure that survives label
   destruction - i.e. exploiting something other than the crash signal.

Probe 3 is the strongest of the three and is standard practice in
learning-to-rank; this project has never run it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util  # noqa: E402
import numpy as np  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    fit_feature_standardizer,
)
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("diagnose_overfitting")

# Fixed a-priori feature ordering, most-to-least justified. Chosen BEFORE
# seeing any result, so the ladder is not selection on the test set.
LADDER = {
    10: ["length", "day_of_week", "u_degree", "v_degree", "collision_count",
         "collision_count_7d", "collision_count_30d", "collision_count_365d",
         "collision_count_1095d", "collision_count_1825d"],
    20: None,   # + casualty breakdown, AADF, POI  (filled at runtime)
    30: None,   # + socio-demographic
    35: None,   # everything (the final model)
}


def main(borough: str = "Lambeth") -> None:
    spec = importlib.util.spec_from_file_location("m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["x", borough]
    spec.loader.exec_module(m)

    inst, ei = m.build_instances(borough)
    cols = list(m.FEATURE_COLUMNS)
    LADDER[35] = cols
    LADDER[30] = [c for c in cols if not c.startswith("imd_") and c not in ("population_density", "has_socio_demographic")][:30]
    LADDER[20] = [c for c in cols if c in LADDER[10] or c.startswith(("n_", "aadf_", "poi_")) or c in ("has_aadf", "has_poi")][:20]

    ho = 7  # a mid-series window with 7 training instances
    print()
    print("=== PROBE 2: feature-count ladder (window %s) ===" % inst[ho][2].date())
    for k in sorted(LADDER):
        sel = LADDER[k]
        idx = [cols.index(c) for c in sel]
        tr = [(x[:, :, idx], y, s) for x, y, s in inst[:ho]]
        hx_raw, hy, _ = inst[ho]
        mean, std = fit_feature_standardizer([x for x, _, _ in tr])
        tri = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in tr]
        hx = apply_feature_standardizer(hx_raw[:, :, idx], mean, std)
        mod = train_gat_temporal_walkforward(tri, ei, epochs=200, horizon=14, zero_inflated=True,
                                             use_residual=True, gat_hidden=16, gru_hidden=32,
                                             heads=3, gat_layers=1)
        p = predict_gat_temporal(mod, hx, ei)
        print("  %2d features -> AccHR@20 = %.4f" % (len(sel), accuracy_hit_rate(hy, p.T, top_fraction=0.20)))

    print()
    print("=== PROBE 3: label-shuffle control (window %s) ===" % inst[ho][2].date())
    print("  targets permuted across segments; features intact.")
    print("  a non-overfitting model should collapse to the ~20%% random baseline.")
    rng = np.random.default_rng(0)
    idx = [cols.index(c) for c in cols]
    tr_raw = inst[:ho]
    hx_raw, hy, _ = inst[ho]
    mean, std = fit_feature_standardizer([x for x, _, _ in tr_raw])
    for label, shuffle in [("real targets", False), ("SHUFFLED targets", True)]:
        tri = []
        for x, y, _ in tr_raw:
            yy = y.copy()
            if shuffle:
                perm = rng.permutation(yy.shape[0])
                yy = yy[perm]
            tri.append((apply_feature_standardizer(x, mean, std), yy.T))
        hx = apply_feature_standardizer(hx_raw, mean, std)
        mod = train_gat_temporal_walkforward(tri, ei, epochs=200, horizon=14, zero_inflated=True,
                                             use_residual=True, gat_hidden=16, gru_hidden=32,
                                             heads=3, gat_layers=1)
        p = predict_gat_temporal(mod, hx, ei)
        print("  %-18s AccHR@20 = %.4f" % (label, accuracy_hit_rate(hy, p.T, top_fraction=0.20)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Lambeth")
