"""Is training deterministic at a fixed seed, and is Westminster different?

**The observation.** Regenerating the headline at 5 seeds reproduced
Lambeth EXACTLY - all 30 windows within the V8 log's 3dp rounding. On
Westminster two windows moved:

| seed | window | log | regenerated | difference |
|---|---|---|---|---|
| 42 | 2023-10-13 | 0.8110 | 0.817177 | +0.006 (a metric tie step) |
| 1 | 2024-10-07 | 0.8450 | 0.755128 | **-0.090** |

The first is explained: AccHR@20 moves in steps of 1/(crashes that
day)/(days), and that window's steps are ~0.012-0.018. The second is not.
That window holds 46 crashes over 13 days, so its largest single step is
0.0769 - a 0.090 move needs between 2 and 7 crashes changing side. That is
a materially different model, not a metric artefact.

**What this script tests.** Train the SAME window at the SAME seed N times
and compare. Three outcomes:

  - identical every time -> training is deterministic, and the difference
    came from something other than the run (a changed input, a different
    code state) which then has to be found.
  - varies by exact metric steps -> the model is stable and only the
    ranking's tie order moves.
  - varies by more than one step -> training itself is non-deterministic
    on this borough, which is a reportable limitation: it means a
    same-seed rerun can produce a materially different model, and every
    per-borough figure inherits that.

Westminster is already known to be the unstable borough - the 2-layer
ablation diverged there on 2 of 6 windows while behaving on Lambeth - so a
borough-specific answer is plausible rather than surprising.

**A mechanism that would explain it, stated as a hypothesis for this probe
to test rather than as a conclusion.** The model uses
`torch_geometric.nn.GATConv`, whose neighbourhood aggregation is
scatter-based; scatter/atomicAdd accumulation order on CUDA is not fixed,
so identical inputs can produce floating-point differences in the last
bits. Nothing in this project pins that down - measured on the current
environment (torch 2.6.0+cu124, pyg 2.8.0):

    torch.are_deterministic_algorithms_enabled()  False
    torch.backends.cudnn.deterministic            False
    CUBLAS_WORKSPACE_CONFIG                       unset

`torch.manual_seed` does seed CUDA, so *initialisation* is reproducible;
the aggregation order is not. Over 200 epochs those differences compound,
usually to nothing visible and occasionally - near a decision boundary -
to a different local optimum.

That is a plausible story, and this project has already been burned once
today by asserting a plausible story about run-to-run differences and
having to withdraw it. So it is written here as the hypothesis the probe
tests, not as the answer. If training proves deterministic, the story is
wrong and the cause is elsewhere.

Run: python scripts/run_determinism_probe.py [n_repeats]
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

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer, fit_feature_standardizer,
)
from greyspot.models.gat_temporal import (  # noqa: E402
    predict_gat_temporal, train_gat_temporal_walkforward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_determinism_probe")

CASES = [
    ("Westminster", pd.Timestamp("2024-10-07"), 1),    # moved by 0.090
    ("Westminster", pd.Timestamp("2023-10-13"), 42),   # moved by one metric step
    ("Lambeth", pd.Timestamp("2024-10-07"), 1),        # reproduced exactly - control
]


def main(n: int = 3) -> None:
    rows = []
    for borough, window, seed in CASES:
        spec = importlib.util.spec_from_file_location(
            "m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
        m = importlib.util.module_from_spec(spec)
        sys.argv = ["x", borough]
        spec.loader.exec_module(m)
        instances, edge_index = m.build_instances(borough)
        idx = next(i for i, (_, _, s) in enumerate(instances) if s == window)
        hx_raw, hy, _ = instances[idx]
        train_slice = instances[:idx]
        mean, std = fit_feature_standardizer([x for x, _, _ in train_slice])
        tri = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
        hx = apply_feature_standardizer(hx_raw, mean, std)

        vals = []
        for rep in range(1, n + 1):
            model = train_gat_temporal_walkforward(
                tri, edge_index, epochs=200, horizon=m.HORIZON, zero_inflated=True,
                use_residual=True, gat_hidden=16, gru_hidden=32, heads=3, gat_layers=1,
                seed=seed,
            )
            v = float(accuracy_hit_rate(hy, predict_gat_temporal(model, hx, edge_index).T,
                                        top_fraction=0.20))
            vals.append(v)
            logger.info("%s %s seed %d, repeat %d/%d: %.6f", borough, window.date(), seed, rep, n, v)
        spread = max(vals) - min(vals)
        rows.append({"borough": borough, "window": window.date(), "seed": seed,
                     "n": n, "min": min(vals), "max": max(vals), "spread": spread,
                     "values": " ".join("%.6f" % v for v in vals)})
        logger.info("=== %s %s seed %d: spread %.6f over %d repeats ===",
                    borough, window.date(), seed, spread, n)

    df = pd.DataFrame(rows)
    print()
    print("=" * 78)
    print("DETERMINISM PROBE: same window, same seed, %d repeats" % n)
    print("=" * 78)
    print("%-14s %-12s %5s %10s  %s" % ("borough", "window", "seed", "spread", "values"))
    for _, r in df.iterrows():
        print("%-14s %-12s %5d %10.6f  %s" % (r.borough, r.window, r.seed, r.spread, r["values"]))
    print()
    if (df.spread < 1e-9).all():
        print("VERDICT: training is DETERMINISTIC at a fixed seed on every case.")
        print("  The observed differences came from something other than rerunning,")
        print("  and that cause must be found before the figures are trusted.")
    else:
        worst = df.loc[df.spread.idxmax()]
        print("VERDICT: training is NOT deterministic at a fixed seed.")
        print("  Worst: %s %s seed %d, spread %.6f over %d repeats."
              % (worst.borough, worst.window, worst.seed, worst.spread, n))
        print("  This is a reportable limitation: a same-seed rerun can produce a")
        print("  materially different model, and every per-borough figure inherits it.")
    out = ROOT / "reports" / "determinism_probe.csv"
    df.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
