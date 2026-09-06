"""Is the committed per-window CSV reproducible on current code?

An open question from 2026-09-05: Westminster's window 2023-10-13 differs
by 0.0117 between the V8 run log (0.8110) and the committed per-window CSV
(0.7993), while the other five windows agree to four decimal places.

Two explanations were considered. Tie-breaking non-determinism was
asserted first and is now contradicted - re-running the S5 config at seed
42 through a different script reproduced its score exactly. The remaining
candidate is provenance: the CSVs were captured when version control was
initialised, so they may predate the code that produced the V8 logs.

This settles it by re-running that ONE window on current code at seed 42
and comparing against both artefacts:

  - matches the CSV (0.7993)  -> the CSV is current; the V8 log is the odd
    one out, and something about that run differed.
  - matches the log (0.8110)  -> the CSV is a stale artefact from earlier
    code, as suspected, and should be regenerated.
  - matches neither           -> the code has moved since both, and every
    committed per-window CSV needs regenerating before publication.

Cheap: one window, one seed, ~2 minutes.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    fit_feature_standardizer,
)
from greyspot.models.gat_temporal import (  # noqa: E402
    predict_gat_temporal,
    train_gat_temporal_walkforward,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_reproducibility_check")

BOROUGH = "Westminster"
TARGET_WINDOW = pd.Timestamp("2023-10-13")
FROM_LOG = 0.8110      # reports/run_logs/v8_westminster.log, seed 42
FROM_CSV = 0.7993      # reports/westminster/ucl_multiwindow_per_window_multiyear.csv


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "m", ROOT / "scripts" / "run_ucl_comparison_multiyear.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["x", BOROUGH]
    spec.loader.exec_module(m)

    instances, edge_index = m.build_instances(BOROUGH)
    idx = next(i for i, (_, _, s) in enumerate(instances) if s == TARGET_WINDOW)
    hx_raw, hy, start = instances[idx]
    train_slice = instances[:idx]
    logger.info("Re-running %s window %s (trained on %d instances, seed 42)",
                BOROUGH, start.date(), len(train_slice))

    mean, std = fit_feature_standardizer([x for x, _, _ in train_slice])
    tri = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
    hx = apply_feature_standardizer(hx_raw, mean, std)
    model = train_gat_temporal_walkforward(
        tri, edge_index, epochs=200, horizon=m.HORIZON, zero_inflated=True,
        use_residual=True, gat_hidden=16, gru_hidden=32, heads=3, gat_layers=1, seed=42,
    )
    got = float(accuracy_hit_rate(hy, predict_gat_temporal(model, hx, edge_index).T,
                                  top_fraction=0.20))

    print()
    print("=" * 70)
    print("REPRODUCIBILITY CHECK: %s %s, seed 42" % (BOROUGH, TARGET_WINDOW.date()))
    print("=" * 70)
    print("  current code      %.4f" % got)
    print("  V8 run log        %.4f  (diff %+.4f)" % (FROM_LOG, got - FROM_LOG))
    print("  committed CSV     %.4f  (diff %+.4f)" % (FROM_CSV, got - FROM_CSV))
    print()
    if abs(got - FROM_CSV) < 0.0005:
        print("VERDICT: matches the COMMITTED CSV.")
        print("  The CSV is current. The V8 log is the outlier for this window;")
        print("  something about that run differed and is worth locating before")
        print("  the V8 per-window figures are relied on further.")
    elif abs(got - FROM_LOG) < 0.0005:
        print("VERDICT: matches the V8 LOG.")
        print("  The committed CSV is a stale artefact predating current code,")
        print("  as suspected. Regenerate the per-window CSVs before publication.")
    else:
        print("VERDICT: matches NEITHER.")
        print("  The code has moved since both artefacts were produced. Every")
        print("  committed per-window CSV must be regenerated before publication.")
    pd.DataFrame([{"borough": BOROUGH, "window": TARGET_WINDOW, "current_code": got,
                   "v8_log": FROM_LOG, "committed_csv": FROM_CSV}]).to_csv(
        ROOT / "reports" / "reproducibility_check.csv", index=False)


if __name__ == "__main__":
    main()
