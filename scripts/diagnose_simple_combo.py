"""Can a SIMPLE model beat the GNN? - the direction every negative
result this session has been pointing at, 2026-09-03.

`diagnose_trivial_baselines.py` found that sorting by a single column
(`poi_amenity_count`) scores 56.89% on Lambeth against the GNN's
63.59% - the whole GAT+GRU+ZIP apparatus is worth about +6.7 points
over one sorted feature. With 6-11 training instances and a signal
dominated by a few static land-use columns, a 40-feature graph network
is enormously overparameterised, and a deliberately simple combination
may generalise better.

This tests parameter-free rank averages of the strongest static
signals. No training, no fitted weights, therefore nothing to overfit
and nothing selected on the evaluation windows - each combination is a
fixed rule stated in advance and applied identically to every window.

Rank-averaging (not raw-value averaging) because the columns are on
wildly different scales - a raw sum would simply return whichever
column has the largest numbers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("diagnose_simple_combo")


def _ranks(v: np.ndarray) -> np.ndarray:
    """Ascending rank, normalised to [0, 1]."""
    return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)


def main(borough_name: str = "Lambeth") -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "base", ROOT / "scripts" / "run_ucl_comparison_multiwindow_os_open_roads.py"
    )
    base = importlib.util.module_from_spec(spec)
    sys.argv = ["x", borough_name]
    spec.loader.exec_module(base)

    instances, _ = base.build_instances(borough_name)
    names = base.FEATURE_COLUMNS
    n, n_windows = len(instances), base.N_WINDOWS

    # Fixed, pre-stated combinations - no weights fitted, none chosen by
    # looking at results.
    combos = {
        "poi_amenity only": ["poi_amenity_count"],
        "all 4 POI classes": ["poi_shop_count", "poi_amenity_count", "poi_leisure_count", "poi_tourism_count"],
        "POI + length": ["poi_amenity_count", "length"],
        "POI + length + aadf": ["poi_amenity_count", "length", "aadf_all_motor_vehicles"],
        "all POI + length": ["poi_shop_count", "poi_amenity_count", "poi_leisure_count", "poi_tourism_count", "length"],
    }

    rows = []
    for idx in range(n - n_windows, n):
        x_raw, y_true, start = instances[idx]
        last = x_raw[-1]  # [N, F]
        row = {"held_out_start": start}
        for label, cols in combos.items():
            present = [c for c in cols if c in names]
            if not present:
                continue
            stacked = np.mean([_ranks(last[:, names.index(c)]) for c in present], axis=0)
            scores = np.tile(stacked, (y_true.shape[0], 1))
            row[label] = accuracy_hit_rate(y_true, scores, 0.20)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print("MEANS (same 6 windows as every model run):")
    for c in df.columns:
        if c != "held_out_start":
            print(f"  {c:26s} {df[c].mean():.4f}")
    print()
    print("  GNN best config              0.6359")
    print("  Gao et al. (Lambeth)         0.7659")
    out = ROOT / "reports" / borough_name.lower().replace(" ", "_") / "simple_combo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Lambeth")
