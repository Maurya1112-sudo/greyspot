"""What does a TRIVIAL baseline score on AccHR@20? - the check that
should have been run first, 2026-09-03.

A model is only worth its complexity if it beats the dumb alternatives
by a margin that matters. This project has spent an entire
investigation moving a GNN between 42% and 70% without ever asking what
a single feature, sorted, would score on the same windows.

That matters especially for AccHR@20 on this data: crashes concentrate
on busy major roads, and ANY proxy for "this is a major road" (traffic
volume, road class, length, POI density) will capture a large share of
them. A random top-20% selection scores ~20% by construction; the real
question is how far above that the cheap heuristics already sit, and
therefore how much of the GNN's 63.59% is genuine spatiotemporal
learning rather than "major roads are dangerous".

Uses the identical windows, identical metric and identical leakage
discipline as every model run, so the numbers are directly comparable.
No training and no GPU - it only sorts existing feature columns.
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
logger = logging.getLogger("diagnose_trivial_baselines")


def main(borough_name: str = "Lambeth") -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "base", ROOT / "scripts" / "run_ucl_comparison_multiwindow_os_open_roads.py"
    )
    base = importlib.util.module_from_spec(spec)
    sys.argv = ["x", borough_name]
    spec.loader.exec_module(base)

    instances, _edge_index = base.build_instances(borough_name)
    feature_names = base.FEATURE_COLUMNS
    n_windows = base.N_WINDOWS
    n = len(instances)

    # Candidate cheap rankings, all computed from the LAST input timestep
    # of the held-out instance's own features (information genuinely
    # available at prediction time - no target leakage).
    candidates = {
        "length": "length",
        "aadf_all_motor_vehicles": "aadf_all_motor_vehicles",
        "u_degree": "u_degree",
        "collision_count_30d": "collision_count_30d",
        "poi_amenity_count": "poi_amenity_count",
    }

    rows = []
    rng = np.random.default_rng(0)
    for held_out_idx in range(n - n_windows, n):
        x_raw, y_true, start = instances[held_out_idx]
        last_step = x_raw[-1]  # [N, F] - features at the final input day

        window = {"held_out_start": start}
        # random control - the floor this metric cannot go below by chance
        rand_scores = rng.random((y_true.shape[0], y_true.shape[1]))
        window["random"] = accuracy_hit_rate(y_true, rand_scores, 0.20)

        for label, col in candidates.items():
            if col not in feature_names:
                continue
            idx = feature_names.index(col)
            per_segment = last_step[:, idx]                       # [N]
            scores = np.tile(per_segment, (y_true.shape[0], 1))   # [horizon, N]
            window[label] = accuracy_hit_rate(y_true, scores, 0.20)
        rows.append(window)

    df = pd.DataFrame(rows)
    logger.info("Per-window trivial-baseline AccHR@20 (%s):", borough_name)
    print(df.to_string(index=False))
    print()
    print("MEANS across the same 6 windows the models are evaluated on:")
    for col in df.columns:
        if col == "held_out_start":
            continue
        print(f"  {col:28s} {df[col].mean():.4f}")
    print()
    print("  GNN (best config, same windows)   0.6359")
    print("  Gao et al. reported (Lambeth)     0.7659")

    out = ROOT / "reports" / borough_name.lower().replace(" ", "_") / "trivial_baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Lambeth")
