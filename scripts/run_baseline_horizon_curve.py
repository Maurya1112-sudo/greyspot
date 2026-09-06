"""How much of a 'historical baseline' depends on its HORIZON.

Gao et al. include a Historical Average baseline and report it at Acc@20 =
0.4520 / 0.4752 / 0.4217 (Lambeth / Tower Hamlets / Westminster; mean
0.4496) against their model's 0.7260. On that evidence a graph network
comfortably beats the historical baseline.

Their dataset is **2019 only**, so their HA can look back at most one year.
Our crash-count sort, given ~8 years, scores 0.8394 on our data - beating
both graph networks in this study. Those two facts are only reconcilable
if the baseline's strength is largely a function of its horizon rather
than of its sophistication.

This script tests that directly and internally: the SAME parameter-free
ranker (sort segments by crash count over the prior N days), swept across
horizons on our data and windows. If the curve rises steeply from a
one-year horizon, then the weakness of published historical baselines is
substantially an artefact of the data window they were computed over, and
the apparent margin of graph networks over them is correspondingly
overstated.

CPU-only; works from the sparse snapped collision list.

Run: python scripts/run_baseline_horizon_curve.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402
from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.network import snap_collisions_to_graph  # noqa: E402
from greyspot.ingest.stats19 import load_local_authority_collisions  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
HISTORY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
HORIZON = 14
EVAL_STARTS = pd.to_datetime(
    ["2023-07-15", "2023-10-13", "2024-01-11", "2024-04-10", "2024-07-09", "2024-10-07"]
)
LOOKBACK_DAYS = [30, 90, 180, 365, 730, 1095, 1825, 2555, 3285]
BOROUGHS = ["Lambeth", "Westminster", "Tower Hamlets"]


def borough_curve(borough: str) -> dict[int, list[float]]:
    b = get_borough(borough)
    graph = ox.load_graphml(
        INTERIM_DIR / f"{slug(b.name)}_os_open_roads_graph.graphml",
        node_dtypes={"osmid": str}, edge_dtypes={"osmid": str},
    )
    segs = [f"{u}_{v}_{k}" for u, v, k in graph.edges(keys=True)]
    sidx = {s: i for i, s in enumerate(segs)}
    n = len(segs)
    col = load_local_authority_collisions(b.ons_code, HISTORY_YEARS, RAW_DIR)
    sn = snap_collisions_to_graph(col, graph).dropna(subset=["segment_id"]).copy()
    sn["cdate"] = pd.to_datetime(sn["date"], format="%d/%m/%Y")

    out: dict[int, list[float]] = {lb: [] for lb in LOOKBACK_DAYS}
    for s in EVAL_STARTS:
        y = np.zeros((HORIZON, n))
        fut = sn[(sn.cdate >= s) & (sn.cdate < s + pd.Timedelta(days=HORIZON))]
        for sid, d in zip(fut.segment_id.values, fut.cdate.values):
            if sid in sidx:
                day = (pd.Timestamp(d) - s).days
                if 0 <= day < HORIZON:
                    y[day, sidx[sid]] += 1
        for lb in LOOKBACK_DAYS:
            hist = sn[(sn.cdate < s) & (sn.cdate >= s - pd.Timedelta(days=lb))]
            counts = np.zeros(n)
            for sid, c in hist.segment_id.value_counts().items():
                if sid in sidx:
                    counts[sidx[sid]] = c
            out[lb].append(accuracy_hit_rate(y, np.tile(counts, (HORIZON, 1)), top_fraction=0.20))
    return out


def main() -> None:
    curves = {b: borough_curve(b) for b in BOROUGHS}
    print("=" * 78)
    print("Crash-count sort: AccHR@20 as a function of LOOKBACK HORIZON")
    print("=" * 78)
    print()
    print("%-10s %10s %10s %10s %10s" % ("lookback", *BOROUGHS, "MEAN"))
    rows = []
    for lb in LOOKBACK_DAYS:
        vals = [float(np.mean(curves[b][lb])) for b in BOROUGHS]
        m = float(np.mean(vals))
        label = "%dd" % lb if lb < 365 else "%.1fyr" % (lb / 365.25)
        print("%-10s %10.4f %10.4f %10.4f %10.4f" % (label, *vals, m))
        rows.append({"lookback_days": lb, **{b: v for b, v in zip(BOROUGHS, vals)}, "mean": m})
    df = pd.DataFrame(rows)

    one_yr = float(df[df.lookback_days == 365]["mean"].iloc[0])
    best = float(df["mean"].max())
    best_lb = int(df.loc[df["mean"].idxmax(), "lookback_days"])
    print()
    print("-" * 78)
    print("For comparison, Gao et al. report (their data, 2019 only):")
    print("    Historical Average    0.4496 mean   (0.4520 / 0.4752 / 0.4217)")
    print("    STZITD-GNN            0.7260 mean")
    print()
    print("On our data the same parameter-free ranker scores:")
    print("    at a 1-year horizon   %.4f" % one_yr)
    print("    at %-4d days          %.4f  (best measured)" % (best_lb, best))
    print("    gain from horizon     %+.4f (%.1f points)" % (best - one_yr, 100 * (best - one_yr)))
    print()
    print("Our GNN scores 0.8014 (5-seed) and the reference architecture 0.6445")
    print("on these windows. A crash-count sort overtakes both once its")
    print("horizon exceeds roughly %d days." % int(df[df["mean"] > 0.8014].lookback_days.min())
          if (df["mean"] > 0.8014).any() else "  (never overtakes on this curve)")

    out = ROOT / "reports" / "baseline_horizon_curve.csv"
    df.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main()
