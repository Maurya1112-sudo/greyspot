"""Why one window disagrees between runs, and five do not.

**The observation.** Westminster's held-out window 2023-10-13 has now
disagreed between independent runs three times:

| comparison | configuration | difference |
|---|---|---|
| V8 log vs committed CSV | 5-year history | 0.0117 |
| S5 single run vs S5 multi-seed, seed 42 | 9-year history | 0.0119 |

Every other window in both comparisons was bit-identical (0.0000). That
pattern rules out ordinary numerical noise, which would perturb all
windows slightly rather than one window discretely.

**The mechanism, and it is exact.** AccHR@20 is a per-day-averaged hit
rate: for each day it computes the share of that day's crashes falling on
the top 20% of predicted-risk segments, then averages over the window's
days. So the smallest possible change is one crash crossing the threshold
on one day, worth

    1 / (crashes that day) / (days in window)

The 2023-10-13 window contains 59 crashes over 14 days, six of them on
each of several days. One crash moving in or out on a 6-crash day is worth
**1/6/14 = 0.011905** - the observed difference to four decimal places.

**Consequence for interpretation.** The metric is a step function of the
ranking, and 94%+ of segments are tied at zero recent crashes, so a
floating-point difference far too small to see in the predictions can
reorder a tie group and move a segment across the 20% cut. Same-seed
reruns are therefore *usually* bit-identical and *occasionally* differ by
one discrete step. The step size is set by the sparsity of the target, not
by the model.

This script quantifies the step size for every window, so the size of any
future discrepancy can be checked against it rather than guessed at.

Run: python scripts/diagnose_metric_granularity.py [Borough]
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

from greyspot.ingest.boroughs import get_borough, slug  # noqa: E402
from greyspot.ingest.network import snap_collisions_to_graph  # noqa: E402
from greyspot.ingest.stats19 import load_local_authority_collisions  # noqa: E402

logging.basicConfig(level=logging.WARNING)

HORIZON = 14
EVAL_STARTS = pd.to_datetime(
    ["2023-07-15", "2023-10-13", "2024-01-11", "2024-04-10", "2024-07-09", "2024-10-07"]
)
OBSERVED = {  # differences actually seen between independent runs
    ("Westminster", "2023-10-13"): 0.0119,
}


def main(borough_name: str = "Westminster") -> None:
    b = get_borough(borough_name)
    graph = ox.load_graphml(
        ROOT / "data" / "interim" / f"{slug(b.name)}_os_open_roads_graph.graphml",
        node_dtypes={"osmid": str}, edge_dtypes={"osmid": str},
    )
    col = load_local_authority_collisions(b.ons_code, [2023, 2024], ROOT / "data" / "raw")
    sn = snap_collisions_to_graph(col, graph).dropna(subset=["segment_id"]).copy()
    sn["cdate"] = pd.to_datetime(sn["date"], format="%d/%m/%Y")

    print("=" * 78)
    print("AccHR@20 granularity: the smallest possible change per window (%s)" % b.name)
    print("=" * 78)
    print()
    print("%-14s %8s %6s %14s %14s %s" %
          ("window", "crashes", "days", "smallest step", "largest step", "observed"))
    rows = []
    for s in EVAL_STARTS:
        fut = sn[(sn.cdate >= s) & (sn.cdate < s + pd.Timedelta(days=HORIZON))]
        if fut.empty:
            continue
        per_day = fut.groupby(fut.cdate.dt.date).size()
        n_days = len(per_day)
        steps = sorted({1.0 / c / n_days for c in per_day.values})
        obs = OBSERVED.get((b.name, str(s.date())))
        match = ""
        if obs is not None:
            near = min(steps, key=lambda x: abs(x - obs))
            match = "%.4f  (= 1 crash on a %d-crash day, %s)" % (
                obs, round(1.0 / near / n_days),
                "EXACT MATCH" if abs(near - obs) < 0.0002 else "no clean match")
        print("%-14s %8d %6d %14.4f %14.4f %s" %
              (str(s.date()), len(fut), n_days, steps[0], steps[-1], match))
        rows.append({"borough": b.name, "window": s.date(), "crashes": len(fut),
                     "days": n_days, "min_step": steps[0], "max_step": steps[-1],
                     "observed_diff": obs})

    print()
    print("The step size is set by TARGET SPARSITY, not by the model: a window with")
    print("few crashes has a coarse metric. That is why a single tie flip is visible")
    print("at the fourth decimal place, and why five of six windows can be")
    print("bit-identical while the sixth moves by ~0.012.")
    out = ROOT / "reports" / ("metric_granularity_%s.csv" % slug(b.name))
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Westminster")
