"""S2 - Empirical Bayes baseline (Highway Safety Manual method).

**Why this matters for the write-up.** The HSM's Empirical Bayes method
is the standard road-safety approach to ranking sites by crash risk, and
it has used multi-year crash history for decades. Without it as a
baseline, a reviewer can reasonably say this project's history-horizon
finding "rediscovers Empirical Bayes". With it, the project can state
precisely how a modern GNN compares to the established method on
identical data and windows.

**The method** (Hauer's formulation, as used in the HSM): the EB
estimate shrinks a site's observed crash count toward a
covariate-predicted mean:

    lambda_EB = w * mu_predicted + (1 - w) * count_observed
    w         = 1 / (1 + mu_predicted / k)

where `k` is the negative-binomial overdispersion parameter. Sites with
few observed crashes are shrunk hard toward the model prediction; sites
with many are trusted. `mu_predicted` here comes from a simple
safety-performance function on segment length and (where available)
traffic volume, fitted on the training period only.

Deliberately CPU-only and low-memory: it works from the sparse snapped
collision list, never materialising a segment-day grid, so it can run
alongside a GPU training job without competing for RAM.
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
from greyspot.ingest.network import graph_to_edges_gdf, snap_collisions_to_graph  # noqa: E402
from greyspot.ingest.stats19 import load_local_authority_collisions  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_s2_empirical_bayes")

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
HISTORY_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
HORIZON = 14
EVAL_STARTS = pd.to_datetime(
    ["2023-07-15", "2023-10-13", "2024-01-11", "2024-04-10", "2024-07-09", "2024-10-07"]
)


def eb_estimate(counts: np.ndarray, exposure: np.ndarray, years: float) -> np.ndarray:
    """Empirical Bayes shrinkage toward a length-based SPF prediction."""
    # Simple SPF: expected crashes proportional to exposure (segment length),
    # calibrated so total predicted equals total observed over the period.
    expo = np.where(exposure > 0, exposure, exposure[exposure > 0].mean() if (exposure > 0).any() else 1.0)
    mu = expo / expo.sum() * counts.sum()
    mu = np.maximum(mu, 1e-9)
    # Negative-binomial overdispersion, method of moments on the observed counts.
    var, mean = counts.var(), counts.mean()
    k = (mean ** 2) / max(var - mean, 1e-6) if var > mean else 1e6
    w = 1.0 / (1.0 + mu / k)
    return w * mu + (1.0 - w) * counts


def main(borough: str = "Lambeth") -> None:
    b = get_borough(borough)
    graph = ox.load_graphml(
        INTERIM_DIR / f"{slug(b.name)}_os_open_roads_graph.graphml",
        node_dtypes={"osmid": str}, edge_dtypes={"osmid": str},
    )
    segs = [f"{u}_{v}_{k}" for u, v, k in graph.edges(keys=True)]
    sidx = {s: i for i, s in enumerate(segs)}
    n = len(segs)

    edges = graph_to_edges_gdf(graph).drop_duplicates(subset="segment_id").set_index("segment_id")
    length = np.array([float(edges.loc[s, "length"]) if s in edges.index else 0.0 for s in segs])

    col = load_local_authority_collisions(b.ons_code, HISTORY_YEARS, RAW_DIR)
    sn = snap_collisions_to_graph(col, graph).dropna(subset=["segment_id"]).copy()
    sn["cdate"] = pd.to_datetime(sn["date"], format="%d/%m/%Y")

    # MATCHED-HORIZON CONTROL (added 2026-09-05): the GNN's deepest feature
    # is 1825 days, but an unbounded cumulative count sees ~8 years. Comparing
    # them directly confounds "simple beats complex" with "more history beats
    # less". Capping the baseline at 1825d makes the horizons identical, so a
    # remaining advantage is attributable to the ESTIMATOR, not the data.
    results = {"EB (HSM method)": [], "raw count (no shrinkage)": [],
               "raw count CAPPED at 1825d (matched to GNN)": []}
    for s in EVAL_STARTS:
        hist = sn[sn.cdate < s]
        hist_capped = sn[(sn.cdate < s) & (sn.cdate >= s - pd.Timedelta(days=1825))]
        counts_capped = np.zeros(n)
        for sid, c in hist_capped.segment_id.value_counts().items():
            if sid in sidx:
                counts_capped[sidx[sid]] = c
        counts = np.zeros(n)
        for sid, c in hist.segment_id.value_counts().items():
            if sid in sidx:
                counts[sidx[sid]] = c
        years = (s - sn.cdate.min()).days / 365.25

        y = np.zeros((HORIZON, n))
        fut = sn[(sn.cdate >= s) & (sn.cdate < s + pd.Timedelta(days=HORIZON))]
        for sid, d in zip(fut.segment_id.values, fut.cdate.values):
            if sid in sidx:
                day = (pd.Timestamp(d) - s).days
                if 0 <= day < HORIZON:
                    y[day, sidx[sid]] += 1

        for name, score in [("EB (HSM method)", eb_estimate(counts, length, years)),
                            ("raw count (no shrinkage)", counts),
                            ("raw count CAPPED at 1825d (matched to GNN)", counts_capped)]:
            results[name].append(accuracy_hit_rate(y, np.tile(score, (HORIZON, 1)), top_fraction=0.20))

    print()
    print("=== S2: EMPIRICAL BAYES BASELINE (%s, same 6 windows) ===" % borough)
    for name, vals in results.items():
        print("  %-42s AccHR@20 = %.4f  (windows: %s)" % (name, float(np.mean(vals)),
              " ".join("%.4f" % v for v in vals)))

    # PER-WINDOW OUTPUT (added 2026-09-05). Previously this script printed
    # its per-window values and kept nothing, so the paired comparison
    # against the GNN could not be computed without re-running it - and the
    # one run that included the matched-horizon arm had its log lost to a
    # machine restart. Rule R12: any number quoted in a doc needs a script
    # in the repo that regenerates it.
    out_dir = ROOT / "reports" / slug(b.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"borough": b.name, "baseline": name, "held_out_start": start, "AccHR": float(v)}
        for name, vals in results.items()
        for start, v in zip(EVAL_STARTS, vals)
    ]
    out_path = out_dir / "s2_empirical_bayes_per_window.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print()
    print("  per-window results written to %s" % out_path)
    # NOTE: the GNN comparison is deliberately NOT printed here. This script
    # previously printed a hardcoded "GNN final model (Lambeth) 0.7944" for
    # EVERY borough, so a Westminster or Tower Hamlets run displayed
    # Lambeth's score under its own heading - and 0.7944 is additionally a
    # single-seed figure, superseded by the 5-seed mean (V8/V11). The
    # comparison now lives in scripts/run_s2_paired_comparison.py, which
    # pairs these windows against the per-seed GNN windows properly.


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Lambeth")
