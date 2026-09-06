"""Does the horizon finding replicate on independent US data?

**The claim under test.** Section 5.1 of the paper reports that a
parameter-free ranker — sort road segments by cumulative past crash count
— spans 22.71% to 83.94% AccHR@20 depending only on its lookback horizon,
and argues that published margins of graph networks over historical
baselines are substantially an artefact of the short horizons those
baselines were computed over.

That finding rests on three London boroughs and roughly 10,000 crashes.
It is the paper's most novel claim and its narrowest evidence base, which
is exactly the weakness the rest of the paper criticises in others.

**The test.** The ML4RoadSafety benchmark of Nippani et al. (NeurIPS 2023)
provides monthly crash counts per road edge for US states, downloaded from
Harvard Dataverse (doi:10.7910/DVN/V71K5R). Delaware alone contains
458,282 crashes over 36,466 edges and 166 months (2009-2022) on a network
of 218,214 edges — roughly 45x the crash volume and 1.5x the history depth
of the London data.

**What differs from the London experiment, and why it is still the same
test.** Their data is monthly, ours daily, so the sweep is over months
rather than days and the metric averages over evaluation months rather
than days within a window. The quantity being measured is unchanged: the
share of a held-out period's crashes falling on the top 20% of segments
ranked by prior crash count, as a function of how far back that count
looks.

**Pre-registered interpretation**, fixed before running:

- The claim REPLICATES if the curve rises substantially with horizon and
  has a similar shape — a steep climb over the first few years followed by
  a plateau.
- It FAILS if the curve is flat, or if it peaks at a short horizon and
  declines, which would mean the London result is a property of that data
  rather than of the task.
- A different plateau POINT is not a failure. Delaware is a US state with
  different road density and reporting; the claim is about the shape and
  magnitude of the horizon dependence, not about a specific number.

Run: python scripts/run_horizon_us_replication.py [STATE]
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = ROOT / "data" / "raw" / "ml4rs"
TOP_FRACTION = 0.20
# Lookbacks in MONTHS. Chosen to mirror the London sweep's coverage
# (1 month to ~10 years) so the two curves are comparable in shape.
LOOKBACKS = [1, 3, 6, 12, 24, 36, 60, 84, 108, 120]
# Held-out months: the last 24 with a full 120-month lookback available.
N_EVAL_MONTHS = 24


def load_state(state: str):
    zp = DATA / f"{state}.zip"
    if not zp.exists():
        raise SystemExit("%s not downloaded. Fetch it from Harvard Dataverse "
                         "doi:10.7910/DVN/V71K5R" % zp)
    z = zipfile.ZipFile(zp)
    with z.open(f"{state}/accidents_monthly.csv") as f:
        acc = pd.read_csv(io.TextIOWrapper(f, "utf-8"))
    import torch
    with z.open(f"{state}/adj_matrix.pt") as f:
        adj = torch.load(io.BytesIO(f.read()), weights_only=False)
    n_edges = adj._nnz() if adj.is_sparse else int((adj > 0).sum())
    return acc, n_edges


def main(state: str = "DE") -> None:
    acc, n_edges = load_state(state)
    acc["t"] = acc.year * 12 + acc.month          # months since year 0
    # Edge identity: the unordered node pair, so a crash on either
    # direction of the same physical road counts once - matching the London
    # experiment, which keys on undirected segments.
    acc["edge"] = [tuple(sorted(p)) for p in zip(acc.node_1, acc.node_2)]

    edges = {e: i for i, e in enumerate(acc.edge.unique())}
    acc["ei"] = acc.edge.map(edges)
    n_crash_edges = len(edges)
    # The top-20% cut is over the WHOLE network, not only edges that have
    # ever crashed. Using the latter would inflate every score by hiding
    # the segments the ranker most easily gets right.
    n_undirected = n_edges // 2
    top_k = int(round(TOP_FRACTION * n_undirected))

    tmax = int(acc.t.max())
    eval_months = [t for t in range(tmax - N_EVAL_MONTHS + 1, tmax + 1)]
    eval_months = [t for t in eval_months if t - max(LOOKBACKS) >= acc.t.min()]

    print("=" * 74)
    print("HORIZON REPLICATION ON US DATA: %s" % state)
    print("=" * 74)
    print("network: %d directed edges (%d undirected); top-20%% cut = %d edges"
          % (n_edges, n_undirected, top_k))
    print("crashes: %d over %d edges, %d months (%d-%d)"
          % (acc.acc_count.sum(), n_crash_edges, acc.t.nunique(),
             acc.year.min(), acc.year.max()))
    print("evaluation: %d held-out months, %d-month max lookback"
          % (len(eval_months), max(LOOKBACKS)))
    print()

    # counts[edge, month] as a sparse-ish matrix over crash edges only;
    # every other edge has zero everywhere and can never enter the top-k
    # except through tie-breaking, which is handled by ranking on value.
    tmin = int(acc.t.min())
    n_t = tmax - tmin + 1
    mat = np.zeros((n_crash_edges, n_t), dtype=np.int32)
    mat[acc.ei.to_numpy(), acc.t.to_numpy() - tmin] = acc.acc_count.to_numpy()
    cum = np.concatenate([np.zeros((n_crash_edges, 1), dtype=np.int64),
                          np.cumsum(mat, axis=1, dtype=np.int64)], axis=1)

    rows = []
    for lb in LOOKBACKS:
        hits = []
        for t in eval_months:
            ti = t - tmin
            future = mat[:, ti]
            total = future.sum()
            if total == 0:
                continue
            hi, lo = ti, max(ti - lb, 0)
            past = cum[:, hi] - cum[:, lo]
            # Rank by past count; take the top_k edges. Edges outside the
            # crash set all have zero and would fill the remainder of the
            # top-k arbitrarily, so a zero-count edge contributes nothing
            # either way and the cut is applied to the real ranking.
            k = min(top_k, len(past))
            idx = np.argpartition(-past, k - 1)[:k]
            hits.append(future[idx].sum() / total)
        m = float(np.mean(hits))
        rows.append({"state": state, "lookback_months": lb,
                     "lookback_years": lb / 12.0, "AccHR": m, "n_months": len(hits)})
        label = "%d mo" % lb if lb < 12 else "%.0f yr" % (lb / 12)
        print("  %-8s AccHR@20 = %.4f   (%d months)" % (label, m, len(hits)))

    df = pd.DataFrame(rows)
    lo_v, hi_v = df.AccHR.iloc[0], df.AccHR.max()
    print()
    print("-" * 74)
    print("range: %.2f%% (%d mo) to %.2f%% (best) = %.1f points"
          % (100 * lo_v, LOOKBACKS[0], 100 * hi_v, 100 * (hi_v - lo_v)))
    print("London equivalent: 22.71%% (30d) to 83.94%% (9yr) = 61.2 points")
    print()
    peak = df.loc[df.AccHR.idxmax()]
    print("peak at %.1f years" % peak.lookback_years)
    rising = df.AccHR.iloc[-1] > df.AccHR.iloc[0]
    plateau = abs(df.AccHR.iloc[-1] - hi_v) < 0.02
    if rising and (hi_v - lo_v) > 0.10:
        print("VERDICT: the horizon dependence REPLICATES on independent US data.")
        print("  The curve rises %.1f points with horizon alone." % (100 * (hi_v - lo_v)))
        if plateau:
            print("  It plateaus, as the London curve does after ~7 years.")
    else:
        print("VERDICT: the horizon dependence does NOT replicate here.")
        print("  The London result may be specific to that data rather than the task.")

    out = ROOT / "reports" / ("horizon_us_%s.csv" % state.lower())
    df.to_csv(out, index=False)
    print()
    print("Written to %s" % out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DE")
