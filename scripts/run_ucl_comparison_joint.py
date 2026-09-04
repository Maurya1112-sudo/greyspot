"""JOINT MULTI-BOROUGH TRAINING - one model trained on all three
boroughs at once, evaluated per borough. Built 2026-09-03.

**The binding constraint this attacks.** Every experiment in this
project trains a separate model per borough on 6-11 walk-forward
instances. That tiny training set is the single most consistent
explanation for the session's long run of nulls: extra features hurt,
extra capacity hurt, ensembling washed out, and a trivial one-column
baseline reaches 56.89% of the GNN's 63.59% - all symptoms of a model
with almost no room to learn anything beyond the strongest static
signal.

Training jointly on Westminster + Lambeth + Tower Hamlets triples the
training SIGNAL without inventing any new data. Stated precisely (the
first draft of this docstring got it wrong): the instance COUNT is
unchanged at 6-11 per window - boroughs are concatenated along the
SEGMENT axis, not the instance axis. What triples is the number of
supervised (node, target) pairs the shared weights see per step,
from ~11k nodes to ~32.7k. For a node-level loss that is the quantity
that matters, but it is not the same thing as having more time windows,
and it does NOT reduce the temporal-sample-size problem: the model
still sees only 6-11 distinct points in time. The three boroughs are adjacent inner-London authorities with
comparable road networks, land use and traffic regimes, so a shared
model is substantively reasonable rather than merely convenient.

**Why this is a genuine methodological step beyond the reference
paper**: Gao et al. train and report each borough separately (their
Table 7.3 lists per-borough models). Nothing in their setup pools the
boroughs. If joint training helps, it is an improvement over their
approach, not a replication of it.

**Implementation.** The three per-borough graphs are combined into ONE
block-diagonal graph: node features are concatenated and each borough's
edge indices are offset by the running node count, so no edge ever
crosses a borough boundary. A GNN handles disconnected components
natively - message passing simply never leaves a component - so this is
mathematically three separate graphs sharing one set of weights, which
is exactly the intent (shared parameters, no fabricated adjacency
between physically unconnected boroughs).

**Evaluation is unchanged and still per borough**: predictions are
sliced back to each borough's own segment range and AccHR@20 computed
on that borough alone, on the identical six windows every other run
uses. Numbers therefore stay directly comparable to every previous
result.

**Leakage**: the walk-forward discipline is per window as always -
training uses only instances strictly before the evaluated window, for
every borough simultaneously. Feature standardisation is fitted on the
combined training instances only.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from greyspot.eval.ucl_metrics import ucl_metric_suite  # noqa: E402
from greyspot.features.daily_temporal import (  # noqa: E402
    apply_feature_standardizer,
    fit_feature_standardizer,
)
from greyspot.models.conformal import manual_split_conformal_interval  # noqa: E402
from greyspot.models.gat_temporal import predict_gat_temporal, train_gat_temporal_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ucl_comparison_joint")

BOROUGHS = ["Westminster", "Lambeth", "Tower Hamlets"]
N_WINDOWS = 6
HORIZON = 14


def build_all():
    """Build instances for every borough using the established real-network
    pipeline, then combine them into one block-diagonal graph."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "base", ROOT / "scripts" / "run_ucl_comparison_multiwindow_os_open_roads.py"
    )
    base = importlib.util.module_from_spec(spec)
    sys.argv = ["x", "Westminster"]
    spec.loader.exec_module(base)

    per_borough = {}
    for b in BOROUGHS:
        inst, ei = base.build_instances(b)
        per_borough[b] = (inst, ei)
        logger.info("%s: %d instances, %d segments", b, len(inst), inst[0][0].shape[1])

    n_inst = min(len(v[0]) for v in per_borough.values())
    # segment ranges for slicing predictions back apart
    offsets, cursor = {}, 0
    for b in BOROUGHS:
        n_seg = per_borough[b][0][0][0].shape[1]
        offsets[b] = (cursor, cursor + n_seg)
        cursor += n_seg
    total_segments = cursor

    # block-diagonal edge index: offset each borough's node ids so no edge
    # crosses a borough boundary
    edges = []
    for b in BOROUGHS:
        ei = per_borough[b][1]
        edges.append(np.asarray(ei) + offsets[b][0])
    joint_edge_index = np.concatenate(edges, axis=1)

    # concatenate features/targets along the segment axis, per instance
    joint = []
    for i in range(n_inst):
        xs = [per_borough[b][0][i][0] for b in BOROUGHS]
        ys = [per_borough[b][0][i][1] for b in BOROUGHS]
        start = per_borough[BOROUGHS[0]][0][i][2]
        joint.append((np.concatenate(xs, axis=1), np.concatenate(ys, axis=1), start))

    logger.info("Joint graph: %d segments, %d edges, %d instances",
                total_segments, joint_edge_index.shape[1], len(joint))
    return joint, joint_edge_index, offsets, per_borough


def main() -> None:
    joint, edge_index, offsets, per_borough = build_all()
    n = len(joint)
    rows = []

    for held_out_idx in range(n - N_WINDOWS, n):
        train_slice = joint[:held_out_idx]
        ho_x_raw, ho_y, start = joint[held_out_idx]

        mean, std = fit_feature_standardizer([x for x, _, _ in train_slice])
        train_instances = [(apply_feature_standardizer(x, mean, std), y.T) for x, y, _ in train_slice]
        ho_x = apply_feature_standardizer(ho_x_raw, mean, std)

        model = train_gat_temporal_walkforward(
            train_instances, edge_index,
            epochs=200, horizon=HORIZON, zero_inflated=True, use_residual=True,
            gat_hidden=16, gru_hidden=32, heads=3, gat_layers=1,
        )
        pred = predict_gat_temporal(model, ho_x, edge_index)  # [N_total, horizon]

        calib_x_raw, calib_y, _ = train_slice[-1]
        calib_pred = predict_gat_temporal(model, apply_feature_standardizer(calib_x_raw, mean, std), edge_index)

        for b in BOROUGHS:
            lo, hi = offsets[b]
            y_true_b = ho_y[:, lo:hi]
            y_pred_b = pred[lo:hi].T
            lower, upper = manual_split_conformal_interval(
                calib_y[:, lo:hi].T.flatten(), calib_pred[lo:hi].flatten(),
                pred[lo:hi].flatten(), confidence_level=0.9,
            )
            lower = lower.reshape(pred[lo:hi].shape).T
            upper = upper.reshape(pred[lo:hi].shape).T
            m = ucl_metric_suite(y_true_b, y_pred_b, lower, upper, top_fraction=0.20)
            m["borough"] = b
            m["held_out_start"] = start
            m["n_train_instances"] = len(train_slice)
            rows.append(m)
            logger.info("  window %s %-15s AccHR@20=%.4f", start.date(), b, m["AccHR"])

    df = pd.DataFrame(rows)
    for b in BOROUGHS:
        sub = df[df.borough == b]
        logger.info("=== JOINT %s: mean=%.4f std=%.4f ===", b, sub.AccHR.mean(), sub.AccHR.std())
        out = ROOT / "reports" / b.lower().replace(" ", "_") / "ucl_multiwindow_per_window_joint.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
    logger.info("Written to reports/<borough>/ucl_multiwindow_per_window_joint.csv")


if __name__ == "__main__":
    main()
