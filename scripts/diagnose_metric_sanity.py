"""Metric sanity check: what does accuracy_hit_rate return for degenerate
predictions?

Run because a claim in docs/final_model.md (§3.5) - that a CONSTANT
prediction scores BELOW random, so no score in this project is a
tie-break artefact - was originally computed inline and left no saved
artefact. Every claim needs a reproducible script behind it.

If a constant prediction scored ABOVE random, the metric's tie-breaking
would be doing work the model should be doing, and every AccHR@20 figure
in this project would be partly an artefact of segment ordering.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from greyspot.eval.ucl_metrics import accuracy_hit_rate  # noqa: E402

N, H = 11596, 14  # Lambeth's real segment count and horizon


def main() -> None:
    rng = np.random.default_rng(0)
    y = np.zeros((H, N))
    for d in range(H):
        for s in rng.choice(N, size=rng.integers(2, 5), replace=False):
            y[d, s] += 1

    cases = [
        ("all-constant (0.001)", np.full((H, N), 0.001)),
        ("all-zeros", np.zeros((H, N))),
        ("random uniform", rng.random((H, N))),
        ("near-constant + tiny noise", np.full((H, N), 0.001) + rng.normal(0, 1e-9, (H, N))),
    ]
    print("METRIC SANITY CHECK (theoretical random = 0.20)")
    print("realistic sparse target: %d segments, %d days, ~2-4 crashes/day" % (N, H))
    print()
    for name, pred in cases:
        scores = [accuracy_hit_rate(y, pred, top_fraction=0.20) for _ in range(5)]
        print("  %-30s AccHR@20 = %.4f" % (name, float(np.mean(scores))))
    print()
    print("PASS if all-constant scores BELOW random: a constant prediction must not")
    print("be rewarded by tie-breaking, or every score would be partly an artefact.")


if __name__ == "__main__":
    main()
