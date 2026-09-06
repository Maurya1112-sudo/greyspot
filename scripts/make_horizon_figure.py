"""Figure 1: AccHR@20 of a parameter-free crash-count sort vs its horizon.

The paper's central figure. Emitted as SVG written directly rather than
via a plotting library: the output is text, so it diffs and
version-controls like the rest of the repository, it is byte-reproducible,
and it adds no dependency to an environment a reader has to recreate.

Reads `reports/baseline_horizon_curve.csv`
(from `scripts/run_baseline_horizon_curve.py`).

Run: python scripts/make_horizon_figure.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

W, H = 760, 470
L, R, T, B = 74, 232, 34, 62          # margins; R is wide to hold the reference labels
PW, PH = W - L - R, H - T - B

BOROUGHS = ["Lambeth", "Westminster", "Tower Hamlets"]
COLOURS = {"Lambeth": "#4c72b0", "Westminster": "#dd8452", "Tower Hamlets": "#55a868"}
# Published / measured reference levels, and whose data each is on.
REFERENCES = [
    (0.4496, "Gao et al. Historical Average", "their data"),
    (0.6445, "reference architecture", "our data"),
    (0.7260, "Gao et al. STZITD-GNN", "their data"),
    (0.8014, "our GNN (5 seeds)", "our data"),
]
YMIN, YMAX = 0.15, 0.90


def main() -> None:
    df = pd.read_csv(ROOT / "reports" / "baseline_horizon_curve.csv")
    xs = df.lookback_days.tolist()
    lx = [math.log10(v) for v in xs]
    lo, hi = min(lx), max(lx)

    def px(days: float) -> float:
        return L + PW * (math.log10(days) - lo) / (hi - lo)

    def py(v: float) -> float:
        return T + PH * (1 - (v - YMIN) / (YMAX - YMIN))

    o: list[str] = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}" font-family="Georgia, \'Times New Roman\', serif">')
    o.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # y grid and axis
    v = 0.2
    while v <= YMAX + 1e-9:
        y = py(v)
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}" stroke="#e8e8e8" stroke-width="1"/>')
        o.append(f'<text x="{L-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#555">'
                 f'{int(v*100)}%</text>')
        v += 0.1
    o.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}" stroke="#333" stroke-width="1.2"/>')
    o.append(f'<line x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}" stroke="#333" stroke-width="1.2"/>')

    # x ticks
    for days, lab in [(30, "30d"), (90, "90d"), (365, "1yr"), (730, "2yr"),
                      (1095, "3yr"), (1825, "5yr"), (2555, "7yr"), (3285, "9yr")]:
        if days not in xs:
            continue
        x = px(days)
        o.append(f'<line x1="{x:.1f}" y1="{T+PH}" x2="{x:.1f}" y2="{T+PH+5}" stroke="#333" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{T+PH+19}" text-anchor="middle" font-size="11" fill="#555">{lab}</text>')
    o.append(f'<text x="{L+PW/2:.0f}" y="{H-16}" text-anchor="middle" font-size="12.5" fill="#333">'
             f'Lookback horizon of the crash-count sort (log scale)</text>')
    o.append(f'<text x="18" y="{T+PH/2:.0f}" text-anchor="middle" font-size="12.5" fill="#333" '
             f'transform="rotate(-90 18 {T+PH/2:.0f})">AccHR@20</text>')

    # reference levels
    for val, label, whose in REFERENCES:
        y = py(val)
        dash = "2,3" if whose == "their data" else "5,3"
        o.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}" stroke="#999" '
                 f'stroke-width="1" stroke-dasharray="{dash}"/>')
        o.append(f'<text x="{L+PW+8}" y="{y-2:.1f}" font-size="10.5" fill="#444">{label}</text>')
        o.append(f'<text x="{L+PW+8}" y="{y+10:.1f}" font-size="9" fill="#999" '
                 f'font-style="italic">{whose} &#183; {val*100:.1f}%</text>')

    # borough curves
    for b in BOROUGHS:
        pts = " ".join(f"{px(d):.1f},{py(v):.1f}" for d, v in zip(xs, df[b]))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{COLOURS[b]}" '
                 f'stroke-width="1.6" opacity="0.55"/>')
    pts = " ".join(f"{px(d):.1f},{py(v):.1f}" for d, v in zip(xs, df["mean"]))
    o.append(f'<polyline points="{pts}" fill="none" stroke="#222" stroke-width="2.6"/>')
    for d, v in zip(xs, df["mean"]):
        o.append(f'<circle cx="{px(d):.1f}" cy="{py(v):.1f}" r="3.2" fill="#222"/>')

    # legend
    ly = T + 8
    o.append(f'<rect x="{L+12}" y="{ly-2}" width="150" height="72" fill="#ffffff" '
             f'opacity="0.9" stroke="#e0e0e0"/>')
    o.append(f'<line x1="{L+22}" y1="{ly+12}" x2="{L+44}" y2="{ly+12}" stroke="#222" stroke-width="2.6"/>')
    o.append(f'<text x="{L+50}" y="{ly+16}" font-size="10.5" fill="#333">mean of 3 boroughs</text>')
    for i, b in enumerate(BOROUGHS):
        yy = ly + 30 + i * 14
        o.append(f'<line x1="{L+22}" y1="{yy}" x2="{L+44}" y2="{yy}" stroke="{COLOURS[b]}" '
                 f'stroke-width="1.6" opacity="0.55"/>')
        o.append(f'<text x="{L+50}" y="{yy+4}" font-size="10.5" fill="#555">{b}</text>')

    o.append("</svg>")

    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig1_horizon_curve.svg"
    out.write_text("\n".join(o), encoding="utf-8")
    print("Written to %s (%d bytes)" % (out, out.stat().st_size))
    print()
    print("Curve spans %.2f%% (30d) to %.2f%% (9yr) - a %.1f-point range."
          % (100 * df["mean"].iloc[0], 100 * df["mean"].iloc[-1],
             100 * (df["mean"].iloc[-1] - df["mean"].iloc[0])))


if __name__ == "__main__":
    main()
