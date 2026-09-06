"""Emit Figure 1 as pgfplots LaTeX, generated from the data.

arXiv accepts JPEG, PNG or PDF figures for pdfLaTeX submissions — **not
SVG**, which is what `make_horizon_figure.py` produces for the web
version. Rather than rasterise (losing vector quality) or add a converter
dependency (cairosvg/inkscape are not installed here, and arXiv only
compiles against TeX Live), the figure is emitted as pgfplots source.

That is the better answer for a submission in any case: pgfplots is in
TeX Live, the output is vector, the figure carries no external file, and
the numbers in the plot are the numbers in
`reports/baseline_horizon_curve.csv` rather than a picture of them.

Run: python scripts/make_arxiv_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

BOROUGHS = ["Lambeth", "Westminster", "Tower Hamlets"]
COLOURS = {"Lambeth": "blue!60!black", "Westminster": "orange!80!black",
           "Tower Hamlets": "green!50!black"}
# Reference levels: (value, label, whose data)
REFS = [
    (44.96, "Gao et al.\\ Historical Average", "their data"),
    (66.57, "reference architecture", "our data"),
    (72.60, "Gao et al.\\ STZITD-GNN", "their data"),
    (80.08, "our GNN", "our data"),
]
TICKS = [(30, "30d"), (90, "90d"), (365, "1yr"), (730, "2yr"),
         (1095, "3yr"), (1825, "5yr"), (2555, "7yr"), (3285, "9yr")]


def main() -> None:
    df = pd.read_csv(ROOT / "reports" / "baseline_horizon_curve.csv")
    o: list[str] = []
    # figure* spans both columns: IEEEtran is two-column, and this plot
    # carries labelled reference lines that become unreadable at column
    # width.
    o.append("\\begin{figure*}[t]")
    o.append("\\centering")
    o.append("\\begin{tikzpicture}")
    o.append("\\begin{axis}[")
    o.append("  width=0.86\\textwidth, height=6.6cm,")
    o.append("  xmode=log, log basis x=10,")
    o.append("  xlabel={Lookback horizon of the crash-count sort (log scale)},")
    o.append("  ylabel={AccHR@20 (\\%)},")
    o.append("  xmin=25, xmax=3600, ymin=15, ymax=92,")
    o.append("  xtick={%s}," % ",".join(str(d) for d, _ in TICKS if d in set(df.lookback_days)))
    o.append("  xticklabels={%s}," % ",".join(l for d, l in TICKS if d in set(df.lookback_days)))
    o.append("  ytick={20,30,40,50,60,70,80,90},")
    o.append("  grid=major, grid style={gray!20},")
    o.append("  tick label style={font=\\footnotesize},")
    o.append("  label style={font=\\small},")
    o.append("  legend style={font=\\footnotesize, at={(0.02,0.98)}, anchor=north west,")
    o.append("                draw=gray!40, fill=white, fill opacity=0.9, text opacity=1},")
    o.append("  legend cell align=left,")
    o.append("]")

    for b in BOROUGHS:
        pts = " ".join("(%d,%.2f)" % (d, 100 * v) for d, v in zip(df.lookback_days, df[b]))
        o.append("\\addplot[%s, thin, opacity=0.55, mark=none] coordinates {%s};"
                 % (COLOURS[b], pts))
        o.append("\\addlegendentry{%s}" % b)
    pts = " ".join("(%d,%.2f)" % (d, 100 * v) for d, v in zip(df.lookback_days, df["mean"]))
    o.append("\\addplot[black, very thick, mark=*, mark size=1.6pt] coordinates {%s};" % pts)
    o.append("\\addlegendentry{mean of 3 boroughs}")

    for val, label, whose in REFS:
        style = "densely dotted" if whose == "their data" else "densely dashed"
        o.append("\\addplot[gray, %s, forget plot] coordinates {(25,%.2f) (3600,%.2f)};"
                 % (style, val, val))
        o.append("\\node[anchor=east, font=\\tiny, gray!50!black] at (axis cs:3550,%.2f) "
                 "{%s\\ (%.1f\\%%)};" % (val + 2.0, label, val))
    o.append("\\end{axis}")
    o.append("\\end{tikzpicture}")
    o.append("\\caption{The same parameter-free crash-count sort, swept across lookback")
    o.append("horizons on our data and windows. The ranker spans %.2f\\%% to %.2f\\%% —"
             % (100 * df["mean"].iloc[0], 100 * df["mean"].iloc[-1]))
    o.append("a %.1f-point range — with nothing changing but how far back it looks."
             % (100 * (df["mean"].iloc[-1] - df["mean"].iloc[0])))
    o.append("Dashed reference lines are measured on our data; dotted lines are the")
    o.append("reference paper's published figures on theirs, so their vertical position")
    o.append("is indicative rather than a controlled comparison.}")
    o.append("\\label{fig:horizon}")
    o.append("\\end{figure*}")

    out = ROOT / "paper" / "arxiv" / "fig_horizon.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(o) + "\n", encoding="utf-8")
    print("Written to %s (%d lines)" % (out, len(o)))


if __name__ == "__main__":
    main()
