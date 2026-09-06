"""Render the preprint to a readable PDF.

**This PDF is for reading and sharing, not for arXiv.** arXiv wants the
LaTeX *source* (`paper/arxiv/`) and rejects TeX-produced PDFs; this is a
separate artefact built directly from `paper/DRAFT_preprint.md` with
ReportLab, because no LaTeX toolchain is installed here.

The two therefore have different provenance, and could in principle drift
apart. To stop that being silent, the script cross-checks a handful of
headline figures against the markdown source and refuses to build if they
disagree with `reports/headline_table.csv`.

Markdown handled: headings, paragraphs, bullet lists, tables, blockquotes,
bold/italic/code spans, footnote definitions. Everything the preprint
actually uses.

Run: python scripts/make_paper_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

SRC = ROOT / "paper" / "DRAFT_preprint.md"
OUT = ROOT / "paper" / "greyspot_preprint.pdf"
FIG_SVG = ROOT / "reports" / "figures" / "fig1_horizon_curve.svg"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#d0d0d0")
BAND = colors.HexColor("#f2f2f2")


def styles():
    ss = getSampleStyleSheet()
    base = dict(fontName="Times-Roman", textColor=INK, leading=13.6)
    return {
        "title": ParagraphStyle("t", ss["Title"], fontName="Times-Bold",
                                fontSize=17, leading=21, textColor=INK, spaceAfter=4),
        "byline": ParagraphStyle("by", ss["Normal"], fontName="Times-Roman",
                                 fontSize=9.5, textColor=MUTED, alignment=1, spaceAfter=14),
        "h1": ParagraphStyle("h1", ss["Normal"], fontName="Times-Bold", fontSize=13,
                             leading=16, textColor=INK, spaceBefore=15, spaceAfter=6),
        "h2": ParagraphStyle("h2", ss["Normal"], fontName="Times-Bold", fontSize=11,
                             leading=14, textColor=INK, spaceBefore=11, spaceAfter=4),
        "body": ParagraphStyle("b", ss["Normal"], fontSize=9.8, alignment=TA_JUSTIFY,
                               spaceAfter=6, **base),
        "quote": ParagraphStyle("q", ss["Normal"], fontSize=9.0, leftIndent=10,
                                rightIndent=6, textColor=MUTED, fontName="Times-Italic",
                                leading=12.4, spaceAfter=6, borderPadding=3),
        "li": ParagraphStyle("li", ss["Normal"], fontSize=9.8, leftIndent=12,
                             bulletIndent=3, spaceAfter=3, **base),
        "cap": ParagraphStyle("c", ss["Normal"], fontSize=8.4, textColor=MUTED,
                              fontName="Times-Italic", leading=11, spaceAfter=8),
        "cell": ParagraphStyle("cl", ss["Normal"], fontName="Times-Roman", fontSize=8.2,
                               leading=10.2, textColor=INK),
        "cellb": ParagraphStyle("cb", ss["Normal"], fontName="Times-Bold", fontSize=8.2,
                                leading=10.2, textColor=INK),
    }


def inline(md: str) -> str:
    """Markdown inline spans -> ReportLab markup."""
    s = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="8.4">\1</font>', s)
    s = re.sub(r"\[\^([^\]]+)\]", r"<super>\1</super>", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)          # links -> text
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s)              # images handled separately
    return s


def build_table(rows: list[list[str]], st) -> Table:
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), st["cellb"]) for c in head]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
    t = Table(data, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def crosscheck() -> None:
    """Refuse to build if the markdown disagrees with the computed table."""
    ht = ROOT / "reports" / "headline_table.csv"
    if not ht.exists():
        print("  (no headline_table.csv - cross-check skipped)")
        return
    d = pd.read_csv(ht)
    text = SRC.read_text(encoding="utf-8")
    bad = []
    for _, r in d.iterrows():
        val = "%.2f" % r["mean"]
        if val not in text:
            bad.append("%s=%s" % (r.borough, val))
    if bad:
        raise SystemExit(
            "REFUSING TO BUILD: the markdown does not contain the computed "
            "headline figures %s. Run make_headline_table.py and propagate "
            "before rendering a PDF that would disagree with the evidence."
            % ", ".join(bad))
    print("  cross-check OK: all %d headline figures present in the source" % len(d))


def main() -> None:
    crosscheck()
    st = styles()
    md = SRC.read_text(encoding="utf-8")
    flow = []

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="What Actually Matters in Road-Level Crash Prediction",
        author="Maurya Patel", subject="Road-level crash prediction: replication study",
    )

    lines = md.splitlines()
    i, para, tbl = 0, [], []

    def flush_para():
        if para:
            flow.append(Paragraph(inline(" ".join(para)), st["body"]))
            para.clear()

    def flush_tbl():
        if tbl:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in tbl
                    if not re.match(r"^\s*\|[\s|:-]+\|\s*$", r)]
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            flow.append(KeepTogether([build_table(rows, st), Spacer(1, 7)]))
            tbl.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            flush_para(); tbl.append(ln); i += 1; continue
        flush_tbl()
        if ln.startswith("# "):
            flush_para()
            flow.append(Paragraph(inline(ln[2:]), st["title"]))
            flow.append(Paragraph("Maurya Patel &nbsp;·&nbsp; University of Westminster "
                                  "&nbsp;·&nbsp; patelmaurya1112@gmail.com", st["byline"]))
        elif ln.startswith("## "):
            flush_para(); flow.append(Paragraph(inline(ln[3:]), st["h1"]))
        elif ln.startswith("### "):
            flush_para(); flow.append(Paragraph(inline(ln[4:]), st["h2"]))
        elif ln.startswith("> "):
            flush_para(); flow.append(Paragraph(inline(ln[2:]), st["quote"]))
        elif ln.strip().startswith(("- ", "* ")):
            flush_para()
            flow.append(Paragraph(inline(ln.strip()[2:]), st["li"], bulletText="•"))
        elif ln.startswith("!["):
            flush_para()
            if FIG_SVG.exists():
                from svglib.svglib import svg2rlg
                d = svg2rlg(str(FIG_SVG))
                # Scale by transforming the contents, not by setting a
                # `scale` attribute - Drawing has no such attribute and
                # assigning one raises. Resizing width/height alone would
                # crop rather than scale.
                sc = doc.width / d.width
                d.scale(sc, sc)
                d.width, d.height = doc.width, d.height * sc
                d.hAlign = "CENTER"
                flow.append(d); flow.append(Spacer(1, 4))
        elif ln.startswith("*Figure") or ln.startswith("*Data:"):
            flush_para(); flow.append(Paragraph(inline(ln.strip("*")), st["cap"]))
        elif ln.strip() == "---":
            flush_para(); flow.append(Spacer(1, 3))
            flow.append(HRFlowable(width="100%", color=RULE, thickness=0.6))
            flow.append(Spacer(1, 5))
        elif not ln.strip():
            flush_para()
        else:
            para.append(ln.strip())
        i += 1
    flush_para(); flush_tbl()

    doc.build(flow)
    print("Written to %s (%.0f KB)" % (OUT, OUT.stat().st_size / 1024))


if __name__ == "__main__":
    main()
