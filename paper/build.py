#!/usr/bin/env python3
"""Render the short paper and its figures from existing results; no GPU work."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL_DEPS = ROOT / ".run/paper-deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
)

RUNTIME = ROOT / "results/cuda-context-study/report/runtime.csv"
DIAG = ROOT / "results/cuda-context-study/diagnostics/q2-q4-priority"
Q2, Q4, CONTROL = "#c66b12", "#245ba8", "#167743"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def figures():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })
    selected = [r for r in read_csv(RUNTIME)
                if r["model"] == "70b" and r["concurrency"] == "8"]
    assert len(selected) == 8
    assert all(r["repetitions"] == "1" and r["output_tokens"] == "512"
               and r["successful_requests"] == "16" and r["failed_requests"] == "0"
               for r in selected)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.25), layout="constrained")
    for quant, color, marker in (("Q4_K_M", Q4, "s"), ("Q2_K", Q2, "o")):
        data = sorted((r for r in selected if r["format"] == quant),
                      key=lambda r: int(r["input_tokens"]))
        prompts = [int(r["input_tokens"]) for r in data]
        assert prompts == [2048, 4096, 8192, 16384]
        for ax, field, divisor in zip(axes,
                ("output_tokens_per_second_mean", "ttft_p95_ms"), (1, 1000)):
            ax.plot(prompts, [float(r[field]) / divisor for r in data],
                    marker=marker, color=color, linewidth=1.7, markersize=4,
                    label=quant)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([2048, 4096, 8192, 16384], ["2k", "4k", "8k", "16k"])
        ax.set_xlabel("Input tokens per request")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Generated tokens / s")
    axes[0].set_title("(a) Aggregate output throughput")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_ylabel("TTFT p95 (s)")
    axes[1].set_title("(b) First-token latency")
    save_figure(fig, "runtime-cost")

    timing = read_csv(DIAG / "controlled-timings.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), layout="constrained", sharey=True)
    for gpu, ax in enumerate(axes):
        for quant, color, marker in (("Q4_K", Q4, "s"), ("Q2_K", Q2, "o")):
            data = {int(r["width"]): float(r["median_operation_us"]) for r in timing
                    if r["gpu"] == str(gpu) and r["variant"] == "baseline"
                    and r["shape"] == "ffn_gate_up" and r["type"] == quant}
            ax.plot([1, 8, 9], [data[n] for n in (1, 8, 9)], label=quant + " stock",
                    marker=marker, color=color, linewidth=1.7, markersize=4)
        early = [r for r in timing if r["gpu"] == str(gpu)
                 and r["variant"] == "early-mmq" and r["shape"] == "ffn_gate_up"
                 and r["type"] == "Q2_K" and r["width"] == "8"]
        assert len(early) == 1
        ax.scatter([8], [float(early[0]["median_operation_us"])], marker="*", s=110,
                   color=CONTROL, label="Q2_K earlier MMQ", zorder=5)
        ax.set_title(f"({'ab'[gpu]}) GPU {gpu}")
        ax.set_xticks([1, 8, 9])
        ax.set_xlim(.5, 9.5)
        ax.set_ylim(0, 660)
        ax.set_xlabel("Tested activation width, N (columns)")
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Complete operation (µs)")
    axes[1].legend(frameon=False, loc="upper left")
    save_figure(fig, "runtime-bottlenecks")


def save_figure(fig, name):
    (HERE / "figures").mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(HERE / "figures" / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def register_fonts():
    for family, prefix in (("DejaVu Serif", "Paper"), ("DejaVu Sans", "Sans")):
        for suffix, weight, style in (("", "normal", "normal"), ("Bold", "bold", "normal"),
                                      ("Italic", "normal", "italic"), ("BoldItalic", "bold", "italic")):
            path = font_manager.findfont(font_manager.FontProperties(
                family=family, weight=weight, style=style))
            pdfmetrics.registerFont(TTFont(prefix + suffix, path))
        pdfmetrics.registerFontFamily(prefix, normal=prefix, bold=prefix + "Bold",
                                      italic=prefix + "Italic", boldItalic=prefix + "BoldItalic")
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono"))
    pdfmetrics.registerFont(TTFont("Code", path))


def inline(element):
    text = ET.tostring(element, encoding="unicode", method="xml")
    text = text[text.index(">") + 1:text.rfind("</")]
    return text.replace("<code>", '<font name="Code" size="8">').replace("</code>", "</font>")


def render_pdf():
    register_fonts()
    width = 528
    body = ParagraphStyle("Body", fontName="Paper", fontSize=9.5, leading=11.9,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    heading = ParagraphStyle("Heading", parent=body, fontName="SansBold", fontSize=11,
                             leading=14, alignment=TA_LEFT, spaceBefore=8, spaceAfter=4,
                             keepWithNext=True)
    title = ParagraphStyle("Title", parent=heading, fontSize=18, leading=22,
                           spaceBefore=0, spaceAfter=8)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8, leading=10,
                             alignment=TA_LEFT, spaceAfter=7)
    reference = ParagraphStyle("Reference", parent=body, fontSize=7.8, leading=10,
                               alignment=TA_LEFT, spaceAfter=4)
    cell = ParagraphStyle("Cell", parent=body, fontName="Sans", fontSize=7.8,
                          leading=10, alignment=TA_LEFT, spaceAfter=0)
    root = ET.fromstring("<root>" + markdown.markdown(
        (HERE / "llama70b.md").read_text(), extensions=["tables"]) + "</root>")
    elements = list(root)
    story, in_refs = [], False
    index = 0
    while index < len(elements):
        node = elements[index]
        if node.tag in ("h1", "h2"):
            in_refs = node.text == "References"
            story.append(Paragraph(inline(node), title if node.tag == "h1" else heading))
        elif node.tag == "p" and node.find("img") is not None:
            src = node.find("img").attrib["src"]
            img = Image(str(HERE / src))
            img.drawHeight *= width / img.drawWidth
            img.drawWidth = width
            parts = [Spacer(1, 4), img]
            if index + 1 < len(elements) and elements[index + 1].tag == "p":
                index += 1
                parts.append(Paragraph(inline(elements[index]), caption))
            story.append(KeepTogether(parts))
        elif node.tag == "p":
            value = inline(node)
            if value.startswith("<em>Table"):
                style = ParagraphStyle("TableCaption", parent=caption, keepWithNext=True)
            else:
                style = reference if in_refs else body
            story.append(Paragraph(value, style))
        elif node.tag == "ol":
            for number, item in enumerate(node, 1):
                story.append(Paragraph(f"{number}. " + inline(item), body))
        elif node.tag == "table":
            rows = [[Paragraph(inline(c), cell) for c in row] for row in node.iter("tr")]
            if len(rows[0]) == 5:
                widths = [68, 75, 97, 102, 186]
            else:
                widths = [57, 31, 166, 90, 95, 89]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), .7, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), .5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]))
            story += [table, Spacer(1, 7)]
        else:
            raise ValueError(f"Unhandled document element: {node.tag}")
        index += 1

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Sans", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        if doc.page > 1:
            canvas.drawString(42, 766, "Runtime Cost and Bottlenecks of Quantized Llama 70B Serving")
        canvas.drawString(42, 25, "Short empirical study · September 2026")
        canvas.drawRightString(570, 25, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(HERE / "llama70b.pdf"), pagesize=(612, 792),
                            leftMargin=36, rightMargin=36, topMargin=42, bottomMargin=42,
                            title="Runtime Cost and Bottlenecks of Quantized Llama 70B Serving")
    doc.build(story, onFirstPage=page, onLaterPages=page)


def provenance():
    paths = [RUNTIME, DIAG / "controlled-timings.csv", DIAG / "dispatch-control.csv",
             DIAG / "primary-hardware-metrics.csv", DIAG / "summary.json",
             ROOT / "results/cuda-context-study/report/capacity.csv"]
    data = {"scope": "Existing 70B measurements only; this build launches no experiments.",
            "inputs": [{"path": str(p.relative_to(ROOT)),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    (HERE / "evidence.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    figures()
    render_pdf()
    provenance()
    print("Wrote paper/llama70b.pdf, two figures (PNG/PDF), and evidence.json.")
