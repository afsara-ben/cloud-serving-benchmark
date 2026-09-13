#!/usr/bin/env python3
"""Build poster plots from saved measurements; optional 64k illustration stays separate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "results/cuda-context-study/report"
Q4, Q2, FP16 = "#245ba8", "#c66b12", "#697586"
INK, MUTED = "#172536", "#576475"
FOOTER = ("Two RTX A6000 GPUs · FP16 KV · 512 outputs/request · prompt reuse off · K = 1,024 input tokens\n"
          "Measured points: one run per setting; thermal limiting occurred in some settings.")


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class Data:
    def __init__(self):
        self.paths = [REPORT / "runtime.csv", REPORT / "capacity-extension.csv"]
        self.rows = {}
        self.used = {}
        for path in self.paths:
            for row in read_csv(path):
                key = (row["model"], row["format"], int(row["concurrency"]), int(row["input_tokens"]))
                assert key not in self.rows, key
                self.rows[key] = (row, path)

    def get(self, model, quant, clients, prompt):
        key = (model, quant, clients, prompt)
        row, path = self.rows[key]
        assert row["output_tokens"] == "512" and row["repetitions"] == "1"
        assert row["failed_requests"] == "0"
        self.used[key] = {
            "model": model, "format": quant, "clients": clients, "input_tokens": prompt,
            "output_tokens": 512, "requests": int(row["successful_requests"]),
            "output_tok_s": float(row["output_tokens_per_second_mean"]),
            "ttft_p95_s": float(row["ttft_p95_ms"]) / 1000,
            "tpot_p95_ms": float(row["tpot_p95_ms"]),
            "source_csv": str(path.relative_to(ROOT)),
            "raw_directory": str((REPORT.parent / row["evidence"]).resolve().relative_to(ROOT)),
            "gpu0_thermal_limit_observed": row["gpu0_thermal_limit_observed"],
            "gpu1_thermal_limit_observed": row["gpu1_thermal_limit_observed"],
        }
        return self.used[key]

    def value(self, model, quant, clients, prompt, field="output_tok_s"):
        return self.get(model, quant, clients, prompt)[field]

    def capacity(self, quant, clients, prompt):
        path = ROOT / f"results/cuda-context-study-70b/{quant}/c{clients}/p{prompt}/capacity-estimate.json"
        if path not in self.paths:
            self.paths.append(path)
        result = json.loads(path.read_text())
        assert result["status"] == "capacity_estimated"
        return result


def theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#b9c1ca", "axes.linewidth": .8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "legend.frameon": False, "legend.fontsize": 11,
    })


def decorate(ax, title, subtitle, grid="y"):
    ax.set_title(title, loc="left", fontsize=18, pad=37)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=11.5,
            color=MUTED, ha="left", va="bottom")
    ax.grid(axis=grid, color="#e5e9ee", linewidth=.8)
    ax.set_axisbelow(True)


def draw_f1(ax, data, estimates=None):
    hypothetical = estimates is not None
    decorate(ax, "F1  Lower bits ≠ lower latency" if not hypothetical else
             "F1  64K illustration — not measured",
             "70B · 8 clients · paired Q2_K / Q4_K_M comparison" if not hypothetical else
             "70B · 8 clients · 64K cannot fit this FP16-KV setup")
    prompts = (2048, 8192, 16384)
    for quant, color, marker, offset in (("Q4_K_M", Q4, "s", -.12), ("Q2_K", Q2, "o", .12)):
        values = [data.value("70b", quant, 8, p, "ttft_p95_s") for p in prompts]
        ax.scatter([i + offset for i in range(3)], values, s=100,
                   marker=marker, color=color, label=quant, zorder=3)
        if hypothetical:
            estimate = next(r for r in estimates if r["format"] == quant)["illustrative_ttft_p95_s"]
            ax.scatter([3 + offset], [estimate], s=125, marker=marker,
                       facecolors="white", edgecolors=color, linewidths=2.2, zorder=3)
            ax.annotate(f"≈ {round(estimate / 10) * 10:,} s", (3 + offset, estimate),
                        xytext=(-39, -31) if quant == "Q4_K_M" else (0, 18),
                        textcoords="offset points", ha="center", color=color,
                        fontsize=11, fontweight="bold")
    for i, p in enumerate(prompts):
        q4 = data.value("70b", "Q4_K_M", 8, p, "ttft_p95_s")
        q2 = data.value("70b", "Q2_K", 8, p, "ttft_p95_s")
        ax.text(i, q2 * 1.27 if hypothetical else q2 + 12,
                f"+{(q2 / q4 - 1) * 100:.1f}%", ha="center", fontsize=12,
                color=Q2, fontweight="bold")
    ax.set_xlabel("Input length per request")
    ax.set_ylabel("TTFT p95 (s)" + (" · logarithmic scale" if hypothetical else ""))
    ax.set_xticks(range(4 if hypothetical else 3),
                  ["2K", "8K", "16K", "64K*\nILLUSTRATIVE"] if hypothetical else ["2K", "8K", "16K"])
    ax.set_xlim(-.5, 3.5 if hypothetical else 2.5)
    if hypothetical:
        ax.set_yscale("log")
        ax.set_ylim(10, 2500)
        ax.set_yticks([10, 100, 1000])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.axvspan(2.5, 3.5, color="#fff3ce", alpha=.8, zorder=0)
        ax.text(3, 13, "NOT MEASURED", ha="center", fontsize=10,
                fontweight="bold", color="#835512")
        ax.legend(loc="upper left", fontsize=10)
    else:
        ax.set_ylim(0, 240)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.legend(loc="upper left")
        q4 = data.value("70b", "Q4_K_M", 8, 2048, "tpot_p95_ms")
        q2 = data.value("70b", "Q2_K", 8, 2048, "tpot_p95_ms")
        ax.text(.035, .62, f"2K TPOT also +{(q2 / q4 - 1) * 100:.1f}%\n(TPOT not plotted)",
                transform=ax.transAxes, fontsize=11, color=MUTED)


def draw_f2(ax, data):
    decorate(ax, "F2  More concurrency ≠ more throughput", "8B · IQ1_M · input lengths shown as separate lines")
    styles = [(2048, "#7143a5", "o", "-", 3), (4096, "#a384bc", "s", "--", 1.8),
              (8192, "#398f98", "^", "-.", 1.8), (16384, "#7e8b9e", "D", ":", 1.8)]
    for p, color, marker, line, width in styles:
        values = [data.value("8b", "IQ1_M", c, p) for c in (8, 16, 32)]
        ax.plot([8, 16, 32], values, label=f"{p // 1024}K", color=color,
                marker=marker, linestyle=line, linewidth=width, markersize=6)
    baseline = data.value("8b", "IQ1_M", 8, 2048)
    dip = data.value("8b", "IQ1_M", 16, 2048)
    recovery = data.value("8b", "IQ1_M", 32, 2048)
    ax.annotate(f"8 → 16 clients: {(dip / baseline - 1) * 100:.1f}%", (16, dip + 9),
                xytext=(18, 397), ha="center", fontsize=13, fontweight="bold",
                color="#7143a5", arrowprops={"arrowstyle": "->", "color": "#7143a5", "lw": 1.3})
    for x, y, offset in ((8, baseline, (17, 13)), (16, dip, (0, -24)), (32, recovery, (-6, 14))):
        ax.annotate(f"{y:.1f}", (x, y), xytext=offset, textcoords="offset points",
                    ha="center", fontsize=11, color="#7143a5")
    ax.set_xlim(6.5, 34)
    ax.set_ylim(0, 435)
    ax.set_xticks([8, 16, 32])
    ax.set_xlabel("Concurrent clients")
    ax.set_ylabel("Aggregate output throughput (tok/s)")
    ax.legend(loc="lower left", ncol=4, fontsize=10, columnspacing=.9, handlelength=1.4)


def draw_f3(ax, data):
    decorate(ax, "F3  Long context weakens concurrency scaling", "70B · Q4_K_M · throughput gain from 8 → 16 clients")
    gains = [100 * (data.value("70b", "Q4_K_M", 16, p) /
                    data.value("70b", "Q4_K_M", 8, p) - 1)
             for p in (2048, 4096, 8192)]
    ax.bar(range(3), gains, width=.56, color=[Q4, "#5682bd", "#8daacf"], zorder=3)
    for i, gain in enumerate(gains):
        ax.text(i, gain + 3, f"+{gain:.1f}%", ha="center", fontsize=16, fontweight="bold")
    ax.set_xticks(range(3), ["2K", "4K", "8K"])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Input length per request")
    ax.set_ylabel("Throughput gain, 8 → 16 clients")
    data.capacity("Q4_K_M", 16, 16384)
    ax.text(.98, .96, "16K omitted:\n16 clients exceed VRAM", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=MUTED)


def draw_f4(ax, data):
    decorate(ax, "F4  Long context erodes quantization speedup", "8B · 32 clients · FP16 versus Q4_K_M", grid="x")
    for y, p in zip((3, 2, 1, 0), (2048, 4096, 8192, 16384)):
        fp = data.value("8b", "FP16", 32, p)
        q4 = data.value("8b", "Q4_K_M", 32, p)
        ax.plot([fp, q4], [y, y], color="#aeb9c7", linewidth=3, zorder=2)
        ax.scatter([fp], [y], s=78, color=FP16, zorder=3)
        ax.scatter([q4], [y], s=125, facecolor="none", edgecolor=Q4, linewidth=2.2, zorder=4)
        ax.text(q4 + 17, y, f"+{(q4 / fp - 1) * 100:.1f}%", va="center", fontsize=13,
                fontweight="bold", color=Q4)
    ax.set_xlim(0, 700)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])
    ax.set_ylim(-.45, 3.7)
    ax.set_yticks([3, 2, 1, 0], ["2K", "4K", "8K", "16K"])
    ax.set_xlabel("Aggregate output throughput (tok/s)")
    ax.set_ylabel("Input length per request")
    handles = [Line2D([], [], marker="o", linestyle="none", color=FP16, label="FP16", markersize=7),
               Line2D([], [], marker="o", linestyle="none", color=Q4, markerfacecolor="white",
                      markeredgewidth=2, label="Q4_K_M", markersize=9)]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=10)


CHARTS = [("f1-precision-latency", draw_f1), ("f2-concurrency-throughput", draw_f2),
          ("f3-concurrency-gain", draw_f3), ("f4-quantization-gap", draw_f4)]

LINE_FORMATS = [("IQ1_M", "#7143a5", "o", "-"),
                ("Q2_K", Q2, "^", "-"),
                ("Q4_K_M", Q4, "s", "-"),
                ("Q8_0", "#398f98", "D", "-."),
                ("FP16", FP16, "P", "--")]
LINE_PROMPTS = (2048, 4096, 8192, 16384)


def context_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(LINE_PROMPTS, ["2K", "4K", "8K", "16K"])
    ax.set_xlim(1800, 18400)
    ax.set_xlabel("Input length per request")


def two_line_figures(data):
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 9.2))
    fig.subplots_adjust(left=.075, right=.96, bottom=.365, top=.745, wspace=.30)
    fig.text(.075, .968, "Runtime Cost", fontsize=29, fontweight="bold")
    fig.text(.075, .928, "Quantization and concurrency across context · two plots, four findings",
             fontsize=16, color=MUTED)
    handles = [Line2D([], [], color=color, marker=marker, linestyle=line, linewidth=2,
                     markersize=6, label=quant) for quant, color, marker, line in LINE_FORMATS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .899),
               ncol=5, fontsize=12, columnspacing=2.8, handlelength=2.5)

    normalized_rows, scaling_rows = [], []
    decorate(axes[0], "Quantization × Context",
             "8B · fixed 32 clients · FP16 baseline at each input length")
    decorate(axes[1], "Concurrency Scaling × Context",
             "8B · throughput change from 8 → 16 clients")
    for quant, color, marker, line in LINE_FORMATS:
        normalized, gains = [], []
        for prompt in LINE_PROMPTS:
            throughput = data.value("8b", quant, 32, prompt)
            baseline = data.value("8b", "FP16", 32, prompt)
            t8 = data.value("8b", quant, 8, prompt)
            t16 = data.value("8b", quant, 16, prompt)
            normalized.append(throughput / baseline)
            gains.append(100 * (t16 / t8 - 1))
            normalized_rows.append({"format": quant, "input_tokens": prompt,
                                    "throughput_tok_s": throughput, "fp16_tok_s": baseline,
                                    "normalized_to_fp16": normalized[-1]})
            scaling_rows.append({"format": quant, "input_tokens": prompt,
                                 "throughput_c8_tok_s": t8, "throughput_c16_tok_s": t16,
                                 "change_pct": gains[-1]})
        for ax, values in zip(axes, (normalized, gains)):
            ax.plot(LINE_PROMPTS, values, color=color, marker=marker, linestyle=line,
                    linewidth=2.4 if quant in ("IQ1_M", "Q4_K_M") else 1.9,
                    markersize=6.5, label=quant)
    for ax in axes:
        context_axis(ax)
    axes[0].set_ylim(.65, 1.30)
    axes[0].set_yticks([.7, .8, .9, 1, 1.1, 1.2, 1.3],
                      ["0.7×", "0.8×", "0.9×", "1.0×", "1.1×", "1.2×", "1.3×"])
    axes[0].set_ylabel("Output throughput / FP16 throughput")
    axes[0].text(.03, (1.01 - .65) / .65, "FP16 = 1.0×", transform=axes[0].transAxes,
                 fontsize=10, color=FP16)
    axes[1].set_ylim(-40, 70)
    axes[1].set_yticks([-40, -20, 0, 20, 40, 60])
    axes[1].yaxis.set_major_formatter(PercentFormatter())
    axes[1].set_ylabel("Throughput change, 8 → 16 clients")
    axes[1].axhspan(-40, 0, color="#f5f0fa", zorder=0)
    axes[1].axhline(0, color="#77818e", linewidth=1.2, zorder=2)

    xleft, xright = axes[0].get_position().x0, axes[1].get_position().x0
    for x, first, second in ((xleft, "F1  Lower bits ≠ better runtime",
                             "F4  Long context erodes quantization gains"),
                            (xright, "F2  More concurrency ≠ more throughput",
                             "F3  Long context weakens concurrency scaling")):
        fig.text(x, .265, first, fontsize=13, fontweight="bold")
        fig.text(x, .23, second, fontsize=13, fontweight="bold")

    q2 = data.get("70b", "Q2_K", 8, 2048)
    q4 = data.get("70b", "Q4_K_M", 8, 2048)
    ttft = 100 * (q2["ttft_p95_s"] / q4["ttft_p95_s"] - 1)
    tpot = 100 * (q2["tpot_p95_ms"] / q4["tpot_p95_ms"] - 1)
    recovery = [data.value("8b", "IQ1_M", c, 2048) for c in (8, 16, 32)]
    box = {"boxstyle": "round,pad=.65", "facecolor": "#f0f3f7", "edgecolor": "none"}
    fig.text(xleft, .126,
             "Separate 70B result · 2K input · 8 clients\n"
             f"Q2 vs. Q4: +{ttft:.0f}% TTFT p95, +{tpot:.0f}% TPOT p95",
             fontsize=12, linespacing=1.7, bbox=box)
    fig.text(xright, .126,
             "8B IQ1_M @ 2K: dip and recovery\n"
             f"{recovery[0]:.0f} → {recovery[1]:.0f} → {recovery[2]:.0f} tok/s (8 → 16 → 32 clients)",
             fontsize=12, linespacing=1.7, bbox=box)
    fig.text(.075, .025, FOOTER, fontsize=10, color=MUTED, linespacing=1.6)
    save(fig, HERE / "runtime-cost-two-line")
    return {"plot1": {"model": "8b", "fixed_clients": 32,
                       "formula": "throughput_format / throughput_FP16 at the same input length",
                       "points": normalized_rows},
            "plot2": {"model": "8b", "clients": [8, 16],
                       "formula": "100 * (throughput_C16 / throughput_C8 - 1)",
                       "points": scaling_rows},
            "separate_70b_latency_annotation": {"clients": 8, "input_tokens": 2048,
                                                "q2_vs_q4_ttft_change_pct": ttft,
                                                "q2_vs_q4_tpot_change_pct": tpot},
            "iq1_recovery_annotation": {"model": "8b", "input_tokens": 2048,
                                        "clients": [8, 16, 32], "throughput_tok_s": recovery}}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + suffix), dpi=240, bbox_inches="tight", pad_inches=.18,
                    facecolor="white")
    plt.close(fig)


def measured_figures(data):
    for name, draw in CHARTS:
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        fig.subplots_adjust(left=.12, right=.95, bottom=.18, top=.80)
        draw(ax, data)
        fig.text(.12, .03, FOOTER, fontsize=8.5, color=MUTED, linespacing=1.55)
        save(fig, HERE / "figures" / name)
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12))
    fig.subplots_adjust(left=.075, right=.96, bottom=.10, top=.84, wspace=.29, hspace=.54)
    fig.text(.075, .973, "Runtime Cost", fontsize=29, fontweight="bold")
    fig.text(.075, .944, "Four findings from quantized Llama serving", fontsize=16, color=MUTED)
    for ax, (_, draw) in zip(axes.flat, CHARTS):
        draw(ax, data)
    fig.text(.075, .028, FOOTER, fontsize=11, color=MUTED, linespacing=1.6)
    save(fig, HERE / "runtime-cost-poster")


def illustration(data):
    estimates = []
    for quant in ("Q4_K_M", "Q2_K"):
        donor16 = data.get("8b", quant, 8, 16384)
        donor64 = data.get("8b", quant, 8, 65536)
        anchor = data.get("70b", quant, 8, 16384)
        capacity = data.capacity(quant, 8, 65536)
        kv_gib = sum(g["kv_bytes"] for g in capacity["gpus"]) / 2 ** 30
        ratio = donor64["ttft_p95_s"] / donor16["ttft_p95_s"]
        estimates.append({
            "kind": "illustrative_extrapolation_NOT_MEASURED", "model": "70b", "format": quant,
            "clients": 8, "input_tokens": 65536, "output_tokens": 512,
            "formula": "70B_16K_TTFT_p95 * (8B_64K_TTFT_p95 / 8B_16K_TTFT_p95)",
            "illustrative_ttft_p95_s": anchor["ttft_p95_s"] * ratio,
            "donor_ratio": ratio, "anchor": anchor, "donor_16k": donor16, "donor_64k": donor64,
            "capacity_status": "excluded before launch", "calculated_fp16_kv_gib": kv_gib,
            "limitations": ["Not a measurement or a validated prediction.",
                            "Eight-client 64K configuration exceeds device VRAM before adding weights.",
                            "Donor 16K and 64K runs use 16 and 8 requests, respectively.",
                            "Cross-model scaling does not preserve kernel, scheduling, or attention costs."],
        })
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    fig.subplots_adjust(left=.12, right=.95, bottom=.24, top=.82)
    draw_f1(ax, data, estimates)
    fig.text(.12, .025,
             "* ILLUSTRATIVE ONLY. 70B 64K = 70B 16K × (8B 64K / 8B 16K), per format.\n"
             "Donor bursts differ: 16 versus 8 requests. These values cannot support a measured finding.\n"
             "At 8 clients, 64K FP16 KV alone needs 161.875 GiB; this setting was excluded before launch.",
             fontsize=9.2, color="#835512", linespacing=1.65)
    save(fig, HERE / "illustrative" / "f1-with-64k-extrapolation")
    (HERE / "illustrative" / "estimates.json").write_text(json.dumps(estimates, indent=2) + "\n")


def provenance(data, include_illustration, two_line=None):
    doc = {
        "scope": "Four poster plots of existing serving measurements; no inference or profiling launched.",
        "optional_64k_illustration_generated": include_illustration,
        "main_poster_contains_synthetic_values": False,
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in data.paths],
        "selected_measured_rows": list(data.used.values()),
        "f3_formula": "100 * (throughput_C16 / throughput_C8 - 1)",
        "f4_formula": "100 * (throughput_Q4_K_M / throughput_FP16 - 1)",
    }
    if two_line is not None:
        doc["scope"] = "Two line plots from existing serving measurements; no inference or profiling launched."
        doc["two_line_layout"] = two_line
    filename = "two-line-data.json" if two_line is not None else "data.json"
    (HERE / filename).write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("four", "two-line"), default="four",
                        help="Choose the original four-plot poster or the alternate two-line layout.")
    parser.add_argument("--include-illustration", action="store_true",
                        help="Also render a separate, explicitly synthetic 70B 64K illustration.")
    args = parser.parse_args()
    if args.layout == "two-line" and args.include_illustration:
        parser.error("The separate 64K illustration belongs to --layout four.")
    theme()
    data = Data()
    if args.layout == "two-line":
        details = two_line_figures(data)
        provenance(data, False, two_line=details)
        print("Wrote alternate two-line poster (PDF/PNG/SVG) and two-line-data.json.")
    else:
        measured_figures(data)
        if args.include_illustration:
            illustration(data)
        provenance(data, args.include_illustration)
        print("Wrote four individual figures and the measured poster (PNG/PDF/SVG).")
        if args.include_illustration:
            print("Separate 64K illustration written with extrapolation, capacity, and protocol labels.")
