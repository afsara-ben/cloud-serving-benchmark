#!/usr/bin/env python3
"""Build Llama 70B posters from validated baseline and continuation measurements."""
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
CONTINUATION = ROOT / "results/context-study/poster-70b-20260914/report"
Q4, Q2, FP16 = "#245ba8", "#c66b12", "#697586"
INK, MUTED = "#172536", "#576475"
FOOTER = ("Two RTX A6000 GPUs · FP16 KV · 512 outputs/request · prompt reuse off · K = 1,024 input tokens\n"
          "One run per setting; thermal limiting occurred. Small gaps do not establish a winner.")
CAPACITY_STATUSES = {"capacity_estimated", "observed_oom", "observed_headroom_limit"}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class Data:
    def __init__(self):
        self.paths = [REPORT / "runtime.csv", REPORT / "capacity-extension.csv"]
        if (CONTINUATION / "runtime.csv").exists():
            self.paths.append(CONTINUATION / "runtime.csv")
        self.rows = {}
        self.used = {}
        self.statuses = {}
        for path in self.paths:
            for row in read_csv(path):
                key = (row["model"], row["format"], int(row["concurrency"]), int(row["input_tokens"]))
                assert key not in self.rows, key
                self.rows[key] = (row, path)
        for report in (REPORT, CONTINUATION):
            path = report / "capacity.csv"
            if not path.exists():
                continue
            self.paths.append(path)
            for row in read_csv(path):
                key = (row["model"], row["format"], int(row["concurrency"]), int(row["input_tokens"]))
                previous = self.statuses.get(key)
                resolved = {"complete", "capacity_estimated", "observed_oom", "observed_headroom_limit"}
                if previous is None or (previous[0]["status"] not in resolved and row["status"] in resolved):
                    self.statuses[key] = (row, path)

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
            "raw_directory": str((path.parent.parent / row["evidence"]).resolve().relative_to(ROOT)),
            "gpu0_thermal_limit_observed": row["gpu0_thermal_limit_observed"],
            "gpu1_thermal_limit_observed": row["gpu1_thermal_limit_observed"],
        }
        return self.used[key]

    def value(self, model, quant, clients, prompt, field="output_tok_s"):
        return self.get(model, quant, clients, prompt)[field]

    def capacity(self, quant, clients, prompt):
        row, source = self.statuses[("70b", quant, clients, prompt)]
        path = (source.parent.parent / row["evidence"] / "capacity-estimate.json").resolve()
        if path not in self.paths:
            self.paths.append(path)
        result = json.loads(path.read_text())
        assert result["status"] == "capacity_estimated"
        return result

    def coverage(self):
        rows = []
        for quant, _, _, _ in LINE_FORMATS[:4]:
            for clients in (8, 16, 32):
                for prompt in LINE_PROMPTS:
                    key = ("70b", quant, clients, prompt)
                    row, source = self.statuses[key]
                    rows.append({"model": "70b", "format": quant, "clients": clients,
                                 "input_tokens": prompt, "status": row["status"],
                                 "reason": row["reason"],
                                 "source_csv": str(source.relative_to(ROOT)),
                                 "evidence": str((source.parent.parent / row["evidence"]).resolve().relative_to(ROOT))})
        return rows

    def require_70b_grid(self, allow_missing=False):
        unresolved = [r for r in self.coverage() if r["status"] not in
                      {"complete", "capacity_estimated", "observed_oom", "observed_headroom_limit"}]
        if unresolved and not allow_missing:
            raise ValueError(f"70B grid still has {len(unresolved)} unresolved settings; finish and report the continuation first.")
        for row in self.coverage():
            key = ("70b", row["format"], row["clients"], row["input_tokens"])
            assert (key in self.rows) == (row["status"] == "complete"), key

    def group_is_capacity_excluded(self, quant, clients):
        return all(self.statuses[("70b", quant, clients, p)][0]["status"] in CAPACITY_STATUSES
                   for p in LINE_PROMPTS)

    def availability_note(self):
        coverage = self.coverage()
        measured = sum(r["status"] == "complete" for r in coverage)
        return f"Available measurements: {measured}/48 settings · Missing and capacity-excluded values are unplotted; see coverage.csv."


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
             "70B · concurrency = 8 · percentages relative to Q4_K_M" if not hypothetical else
             "70B · concurrency = 8 · 64K cannot fit this FP16-KV setup")
    prompts = LINE_PROMPTS
    n = len(prompts)
    formats = (("Q4_K_M", Q4, "s", -.12), ("Q2_K", Q2, "o", .12)) if hypothetical else (
        ("IQ1_M", "#7143a5", "^", -.25), ("Q2_K", Q2, "o", 0), ("Q4_K_M", Q4, "s", .25))
    if not hypothetical:
        data.f1_series, data.f1_changes = [], []
    for quant, color, marker, offset in formats:
        values = [data.value("70b", quant, 8, p, "ttft_p95_s") for p in prompts]
        positions = [i + offset for i in range(n)]
        if hypothetical:
            ax.scatter(positions, values, s=100, marker=marker, color=color, label=quant, zorder=3)
        else:
            ax.bar(positions, values, width=.22, color=color, label=quant, zorder=3)
            for x, value in zip(positions, values):
                ax.text(x, value + 3, f"{value:.1f}", ha="center", va="bottom", fontsize=9,
                        color=color, fontweight="bold")
            data.f1_series.append({"model": "70b", "format": quant, "clients": 8,
                                   "input_tokens": list(prompts), "ttft_p95_s": values})
        if hypothetical:
            estimate = next(r for r in estimates if r["format"] == quant)["illustrative_ttft_p95_s"]
            ax.scatter([n + offset], [estimate], s=125, marker=marker,
                       facecolors="white", edgecolors=color, linewidths=2.2, zorder=3)
            ax.annotate(f"≈ {round(estimate / 10) * 10:,} s", (n + offset, estimate),
                        xytext=(-39, -31) if quant == "Q4_K_M" else (0, 18),
                        textcoords="offset points", ha="center", color=color,
                        fontsize=11, fontweight="bold")
    for i, p in enumerate(prompts):
        q4 = data.value("70b", "Q4_K_M", 8, p, "ttft_p95_s")
        q2 = data.value("70b", "Q2_K", 8, p, "ttft_p95_s")
        if hypothetical:
            ax.text(i, q2 * 1.27, f"+{(q2 / q4 - 1) * 100:.1f}%", ha="center", fontsize=11,
                    color=Q2, fontweight="bold")
        else:
            iq1 = data.value("70b", "IQ1_M", 8, p, "ttft_p95_s")
            top = max(q4, q2, iq1)
            for quant, label, value, color, gap in (("Q2_K", "Q2", q2, Q2, 21),
                                                    ("IQ1_M", "IQ1", iq1, "#7143a5", 37)):
                change = 100 * (value / q4 - 1)
                ax.text(i, top + gap, f"{label} {change:+.1f}%", ha="center", va="bottom",
                        fontsize=9, color=color, fontweight="bold")
                data.f1_changes.append({"format": quant, "reference": "Q4_K_M", "input_tokens": p,
                                        "ttft_change_pct": change})
    ax.set_xlabel("Context length per request")
    ax.set_ylabel("TTFT p95 (s)" + (" · logarithmic scale" if hypothetical else ""))
    labels = [f"{p // 1024}K" for p in prompts]
    ax.set_xticks(range(n + int(hypothetical)), labels + (["64K*\nILLUSTRATIVE"] if hypothetical else []))
    ax.set_xlim(-.5, n - .5 + int(hypothetical))
    if hypothetical:
        ax.set_yscale("log")
        ax.set_ylim(10, 2500)
        ax.set_yticks([10, 100, 1000])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.axvspan(n - .5, n + .5, color="#fff3ce", alpha=.8, zorder=0)
        ax.text(n, 13, "NOT MEASURED", ha="center", fontsize=10,
                fontweight="bold", color="#835512")
        ax.legend(loc="upper left", fontsize=10)
    else:
        ax.set_ylim(0, 285)
        ax.set_yticks([0, 50, 100, 150, 200, 250])
        ax.legend(loc="upper left", ncol=3, fontsize=9.5, columnspacing=1.2, handlelength=1.3)
        ax.tick_params(axis="both", labelcolor="#111111")
        for axis_label in (ax.xaxis.label, ax.yaxis.label):
            axis_label.set_color("#111111")
            axis_label.set_fontsize(axis_label.get_fontsize() + 2)


def draw_f2(ax, data):
    decorate(ax, "F2  Higher concurrency can lower throughput",
             "70B · color = quantization · line / marker = input length")
    maximum = 0
    data.f2_series = []
    prompts = [(2048, "o", "-"), (4096, "^", "--"),
               (8192, "s", ":"), (16384, "D", "-.")]
    formats = []
    for quant, color, _, _ in LINE_FORMATS[:4]:
        measured = False
        for prompt, marker, line in prompts:
            points = [data.value("70b", quant, c, prompt) if ("70b", quant, c, prompt) in data.rows
                      else None for c in (8, 16, 32)]
            data.f2_series.append({"model": "70b", "format": quant, "input_tokens": prompt,
                                   "clients": [8, 16, 32], "throughput_tok_s": points})
            available = [v for v in points if v is not None]
            if not available:
                continue
            measured = True
            maximum = max(maximum, *available)
            ax.plot([8, 16, 32], [v if v is not None else float("nan") for v in points], color=color,
                    marker=marker, linestyle=line, linewidth=2, markersize=6,
                    markerfacecolor="white" if prompt == 16384 else color, markeredgewidth=1.3)
        if measured:
            formats.append(Line2D([], [], color=color, linewidth=2.3, label=quant))
    ax.set_xlim(6, 34)
    ax.set_ylim(0, maximum * 1.48)
    ax.set_xticks([8, 16, 32])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Aggregate output throughput (tok/s)")
    quant_legend = ax.legend(handles=formats, loc="upper left", ncol=4, fontsize=9.5)
    ax.add_artist(quant_legend)
    prompt_handles = [Line2D([], [], color=INK, marker=marker, linestyle=line,
                             markerfacecolor="white" if prompt == 16384 else INK,
                             linewidth=1.8, markersize=5, label=f"{prompt // 1024}K")
                      for prompt, marker, line in prompts]
    ax.legend(handles=prompt_handles, loc="upper left", bbox_to_anchor=(0, .91), ncol=4,
              fontsize=9.5, columnspacing=1.5)
    iq8 = data.value("70b", "IQ1_M", 8, 2048)
    iq16 = data.value("70b", "IQ1_M", 16, 2048)
    ax.annotate(f"IQ1_M at 2K: {100 * (iq16 / iq8 - 1):+.1f}%\nconcurrency: 8 → 16", (16, iq16),
                xytext=(25.5, 40), textcoords="data", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#7143a5")
    missing = []
    for quant, _, _, _ in LINE_FORMATS[:4]:
        clients = [c for c in (8, 16, 32) if not any(("70b", quant, c, p) in data.rows for p in LINE_PROMPTS)]
        if not clients:
            continue
        if all(data.group_is_capacity_excluded(quant, c) for c in clients):
            missing.append(f"{quant}: C" + "/".join(map(str, clients)) + " VRAM-excluded")
        else:
            missing.append(f"{quant}: " + ("no measurements" if len(clients) == 3 else
                                           "C" + "/".join(map(str, clients)) + " not measured"))
    missing.append("16K: C8 only; other gaps listed in coverage.csv")
    ax.text(.98, .04, "\n".join(missing), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})


def draw_f3(ax, data):
    from matplotlib.patches import Patch
    decorate(ax, "F3  Throughput versus concurrency",
             "70B · bar labels: tok/s · top labels: change from C8 to C16")
    data.f3_series = []
    data.f3_changes = []
    for (quant, color, _, _), offset in zip(LINE_FORMATS[:3], (-.28, 0, .28)):
        pair_values = {}
        for clients, shift in ((8, -.065), (16, .065)):
            values = [data.value("70b", quant, clients, p) if ("70b", quant, clients, p) in data.rows
                      else None for p in LINE_PROMPTS]
            pair_values[clients] = values
            data.f3_series.append({"model": "70b", "format": quant, "clients": clients,
                                   "input_tokens": list(LINE_PROMPTS), "throughput_tok_s": values})
            positions = [i + offset + shift for i in range(len(LINE_PROMPTS))]
            measured = [(x, value) for x, value in zip(positions, values) if value is not None]
            ax.bar([x for x, _ in measured], [v for _, v in measured], width=.115, facecolor="white" if clients == 8 else color,
                   edgecolor=color, hatch="////" if clients == 8 else None, linewidth=1.1, zorder=3)
            for x, value in zip(positions, values):
                if value is None:
                    ax.text(x, 2, "N/A", ha="center", va="bottom", rotation=90,
                            fontsize=7, color=MUTED)
                    continue
                ax.text(x, value + 1.8, f"{value:.1f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=color)
        for i, prompt in enumerate(LINE_PROMPTS):
            c8, c16 = pair_values[8][i], pair_values[16][i]
            if c8 is None or c16 is None:
                continue
            change = 100 * (c16 / c8 - 1)
            data.f3_changes.append({"format": quant, "input_tokens": prompt, "c8_to_c16_pct": change})
            x, y = i + offset, max(c8, c16) + 8
            ax.plot([x - .065, x - .065, x + .065, x + .065], [y - 1, y, y, y - 1],
                    color=color, linewidth=.9, zorder=4)
            ax.text(x, y + 1, f"{change:+.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=color)
    ax.set_xticks(range(len(LINE_PROMPTS)), [f"{p // 1024}K" for p in LINE_PROMPTS])
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Context length per request")
    ax.set_ylabel("Aggregate output throughput (tok/s)")
    quant_legend = ax.legend(handles=[Patch(facecolor=color, label=quant) for quant, color, _, _ in LINE_FORMATS[:3]],
                            loc="upper left", ncol=3, fontsize=9.5, columnspacing=1.2, handlelength=1.4)
    ax.add_artist(quant_legend)
    ax.legend(handles=[Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="Concurrency = 8"),
                       Patch(facecolor=MUTED, label="Concurrency = 16")],
              loc="upper left", bbox_to_anchor=(0, .90), ncol=2, fontsize=9.5)
    data.capacity("Q4_K_M", 16, 16384)
    data.capacity("Q2_K", 16, 16384)
    iq_status = data.statuses[("70b", "IQ1_M", 16, 16384)][0]["status"]
    iq_reason = "VRAM-excluded" if iq_status in CAPACITY_STATUSES else "not measured"
    ax.text(.98, .96, "16K pairs unavailable: Q2/Q4 C16 exceed VRAM\nIQ1_M at C16/16K: " + iq_reason,
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)


def draw_f4(ax, data):
    decorate(ax, "F4  Context changes quantization gains",
             "70B · concurrency = 8 · IQ1_M versus Q4_K_M", grid="x")
    largest = 0
    for y, prompt in zip((3, 2, 1, 0), LINE_PROMPTS):
        iq1 = data.value("70b", "IQ1_M", 8, prompt)
        q4 = data.value("70b", "Q4_K_M", 8, prompt)
        gain = 100 * (iq1 / q4 - 1)
        largest = max(largest, iq1, q4)
        ax.plot([q4, iq1], [y, y], color="#aeb9c7", linewidth=3, zorder=2)
        ax.scatter([iq1], [y], s=85, color="#7143a5", zorder=3)
        ax.scatter([q4], [y], s=125, facecolor="none", edgecolor=Q4, linewidth=2.2, zorder=4)
        ax.annotate(f"{gain:+.1f}%", (max(q4, iq1), y), xytext=(14, 0),
                    textcoords="offset points", va="center", fontsize=13,
                    fontweight="bold", color="#7143a5" if gain >= 0 else Q4)
    ax.set_xlim(0, largest * 1.32)
    ax.set_ylim(-.45, 3.75)
    ax.set_yticks([3, 2, 1, 0], ["2K", "4K", "8K", "16K"])
    ax.set_xlabel("Aggregate output throughput (tok/s)")
    ax.set_ylabel("Context length per request")
    handles = [Line2D([], [], marker="o", linestyle="none", color="#7143a5", label="IQ1_M", markersize=7),
               Line2D([], [], marker="o", linestyle="none", color=Q4, markerfacecolor="white",
                      markeredgewidth=2, label="Q4_K_M", markersize=9)]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=10)
    ax.text(.98, .97, "Labels: IQ1_M / Q4_K_M − 1", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=MUTED)


def draw_capacity(ax, data):
    decorate(ax, "Available context measurements",
             "70B · largest validated input within the 2K–16K grid")
    from matplotlib.patches import Rectangle
    colors = {2048: "#e4edf6", 4096: "#b8d1e9", 8192: "#719fc9", 16384: "#245ba8"}
    for y, (quant, _, _, _) in enumerate(LINE_FORMATS[:4]):
        for x, clients in enumerate((8, 16, 32)):
            available = [p for p in LINE_PROMPTS if ("70b", quant, clients, p) in data.rows]
            maximum = max(available, default=None)
            if maximum:
                data.get("70b", quant, clients, maximum)
            ax.add_patch(Rectangle((x - .48, y - .46), .96, .92,
                                  facecolor=colors.get(maximum, "#edf0f3"), edgecolor="white"))
            absent = "VRAM\nexcluded" if data.group_is_capacity_excluded(quant, clients) else "Not\nmeasured"
            ax.text(x, y, f"{maximum // 1024}K" if maximum else absent,
                    ha="center", va="center", fontsize=20 if maximum else 11,
                    color="white" if maximum == 16384 else INK, fontweight="bold")
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(3.5, -.5)
    ax.set_xticks([0, 1, 2], ["8", "16", "32"])
    ax.set_yticks(range(4), [q for q, _, _, _ in LINE_FORMATS[:4]])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Quantization")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    ax.set_xlabel("Context length per request")


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
             "8B · fixed concurrency = 32 · FP16 baseline at each input length")
    decorate(axes[1], "Concurrency Scaling × Context",
             "8B · throughput change from concurrency 8 → 16")
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
    axes[1].set_ylabel("Throughput change, concurrency 8 → 16")
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
             "Separate 70B result · 2K input · concurrency = 8\n"
             f"Q2 vs. Q4: +{ttft:.0f}% TTFT p95, +{tpot:.0f}% TPOT p95",
             fontsize=12, linespacing=1.7, bbox=box)
    fig.text(xright, .126,
             "8B IQ1_M @ 2K: dip and recovery\n"
             f"{recovery[0]:.0f} → {recovery[1]:.0f} → {recovery[2]:.0f} tok/s (concurrency: 8 → 16 → 32)",
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


CONCURRENCY_STYLES = [(8, "o", "-"), (16, "^", "--"), (32, "s", ":")]


def draw_merged_throughput(ax, data):
    decorate(ax, "F2–F4  Throughput across prompt lengths",
             "Llama 70B only · color = quantization · line / marker = concurrency")
    series, maximum = [], 0
    for quant, color, _, _ in LINE_FORMATS[:4]:
        for clients, marker, line in CONCURRENCY_STYLES:
            points = [data.value("70b", quant, clients, p)
                      if ("70b", quant, clients, p) in data.rows else None for p in LINE_PROMPTS]
            if not any(v is not None for v in points):
                continue
            values = [v if v is not None else float("nan") for v in points]
            maximum = max(maximum, *(v for v in points if v is not None))
            ax.plot(LINE_PROMPTS, values, color=color, marker=marker, linestyle=line,
                    linewidth=2.1, markersize=7, markerfacecolor="white" if clients == 32 else color,
                    markeredgewidth=1.5, zorder=3)
            series.append({"model": "70b", "format": quant, "clients": clients,
                           "input_tokens": list(LINE_PROMPTS), "throughput_tok_s": points})
    context_axis(ax)
    ax.set_xlim(1750, 19500)
    ax.set_ylim(0, maximum * 1.14)
    ax.set_xlabel("Prompt length per request (tokens)")
    ax.set_ylabel("Aggregate output throughput (tok/s)")
    formats = [Line2D([], [], color=color, linewidth=2.5,
                     label=quant if any(s["format"] == quant for s in series) else quant + " (not measured)")
               for quant, color, _, _ in LINE_FORMATS[:4]]
    clients = [Line2D([], [], color=INK, marker=marker, linestyle=line,
                      markerfacecolor="white" if count == 32 else INK,
                      linewidth=2, label=f"Concurrency = {count}")
               for count, marker, line in CONCURRENCY_STYLES]
    legend = ax.legend(handles=formats, loc="upper center", bbox_to_anchor=(.5, -.17),
                       ncol=4, fontsize=10.5, columnspacing=1.5, handlelength=2)
    ax.add_artist(legend)
    ax.legend(handles=clients, loc="upper center", bbox_to_anchor=(.5, -.26),
              ncol=3, fontsize=10.5, columnspacing=2, handlelength=2.7)
    iq1 = [data.value("70b", "IQ1_M", c, 2048) if ("70b", "IQ1_M", c, 2048) in data.rows
           else None for c in (8, 16, 32)]
    gains = [{"input_tokens": p, "c8_to_c16_pct": 100 * (
              data.value("70b", "Q4_K_M", 16, p) / data.value("70b", "Q4_K_M", 8, p) - 1)}
             for p in LINE_PROMPTS[:3]]
    quantization_gains = [{"input_tokens": p, "iq1_over_q4_pct": 100 * (
                          data.value("70b", "IQ1_M", 8, p) / data.value("70b", "Q4_K_M", 8, p) - 1)}
                         for p in LINE_PROMPTS]
    excluded = [r for r in data.coverage() if r["status"] in CAPACITY_STATUSES]
    missing = [r for r in data.coverage() if r["status"] not in CAPACITY_STATUSES | {"complete"}]
    q8_maximum = {c: max((p for p in LINE_PROMPTS if ("70b", "Q8_0", c, p) in data.rows), default=None)
                  for c in (8, 16, 32)}
    return {"model": "70b", "series": series, "x_axis": "prompt tokens, log2 spacing",
            "y_axis": "aggregate output tokens/s, linear scale",
            "f2": {"model": "70b", "format": "IQ1_M", "input_tokens": 2048,
                   "clients": [8, 16, 32], "throughput_tok_s": iq1},
            "f3": {"model": "70b", "format": "Q4_K_M", "clients": [8, 16], "gains": gains},
            "f4": {"model": "70b", "clients": 8, "formats": ["IQ1_M", "Q4_K_M"], "gains": quantization_gains},
            "capacity": {"q8_maximum_input_tokens": q8_maximum,
                         "q8_groups_excluded": {c: data.group_is_capacity_excluded("Q8_0", c) for c in (8, 16, 32)}},
            "unplotted_capacity_exclusions": excluded, "unplotted_missing_settings": missing}


def merged_notes(fig, x, y, details, fontsize=11):
    values = details["f2"]["throughput_tok_s"]
    fig.text(x, y, "F2  IQ1_M at 2K, concurrency = 8 / 16 / 32: "
             + " / ".join(f"{v:.1f} tok/s" if v is not None else "not measured" for v in values),
             fontsize=fontsize, color=INK)
    gains = ", ".join(f"{r['c8_to_c16_pct']:+.1f}% at {r['input_tokens'] // 1024}K"
                      for r in details["f3"]["gains"])
    fig.text(x, y - .029, "F3  Q4_K_M gain from concurrency 8 → 16: " + gains,
             fontsize=fontsize, color=INK)
    ratios = ", ".join(f"{r['iq1_over_q4_pct']:+.1f}% at {r['input_tokens'] // 1024}K"
                       for r in details["f4"]["gains"])
    fig.text(x, y - .058,
             "F4  IQ1_M / Q4_K_M throughput change at concurrency = 8: " + ratios,
             fontsize=fontsize, color=INK)
    limits = "; ".join(f"{p // 1024}K at concurrency = {c}" if p else
                       f"Concurrency = {c}: VRAM-excluded" if details["capacity"]["q8_groups_excluded"][c] else
                       f"Concurrency = {c}: not measured"
                       for c, p in details["capacity"]["q8_maximum_input_tokens"].items())
    if all(p is None for p in details["capacity"]["q8_maximum_input_tokens"].values()):
        limits = "no measured values at concurrency = 8, 16 or 32"
    fig.text(x, y - .087,
             "Q8_0: " + limits + ". Missing/capacity-excluded values are unplotted.",
             fontsize=fontsize - .5, color=MUTED)


def merged_figures(data):
    fig, ax = plt.subplots(figsize=(12.6, 9.5))
    fig.subplots_adjust(left=.095, right=.95, bottom=.37, top=.85)
    details = draw_merged_throughput(ax, data)
    merged_notes(fig, .095, .145, details, fontsize=10)
    fig.text(.095, .022, FOOTER, fontsize=9, color=MUTED, linespacing=1.6)
    save(fig, HERE / "figures" / "f2-f4-throughput-context")

    fig, axes = plt.subplots(1, 2, figsize=(19, 9.6), gridspec_kw={"width_ratios": [1, 1.65]})
    fig.subplots_adjust(left=.06, right=.97, bottom=.34, top=.77, wspace=.26)
    fig.text(.06, .963, "Runtime Cost · Llama 70B", fontsize=29, fontweight="bold")
    fig.text(.06, .919, "Latency and throughput across quantization, concurrency and input length",
             fontsize=16, color=MUTED)
    fig.text(.06, .879, data.availability_note(), fontsize=11, color=MUTED)
    draw_f1(axes[0], data)
    draw_merged_throughput(axes[1], data)
    merged_notes(fig, .06, .151, details)
    fig.text(.06, .026, FOOTER, fontsize=10, color=MUTED, linespacing=1.6)
    save(fig, HERE / "runtime-cost-merged")
    return details


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + suffix), dpi=240, bbox_inches="tight", pad_inches=.18,
                    facecolor="white")
        if suffix == "svg":
            svg = path.with_suffix(".svg")
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def measured_figures(data):
    for name, draw in CHARTS:
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        fig.subplots_adjust(left=.12, right=.95, bottom=.18, top=.80)
        draw(ax, data)
        fig.text(.12, .03, FOOTER, fontsize=8.5, color=MUTED, linespacing=1.55)
        save(fig, HERE / "figures" / name)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    fig.subplots_adjust(left=.15, right=.95, bottom=.18, top=.80)
    draw_capacity(ax, data)
    fig.text(.15, .03, FOOTER, fontsize=8.5, color=MUTED, linespacing=1.55)
    save(fig, HERE / "figures" / "context-capacity")
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12))
    fig.subplots_adjust(left=.075, right=.96, bottom=.10, top=.84, wspace=.29, hspace=.54)
    fig.text(.075, .973, "Runtime Cost · Llama 70B", fontsize=29, fontweight="bold")
    fig.text(.075, .944, "Four views of measured 70B serving · IQ1_M, Q2_K and Q4_K_M · Q8_0 not yet measured"
             if not any(key[0:2] == ("70b", "Q8_0") for key in data.rows) else
             "Four views of measured 70B serving · IQ1_M, Q2_K, Q4_K_M and Q8_0", fontsize=15, color=MUTED)
    fig.text(.075, .916, data.availability_note(), fontsize=10.5, color=MUTED)
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
             "At concurrency = 8, 64K FP16 KV alone needs 161.875 GiB; this setting was excluded before launch.",
             fontsize=9.2, color="#835512", linespacing=1.65)
    save(fig, HERE / "illustrative" / "f1-with-64k-extrapolation")
    (HERE / "illustrative" / "estimates.json").write_text(json.dumps(estimates, indent=2) + "\n")


def provenance(data, include_illustration, two_line=None, merged=None):
    doc = {
        "scope": "Four Llama 70B plots from independently validated baseline and September 14 continuation measurements.",
        "optional_64k_illustration_generated": include_illustration,
        "main_poster_contains_synthetic_values": False,
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in data.paths],
        "selected_measured_rows": list(data.used.values()),
        "f1_comparison": {"model": "70b", "clients": 8, "formats": ["IQ1_M", "Q2_K", "Q4_K_M"],
                          "metric": "TTFT p95 in seconds", "percentage_labels": "100 * (format / Q4_K_M - 1), for IQ1_M and Q2_K"},
        "f3_formula": "100 * (throughput_C16 / throughput_C8 - 1)",
        "f4_formula": "100 * (throughput_IQ1_M / throughput_Q4_K_M - 1), at concurrency = 8 and the same input length.",
        "capacity_summary_definition": "Maximum input_tokens with a validated complete run, within the requested 2K/4K/8K/16K grid, for each format and concurrency.",
    }
    if two_line is not None:
        doc["scope"] = "Two line plots from existing serving measurements; no inference or profiling launched."
        doc["two_line_layout"] = two_line
        doc["f4_formula"] = "100 * (throughput_Q4_K_M / throughput_FP16 - 1)"
    if merged is not None:
        doc["scope"] = "Llama 70B only: F1 latency plus F2–F4 throughput across context, from validated baseline and continuation runs."
        doc["merged_layout"] = merged
    if hasattr(data, "f1_series"):
        doc["f1_series"] = data.f1_series
        doc["f1_changes"] = data.f1_changes
    if hasattr(data, "f2_series"):
        doc["f2_series"] = data.f2_series
        doc["f2_encoding"] = {"x": "concurrency", "y": "aggregate output tokens/s",
                              "color": "weight quantization", "line_and_marker": "input tokens per request"}
    if hasattr(data, "f3_series"):
        doc["f3_series"] = data.f3_series
        doc["f3_changes"] = data.f3_changes
        doc["f3_formula"] = "Bar heights: raw output_tokens_per_second. Top labels: 100 * (throughput_C16 / throughput_C8 - 1)."
    if two_line is None:
        assert all(row["model"] == "70b" for row in data.used.values()) or include_illustration
        coverage = data.coverage()
        doc["coverage"] = coverage
        doc["grid_complete"] = all(r["status"] in CAPACITY_STATUSES | {"complete"} for r in coverage)
        doc["missing_values_policy"] = "Absent measurements are labeled and never interpolated or plotted as zero. Only recorded capacity exclusions are classified as such."
        doc["continuation_launch"] = "results/context-study/poster-70b-20260914/launch.json"
        doc["measurement_notes"] = [
            "One validated run per setting; 2C requests after a discarded C-request warmup.",
            "Same pinned server binary, workload and GPU placement; runner revisions are recorded separately.",
            "Baseline and continuation were collected in separate sessions under automatic clocks.",
            "Capacity exclusions are not zero-throughput measurements. No synthetic values are plotted."]
        with (HERE / "coverage.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
            writer.writeheader()
            writer.writerows(coverage)
        lines = ["# Llama 70B inference coverage", "",
                 "Aggregate output tokens/s. One validated run per setting, 512 output tokens per request, FP16 KV and no prompt reuse.", "",
                 "| Format | Concurrency | 2K | 4K | 8K | 16K |", "|---|---:|---:|---:|---:|---:|"]
        for quant, _, _, _ in LINE_FORMATS[:4]:
            for clients in (8, 16, 32):
                cells = []
                for prompt in LINE_PROMPTS:
                    key = ("70b", quant, clients, prompt)
                    if key in data.rows:
                        row, source = data.rows[key]
                        cells.append(f"{float(row['output_tokens_per_second_mean']):.2f}")
                    else:
                        status = data.statuses[key][0]["status"]
                        cells.append("VRAM estimate" if status == "capacity_estimated" else
                                     "Not measured" if status == "missing" else status)
                lines.append("| " + " | ".join([quant, str(clients), *cells]) + " |")
        lines += ["", "Per-setting statuses and evidence paths: [coverage.csv](coverage.csv).",
                  "Not measured means no validated result exists. Some of these settings are expected to exceed VRAM, but their per-cell capacity screen had not run when the queue was paused.",
                  "VRAM estimate means a saved per-cell capacity exclusion exists; it is not an observed OOM or a throughput measurement.",
                  "Baseline and continuation are separate sessions on the same two RTX A6000 GPUs. Excluded settings have no throughput value.", ""]
        (HERE / "coverage.md").write_text("\n".join(lines))
    filename = "merged-data.json" if merged is not None else "two-line-data.json" if two_line is not None else "data.json"
    (HERE / filename).write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("four", "two-line", "merged"), default="four",
                        help="Choose four plots, normalized two-line plots, or F1 plus merged F2–F4 throughput.")
    parser.add_argument("--include-illustration", action="store_true",
                        help="Also render a separate, explicitly synthetic 70B 64K illustration.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Render available 70B measurements with explicit missing-value labels.")
    args = parser.parse_args()
    if args.layout != "four" and args.include_illustration:
        parser.error("The separate 64K illustration belongs to --layout four.")
    theme()
    data = Data()
    if args.layout != "two-line":
        data.require_70b_grid(allow_missing=args.allow_missing)
    if args.layout == "merged":
        details = merged_figures(data)
        provenance(data, False, merged=details)
        print("Wrote F1 plus merged F2–F4 poster, standalone throughput figure, and merged-data.json.")
    elif args.layout == "two-line":
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
