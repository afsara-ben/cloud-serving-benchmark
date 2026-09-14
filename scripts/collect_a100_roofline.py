#!/usr/bin/env python3
"""Collect scalar FP32 operator rooflines on A100 SXM 80GB or RTX PRO 6000.

Use --hardware rtxpro6000 for SM120 and ceilings selected from the saved GPU
edition. A100 remains the default. Both paths use one-GPU, not summed, ceilings.

Select completed 70B serving cells at the same input length and concurrency.
plan prints commands only; run captures serially; report reads saved captures.
Requires the annotated SM80 server built by benchmark/matrix_diagnostic.py.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

import profile_a100_setting as profiler
from capacity import layer_devices
from cuda_hardware import HARDWARE, rated_roof

ROOT = Path(__file__).resolve().parents[1]
DATASHEET = "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf"
PEAK = 19.5e12
BANDWIDTH = 2039e9
OPCODES = "sass__thread_inst_executed_true_per_opcode"
OPERATORS = {
    "ffn_gate_up": ("FFN gate/up", ("blk.*.ffn_gate.weight", "blk.*.ffn_up.weight")),
    "ffn_down": ("FFN down", ("blk.*.ffn_down.weight",)),
    "attn_q": ("Q projection", ("blk.*.attn_q.weight",)),
    "attn_k": ("K projection", ("blk.*.attn_k.weight",)),
    "attn_v": ("V projection", ("blk.*.attn_v.weight",)),
    "attn_output": ("Attention output", ("blk.*.attn_output.weight",)),
    "flash_attention": ("FlashAttention", ("op.FLASH_ATTN_EXT",)),
    "lm_head": ("LM head", ("output.weight",)),
}
COORDINATES = ("fadd", "fmul", "ffma", "fp32_flops", "duration_s", "dram_read_bytes",
               "dram_write_bytes", "dram_bytes", "fp32_flops_per_byte", "fp32_flops_per_s")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def operation_regex(name):
    if name == "flash_attention":
        return r"csb_op:role=node_[0-9]+:type=[^:]+:op=FLASH_ATTN_EXT:.*"
    roles = [re.escape(role).replace(r"\.\*\.", r"\.[0-9]+\.") for role in OPERATORS[name][1]]
    return "csb_op:role=(" + "|".join(roles) + "):.*"


def number(item):
    value = item.get("value")
    if (item.get("available") is not True or item.get("ambiguous")
            or isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("Missing, ambiguous or invalid counter")
    return value


def measured(launch):
    metrics = launch["metrics"]
    instances = launch.get("metric_instances", {}).get(OPCODES, [])
    if not instances:
        raise ValueError("Missing predicated-on SASS thread instruction instances")
    counts = {}
    for item in instances:
        opcode = item["instance"]
        if opcode in counts:
            raise ValueError("Duplicate SASS opcode instance")
        counts[opcode] = number(item)
    # Only a complete inventory justifies treating absent opcodes as zero.
    if not math.isclose(sum(counts.values()), number(metrics[OPCODES]), rel_tol=1e-6, abs_tol=1):
        raise ValueError("SASS opcode inventory does not match its aggregate count")
    duration_metric = metrics["gpu__time_duration.sum"]
    scales = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.}
    if duration_metric.get("unit") not in scales:
        raise ValueError("Unknown kernel-duration unit")
    duration = number(duration_metric) * scales[duration_metric["unit"]]
    byte_values = []
    for name in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
        if metrics[name].get("unit") not in ("byte", "bytes"):
            raise ValueError("DRAM counters must be in bytes")
        byte_values.append(number(metrics[name]))
    traffic = sum(byte_values)
    if duration <= 0 or traffic <= 0:
        raise ValueError("Duration and DRAM traffic must be positive")
    add, mul, fma = (counts.get(name, 0) for name in ("FADD", "FMUL", "FFMA"))
    work = add + mul + 2 * fma
    return dict(zip(COORDINATES, (add, mul, fma, work, duration, *byte_values,
                                  traffic, work / traffic, work / duration)))


def make_plan(args):
    jobs, cells, formats = [], [], set()
    workload = None
    hardware = getattr(args, "hardware", "a100")
    roof = {"hardware": "A100 SXM 80GB", "peak_fp32_flops_per_s": PEAK,
            "peak_dram_bytes_per_s": BANDWIDTH, "roof_source": DATASHEET, "ceilings_are_per_gpu": True}
    for directory in args.cells:
        cell, devices = profiler.load_cell(directory, hardware) if hardware != "a100" else profiler.load_cell(directory)
        if hardware == "rtxpro6000":
            gpu_rows = profiler.validate_rtx_cell(cell, devices)
            roofs = [rated_roof(d["name"]) for d in gpu_rows]
            if any(item != roofs[0] for item in roofs):
                raise ValueError("Use the same GPU edition on both devices for this comparison")
            if cells and roofs[0] != roof:
                raise ValueError("Source cells have different GPU editions")
            roof = roofs[0]
        quant = cell["quantization"]
        if (cell.get("model_family") != "Llama-3.3-70B-Instruct" or cell.get("model", {}).get("layers") != 80
                or quant not in ("IQ1_M", "Q2_K", "Q4_K_M", "Q8_0")):
            raise ValueError("Select 70B IQ1_M/Q2_K/Q4_K_M/Q8_0 serving cells")
        if quant in formats:
            raise ValueError("Select one cell per format; use separate outputs for different workloads")
        formats.add(quant)
        cfg = cell["configuration"]
        if any(cfg.get(key) != "f16" for key in ("SERVER_CACHE_TYPE_K", "SERVER_CACHE_TYPE_V")):
            raise ValueError("Expected FP16 KV in the saved serving cell")
        shape = (cell["concurrency"], cell["prompt_tokens"], devices, cfg["SERVER_TENSOR_SPLIT"])
        if workload is not None and workload != shape:
            raise ValueError("Cells must share concurrency, input length, visible GPUs and layer split")
        workload = shape
        split = [float(value) for value in cfg["SERVER_TENSOR_SPLIT"].split(",")]
        if len(split) != len(devices):
            raise ValueError("GPU visibility and layer split differ")
        output_gpu = layer_devices(cell["model"]["layers"], split)[-1]
        cell_path = Path(cell["cell_file"])
        cells.append({"path": str(cell_path), "sha256": hashlib.sha256(cell_path.read_bytes()).hexdigest(),
                      "format": quant})
        for phase in args.phases:
            for gpu in range(len(devices)):
                for operator in args.operators:
                    if operator == "lm_head" and gpu != output_gpu:
                        continue
                    relative = Path("captures") / quant / phase / f"gpu{gpu}" / operator
                    command = [sys.executable, str(ROOT / "scripts/profile_a100_setting.py"), "counters",
                               "--cell", str(cell_path.parent), "--phase", phase, "--device", str(gpu),
                               "--scalar-roofline", "--operation-regex", operation_regex(operator),
                               "--kernel-regex", "flash_attn" if operator == "flash_attention" else "mul_mat|gemm|gemv|mma",
                               "--launch-count", str(args.launch_count), "--output", str(args.output / relative),
                               "--diagnostic-server", str(args.diagnostic_server.resolve()), "--ncu", args.ncu]
                    if hardware != "a100":
                        command += ["--hardware", hardware]
                    jobs.append({"format": quant, "phase": phase, "gpu": gpu, "physical_gpu": devices[gpu],
                                 "operator": operator, "capture": str(relative), "command": command})
    return {"schema_version": 1, "hardware_family": hardware, "source_cells": cells,
            "concurrency": workload[0], "prompt_tokens": workload[1], "devices": workload[2],
            "output_tokens": 512, "phases": args.phases, "operators": args.operators,
            "launch_cap_per_operator_phase_gpu": args.launch_count,
            "warmup_requests_per_capture": workload[0], "profiled_requests_per_capture": workload[0],
            **roof, "jobs": jobs}


def export_values(output, plan):
    samples, coverage = [], []
    for job in plan["jobs"]:
        directory = output / job["capture"]
        status = {key: job[key] for key in ("format", "phase", "gpu", "operator", "capture")}
        try:
            marker, metadata = read(directory / "study-capture.json"), read(directory / "metadata.json")
            if (marker.get("exit_code") != 0 or metadata.get("status") != "captured"
                    or metadata.get("counter_status") != "collected"):
                raise ValueError("Capture did not pass profiler validation")
            source_cell = next(cell for cell in plan["source_cells"] if cell["format"] == job["format"])
            if hashlib.sha256(Path(source_cell["path"]).read_bytes()).hexdigest() != source_cell["sha256"]:
                raise ValueError("Source serving cell changed after capture planning")
            settings = marker.get("settings", {})
            expected_filter = f"regex:csb_batch:phase={job['phase']}:.*/*/{operation_regex(job['operator'])}"
            if (marker.get("source_cell") != source_cell["path"]
                    or settings.get("PROFILE_NVTX_INCLUDE") != expected_filter
                    or settings.get("PROFILE_DEVICES") != str(job["gpu"])):
                raise ValueError("Saved capture source or filters differ from the roofline plan")
            path = directory / "counter_summary.json"
            launches = read(path)["launches"]
            if not launches:
                raise ValueError("No captured kernel launches")
            selected = []
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for launch in launches:
                op = launch.get("operation", {})
                if (str(launch["device"]) != str(job["gpu"]) or op.get("phase") != job["phase"]
                        or op.get("operation_match") != "nvtx_same_capture"
                        or op.get("role") not in OPERATORS[job["operator"]][1]):
                    raise ValueError("Counter GPU, phase or operator differs from requested capture")
                values = measured(launch)
                selected.append({**{key: job[key] for key in ("format", "phase", "gpu", "physical_gpu", "operator")},
                                 **{key: op.get(key, "") for key in ("role", "type", "m", "n", "k", "batch_width", "fusion")},
                                 **{key: launch.get(key, "") for key in ("kernel", "grid", "block")},
                                 "launch_id": launch["id"], **values,
                                 "source": str(path.relative_to(output)), "source_sha256": source_hash})
            samples.extend(selected)
            status.update(status="collected", samples=len(selected))
        except (OSError, ValueError, KeyError, TypeError) as error:
            status.update(status="unavailable", reason=str(error))
        coverage.append(status)
    groups = collections.defaultdict(list)
    keys = ("format", "phase", "gpu", "physical_gpu", "operator", "role", "type", "m", "n", "k",
            "batch_width", "fusion", "kernel", "grid", "block", "source")
    for sample in samples:
        groups[tuple(sample[key] for key in keys)].append(sample)
    points = [dict(zip(keys, identity)) | {key: statistics.median(row[key] for row in rows) for key in COORDINATES}
              | {"samples": len(rows)} for identity, rows in groups.items()]
    for name, rows, fields in (("roofline-samples.csv", samples, [*keys, "launch_id", *COORDINATES, "source_sha256"]),
                               ("roofline-points.csv", points, [*keys, *COORDINATES, "samples"])):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    audit = {"status": "complete" if all(row["status"] == "collected" for row in coverage) else "incomplete",
             "captures": coverage, "captured_launches": len(samples), "configuration_groups": len(points),
             "zero_fp32_groups_not_plotted": sum(point["fp32_flops"] == 0 for point in points),
             "formulas": {"F": "FADD + FMUL + 2*FFMA (predicated-on thread instructions)",
                          "B": "DRAM read bytes + DRAM write bytes", "x": "F/B", "y": "F/duration_s",
                          "roof": "min(per_GPU_FP32_peak, x * per_GPU_DRAM_bandwidth)"},
             "excludes": "FP16, Tensor Core, integer and special-function work; these are profiler timings"}
    write(output / "roofline-validation.json", audit)
    return points, audit


def plot_values(output, plan, points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import numpy as np

    formats = [cell["format"] for cell in plan["source_cells"]]
    peak, bandwidth = plan["peak_fp32_flops_per_s"], plan["peak_dram_bytes_per_s"]
    styles = {name: (f"C{i}", marker) for i, (name, marker) in enumerate(zip(OPERATORS, "o^sDvh*P"))}
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})
    with PdfPages(output / "roofline-explained.pdf") as pdf:
        for gpu, physical in enumerate(plan["devices"]):
            fig, axes = plt.subplots(len(plan["phases"]), len(formats), squeeze=False,
                                     figsize=(max(10, 5 * len(formats)), 4 * len(plan["phases"]) + 1.5))
            for row, phase in enumerate(plan["phases"]):
                for col, quant in enumerate(formats):
                    ax = axes[row, col]
                    selected = [p for p in points if (p["format"], p["phase"], p["gpu"]) == (quant, phase, gpu)
                                and p["fp32_flops"] > 0]
                    intensities = [p["fp32_flops_per_byte"] for p in selected] + [peak / bandwidth]
                    rates = [p["fp32_flops_per_s"] / 1e12 for p in selected] + [peak / 1e12]
                    x = np.geomspace(min(intensities) / 5, max(intensities) * 5, 300)
                    ax.loglog(x, x * bandwidth / 1e12, "--", color="#447D98")
                    ax.axhline(peak / 1e12, color="#537650")
                    ax.axvline(peak / bandwidth, color="#B64D4D", linestyle=":")
                    for point in selected:
                        color, marker = styles[point["operator"]]
                        ax.scatter(point["fp32_flops_per_byte"], point["fp32_flops_per_s"] / 1e12,
                                   color=color, marker=marker, s=50, zorder=3)
                    present = {p["operator"] for p in selected}
                    absent = [OPERATORS[name][0] for name in plan["operators"] if name not in present]
                    ax.text(.02, .02, "No positive FP32 point: " + ", ".join(absent) if absent else "",
                            transform=ax.transAxes, fontsize=7, wrap=True)
                    ax.set(xlim=(x[0], x[-1]), ylim=(min(rates) / 5, max(rates) * 3),
                           title=f"70B {quant} · {phase}", xlabel="Scalar FP32 FLOPs / DRAM byte",
                           ylabel="Scalar FP32 TFLOP/s")
                    ax.grid(which="major", alpha=.25)
            handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=OPERATORS[name][0])
                       for name, (color, marker) in styles.items() if name in plan["operators"]]
            handles += [Line2D([], [], color="#447D98", linestyle="--", label="Rated DRAM ceiling"),
                        Line2D([], [], color="#537650", label="Rated FP32 ceiling"),
                        Line2D([], [], color="#B64D4D", linestyle=":", label="Compute/memory boundary")]
            fig.suptitle(f"{plan['hardware']} · physical GPU {physical} · C{plan['concurrency']} · "
                         f"{plan['prompt_tokens']} input / 512 output tokens", fontsize=15)
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045), ncol=4)
            fig.text(.03, .018, f"Per-GPU rated ceilings: {peak / 1e12:g} TFLOP/s; {bandwidth / 1e9:g} GB/s. "
                     "Medians per kernel/shape/capture. Tensor Core, FP16 and integer work excluded.", fontsize=8)
            fig.tight_layout(rect=(0, .14, 1, .95))
            pdf.savefig(fig)
            for extension in ("png", "svg"):
                fig.savefig(output / f"roofline-explained-gpu{gpu}.{extension}", dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("plan", "run", "report"), default="plan")
    parser.add_argument("--hardware", choices=HARDWARE, default="a100")
    parser.add_argument("--cells", nargs="+", type=Path, help="Completed serving cell directories, one per format")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operators", nargs="+", choices=tuple(OPERATORS), default=list(OPERATORS))
    parser.add_argument("--phases", nargs="+", choices=("prefill", "decode"), default=["prefill", "decode"])
    parser.add_argument("--launch-count", type=int, default=5, help="Maximum launches per operator/phase/GPU capture")
    parser.add_argument("--diagnostic-server", type=Path)
    parser.add_argument("--ncu", default=profiler.os.environ.get("NCU_BIN", "ncu"))
    args = parser.parse_args()
    args.diagnostic_server = args.diagnostic_server or ROOT / f".run/{args.hardware}-profile-build/baseline/llama-server"
    args.output = args.output.resolve()
    if args.action == "report":
        plan = read(args.output / "roofline-plan.json")
    else:
        if not args.cells or args.launch_count < 1:
            parser.error("Select --cells and a positive --launch-count")
        if len(set(args.operators)) != len(args.operators) or len(set(args.phases)) != len(args.phases):
            parser.error("Operator and phase selections must be unique")
        plan = make_plan(args)
        if args.action == "plan":
            print(json.dumps(plan | {"capture_count": len(plan["jobs"]), "inference_executed": False}, indent=2))
            return
        # Rated ceilings are valid for this exact hardware; never reuse A6000 constants.
        names = subprocess.check_output(["nvidia-smi", "-i", ",".join(plan["devices"]),
                                         "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
        if args.hardware == "a100":
            if len(names) != len(plan["devices"]) or any(not re.search(r"A100.*SXM.*80GB", name) for name in names):
                raise ValueError(f"Expected A100 SXM 80GB GPUs for the rated ceilings; received {names}")
        elif len(names) != len(plan["devices"]) or any(rated_roof(name)["hardware"] != plan["hardware"] for name in names):
            raise ValueError("Actual GPU edition differs from the saved serving hardware")
        profiler.validate_build(args.diagnostic_server.resolve(), args.hardware)
        profiler.validate_sections(args.ncu, ["InstructionStats"])
        path = args.output / "roofline-plan.json"
        if path.exists() and read(path) != plan:
            raise ValueError("Capture plan changed; choose a fresh --output")
        if args.output.exists() and not path.exists() and any(args.output.iterdir()):
            raise ValueError("Choose an empty output directory or resume its matching roofline-plan.json")
        args.output.mkdir(parents=True, exist_ok=True)
        write(path, plan)
        for job in plan["jobs"]:
            print(f"{job['format']} {job['phase']} GPU{job['gpu']} {job['operator']}", flush=True)
            result = subprocess.run(job["command"], cwd=ROOT)
            if result.returncode:
                export_values(args.output, plan)
                raise RuntimeError("Capture failed; partial CSVs saved. Inspect its logs; no automatic retry.")
    points, audit = export_values(args.output, plan)
    plot_values(args.output, plan, points)
    print(f"{audit['status']}: {len(points)} operator/configuration points; {args.output / 'roofline-explained.pdf'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
