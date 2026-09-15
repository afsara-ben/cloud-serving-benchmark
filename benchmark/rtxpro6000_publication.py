"""Publication figures from one validated RTX PRO 6000 study.

No original A6000 data or interpretations are inputs. No GPU jobs are launched.
Missing evidence stops complete builds; poster snapshots may explicitly opt in.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import report_context_study as context
import report_serving as serving
from capacity import layer_devices
from cuda_hardware import rated_roof

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"IQ1_M": "#8b5fbf", "Q2_K": "#d47916", "Q4_K_M": "#247a49", "Q8_0": "#3479aa"}
Q4, Q2 = "Q4_K_M", "Q2_K"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def destination(path, root, kind):
    path = path.resolve()
    for historical in (ROOT / "paper/poster", ROOT / "paper/runtime-bottlenecks-report"):
        if path == historical or path.is_relative_to(historical):
            raise ValueError("Use a separate publication output, outside the historical paper directories")
    marker = {"study_root": str(root.resolve()), "kind": kind}
    owner = path / "publication-source.json"
    if path.exists() and any(path.iterdir()) and (not owner.is_file() or read(owner) != marker):
        raise ValueError(f"Output belongs to another publication; choose an empty directory: {path}")
    if path.exists() and any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Publication output must not contain symbolic links")
    path.mkdir(parents=True, exist_ok=True)
    write(owner, marker)
    (path / "figures").mkdir(exist_ok=True)
    return path


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save_figure(figure, path):
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(path.with_suffix("." + suffix), dpi=170)


def change(before, after):
    return 100 * (after / before - 1) if before else None


def percentage(value):
    return "undefined" if value is None else f"{value:+.1f}%"


def actual_value(value):
    return f"{value:.3g}" if 0 < abs(value) < .01 else f"{value:,.2f}"


def serving_data(root, allow_missing=False):
    audit = serving.build_report(root, plots=False)
    if audit["scope"].get("hardware") != "rtxpro6000":
        raise ValueError("Select an RTX PRO 6000 serving scope")
    if not allow_missing and audit["status"] != "complete":
        raise ValueError(f"Serving grid has {len(audit['unresolved_cells'])} unresolved cells; see serving-report/completion-audit.json")
    runtime = context.read_csv(root / "serving-report/runtime.csv")
    rows, fixtures, sources = [], {}, {}
    for row in runtime:
        folder = (root / row["evidence"]).resolve()
        # report_serving has already revalidated these raw requests/resources.
        raw_path = folder / "r1/raw.json"
        raw = read(raw_path)
        c, p = int(row["concurrency"]), int(row["input_tokens"])
        hashes = [r["input_sha256"] for r in sorted(raw["requests"], key=lambda r: r["request_index"])]
        if not all(hashes) or len(hashes) != 2 * c:
            raise ValueError(f"Missing prompt hashes or unmatched request count: {folder}")
        if (c, p) in fixtures and fixtures[c, p] != hashes:
            raise ValueError(f"Input prompts differ across formats at C{c}/P{p}")
        fixtures[c, p] = hashes
        rows.append({"format": row["format"], "concurrency": c, "input_tokens": p,
                     "ttft_p95_s": float(row["ttft_p95_ms"]) / 1000,
                     "output_tok_s": float(row["output_tokens_per_second_mean"]),
                     "tpot_p95_ms": float(row["tpot_p95_ms"]), "source": str(raw_path),
                     "thermal_limit_observed": any(row.get(f"gpu{gpu}_thermal_limit_observed") == "True"
                         for gpu in audit["scope"]["device_indices"])})
        for path in (raw_path, folder / "cell.json", folder / "r1/resources.json", folder / "r1/placement.json",
                     folder / "r1/telemetry.csv", folder / "warmup-discarded.json"):
            sources[str(path)] = sha(path)
    manifest = read(root / "70b/manifest.json") if (root / "70b/manifest.json").exists() else {}
    for filename in ("serving-scope.json", "70b/manifest.json", "70b/progress.json", "serving-report/runtime.csv", "serving-report/capacity.csv"):
        path = root / filename
        if path.is_file():
            sources[str(path)] = sha(path)
    return {"rows": rows, "coverage": context.read_csv(root / "serving-report/capacity.csv"),
            "audit": audit, "devices": manifest.get("devices", []), "sources": sources}


def render_poster(data, output, source_roots):
    scope = data["audit"]["scope"]
    prompts = scope["prompt_lengths"] + scope["capacity_lengths"]
    clients = scope["concurrencies"]
    formats = scope["model_formats"]["70b"]
    lookup = {(r["format"], r["concurrency"], r["input_tokens"]): r for r in data["rows"]}
    plt = plotting()
    import numpy as np
    from matplotlib.lines import Line2D
    x = np.arange(len(prompts))

    def value(q, c, p, metric="output_tok_s"):
        return lookup.get((q, c, p), {}).get(metric, math.nan)

    def bars(axis, positions, values, width, annotations=None, **kwargs):
        plotted = axis.bar(positions, values, width, **kwargs)
        axis.bar_label(plotted, labels=annotations or [f"{v:.1f}" if math.isfinite(v) else "" for v in values], fontsize=6, padding=2)
        for pos, val in zip(positions, values):
            if not math.isfinite(val):
                axis.text(pos, .01, "N/A", transform=axis.get_xaxis_transform(), ha="center", rotation=90, fontsize=6)

    def f1(ax):
        width = .8 / len(formats)
        for i, q in enumerate(formats):
            ys = [value(q, 8, p, "ttft_p95_s") for p in prompts]
            annotations = []
            for p, y in zip(prompts, ys):
                reference = value(Q4, 8, p, "ttft_p95_s")
                label = f"{y:.1f}" if math.isfinite(y) else ""
                if q != Q4 and math.isfinite(y) and math.isfinite(reference):
                    label += "\n" + percentage(change(reference, y))
                annotations.append(label)
            bars(ax, x + (i - (len(formats) - 1) / 2) * width,
                 ys, width, annotations, color=COLORS[q], label=q)
        ax.set(title="F1  TTFT at C8 (percentages relative to Q4_K_M)", ylabel="TTFT p95 (s)", xticks=x,
               xticklabels=[f"{p // 1024}K" for p in prompts], xlabel="Input tokens")
        ax.legend(fontsize=7, ncol=2); ax.margins(y=.2)

    def f2(ax):
        for q in formats:
            for i, p in enumerate(prompts):
                ys = [value(q, c, p) for c in clients]
                if any(math.isfinite(v) for v in ys):
                    ax.plot(clients, ys, color=COLORS[q], marker="o^sDv"[i], linestyle=["-", "--", ":", "-.", "--"][i])
        handles = [Line2D([], [], color=COLORS[q], label=q) for q in formats]
        handles += [Line2D([], [], color="#555", marker="o^sDv"[i], linestyle="none", label=f"{p // 1024}K") for i, p in enumerate(prompts)]
        ax.legend(handles=handles, fontsize=6, ncol=3)
        ax.set(title="F2  Throughput versus concurrent clients", xlabel="Concurrent clients", ylabel="Output tokens/s", xticks=clients)

    def f3(ax):
        selected = ["IQ1_M", Q2, Q4]
        width = .8 / 6
        for i, q in enumerate(selected):
            for j, c in enumerate((8, 16)):
                ys = [value(q, c, p) for p in prompts]
                bars(ax, x + (i * 2 + j - 2.5) * width, ys, width, color=COLORS[q],
                     hatch="///" if c == 8 else "", alpha=.65 if c == 8 else 1, label=f"{q} C{c}")
            for index, p in enumerate(prompts):
                a, b = value(q, 8, p), value(q, 16, p)
                if math.isfinite(a) and math.isfinite(b):
                    ax.annotate(percentage(change(a, b)), (x[index] + (i * 2 - 2) * width, max(a, b)),
                                xytext=(0, 18), textcoords="offset points", ha="center", fontsize=6)
        ax.set(title="F3  Throughput at 8 and 16 clients", ylabel="Output tokens/s", xticks=x,
               xticklabels=[f"{p // 1024}K" for p in prompts], xlabel="Input tokens")
        ax.legend(fontsize=6, ncol=3); ax.margins(y=.25)

    def f4(ax):
        for i, p in enumerate(prompts):
            a, b = value(Q4, 8, p), value("IQ1_M", 8, p)
            if math.isfinite(a) and math.isfinite(b):
                ax.plot([a, b], [i, i], color="#a2a9b2", linewidth=2)
                ax.text(max(a, b), i - .13, " " + percentage(change(a, b)), fontsize=8)
        for q in (Q4, "IQ1_M"):
            ax.scatter([value(q, 8, p) for p in prompts], x, color=COLORS[q], label=q)
        ax.set(title="F4  IQ1_M throughput relative to Q4_K_M at C8", xlabel="Output tokens/s", yticks=x,
               yticklabels=[f"{p // 1024}K" for p in prompts], ylabel="Input tokens")
        ax.invert_yaxis(); ax.margins(x=.35, y=.2); ax.legend(fontsize=7)

    state = "validated grid" if data["audit"]["status"] == "complete" else "INCOMPLETE SNAPSHOT"
    gpu_names = " / ".join(dict.fromkeys(d["name"] for d in data["devices"])) or "RTX PRO 6000 (no measured hardware yet)"
    footer = (f"2 GPUs: {gpu_names}\nFP16 KV; 512 outputs/request; C warmup + 2C measured requests at every input length; one run/setting. "
              "Missing/excluded cells have no point.\n")
    if any(r["thermal_limit_observed"] for r in data["rows"]):
        footer += "Thermal limiting observed in saved measurements. "
    footer += "Ratios do not establish statistical significance; k = 1024."
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for draw, ax in zip((f1, f2, f3, f4), axes.flat):
        draw(ax); ax.set_axisbelow(True); ax.grid(axis="y", alpha=.15)
        if draw != f4:
            ax.set_ylim(bottom=0)
    fig.suptitle(f"Runtime Cost - Llama 70B - {state}", fontsize=17)
    fig.text(.5, .012, footer, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .075, 1, .95), h_pad=3)
    save_figure(fig, output / "runtime-cost-poster"); plt.close(fig)
    for name, draw in zip(("f1-precision-latency", "f2-concurrency-throughput", "f3-concurrency-gain", "f4-quantization-gap"), (f1, f2, f3, f4)):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        draw(ax); fig.tight_layout()
        save_figure(fig, output / "figures" / name); plt.close(fig)
    csv_write(output / "coverage.csv", data["coverage"])
    write(output / "data.json", data)
    write(output / "evidence.json", {"study_roots": [str(root) for root in source_roots],
                                     "sources": data["sources"], "builder_sha256": sha(Path(__file__))})
    return output / "runtime-cost-poster.pdf"


def build_poster(root, output=None, allow_missing=False):
    root = root.resolve()
    data = serving_data(root, allow_missing)
    output = destination(output or root / "paper/poster", root, "runtime-cost-poster")
    return render_poster(data, output, [root])


def build_poster_exports(export_roots, output=None):
    """Combine four independently validated per-format serving exports."""
    paths, indexes = [], []
    for root in map(Path, export_roots):
        root = root.resolve()
        index_path = root / "index.json"
        index = read(index_path)
        recorded = {item["path"]: item["sha256"] for item in index.get("files", [])}
        selected = sorted(root.glob("*-serving-r1.json"))
        if any(recorded.get(path.name) != sha(path) for path in selected):
            raise ValueError(f"Serving export hash does not match {index_path}")
        paths.extend(selected); indexes.append(index_path)
    documents = [read(path) for path in paths]
    formats = [document.get("format") for document in documents]
    if set(formats) != set(serving.FORMATS["70b"]) or len(formats) != len(set(formats)):
        raise ValueError("Exports must contain exactly one serving JSON for each 70B format")
    if any(document.get("hardware") != "rtxpro6000" or document.get("repetitions") != 1
           or len(document.get("coverage", [])) != 15 for document in documents):
        raise ValueError("Serving export protocol or 15-cell format coverage is incomplete")
    names = [{device.get("name") for device in document.get("devices", [])} for document in documents]
    if any(len(group) != 1 for group in names) or len(set.union(*names)) != 1:
        raise ValueError("All shard exports must use the same RTX PRO 6000 edition")
    rows = [row for document in documents for row in document["rows"]]
    keys = {(row["format"], row["concurrency"], row["input_tokens"]) for row in rows}
    coverage = [row for document in documents for row in document["coverage"]]
    coverage_keys = {(row["format"], int(row["concurrency"]), int(row["input_tokens"])) for row in coverage}
    resolved = {"complete", "capacity_estimated", "observed_oom", "observed_headroom_limit"}
    if len(keys) != len(rows) or len(coverage) != 60 or len(coverage_keys) != 60 or any(row["status"] not in resolved for row in coverage):
        raise ValueError("Combined serving exports do not provide a resolved unique 60-cell grid")
    scope = serving.scope_document(["0", "1"], "rtxpro6000")
    sources = {str(path): sha(path) for path in paths + indexes}
    data = {"rows": rows, "coverage": coverage,
            "audit": {"status": "complete", "scope": scope},
            "devices": [device for document in documents for device in document["devices"]], "sources": sources}
    provenance_root = ROOT / "results/rtxpro6000-exports"
    output = destination(output or provenance_root / "combined-poster", provenance_root, "runtime-cost-poster-from-exports")
    return render_poster(data, output, [Path(root).resolve() for root in export_roots])


METRICS = {
    "instructions": "sm__inst_executed.sum",
    "dram_reads": "dram__bytes_read.sum",
    "dram_writes": "dram__bytes_write.sum",
    "tensor_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "alu_pct": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "ipc": "sm__inst_executed.avg.per_cycle_active",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers": "launch__registers_per_thread",
    "register_blocks": "launch__occupancy_limit_registers",
}


def number(item):
    value = item.get("value")
    if (item.get("available") is not True or item.get("ambiguous") or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
        raise ValueError("Unavailable, ambiguous or invalid counter")
    return value


def duration_s(launch):
    item = launch["metrics"]["gpu__time_duration.sum"]
    factors = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1}
    if item.get("unit") not in factors or number(item) <= 0:
        raise ValueError("Invalid kernel duration/unit")
    return number(item) * factors[item["unit"]]


def opcode_counts(launch):
    name = "sass__inst_executed_per_opcode"
    items = launch.get("metric_instances", {}).get(name, [])
    counts = {item["instance"]: number(item) for item in items}
    total = number(launch["metrics"]["sm__inst_executed.sum"])
    if (not items or len(counts) != len(items) or
            not math.isclose(sum(counts.values()), number(launch["metrics"][name]), rel_tol=1e-6, abs_tol=1) or
            not math.isclose(sum(counts.values()), total, rel_tol=1e-6, abs_tol=1)):
        raise ValueError("Opcode inventory does not reconcile with executed instructions")
    # Complete, reconciled inventories justify zero for an absent opcode.
    return counts


def tensor_coordinates(launch):
    """Apply the installed section's work/peak expressions per arithmetic path.

    Blackwell uses op_imma/op_hmma names. Never substitute FP4 marketing peaks
    or assume the Ampere INT8 name. Ignore overlapping aggregate path counters
    when operation-specific counters are present.
    """
    metrics = launch["metrics"]
    suffix = ".sum.per_cycle_elapsed"
    names = [name for name in metrics if name.startswith("sm__ops_path_tensor_") and name.endswith(suffix)]
    specific = [name for name in names if name.startswith("sm__ops_path_tensor_op_")]
    if specific:
        names = specific
    if not names:
        raise ValueError("No Tensor Core arithmetic path counters were exported")
    hz = number(metrics["sm__cycles_elapsed.avg.per_second"])
    traffic = number(metrics["dram__bytes.sum.per_second"])
    bandwidth = number(metrics["dram__bytes.sum.peak_sustained"]) * number(metrics["dram__cycles_elapsed.avg.per_second"])
    if min(hz, traffic, bandwidth) <= 0:
        raise ValueError("Invalid Tensor Core roofline frequency/traffic/peak")
    rows = []
    for name in names:
        base = name[:-len(suffix)]
        work = number(metrics[name]) * hz
        peak = number(metrics[base + ".sum.peak_sustained"]) * hz
        if work > 0 and peak <= 0:
            raise ValueError("Positive tensor work without its matching peak")
        rows.append({"arithmetic_path": base.removeprefix("sm__ops_path_tensor_"),
                     "work_ops_per_s": work, "peak_work_ops_per_s": peak,
                     "traffic_bytes_per_s": traffic, "peak_traffic_bytes_per_s": bandwidth,
                     "arithmetic_intensity_ops_per_byte": work / traffic,
                     "work_pct_of_peak": 100 * work / peak if peak else 0,
                     "traffic_pct_of_peak": 100 * traffic / bandwidth, "work_metric": name})
    return rows


def combined_cases(cases):
    """Sum the per-configuration groups so one phase/GPU total covers the whole projection.

    Decode samples a matmul and its stream-k fixup separately; a report figure wants
    their combined cost, while the exported cases keep each group distinct.
    """
    structural = ("gpu", "n", "transformer_layers_on_gpu", "projected_gate_up_launches")
    merged = {}
    for case in cases:
        key = (case["format"], case["phase"], case["gpu"])
        if key not in merged:
            merged[key] = dict(case) | {"kernel_configuration": [case["kernel_configuration"]],
                                        "roles": collections.Counter(case["roles"])}
            continue
        total = merged[key]
        for field, value in case.items():
            if field in ("opcodes", "projected_opcodes"):
                total[field] = {name: total[field][name] + value[name] for name in value}
            elif field == "roles":
                total[field] += collections.Counter(value)
            elif field == "kernel_configuration":
                total[field] = total[field] + [value]
            elif isinstance(value, (int, float)) and field not in structural:
                total[field] += value
    return {key: value | {"roles": dict(value["roles"])} for key, value in merged.items()}


def kernel_family(name):
    """Kernel identity without template arguments, which encode the quantization type."""
    return name.split("<", 1)[0].strip()


def bottleneck_data(root, captures):
    root, captures = root.resolve(), captures.resolve()
    sys.path.insert(0, str(ROOT / "scripts"))
    import collect_a100_roofline as operators
    import profile_a100_setting as profiler
    from profile_rtxpro6000 import comparison_sources, full_jobs
    plan_path = captures / "paper-profile-plan.json"
    plan = read(plan_path)
    if (plan.get("hardware") != "rtxpro6000" or Path(plan["study_root"]).resolve() != root
            or plan.get("concurrency") != 8 or plan.get("output_tokens") != 512):
        raise ValueError("Profile plan is not the selected RTX PRO 6000 C8 study")
    analysis_scope = plan.get("analysis_scope", "complete_operator_inventory")
    minimal = analysis_scope == "sampled_gate_up_extrapolation"
    if analysis_scope not in ("complete_operator_inventory", "sampled_gate_up_extrapolation"):
        raise ValueError("Unknown profile analysis scope")
    if minimal and (plan.get("full_capture_count") != 8 or plan.get("operator_capture_count") != 0
                    or plan.get("operator_command") is not None):
        raise ValueError("Minimal image profile plan must contain exactly eight full captures and no operator captures")
    formats = plan.get("formats", [Q4, Q2])
    if len(formats) != 2 or len(set(formats)) != 2 or set(formats) - set(serving.FORMATS["70b"]):
        raise ValueError("Profile plan must compare exactly two supported formats")
    # Derive the scientific selection again rather than trusting saved job labels.
    jobs = full_jobs(root, captures, plan["prompt_tokens"], Path("unused"), "unused", formats)
    if len(plan.get("full_jobs", [])) != len(jobs):
        raise ValueError("The full-section profile plan is incomplete")
    expected_sources = comparison_sources([root / "70b" / q / "c8" / f"p{plan['prompt_tokens']}" for q in formats])
    if plan.get("source_cells") != expected_sources:
        raise ValueError("Serving source changed after profile planning")
    sources = {str(plan_path): sha(plan_path)}
    devices, cases, metric_rows, instruction_rows, roofs = None, [], [], [], []
    layer_counts = None
    for source in plan["source_cells"]:
        path = Path(source["path"])
        if sha(path) != source["sha256"]:
            raise ValueError("Serving source changed after profile planning")
        cell, ids = profiler.load_cell(path, "rtxpro6000")
        gpu_rows = profiler.validate_rtx_cell(cell, ids)
        if devices is not None and devices != gpu_rows:
            raise ValueError("Hardware differs across comparison formats")
        devices = gpu_rows
        split = [float(value) for value in cell["configuration"]["SERVER_TENSOR_SPLIT"].split(",")]
        assignments = layer_devices(cell["model"]["layers"], split)
        counts = [assignments[:-1].count(gpu) for gpu in range(len(devices))]
        if sum(counts) != cell["model"]["layers"]:
            raise ValueError("Transformer layer placement does not cover the model")
        if layer_counts is not None and layer_counts != counts:
            raise ValueError("Layer placement differs across comparison formats")
        layer_counts = counts
        sources[str(path)] = sha(path)
        raw_path = path.parent / "r1/raw.json"
        sources[str(raw_path)] = sha(raw_path)
    if devices is None or len(plan["source_cells"]) != 2:
        raise ValueError("Expected two serving source cells")
    sampled_operator_points, extrapolation_rows, unplottable_groups = [], [], []
    for job in jobs:
        directory = captures / job["capture"]
        marker, metadata, summary = (read(directory / name) for name in ("study-capture.json", "metadata.json", "counter_summary.json"))
        source = (Path(job["cell"]) / "cell.json").resolve()
        if (marker.get("exit_code") != 0 or Path(marker.get("source_cell", "")).resolve() != source
                or metadata.get("status") != "captured" or metadata.get("counter_status") != "collected"
                or metadata.get("server_code_commit") != context.PIN
                or metadata.get("warmup_requests") != 8 or metadata.get("profiled_requests") != 8):
            raise ValueError(f"Capture/source/protocol validation failed: {directory}")
        identities = list(csv.reader((metadata.get("gpu_identity_csv") or "").splitlines(), skipinitialspace=True))
        capture_gpus = {row[0]: row[1:3] for row in identities if len(row) >= 3}
        if any(capture_gpus.get(str(d["index"])) != [d["uuid"], d["name"]] for d in devices):
            raise ValueError(f"Capture GPU identities differ from the serving measurement: {directory}")
        nvtx = f"regex:csb_batch:phase={job['phase']}:.*/*/{job['operation_regex']}"
        settings = marker.get("settings", {})
        if settings.get("PROFILE_NVTX_INCLUDE") != nvtx or settings.get("PROFILE_DEVICES") != str(job["gpu"]):
            raise ValueError("Saved profile filters differ from the intended comparison")
        build = metadata.get("server_build_provenance", {})
        if (build.get("status") != "verified_adjacent_build" or
                "-DCMAKE_CUDA_ARCHITECTURES=120" not in build.get("build", {}).get("configure_command", [])):
            raise ValueError("Capture does not establish an SM120 diagnostic build")
        launches = summary["launches"]
        if len(launches) != 5:
            raise ValueError(f"Expected five matched gate/up launches: {directory}")
        # Decode splits each projection into a matmul plus a stream-k fixup, so
        # every launch configuration becomes its own sample group.
        groups = collections.defaultdict(lambda: {"roles": collections.Counter(), "values": [],
                                                  "opcodes": [], "scalars": [], "launches": [], "no_traffic": 0})
        for launch in launches:
            op = launch.get("operation", {})
            if (str(launch["device"]) != str(job["gpu"]) or
                    any(str(op.get(k)) != str(v) for k, v in {"phase": job["phase"], "m": 28672,
                        "n": job["n"], "k": 8192, "type": job["tensor_type"], "operation_match": "nvtx_same_capture"}.items()) or
                    op.get("role") not in ("blk.*.ffn_gate.weight", "blk.*.ffn_up.weight")):
                raise ValueError(f"Unmatched phase/geometry/role/GPU in {directory}")
            group = groups[tuple(str(launch.get(k, "")) for k in ("kernel", "grid", "block"))]
            group["roles"][op["role"]] += 1
            row = {key: number(launch["metrics"][name]) for key, name in METRICS.items()}
            row["duration_s"] = duration_s(launch)
            group["values"].append(row); group["opcodes"].append(opcode_counts(launch))
            try:
                group["scalars"].append(operators.measured(launch))
            except ValueError:
                # A stream-k fixup can finish entirely in cache. Without DRAM traffic its
                # arithmetic intensity is undefined, so it is counted but never plotted.
                group["no_traffic"] += 1
            group["launches"].append(launch)
            try:
                tensor_points = list(tensor_coordinates(launch))
            except ValueError:
                # The same cache-resident fixup: no DRAM traffic, so no roofline coordinate.
                tensor_points = []
            for point in tensor_points:
                roofs.append({"format": job["format"], "phase": job["phase"], "gpu": job["gpu"],
                              "launch_id": launch["id"], **point, "source": str(directory / "counter_summary.json")})
            for name, item in launch["metrics"].items():
                metric_rows.append({"format": job["format"], "phase": job["phase"], "gpu": job["gpu"],
                    "launch_id": launch["id"], "metric": name, "value": item.get("value"),
                    "unit": item.get("unit"), "available": item.get("available"), "source": str(directory)})
        for configuration, group in groups.items():
            values, opcodes = group["values"], group["opcodes"]
            case = {"format": job["format"], "phase": job["phase"], "gpu": job["gpu"], "n": job["n"],
                    "samples": len(values), "roles": dict(group["roles"]), "kernel_configuration": list(configuration),
                    **{key: statistics.median(row[key] for row in values) for key in values[0]}}
            # Median of the per-launch categories preserves the measured grouping.
            categories = [{"FFMA": counts.get("FFMA", 0), "I2FP": counts.get("I2FP", 0),
                           "Other": sum(v for k, v in counts.items() if k not in ("FFMA", "I2FP"))} for counts in opcodes]
            case["opcodes"] = {key: statistics.median(row[key] for row in categories) for key in categories[0]}
            # A Llama transformer layer invokes one gate and one up projection. The
            # capture samples five matching launches; scale its per-launch median to
            # one matched matrix batch across the transformer layers placed here.
            projection_factor = 2 * layer_counts[job["gpu"]]
            case.update(transformer_layers_on_gpu=layer_counts[job["gpu"]],
                        projected_gate_up_launches=projection_factor,
                        projected_instructions=case["instructions"] * projection_factor,
                        projected_dram_reads=case["dram_reads"] * projection_factor,
                        projected_dram_writes=case["dram_writes"] * projection_factor,
                        projected_opcodes={key: value * projection_factor for key, value in case["opcodes"].items()})
            extrapolation_rows.append({"format": job["format"], "phase": job["phase"], "gpu": job["gpu"],
                "sampled_launches": len(values), "transformer_layers_on_gpu": layer_counts[job["gpu"]],
                "gate_up_launches_for_one_matrix_batch": projection_factor,
                "median_instructions_per_launch": case["instructions"],
                "estimated_gate_up_instructions": case["projected_instructions"],
                "median_dram_read_bytes_per_launch": case["dram_reads"],
                "estimated_gate_up_dram_read_bytes": case["projected_dram_reads"],
                "median_dram_write_bytes_per_launch": case["dram_writes"],
                "estimated_gate_up_dram_write_bytes": case["projected_dram_writes"],
                "scope": "one matrix batch across gate/up operations on this GPU"})
            case["roofline_samples"] = len(group["scalars"])
            case["launches_without_dram_traffic"] = group["no_traffic"]
            if group["scalars"]:
                sampled_operator_points.append({"format": job["format"], "phase": job["phase"], "gpu": job["gpu"],
                    "physical_gpu": devices[job["gpu"]]["index"], "operator": "sampled_gate_up",
                    "role": "blk.*.ffn_gate/up.weight", "type": job["tensor_type"], "m": "28672", "n": str(job["n"]),
                    "k": "8192", "batch_width": "", "fusion": "", "kernel": configuration[0],
                    "grid": configuration[1], "block": configuration[2],
                    **{key: statistics.median(row[key] for row in group["scalars"]) for key in operators.COORDINATES},
                    "samples": len(group["scalars"]), "source": str(directory / "counter_summary.json")})
            else:
                unplottable_groups.append(case["kernel_configuration"])
            for launch, counts in zip(group["launches"], opcodes):
                instruction_rows.extend({"format": job["format"], "phase": job["phase"], "gpu": job["gpu"],
                                         "launch_id": launch["id"], "opcode": key, "instructions": value} for key, value in counts.items())
            cases.append(case)
        for filename in ("study-capture.json", "metadata.json", "counter_summary.json"):
            path = directory / filename
            sources[str(path)] = sha(path)
    for phase in ("prefill", "decode"):
        for gpu in (0, 1):
            selected = [c for c in cases if (c["phase"], c["gpu"]) == (phase, gpu)]
            by_format = collections.defaultdict(dict)
            for entry in selected:
                by_format[entry["format"]][kernel_family(entry["kernel_configuration"][0])] = entry["roles"]
            grouped = list(by_format.values())
            if len(grouped) != 2 or grouped[0].keys() != grouped[1].keys():
                raise ValueError("Compared formats sampled different gate/up kernel configurations")
            if any(grouped[0][family] != grouped[1][family] for family in grouped[0]):
                raise ValueError("Compared formats have different gate/up launch proportions")
    if minimal:
        roof = rated_roof(devices[0]["name"])
        if any(rated_roof(device["name"]) != roof for device in devices):
            raise ValueError("Both GPUs must have the same rated ceilings")
        points = sampled_operator_points
        operator_plan = {"schema_version": 1, "hardware_family": "rtxpro6000", **roof,
            "prompt_tokens": plan["prompt_tokens"], "concurrency": 8, "devices": [str(d["index"]) for d in devices],
            "phases": ["prefill", "decode"], "operators": ["sampled_gate_up"],
            "launch_cap_per_operator_phase_gpu": 5, "jobs": plan["full_jobs"],
            "method": "Scalar FP32 coordinates reused from the eight full-section gate/up captures"}
        operator_audit = {"status": "complete", "captures": 8, "captured_launches": 40,
            "configuration_groups": len(points),
            "groups_without_dram_traffic_not_plotted": len(unplottable_groups),
            "zero_fp32_groups_not_plotted": sum(point["fp32_flops"] == 0 for point in points),
            "scope": "sampled gate/up only; no complete operator inventory"}
    else:
        operator_root = captures / "operators"
        operator_plan = read(operator_root / "roofline-plan.json")
        if (operator_plan.get("hardware_family") != "rtxpro6000" or operator_plan.get("prompt_tokens") != plan["prompt_tokens"]
                or operator_plan.get("concurrency") != 8 or set(operator_plan.get("operators", [])) != set(operators.OPERATORS)
                or operator_plan.get("phases") != ["prefill", "decode"] or len(operator_plan.get("jobs", [])) != 60
                or operator_plan.get("launch_cap_per_operator_phase_gpu") != plan.get("operator_launch_cap_per_capture", 5)
                or {Path(c["path"]).resolve() for c in operator_plan["source_cells"]} != {Path(c["path"]).resolve() for c in plan["source_cells"]}):
            raise ValueError("Operator collection does not cover the same complete workload")
        points, operator_audit = operators.export_values(operator_root, operator_plan)
        if operator_audit["status"] != "complete":
            raise ValueError("Operator captures are incomplete; see operators/roofline-validation.json")
        for point in points:
            path = operator_root / point["source"]
            sources[str(path)] = sha(path)
        sources[str(operator_root / "roofline-plan.json")] = sha(operator_root / "roofline-plan.json")
    for name in ("peak_fp32_flops_per_s", "peak_dram_bytes_per_s"):
        if any(rated_roof(d["name"])[name] != operator_plan[name] for d in devices):
            raise ValueError("Operator roofline uses incorrect GPU ceilings")
    return {"prompt_tokens": plan["prompt_tokens"], "devices": devices, "cases": cases, "metric_rows": metric_rows,
            "instruction_rows": instruction_rows, "tensor_samples": roofs, "operator_points": points,
            "operator_plan": operator_plan, "operator_audit": operator_audit, "sources": sources,
            "formats": formats,
            "analysis_scope": analysis_scope, "layer_counts": layer_counts, "extrapolation_rows": extrapolation_rows}


def bottleneck_figures(output, data):
    plt = plotting()
    import numpy as np
    from matplotlib.lines import Line2D
    cases = combined_cases(data["cases"])
    reference, comparison = data["formats"]
    estimated = data["analysis_scope"] == "sampled_gate_up_extrapolation"
    for phase in ("prefill", "decode"):
        specs = ([("ipc", 1, "IPC (active cycles)"), ("alu_pct", 1, "ALU (% peak, elapsed)"),
                  ("tensor_pct", 1, "Tensor (% peak, elapsed)")] if phase == "prefill" else
                 [("registers", 1, "Registers per thread"), ("register_blocks", 1, "Register-limited blocks/SM"),
                  ("occupancy_pct", 1, "Achieved occupancy (%)")])
        specs += [("projected_instructions" if estimated else "instructions", 1e6,
                   "Estimated GPU0 gate/up instructions (M)" if estimated else "Executed instructions (M)"),
                  ("projected_dram_reads" if estimated else "dram_reads", 1e6,
                   "Estimated GPU0 gate/up DRAM reads (MB)" if estimated else "DRAM reads (MB)"),
                  ("duration_s", 1e-3 if phase == "prefill" else 1e-6, "Duration (ms)" if phase == "prefill" else "Duration (us)")]
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
        for ax, (key, divisor, label) in zip(axes.flat, specs):
            values = [cases[q, phase, 0][key] / divisor for q in (reference, comparison)]
            bars = ax.bar([0, 1], values, color=[COLORS[reference], COLORS[comparison]], width=.6)
            ax.bar_label(bars, labels=[actual_value(v) for v in values], padding=3, fontsize=9)
            ax.set(xticks=[0, 1], xticklabels=[reference, comparison], ylabel=label,
                   title=percentage(change(*values)) + f" ({comparison} vs {reference})", ylim=(0, (max(values) or 1) * 1.25))
            ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
        suffix = f"; estimated totals span {cases[reference, phase, 0]['transformer_layers_on_gpu']} layers" if estimated else ""
        fig.suptitle(f"{phase.title()} gate/up - GPU0 - M=28672, N={512 if phase == 'prefill' else 8}, K=8192{suffix}")
        fig.tight_layout(rect=(0, 0, 1, .93))
        name = "prefill-actual-values" if phase == "prefill" else "decode-register-pressure"
        save_figure(fig, output / "figures" / name); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, phase in zip(axes, ("prefill", "decode")):
        x = np.arange(3)
        for i, q in enumerate((reference, comparison)):
            inventory = cases[q, phase, 0]["projected_opcodes" if estimated else "opcodes"]
            values = [inventory[key] / 1e6 for key in ("FFMA", "I2FP", "Other")]
            bars = ax.bar(x + (i - .5) * .36, values, .36, color=COLORS[q], label=q)
            ax.bar_label(bars, labels=[actual_value(v) for v in values], fontsize=8, padding=3)
        ax.set(title=phase.title(), xticks=x, xticklabels=["FFMA", "I2FP", "Other"], ylabel="Executed instructions (M)")
        ax.margins(y=.25); ax.legend(); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("Estimated gate/up instruction mix across GPU0 layers" if estimated else
                 "Instruction mix - GPU0 - complete reconciled opcode inventories")
    fig.tight_layout(rect=(0, 0, 1, .92))
    save_figure(fig, output / "figures/instruction-overhead-actual-values"); plt.close(fig)

    # Each row is an actual arithmetic path, each column a phase. Zero work is
    # retained in CSV but has no logarithmic plot point.
    groups = collections.defaultdict(list)
    for row in data["tensor_samples"]:
        if row["gpu"] == 0:
            groups[row["arithmetic_path"], row["phase"], row["format"]].append(row)
    paths = sorted({key[0] for key, rows in groups.items() if any(r["work_ops_per_s"] > 0 for r in rows)})
    fig, axes = plt.subplots(max(1, len(paths)), 2, figsize=(10.5, max(4.5, 3.6 * len(paths))), squeeze=False)
    for row_index, path in enumerate(paths or [None]):
        for col, phase in enumerate(("prefill", "decode")):
            ax = axes[row_index, col]
            positive = False
            for q in (reference, comparison):
                samples = groups.get((path, phase, q), [])
                if not samples:
                    continue
                point = {k: statistics.median(s[k] for s in samples) for k in
                         ("work_ops_per_s", "peak_work_ops_per_s", "peak_traffic_bytes_per_s", "arithmetic_intensity_ops_per_byte")}
                if point["work_ops_per_s"] <= 0:
                    continue
                positive = True
                peak, bandwidth = point["peak_work_ops_per_s"], point["peak_traffic_bytes_per_s"]
                intensity = point["arithmetic_intensity_ops_per_byte"]
                ridge = peak / bandwidth
                xs = np.geomspace(min(intensity, ridge) / 10, max(intensity, ridge) * 10, 150)
                ax.loglog(xs, np.minimum(peak, xs * bandwidth) / 1e12, "--", color=COLORS[q], alpha=.7)
                ax.scatter(intensity, point["work_ops_per_s"] / 1e12, color=COLORS[q], marker="o" if q == reference else "D", label=q)
            if positive:
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "No positive tensor work in saved counters", ha="center", transform=ax.transAxes, fontsize=8)
            ax.set(title=f"{phase}: {path or 'no active path'}", xlabel="Tensor operations / DRAM byte", ylabel="Executed tensor TOPS")
            ax.title.set_fontsize(7); ax.grid(alpha=.2)
    fig.suptitle("Tensor Core roofline - GPU0 - actual arithmetic paths and same-capture peaks")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, output / "figures/prefill-tensor-roofline"); plt.close(fig)

    operator_plan = data["operator_plan"]
    peak, bandwidth = operator_plan["peak_fp32_flops_per_s"], operator_plan["peak_dram_bytes_per_s"]
    operators = operator_plan["operators"]
    styles = {name: (f"C{i}", "o^sDvh*P"[i]) for i, name in enumerate(operators)}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    for i, phase in enumerate(("prefill", "decode")):
        for gpu in (0, 1):
            ax = axes[i, gpu]
            points = [p for p in data["operator_points"] if p["phase"] == phase and p["gpu"] == gpu and p["fp32_flops"] > 0]
            xs = [p["fp32_flops_per_byte"] for p in points] + [peak / bandwidth]
            domain = np.geomspace(min(xs) / 5, max(xs) * 5, 150)
            ax.loglog(domain, np.minimum(peak, domain * bandwidth) / 1e12, "--", color="#697586")
            for point in points:
                color, marker = styles[point["operator"]]
                ax.scatter(point["fp32_flops_per_byte"], point["fp32_flops_per_s"] / 1e12,
                           edgecolors=color, facecolors=color if point["format"] == reference else "none", marker=marker, s=45)
            ax.set(title=f"{phase.title()} - GPU{gpu}", xlabel="Scalar FP32 FLOPs / DRAM byte", ylabel="Scalar FP32 TFLOP/s")
            ax.grid(alpha=.2)
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", label=name) for name, (color, marker) in styles.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(.5, .015))
    fig.suptitle(("Sampled gate/up scalar FP32 rooflines" if estimated else "Operator scalar FP32 rooflines") +
                 f" - {reference} filled, {comparison} hollow")
    fig.tight_layout(rect=(0, .09, 1, .95))
    save_figure(fig, output / "figures/operator-roofline-combined"); plt.close(fig)


def report_document(output, data):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, Spacer, PageBreak
    from xml.sax.saxutils import escape
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    cases = combined_cases(data["cases"])
    reference, comparison = data["formats"]
    estimated = data["analysis_scope"] == "sampled_gate_up_extrapolation"
    elements, markdown = [], []

    def para(text, title=False):
        elements.append(Paragraph(escape(text), styles["Heading1" if title else "BodyText"]))
        elements.append(Spacer(1, 7))
        markdown.extend([("# " if title else "") + text, ""])

    def figure(name, height=290):
        path = output / "figures" / (name + ".png")
        image = Image(str(path))
        scale = min(510 / image.imageWidth, height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        elements.append(image); elements.append(Spacer(1, 8))
        markdown.extend([f"![{name}](figures/{name}.png)", ""])

    def table(rows):
        tab = Table(rows, repeatRows=1, hAlign="LEFT")
        tab.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf4")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("GRID", (0, 0), (-1, -1), .3, colors.lightgrey)]))
        elements.append(tab); elements.append(Spacer(1, 10))
        markdown.append("| " + " | ".join(map(str, rows[0])) + " |")
        markdown.append("| " + " | ".join("---" for _ in rows[0]) + " |")
        markdown.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows[1:]]); markdown.append("")

    names = " / ".join(dict.fromkeys(d["name"] for d in data["devices"]))
    para(f"Runtime bottlenecks of {reference} and {comparison} quantization", True)
    para(f"Llama 3.3 70B; two GPUs: {names}; C8; {data['prompt_tokens']:,} input and 512 output tokens; FP16 KV.")
    if estimated:
        para("Reduced image scope: eight full-section captures sample five gate/up launches per format, phase and GPU. "
             "Values labelled estimated scale each per-launch median across the gate and up operations in the transformer layers assigned to that GPU. "
             "The scalar roofline reuses these captures; no separate operator inventory was collected.")
    for phase in ("prefill", "decode"):
        if phase == "decode":
            elements.append(PageBreak()); para("Decode gate/up measurements", True)
        a, b = cases[reference, phase, 0], cases[comparison, phase, 0]
        instruction_key = "projected_instructions" if estimated else "instructions"
        dram_key = "projected_dram_reads" if estimated else "dram_reads"
        para(f"GPU0 {phase}: {comparison} kernel duration changes {percentage(change(a['duration_s'], b['duration_s']))}, "
             f"executed instructions {percentage(change(a[instruction_key], b[instruction_key]))}, and DRAM reads "
             f"{percentage(change(a[dram_key], b[dram_key]))} relative to {reference}. " +
             ("Duration is observed per launch; labelled instruction and DRAM totals are layer-placement estimates."
              if estimated else "These are observed kernel comparisons."))
        figure("prefill-actual-values" if phase == "prefill" else "decode-register-pressure", 310)
        para("Five sampled gate/up launches per format/phase/GPU; medians within a matched matrix geometry and a single kernel configuration. "
             "Pipeline percentages describe different resources and must not be added. Register pressure and occupancy do not by themselves isolate a latency cause.")
        if set(data["formats"]) == {Q2, Q4}:
            para("The pinned Blackwell dispatch selects MMQ for Q2_K/Q4_K above five activation columns. "
                 "This eight-column decode workload therefore does not inherit the A6000 MMVQ explanation. Actual kernel names are retained in report-data.json.")
        else:
            para("Actual kernel names and matrix geometry are retained in report-data.json; conclusions from another quantization pair are not transferred to this shard.")
    elements.append(PageBreak()); para("Executed instruction mix", True)
    figure("instruction-overhead-actual-values", 280)
    rows = [["Phase / opcode", f"{reference} (M)", f"{comparison} (M)", "Change"]]
    for phase in ("prefill", "decode"):
        a, b = cases[reference, phase, 0], cases[comparison, phase, 0]
        for name in ("FFMA", "I2FP", "Other"):
            a_inventory = a["projected_opcodes" if estimated else "opcodes"]
            b_inventory = b["projected_opcodes" if estimated else "opcodes"]
            rows.append([f"{phase} / {name}", actual_value(a_inventory[name] / 1e6), actual_value(b_inventory[name] / 1e6),
                         percentage(change(a_inventory[name], b_inventory[name]))])
    table(rows)
    para("FFMA and I2FP counts are read from complete SASS opcode inventories and checked against total instructions per launch. "
         "An absent opcode is zero only after that reconciliation. Other includes every remaining opcode. " +
         ("The displayed totals multiply per-launch medians by two gate/up operations per placed transformer layer; they estimate one matched matrix batch. "
          if estimated else "") + "Instruction-count differences do not establish separate runtime costs.")
    elements.append(PageBreak()); para("Tensor Core arithmetic paths", True)
    figure("prefill-tensor-roofline", 440)
    para("Work and matching peak counters are selected by their actual exported arithmetic path, including Blackwell op_imma/op_hmma paths. "
         "Each coordinate is calculated per launch before aggregation. Dashed ceilings use same-capture sustained peaks and observed clocks; "
         "Quantized weight storage bits are not a compute precision. Zero-work paths remain in tensor-roofline-samples.csv and have no logarithmic point.")
    para("The plot covers tensor work in the selected kernels, not whole-serving performance or a unique bottleneck classification.")
    elements.append(PageBreak()); para("Sampled gate/up scalar FP32 rooflines" if estimated else "Scalar FP32 operator rooflines", True)
    figure("operator-roofline-combined", 450)
    roof = data["operator_plan"]
    para(f"Per-GPU rated ceilings: {roof['peak_fp32_flops_per_s'] / 1e12:g} TFLOP/s and "
         f"{roof['peak_dram_bytes_per_s'] / 1e9:g} GB/s. F = FADD + FMUL + 2*FFMA from predicated-on thread counts; "
         "B = DRAM reads + writes; x = F/B; y = F/duration. FP16, tensor, integer and special-function work are excluded.")
    if estimated:
        para(f"The same eight full-section captures provide {len(data['operator_points'])} gate/up kernel/configuration groups; "
             f"five matching launches contribute to each group. {data['operator_audit']['zero_fp32_groups_not_plotted']} groups have zero scalar FP32 work "
             "and are retained in CSV without invented plot positions. This page describes sampled gate/up kernels only. "
             "Normalization, attention, activations, KV updates, the output head and other helpers were not captured.")
    else:
        para(f"All 60 requested operator captures passed validation with up to "
             f"{roof['launch_cap_per_operator_phase_gpu']} matching launch(es) per capture; "
             f"{len(data['operator_points'])} kernel/configuration groups. "
             f"{data['operator_audit']['zero_fp32_groups_not_plotted']} groups have zero scalar FP32 work and are retained in CSV without invented plot positions. "
             "FlashAttention includes fused softmax; normalization, activations, KV updates and other unselected helpers are outside this inventory.")
    elements.append(PageBreak()); para("Both GPUs and reproducibility", True)
    rows = [["Phase", "GPU", f"{reference} time (us)", f"{comparison} time (us)", "Change", "Samples/format"]]
    for phase in ("prefill", "decode"):
        for gpu in (0, 1):
            a, b = cases[reference, phase, gpu], cases[comparison, phase, gpu]
            rows.append([phase, str(gpu), f"{a['duration_s'] * 1e6:.3f}", f"{b['duration_s'] * 1e6:.3f}",
                         percentage(change(a["duration_s"], b["duration_s"])), "5"])
    table(rows)
    para("Each capture uses eight discarded warmup requests and eight profiled requests. Both server GPUs remain visible; only one GPU's counters are selected per capture. "
         "NVTX labels establish actual phase, tensor role and matrix dimensions. CUDA graphs are disabled for attribution. Cache and clock controls are none. "
         "Profiler replay timings are separate from the unprofiled serving measurements.")
    para("Each setting has one capture, not independent repetitions. GPU agreement does not prove universality across workloads. "
         "The resulting report describes the saved counters; it does not copy the previous GPU's performance conclusions.")
    para(f"Pinned llama.cpp: {context.PIN}. Source hashes: evidence.json. Complete launch metrics: hardware-metrics.csv. "
         "Opcode counts: instruction-instances.csv. Tensor coordinates: tensor-roofline-samples.csv. Scalar coordinates: operator-roofline-data.csv. "
         "Layer projection inputs: gate-up-layer-extrapolation.csv. Rated scalar roofline source: " + roof["roof_source"])
    doc = SimpleDocTemplate(str(output / "runtime-bottlenecks-report.pdf"), pagesize=(612, 792),
                            leftMargin=42, rightMargin=42, topMargin=35, bottomMargin=35)
    def page_number(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 20, "RTX PRO 6000 - 70B - kernel profiling")
        canvas.drawRightString(570, 20, str(doc.page))
    doc.build(elements, onFirstPage=page_number, onLaterPages=page_number)
    (output / "runtime-bottlenecks-report.md").write_text("\n".join(markdown))


def build_bottlenecks(root, captures, output=None):
    root, captures = root.resolve(), captures.resolve()
    data = bottleneck_data(root, captures)
    output = destination(output or root / "paper" / f"runtime-bottlenecks-c8-p{data['prompt_tokens']}",
                         root, f"runtime-bottlenecks-c8-p{data['prompt_tokens']}")
    bottleneck_figures(output, data)
    report_document(output, data)
    csv_write(output / "hardware-metrics.csv", data["metric_rows"])
    csv_write(output / "instruction-instances.csv", data["instruction_rows"])
    csv_write(output / "tensor-roofline-samples.csv", data["tensor_samples"])
    csv_write(output / "operator-roofline-data.csv", data["operator_points"])
    csv_write(output / "gate-up-layer-extrapolation.csv", data["extrapolation_rows"])
    write(output / "report-data.json", data)
    write(output / "evidence.json", {"study_root": str(root), "capture_root": str(captures),
                                     "sources": data["sources"], "builder_sha256": sha(Path(__file__))})
    return output / "runtime-bottlenecks-report.pdf"
