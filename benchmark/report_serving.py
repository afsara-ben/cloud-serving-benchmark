#!/usr/bin/env python3
"""Report the selected A100 serving scope from saved evidence, without GPU work."""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import report_context_study as context

MODELS = ("1b", "8b", "70b")
FORMATS = {model: ("IQ1_M", "Q2_K", "Q4_K_M", "Q8_0") + (() if model == "70b" else ("FP16",))
           for model in MODELS}
CONCURRENCIES = (8, 16, 32, 64)
PROMPTS = (2048, 4096, 8192, 16384)


def scope_document(devices):
    return {"schema_version": 1, "model_formats": {model: list(formats) for model, formats in FORMATS.items()},
            "device_indices": list(devices), "concurrencies": list(CONCURRENCIES),
            "prompt_lengths": list(PROMPTS), "capacity_lengths": [], "output_tokens": 512,
            "repetitions": 1, "warmup_requests_per_cell": "concurrency",
            "measured_requests_per_cell": "2 * concurrency", "profiling": False}


def columns(devices):
    return [("model", "Model"), ("format", "Format"), ("concurrency", "Clients"), ("input_tokens", "Input tokens"),
            ("ttft_p50_ms", "TTFT p50 ms"), ("ttft_p95_ms", "TTFT p95 ms"),
            ("tpot_p50_ms", "TPOT p50 ms"), ("tpot_p95_ms", "TPOT p95 ms"),
            ("output_tokens_per_second_mean", "Generated tok/s"),
            *[(f"gpu{device}_peak_vram_gib", f"GPU {device} peak GiB") for device in devices]]


def plot_metrics(rows, output, devices):
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    artifacts = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        for name, metric, label in (
                ("throughput", "output_tokens_per_second_mean", "Generated tokens / second"),
                ("ttft", "ttft_p95_ms", "TTFT p95 (ms)"),
                ("tpot", "tpot_p95_ms", "TPOT p95 (ms)"),
                ("memory", None, "Peak sampled GPU memory (GiB)")):
            figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained", squeeze=False)
            for axis, concurrency in zip(axes.flat, CONCURRENCIES):
                for index, quant in enumerate(FORMATS[model]):
                    lookup = {row["input_tokens"]: row for row in model_rows
                              if row["concurrency"] == concurrency and row["format"] == quant}
                    metrics = [(f"gpu{gpu}_peak_vram_gib", f"{quant} GPU {gpu}") for gpu in devices] if metric is None else [(metric, quant)]
                    for device_order, (key, legend) in enumerate(metrics):
                        values = [lookup.get(prompt, {}).get(key) for prompt in PROMPTS]
                        if not any(context.finite(value) for value in values):
                            continue
                        axis.plot(PROMPTS, [value if context.finite(value) else math.nan for value in values],
                                  marker="o", linestyle="--" if device_order else "-", color=f"C{index}", label=legend)
                axis.set_xscale("log", base=2)
                axis.set_xticks(PROMPTS, labels=("2k", "4k", "8k", "16k"))
                axis.set_xlabel("Input tokens (k = 1024)")
                axis.set_ylabel(label)
                axis.set_title(f"{concurrency} clients")
                axis.grid(alpha=.25)
                if axis.lines:
                    axis.set_ylim(bottom=0)
                    axis.legend(fontsize=7)
                else:
                    axis.text(.5, .5, "No validated measurements", ha="center", transform=axis.transAxes)
            figure.suptitle(f"{model.upper()} | {len(devices)} GPU(s) | FP16 KV | 512 output tokens | one measured run")
            for suffix in ("png", "svg"):
                path = output / f"{model}-{name}.{suffix}"
                figure.savefig(path, dpi=180)
                artifacts.append(str(path.relative_to(output.parent)))
            plt.close(figure)
    return artifacts


def build_report(study_root, plots=True):
    root = study_root.resolve()
    scope = context.read_json(root / "serving-scope.json")
    devices = scope.get("device_indices", [])
    if (len(devices) not in (1, 2) or len(set(devices)) != len(devices)
            or any(not isinstance(device, str) or not device.isdecimal() for device in devices)
            or scope != scope_document(devices)):
        raise ValueError("Missing or incompatible serving-scope.json; use scripts/run_a100_serving.py")
    if not context.shared_layout(root):
        raise ValueError("Expected the launcher's shared results directory")
    roots = context.model_roots(root)
    output = root / "serving-report"
    if output.is_symlink() or any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Serving report output must not contain symbolic links")
    output.mkdir(exist_ok=True)
    runtime, capacity = [], []
    for model in MODELS:
        manifest = context.read_evidence_json(root, roots[model] / "manifest.json")
        progress = context.read_evidence_json(root, roots[model] / "progress.json")
        identity_errors = []
        if manifest:
            if [str(device.get("index")) for device in manifest.get("devices", [])] != devices:
                identity_errors.append("Manifest GPUs differ from the declared serving scope")
            if (manifest.get("llama_cpp_commit") != context.PIN or not manifest.get("binary_sha256")
                    or not manifest.get("code_sha256")):
                identity_errors.append("Pinned source, binary, or code provenance missing")
            if (manifest.get("concurrencies") != list(CONCURRENCIES) or manifest.get("prompt_lengths") != list(PROMPTS)
                    or manifest.get("capacity_lengths") != [] or manifest.get("repetitions") != 1):
                identity_errors.append("Manifest grid differs from the declared serving scope")
            if {item.get("quant") for item in manifest.get("models", [])} != set(FORMATS[model]):
                identity_errors.append("Manifest formats differ from the declared serving scope")
        for quant in FORMATS[model]:
            for concurrency in CONCURRENCIES:
                for prompt in PROMPTS:
                    row, runs = context.load_cell(root, model, quant, concurrency, prompt, manifest, progress,
                                                  device_indices=devices)
                    errors = list(identity_errors)
                    if runs:
                        entry = next((item for item in manifest.get("models", []) if item.get("quant") == quant), {})
                        if not entry.get("sha256"):
                            errors.append("Model hash missing")
                        measured = context.aggregate(row, runs, device_indices=devices)
                        if any(not context.finite(measured.get(f"gpu{gpu}_peak_vram_gib")) for gpu in devices):
                            errors.append("Measured HTTP window lacks GPU memory samples")
                        if not errors:
                            runtime.append(measured)
                    if errors:
                        row.update(status="invalid", validation_errors="; ".join(filter(None, [row["validation_errors"], *errors])))
                    capacity.append(row)
    fields = columns(devices)
    context.write_csv(output / "runtime.csv", runtime, None if runtime else [key for key, _ in fields])
    context.write_csv(output / "capacity.csv", capacity)
    excluded = sum(row["status"] in context.CAPACITY_STATUSES for row in capacity)
    unresolved = [row for row in capacity if row["status"] not in context.RESOLVED_STATUSES]
    artifacts = plot_metrics(runtime, output / "plots", devices) if plots else []
    audit = {"schema_version": 1, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "status": "incomplete" if unresolved else "complete", "scope": scope,
             "required_cells": len(capacity), "measured_cells": len(runtime), "capacity_exclusions": excluded,
             "unresolved_cells": unresolved, "plots": artifacts,
             "completion_definition": f"All {len(capacity)} serving cells measured or capacity-excluded; profiling is outside this scope."}
    context.write_json(output / "completion-audit.json", audit)
    lines = ["# A100 serving metrics", "", f"Status: **{audit['status']}**. {len(runtime)}/{len(capacity)} settings measured; "
             f"{excluded} capacity-excluded; {len(unresolved)} unresolved.", "",
             f"{len(devices)} GPU(s), physical indices {', '.join(devices)}. 1B/8B: five formats; 70B: IQ1_M/Q2_K/Q4_K_M/Q8_0.", "",
             "One measured run per setting: C discarded warmup requests and 2C measured requests, exactly 512 output tokens each. "
             "FP16 KV, prompt reuse off. Throughput is aggregate generated tokens divided by the measured HTTP window. "
             "TTFT/TPOT percentiles describe requests within that run; run-to-run variation is unmeasured. "
             "Memory is device-total SMI sampling within that same window, in GiB.", "",
             "[Runtime CSV](runtime.csv) · [All cell statuses](capacity.csv) · [Completion audit](completion-audit.json)", "",
             context.markdown_table(runtime, fields)]
    for artifact in artifacts:
        if artifact.endswith(".png"):
            lines += [f"![{Path(artifact).stem}]({artifact})", ""]
    (output / "index.md").write_text("\n".join(lines))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        audit = build_report(args.study_root, plots=not args.no_plots)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.study_root / 'serving-report/index.md'}: {audit['measured_cells']} measured settings")


if __name__ == "__main__":
    main()
