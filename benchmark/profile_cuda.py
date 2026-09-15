#!/usr/bin/env python3
"""Capture one warmed concurrent HTTP burst with Nsight or rocprofv3.

The remaining command-line arguments after -- are the exact llama-server
command. Profiler timings are diagnostic and never performance benchmark rows.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import collections
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from load_test import ServerTokenizer, finalize_record, make_messages, stream_completion


DEFAULT_METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.avg.per_cycle_active",
    "sm__inst_executed.avg.per_cycle_elapsed",
    "sm__inst_executed.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.max_rate",
])

# Keep replay groups small. Section identifiers are resolved by the installed
# Nsight version and preserve its architecture-specific metric definitions.
METRIC_GROUPS = {
    "memory": {"metrics": DEFAULT_METRICS, "sections": []},
    "instructions": {"metrics": "gpu__time_duration.sum,sm__inst_executed.sum,sm__inst_executed.avg.per_cycle_active,sm__inst_executed.avg.per_cycle_elapsed",
                     "sections": ["InstructionStats"]},
    "occupancy": {"metrics": "gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
                  "sections": ["LaunchStats", "Occupancy"]},
    "stalls": {"metrics": "gpu__time_duration.sum", "sections": ["SchedulerStats", "WarpStateStats"]},
    "tensor": {"metrics": "gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
               "sections": []},
}

SECTION_REQUIREMENTS = {
    "SpeedOfLight": ("gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                     "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    "ComputeWorkloadAnalysis": ("sm__inst_executed.avg.per_cycle_active", "sm__inst_executed.avg.per_cycle_elapsed",
                                "sm__inst_issued.avg.per_cycle_active",
                                "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
                                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
                                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    "MemoryWorkloadAnalysis": ("dram__bytes.sum.per_second", "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
                               "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
                               "gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed",
                               "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    "InstructionStats": ("smsp__inst_executed.sum", "sass__inst_executed_per_opcode"),
    "LaunchStats": ("launch__registers_per_thread", "launch__shared_mem_per_block"),
    "Occupancy": ("launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
                  "sm__warps_active.avg.pct_of_peak_sustained_active"),
    "SchedulerStats": ("smsp__warps_eligible.avg.per_cycle_active", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
    "WarpStateStats": tuple("smsp__average_warps_issue_stalled_" + cause + "_per_issue_active.ratio"
                            for cause in ("long_scoreboard", "short_scoreboard", "wait", "math_pipe_throttle")),
}

# These are opt-in A100 section collections; existing study metric groups keep
# their original scope. Requirements check essential raw evidence, not a claim
# that every optional chart metric is available on every architecture/version.
A100_SECTIONS = ("SpeedOfLight", "ComputeWorkloadAnalysis", "InstructionStats", "WarpStateStats",
                 "SchedulerStats", "Occupancy", "MemoryWorkloadAnalysis", "LaunchStats")
ROOFLINE_SECTIONS = {
    "overview": "SpeedOfLight_RooflineChart",
    "half": "SpeedOfLight_HierarchicalHalfRooflineChart",
    "single": "SpeedOfLight_HierarchicalSingleRooflineChart",
    "double": "SpeedOfLight_HierarchicalDoubleRooflineChart",
    "tensor": "SpeedOfLight_HierarchicalTensorRooflineChart",
}
ROOFLINE_MEMORY_METRICS = ("dram__bytes.sum.per_second", "dram__bytes.sum.peak_sustained",
                           "dram__cycles_elapsed.avg.per_second", "sm__cycles_elapsed.avg.per_second")
for precision, prefix in (("half", "h"), ("single", "f"), ("double", "d")):
    SECTION_REQUIREMENTS[ROOFLINE_SECTIONS[precision]] = ROOFLINE_MEMORY_METRICS + (
        f"sm__sass_thread_inst_executed_op_{prefix}fma_pred_on.sum.peak_sustained",
        *(f"smsp__sass_thread_inst_executed_op_{prefix}{operation}_pred_on.sum.per_cycle_elapsed"
          for operation in ("add", "mul", "fma")),
        "lts__lts2xbar_cycles_active.sum.peak_sustained", "lts__lts2xbar_cycles_active.sum.per_second",
        "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained", "l1tex__lsu_writeback_active_mem_lg.sum.per_second")
SECTION_REQUIREMENTS[ROOFLINE_SECTIONS["overview"]] = ROOFLINE_MEMORY_METRICS + tuple(
    metric for precision in ("single", "double") for metric in SECTION_REQUIREMENTS[ROOFLINE_SECTIONS[precision]]
    if metric.startswith(("sm__sass", "smsp__sass")))
SECTION_REQUIREMENTS[ROOFLINE_SECTIONS["tensor"]] = ROOFLINE_MEMORY_METRICS + (
    "lts__lts2xbar_cycles_active.sum.peak_sustained", "lts__lts2xbar_cycles_active.sum.per_second",
    "l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained", "l1tex__lsu_writeback_active_mem_lg.sum.per_second")


def a100_section_collection(rooflines=()):
    unknown = set(rooflines) - ROOFLINE_SECTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown roofline selections: {sorted(unknown)}")
    # Explicit load/store and L2 sectors supplement the section's memory summary.
    metrics = DEFAULT_METRICS.split(",") + [
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum"]
    return {"metrics": ",".join(dict.fromkeys(metrics)),
            "sections": list(A100_SECTIONS) + list(dict.fromkeys(ROOFLINE_SECTIONS[name] for name in rooflines))}


# Blackwell renamed the DRAM byte counters to dram__bytes_op_read/write. Record
# and validate them under the Ampere spelling so one canonical key serves every
# architecture and results stay comparable across GPUs.
CANONICAL_METRICS = {
    "dram__bytes_op_read.sum": "dram__bytes_read.sum",
    "dram__bytes_op_write.sum": "dram__bytes_write.sum",
}


def canonical_metric(name: str) -> str:
    return CANONICAL_METRICS.get(name, name)


def combine_metric_groups(names: list[str]) -> dict:
    return {"metrics": ",".join(dict.fromkeys(metric for name in names for metric in METRIC_GROUPS[name]["metrics"].split(","))),
            "sections": list(dict.fromkeys(section for name in names for section in METRIC_GROUPS[name]["sections"]))}


def matrix_geometry_known(operation: dict) -> bool:
    return all(str(operation.get(field, "")).isdigit() and int(operation[field]) > 0 for field in ("m", "n", "k"))


def validate_counter_capture(summary: dict, metrics: str | list[str], sections: list[str],
                             required_launches: int = 5, expected_devices: list[str] | None = None,
                             require_nvtx: bool = False) -> dict:
    """Validate explicit metrics and essential section evidence without zero filling.

    Launch counts here are per device; callers must additionally validate their
    exact selected operation signatures and documented sample availability.
    """
    requested = metrics.split(",") if isinstance(metrics, str) else metrics
    requested = [canonical_metric(name) for name in requested if ":" not in name]
    missing, section_missing = set(), {name: set() for name in sections}
    counts = collections.Counter(str(launch["device"]) for launch in summary.get("launches", []))
    invalid_nvtx = []
    for launch in summary.get("launches", []):
        def available(name: str) -> bool:
            value = launch.get("metrics", {}).get(name, {})
            return (value.get("available") is True and not value.get("ambiguous")
                    and isinstance(value.get("value"), (int, float)) and not isinstance(value["value"], bool)
                    and math.isfinite(value["value"]))
        missing.update(name for name in requested if not available(name))
        for section in sections:
            if section not in SECTION_REQUIREMENTS:
                section_missing[section].add("unregistered_section_requirements")
                continue
            # Blackwell's hierarchical charts include shared-memory traffic in
            # the L1/TEX writeback metric (mem_lgds); retain its native name.
            def section_available(name):
                return available(name) or ("lsu_writeback_active_mem_lg." in name and
                    available(name.replace("_mem_lg.", "_mem_lgds.")))
            section_missing[section].update(name for name in SECTION_REQUIREMENTS[section] if not section_available(name))
            if section == ROOFLINE_SECTIONS["tensor"]:
                # Require work and its corresponding precision-specific ceiling.
                # A zero work count is valid; a missing work count is not zero.
                work_suffix = ".sum.per_cycle_elapsed"
                work_metrics = [name for name in launch.get("metrics", {})
                                if name.startswith("sm__ops_path_tensor_") and name.endswith(work_suffix)]
                if not any(available(name) and available(name[:-len(work_suffix)] + ".sum.peak_sustained")
                           for name in work_metrics):
                    section_missing[section].add("matching_tensor_work_and_peak_metrics")
            if section == "InstructionStats":
                instances = launch.get("metric_instances", {}).get("sass__inst_executed_per_opcode", [])
                if not instances or not all(item.get("available") is True and not item.get("ambiguous")
                    and isinstance(item.get("value"), (int, float)) and not isinstance(item["value"], bool)
                    and math.isfinite(item["value"]) for item in instances):
                    section_missing[section].add("sass__inst_executed_per_opcode:available_instances")
        if require_nvtx:
            operation = launch.get("operation", {})
            if (operation.get("operation_match") != "nvtx_same_capture" or operation.get("phase") not in ("prefill", "decode")
                or not operation.get("role") or not operation.get("type")
                or (launch.get("class") != "attention" and not matrix_geometry_known(operation))):
                invalid_nvtx.append({"device": launch["device"], "id": launch["id"]})
    expected = expected_devices if expected_devices is not None else sorted(counts)
    errors = []
    if not counts:
        errors.append("No collected kernel launches")
    for device in expected:
        if counts[str(device)] < required_launches:
            errors.append(f"GPU {device}: {counts[str(device)]}/{required_launches} required launches")
    if missing:
        errors.append("Requested counters missing or unavailable: " + ", ".join(sorted(missing)))
    for section, names in section_missing.items():
        if names:
            errors.append(section + " essential evidence missing or unavailable: " + ", ".join(sorted(names)))
    if invalid_nvtx:
        errors.append(f"Missing same-capture NVTX operation/phase attribution for {len(invalid_nvtx)} launches")
    return {"valid": not errors, "missing_or_unavailable": sorted(missing),
            "section_coverage": {name: {"valid": not missing_names, "missing_or_unavailable": sorted(missing_names)}
                                 for name, missing_names in section_missing.items()},
            "launch_counts": dict(counts), "invalid_nvtx_launches": invalid_nvtx, "errors": errors}


def kernel_class(name: str) -> str:
    if "mul_mat_vec_q" in name:
        return "quantized_matvec"
    if "mul_mat_q" in name:
        return "quantized_matmul"
    if "dequantize" in name:
        return "dequantization"
    if "quantize" in name:
        return "activation_quantization"
    # Attention kernels may contain "mma"; classify them before generic matrix kernels.
    if any(token in name.lower() for token in ["fattn", "attention", "flash_attn", "flashattn"]):
        return "attention"
    if "Cijk_" in name or any(token in name.lower() for token in ["gemm", "gemv", "mul_mat", "mma"]):
        return "other_matrix_multiply"
    if "norm" in name:
        return "normalization"
    if "rope" in name:
        return "rotary_position"
    return "other"


def union_duration(intervals: list[tuple[int, int]]) -> int:
    total = 0
    last_end = -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, last_end))
        last_end = max(last_end, end)
    return total


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_ncu_report(report: pathlib.Path) -> pathlib.Path | None:
    """Accept both compressed and legacy Nsight Compute report formats."""
    return next((path for suffix in (".ncu-repz", ".ncu-rep")
                 if (path := report.with_suffix(suffix)).is_file()), None)


def server_build_provenance(server: pathlib.Path) -> dict:
    """Prefer verified adjacent build metadata; never inherit an enclosing repo HEAD."""
    server = server.resolve()
    manifest = server.parent / "build.json"
    if manifest.is_file():
        document = json.loads(manifest.read_text())
        expected = {server: document.get("server_sha256"),
                    server.parent / "libggml-cuda.so": document.get("cuda_library_sha256")}
        if document.get("server_impl_sha256"):
            expected[server.parent / "libllama-server-impl.so"] = document["server_impl_sha256"]
        if document.get("build_diff_sha256"):
            expected[server.parent / "build.diff"] = document["build_diff_sha256"]
        verified = {}
        for path, required_hash in expected.items():
            if not path.is_file() or not required_hash:
                verified[str(path)] = False
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            verified[str(path)] = digest.hexdigest() == required_hash
        valid = all(verified.values()) and bool(re.fullmatch(r"[0-9a-f]{40}", str(document.get("commit", ""))))
        return {"status": "verified_adjacent_build" if valid else "unverified_adjacent_build", "manifest": str(manifest),
                "commit": document.get("commit") if valid else None, "verified_files": verified,
                "build": document, "commit_scope": "Pinned source commit plus the recorded diagnostic patches"}
    candidate = server.parents[2] if len(server.parents) >= 3 else None
    if candidate and (candidate / "include/llama.h").is_file() and (candidate / "ggml/src").is_dir():
        revision = subprocess.run(["git", "-C", str(candidate), "rev-parse", "--show-toplevel", "HEAD"],
                                  capture_output=True, text=True, timeout=15)
        lines = revision.stdout.splitlines()
        if revision.returncode == 0 and len(lines) == 2 and pathlib.Path(lines[0]).resolve() == candidate.resolve():
            return {"status": "source_checkout", "commit": lines[1], "source_root": str(candidate),
                    "commit_scope": "Source checkout identity; binary hashes are recorded separately"}
    return {"status": "unknown", "commit": None, "reason": "No verified adjacent build or exact llama.cpp source checkout"}


def operation_from_nvtx(labels: list[str]) -> dict:
    operation, batch = {}, {}
    for label in labels:
        parsed = dict(re.findall(r"(?:^|:)([A-Za-z_]+)=([^:]+)", label))
        if label.startswith("csb_batch:"):
            batch = parsed
        elif label.startswith("csb_op:"):
            operation = parsed
    if not operation:
        return {}
    if "role" in operation:
        operation["tensor_name"] = operation["role"]
        operation["role"] = re.sub(r"\bblk\.\d+\.", "blk.*.", operation["role"])
        if re.fullmatch(r"node_[0-9]+", operation["role"]) and operation.get("op") == "FLASH_ATTN_EXT":
            operation["role"] = "op.FLASH_ATTN_EXT"
    if batch:
        operation.update(phase=batch.get("phase", "unknown"), batch_width=batch.get("width", ""),
                         prefill_tokens=batch.get("prefill_tokens", ""), decode_tokens=batch.get("decode_tokens", ""))
    operation.setdefault("phase", "unknown")
    operation["operation_match"] = "nvtx_same_capture"
    return operation


def attach_ncu_nvtx(report: pathlib.Path, profiler: str, summary: dict) -> dict:
    """Read NVTX stacks from the matching installed profiler's report API.

    CSV launch IDs are report action ordinals. Device, function and launch
    geometry are cross-checked before attaching same-capture operation labels.
    """
    extras = pathlib.Path(profiler).resolve().parent / "extras/python"
    sys.path.insert(0, str(extras))
    try:
        import ncu_report
        context = ncu_report.load_report(str(report))
        ordinal = 0
        attributed = 0
        associations = []
        by_id = {launch["id"]: launch for launch in summary["launches"]}
        for range_index in range(context.num_ranges()):
            capture_range = context.range_by_idx(range_index)
            for action_index in range(capture_range.num_actions()):
                action = capture_range.action_by_idx(action_index)
                launch = by_id.get(str(ordinal))
                ordinal += 1
                if launch is None:
                    continue
                def number(name: str) -> int:
                    return action.metric_by_name(name).as_uint64()
                geometry = {key: tuple(number(f"launch__{key}_dim_{axis}") for axis in "xyz") for key in ("grid", "block")}
                if (str(number("launch__device_id")) != str(launch["device"]) or action.name() not in launch["kernel"]
                    or any(geometry[key] != tuple(map(int, re.findall(r"\d+", launch[key]))) for key in geometry)):
                    raise ValueError("Report API/CSV launch identity mismatch; refusing NVTX attribution")
                state = action.nvtx_state()
                labels = []
                if state:
                    for domain_id in state.domains():
                        domain = state.domain_by_id(domain_id)
                        labels.extend(domain.push_pop_range(index).name() for index in range(len(domain.push_pop_ranges())))
                operation = operation_from_nvtx(labels)
                associations.append((launch, labels, operation))
                if operation:
                    attributed += 1
        for launch, labels, operation in associations:
            launch["nvtx_ranges"] = labels
            launch["operation"] = operation
        return {"status": "read", "attributed_launches": attributed, "report_api": str(extras)}
    finally:
        sys.path.remove(str(extras))


def summarize_ncu(csv_path: pathlib.Path, output: pathlib.Path) -> dict:
    """Retain per-launch counters and units; unavailable values never become zero."""
    # InstructionStats exports source/SASS instance lists in single wide CSV
    # cells, which readily exceed the csv module's default 128 KiB limit.
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
    source_rows = list(csv.reader(io.StringIO(csv_path.read_text(errors="replace"))))
    header_index = next((i for i, row in enumerate(source_rows)
                         if "ID" in row and "Kernel Name" in row), None)
    if header_index is None:
        raise ValueError("No Nsight Compute CSV header found")
    header = source_rows[header_index]
    launches = {}

    def add_metric(launch: dict, name: str, unit: str, raw: str,
                   section: str = "", instance: str = "") -> None:
        name = canonical_metric(name)
        aggregate_raw = raw
        # --print-metric-instances details exports wide cells as
        # "aggregate (correlation ID: value; correlation ID: value)".
        detailed = re.fullmatch(r"(.*?)\s+\((.*)\)", raw.strip(), re.DOTALL)
        if detailed and ": " in detailed[2]:
            aggregate_raw = detailed[1]
            for component in detailed[2].split("; "):
                label, separator, component_raw = component.rpartition(": ")
                if separator:
                    add_metric(launch, name, unit, component_raw, section, label)
        try:
            value = float(aggregate_raw.strip().replace(",", ""))
            if not math.isfinite(value):
                value = None
        except ValueError:
            value = None
        metric = {"value": value, "unit": unit, "raw": raw,
                  "available": value is not None, "section": section}
        if instance:
            # SASS opcode/correlation IDs are labels, not scalar totals. Never
            # overwrite the aggregate with the final opcode's instruction count.
            launch["metric_instances"].setdefault(name, []).append(
                {"instance": instance, **metric})
            return
        previous = launch["metrics"].get(name)
        if previous and (previous.get("ambiguous") or previous["unit"] != unit
                         or (previous.get("value") if previous.get("value") is not None else previous["raw"])
                         != (value if value is not None else raw)):
            metric["observations"] = previous.get("observations", [
                {key: previous[key] for key in ("value", "unit", "raw", "section")}]) + [
                {key: metric[key] for key in ("value", "unit", "raw", "section")}]
            metric.update(value=None, available=False, ambiguous=True)
        launch["metrics"][name] = metric

    wide_units = {}
    for cells in source_rows[header_index + 1:]:
        if len(cells) != len(header):
            continue  # Profiler progress/diagnostic lines may surround the CSV.
        row = dict(zip(header, cells))
        if not row.get("ID", "").isdigit():
            if "Metric Name" not in header and not row.get("Kernel Name"):
                wide_units = row
            continue
        key = (row["ID"], row.get("Process ID", ""), row.get("Device", ""),
               row.get("Context", ""), row.get("Stream", ""))
        launch = launches.setdefault(key, {
            "id": row["ID"], "process_id": row.get("Process ID", ""),
            "kernel": row["Kernel Name"], "class": kernel_class(row["Kernel Name"]),
            "device": row.get("Device", ""), "context": row.get("Context", ""),
            "stream": row.get("Stream", ""), "grid": row.get("Grid Size", ""),
            "block": row.get("Block Size", ""), "metrics": {}, "metric_instances": {},
        })
        if "Metric Name" in header:
            add_metric(launch, row["Metric Name"], row.get("Metric Unit", ""), row.get("Metric Value", ""),
                       row.get("Section Name", ""), row.get("Metric Instance", row.get("Correlation ID", "")))
        else:
            for name in header:
                if "__" in name:
                    add_metric(launch, name, wide_units.get(name, ""), row[name])
    if not launches or not any(launch["metrics"] or launch["metric_instances"] for launch in launches.values()):
        raise ValueError("Nsight Compute CSV contains no per-launch metrics")

    metric_columns = {
        "dram_read_bytes": "dram__bytes_read.sum", "dram_write_bytes": "dram__bytes_write.sum",
        "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1_hit_pct": "l1tex__t_sector_hit_rate.pct", "l2_hit_pct": "lts__t_sector_hit_rate.pct",
        "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "tensor_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "tensor_elapsed_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm_ipc_active": "sm__inst_executed.avg.per_cycle_active",
        "sm_ipc_elapsed": "sm__inst_executed.avg.per_cycle_elapsed",
        "sm_instructions": "sm__inst_executed.sum",
        "smsp_ipc_active": "smsp__inst_executed.avg.per_cycle_active",
        "global_load_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "global_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "global_load_bytes_per_sector": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
        "global_load_bytes_per_sector_max_rate": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.max_rate",
        "registers_per_thread": "launch__registers_per_thread",
        "shared_bytes_per_block": "launch__shared_mem_per_block",
        "occupancy_limit_registers": "launch__occupancy_limit_registers",
        "occupancy_limit_shared_mem": "launch__occupancy_limit_shared_mem",
        "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
        "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "local_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "local_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    }
    flat_rows = []
    for launch in launches.values():
        metrics = launch["metrics"]
        row = {key: launch[key] for key in ["id", "kernel", "class", "device", "grid", "block"]}
        row.update({label: metrics.get(name, {}).get("value") for label, name in metric_columns.items()})
        duration = metrics.get("gpu__time_duration.sum", {})
        factors = {"ns": 1, "nsecond": 1, "us": 1000, "usecond": 1000,
                   "ms": 1000000, "msecond": 1000000, "s": 1000000000, "second": 1000000000}
        factor = factors.get(duration.get("unit"))
        ns = duration["value"] * factor if duration.get("value") is not None and factor else None
        row["duration_us"] = ns / 1000 if ns is not None else None
        read, written = row["dram_read_bytes"], row["dram_write_bytes"]
        row["dram_bandwidth_gb_s"] = (read + written) / ns if ns and read is not None and written is not None else None
        sectors = metrics.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", {}).get("value")
        requests = metrics.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", {}).get("value")
        row["global_load_sectors_per_request"] = sectors / requests if requests and sectors is not None else None
        row["unavailable_metrics"] = ";".join(name for name, metric in metrics.items() if not metric["available"])
        flat_rows.append(row)
    summary = {
        "sample_count": len(launches), "launches": list(launches.values()),
        "observed_cuda_devices": sorted({launch["device"] for launch in launches.values()}),
        "unavailable_metric_values": sum(not metric["available"] for launch in launches.values()
                                         for metric in launch["metrics"].values()),
        "scope": "Selected kernel launches under replay; not whole-server utilization or performance.",
        "derived_metrics": {
            "dram_bandwidth_gb_s": "(DRAM read bytes + write bytes) / kernel duration in nanoseconds; decimal GB/s",
            "global_load_sectors_per_request": "32-byte global-load sectors per LSU request; ideal value depends on instruction width and active lanes",
            "sm_ipc_active": "Average executed warp instructions per active SM cycle; not directly comparable to AMD IPC conventions",
            "sm_ipc_elapsed": "Average executed warp instructions per elapsed SM cycle, including inactive cycles",
            "global_load_bytes_per_sector": "NVIDIA global-load bytes/sector ratio; compare with its reported max_rate and access pattern, not an AMD coalescing percentage",
        },
    }
    (output / "counter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "counter_summary.csv", flat_rows)
    # Long form retains every collected section counter and opcode instance,
    # including unavailable values. The compact table is intentionally a subset.
    write_csv(output / "counter_metrics.csv", [
        {"id": launch["id"], "device": launch["device"], "kernel": launch["kernel"],
         "metric": name, "instance": instance.get("instance", ""),
         **{key: instance[key] for key in ("value", "unit", "raw", "available", "section")}}
        for launch in launches.values()
        for name in set(launch["metrics"]) | set(launch["metric_instances"])
        for instance in ([launch["metrics"][name]] if name in launch["metrics"] else [])
                        + launch["metric_instances"].get(name, [])])
    return summary


def validate_rocm_region_version(rocm_root: pathlib.Path, profiler_help: str, *, counters: bool = True) -> dict:
    """Check documented region support; the remote sentinel must also pass."""
    if "--selected-regions" not in profiler_help:
        raise ValueError("rocprofv3 lacks --selected-regions; this runner will not profile warmup/model loading")
    version_text = ""
    version_file = None
    for name in ["version", "version-dev"]:
        candidate = pathlib.Path(rocm_root) / ".info" / name
        if candidate.exists():
            version_text = candidate.read_text().strip()
            version_file = str(candidate)
            if re.search(r"\d+\.\d+(?:\.\d+)?", version_text):
                break
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", version_text)
    version = tuple(int(part or 0) for part in match.groups()) if match else None
    if counters and (version is None or version < (7, 14, 0)):
        raise ValueError("Gated rocprofv3 counters require ROCm >= 7.14.0 and a passing region sentinel; "
                         "older --selected-regions implementations gate traces but not counters")
    return {"rocm_version": version_text or None, "version_file": version_file,
            "selected_regions": True, "counter_regions_minimum_rocm": "7.14.0",
            "remote_region_sentinel_required": counters}


def summarize_rocprof(output: pathlib.Path, *, counters: bool, requested_metrics: list[str] | None = None) -> dict:
    """Parse native rocprofv3 CSV without treating missing counters as zero."""
    def launch_shape(raw: dict, name: str) -> str:
        # Counter CSV uses a scalar size; trace CSV preserves XYZ dimensions.
        if raw.get(name):
            return raw[name]
        dimensions = [raw.get(f"{name}_{axis}") for axis in "XYZ"]
        return "x".join(dimensions) if all(dimensions) else ""

    pattern = "*counter_collection.csv" if counters else "*kernel_trace.csv"
    paths = sorted(output.rglob(pattern))
    launches = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            needed = {"Kernel_Name", "Dispatch_Id", "Start_Timestamp", "End_Timestamp"}
            if counters:
                needed |= {"Counter_Name", "Counter_Value"}
            if not needed <= set(reader.fieldnames or []):
                raise ValueError(f"Unsupported rocprofv3 CSV columns: {path.name}")
            for raw in reader:
                if not raw.get("Kernel_Name"):
                    continue
                key = (str(path.relative_to(output)), raw.get("Process_Id", ""),
                       raw.get("Agent_Id", ""), raw["Dispatch_Id"])
                start, end = int(raw["Start_Timestamp"]), int(raw["End_Timestamp"])
                if end <= start:
                    raise ValueError(f"Invalid ROCm dispatch timestamps: {raw['Dispatch_Id']}")
                launch = launches.setdefault(key, {
                    "id": raw["Dispatch_Id"], "source": key[0], "agent": raw.get("Agent_Id", ""),
                    "kernel": raw["Kernel_Name"], "class": kernel_class(raw["Kernel_Name"]),
                    "grid": launch_shape(raw, "Grid_Size"), "block": launch_shape(raw, "Workgroup_Size"),
                    "start_ns": start, "end_ns": end, "duration_ns": end - start,
                    "vgpr_count": raw.get("VGPR_Count", ""), "sgpr_count": raw.get("SGPR_Count", ""),
                    "lds_bytes": raw.get("LDS_Block_Size", ""), "metrics": {},
                })
                if counters:
                    name = raw["Counter_Name"]
                    try:
                        value = float((raw.get("Counter_Value") or "").replace(",", ""))
                        if not math.isfinite(value):
                            value = None
                    except ValueError:
                        value = None
                    if name in launch["metrics"]:
                        raise ValueError(f"Duplicate ROCm counter/dimension rows for dispatch {launch['id']}; inspect raw CSV")
                    launch["metrics"][name] = {"value": value, "raw": raw["Counter_Value"],
                        "unit": raw.get("Counter_Unit") or ("KiB" if name in {"FETCH_SIZE", "WRITE_SIZE", "FetchSize", "WriteSize"}
                            else "resident waves/CU" if name in {"MeanOccupancyPerCU", "MeanOccupancyPerActiveCU"} else None),
                        "available": value is not None,
                        "dimensions": {key: value for key, value in raw.items() if key and "dimension" in key.lower()} }
    if not launches:
        raise ValueError("No ROCm dispatch counter rows found" if counters else "No ROCm kernel trace rows found")
    rows = list(launches.values())
    if counters:
        requested = requested_metrics or []
        missing = sorted({name for launch in rows for name in requested
                          if not launch["metrics"].get(name, {}).get("available", False)})
        flat = []
        for launch in rows:
            metric = {name: value["value"] for name, value in launch["metrics"].items()}
            def ratio(numerator: str, denominator: str) -> float | None:
                a, b = metric.get(numerator), metric.get(denominator)
                return a / b if a is not None and b else None
            hit, miss = metric.get("TCC_HIT"), metric.get("TCC_MISS")
            valu, mfma = metric.get("SQ_INSTS_VALU"), metric.get("SQ_INSTS_MFMA")
            read = metric.get("FETCH_SIZE", metric.get("FetchSize"))
            write = metric.get("WRITE_SIZE", metric.get("WriteSize"))
            access_ratio = ratio("TA_TOTAL_WAVEFRONTS_sum", "TCP_TOTAL_ACCESSES_sum")
            launch["derived"] = {
                "l2_hit_pct": 100 * hit / (hit + miss) if hit is not None and miss is not None and hit + miss else None,
                "valu_non_mfma_instructions": valu - mfma if valu is not None and mfma is not None else None,
                "amd_ipc_convention": ratio("SQ_INSTS", "SQ_BUSY_CU_CYCLES"),
                "amd_coalescing_pct": 1600 * access_ratio if access_ratio is not None else None,
                "read_bandwidth_gb_s": 1024 * read / launch["duration_ns"] if read is not None else None,
                "write_bandwidth_gb_s": 1024 * write / launch["duration_ns"] if write is not None else None,
            }
            flat.append({key: launch[key] for key in ["id", "agent", "kernel", "class", "grid", "block", "duration_ns"]}
                        | {name: metric.get(name) for name in requested} | launch["derived"])
        summary = {"sample_count": len(rows), "launches": rows,
            "missing_or_unavailable_requested_metrics": missing,
            "scope": "One PMC group; selected dispatches may be serialized. Not unprofiled serving performance.",
            "counter_definitions": "Retain the installed rocprofv3 catalog; unlabelled raw counter units are not inferred.",
            "derived_formulas": {"amd_ipc_convention": "SQ_INSTS/SQ_BUSY_CU_CYCLES; AMD convention, not NVIDIA SMSP IPC",
                "amd_coalescing_pct": "100*64*TA_TOTAL_WAVEFRONTS_sum/(4*TCP_TOTAL_ACCESSES_sum); AMD gfx942 convention",
                "bandwidth": "1024*FETCH_SIZE_or_WRITE_SIZE/dispatch_duration_ns; decimal GB/s; operands from same dispatch/pass only",
                "valu_non_mfma": "SQ_INSTS_VALU-SQ_INSTS_MFMA; VALU includes MFMA"}}
        (output / "counter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_csv(output / "counter_summary.csv", flat)
        return summary
    total = sum(row["duration_ns"] for row in rows)
    classes = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for row in rows:
        classes[row["class"]].append(row)
        groups[(row["kernel"], row["agent"], row["grid"], row["block"], row["vgpr_count"], row["sgpr_count"], row["lds_bytes"])].append(row)
    class_rows = [{"class": name, "calls": len(values), "total_ms": sum(r["duration_ns"] for r in values) / 1e6,
                   "kernel_time_share_pct": 100 * sum(r["duration_ns"] for r in values) / total}
                  for name, values in classes.items()]
    class_rows.sort(key=lambda r: r["total_ms"], reverse=True)
    grouped = []
    for values in groups.values():
        first = values[0]
        duration = sum(r["duration_ns"] for r in values)
        grouped.append({key: first[key] for key in ["kernel", "class", "agent", "grid", "block", "vgpr_count", "sgpr_count", "lds_bytes"]}
                       | {"calls": len(values), "total_ms": duration / 1e6, "mean_us": duration / len(values) / 1000,
                          "kernel_time_share_pct": 100 * duration / total})
    grouped.sort(key=lambda row: row["total_ms"], reverse=True)
    summary = {"kernel_calls": len(rows), "sum_kernel_ms": total / 1e6, "classes": class_rows,
               "scope": "ROCm selected-region trace; timing includes profiler overhead; no hardware counters inferred."}
    (output / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "kernels.csv", [{k: v for k, v in row.items() if k != "metrics"} for row in rows])
    write_csv(output / "kernel_summary.csv", grouped)
    write_csv(output / "kernel_classes.csv", class_rows)
    return summary


def summarize_sqlite(sqlite_path: pathlib.Path, output: pathlib.Path,
                     operation_metadata: pathlib.Path | None = None) -> dict:
    """Summarize CUDA activity without inferring unavailable hardware counters."""
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise SystemExit("No CUDA kernel activity table found; capture is invalid for kernel analysis")
    kernels = [dict(row) for row in connection.execute("""
        SELECT k.start, k.end, s.value AS kernel, k.deviceId, k.streamId,
               k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
               k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
               k.localMemoryPerThread, k.graphNodeId, k.correlationId
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS s ON s.id=k.demangledName
        ORDER BY k.start
    """)]
    if not kernels:
        raise SystemExit("The trace contains no CUDA kernels")
    operations = []
    if operation_metadata:
        document = json.loads(operation_metadata.read_text())
        operations = document.get("operations", []) if isinstance(document, dict) else document
        for operation in operations:
            if "deviceId" not in operation or not ({"correlationId", "graphNodeId"} & set(operation)):
                raise ValueError("Operation metadata must carry deviceId and correlationId or graphNodeId")
    # Match diagnostic labels by CPU launch correlation, never by overlapping
    # asynchronous GPU time or similar-looking launch grids.
    nvtx_by_correlation = collections.defaultdict(list)
    if {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"} <= tables:
        nvtx_columns = {row[1] for row in connection.execute("PRAGMA table_info(NVTX_EVENTS)")}
        if {"start", "end", "globalTid", "text"} <= nvtx_columns:
            text_expr = "COALESCE(n.text, s.value)" if "textId" in nvtx_columns else "n.text"
            text_join = "LEFT JOIN StringIds s ON n.textId=s.id" if "textId" in nvtx_columns else ""
            # Sweep nested ranges per CPU thread; a SQL interval cross-join
            # becomes quadratic for long contexts and millions of launches.
            ranges_by_thread = collections.defaultdict(list)
            query = f"""SELECT n.globalTid, n.start, n.end, {text_expr} AS label
                FROM NVTX_EVENTS n {text_join}
                WHERE ({text_expr} LIKE 'csb_op:%' OR {text_expr} LIKE 'csb_batch:%') AND n.end IS NOT NULL
                ORDER BY n.start, n.end DESC"""
            for row in connection.execute(query):
                ranges_by_thread[row["globalTid"]].append(dict(row))
            cursors = collections.defaultdict(int)
            active_by_thread = collections.defaultdict(list)
            for runtime in connection.execute("SELECT correlationId, globalTid, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"):
                thread = runtime["globalTid"]
                ranges = ranges_by_thread.get(thread, [])
                cursor = cursors[thread]
                active = active_by_thread[thread]
                while cursor < len(ranges) and ranges[cursor]["start"] <= runtime["start"]:
                    active.append(ranges[cursor])
                    cursor += 1
                cursors[thread] = cursor
                active = [row for row in active if row["end"] >= runtime["start"]]
                active_by_thread[thread] = active
                labels = [row["label"] for row in active if row["end"] >= runtime["end"]]
                if labels:
                    nvtx_by_correlation[runtime["correlationId"]] = labels
    grouped = collections.defaultdict(list)
    classes = collections.defaultdict(list)
    for kernel in kernels:
        kernel["duration_ns"] = kernel["end"] - kernel["start"]
        kernel["class"] = kernel_class(kernel["kernel"])
        matches = [op for op in operations if op["deviceId"] == kernel["deviceId"] and
                   ((op.get("correlationId") is not None and op["correlationId"] == kernel["correlationId"]) or
                    (op.get("graphNodeId") not in (None, 0) and op["graphNodeId"] == kernel["graphNodeId"]))]
        operation = matches[0] if len(matches) == 1 else operation_from_nvtx(nvtx_by_correlation.get(kernel["correlationId"], []))
        for field in ["phase", "role", "type", "m", "n", "k", "fusion"]:
            kernel[field] = operation.get(field, "unknown" if field == "phase" else "")
        kernel["tensor_name"] = operation.get("tensor_name", "")
        kernel["operation_match"] = "external_id" if len(matches) == 1 else "nvtx_launch_correlation" if operation else "unavailable"
        # Keep different batch shapes/resources separate rather than averaging them away.
        key = tuple(kernel[field] for field in ["kernel", "deviceId", "gridX", "gridY", "gridZ",
                    "blockX", "blockY", "blockZ", "registersPerThread", "staticSharedMemory", "dynamicSharedMemory",
                    "phase", "role", "type", "m", "n", "k", "fusion"])
        grouped[key].append(kernel)
        classes[kernel["class"]].append(kernel["duration_ns"])
    total = sum(kernel["duration_ns"] for kernel in kernels)
    grouped_rows = []
    for values in grouped.values():
        first = values[0]
        durations = [value["duration_ns"] for value in values]
        grouped_rows.append({
            "kernel": first["kernel"], "class": first["class"], "calls": len(values),
            "total_ms": sum(durations) / 1e6, "kernel_time_share_pct": 100 * sum(durations) / total,
            "mean_us": statistics.mean(durations) / 1000, "median_us": statistics.median(durations) / 1000,
            "min_us": min(durations) / 1000, "max_us": max(durations) / 1000,
            "grid": "x".join(str(first[f"grid{dim}"]) for dim in "XYZ"),
            "block": "x".join(str(first[f"block{dim}"]) for dim in "XYZ"),
            "registers_per_thread": first["registersPerThread"],
            "shared_bytes_per_block": first["staticSharedMemory"] + first["dynamicSharedMemory"],
            "local_bytes_per_thread": first["localMemoryPerThread"], "device": first["deviceId"],
            **{field: first[field] for field in ["phase", "role", "type", "m", "n", "k", "fusion", "operation_match"]},
            "contributing_tensor_names": json.dumps(sorted({value["tensor_name"] for value in values if value["tensor_name"]})),
            "tensor_launch_counts": json.dumps(dict(collections.Counter(value["tensor_name"] for value in values if value["tensor_name"]))),
        })
    grouped_rows.sort(key=lambda row: row["total_ms"], reverse=True)
    write_csv(output / "kernels.csv", kernels)
    write_csv(output / "kernel_summary.csv", grouped_rows)
    class_rows = [{"class": name, "calls": len(durations), "total_ms": sum(durations) / 1e6,
                   "kernel_time_share_pct": 100 * sum(durations) / total}
                  for name, durations in sorted(classes.items(), key=lambda item: -sum(item[1]))]
    write_csv(output / "kernel_classes.csv", class_rows)

    copies = []
    copy_device_pairs = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        copies = [dict(row) for row in connection.execute("""
            SELECT e.label AS direction, e.name AS copy_kind, count(*) AS calls, sum(m.bytes) AS bytes,
                   sum(m.end-m.start)/1000000.0 AS total_ms
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
            LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS e ON m.copyKind=e.id
            GROUP BY m.copyKind ORDER BY total_ms DESC
        """)]
        copy_columns = {row[1] for row in connection.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")}
        if {"srcDeviceId", "dstDeviceId"} <= copy_columns:
            execution_device = "m.deviceId" if "deviceId" in copy_columns else "NULL"
            source_kind = destination_kind = "NULL"
            memory_kind_joins = ""
            if "ENUM_CUDA_MEM_KIND" in tables and {"srcKind", "dstKind"} <= copy_columns:
                source_kind, destination_kind = "source_kind.label", "destination_kind.label"
                memory_kind_joins = """
                    LEFT JOIN ENUM_CUDA_MEM_KIND AS source_kind ON m.srcKind=source_kind.id
                    LEFT JOIN ENUM_CUDA_MEM_KIND AS destination_kind ON m.dstKind=destination_kind.id
                """
            copy_device_pairs = [dict(row) for row in connection.execute(f"""
                SELECT e.label AS direction, e.name AS copy_kind,
                       {execution_device} AS execution_device,
                       m.srcDeviceId AS source_device, m.dstDeviceId AS destination_device,
                       {source_kind} AS source_memory_kind, {destination_kind} AS destination_memory_kind,
                       count(*) AS calls, sum(m.bytes) AS bytes,
                       sum(m.end-m.start)/1000000.0 AS total_ms
                FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
                LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS e ON m.copyKind=e.id
                {memory_kind_joins}
                GROUP BY m.copyKind, execution_device, m.srcDeviceId, m.dstDeviceId,
                         source_memory_kind, destination_memory_kind ORDER BY total_ms DESC
            """)]
    write_csv(output / "memory_copies.csv", copy_device_pairs or copies)
    api_rows = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api_rows = [dict(row) for row in connection.execute("""
            SELECT s.value AS api, count(*) AS calls, sum(r.end-r.start)/1000000.0 AS total_ms
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r JOIN StringIds AS s ON r.nameId=s.id
            GROUP BY r.nameId ORDER BY total_ms DESC
        """)]
    write_csv(output / "cuda_api.csv", api_rows)
    devices = sorted({kernel["deviceId"] for kernel in kernels})
    timelines = []
    active_ns_by_device = {}
    for device in devices:
        intervals = [(kernel["start"], kernel["end"]) for kernel in kernels if kernel["deviceId"] == device]
        span = max(end for _, end in intervals) - min(start for start, _ in intervals)
        active = union_duration(intervals)
        active_ns_by_device[device] = active
        timelines.append({"device": device, "kernel_calls": len(intervals),
                          "sum_kernel_ms": sum(end - start for start, end in intervals) / 1e6,
                          "first_to_last_kernel_ms": span / 1e6,
                          "kernel_active_union_ms": active / 1e6,
                          "kernel_timeline_active_pct": 100 * active / span if span else None})
    two_device_overlap = None
    if len(devices) == 2:
        intervals = [(kernel["start"], kernel["end"]) for kernel in kernels]
        span = max(end for _, end in intervals) - min(start for start, _ in intervals)
        # Inclusion-exclusion on the two device unions counts simultaneous
        # activity once, even when kernels overlap within either device.
        overlap = sum(active_ns_by_device.values()) - union_duration(intervals)
        two_device_overlap = {
            "devices": devices, "both_devices_kernel_active_ms": overlap / 1e6,
            "overall_kernel_span_ms": span / 1e6,
            "fraction_of_overall_kernel_span": overlap / span if span else None,
            "scope": "Intersection of per-device CUDA kernel activity unions; not hardware utilization or scaling efficiency.",
        }
    summary = {
        "kernel_calls": len(kernels), "sum_kernel_ms": total / 1e6,
        "classes": class_rows, "timelines": timelines, "memory_copies": copies,
        "observed_cuda_devices": devices, "copy_device_pairs": copy_device_pairs,
        "two_device_kernel_overlap": two_device_overlap,
        "operation_attribution": {"matched_kernel_calls": sum(k["operation_match"] != "unavailable" for k in kernels),
                                  "phase_known_kernel_calls": sum(k["phase"] in ("prefill", "decode") for k in kernels),
                                  "method": "Explicit device/correlation or graph-node IDs, or CPU-launch NVTX containment"},
        "peer_copy_bytes": sum(row["bytes"] for row in copies if row["copy_kind"] == "CUDA_MEMCPY_KIND_PTOP"),
        "peer_copy_bytes_scope": "Only PTOP-labelled activity records. Peer API transfers can appear as paired differently labelled records; zero does not establish zero inter-GPU traffic.",
        "limitations": [
            "Trace timings include profiler overhead; use separate unprofiled HTTP results for performance.",
            "Kernel classes describe function names, not exact whole-request prefill/decode boundaries.",
            "Sum of kernel durations counts overlapping kernels separately; union time does not.",
            "Timeline-active percentage is not SM utilization or achieved occupancy.",
            "Copies are explicit CUDA transfers, not kernel DRAM traffic or memory bandwidth.",
            "Peer-copy bytes describe explicit transfers; they do not establish NVLink versus PCIe routing or include peer loads in kernels.",
            "Registers/shared memory are launch resource usage, not achieved occupancy or instruction counters.",
        ],
    }
    (output / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# CUDA serving trace", "", f"{len(kernels):,} kernel calls; {total / 1e6:.2f} ms summed kernel time.", "",
             "| Kernel class | Calls | GPU time (ms) | Share |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row['class']} | {row['calls']} | {row['total_ms']:.2f} | {row['kernel_time_share_pct']:.1f}% |"
                 for row in class_rows)
    if two_device_overlap:
        lines += ["", f"Both GPUs have kernels active for {two_device_overlap['both_devices_kernel_active_ms']:.2f} ms "
                  f"({100 * two_device_overlap['fraction_of_overall_kernel_span']:.1f}% of the overall kernel span). "
                  "This is CUDA trace activity overlap, not hardware utilization or scaling efficiency."]
    lines += ["", "Kernel-time shares include overlap. These are profiler observations, not serving throughput.",
              "Hardware counters (DRAM bandwidth, cache hits, achieved occupancy, instruction mix) are not present in this trace.",
              "See `kernel_summary.csv` for individual functions, launch shapes, registers and shared memory.", ""]
    (output / "profile_summary.md").write_text("\n".join(lines))
    print(f"Summarized {len(kernels):,} CUDA kernel calls in {output}")
    return summary



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=["nsys", "ncu", "rocprof-trace", "rocprof-counters"], required=True)
    parser.add_argument("--profiler", required=True)
    parser.add_argument("--gate-library", type=pathlib.Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", required=True, help="Model alias served by llama-server")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--topics", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / "prompts/topics.txt")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--repeat-prompt", action="store_true",
                        help="Use the same prepared input for every request")
    parser.add_argument("--prefix-reuse", action="store_true",
                        help="Prime slots, warm with prefix reuse, then capture a reuse burst")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-regex", default="mul_mat_q|mul_mat_vec_q")
    parser.add_argument("--launch-count", type=int, default=4)
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--metric-group", choices=sorted(METRIC_GROUPS))
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--devices", help="NCU device IDs after CUDA visibility mapping; keeps both server GPUs visible")
    parser.add_argument("--expected-devices", help="Comma-separated logical GPU IDs whose trace/counter coverage is required")
    parser.add_argument("--nvtx-include", help="Optional diagnostic NVTX operation/phase filter")
    parser.add_argument("--require-nvtx", action="store_true",
                        help="Fail counter collection unless every launch has same-capture operation/phase labels")
    parser.add_argument("--operation-metadata", type=pathlib.Path,
                        help="Diagnostic operation JSON; must carry matching CUDA correlation/graph-node IDs")
    parser.add_argument("--exact-prompt-tokens", action="store_true")
    parser.add_argument("--prompt-fixture", type=pathlib.Path,
                        help="Reuse the first C canonical messages from the serving cell's prompts.json")
    parser.add_argument("--cache-control", choices=["none", "all"], default="none")
    parser.add_argument("--clock-control", choices=["none", "base"], default="none")
    parser.add_argument("--filter-mode", choices=["global", "per-gpu", "per-launch-config"], default="global")
    parser.add_argument("--rocm-root", type=pathlib.Path, default=pathlib.Path(os.environ.get("ROCM_PATH", "/opt/rocm")))
    parser.add_argument("--kernel-iteration-range", default=None,
                        help="ROCm per-kernel dispatch ordinals within the selected region, e.g. [1-2]")
    parser.add_argument("server_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    is_rocm = args.tool.startswith("rocprof-")
    if args.metric_group:
        if is_rocm:
            parser.error("--metric-group is specific to NVIDIA counters")
        group = METRIC_GROUPS[args.metric_group]
        args.metrics = args.metrics or group["metrics"]
        args.section = list(dict.fromkeys(args.section + group["sections"]))
    args.metrics = ",".join(re.split(r"[,\s]+", (args.metrics or ("SQ_WAVES" if is_rocm else DEFAULT_METRICS)).strip()))
    server = args.server_command
    if server[:1] == ["--"]:
        server = server[1:]
    if not server or args.prompt_tokens < 1 or args.output_tokens < 1 or args.concurrency < 1:
        parser.error("A server command, positive concurrency and positive token counts are required")
    for value in (args.devices, args.expected_devices):
        if value and not re.fullmatch(r"\d+(,\d+)*", value):
            parser.error("Device selection must be comma-separated nonnegative integers")
    if args.launch_count < 1 or args.launch_skip < 0:
        parser.error("Launch count must be positive and launch skip nonnegative")
    if args.prefix_reuse and not args.repeat_prompt:
        parser.error("--prefix-reuse requires --repeat-prompt for the controlled identical-input comparison")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    gate = args.output / "capture"
    report = args.output / "profile"
    target = ["env", f"LD_PRELOAD={args.gate_library.resolve()}", f"CSB_PROFILE_TOOL={args.tool}",
              f"CSB_PROFILE_PREFIX={gate}", *server]
    if args.tool == "nsys":
        command = [args.profiler, "profile", "--trace=cuda,nvtx",
                   "--sample=none", "--cpuctxsw=none", "--cuda-graph-trace=node",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                   "--force-overwrite=true", f"--output={report}", *target]
    elif args.tool == "ncu":
        command = [args.profiler, "--config-file", "off", "--profile-from-start", "off",
                   "--replay-mode", "kernel", "--graph-profiling", "node",
                   "--rename-kernels", "off",
                   "--clock-control", args.clock_control, "--cache-control", args.cache_control,
                   "--filter-mode", args.filter_mode,
                   "--kernel-name-base", "demangled", "--kernel-name", f"regex:{args.kernel_regex}",
                   "--launch-count", str(args.launch_count), "--launch-skip", str(args.launch_skip),
                   "--metrics", args.metrics, "--csv", "--page", "raw", "--print-units", "base",
                   "--log-file", str(args.output / "ncu.csv"),
                   "--export", str(report)]
        for section in args.section:
            command += ["--section", section]
        if args.devices:
            command += ["--devices", args.devices]
        if args.nvtx_include or os.environ.get("CSB_NVTX_OPS") == "1":
            command += ["--nvtx"]
        if args.nvtx_include:
            command += ["--nvtx-include", args.nvtx_include]
        command += target
    else:
        args.kernel_iteration_range = args.kernel_iteration_range or f"[{args.launch_skip + 1}-{args.launch_skip + args.launch_count}]"
        # rocprofv3 builds its own LD_PRELOAD injection list; use its supported
        # --preload option instead of replacing that list in an env child.
        command = [args.profiler, "--preload", str(args.gate_library.resolve()),
                   "--selected-regions", "--hip-trace", "--kernel-trace", "--marker-trace",
                   "--memory-copy-trace", "--output-format", "csv", "--output-directory", str(args.output / "rocprof"),
                   "--output-file", "capture"]
        if args.tool == "rocprof-counters":
            command += ["--kernel-include-regex", args.kernel_regex, "--kernel-iteration-range", args.kernel_iteration_range,
                        "--pmc", *args.metrics.split(",")]
        command += ["--", *server]
    metadata = {
        "tool": args.tool, "command": command, "server_command": server,
        "concurrency": args.concurrency, "target_prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens, "profiled_requests": args.concurrency,
        "topics_file": str(args.topics.resolve()), "seed": args.seed,
        "repeat_prompt": args.repeat_prompt, "prefix_reuse": args.prefix_reuse,
        "cache_policy": "require" if args.prefix_reuse else "forbid",
        "priming_requests": args.concurrency if args.prefix_reuse else 0,
        "warmup_requests": args.concurrency, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timing_use": "diagnostic only; do not mix with unprofiled serving performance",
        "exact_prompt_tokens": args.exact_prompt_tokens,
        "phase_scope": "warm-prefix diagnostic; residual prefill must be measured" if args.prefix_reuse else "uncached prefill and decode burst",
        "operation_metadata": str(args.operation_metadata.resolve()) if args.operation_metadata else None,
        "prompt_fixture": str(args.prompt_fixture.resolve()) if args.prompt_fixture else None,
        "environment": {key: os.environ.get(key) for key in
                        ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_P2P", "GGML_CUDA_REGISTER_HOST"]},
    }
    if not is_rocm:
        metadata["cuda_configuration"] = {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "server_split_parameters": {flag: server[server.index(flag) + 1]
                                        for flag in ["--split-mode", "--tensor-split", "--main-gpu"] if flag in server},
            "capture_scope": "all visible server devices, synchronized before the first stop",
            "profiler_api_scope": "each device primary context" if args.tool == "ncu" else "one Nsight Systems capture session",
            "peer_access_note": "Capability is not proof of enabled peer access or of NVLink routing",
        }
    if args.tool == "ncu":
        metadata["counter_collection"] = {
            "metrics": args.metrics.split(","), "kernel_regex": args.kernel_regex,
            "metric_group": args.metric_group, "sections": args.section,
            "metric_groups": [name for name, group in METRIC_GROUPS.items()
                              if set(group["metrics"].split(",")) <= set(args.metrics.split(","))
                              and set(group["sections"]) <= set(args.section)],
            "devices": args.devices, "expected_devices": args.expected_devices,
            "nvtx_include": args.nvtx_include,
            "require_nvtx": args.require_nvtx,
            "launch_skip": args.launch_skip, "launch_count": args.launch_count,
            "filter_mode": args.filter_mode, "replay_mode": "kernel", "graph_profiling": "node",
            "rename_kernels": False,
            "cache_control": args.cache_control, "clock_control": args.clock_control,
            "launch_count_scope": {"per-launch-config": "per grid/block/shared-memory configuration; not a per-device guarantee",
                                   "per-gpu": "per GPU", "global": "all matching launches"}[args.filter_mode],
        }
    elif args.tool == "rocprof-counters":
        metadata["counter_collection"] = {
            "metrics": args.metrics.split(","), "kernel_regex": args.kernel_regex,
            "kernel_iteration_range": args.kernel_iteration_range, "iteration_scope": "per kernel inside one selected region",
            "gate": "ROCTx process-wide Resume(0)/Pause(0) with --selected-regions",
            "passes": 1, "application_replay": False, "clock_control": "unmodified",
            "cache_control": "unmodified", "dispatch_serialization": "rocprofv3 counter collection may serialize dispatches",
        }
    # Nsight embeds its inherited environment in reports. The local-model server
    # needs no download credentials or editor/session environment.
    profiler_environment = {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP",
        "LANG", "LC_ALL", "LC_CTYPE", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "CUDA_CACHE_PATH",
        "CUDA_HOME", "CUDA_PATH", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS", "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_FORCE_MMQ",
        "GGML_CUDA_P2P", "GGML_CUDA_REGISTER_HOST",
        "GGML_CUDA_FORCE_CUBLAS", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
        "CSB_NVTX_OPS",
        "ROCM_PATH", "HIP_PATH", "HSA_XNACK", "HSA_OVERRIDE_GFX_VERSION",
    }}
    if is_rocm:
        profiler_environment["CSB_PROFILE_PREFIX"] = str(gate)
    metadata["profiler_environment_keys"] = sorted(profiler_environment)
    metadata["server_build_provenance"] = server_build_provenance(pathlib.Path(server[0]))
    metadata["server_code_commit"] = metadata["server_build_provenance"]["commit"]
    if args.tool == "ncu":
        version = subprocess.run([args.profiler, "--version"], capture_output=True, text=True,
                                 env=profiler_environment, timeout=15)
        metadata["profiler_version"] = version.stdout.strip()
        driver_version = pathlib.Path("/proc/driver/nvidia/version")
        metadata["driver_version"] = driver_version.read_text() if driver_version.exists() else None
        try:
            identity = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,driver_version",
                                       "--format=csv,noheader"], capture_output=True, text=True,
                                      env=profiler_environment, timeout=15)
            metadata["gpu_identity_csv"] = identity.stdout.strip() if identity.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            metadata["gpu_identity_csv"] = None
    elif is_rocm:
        try:
            help_result = subprocess.run([args.profiler, "--help"], capture_output=True, text=True,
                                         env=profiler_environment, timeout=30)
            help_text = help_result.stdout + help_result.stderr
            required_flags = ["--preload", "--selected-regions", "--hip-trace", "--kernel-trace", "--marker-trace", "--memory-copy-trace"]
            if args.tool == "rocprof-counters":
                required_flags += ["--pmc", "--kernel-include-regex", "--kernel-iteration-range"]
            missing_flags = [flag for flag in required_flags if flag not in help_text]
            if help_result.returncode or missing_flags:
                raise ValueError("Installed rocprofv3 lacks required capabilities: " + ", ".join(missing_flags))
            metadata["rocm_region_capability"] = validate_rocm_region_version(args.rocm_root, help_text,
                                                                           counters=args.tool == "rocprof-counters")
            version = subprocess.run([args.profiler, "--version"], capture_output=True, text=True,
                                     env=profiler_environment, timeout=30)
            metadata["profiler_version"] = (version.stdout + version.stderr).strip()
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            metadata.update(status="failed", counter_status="unsupported_profiler", error=str(error))
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(str(error), flush=True)
            return 2
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log = (args.output / "server-profiler.log").open("w")
    process = None
    exit_code = 0

    def require_running() -> None:
        if process.poll() is not None:
            raise RuntimeError(f"Profiler/server exited with {process.returncode}; see server-profiler.log and ncu.csv")
        error = pathlib.Path(str(gate) + ".error")
        if error.exists():
            raise RuntimeError("GPU profiler gate: " + error.read_text())

    def wait_gate(suffix: str) -> None:
        deadline = time.monotonic() + min(args.timeout, 60)
        while not pathlib.Path(str(gate) + suffix).exists():
            require_running()
            if time.monotonic() > deadline:
                raise TimeoutError(f"Profiler gate did not acknowledge {suffix}")
            time.sleep(0.02)

    def burst(prepared: list, cache_prompt: bool = False) -> list:
        barrier = threading.Barrier(args.concurrency)

        def request(index):
            messages, prompt_tokens = prepared[index]
            barrier.wait()
            record = stream_completion(args.base_url, args.model, messages,
                                       args.output_tokens, args.timeout, args.seed + index,
                                       cache_prompt=cache_prompt)
            finalize_record(record, prompt_tokens, args.output_tokens,
                            cache_policy="require" if cache_prompt else "forbid",
                            expected_prompt_tokens=args.prompt_tokens if args.exact_prompt_tokens else None)
            record.pop("_text", None)
            input_sha256 = hashlib.sha256(json.dumps(messages, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
            return {"index": index, "prepared_prompt_tokens": prompt_tokens,
                    "input_sha256": input_sha256, "prompt_cache_requested": cache_prompt, **record}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            return list(pool.map(request, range(args.concurrency)))

    try:
        try:
            with urllib.request.urlopen(args.base_url + "/health", timeout=2):
                raise RuntimeError("A server already occupies the requested URL")
        except urllib.error.URLError:
            pass
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=profiler_environment)
        deadline = time.monotonic() + args.timeout
        while True:
            require_running()
            try:
                with urllib.request.urlopen(args.base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except urllib.error.URLError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Server did not become healthy")
            time.sleep(0.2)
        tokenizer = ServerTokenizer(args.base_url, args.timeout)
        topics = [line.strip() for line in args.topics.read_text().splitlines() if line.strip()]
        if not topics:
            raise ValueError("The topics file is empty")
        if args.prompt_fixture:
            fixture = json.loads(args.prompt_fixture.read_text())
            if len(fixture) < args.concurrency:
                raise ValueError("Prompt fixture has fewer messages than concurrent clients")
            prepared = [(row["messages"], tokenizer.count_messages(row["messages"])) for row in fixture[:args.concurrency]]
            if args.repeat_prompt:
                prepared = [prepared[0]] * args.concurrency
        elif args.repeat_prompt:
            prepared = [make_messages(tokenizer, topics[0], 0, args.prompt_tokens, exact=args.exact_prompt_tokens)] * args.concurrency
        else:
            prepared = [make_messages(tokenizer, topics[i % len(topics)], i, args.prompt_tokens, exact=args.exact_prompt_tokens)
                        for i in range(args.concurrency)]
        if args.exact_prompt_tokens and any(tokens != args.prompt_tokens for _, tokens in prepared):
            raise ValueError("Prepared prompt count does not equal the requested study input length")
        (args.output / "prompts.json").write_text(json.dumps(prepared, indent=2) + "\n")
        metadata["input_sha256_definition"] = "SHA256 of messages JSON, sorted keys, compact separators, UTF-8 without ASCII escaping"
        if args.prefix_reuse:
            # All concurrent slots first receive the full input with reuse disabled.
            # A second, reused burst warms the resulting short-prefill CUDA shapes.
            metadata["priming"] = burst(prepared, cache_prompt=False)
            if any(not record["ok"] for record in metadata["priming"]):
                raise RuntimeError("Prefix-cache priming failed")
        metadata["warmup"] = burst(prepared, cache_prompt=args.prefix_reuse)
        if any(not record["ok"] for record in metadata["warmup"]):
            raise RuntimeError("Warmup failed")
        pathlib.Path(str(gate) + ".start").touch()
        wait_gate(".started")
        if not is_rocm:
            metadata["cuda_configuration"].update(json.loads(pathlib.Path(str(gate) + ".devices").read_text()))
        metadata["requests"] = burst(prepared, cache_prompt=args.prefix_reuse)
        pathlib.Path(str(gate) + ".stop").touch()
        wait_gate(".stopped")
        if any(not record["ok"] for record in metadata["requests"]):
            raise RuntimeError("One or more profiled requests failed")
        metadata["status"] = "captured"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        print(metadata["error"], flush=True)
        exit_code = 2
    finally:
        if process is not None:
            # Signal only the server first so the profiler can finalize/export.
            pid_file = pathlib.Path(str(gate) + ".pid")
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            metadata["profiler_exit_code"] = process.returncode
            if process.returncode != 0 and not exit_code:
                metadata["status"] = "failed"
                metadata["error"] = f"Profiler exited with {process.returncode}"
                exit_code = 2
        log.close()
        if args.tool == "ncu":
            counter_log = args.output / "ncu.csv"
            counter_text = counter_log.read_text(errors="replace") if counter_log.exists() else ""
            counter_text += (args.output / "server-profiler.log").read_text(errors="replace")
            if "ERR_NVGPUCTRPERM" in counter_text:
                metadata["counter_status"] = "permission_denied"
                metadata["status"] = "failed"
                metadata["error"] = "ERR_NVGPUCTRPERM: NVIDIA hardware-counter permission denied"
                exit_code = 2
            elif "driver resource was unavailable" in counter_text.lower():
                metadata["counter_status"] = "resource_unavailable"
                metadata["status"] = "failed"
                metadata["error"] = "NVIDIA profiling resource unavailable; another counter collector such as DCGM may hold it"
                exit_code = 2
            elif (counter_report := find_ncu_report(report)) is None:
                metadata["counter_status"] = "no_report"
                metadata["status"] = "failed"
                metadata.setdefault("error", "Nsight Compute produced no counter report")
                exit_code = 2
            else:
                metadata["counter_status"] = "report_created"
                metadata["counter_report"] = str(counter_report)
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if is_rocm:
        try:
            summary = summarize_rocprof(args.output, counters=args.tool == "rocprof-counters",
                                       requested_metrics=args.metrics.split(",") if args.tool == "rocprof-counters" else None)
            missing = summary.get("missing_or_unavailable_requested_metrics", [])
            metadata["missing_or_unavailable_requested_metrics"] = missing
            if missing:
                raise ValueError("Requested ROCm counters missing or unavailable: " + ", ".join(missing))
            if args.tool == "rocprof-counters":
                metadata["counter_status"] = "collected"
                metadata["counter_sample_count"] = summary["sample_count"]
        except ValueError as error:
            log_text = (args.output / "server-profiler.log").read_text(errors="replace").lower()
            metadata["status"] = "failed"
            metadata["counter_status"] = "permission_denied" if "permission denied" in log_text else "incomplete_metrics"
            metadata["error"] = str(error)
            exit_code = 2
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.tool == "nsys" and report.with_suffix(".nsys-rep").exists():
        with (args.output / "export.log").open("w") as export_log:
            exported = subprocess.run([args.profiler, "export", "--type=sqlite", "--force-overwrite=true",
                                       f"--output={report}.sqlite", f"{report}.nsys-rep"],
                                      stdout=export_log, stderr=subprocess.STDOUT, env=profiler_environment)
        if exported.returncode:
            return 2
        summary = summarize_sqlite(report.with_suffix(".sqlite"), args.output, args.operation_metadata)
        metadata["observed_cuda_devices"] = summary["observed_cuda_devices"]
        metadata["peer_copy_bytes"] = summary["peer_copy_bytes"]
        metadata["peer_copy_bytes_scope"] = summary["peer_copy_bytes_scope"]
        if args.expected_devices:
            missing = set(args.expected_devices.split(",")) - {str(device) for device in summary["observed_cuda_devices"]}
            metadata["missing_expected_devices"] = sorted(missing)
            if missing:
                metadata.update(status="failed", error="Missing trace coverage for logical GPU(s): " + ",".join(sorted(missing)))
                exit_code = 2
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    elif args.tool == "ncu" and metadata.get("counter_status") == "report_created":
        export_command = [args.profiler, "--import", metadata["counter_report"],
                          "--page", "raw", "--csv", "--print-units", "base",
                          "--print-metric-instances", "details", "--rename-kernels", "off"]
        metadata["counter_export_command"] = export_command
        with (args.output / "ncu_raw.csv").open("w") as raw, (args.output / "export.log").open("w") as export_log:
            exported = subprocess.run(export_command, stdout=raw, stderr=export_log, env=profiler_environment)
        try:
            if exported.returncode:
                raise RuntimeError(f"Nsight Compute CSV export exited with {exported.returncode}")
            summary = summarize_ncu(args.output / "ncu_raw.csv", args.output)
            try:
                metadata["nvtx_attribution"] = attach_ncu_nvtx(pathlib.Path(metadata["counter_report"]), args.profiler, summary)
            except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as error:
                metadata["nvtx_attribution"] = {"status": "unavailable", "reason": str(error)}
            (args.output / "counter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            metadata["observed_cuda_devices"] = summary["observed_cuda_devices"]
            if args.expected_devices:
                missing_devices = set(args.expected_devices.split(",")) - set(summary["observed_cuda_devices"])
                metadata["missing_expected_devices"] = sorted(missing_devices)
                if missing_devices:
                    raise ValueError("Missing counter coverage for logical GPU(s): " + ",".join(sorted(missing_devices)))
            validation = validate_counter_capture(summary, args.metrics, args.section, required_launches=1,
                expected_devices=args.expected_devices.split(",") if args.expected_devices else None,
                require_nvtx=args.require_nvtx)
            metadata["counter_validation"] = validation
            metadata["missing_or_unavailable_requested_metrics"] = validation["missing_or_unavailable"]
            if not validation["valid"]:
                metadata["counter_status"] = "incomplete_attribution" if validation["invalid_nvtx_launches"] else "incomplete_metrics"
                raise ValueError("; ".join(validation["errors"]))
            metadata["counter_status"] = "collected"
            metadata["counter_sample_count"] = summary["sample_count"]
            metadata["unavailable_metric_values"] = summary["unavailable_metric_values"]
        except (ValueError, RuntimeError, csv.Error) as error:
            metadata["status"] = "failed"
            if metadata.get("counter_status") not in ("incomplete_metrics", "incomplete_attribution"):
                metadata["counter_status"] = "export_failed"
            metadata["error"] = str(error)
            exit_code = 2
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Profile {metadata['status']}: {args.output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
