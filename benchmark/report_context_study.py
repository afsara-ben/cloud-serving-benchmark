#!/usr/bin/env python3
"""Report the approved context study without changing any measurement artifacts.

Only complete, independently revalidated cells enter runtime tables. The fixed
approved scope is audited even when a runner manifest requested a smaller subset.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

from capacity import GIB, slot_capacity
from context_protocol import producer_manifest
from profile_study import operation_id
from summarize import percentile

FORMATS = {family: ("FP16", "Q8_0", "Q4_K_M", "Q2_K", "IQ1_M") for family in ("1b", "8b", "70b")}
CONCURRENCIES = (8, 16, 32, 64)
PROMPTS = (2048, 4096, 8192, 16384)
EXTENSION = (32768, 65536)
REPETITIONS = 1
CAPTURES = 1
OUTPUT_TOKENS = 512
RUNTIME_PLOTS = (
    ("throughput", "output_tokens_per_second_mean", "Generated tokens / second"),
    ("ttft", "ttft_p95_ms", "TTFT p95 (ms)"),
    ("tpot", "tpot_p95_ms", "TPOT p95 (ms)"),
    ("memory", "gpu0_peak_vram_gib", "Peak sampled GPU memory (GiB)"),
)
PIN = "3f5e94d7c2ab2267fe39852051777fe30c1f49ef"
CAPACITY_STATUSES = {"capacity_estimated", "observed_oom", "observed_headroom_limit",
                     "native_context_unsupported"}
RESOLVED_STATUSES = CAPACITY_STATUSES | {"complete"}
EXACT_TRACE_MATCHES = {"verified", "external_id", "nvtx_launch_correlation", "nvtx_same_capture"}
COUNTER_GROUPS = ("memory", "instructions", "occupancy", "stalls", "tensor")
REQUIRED_COUNTERS = {
    "memory": ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
               "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct"),
    "instructions": ("sm__inst_executed.sum", "sm__inst_executed.avg.per_cycle_active"),
    "occupancy": ("sm__warps_active.avg.pct_of_peak_sustained_active", "launch__registers_per_thread",
                  "launch__shared_mem_per_block", "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
                  "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"),
    "stalls": ("smsp__warps_eligible.avg.per_cycle_active", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
    "tensor": ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",),
}


def shared_layout(root):
    declared = any((root / family).is_symlink() for family in FORMATS)
    named = root.name == "cuda-context-study"
    return (root.parent.name == "results" and not root.name.startswith(("cuda-study", "prefix-study"))
            and re.fullmatch(r"[A-Za-z0-9_.-]+", root.name) is not None and (named or declared))


def model_roots(study_root):
    """Declare only the three model roots; do not discover arbitrary siblings."""
    root = study_root.resolve()
    roots = {}
    for family in FORMATS:
        alias = root / family
        if shared_layout(root):
            target = root.parent / f"{root.name}-{family}"
            if target.is_symlink() or ((alias.exists() or alias.is_symlink()) and alias.resolve() != target):
                raise ValueError(f"Model alias {alias} must point to canonical directory {target}")
        else:
            target = alias.resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Model directory {alias} leaves the context study")
        roots[family] = target
    return roots


def evidence_allowed(study_root, path):
    source = path.resolve()
    return any(source.is_relative_to(root) for root in (study_root.resolve(), *model_roots(study_root).values()))


def evidence_path(study_root, path):
    """Display canonical evidence relative to the overview, including siblings."""
    return os.path.relpath(path.resolve(), study_root.resolve())


def read_evidence_json(study_root, path):
    return read_json(path) if evidence_allowed(study_root, path) else {}


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validation_errors(raw, concurrency, prompt, requests, repetition=None):
    """Check records as well as summaries, so a stale 'complete' flag cannot pass."""
    errors = []
    config, summary, records = raw.get("configuration", {}), raw.get("summary", {}), raw.get("requests", [])
    expected = {"concurrency": concurrency, "requests": requests, "target_prompt_tokens": prompt,
                "max_output_tokens": OUTPUT_TOKENS, "exact_prompt_tokens": True,
                "prompt_cache_requested": False, "prefix_reuse": False, "repeat_prompt": False,
                "ignore_eos": True, "warmup_requests": 0}
    if raw.get("schema_version", 0) < 3:
        errors.append("Exact-token schema version 3 or newer is required")
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"configuration.{key}: expected {value!r}, received {config.get(key)!r}")
    if repetition is not None and raw.get("repetition") != repetition:
        errors.append("Repetition identity differs from its directory")
    if not isinstance(records, list) or len(records) != requests:
        errors.append("Request count is missing or incomplete")
        records = records if isinstance(records, list) else []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"request {index}: invalid record")
            continue
        counts = {"prepared_prompt_tokens": prompt, "prompt_tokens": prompt,
                  "evaluated_prompt_tokens": prompt, "completion_tokens": OUTPUT_TOKENS,
                  "cached_prompt_tokens": 0}
        if record.get("ok") is not True or record.get("validation_errors"):
            errors.append(f"request {index}: failed validation")
        if any(record.get(key) != value for key, value in counts.items()):
            errors.append(f"request {index}: exact input/output/cache counts do not match")
        if record.get("fixed_output_length_valid") is not True:
            errors.append(f"request {index}: output length was not validated")
        for metric in ("duration_seconds", "ttft_seconds", "tpot_seconds"):
            if not finite(record.get(metric)) or record[metric] < 0:
                errors.append(f"request {index}: invalid {metric}")
        if (finite(record.get("duration_seconds")) and finite(record.get("ttft_seconds"))
                and record["ttft_seconds"] > record["duration_seconds"]):
            errors.append(f"request {index}: TTFT exceeds total duration")
    for key, value in {"successful_requests": requests, "failed_requests": 0,
                       "output_length_invalid_requests": 0, "cached_prompt_tokens": 0,
                       "cache_counter_available_requests": requests,
                       "max_client_inflight": concurrency}.items():
        if summary.get(key) != value:
            errors.append(f"summary.{key}: expected {value}")
    wall = summary.get("wall_seconds")
    if not finite(wall) or wall <= 0:
        errors.append("Positive finite serving wall time is required")
    else:
        rate = summary.get("output_tokens_per_second")
        expected_rate = requests * OUTPUT_TOKENS / wall
        if not finite(rate) or not math.isclose(rate, expected_rate, rel_tol=1e-6):
            errors.append("Output throughput disagrees with exact generated tokens / wall time")
    return errors


def placement_errors(placement, concurrency, prompt, gpu_count=2):
    errors = []
    if placement.get("valid") is not True or placement.get("errors"):
        errors.append("Full GPU placement validation is absent or failed")
    if placement.get("slots") != concurrency or placement.get("slot_capacity") != slot_capacity(prompt):
        errors.append("Slot count or padded per-slot context does not match")
    if placement.get("headroom_valid") is not True:
        errors.append("Startup 2 GiB per-GPU headroom validation is absent or failed")
    kv = placement.get("kv_buffers_mib", {})
    expected_kv = {f"CUDA{index}" for index in range(gpu_count)}
    try:
        # CUDA log parsers preserve numeric strings; indices are visible ordinals.
        if not all(math.isfinite(float(kv.get(key, 0))) and float(kv.get(key, 0)) > 0 for key in expected_kv):
            errors.append("Every selected GPU requires positive KV allocation evidence")
        if {key for key in kv if key.startswith("CUDA")} != expected_kv:
            errors.append("KV allocation devices differ from the selected GPUs")
    except (TypeError, ValueError):
        errors.append("Invalid GPU KV allocation evidence")
    try:
        if float(kv.get("CPU", 0)) > 0:
            errors.append("CPU KV fallback detected")
    except (TypeError, ValueError):
        errors.append("Invalid CPU KV allocation evidence")
    offload = placement.get("offload", [])
    if len(offload) != 2 or not isinstance(offload[0], int) or offload[0] <= 0 or offload[0] != offload[1]:
        errors.append("Full transformer/output layer offload is not established")
    return errors


def resource_errors(resources, device_indices=None):
    errors = []
    swap, memory = resources.get("swap", {}), resources.get("memory", {})
    if swap.get("valid") is not True or swap.get("max_server_swap_bytes") != 0 or swap.get("sample_errors"):
        errors.append("Zero server swap was not validated")
    gpus = memory.get("gpus", [])
    observed = {str(g.get("index")) for g in gpus}
    expected_count = 2 if device_indices is None else len(device_indices)
    if (memory.get("valid") is not True or len(gpus) != expected_count or len(observed) != expected_count
            or (device_indices is not None and observed != set(device_indices))):
        errors.append("Selected-GPU runtime memory validation is absent or failed")
    for gpu in gpus:
        if (gpu.get("samples_available") is not True or not finite(gpu.get("peak_used_bytes"))
                or gpu["peak_used_bytes"] <= 0 or not finite(gpu.get("minimum_headroom_bytes"))
                or gpu["minimum_headroom_bytes"] < 2 * GIB):
            errors.append(f"GPU {gpu.get('index')}: runtime 2 GiB headroom was not established")
    return errors


def telemetry_samples(path, raw):
    """Keep telemetry inside measured HTTP time; warmup samples never enter plots."""
    summary = raw.get("summary", {})
    start, end = summary.get("started_unix_seconds"), summary.get("finished_unix_seconds")
    rows = []
    if not finite(start) or not finite(end):
        return rows
    try:
        with path.open() as stream:
            for original in csv.DictReader(stream, skipinitialspace=True):
                row = {key.strip(): value.strip() for key, value in original.items() if key and value is not None}
                try:
                    timestamp = dt.datetime.strptime(row.get("timestamp", ""), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
                except ValueError:
                    continue
                if start <= timestamp <= end:
                    rows.append(row)
    except OSError:
        pass
    return rows


def numeric_column(rows, prefixes):
    values = []
    for row in rows:
        key = next((key for key in row if any(key.startswith(prefix) for prefix in prefixes)), None)
        if key:
            match = re.match(r"[-+]?\d+(?:\.\d+)?", row[key])
            if match:
                values.append(float(match[0]))
    return values


def thermal_observations(rows):
    """An unavailable flag is unknown, including when the other flag is valid."""
    flags = {}
    for kind in ("sw", "hw"):
        values = []
        for row in rows:
            raw = next((value for key, value in row.items() if key.endswith(f".{kind}_thermal_slowdown")), "")
            normalized = re.sub(r"\s+", "", str(raw)).lower()
            values.append({"active": True, "notactive": False}.get(normalized))
        flags[kind] = any(values) if values and all(value is not None for value in values) else None
    return flags, any(flags.values()) if all(value is not None for value in flags.values()) else None


def telemetry_metrics(runs, device_indices=("0", "1")):
    devices = collections.defaultdict(list)
    for run in runs:
        for row in telemetry_samples(run["path"] / "telemetry.csv", run["raw"]):
            devices[row.get("index", "unknown")].append(row)
    result = {}
    for index in device_indices:
        rows = devices[index]
        result[f"gpu{index}_telemetry_samples"] = len(rows)
        for label, prefixes, reducer in (
            ("peak_vram_gib", ("memory.used",), lambda x: max(x) / 1024),
            ("power_mean_w", ("power.draw",), statistics.fmean),
            ("temperature_max_c", ("temperature.gpu",), max),
            ("sm_clock_min_mhz", ("clocks.current.sm", "clocks.sm"), min),
            ("sm_clock_max_mhz", ("clocks.current.sm", "clocks.sm"), max),
            ("memory_clock_min_mhz", ("clocks.current.memory", "clocks.mem"), min),
        ):
            values = numeric_column(rows, prefixes)
            result[f"gpu{index}_{label}"] = reducer(values) if values else None
        flags, thermal = thermal_observations(rows)
        result[f"gpu{index}_sw_thermal_limit_observed"] = flags["sw"]
        result[f"gpu{index}_hw_thermal_limit_observed"] = flags["hw"]
        result[f"gpu{index}_thermal_limit_observed"] = thermal
    return result


def server_concurrency(path):
    """Count task lifetimes and decode tokens explicitly logged by the server.

    The logged `n_batch (effective)` is a ceiling, not the actual batch width.
    Decode-token additions can coexist with prompt tokens, so they are a separate
    sequence-count observation and never substituted for matrix activation width.
    """
    active, maximum, starts, stops, inconsistent = {}, 0, 0, 0, False
    added, histogram = set(), collections.Counter()
    try:
        with path.open(errors="replace") as stream:
            for line in stream:
                match = re.search(r"\bslot\s+\S+:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*(.*)", line)
                if match:
                    slot, task, event = int(match[1]), int(match[2]), match[3]
                    if "processing task," in event:
                        inconsistent |= slot in active
                        active[slot] = task
                        starts += 1
                        maximum = max(maximum, len(active))
                    elif "stop processing:" in event:
                        inconsistent |= active.get(slot) != task
                        active.pop(slot, None)
                        stops += 1
                    elif "slot decode token," in event:
                        added.add(slot)
                batch = re.search(r"n_batch \(effective\) = \d+, off = (\d+)", line)
                if batch and int(batch[1]) == 0:
                    if added:
                        histogram[len(added)] += 1
                    added.clear()
    except OSError:
        pass
    return {"max_active_slots": maximum if starts else None, "started_tasks": starts, "completed_tasks": stops,
            "task_events_complete": starts > 0 and starts == stops and not active and not inconsistent,
            "decode_sequences_added_per_batch": dict(sorted(histogram.items())),
            "matrix_activation_width": None,
            "matrix_width_status": "Requires same-operation phase/matrix metadata; n_batch is only a ceiling"}


def load_cell(root, family, quant, concurrency, prompt, manifest, progress, device_indices=None):
    key = f"{quant}/c{concurrency}/p{prompt}"
    folder = model_roots(root)[family] / key
    document = read_evidence_json(root, folder / "cell.json")
    experiment = "serving_grid" if prompt in PROMPTS else "capacity_extension"
    requested = concurrency * (2 if experiment == "serving_grid" else 1)
    row = {"model": family, "format": quant, "concurrency": concurrency,
           "input_tokens": prompt, "output_tokens": OUTPUT_TOKENS, "experiment": experiment,
           "status": document.get("status", "missing"), "validated_repetitions": 0,
           "required_repetitions": REPETITIONS, "requests_per_repetition": requested,
           "reason": document.get("reason", "No cell result"),
           "evidence": evidence_path(root, folder), "validation_errors": ""}
    if not document:
        return row, []
    errors = []
    for field, value in {"quantization": quant, "concurrency": concurrency, "prompt_tokens": prompt,
                         "output_tokens": OUTPUT_TOKENS, "experiment": experiment,
                         "requests_per_repetition": requested}.items():
        if document.get(field) != value:
            errors.append(f"cell.{field}: expected {value!r}")
    controls = document.get("configuration", {})
    expected_controls = {"LLAMA_CPP_REF": PIN, "SERVER_PARALLEL": str(concurrency),
        "SERVER_CTX_SIZE": str(concurrency * slot_capacity(prompt)), "SERVER_BATCH_SIZE": "2048",
        "SERVER_UBATCH_SIZE": "512", "MAX_OUTPUT_TOKENS": "512", "FLASH_ATTN": "on",
        "SERVER_SPLIT_MODE": "layer", "SERVER_TENSOR_SPLIT": "1,1" if device_indices is None else ",".join("1" for _ in device_indices), "SERVER_CACHE_TYPE_K": "f16",
        "SERVER_CACHE_TYPE_V": "f16", "SERVER_NO_CONTEXT_SHIFT": "1", "SERVER_FIT": "off"}
    for field, value in expected_controls.items():
        if str(controls.get(field)) != value:
            errors.append(f"cell.configuration.{field}: expected {value!r}")
    fingerprint = progress.get("fingerprint")
    history = read_evidence_json(root, model_roots(root)[family] / "protocol-history.json")
    producer = producer_manifest(manifest, history, document.get("fingerprint")) if manifest else None
    if not fingerprint or producer is None:
        errors.append("Cell producer is not the current or a proven compatible earlier manifest")
    recorded_repetitions = document.get("repetitions")
    if not isinstance(recorded_repetitions, int) or recorded_repetitions < REPETITIONS:
        errors.append("Cell does not declare the required measured run")
    if producer and producer.get("repetitions", recorded_repetitions) != recorded_repetitions:
        errors.append("Cell repetition budget differs from its original producer manifest")
    if manifest:
        expected_fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if fingerprint != expected_fingerprint:
            errors.append("Manifest fingerprint differs from progress")
    else:
        errors.append("Missing study manifest")
    status = row["status"]
    if progress.get("cells", {}).get(key, {}).get("status") != status:
        errors.append("Cell and progress statuses disagree")
    if status in CAPACITY_STATUSES:
        screen = read_evidence_json(root, folder / "capacity-estimate.json")
        if status in ("capacity_estimated", "native_context_unsupported") and screen.get("status") != status:
            errors.append("Predicted exclusion is missing its matching capacity calculation")
        if status.startswith("observed_") and not any(evidence_allowed(root, folder / name) and (folder / name).exists() for name in ("server.log", "placement.json")):
            errors.append("Observed capacity failure has no startup/placement evidence")
        row["status"] = "invalid" if errors else status
        row["validation_errors"] = "; ".join(errors)
        return row, []
    runs = []
    for repetition in range(1, REPETITIONS + 1):
        rep_path = folder / f"r{repetition}"
        raw = read_evidence_json(root, rep_path / "raw.json")
        if not raw:
            errors.append(f"r{repetition}: missing or invalid raw JSON")
            continue
        resources = read_evidence_json(root, rep_path / "resources.json")
        try:
            rep_errors = validation_errors(raw, concurrency, prompt, requested, repetition)
            rep_errors += placement_errors(read_evidence_json(root, rep_path / "placement.json"), concurrency, prompt,
                                           2 if device_indices is None else len(device_indices))
            rep_errors += resource_errors(resources, device_indices)
            if any(not evidence_allowed(root, rep_path / name) for name in ("telemetry.csv", "server.log")):
                rep_errors.append("Runtime evidence leaves the declared study roots")
        except (ValueError, TypeError, AttributeError, KeyError):
            rep_errors = ["Malformed raw, placement or resource document"]
        if progress.get("cells", {}).get(f"{key}/r{repetition}", {}).get("status") != "complete":
            rep_errors.append("Runner repetition status is not complete")
        study = raw.get("study_configuration", {})
        if study.get("fingerprint") != document.get("fingerprint") or study.get("status") != "complete":
            rep_errors.append("Raw result resume identity or validation status differs")
        if rep_errors:
            errors.extend(f"r{repetition}: {error}" for error in rep_errors)
        else:
            runs.append({"path": rep_path, "raw": raw, "resources": resources})
    row["validated_repetitions"] = len(runs)
    if status == "complete":
        errors += [f"warmup: {error}" for error in validation_errors(
            read_evidence_json(root, folder / "warmup-discarded.json"), concurrency, prompt, concurrency, 0)]
        errors += [f"warmup: {error}" for error in resource_errors(read_evidence_json(root, folder / "warmup-resources.json"), device_indices)]
        if errors or len(runs) != REPETITIONS:
            row["status"] = "invalid" if len(runs) == REPETITIONS else "incomplete"
    row["validation_errors"] = "; ".join(errors)
    return row, runs if row["status"] == "complete" and not errors else []


def aggregate(row, runs, device_indices=("0", "1")):
    records = [record for run in runs for record in run["raw"]["requests"]]
    rates = [run["raw"]["summary"]["output_tokens_per_second"] for run in runs]
    result = {key: row[key] for key in ("model", "format", "concurrency", "input_tokens", "output_tokens", "experiment")}
    result.update(repetitions=len(runs), successful_requests=len(records), failed_requests=0,
                  output_tokens_per_second_mean=statistics.fmean(rates),
                  output_tokens_per_second_stddev=statistics.stdev(rates) if len(rates) > 1 else None,
                  min_max_client_inflight=min(run["raw"]["summary"]["max_client_inflight"] for run in runs))
    for name, key in (("ttft", "ttft_seconds"), ("tpot", "tpot_seconds"), ("e2e", "duration_seconds")):
        values = [record[key] * 1000 for record in records]
        for label, quantile in (("p50", .5), ("p95", .95)):
            result[f"{name}_{label}_ms"] = percentile(values, quantile)
    result.update(telemetry_metrics(runs, device_indices))
    concurrency = [server_concurrency(run["path"] / "server.log") for run in runs]
    observed = [item["max_active_slots"] for item in concurrency if item["max_active_slots"] is not None]
    result["server_max_active_slots_min_across_runs"] = min(observed) if len(observed) == len(runs) else None
    result["server_task_events_complete"] = all(item["task_events_complete"] and item["started_tasks"] == row["requests_per_repetition"] for item in concurrency)
    histogram = collections.Counter()
    for item in concurrency:
        histogram.update(item["decode_sequences_added_per_batch"])
    result["decode_sequences_added_per_batch_histogram"] = json.dumps(dict(sorted(histogram.items()))) if histogram else None
    result["runtime_matrix_activation_width_status"] = "unmeasured: decode additions can share a batch with prompt tokens"
    result["evidence"] = row["evidence"]
    return result


def maximum_inputs(capacity):
    rows, joint = [], []
    lookup = {(row["model"], row["format"], row["concurrency"], row["input_tokens"]): row for row in capacity}
    for family, formats in FORMATS.items():
        for concurrency in CONCURRENCIES:
            passing, excluded, unresolved = {}, [], []
            for quant in formats:
                cells = [lookup[(family, quant, concurrency, prompt)] for prompt in PROMPTS + EXTENSION]
                passes = [row["input_tokens"] for row in cells if row["status"] == "complete"]
                remaining = [row["input_tokens"] for row in cells if row["status"] not in RESOLVED_STATUSES]
                capacity_excluded = all(row["status"] in CAPACITY_STATUSES for row in cells)
                weight_excluded = family == "70b" and quant == "FP16" and all(row["status"] == "capacity_estimated" for row in cells)
                if passes:
                    passing[quant] = set(passes)
                elif capacity_excluded:
                    excluded.append(quant)
                else:
                    unresolved.append(quant)
                rows.append({"model": family, "format": quant, "concurrency": concurrency,
                             "largest_validated_input_tokens": max(passes) if passes else None,
                             "search_status": "complete" if not remaining else "incomplete",
                             "capacity_excluded": capacity_excluded, "fp16_weight_capacity_excluded": weight_excluded,
                             "untested_or_invalid_input_tokens": ";".join(map(str, remaining)),
                             "native_131072_input_status": "unsupported_input_plus_512_exceeds_131072"})
            common = set.intersection(*passing.values()) if passing and not unresolved else set()
            joint.append({"model": family, "concurrency": concurrency,
                          "runnable_formats": ";".join(passing), "capacity_excluded_formats": ";".join(excluded),
                          "unresolved_formats": ";".join(unresolved),
                          "longest_jointly_validated_input_tokens": max(common) if common else None,
                          "search_status": "complete" if all(row["search_status"] == "complete" for row in rows if row["model"] == family and row["concurrency"] == concurrency) else "incomplete"})
    return rows, joint


def profile_context(path, study_root):
    relative = Path(evidence_path(study_root, path)).parts
    if "profiles" not in relative:
        return None
    parts = relative[relative.index("profiles") + 1:]
    if len(parts) < 5:
        return None
    family = parts[0].lower()
    if family not in FORMATS:
        match = re.search(r"(?:^|[-_])(1b|8b|70b)(?:[-_]|$)", family)
        if not match:
            return None
        family = match[1]
    try:
        return {"model": family, "format": parts[1], "concurrency": int(parts[2][1:]),
                "input_tokens": int(parts[3][1:])}
    except ValueError:
        return None


def profile_paths(study_root, filename):
    roots = [study_root / "profiles"] + [root / "profiles" for root in model_roots(study_root).values()]
    return sorted({path.resolve() for root in roots if evidence_allowed(study_root, root)
                   for path in root.rglob(filename) if evidence_allowed(study_root, path)})


def read_csv(path):
    try:
        with path.open() as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError):
        return []


def counter_shape(selection, launch):
    matched = next((row for row in selection.get("launches", []) if str(row.get("launch_id")) == str(launch.get("id"))), None)
    matched = matched if matched is not None else selection
    operation = matched.get("operation", matched)
    verified = matched.get("matching_verified") is True
    # Selection intent is not evidence that NCU actually captured that operation.
    if verified:
        for field in ("kernel", "grid", "block"):
            if operation.get(field) and str(operation[field]) != str(launch.get(field)):
                verified = False
    fields = ("phase", "role", "type", "m", "n", "k", "fusion")
    result = {field: operation.get(field, "") if verified else "" for field in fields}
    result["phase"] = result["phase"] or "unknown"
    result["operation_match"] = "verified" if verified else matched.get("operation_match", "unresolved")
    result["kernel_launch_shape_matched"] = matched.get("kernel_launch_shape_matched", verified)
    return result


def capture_bundle(path):
    return next((parent for parent in path.parents if re.fullmatch(r"capture\d+(?:-attempt\d+)?", parent.name)), path.parent)


def operation_coverage(study_root):
    """Validate one chosen capture per selected operation; never add partial runs."""
    rows, accepted = [], {}
    for path in profile_paths(study_root, "coverage.json"):
        context = profile_context(path, study_root)
        if context is None:
            continue
        document = read_json(path)
        for entry in document.get("operation_coverage", []):
            op = entry.get("operation", {})
            identifier = entry.get("operation_id")
            source_name = entry.get("source_capture")
            source = (path.parent / (source_name if isinstance(source_name, str) else "__missing__")).resolve()
            errors = []
            if not evidence_allowed(study_root, source):
                errors.append("Source capture leaves the declared study roots")
            selection = read_evidence_json(study_root, source / "selection.json")
            metadata = read_evidence_json(study_root, source / "metadata.json")
            counters = read_evidence_json(study_root, source / "counter_summary.json")
            declared = next((item for item in selection.get("operation_coverage", []) if item.get("operation_id") == identifier), {})
            verified = {str(item.get("launch_id")): item for item in selection.get("launches", []) if item.get("matching_verified") is True}
            observed = 0
            for launch in counters.get("launches", []):
                item = verified.get(str(launch.get("id")))
                if item:
                    signature = launch | item.get("operation", {})
                    observed += operation_id(signature) == identifier
            available, required = entry.get("trace_available_launches"), entry.get("required_launches")
            if (not isinstance(available, int) or available <= 0 or not isinstance(required, int)
                    or required != min(5, available)):
                errors.append("Positive trace availability and required sample count are missing or inconsistent")
            if identifier != operation_id(op):
                errors.append("Selected operation identity differs from its signature")
            if metadata.get("status") != "captured" or metadata.get("counter_status") != "collected":
                errors.append("Chosen counter capture is not validated complete")
            if (entry.get("complete") is not True or declared.get("complete") is not True
                    or declared.get("required_launches") != required or declared.get("trace_available_launches") != available
                    or declared.get("operation") != op):
                errors.append("Chosen source does not independently satisfy this operation's coverage")
            if not isinstance(required, int) or observed < required:
                errors.append("Too few independently verified launches in the chosen source")
            capture = selection.get("capture")
            if capture not in (1, 2, 3):
                errors.append("Logical independent capture number is missing or outside 1..3")
            groups = selection.get("metric_groups", []) or metadata.get("counter_collection", {}).get("metric_groups", [])
            row = context | op | {"operation_id": identifier, "capture": capture,
                "metric_groups_covered": ";".join(groups), "required_launches": required,
                "verified_launches": observed, "availability_limited": isinstance(available, int) and 0 < available < 5,
                "status": "complete" if not errors else "incomplete", "reason": "; ".join(errors),
                "source_capture": evidence_path(study_root, source),
                "coverage_source": evidence_path(study_root, path)}
            rows.append(row)
            if not errors:
                accepted[(str(source), identifier)] = row
    return accepted, rows


def profile_tables(study_root, coverage=None):
    """Retain exact launch/matrix shapes, device identity and native metric units."""
    kernels, counters, captures = [], [], []
    accepted, coverage_rows = coverage if coverage is not None else operation_coverage(study_root)
    chosen_operations = {(row["coverage_source"], row["operation_id"]) for row in coverage_rows}
    for path in profile_paths(study_root, "kernel_summary.csv"):
        context = profile_context(path, study_root)
        if context is None:
            continue
        metadata = read_evidence_json(study_root, path.parent / "metadata.json")
        if metadata.get("error") or metadata.get("status") in ("failed", "invalid"):
            continue
        for row in read_csv(path):
            try:
                calls, duration = int(row["calls"]), float(row["total_ms"])
                if calls <= 0 or not math.isfinite(duration) or duration < 0:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            shape = {key: row.get(key, "") for key in (
                "kernel", "class", "device", "grid", "block", "registers_per_thread",
                "shared_bytes_per_block", "local_bytes_per_thread", "phase", "role", "type",
                "m", "n", "k", "fusion", "operation_match")}
            shape["phase"] = shape["phase"] or "unknown"
            kernels.append(context | shape | {"calls": calls, "total_ms": duration,
                "mean_us": duration * 1000 / calls, "source": evidence_path(study_root, path)})
    for path in profile_paths(study_root, "counter_summary.json"):
        context = profile_context(path, study_root)
        if context is None:
            continue
        document = read_json(path)
        metadata = read_evidence_json(study_root, path.parent / "metadata.json")
        selection = read_evidence_json(study_root, path.parent / "selection.json")
        if not selection:
            selection = read_evidence_json(study_root, path.parent.parent.parent / "selection.json")
        capture_id = evidence_path(study_root, path.parent)
        bundle = capture_bundle(path)
        logical_capture = evidence_path(study_root, bundle)
        capture_number = re.fullmatch(r"capture(\d+)(?:-attempt\d+)?", bundle.name)
        included_in_protocol = capture_number is None or int(capture_number[1]) <= CAPTURES
        capture_valid = metadata.get("status") == "captured" and metadata.get("counter_status") == "collected"
        capture = context | {"source": capture_id, "counter_status": metadata.get("counter_status", "unknown"),
                             "matched_launches": 0, "launches": len(document.get("launches", []))}
        for launch in document.get("launches", []):
            shape = counter_shape(selection, launch)
            capture["matched_launches"] += shape["operation_match"] == "verified"
            base = context | shape | {key: launch.get(key, "") for key in ("kernel", "device", "grid", "block")}
            identifier = operation_id(base)
            proof = accepted.get((str(path.parent.resolve()), identifier))
            expected_key = (evidence_path(study_root, bundle / "coverage.json"), identifier)
            chosen = proof is not None or expected_key not in chosen_operations
            base["operation_id"] = identifier
            # Unmatched selections cannot be averaged across independent captures.
            base["selection_scope"] = evidence_path(study_root, bundle.parent) if shape["operation_match"] == "verified" else capture_id
            base["metric_group"] = selection.get("metric_group", bundle.parent.name)
            groups = metadata.get("counter_collection", {}).get("metric_groups", [])
            base["metric_groups_covered"] = ";".join(groups) if groups else base["metric_group"]
            base["capture_validated"] = capture_valid
            base["selected_for_summary"] = chosen and included_in_protocol
            base["operation_coverage_validated"] = proof is not None
            base["required_launches_per_capture"] = proof["required_launches"] if proof is not None else 5
            observations = [(name, "", metric) for name, metric in launch.get("metrics", {}).items()]
            observations += [(name, metric.get("instance", ""), metric)
                             for name, instances in launch.get("metric_instances", {}).items() for metric in instances]
            for name, instance, metric in observations:
                value = metric.get("value")
                available = metric.get("available") is True and finite(value) and not metric.get("ambiguous")
                counters.append(base | {"metric": name, "instance": instance, "unit": metric.get("unit", ""),
                                        "value": value if available else None, "available": available,
                                        "raw": metric.get("raw", ""), "capture": capture_id,
                                        "logical_capture": re.sub(r"-attempt\d+$", "", logical_capture),
                                        "launch_id": launch.get("id", "")})
        captures.append(capture)
    grouped = collections.defaultdict(list)
    excluded = {"value", "available", "raw", "capture", "logical_capture", "launch_id"}
    for row in counters:
        if not row["selected_for_summary"]:
            continue
        key = tuple((field, str(value)) for field, value in row.items() if field not in excluded)
        grouped[key].append(row)
    counter_summary = []
    for key, observations in sorted(grouped.items()):
        values = [row["value"] for row in observations if row["available"]]
        by_capture = collections.Counter(row["logical_capture"] for row in observations if row["available"])
        counter_summary.append(dict(key) | {"available_launches": len(values),
            "unavailable_launches": len(observations) - len(values), "independent_captures": len(by_capture),
            "minimum_launches_per_capture": min(by_capture.values()) if by_capture else 0,
            "mean": statistics.fmean(values) if values else None,
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "sampling_requirement_met": len(by_capture) >= CAPTURES and min(by_capture.values()) >= int(dict(key)["required_launches_per_capture"])
                                        and dict(key).get("capture_validated") == "True" and dict(key).get("operation_coverage_validated") == "True" if by_capture else False,
            "source_captures": ";".join(sorted({row["capture"] for row in observations}))})
    return kernels, counters, counter_summary, captures


def profile_requirements(capacity, joint, kernels, counter_summary, selected_operations=(), coverage_rows=()):
    lookup = {(row["model"], row["format"], row["concurrency"], row["input_tokens"]): row for row in capacity}
    joint_lookup = {(row["model"], row["concurrency"]): row for row in joint}
    rows = []
    for family, formats in FORMATS.items():
        for concurrency in CONCURRENCIES:
            common = joint_lookup[(family, concurrency)]["longest_jointly_validated_input_tokens"]
            for quant in formats:
                endpoints = set((2048, 16384))
                if common:
                    endpoints.add(common)
                for prompt in sorted(endpoints):
                    key = (family, quant, concurrency, prompt)
                    source = lookup[key]
                    row = dict(zip(("model", "format", "concurrency", "input_tokens"), key))
                    if source["status"] in CAPACITY_STATUSES:
                        rows.append(row | {"status": "capacity_excluded", "reason": source["status"]})
                        continue
                    traces = [k for k in kernels if tuple(k[field] for field in row) == key]
                    counters = [k for k in counter_summary if all(str(k[field]) == str(value) for field, value in row.items())]
                    missing = []
                    if source["status"] != "complete":
                        missing.append("validated source serving/capacity cell")
                    selected = [operation for operation in selected_operations
                                if all(str(operation[field]) == str(value) for field, value in row.items())]
                    if not selected:
                        missing.append("selected dominant operations artifact")
                    for operation in selected:
                        identifier = operation["operation_id"]
                        for group in COUNTER_GROUPS:
                            covered = {item["capture"] for item in coverage_rows if item["status"] == "complete"
                                       and item["operation_id"] == identifier and group in item["metric_groups_covered"].split(";")
                                       and all(str(item[field]) == str(value) for field, value in row.items())}
                            if not set(range(1, CAPTURES + 1)).issubset(covered):
                                missing.append(f"Operation {identifier}/{group}: selected-operation coverage missing in capture 1")
                            samples = [item for item in counters if item["operation_id"] == identifier
                                       and item["sampling_requirement_met"] and group in item["metric_groups_covered"].split(";")]
                            metrics = {item["metric"] for item in samples}
                            absent = [name for name in REQUIRED_COUNTERS[group] if name not in metrics]
                            if group == "instructions" and not any(item["instance"] or "sass" in item["metric"] for item in samples):
                                absent.append("instruction_mix")
                            if group == "stalls" and not any("stall" in item["metric"] for item in samples):
                                absent.append("scheduler_stall_reasons")
                            if absent:
                                missing.append(f"Operation {identifier}/{group}: unavailable required counters {','.join(absent)}")
                    for device in ("0", "1"):
                        for phase in ("prefill", "decode"):
                            if not any(str(k["device"]) == device and k["phase"] == phase and k["operation_match"] in EXACT_TRACE_MATCHES for k in traces):
                                missing.append(f"GPU{device} {phase} trace operation metadata")
                            for group in COUNTER_GROUPS:
                                selected = [k for k in counters if str(k["device"]) == device and k["phase"] == phase
                                            and group in k["metric_groups_covered"].split(";")
                                            and k["operation_match"] == "verified" and k["sampling_requirement_met"]]
                                available = {k["metric"] for k in selected}
                                absent = [name for name in REQUIRED_COUNTERS[group] if name not in available]
                                if group == "instructions" and not any(k["instance"] or "sass" in k["metric"] for k in selected):
                                    absent.append("instruction_mix")
                                if group == "stalls" and not any("stall" in k["metric"] for k in selected):
                                    absent.append("scheduler_stall_reasons")
                                if absent:
                                    missing.append(f"GPU{device} {phase} {group}: missing one capture with five matched samples for {','.join(absent)}")
                    rows.append(row | {"status": "incomplete" if missing else "complete", "reason": "; ".join(missing)})
                if not common and quant not in joint_lookup[(family, concurrency)]["capacity_excluded_formats"].split(";"):
                    rows.append({"model": family, "format": quant, "concurrency": concurrency, "input_tokens": "joint_longest",
                                 "status": "incomplete", "reason": "Longest jointly feasible input has not been established"})
    return rows


def diagnostic_tables(study_root):
    root = study_root / "diagnostics"
    rows = []
    for path in sorted(root.rglob("run.json")) if evidence_allowed(study_root, root) else []:
        if not evidence_allowed(study_root, path):
            continue
        identity = read_json(path).get("identity", {})
        operation = read_evidence_json(study_root, path.parent / "operation.json")
        complete = read_evidence_json(study_root, path.parent / "complete.json")
        build = identity.get("build", {})
        passed = complete.get("correctness", {}).get("passed") is True and operation.get("correctness", {}).get("passed") is True
        provenance = (build.get("commit") == PIN and all(build.get(key) for key in ("binary_sha256", "cuda_library_sha256", "build_diff_sha256")))
        row = {key: identity.get(key) for key in ("variant", "shape", "type", "m", "n", "k", "phase", "repetition", "tool", "metric_group", "gpu")}
        row["metric_groups_covered"] = ";".join(identity.get("metric_groups", []) or [str(identity.get("metric_group", ""))])
        row.update(status="complete" if passed and provenance else "incomplete", correctness_passed=passed,
                   provenance_present=bool(provenance), median_operation_us=complete.get("median_operation_us"),
                   reference_relative_l2=operation.get("correctness", {}).get("relative_l2"),
                   source=evidence_path(study_root, path.parent))
        if row["tool"] == "ncu":
            row["counter_validation_passed"] = complete.get("counter_validation", {}).get("valid") is True
            launches = read_evidence_json(study_root, path.parent / "counter_summary.json").get("launches", [])
            row["counter_launches"] = len(launches)
            counts = collections.Counter((launch.get("kernel"), launch.get("grid"), launch.get("block"), launch.get("device")) for launch in launches)
            row["minimum_counter_launches_per_shape"] = min(counts.values()) if counts else 0
            row["counter_metric_names"] = ";".join(sorted({name for launch in launches for name, metric in launch.get("metrics", {}).items()
                                                           if metric.get("available") is True and finite(metric.get("value"))}))
            if not row["counter_validation_passed"] or not launches or row["minimum_counter_launches_per_shape"] < 5:
                row["status"] = "incomplete"
        rows.append(row)
    try:
        if not evidence_allowed(study_root, root / "plan.json"):
            raise ValueError("Diagnostic plan leaves the declared study roots")
        plan = json.loads((root / "plan.json").read_text())
        cases = plan if isinstance(plan, list) else plan.get("cases", [])
    except (OSError, ValueError, AttributeError):
        cases = []
    if not cases:
        shapes = {"ffn_gate_up": (28672, 8192), "ffn_down": (8192, 28672),
                  "attn_q_output": (8192, 8192), "attn_k_v": (1024, 8192)}
        cases = [{"variant": "baseline", "shape": shape, "type": kind, "m": m, "n": width, "k": k}
                 for shape, (m, k) in shapes.items() for kind in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ1_M", "FP16")
                 for width in (1, 8, 9, 16, 32, 64)]
        cases += [{"variant": "early-mmq", "shape": shape, "type": kind, "m": m, "n": 8, "k": k}
                  for shape, (m, k) in shapes.items() for kind in ("Q2_K", "Q3_K")]
    requirements = []
    for case in cases:
        fields = ("variant", "shape", "type", "m", "n", "k") + (("gpu",) if "gpu" in case else ())
        matching = [row for row in rows if row["status"] == "complete" and all(str(row[field]) == str(case.get(field)) for field in fields)]
        absent = []
        for group in ("timing", "nsys") + COUNTER_GROUPS:
            found = {row["repetition"] for row in matching if group in row["metric_groups_covered"].split(";")}
            minimum = 1
            if len(found) < minimum:
                absent.append(f"{group}: {len(found)}/{minimum} independent captures")
            if group in COUNTER_GROUPS:
                for metric in REQUIRED_COUNTERS[group]:
                    found = {row["repetition"] for row in matching if group in row["metric_groups_covered"].split(";") and metric in row.get("counter_metric_names", "").split(";")}
                    if len(found) < CAPTURES:
                        absent.append(f"{group}/{metric}: missing counter repetitions")
        requirements.append({key: case.get(key) for key in fields} | {"status": "incomplete" if absent else "complete",
                            "observed_gpus": ";".join(sorted({str(row["gpu"]) for row in matching})), "reason": "; ".join(absent)})
    # A supplied plan can select dominant shapes, but cannot omit requested widths/interventions.
    for kind in ("Q2_K", "Q4_K", "IQ1_M"):
        widths = {case.get("n") for case in cases if case.get("variant") == "baseline" and case.get("type") == kind}
        if not set((1, 8, 9, 16, 32, 64)).issubset(widths):
            requirements.append({"type": kind, "status": "incomplete", "reason": "Plan omits approved activation widths"})
    for kind in ("Q2_K", "Q3_K"):
        if not any(case.get("variant") == "early-mmq" and case.get("type") == kind and case.get("n") == 8 for case in cases):
            requirements.append({"type": kind, "status": "incomplete", "reason": "Plan omits width-eight earlier-MMQ intervention"})
    return rows, requirements


def interpretation_review(study_root):
    study_root = study_root.resolve()
    marker = study_root / "report/interpretation-reviewed.json"
    review = read_json(marker)
    errors = []
    if review.get("status") != "reviewed" or not review.get("reviewer") or not review.get("note"):
        errors.append("Root interpretation review is missing")
    try:
        reviewed = dt.datetime.fromisoformat(review.get("reviewed_utc", "").replace("Z", "+00:00"))
        if reviewed.tzinfo is None:
            raise ValueError("timezone missing")
        reviewed_at = reviewed.timestamp()
    except (ValueError, TypeError):
        reviewed_at = None
        errors.append("Review timestamp is missing or invalid")
    paths = review.get("source_paths", [])
    if not isinstance(paths, list) or not paths:
        errors.append("Reviewed source paths are missing")
        paths = []
    for name in paths:
        source = (study_root / str(name)).resolve()
        if not evidence_allowed(study_root, source) or not source.is_file():
            errors.append(f"Reviewed source unavailable: {name}")
        elif reviewed_at is not None and source.stat().st_mtime > reviewed_at:
            errors.append(f"Reviewed source changed after review: {name}")
    return {"id": "evidence_backed_interpretation", "status": "requires_review" if errors else "complete",
            "missing_or_invalid": errors, "review_marker": str(marker.relative_to(study_root))}


def existing_runtime_plots(runtime, output):
    """Keep saved plot links during status refreshes without creating images."""
    families = {row["model"] for row in runtime}
    artifacts = []
    for family in FORMATS:
        if family not in families:
            continue
        for name, _, _ in RUNTIME_PLOTS:
            for suffix in ("png", "svg"):
                path = output / f"{family}-{name}.{suffix}"
                if path.is_file():
                    artifacts.append(str(path.relative_to(output.parent)))
    return artifacts


def plot_runtime(runtime, output):
    if not runtime:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    artifacts = []
    colors = {"FP16": "#555555", "Q8_0": "#0072B2", "Q4_K_M": "#009E73", "IQ1_M": "#D55E00", "Q2_K": "#CC79A7"}
    for family in FORMATS:
        rows = [row for row in runtime if row["model"] == family]
        if not rows:
            continue
        for name, metric, ylabel in RUNTIME_PLOTS:
            figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
            axes.flat[0].set_xscale("log", base=2)
            axes.flat[0].set_xlim(PROMPTS[0] / 2 ** .15, PROMPTS[-1] * 2 ** .15)
            for ax, concurrency in zip(axes.flat, CONCURRENCIES):
                for quant in FORMATS[family]:
                    points = sorted((row for row in rows if row["concurrency"] == concurrency and row["format"] == quant), key=lambda row: row["input_tokens"])
                    points = [row for row in points if finite(row.get(metric))]
                    if not points:
                        continue
                    lookup = {row["input_tokens"]: row for row in points}
                    x = PROMPTS
                    y = [lookup[prompt][metric] if prompt in lookup else math.nan for prompt in PROMPTS]
                    if name == "throughput" and any(finite(point.get("output_tokens_per_second_stddev")) for point in points):
                        ax.errorbar(x, y, yerr=[lookup[prompt]["output_tokens_per_second_stddev"] if prompt in lookup else math.nan for prompt in PROMPTS],
                                    marker="o", capsize=3, label=quant, color=colors[quant])
                    else:
                        ax.plot(x, y, marker="o", label=quant + (" GPU0" if name == "memory" else ""), color=colors[quant])
                    if name == "memory":
                        other = [lookup[prompt].get("gpu1_peak_vram_gib", math.nan) if prompt in lookup else math.nan for prompt in PROMPTS]
                        ax.plot(x, other, marker="x", linestyle="--", label=quant + " GPU1", color=colors[quant])
                ax.set_title(f"{concurrency} clients")
                ax.set_xticks(PROMPTS, labels=("2k", "4k", "8k", "16k"))
                ax.grid(alpha=.25)
                ax.set_xlabel("Input tokens (k = 1024)")
                ax.set_ylabel(ylabel)
                if ax.lines:
                    ax.set_ylim(bottom=0)
                    ax.legend(fontsize=7)
                else:
                    ax.set_yticks([])
                    ax.text(.5, .5, "No validated cells", ha="center", va="center", transform=ax.transAxes, color="#555555")
            figure.suptitle(f"{family.upper()} | two GPUs | FP16 KV | 512 output tokens | reuse off\nOnly complete validated serving cells; gaps are unmeasured")
            for suffix in ("png", "svg"):
                path = output / f"{family}-{name}.{suffix}"
                figure.savefig(path, dpi=180)
                artifacts.append(str(path.relative_to(output.parent)))
            plt.close(figure)
    return artifacts


def markdown_table(rows, columns):
    if not rows:
        return "No complete validated measurements available.\n"
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append("unmeasured" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_model_summaries(study_root, roots, manifests, runtime, extension, capacity, audit, columns):
    """Write compact derived summaries beside each manifest, with original evidence links."""
    for family, root in roots.items():
        pending = not manifests.get(family)
        if pending and not (shared_layout(study_root) and root.is_dir()):
            continue
        for name in ("summary.csv", "summary.md"):
            if (root / name).is_symlink():
                raise ValueError(f"Model summary must not be a symbolic link: {root / name}")
        rows = [row | {"evidence": evidence_path(root, study_root / row["evidence"])}
                for row in runtime if row["model"] == family]
        write_csv(root / "summary.csv", rows, None if rows else [key for key, _ in columns] + ["evidence"])
        linked_rows = [row | {"evidence": " · ".join(
            f"[r{rep}]({row['evidence']}/r{rep}/raw.json)" for rep in range(1, REPETITIONS + 1))} for row in rows]
        cells = [row for row in capacity if row["model"] == family]
        extension_count = sum(row["model"] == family for row in extension)
        excluded = sum(row["status"] in CAPACITY_STATUSES for row in cells)
        shared = evidence_path(root, study_root / "report")
        provenance = "Manifest pending" if pending else "[Manifest](manifest.json)"
        provenance += " · [Progress](progress.json)" if (root / "progress.json").is_file() else " · Progress pending"
        text = [f"# {family.upper()} context study", "",
                "Status: **pending**. The runner manifest is not available; no validated measurements are reported." if pending else
                f"{len(rows)} validated serving cells; {extension_count} validated capacity-extension cells; "
                f"{excluded} recorded capacity exclusions. Full-study audit: **{audit['status']}**.", "",
                "Serving measurements use the first validated run only, exact 512 output tokens and no prompt reuse. "
                "Only complete validated serving cells enter this table and summary.csv. Capacity extensions remain separate.", "",
                f"[Full report]({shared}/index.md) · [Summary CSV](summary.csv) · {provenance}", "",
                f"[Capacity extension measurements]({shared}/capacity-extension.md) · [All capacity statuses]({shared}/capacity.csv) · "
                f"[Completion audit]({shared}/completion-audit.json)", "",
                markdown_table(linked_rows, columns + [("evidence", "Raw repetitions")])]
        (root / "summary.md").write_text("\n".join(text))


def build_report(study_root, plots=True):
    study_root = study_root.resolve()
    if not study_root.is_dir():
        raise ValueError(f"Study root does not exist: {study_root}")
    # Prevent accidental writes into prior cuda-study/prefix-study result trees.
    if "context-study" not in study_root.parts and not shared_layout(study_root):
        raise ValueError("Study root must be inside a context-study directory; prior studies are read-only")
    roots = model_roots(study_root)
    output = study_root / "report"
    if output.is_symlink():
        raise ValueError("Report output must not be a symbolic link")
    output.mkdir(exist_ok=True)
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Report files must not be symbolic links to external artifacts")
    capacity, runtime, extension, manifests, accepted_runs = [], [], [], {}, {}
    for family, formats in FORMATS.items():
        manifest = read_evidence_json(study_root, roots[family] / "manifest.json")
        progress = read_evidence_json(study_root, roots[family] / "progress.json")
        manifests[family] = manifest
        for quant in formats:
            for concurrency in CONCURRENCIES:
                for prompt in PROMPTS + EXTENSION:
                    row, runs = load_cell(study_root, family, quant, concurrency, prompt, manifest, progress)
                    capacity.append(row)
                    if runs:
                        measured = aggregate(row, runs)
                        (runtime if prompt in PROMPTS else extension).append(measured)
                        accepted_runs[(family, quant, concurrency, prompt)] = runs
    maximum, joint = maximum_inputs(capacity)
    coverage = operation_coverage(study_root)
    kernels, counter_launches, counters, captures = profile_tables(study_root, coverage)
    selected_operations = []
    for path in profile_paths(study_root, "operations.json"):
        context = profile_context(path, study_root)
        if context:
            selected_operations.extend(context | operation | {"operation_id": operation_id(operation)}
                                       for operation in read_json(path).get("operations", []))
    profiles = profile_requirements(capacity, joint, kernels, counters, selected_operations, coverage[1])
    diagnostics, diagnostic_coverage = diagnostic_tables(study_root)
    artifacts = (plot_runtime(runtime, output / "plots") if plots
                 else existing_runtime_plots(runtime, output / "plots"))
    tables = {"runtime.csv": runtime, "capacity-extension.csv": extension, "capacity.csv": capacity,
              "max-inputs.csv": maximum, "joint-inputs.csv": joint, "profile-kernels.csv": kernels,
              "profile-counter-launches.csv": counter_launches, "profile-counters.csv": counters,
              "profile-captures.csv": captures, "profile-requirements.csv": profiles,
              "profile-operation-coverage.csv": coverage[1],
              "diagnostic-operations.csv": diagnostics, "diagnostic-requirements.csv": diagnostic_coverage}
    fallback_fields = {"runtime.csv": ["model", "format", "concurrency", "input_tokens", "output_tokens", "output_tokens_per_second_mean"],
                       "capacity-extension.csv": ["model", "format", "concurrency", "input_tokens", "output_tokens", "output_tokens_per_second_mean"],
                       "profile-kernels.csv": ["model", "format", "kernel", "phase", "device", "grid", "block", "m", "n", "k", "calls", "total_ms"],
                       "profile-counters.csv": ["model", "format", "kernel", "phase", "device", "grid", "block", "m", "n", "k", "metric", "unit", "mean"],
                       "profile-counter-launches.csv": ["model", "format", "kernel", "device", "metric", "unit", "value", "available"],
                       "profile-captures.csv": ["model", "format", "concurrency", "input_tokens", "source", "counter_status", "matched_launches"],
                       "diagnostic-operations.csv": ["variant", "shape", "type", "m", "n", "k", "phase", "repetition", "tool", "metric_group", "gpu", "status", "source"]}
    fallback_fields["profile-operation-coverage.csv"] = ["model", "format", "concurrency", "input_tokens", "operation_id", "capture", "status", "source_capture"]
    for filename, rows in tables.items():
        write_csv(output / filename, rows, fallback_fields.get(filename) if not rows else None)
    matching = []
    for family in FORMATS:
        for concurrency in CONCURRENCIES:
            for prompt in PROMPTS + EXTENSION:
                available = [(quant, accepted_runs[(family, quant, concurrency, prompt)]) for quant in FORMATS[family]
                             if (family, quant, concurrency, prompt) in accepted_runs]
                if len(available) < 2:
                    continue
                signatures = [tuple(tuple(record.get("input_sha256") for record in run["raw"]["requests"]) for run in runs)
                              for _, runs in available]
                complete_hashes = all(all(all(value for value in rep) for rep in signatures_for_format) for signatures_for_format in signatures)
                matching.append({"model": family, "concurrency": concurrency, "input_tokens": prompt,
                                 "formats": [quant for quant, _ in available],
                                 "status": "complete" if complete_hashes and all(sig == signatures[0] for sig in signatures) else "mismatch_or_missing_hashes"})
    requirements = []
    for experiment in ("serving_grid", "capacity_extension"):
        cells = [row for row in capacity if row["experiment"] == experiment]
        missing = [row["evidence"] for row in cells if row["status"] not in RESOLVED_STATUSES]
        requirements.append({"id": experiment, "status": "complete" if not missing else "incomplete",
            "required_cells": len(cells), "measured_cells": sum(row["status"] == "complete" for row in cells),
            "capacity_excluded_cells": sum(row["status"] in CAPACITY_STATUSES for row in cells),
            "missing_or_invalid": missing})
    manifest_issues = []
    for family, manifest in manifests.items():
        if not manifest:
            manifest_issues.append(f"{family}: missing manifest")
            continue
        if manifest.get("llama_cpp_commit") != PIN or len(manifest.get("devices", [])) != 2:
            manifest_issues.append(f"{family}: pinned backend/two-device identity not established")
        if not manifest.get("binary_sha256") or not manifest.get("code_sha256"):
            manifest_issues.append(f"{family}: binary/code provenance missing")
        for model in manifest.get("models", []):
            if family == "70b" and model.get("quant") == "FP16" and model.get("available") is False:
                continue
            if not model.get("sha256"):
                manifest_issues.append(f"{family}/{model.get('quant')}: model hash missing")
    requirements.append({"id": "pinned_configuration_and_provenance", "status": "incomplete" if manifest_issues else "complete", "missing_or_invalid": manifest_issues})
    missing_profiles = [row for row in profiles if row["status"] == "incomplete"]
    requirements.append({"id": "targeted_profiling", "status": "incomplete" if missing_profiles else "complete",
                         "required_endpoints": len(profiles), "missing_or_invalid": missing_profiles,
                         "counter_failures": [evidence_path(study_root, path) for path in profile_paths(study_root, "metadata.json")
                                              if read_json(path).get("counter_status") in ("permission_denied", "failed", "unavailable", "profiling_resource_busy")]})
    telemetry_gaps = [row["evidence"] for row in runtime + extension if any(row.get(f"gpu{gpu}_telemetry_samples", 0) == 0
                      or row.get(f"gpu{gpu}_thermal_limit_observed") is None for gpu in (0, 1))]
    requirements.append({"id": "runtime_telemetry_including_thermal_state", "status": "incomplete" if telemetry_gaps or not runtime else "complete",
                         "missing_or_invalid": telemetry_gaps, "scope": "Device-total SMI samples restricted to measured HTTP windows"})
    active_gaps = [row["evidence"] for row in runtime + extension if not row["server_task_events_complete"]
                   or row["server_max_active_slots_min_across_runs"] != row["concurrency"]]
    requirements.append({"id": "actual_server_concurrency", "status": "incomplete" if active_gaps or not runtime else "complete",
                         "missing_or_invalid": active_gaps,
                         "interpretation": "Task start/release events establish overlapping active slots. Matrix widths require the independent profiling requirement."})
    requirements.append({"id": "matched_inputs_across_formats", "status": "complete" if matching and all(row["status"] == "complete" for row in matching)
                         and all(row["status"] == "complete" for row in requirements[:2]) else "incomplete", "comparisons": matching})
    requirements.append({"id": "q2_q4_controlled_kernel_intervention",
                         "status": "complete" if diagnostic_coverage and all(row["status"] == "complete" for row in diagnostic_coverage) else "incomplete",
                         "required_cases": len(diagnostic_coverage),
                         "missing_or_invalid": [row for row in diagnostic_coverage if row["status"] != "complete"],
                         "scope": "Private-build isolated complete operations; observed GPU is explicit and timing is never serving throughput"})
    requirements.append(interpretation_review(study_root))
    audit = {"schema_version": 1, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "study_root": str(study_root), "status": "complete" if all(row["status"] == "complete" for row in requirements) else "incomplete",
             "scope": {"formats": FORMATS, "concurrencies": CONCURRENCIES, "serving_inputs": PROMPTS,
                       "capacity_extension_inputs": EXTENSION, "output_tokens": OUTPUT_TOKENS, "repetitions": REPETITIONS},
             "cell_status_counts": dict(collections.Counter(row["status"] for row in capacity)),
             "requirements": requirements, "plots": artifacts,
             "measurement_policy": "Only complete cells with a valid first run enter runtime CSV/plots. Earlier extra repetitions are excluded. Exclusions are not measured throughput.",
             "limitations": ["Largest validated power of two is a tested result for fixed layout, not an interpolated capacity boundary.",
                             "Profile kernel sums and replay counters are diagnostic; never substitute for HTTP timing.",
                             "Missing metric values stay unavailable. No averages mix devices, units, phases, matrix shapes or launch resources."]}
    write_json(output / "completion-audit.json", audit)
    columns = [("model", "Model"), ("format", "Format"), ("concurrency", "Clients"), ("input_tokens", "Input tokens"),
               ("output_tokens_per_second_mean", "Output tok/s"),
               ("ttft_p95_ms", "TTFT p95 ms"), ("tpot_p95_ms", "TPOT p95 ms"), ("e2e_p95_ms", "E2E p95 ms")]
    (output / "runtime.md").write_text("# Validated serving measurements\n\nOne measured run of two requests/client; exact 512 output tokens; reuse off. "
        "Throughput uses r1 only; run-to-run variability is unmeasured. Latency percentiles describe requests within that run. Earlier extra repetitions remain as provenance and are excluded.\n\n" + markdown_table(runtime, columns))
    (output / "capacity-extension.md").write_text("# Validated capacity extension bursts\n\nOne measured burst of one request/client. "
        "Burst size differs from the serving grid; these measurements are reported separately.\n\n" + markdown_table(extension, columns))
    (output / "capacity.md").write_text("# Capacity for the fixed two-GPU layout\n\nPredicted exclusions and observed failures remain separate. "
        "Missing cells are not evidence of infeasibility. 70B FP16 is explicitly excluded only when a recorded weight lower bound establishes the capacity limit.\n\n"
        + markdown_table(maximum, [("model", "Model"), ("format", "Format"), ("concurrency", "Clients"),
                                  ("largest_validated_input_tokens", "Largest validated power-of-two input"),
                                  ("search_status", "Search"), ("capacity_excluded", "Weight capacity excluded")])
        + "\n" + markdown_table(joint, [("model", "Model"), ("concurrency", "Clients"), ("runnable_formats", "Runnable formats"),
                                         ("capacity_excluded_formats", "Excluded formats"), ("unresolved_formats", "Unresolved formats"),
                                         ("longest_jointly_validated_input_tokens", "Longest jointly validated input")]))
    format_progress = []
    for family, formats in FORMATS.items():
        for quant in formats:
            cells = [row for row in capacity if row["model"] == family and row["format"] == quant]
            main = [row for row in cells if row["experiment"] == "serving_grid"]
            extra = [row for row in cells if row["experiment"] == "capacity_extension"]
            unresolved = sum(row["status"] not in RESOLVED_STATUSES for row in cells)
            status = ("not started" if all(row["status"] == "missing" for row in cells) else
                      "capacity excluded" if all(row["status"] in CAPACITY_STATUSES for row in cells) else
                      "incomplete" if unresolved else "complete")
            format_progress.append({"model": family, "format": quant, "status": status,
                "main_measured": f"{sum(row['status'] == 'complete' for row in main)}/{len(main)}",
                "extension_measured": f"{sum(row['status'] == 'complete' for row in extra)}/{len(extra)}",
                "capacity_exclusions": sum(row["status"] in CAPACITY_STATUSES for row in cells),
                "unresolved": unresolved})
    profiling_status = next(row["status"] for row in requirements if row["id"] == "targeted_profiling")
    lines = ["# Context study results", "", f"Audit status: **{audit['status']}**. {len(runtime)} validated serving cells and {len(extension)} validated capacity-extension cells.", "",
             "Inference completion by model and format is shown below. Measured counts exclude capacity rejections; unresolved cells still need work.", "",
             markdown_table(format_progress, [("model", "Model"), ("format", "Format"), ("status", "Inference status"),
                 ("main_measured", "Main measured"), ("extension_measured", "Extensions measured"),
                 ("capacity_exclusions", "Capacity exclusions"), ("unresolved", "Unresolved")]), "",
             f"Targeted hardware profiling: **{profiling_status}**. Its completion is tracked separately from inference.", "",
             "Each accepted cell has one valid first run, exact token counts, matching current or proven compatible producer identity, full GPU placement, zero sampled server swap, and at least 2 GiB sampled headroom per GPU. "
             "Configured slots and client overlap do not alone prove decode matrix widths; profiling coverage remains explicit in the audit.", "",
             "- [Serving measurements](runtime.md) ([CSV](runtime.csv))",
             "- [Capacity extension bursts](capacity-extension.md) ([CSV](capacity-extension.csv))",
             "- [Capacity and largest validated inputs](capacity.md) ([all cell statuses](capacity.csv))",
             "- [Completion audit](completion-audit.json)",
             "- [Exact trace kernel shapes](profile-kernels.csv), [counter samples](profile-counter-launches.csv), [matched counter summary](profile-counters.csv)",
             "- [Required profiling coverage](profile-requirements.csv)",
             "- [Selected-operation capture coverage](profile-operation-coverage.csv)",
             "- [Isolated diagnostic operations](diagnostic-operations.csv) and [required coverage](diagnostic-requirements.csv)", "",
             "Kernel durations sum captured launches. Counter statistics retain exact shape, phase, device and native unit; unknown phase remains unknown. "
             "Unavailable counter values do not become zero. Hardware replay timing is separate from serving throughput. No causal conclusion is generated here.", ""]
    for artifact in artifacts:
        if artifact.endswith(".png"):
            lines += [f"![{Path(artifact).stem}]({artifact})", ""]
    lines += ["Regenerate without executing GPU work:", "", "```bash", f"python3 benchmark/report_context_study.py --study-root {study_root}", "```", ""]
    (output / "index.md").write_text("\n".join(lines))
    write_model_summaries(study_root, roots, manifests, runtime, extension, capacity, audit, columns)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True,
                        help="results/cuda-context-study overview, or legacy results/context-study/RUN_TAG")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation but retain existing plot embeds, for incremental status refreshes")
    args = parser.parse_args()
    try:
        audit = build_report(args.study_root, plots=not args.no_plots)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.study_root / 'report/index.md'}; approved study status: {audit['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
