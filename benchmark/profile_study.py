#!/usr/bin/env python3
"""Select and capture context-study profiles, independently from serving timing.

Selection is read-only by default. --execute runs the selected stage sequentially;
never run it alongside another GPU workload. Resume requires completed, matching
capture metadata. Unknown matrix shapes/phases remain explicitly unknown.
"""
from __future__ import annotations

import argparse
import collections
import csv
import functools
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import time

from profile_cuda import METRIC_GROUPS, combine_metric_groups, matrix_geometry_known

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX_CLASSES = {"quantized_matvec", "quantized_matmul", "other_matrix_multiply", "dequantization"}
OPERATION_FIELDS = ("device", "kernel", "grid", "block", "phase", "role", "type", "m", "n", "k", "fusion")
EXCLUDED_RESULT_DIRECTORIES = {"validation", "profiles", "archive", "archives", "archived"}


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def study_cell_paths(study_root: pathlib.Path) -> list[pathlib.Path]:
    """Follow only the shared study's explicit family aliases; deduplicate targets."""
    def excluded(name: str) -> bool:
        return name in EXCLUDED_RESULT_DIRECTORIES or name.startswith(("archive-", "cuda-study", "prefix-study"))

    study_root = study_root.resolve()
    if any(excluded(part) for part in study_root.parts):
        return []
    roots = {study_root}
    for family in ("1b", "8b", "70b"):
        alias = study_root / family
        if alias.is_symlink():
            target = alias.resolve()
            expected = study_root.parent / (study_root.name + "-" + family)
            if target == expected and target.is_dir():
                roots.add(target)
    paths = set()
    for root in sorted(roots):
        if any(excluded(part) for part in root.parts):
            continue
        for folder, directories, files in os.walk(root, followlinks=False):
            directories[:] = sorted(name for name in directories if not excluded(name))
            if "cell.json" in files:
                path = (pathlib.Path(folder) / "cell.json").resolve()
                if not any(excluded(part) for part in path.parts):
                    paths.add(path)
    return sorted(paths)


def select_cells(study_root: pathlib.Path) -> dict:
    cells = []
    for path in study_cell_paths(study_root):
        document = json.loads(path.read_text())
        if not {"model_family", "quantization", "concurrency", "prompt_tokens", "output_tokens", "status"} <= set(document):
            continue
        document = dict(document, cell_file=str(path.resolve()))
        original_config = document.get("config_file")
        colocated_config = path.parent / "study.env"
        if original_config or colocated_config.is_file():
            # A moved cell must execute and fingerprint its own configuration.
            # Missing colocated configuration fails before execution; an old
            # absolute path is provenance, never a fallback to a different run.
            if original_config:
                document["original_config_file"] = original_config
            document["config_file"] = str(colocated_config.resolve())
        cells.append(document)
    groups = collections.defaultdict(list)
    for cell in cells:
        groups[(cell["model_family"], cell["concurrency"])].append(cell)
    selected, skipped, joint = [], [], []
    for (family, concurrency), rows in sorted(groups.items()):
        by_format = collections.defaultdict(dict)
        for cell in rows:
            by_format[cell["quantization"]][cell["prompt_tokens"]] = cell
        passing = {quant: {p for p, row in lengths.items()
                           if row["status"] == "complete" and row["output_tokens"] == 512}
                   for quant, lengths in by_format.items()}
        runnable = {quant: lengths for quant, lengths in passing.items() if lengths}
        common = set.intersection(*runnable.values()) if runnable else set()
        maximum = max(common) if common else None
        joint.append({"model_family": family, "concurrency": concurrency,
                      "runnable_formats": sorted(runnable), "longest_jointly_feasible_prompt": maximum,
                      "unmeasured_or_infeasible_formats": sorted(set(by_format) - set(runnable))})
        endpoints = {2048, 16384} | ({maximum} if maximum is not None else set())
        for quant, lengths in sorted(by_format.items()):
            for prompt in sorted(endpoints):
                row = lengths.get(prompt)
                if row and prompt in passing[quant]:
                    selected.append(dict(row, endpoint_reasons=[reason for condition, reason in
                        [(prompt == 2048, "2k"), (prompt == 16384, "16k"), (prompt == maximum, "joint_maximum")]
                        if condition]))
                else:
                    skipped.append({"model_family": family, "quantization": quant, "concurrency": concurrency,
                                    "prompt_tokens": prompt, "status": row["status"] if row else "not_measured",
                                    "reason": "Only complete exact-512-output serving cells are profiled"})
    return {"schema_version": 1, "source": str(study_root.resolve()), "selected": selected,
            "skipped": skipped, "joint_maxima": joint,
            "selection_scope": "Joint maximum among formats with at least one complete 512-output cell; rerun selection after serving completes"}


def normalize_dimensions(value: str) -> tuple[int, ...]:
    return tuple(map(int, re.findall(r"\d+", str(value))))


def dominant_operations(rows: list[dict], coverage: float = .90) -> list[dict]:
    """Cover matrix/attention time per device and observed phase, not call count."""
    buckets = collections.defaultdict(list)
    for row in rows:
        family = "matrix" if row["class"] in MATRIX_CLASSES else "attention" if row["class"] == "attention" else None
        if family:
            buckets[(str(row["device"]), row.get("phase", "unknown"), family)].append(row)
    selected = []
    for (device, phase, family), values in sorted(buckets.items()):
        total = sum(float(row["total_ms"]) for row in values)
        covered = 0.0
        for row in sorted(values, key=lambda row: float(row["total_ms"]), reverse=True):
            selected.append(dict(row, selection_family=family, selection_device=device,
                                 selection_phase=phase, bucket_total_ms=total))
            covered += float(row["total_ms"])
            if covered >= coverage * total:
                break
    return selected


def operation_signature(operation: dict) -> dict:
    return {field: "x".join(map(str, normalize_dimensions(operation.get(field, "")))) if field in ("grid", "block")
            else str(operation.get(field, "")) for field in OPERATION_FIELDS}


def operation_id(operation: dict) -> str:
    return hashlib.sha256(json.dumps(operation_signature(operation), sort_keys=True).encode()).hexdigest()[:16]


def counter_matches(summary: dict, operations: list[dict], device: str, requested_launches: int = 5) -> dict:
    matches = []
    observed = collections.Counter()
    for launch in summary.get("launches", []):
        candidates = [row for row in operations if str(row["device"]) == str(launch["device"]) == device
                      and row["kernel"] == launch["kernel"]
                      and normalize_dimensions(row["grid"]) == normalize_dimensions(launch["grid"])
                      and normalize_dimensions(row["block"]) == normalize_dimensions(launch["block"])]
        operation = launch.get("operation", {})
        exact = [candidate for candidate in candidates
                 if all(str(candidate.get(field, "")) == str(operation.get(field, ""))
                        for field in ("phase", "role", "type", "m", "n", "k", "fusion"))]
        verified = bool(exact and operation.get("operation_match") == "nvtx_same_capture"
                        and operation.get("phase") in ("prefill", "decode")
                        and operation.get("role")
                        and (launch.get("class") == "attention" or matrix_geometry_known(operation)))
        if verified:
            observed.update({operation_id(candidate) for candidate in exact})
        matches.append({"launch_id": launch["id"], "kernel_launch_shape_matched": bool(candidates),
                        "matching_verified": verified, "operation_match": "nvtx_same_capture" if verified else "kernel_grid_block_only" if candidates else "unmatched",
                        "candidate_operations": candidates, "operation": operation,
                        "phase": operation.get("phase", "unknown") if verified else "unknown"})
    coverage = []
    for operation in operations:
        available = int(operation["calls"]) if str(operation.get("calls", "")).isdigit() else None
        required = min(requested_launches, available) if available is not None and available > 0 else requested_launches
        count = observed[operation_id(operation)]
        coverage.append({"operation_id": operation_id(operation), "operation": operation_signature(operation),
                         "trace_available_launches": available, "required_launches": required,
                         "observed_launches": count, "availability_limited": available is not None and 0 < available < requested_launches,
                         "complete": available is not None and available > 0 and count >= required})
    complete = bool(coverage) and all(row["complete"] for row in coverage)
    return {"matching_verified": complete, "coverage_complete": complete, "operation_coverage": coverage,
            "all_observed_launches_verified": bool(matches) and all(row["matching_verified"] for row in matches),
            "operation_match": "see_per_launch_verification",
            "launches": matches,
            "matched_launch_shape_count": sum(row["kernel_launch_shape_matched"] for row in matches),
            "verified_operation_count": sum(row["matching_verified"] for row in matches),
            "limitation": "Only NVTX metadata from the same counter capture proves matrix dimensions, tensor role and phase; incidental unselected shapes remain excluded"}


def operation_nvtx_filter(operation: dict) -> str:
    """Match the real batch phase and semantic tensor role/dimensions together."""
    role = re.escape(str(operation["role"])).replace(r"\.\*\.", r"\.[0-9]+\.")
    anonymous_attention = operation["role"] == "op.FLASH_ATTN_EXT"
    if anonymous_attention:
        role = r"node_[0-9]+"
    inner = "csb_op:role=" + role + ":type=" + re.escape(str(operation["type"]))
    if anonymous_attention:
        inner += ":op=FLASH_ATTN_EXT"
    if all(operation.get(field) for field in ("m", "n", "k")):
        inner += "".join(":" + field + "=" + re.escape(str(operation[field])) for field in ("m", "n", "k"))
    if operation.get("fusion"):
        inner += ":(?:[^:]+:)*fusion=" + re.escape(str(operation["fusion"]))
    inner += ":.*"
    return "regex:csb_batch:phase=" + str(operation["phase"]) + ":.*/*/" + inner


def cell_slug(cell: dict) -> pathlib.Path:
    return pathlib.Path(cell["model_family"]) / cell["quantization"] / f"c{cell['concurrency']}" / f"p{cell['prompt_tokens']}"


@functools.lru_cache(maxsize=32)
def file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostic_identity(server: pathlib.Path | None) -> dict | None:
    if server is None:
        return None
    paths = [server.resolve()] + sorted(server.resolve().parent.glob("*.so*"))
    paths += [server.resolve().parent / name for name in ("build.json", "build.diff", "server-phase.diff")]
    return {str(path): file_sha256(str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in paths if path.is_file()}


def capture_identity(cell: dict, extra: dict, script: pathlib.Path) -> str:
    dependencies = {pathlib.Path(__file__).resolve(), ROOT / "benchmark/profile_cuda.py",
                    ROOT / "benchmark/profile_gate.cpp", ROOT / "benchmark/load_test.py",
                    ROOT / "scripts/common.sh", script.resolve(), pathlib.Path(cell["config_file"]).resolve()}
    dependencies.add(ROOT / "config/experiment.env")
    fixture = pathlib.Path(cell["cell_file"]).parent / "prompts.json"
    if fixture.exists():
        dependencies.add(fixture.resolve())
    template = cell.get("configuration", {}).get("SERVER_CHAT_TEMPLATE_FILE")
    if template:
        dependencies.add(pathlib.Path(template).resolve())
    # Generated cell environments can source a caller-provided root config.
    # Follow literal source directives without executing configuration code.
    pending = list(dependencies)
    while pending:
        path = pending.pop()
        if path.suffix not in (".env", ".sh"):
            continue
        for line in path.read_text().splitlines():
            try:
                words = shlex.split(line, comments=True)
            except ValueError:
                continue
            if len(words) == 2 and words[0] in ("source", ".") and "$" not in words[1]:
                candidate = pathlib.Path(words[1])
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                candidate = candidate.resolve()
                if candidate.exists() and candidate not in dependencies:
                    dependencies.add(candidate)
                    pending.append(candidate)
    profiler_name = "NCU_BIN" if extra.get("PROFILE_TOOL") == "ncu" else "NSYS_BIN"
    profiler = os.environ.get(profiler_name) or shutil.which("ncu" if profiler_name == "NCU_BIN" else "nsys")
    if profiler:
        dependencies.add(pathlib.Path(profiler).resolve())
    # Endpoint reasons can change when additional formats finish the serving
    # sweep. They describe selection, not the workload or capture identity.
    source = {"cell": {key: value for key, value in cell.items() if key != "endpoint_reasons"}, "capture": extra,
              "source_sha256": {str(path): file_sha256(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                                for path in sorted(dependencies)},
              "profiler_environment": {key: value for key, value in os.environ.items()
                                       if key.startswith(("GGML_", "CSB_")) or key in ("LD_LIBRARY_PATH", "PROFILE_PORT", "NCU_BIN", "NSYS_BIN")}}
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def run_capture(cell: dict, destination: pathlib.Path, settings: dict, *, execute: bool,
                script: pathlib.Path, attempts: int, diagnostic_server: pathlib.Path | None,
                existing_only: bool = False, required_operations: list[dict] | None = None,
                requested_launches: int = 5) -> pathlib.Path | None:
    build_identity = diagnostic_identity(diagnostic_server)
    identity = capture_identity(cell, settings | {"diagnostic_build": build_identity,
                                "required_operations": required_operations, "requested_launches": requested_launches}, script)
    base_env = {key: value for key, value in os.environ.items() if not key.startswith("PROFILE_") or key == "PROFILE_PORT"} | {
                            "CONFIG_FILE": str(cell["config_file"]), "PROFILE_CONCURRENCY": str(cell["concurrency"]),
                            "EXACT_PROMPT_TOKENS": "1", "PROFILE_REPEAT_PROMPT": "0", "PROFILE_PREFIX_REUSE": "0"} | settings
    fixture = pathlib.Path(cell["cell_file"]).parent / "prompts.json"
    if fixture.exists():
        base_env["PROFILE_PROMPT_FIXTURE"] = str(fixture)
    if diagnostic_server:
        # The generated config sources ordinary settings first, then substitutes
        # this diagnostic executable. It never changes the serving cell config.
        base_env["PROFILE_SERVER_BIN"] = str(diagnostic_server.resolve())
        base_env["CSB_NVTX_OPS"] = "1"
        base_env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        base_env["LD_LIBRARY_PATH"] = str(diagnostic_server.resolve().parent) + ":" + base_env.get("LD_LIBRARY_PATH", "")

    def validated(actual: pathlib.Path, metadata: dict) -> bool:
        if metadata.get("status") != "captured" or (settings["PROFILE_TOOL"] == "ncu" and metadata.get("counter_status") != "collected"):
            return False
        needed = ("counter_summary.json", "ncu_raw.csv") if settings["PROFILE_TOOL"] == "ncu" else ("kernel_summary.csv", "profile_summary.json")
        if not all((actual / name).is_file() for name in needed):
            return False
        if required_operations is None:
            return True
        summary = json.loads((actual / "counter_summary.json").read_text())
        audit = counter_matches(summary, required_operations, str(settings["PROFILE_DEVICES"]), requested_launches)
        if execute:
            write_json(actual / "sampling-validation.json", audit)
            metadata["study_sampling_status"] = "complete" if audit["coverage_complete"] else "incomplete"
            write_json(actual / "metadata.json", metadata)
        return audit["coverage_complete"]

    existing_attempts = [int(match[1]) for path in destination.parent.glob(destination.name + "-attempt*")
                         if (match := re.fullmatch(re.escape(destination.name) + r"-attempt(\d+)", path.name))]
    for attempt in range(1, max([attempts, *existing_attempts]) + 1):
        actual = destination if attempt == 1 else destination.with_name(destination.name + f"-attempt{attempt}")
        metadata_path = actual / "metadata.json"
        marker = actual / "study-capture.json"
        if marker.exists():
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            saved = json.loads(marker.read_text())
            if saved["identity"] != identity:
                raise ValueError(f"Capture identity changed; choose a fresh profile output directory: {actual}")
            if saved.get("exit_code") == 0 and validated(actual, metadata):
                return actual
            continue
        if actual.exists():
            raise ValueError(f"Capture directory exists without finalized identity: {actual}")
        if existing_only or attempt > attempts:
            continue
        command = ["bash", str(script)]
        print(json.dumps({"command": command, "output": str(actual), "settings": settings,
                          "source_cell": cell["cell_file"], "execute": execute}), flush=True)
        if not execute:
            return None
        actual.parent.mkdir(parents=True, exist_ok=True)
        base_env["PROFILE_OUTPUT"] = str(actual)
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result = subprocess.run(command, env=base_env, cwd=ROOT)
        # profile_cuda creates the capture folder exclusively. Keep an attempt
        # record even if setup fails before that point.
        write_json(marker, {"identity": identity, "source_cell": cell["cell_file"],
                            "source_configuration": cell["configuration"], "settings": settings,
                            "command": command, "started_utc": started, "exit_code": result.returncode,
                            "diagnostic_build_sha256": build_identity,
                            "diagnostic_server": str(diagnostic_server) if diagnostic_server else None})
        if result.returncode == 0 and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if validated(actual, metadata):
                return actual
    if existing_only and not execute:
        return None
    raise RuntimeError(f"No complete capture after {attempts} attempt(s): {destination}")


def ensure_operation_coverage(cell: dict, primary: pathlib.Path, operations: list[dict],
                              settings: dict, capture_metadata: dict, *, script: pathlib.Path,
                              attempts: int, diagnostic_server: pathlib.Path, requested_launches: int) -> dict:
    """Supplement only starved operation signatures, preserving partial evidence."""
    device = str(settings["PROFILE_DEVICES"])
    primary_summary = json.loads((primary / "counter_summary.json").read_text())
    primary_audit = counter_matches(primary_summary, operations, device, requested_launches)
    write_json(primary / "selection.json", dict(capture_metadata, operations=operations, **primary_audit))
    coverage = {"schema_version": 1, "status": "incomplete", "operation_coverage": [],
                "source_capture_paths": "Relative to this coverage.json directory",
                "sampling_rule": "Each selected operation requires min(requested launches, positive observed trace availability) in one individual capture",
                **capture_metadata}
    by_id = {operation_id(operation): operation for operation in operations}
    for entry in primary_audit["operation_coverage"]:
        coverage["operation_coverage"].append(dict(entry, source_capture="." if entry["complete"] else None))
    write_json(primary / "coverage.json", coverage)
    for index, entry in enumerate(coverage["operation_coverage"]):
        if entry["complete"]:
            continue
        if entry["trace_available_launches"] is None or entry["trace_available_launches"] < 1:
            raise RuntimeError("Missing positive trace availability for selected operation " + entry["operation_id"])
        operation = by_id[entry["operation_id"]]
        targeted_settings = settings | {
            "PROFILE_KERNEL_REGEX": "^" + re.escape(operation["kernel"]) + "$",
            "PROFILE_NVTX_INCLUDE": operation_nvtx_filter(operation),
            "PROFILE_LAUNCH_COUNT": str(entry["required_launches"]),
        }
        target = run_capture(cell, primary / "targeted" / entry["operation_id"], targeted_settings,
                             execute=True, script=script, attempts=attempts, diagnostic_server=diagnostic_server,
                             required_operations=[operation], requested_launches=requested_launches)
        targeted_summary = json.loads((target / "counter_summary.json").read_text())
        targeted_audit = counter_matches(targeted_summary, [operation], device, requested_launches)
        write_json(target / "selection.json", dict(capture_metadata, operations=[operation],
                   logical_capture=str(primary), **targeted_audit))
        resolved = targeted_audit["operation_coverage"][0]
        coverage["operation_coverage"][index] = dict(resolved, source_capture=str(target.relative_to(primary)))
        write_json(primary / "coverage.json", coverage)
        if not resolved["complete"]:
            raise RuntimeError("Targeted operation sampling remains incomplete: " + str(target))
    coverage["status"] = "complete" if all(row["complete"] for row in coverage["operation_coverage"]) else "incomplete"
    write_json(primary / "coverage.json", coverage)
    if coverage["status"] != "complete":
        raise RuntimeError("Selected-operation sampling remains incomplete: " + str(primary))
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", "--study-dir", dest="study_root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--stage", choices=["select", "trace", "counters", "all"], default="select")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--metric-groups", nargs="+", choices=sorted(METRIC_GROUPS), default=list(METRIC_GROUPS))
    parser.add_argument("--split-metric-groups", action="store_true",
                        help="Separate metric groups only when a combined capture cannot collect them")
    parser.add_argument("--captures", type=int, default=3)
    parser.add_argument("--launches", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=.90)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--concurrencies", nargs="+", type=int)
    parser.add_argument("--diagnostic-server", type=pathlib.Path)
    parser.add_argument("--script", type=pathlib.Path, default=ROOT / "scripts/07_profile_cuda.sh")
    args = parser.parse_args()
    if min(args.captures, args.launches, args.attempts) < 1 or not 0 < args.coverage <= 1:
        parser.error("Positive captures/launches/attempts and coverage in (0,1] are required")
    if args.execute and args.stage in ("all", "counters") and not args.diagnostic_server:
        parser.error("Phase-specific counters require --diagnostic-server with csb_batch/csb_op NVTX labels")
    if args.execute and args.stage != "select" and args.diagnostic_server:
        pending = args.diagnostic_server.resolve().parent.parent / "pending-instrumentation.json"
        if pending.exists():
            parser.error(f"Diagnostic instrumentation has uncompiled changes: {pending}; rebuild it before profiling")
    selection = select_cells(args.study_root.resolve())
    output = (args.output or args.study_root / "profiles").resolve()
    if args.execute:
        write_json(output / "profile-selection.json", selection)
    else:
        print(json.dumps(selection, indent=2))
    if args.stage == "select":
        return 0
    cells = [cell for cell in selection["selected"]
             if (not args.models or cell["model_family"] in args.models)
             and (not args.formats or cell["quantization"] in args.formats)
             and (not args.concurrencies or cell["concurrency"] in args.concurrencies)]
    for cell in cells:
        profile_cell = output / cell_slug(cell)
        visibility = str(cell["configuration"].get("GPU_DEVICE", "0,1"))
        devices = [str(index) for index in range(len(visibility.split(",")))]
        trace = profile_cell / "trace"
        if args.stage in ("trace", "all", "counters"):
            captured = run_capture(cell, trace, {"PROFILE_TOOL": "nsys", "PROFILE_EXPECTED_DEVICES": ",".join(devices)},
                                  execute=args.execute, script=args.script, attempts=args.attempts,
                                  diagnostic_server=args.diagnostic_server, existing_only=args.stage == "counters")
            trace = captured or trace
        if args.stage not in ("counters", "all"):
            continue
        if not (trace / "kernel_summary.csv").exists():
            if args.execute:
                raise RuntimeError(f"Collect and validate the trace before counters: {trace}")
            print(json.dumps({"pending_trace": str(trace), "counter_selection": "requires measured dominant kernel signatures"}))
            continue
        with (trace / "kernel_summary.csv").open() as handle:
            operations = dominant_operations(list(csv.DictReader(handle)), args.coverage)
        if args.execute:
            write_json(profile_cell / "operations.json", {"coverage_target": args.coverage, "operations": operations,
                "selection_basis": "Descending summed matrix-path time including dequantization, and attention time, per observed phase and device",
                "phase_limit": "Unknown unless explicitly labeled; generic kernel class does not establish phase"})
        metric_batches = [(name, [name]) for name in args.metric_groups] if args.split_metric_groups else [("combined", args.metric_groups)]
        for phase in ("prefill", "decode"):
            for device in devices:
                selected_ops = [row for row in operations if row["selection_device"] == device and row["selection_phase"] == phase]
                if not selected_ops:
                    raise RuntimeError(f"No phase-attributed dominant {phase} kernels on expected GPU {device}: {trace}")
                names = sorted({row["kernel"] for row in selected_ops})
                regex = "^(?:" + "|".join(re.escape(name) for name in names) + ")$"
                for group, components in metric_batches:
                    combined = combine_metric_groups(components)
                    for capture in range(1, args.captures + 1):
                        destination = profile_cell / "counters" / "dominant" / phase / f"gpu{device}" / group / f"capture{capture}"
                        settings = {"PROFILE_TOOL": "ncu", "PROFILE_METRICS": combined["metrics"], "PROFILE_SECTIONS": ",".join(combined["sections"]),
                                    "PROFILE_DEVICES": device, "PROFILE_EXPECTED_DEVICES": device,
                                    "PROFILE_KERNEL_REGEX": regex, "PROFILE_FILTER_MODE": "per-launch-config",
                                    "PROFILE_NVTX_INCLUDE": "regex:csb_batch:phase=" + phase + ":.*/",
                                    "PROFILE_REQUIRE_NVTX": "1",
                                    "PROFILE_LAUNCH_COUNT": str(args.launches), "PROFILE_LAUNCH_SKIP": "0",
                                    "PROFILE_CACHE_CONTROL": "none", "PROFILE_CLOCK_CONTROL": "none"}
                        result = run_capture(cell, destination, settings, execute=args.execute, script=args.script,
                                             attempts=args.attempts, diagnostic_server=args.diagnostic_server)
                        if result and args.execute:
                            ensure_operation_coverage(cell, result, selected_ops, settings, {
                                "capture": capture, "phase": phase, "metric_group": group, "metric_groups": components,
                                "requested_launches_per_config": args.launches}, script=args.script, attempts=args.attempts,
                                diagnostic_server=args.diagnostic_server, requested_launches=args.launches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
