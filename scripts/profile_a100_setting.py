#!/usr/bin/env python3
"""Capture one saved CUDA serving setting, one phase and one GPU at a time.

Use --hardware rtxpro6000 for the matched two-GPU 70B scope including 32K.

The default action is plan (no inference). Counters collect the requested Nsight
sections together in one capture with a GLOBAL launch cap. No endpoint sweep,
automatic trace, supplemental capture or retry is scheduled.

Build the annotated server once on the A100 host:
  python3 benchmark/matrix_diagnostic.py build --server --cuda-arch 80 --build-root .run/a100-profile-build

Example (a completed serving cell):
  python3 scripts/profile_a100_setting.py counters --cell results/cuda-context-study-a100-2gpu-8b/Q4_K_M/c8/p2048 --phase decode --device 0 --launch-count 5 --roofline tensor
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from profile_cuda import a100_section_collection, ROOFLINE_SECTIONS, server_build_provenance
from profile_study import run_capture
from report_context_study import PIN
from report_serving import CONCURRENCIES, PROMPTS
from run_a100_serving import check_devices
from cuda_hardware import HARDWARE, architecture, matches

# ncu only accepts the Blackwell spelling on RTX PRO 6000; profile_cuda records
# the results back under the canonical Ampere names.
BLACKWELL_METRIC_NAMES = {
    "dram__bytes_read.sum": "dram__bytes_op_read.sum",
    "dram__bytes_write.sum": "dram__bytes_op_write.sum",
}


def device_metrics(metrics, hardware):
    if hardware != "rtxpro6000":
        return metrics
    return ",".join(BLACKWELL_METRIC_NAMES.get(name, name) for name in metrics.split(","))


SCALAR_ROOFLINE_METRICS = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "sass__thread_inst_executed_true_per_opcode",
)


def load_cell(directory, hardware="a100"):
    directory = directory.resolve()
    if directory.name == "cell.json":
        directory = directory.parent
    path = directory / "cell.json"
    cell = json.loads(path.read_text())
    if cell.get("status") != "complete" or not (directory / "r1/raw.json").is_file():
        raise ValueError("Select a completed serving cell with r1/raw.json; no inference is started to fill missing results")
    configuration = cell.get("configuration", {})
    devices = str(configuration.get("GPU_DEVICE", "")).split(",")
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices) or any(not gpu.isdecimal() for gpu in devices):
        raise ValueError("Saved setting must select one or two CUDA GPUs")
    prompts = (*PROMPTS, 32768) if hardware == "rtxpro6000" else PROMPTS
    if (cell.get("concurrency") not in CONCURRENCIES or cell.get("prompt_tokens") not in prompts
            or cell.get("output_tokens") != 512):
        raise ValueError(f"Expected a {hardware} serving setting with supported clients/inputs and 512 output tokens")
    if configuration.get("LLAMA_CPP_REF") != PIN:
        raise ValueError("Serving cell does not use the pinned llama.cpp revision")
    for filename in ("study.env", "prompts.json"):
        if not (directory / filename).is_file():
            raise ValueError(f"Missing saved serving evidence: {directory / filename}")
    cell.update(cell_file=str(path), config_file=str(directory / "study.env"))
    if hardware == "rtxpro6000":
        validate_rtx_cell(cell, devices)
    return cell, devices


def validate_rtx_cell(cell, devices):
    """Revalidate the source measurement and hardware without launching a GPU job."""
    import report_context_study as context
    model_root = Path(cell["cell_file"]).parents[3]
    if not model_root.name.endswith("-70b") or len(devices) != 2:
        raise ValueError("Expected a two-GPU 70B shared study layout")
    root = model_root.parent / model_root.name[:-4]
    manifest = context.read_json(model_root / "manifest.json")
    progress = context.read_json(model_root / "progress.json")
    gpu_rows = manifest.get("devices", [])
    if (len(gpu_rows) != 2 or [str(d.get("index")) for d in gpu_rows] != devices
            or any(not matches(d.get("name", ""), "rtxpro6000") for d in gpu_rows)
            or manifest.get("extension_request_waves") != 2):
        raise ValueError("Source manifest must describe the matched RTX PRO 6000 protocol")
    row, runs = context.load_cell(root, "70b", cell["quantization"], cell["concurrency"], cell["prompt_tokens"],
                                  manifest, progress, device_indices=devices)
    if not runs or row["status"] != "complete":
        raise ValueError("Serving source failed independent validation: " + row["validation_errors"])
    return gpu_rows


def capture_settings(args, gpu_count):
    if not 0 <= args.device < gpu_count:
        raise ValueError(f"--device is a logical index in the saved visibility list: 0 through {gpu_count - 1}")
    if args.launch_count < 1 or args.launch_skip < 0:
        raise ValueError("Use a positive --launch-count and a nonnegative --launch-skip")
    tool = "nsys" if args.action == "trace" else "ncu"
    if tool == "nsys":
        return {"PROFILE_TOOL": tool, "PROFILE_EXPECTED_DEVICES": ",".join(map(str, range(gpu_count)))}
    if getattr(args, "scalar_roofline", False):
        if args.roofline:
            raise ValueError("--scalar-roofline uses InstructionStats; omit --roofline")
        collection = {"metrics": ",".join(SCALAR_ROOFLINE_METRICS), "sections": ["InstructionStats"]}
    else:
        collection = a100_section_collection(args.roofline)
        if getattr(args, "hardware", "a100") == "rtxpro6000":
            collection["metrics"] += (",l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,"
                                      "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")
    collection["metrics"] = device_metrics(collection["metrics"], getattr(args, "hardware", "a100"))
    nvtx = f"regex:csb_batch:phase={args.phase}:.*/"
    if getattr(args, "operation_regex", None):
        nvtx = f"regex:csb_batch:phase={args.phase}:.*/*/{args.operation_regex}"
    return {"PROFILE_TOOL": tool, "PROFILE_METRICS": collection["metrics"],
            "PROFILE_SECTIONS": ",".join(collection["sections"]),
            "PROFILE_DEVICES": str(args.device), "PROFILE_EXPECTED_DEVICES": str(args.device),
            "PROFILE_KERNEL_REGEX": args.kernel_regex, "PROFILE_FILTER_MODE": "global",
            "PROFILE_LAUNCH_COUNT": str(args.launch_count), "PROFILE_LAUNCH_SKIP": str(args.launch_skip),
            "PROFILE_NVTX_INCLUDE": nvtx, "PROFILE_REQUIRE_NVTX": "1",
            "PROFILE_CACHE_CONTROL": "none", "PROFILE_CLOCK_CONTROL": "none"}


def validate_sections(profiler, sections):
    # Listing sections never runs a kernel. Fail before warmup for absent sections.
    result = subprocess.run([profiler, "--config-file", "off", "--list-sections"], capture_output=True,
                            text=True, check=True, timeout=30)
    missing = [section for section in sections
               if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(section) + r"(?![A-Za-z0-9_])", result.stdout)]
    if missing:
        raise ValueError("Installed Nsight Compute lacks sections: " + ", ".join(missing))
    return result.stdout


def validate_build(server, hardware="a100"):
    if not server.is_file():
        raise ValueError("Build the annotated server first: python3 benchmark/matrix_diagnostic.py build "
                         f"--server --cuda-arch {architecture(hardware)} --build-root .run/{hardware}-profile-build")
    if (server.parent.parent / "pending-instrumentation.json").exists():
        raise ValueError("Diagnostic instrumentation has uncompiled changes; rebuild before profiling")
    provenance = server_build_provenance(server)
    if (provenance["status"] != "verified_adjacent_build" or provenance["commit"] != PIN
            or not provenance.get("build", {}).get("server_phase_diff_sha256")):
        raise ValueError("Diagnostic server must have verified pinned build metadata and phase annotations")
    if hardware == "rtxpro6000":
        command = provenance.get("build", {}).get("configure_command", [])
        if f"-DCMAKE_CUDA_ARCHITECTURES={architecture(hardware)}" not in command:
            raise ValueError("Rebuild the diagnostic server with --cuda-arch 120")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", choices=("plan", "sections", "trace", "counters"), default="plan")
    parser.add_argument("--hardware", choices=HARDWARE, default="a100")
    parser.add_argument("--cell", type=Path, help="One completed setting directory containing cell.json and study.env")
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode", help="Counter phase; trace captures both")
    parser.add_argument("--device", type=int, default=0, help="Logical counter GPU index within the saved GPU_DEVICE list")
    parser.add_argument("--kernel-regex", default="mul_mat|gemm|gemv|mma", help="Demangled kernel-name filter; refine using an optional trace")
    parser.add_argument("--operation-regex", help="Optional inner csb_op NVTX range regex, matched within the selected phase")
    parser.add_argument("--scalar-roofline", action="store_true",
                        help="Collect duration, DRAM bytes and predicated-on SASS thread counts for scalar FP32 coordinates")
    parser.add_argument("--launch-count", type=int, default=5, help="Maximum total matching launches on the selected GPU")
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--roofline", nargs="+", choices=tuple(ROOFLINE_SECTIONS), default=[], help="Optional precision-specific roofline sections")
    parser.add_argument("--diagnostic-server", type=Path)
    parser.add_argument("--output", type=Path, help="Fresh output directory; default derived from the cell and capture options")
    parser.add_argument("--ncu", default=os.environ.get("NCU_BIN") or shutil.which("ncu") or "ncu")
    parser.add_argument("--nsys", default=os.environ.get("NSYS_BIN") or shutil.which("nsys") or "nsys")
    args = parser.parse_args()
    args.diagnostic_server = args.diagnostic_server or ROOT / f".run/{args.hardware}-profile-build/baseline/llama-server"
    if args.action == "sections":
        collection = a100_section_collection(args.roofline)
        print(validate_sections(args.ncu, collection["sections"]))
        return
    if args.cell is None:
        parser.error("--cell is required; select the exact saved serving setting")
    cell, devices = load_cell(args.cell, args.hardware)
    settings = capture_settings(args, len(devices))
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    tag = "trace" if args.action == "trace" else f"ncu-{args.phase}-gpu{args.device}"
    output = (args.output or Path(cell["cell_file"]).parent / "profiles" / f"{tag}-{signature}").resolve()
    if args.action == "plan":
        print(json.dumps({"source_cell": cell["cell_file"], "output": str(output), "settings": settings,
                          "visible_physical_gpus": devices, "counter_physical_gpu": devices[args.device],
                          "diagnostic_server": str(args.diagnostic_server.resolve()),
                          "warmup_requests": cell["concurrency"], "profiled_requests": cell["concurrency"],
                          "output_tokens_per_request": 512, "capture_attempts": 1,
                          "launch_limit": "global across matching kernels on one selected GPU",
                          "additional_kernel_replay_passes": "Chosen by Nsight for requested counters/sections",
                          "inference_executed": False}, indent=2))
        return
    validate_build(args.diagnostic_server.resolve(), args.hardware)
    profiler_key = "NSYS_BIN" if args.action == "trace" else "NCU_BIN"
    profiler_arg = args.nsys if args.action == "trace" else args.ncu
    profiler = shutil.which(profiler_arg)
    if not profiler:
        raise ValueError(f"Profiler not found: {profiler_arg}")
    os.environ[profiler_key] = str(Path(profiler).resolve())
    if args.action == "counters":
        validate_sections(profiler, settings["PROFILE_SECTIONS"].split(","))
        extras = Path(profiler).resolve().parent / "extras/python"
        if not list(extras.glob("ncu_report*")):
            raise ValueError(f"Nsight Python Report Interface missing from {extras}; install the full Nsight Compute package")
    (ROOT / ".run").mkdir(exist_ok=True)
    with (ROOT / ".run/remote-experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another benchmark/profile command is active in this checkout") from None
        check_devices(devices, os.environ, idle=True, hardware=args.hardware)
        if args.hardware == "rtxpro6000":
            saved = validate_rtx_cell(cell, devices)
            from capacity import query_memory
            actual = query_memory(devices)
            if [d.get("uuid") for d in saved] != [d["uuid"] for d in actual]:
                raise ValueError("Selected GPU UUIDs differ from the serving measurement")
        captured = run_capture(cell, output, settings, execute=True, script=ROOT / "scripts/07_profile_cuda.sh",
                               attempts=1, diagnostic_server=args.diagnostic_server.resolve(),
                               requested_launches=1)
    print(f"Profile saved: {captured}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
