#!/usr/bin/env python3
"""Collect the 70B bottlenecks figure inputs on RTX PRO 6000, serially.

Each workload uses 8 full-section gate/up captures and 60 targeted scalar
operator captures. plan is read-only and works before serving has completed.
run requires completed Q2_K and Q4_K_M cells at C8. No automatic retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import collect_a100_roofline as operators
import profile_a100_setting as profiler
from run_a100_serving import check_devices

ROOT = Path(__file__).resolve().parents[1]


def comparison_sources(cells):
    """Require matching measured inputs before comparing quantization formats."""
    sources, reference = [], None
    for directory in cells:
        cell, _ = profiler.load_cell(directory, "rtxpro6000")
        path = Path(cell["cell_file"])
        raw = json.loads((path.parent / "r1/raw.json").read_text())
        hashes = [r["input_sha256"] for r in sorted(raw["requests"], key=lambda r: r["request_index"])]
        if len(hashes) != 2 * cell["concurrency"] or not all(hashes):
            raise ValueError("Missing source prompt hashes or unmatched request count")
        if reference is not None and reference != hashes:
            raise ValueError("Input prompts differ across profiling comparison formats")
        reference = hashes
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return sources


def full_jobs(root, output, prompt, server, ncu):
    jobs = []
    for quant, tensor_type in (("Q4_K_M", "q4_K"), ("Q2_K", "q2_K")):
        cell = root / "70b" / quant / "c8" / f"p{prompt}"
        for phase in ("prefill", "decode"):
            n = 512 if phase == "prefill" else 8
            operation = rf"csb_op:role=blk\.[0-9]+\.ffn_(gate|up)\.weight:type={tensor_type}:m=28672:n={n}:k=8192:.*"
            for gpu in (0, 1):
                destination = output / "full" / quant / phase / f"gpu{gpu}"
                command = [sys.executable, str(ROOT / "scripts/profile_a100_setting.py"), "counters",
                           "--hardware", "rtxpro6000", "--cell", str(cell), "--phase", phase,
                           "--device", str(gpu), "--operation-regex", operation, "--roofline", "tensor",
                           "--launch-count", "5", "--diagnostic-server", str(server), "--ncu", ncu,
                           "--output", str(destination)]
                jobs.append({"format": quant, "tensor_type": tensor_type, "phase": phase, "gpu": gpu,
                             "n": n, "cell": str(cell), "capture": str(destination.relative_to(output)),
                             "operation_regex": operation, "command": command})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("plan", "preflight", "pilot", "run", "report"), nargs="?", default="plan")
    parser.add_argument("--study-root", type=Path, default=ROOT / "results/cuda-context-study-rtxpro6000-2gpu")
    parser.add_argument("--prompt-tokens", type=int, choices=(2048, 4096, 8192, 16384, 32768), default=2048)
    parser.add_argument("--stage", choices=("full", "operators", "all"), default="all")
    parser.add_argument("--gpus", default="0,1", help="Only used by preflight; captures use saved serving GPUs")
    parser.add_argument("--diagnostic-server", type=Path, default=ROOT / ".run/rtxpro6000-profile-build/baseline/llama-server")
    parser.add_argument("--ncu", default=os.environ.get("NCU_BIN") or shutil.which("ncu") or "ncu")
    parser.add_argument("--output", type=Path, help="Capture root; use a fresh directory to retry failed captures")
    args = parser.parse_args()
    root = args.study_root.resolve()
    output = (args.output or root / "profiles" / f"c8-p{args.prompt_tokens}").resolve()
    server = args.diagnostic_server.resolve()
    if args.action == "preflight":
        devices = args.gpus.split(",")
        if len(set(devices)) != 2 or any(not d.isdecimal() for d in devices):
            parser.error("Select two unique integer GPU indices")
        check_devices(devices, os.environ, idle=True, hardware="rtxpro6000")
        ncu = shutil.which(args.ncu)
        if not ncu:
            raise ValueError(f"Nsight Compute executable not found: {args.ncu!r}. "
                             "Load the site's Nsight Compute module, or install the full package "
                             "using RTX_PRO_6000.md, then export NCU_BIN=/absolute/path/to/ncu. "
                             "NCU_BIN must name the executable, not its directory. "
                             "This preflight is needed for profiling; serving build/prepare/run can proceed without Nsight.")
        extras = Path(ncu).resolve().parent / "extras/python"
        if not list(extras.glob("ncu_report*")):
            raise ValueError(f"Nsight Python Report Interface missing: {extras}")
        help_text = subprocess.check_output([ncu, "--help"], text=True)
        missing = [flag for flag in ("--config-file", "--rename-kernels", "--graph-profiling", "--nvtx-include", "--filter-mode") if flag not in help_text]
        if missing:
            raise ValueError("Installed Nsight Compute lacks required CLI features: " + ", ".join(missing))
        profiler.validate_sections(ncu, profiler.a100_section_collection(["tensor"])["sections"])
        for device in devices:
            env = dict(os.environ, GPU_DEVICE=device, NCU_BIN=str(Path(ncu).resolve()),
                       NVCC_PREPEND_FLAGS="-arch=sm_120", RUN_TAG="rtxpro6000-preflight")
            subprocess.run(["bash", "execute.sh", "doctor", "cuda"], cwd=ROOT, env=env, check=True)
        print("Both GPUs passed the real counter probe. Validate one full capture before the full campaign.")
        return
    if args.action == "report":
        from rtxpro6000_publication import build_bottlenecks
        print(build_bottlenecks(root, output))
        return
    jobs = full_jobs(root, output, args.prompt_tokens, server, args.ncu)
    cells = [root / "70b" / quant / "c8" / f"p{args.prompt_tokens}" for quant in ("Q4_K_M", "Q2_K")]
    operator_command = [sys.executable, str(ROOT / "scripts/collect_a100_roofline.py"), "run",
                        "--hardware", "rtxpro6000", "--cells", *map(str, cells), "--output", str(output / "operators"),
                        "--diagnostic-server", str(server), "--ncu", args.ncu]
    plan = {"schema_version": 1, "hardware": "rtxpro6000", "study_root": str(root),
            "prompt_tokens": args.prompt_tokens, "concurrency": 8, "output_tokens": 512,
            "warmup_requests_per_capture": 8, "profiled_requests_per_capture": 8,
            "full_capture_count": 8, "operator_capture_count": 60, "launch_cap_per_capture": 5,
            "full_jobs": jobs, "operator_command": operator_command}
    if args.action == "plan":
        print(json.dumps(plan | {"selected_stage": args.stage, "inference_executed": False}, indent=2))
        return
    profiler.validate_build(server, "rtxpro6000")
    plan["source_cells"] = comparison_sources(cells)
    path = output / "paper-profile-plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Profile plan changed; choose a fresh --output")
    if output.exists() and not path.exists() and any(output.iterdir()):
        raise ValueError("Select an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    if args.action == "pilot" or args.stage in ("full", "all"):
        for job in jobs[:1] if args.action == "pilot" else jobs:
            print(f"Full sections: {job['format']} {job['phase']} GPU{job['gpu']}", flush=True)
            subprocess.run(job["command"], cwd=ROOT, check=True)
    if args.action == "pilot":
        print("Pilot capture completed; run resumes this same capture in the full campaign.")
        return
    if args.stage in ("operators", "all"):
        subprocess.run(operator_command, cwd=ROOT, check=True)
    if args.stage == "all":
        from rtxpro6000_publication import build_bottlenecks
        print(build_bottlenecks(root, output))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
