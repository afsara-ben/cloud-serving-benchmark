#!/usr/bin/env python3
"""Run CUDA serving measurements serially with an explicit hardware scope.

The A100 default retains its 2K-16K scope. --hardware rtxpro6000 selects 70B,
two SM120 GPUs, C8/16/32 and 2K-32K inputs, all with two measured request waves.

Examples (run from the repository checkout):
  python3 scripts/run_a100_serving.py plan --gpus 0,1
  python3 scripts/run_a100_serving.py build --gpus 0,1
  python3 scripts/run_a100_serving.py prepare --gpus 0,1
  python3 scripts/run_a100_serving.py run --gpus 0,1
  python3 scripts/run_a100_serving.py run --gpus 0 --models 8b --formats Q4_K_M --concurrencies 8 --prompt-lengths 2048
  python3 scripts/run_a100_serving.py report --gpus 0

Build and prepare once before running. Preparing a model size prepares its full
format set so later individual-cell selections keep the same resume identity.
70B FP16 is excluded from this workflow. Use a new --study-name when changing
hardware, binaries, source code or models. No action defaults to plan.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import report_serving as report
from cuda_hardware import HARDWARE, architecture, matches

DEFAULT_CONCURRENCIES = ",".join(map(str, report.CONCURRENCIES))


def selection(args):
    hardware = getattr(args, "hardware", "a100")
    scope = report.scope_document([], hardware)
    def values(text, allowed, label, convert=str):
        result = [convert(value.strip()) for value in text.split(",") if value.strip()]
        if not result or len(set(result)) != len(result) or set(result) - set(allowed):
            raise ValueError(f"{label} must select unique values from {', '.join(map(str, allowed))}")
        return result
    formats = values(args.formats, report.FORMATS["1b"], "--formats") if args.formats else None
    concurrencies = values(args.concurrencies, scope["concurrencies"], "--concurrencies", int)
    prompts = values(args.prompt_lengths, scope["prompt_lengths"] + scope["capacity_lengths"], "--prompt-lengths", int)
    selected = {}
    if set(args.models) - set(scope["model_formats"]):
        raise ValueError("The RTX PRO 6000 paper workflow selects 70B only")
    for model in scope["model_formats"]:
        if model not in args.models:
            continue
        chosen = [quant for quant in report.FORMATS[model] if formats is None or quant in formats]
        if not chosen:
            continue
        selected[model] = [f"{quant}/c{concurrency}/p{prompt}" for quant in chosen
                           for concurrency in concurrencies for prompt in prompts]
    if not selected:
        raise ValueError("No applicable settings selected; 70B FP16 is excluded")
    return selected


def manifest_path(model, hardware="a100"):
    return ROOT / "models" / f"{hardware}-serving-manifest-{model}.json"


def run_command(argv, environment):
    print(shlex.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, env=environment, check=True)


def check_build_tools(environment):
    required = ("git", "cmake", "ninja", "c++", "nvcc", "curl", "flock")
    missing = [name for name in required if not shutil.which(name, path=environment.get("PATH"))]
    if missing:
        raise ValueError("Missing build prerequisites: " + ", ".join(missing) + ". "
                         "Install CMake/Ninja in the active environment with "
                         "python3 -m pip install -r scripts/requirements-rtxpro6000.txt. "
                         "Load the site's CUDA/compiler modules for nvcc/c++; see RTX_PRO_6000.md. "
                         "Run the gguf-py install only after build succeeds.")


def check_devices(devices, environment, idle=False, hardware="a100"):
    result = subprocess.check_output(["nvidia-smi", "-i", ",".join(devices), "--query-gpu=name", "--format=csv,noheader"],
                                     env=environment, text=True)
    names = result.splitlines()
    if len(names) != len(devices) or any(not matches(name, hardware) for name in names):
        raise ValueError(f"Expected {len(devices)} {hardware} GPUs; received {names}")
    if hardware == "rtxpro6000":
        capabilities = subprocess.check_output(["nvidia-smi", "-i", ",".join(devices),
            "--query-gpu=compute_cap,memory.total", "--format=csv,noheader,nounits"], env=environment, text=True)
        rows = [row.split(",") for row in capabilities.splitlines()]
        if (len(devices) != 2 or len(rows) != 2 or
                any(row[0].strip() != "12.0" or float(row[1]) < 90000 for row in rows)):
            raise ValueError("Select two whole SM120 RTX PRO 6000 96GB GPUs (no MIG slices)")
    if idle:
        processes = subprocess.check_output(["nvidia-smi", "-i", ",".join(devices), "--query-compute-apps=pid",
                                             "--format=csv,noheader,nounits"], env=environment, text=True)
        if any(line.strip().isdigit() for line in processes.splitlines()):
            raise ValueError("A selected GPU has an active compute process; use an idle allocation")


def preparation_entries(model, hardware="a100"):
    source = manifest_path(model, hardware)
    if not source.is_file():
        source = ROOT / "models" / f"study-manifest-{model}.json"
    entries = json.loads(source.read_text())
    indexed = {entry["quant"]: entry for entry in entries}
    prepared, missing = [], []
    for quant in report.FORMATS[model]:
        entry = dict(indexed[quant])
        local = ROOT / "models" / entry["filename"]
        original = Path(entry.get("path", local)).expanduser()
        # Prefer the portable checkout-local copy; retain a valid original path.
        path = local if local.is_file() or not original.is_file() else original
        entry["path"] = str(path.resolve())
        if hardware == "rtxpro6000":
            # Llama 3 role tokens are shared by 3.1/3.3. Avoid GGUF-specific
            # date/system preambles changing inputs across the four formats.
            entry["chat_template_file"] = "config/templates/llama-3.1-benchmark.jinja"
        if not path.is_file() and not entry.get("repo"):
            missing.append(str(local))
        prepared.append(entry)
    if missing:
        raise ValueError("Copy the existing GGUF files to these paths before prepare:\n  " +
                         "\n  ".join(missing) + "\nTheir original upstream sources are not recorded; "
                         "prepare cannot download substitutes.")
    return prepared


def prepare_models(models, devices, hardware="a100"):
    spec = importlib.util.spec_from_file_location("prepare_study", ROOT / "scripts/09_prepare_study_models.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Check local-only file availability before starting any downloads.
    entries = {model: preparation_entries(model, hardware) for model in models}
    args = SimpleNamespace(include_remote_only=False, devices=",".join(devices),
                           log_dir=ROOT / f"results/{hardware}-model-preparation", source_dir=ROOT / ".run/model-sources",
                           token_file=Path.home() / ".hf_token")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for model, items in entries.items():
        for index, entry in enumerate(items):
            items[index] = helper.prepare(entry, args)
            helper.write_json(manifest_path(model, hardware), items)


def initialize_series(root, devices, hardware="a100"):
    scope = report.scope_document(devices, hardware)
    path = root / "serving-scope.json"
    if path.exists() and json.loads(path.read_text()) != scope:
        raise ValueError("Existing serving scope differs; choose a new --study-name")
    root.mkdir(parents=True, exist_ok=True)
    for model in report.MODELS:
        link = root / model
        target = Path("..") / f"{root.name}-{model}"
        if link.is_symlink():
            if os.readlink(link) != str(target):
                raise ValueError(f"Unexpected model link: {link}")
        elif link.exists():
            raise ValueError(f"Expected a model link: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    if not path.exists():
        path.write_text(json.dumps(scope, indent=2) + "\n")


def study_command(action, model, cells, hardware="a100"):
    scope = report.scope_document([], hardware)
    command = ["bash", "execute.sh", action, "cuda", model, "--manifest", str(manifest_path(model, hardware)),
               "--formats", ",".join(scope["model_formats"][model]), "--concurrencies", ",".join(map(str, scope["concurrencies"])),
               "--prompt-lengths", ",".join(map(str, scope["prompt_lengths"])),
               "--capacity-lengths", ",".join(map(str, scope["capacity_lengths"])), "--repetitions", "1"]
    if hardware == "rtxpro6000":
        command += ["--extension-request-waves", "2"]
    for cell in cells:
        command.extend(("--only-cell", cell))
    return command


def main(default_hardware="a100"):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", choices=("plan", "build", "prepare", "run", "report"), default="plan")
    parser.add_argument("--hardware", choices=HARDWARE, default=default_hardware)
    parser.add_argument("--gpus", default="0,1", help="Physical GPU indices; RTX paper scope requires two whole GPUs")
    parser.add_argument("--study-name", help="Fresh results series; default: cuda-context-study-HARDWARE-Ngpu")
    parser.add_argument("--models", nargs="+", choices=report.MODELS)
    parser.add_argument("--formats", help="Comma-separated IQ1_M,Q2_K,Q4_K_M,Q8_0,FP16; default: all applicable formats")
    parser.add_argument("--concurrencies",
                        help=f"Comma-separated selection of {DEFAULT_CONCURRENCIES}")
    parser.add_argument("--prompt-lengths", help="Comma-separated inputs; RTX scope includes 32768")
    parser.add_argument("--no-plots", action="store_true", help="Generate CSV/Markdown only")
    args = parser.parse_args()
    scope = report.scope_document([], args.hardware)
    default_concurrencies = ",".join(map(str, scope["concurrencies"]))
    default_prompts = ",".join(map(str, scope["prompt_lengths"] + scope["capacity_lengths"]))
    args.models = args.models or list(scope["model_formats"])
    args.concurrencies = args.concurrencies or default_concurrencies
    args.prompt_lengths = args.prompt_lengths or default_prompts
    devices = args.gpus.split(",")
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices) or any(not re.fullmatch(r"\d+", gpu) for gpu in devices):
        parser.error("--gpus requires one or two unique integer device indices")
    if args.hardware == "rtxpro6000" and len(devices) != 2:
        parser.error("The RTX PRO 6000 paper scope requires exactly two GPUs")
    name = args.study_name or f"cuda-context-study-{args.hardware}-{len(devices)}gpu"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..", "cuda-context-study") or name.startswith(("cuda-study", "prefix-study")):
        parser.error("Choose a distinct, simple --study-name for this serving series")
    selected = selection(args)
    root = ROOT / "results" / name
    environment = dict(os.environ, GPU_DEVICE=",".join(devices), SERVER_TENSOR_SPLIT=",".join("1" for _ in devices),
                       STUDY_NAME=name, ACCELERATOR_BACKEND="cuda", CUDA_ARCHITECTURES=architecture(args.hardware),
                       CONFIG_FILE=str(ROOT / "config/context-study.env"))
    if args.action == "plan":
        warmup = sum(int(cell.split("/")[1][1:]) for cells in selected.values() for cell in cells)
        print(json.dumps({"devices": devices, "output": str(root), "selected_cells": selected,
                          "settings": sum(map(len, selected.values())), "warmup_requests_if_all_feasible": warmup,
                          "measured_requests_if_all_feasible": warmup * 2, "output_tokens_per_request": 512,
                          "repetitions": 1, "capacity_extensions": bool(scope["capacity_lengths"]), "profiling": False,
                          "hardware": args.hardware, "cuda_architecture": architecture(args.hardware),
                          "execution_order": list(selected), "inference_executed": False}, indent=2))
        return
    if args.action == "build":
        if args.hardware == "rtxpro6000":
            check_build_tools(environment)
        check_devices(devices, environment, idle=True, hardware=args.hardware)
        run_command(["bash", "scripts/01_build_llama_cpp.sh"], environment)
        return
    if args.action == "prepare":
        if args.formats or args.concurrencies != default_concurrencies or args.prompt_lengths != default_prompts:
            parser.error("prepare operates on all formats of each selected --models size; cell selectors apply to plan/run")
        check_devices(devices, environment, hardware=args.hardware)
        prepare_models(selected, devices, args.hardware)
        return
    if args.action == "run":
        # Validate all selected model manifests before the first inference job.
        for model in selected:
            path = manifest_path(model, args.hardware)
            if not path.is_file():
                raise ValueError(f"Missing {path}; run prepare --models {model} first")
            entries = json.loads(path.read_text())
            if [entry["quant"] for entry in entries] != list(report.FORMATS[model]):
                raise ValueError(f"Unexpected model formats in {path}; rerun prepare")
            for entry in entries:
                if not Path(entry["path"]).is_file():
                    raise ValueError(f"Missing prepared model: {entry['path']}; rerun prepare")
        check_devices(devices, environment, idle=True, hardware=args.hardware)
        initialize_series(root, devices, args.hardware)
        for model, cells in selected.items():
            try:
                run_command(study_command("study", model, cells, args.hardware), environment)
            finally:
                # Save available metrics even if a later cell is interrupted.
                report.build_report(root, plots=False)
    audit = report.build_report(root, plots=not args.no_plots)
    print(f"Report: {root / 'serving-report/index.md'} ({audit['measured_cells']} measured settings)")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
