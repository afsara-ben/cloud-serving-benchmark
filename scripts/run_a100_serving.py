#!/usr/bin/env python3
"""Run A100 serving measurements serially; no profiling or long-context extensions.

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
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import report_serving as report

DEFAULT_CONCURRENCIES = ",".join(map(str, report.CONCURRENCIES))


def selection(args):
    def values(text, allowed, label, convert=str):
        result = [convert(value.strip()) for value in text.split(",") if value.strip()]
        if not result or len(set(result)) != len(result) or set(result) - set(allowed):
            raise ValueError(f"{label} must select unique values from {', '.join(map(str, allowed))}")
        return result
    formats = values(args.formats, report.FORMATS["1b"], "--formats") if args.formats else None
    concurrencies = values(args.concurrencies, report.CONCURRENCIES, "--concurrencies", int)
    prompts = values(args.prompt_lengths, report.PROMPTS, "--prompt-lengths", int)
    selected = {}
    for model in report.MODELS:
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


def manifest_path(model):
    return ROOT / "models" / f"a100-serving-manifest-{model}.json"


def run_command(argv, environment):
    print(shlex.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, env=environment, check=True)


def check_devices(devices, environment, idle=False):
    result = subprocess.check_output(["nvidia-smi", "-i", ",".join(devices), "--query-gpu=name", "--format=csv,noheader"],
                                     env=environment, text=True)
    names = result.splitlines()
    if len(names) != len(devices) or any("A100" not in name for name in names):
        raise ValueError(f"Expected {len(devices)} A100 GPUs; received {names}")
    if idle:
        processes = subprocess.check_output(["nvidia-smi", "-i", ",".join(devices), "--query-compute-apps=pid",
                                             "--format=csv,noheader,nounits"], env=environment, text=True)
        if any(line.strip().isdigit() for line in processes.splitlines()):
            raise ValueError("A selected GPU has an active compute process; use an idle allocation")


def preparation_entries(model):
    source = manifest_path(model)
    if not source.is_file():
        source = ROOT / "models" / f"study-manifest-{model}.json"
    entries = json.loads(source.read_text())
    indexed = {entry["quant"]: entry for entry in entries}
    prepared = []
    for quant in report.FORMATS[model]:
        entry = dict(indexed[quant])
        local = ROOT / "models" / entry["filename"]
        original = Path(entry.get("path", local)).expanduser()
        # Prefer the portable checkout-local copy; retain a valid original path.
        path = local if local.is_file() or not original.is_file() else original
        entry["path"] = str(path.resolve())
        if not path.is_file() and not entry.get("repo"):
            raise ValueError(f"Copy the existing {model} {quant} GGUF to {local}. Its original upstream source is not recorded.")
        prepared.append(entry)
    return prepared


def prepare_models(models, devices):
    spec = importlib.util.spec_from_file_location("prepare_study", ROOT / "scripts/09_prepare_study_models.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Check local-only file availability before starting any downloads.
    entries = {model: preparation_entries(model) for model in models}
    args = SimpleNamespace(include_remote_only=False, devices=",".join(devices),
                           log_dir=ROOT / "results/a100-model-preparation", source_dir=ROOT / ".run/model-sources",
                           token_file=Path.home() / ".hf_token")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for model, items in entries.items():
        for index, entry in enumerate(items):
            items[index] = helper.prepare(entry, args)
            helper.write_json(manifest_path(model), items)


def initialize_series(root, devices):
    scope = report.scope_document(devices)
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


def study_command(action, model, cells):
    command = ["bash", "execute.sh", action, "cuda", model, "--manifest", str(manifest_path(model)),
               "--formats", ",".join(report.FORMATS[model]), "--concurrencies", DEFAULT_CONCURRENCIES,
               "--prompt-lengths", "2048,4096,8192,16384", "--capacity-lengths", "", "--repetitions", "1"]
    for cell in cells:
        command.extend(("--only-cell", cell))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", choices=("plan", "build", "prepare", "run", "report"), default="plan")
    parser.add_argument("--gpus", default="0,1", help="One or two physical GPU indices, for example 0 or 0,1")
    parser.add_argument("--study-name", help="Fresh results series; default: cuda-context-study-a100-1gpu or -2gpu")
    parser.add_argument("--models", nargs="+", choices=report.MODELS, default=list(report.MODELS))
    parser.add_argument("--formats", help="Comma-separated IQ1_M,Q2_K,Q4_K_M,Q8_0,FP16; default: all applicable formats")
    parser.add_argument("--concurrencies", default=DEFAULT_CONCURRENCIES,
                        help=f"Comma-separated selection of {DEFAULT_CONCURRENCIES}")
    parser.add_argument("--prompt-lengths", default="2048,4096,8192,16384", help="Comma-separated input-token selection")
    parser.add_argument("--no-plots", action="store_true", help="Generate CSV/Markdown only")
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices) or any(not re.fullmatch(r"\d+", gpu) for gpu in devices):
        parser.error("--gpus requires one or two unique integer device indices")
    name = args.study_name or f"cuda-context-study-a100-{len(devices)}gpu"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", "..", "cuda-context-study") or name.startswith(("cuda-study", "prefix-study")):
        parser.error("Choose a distinct, simple --study-name for this serving series")
    selected = selection(args)
    root = ROOT / "results" / name
    environment = dict(os.environ, GPU_DEVICE=",".join(devices), SERVER_TENSOR_SPLIT=",".join("1" for _ in devices),
                       STUDY_NAME=name, ACCELERATOR_BACKEND="cuda", CUDA_ARCHITECTURES="80",
                       CONFIG_FILE=str(ROOT / "config/context-study.env"))
    if args.action == "plan":
        warmup = sum(int(cell.split("/")[1][1:]) for cells in selected.values() for cell in cells)
        print(json.dumps({"devices": devices, "output": str(root), "selected_cells": selected,
                          "settings": sum(map(len, selected.values())), "warmup_requests_if_all_feasible": warmup,
                          "measured_requests_if_all_feasible": warmup * 2, "output_tokens_per_request": 512,
                          "repetitions": 1, "capacity_extensions": False, "profiling": False,
                          "execution_order": list(selected), "inference_executed": False}, indent=2))
        return
    if args.action == "build":
        check_devices(devices, environment, idle=True)
        run_command(["bash", "scripts/01_build_llama_cpp.sh"], environment)
        return
    if args.action == "prepare":
        if args.formats or args.concurrencies != DEFAULT_CONCURRENCIES or args.prompt_lengths != "2048,4096,8192,16384":
            parser.error("prepare operates on all formats of each selected --models size; cell selectors apply to plan/run")
        check_devices(devices, environment)
        prepare_models(selected, devices)
        return
    if args.action == "run":
        # Validate all selected model manifests before the first inference job.
        for model in selected:
            path = manifest_path(model)
            if not path.is_file():
                raise ValueError(f"Missing {path}; run prepare --models {model} first")
            entries = json.loads(path.read_text())
            if [entry["quant"] for entry in entries] != list(report.FORMATS[model]):
                raise ValueError(f"Unexpected model formats in {path}; rerun prepare")
            for entry in entries:
                if not Path(entry["path"]).is_file():
                    raise ValueError(f"Missing prepared model: {entry['path']}; rerun prepare")
        check_devices(devices, environment, idle=True)
        initialize_series(root, devices)
        for model, cells in selected.items():
            try:
                run_command(study_command("study", model, cells), environment)
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
