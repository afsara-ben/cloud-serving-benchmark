#!/usr/bin/env python3
"""Run/resume the fixed-workload CUDA or ROCm HTTP study; dry-run touches no GPU."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/cuda-study.env"
PIN = "3f5e94d7c2ab2267fe39852051777fe30c1f49ef"
KEYS = "EXPERIMENT_NAME ACCELERATOR_BACKEND LLAMA_CPP_REF GPU_DEVICE SERVER_HOST SERVER_PORT SERVER_PARALLEL SERVER_CTX_SIZE SERVER_BATCH_SIZE SERVER_UBATCH_SIZE GPU_LAYERS FLASH_ATTN CONCURRENCY_LEVELS REQUESTS_PER_LEVEL REPETITIONS TARGET_PROMPT_TOKENS MAX_OUTPUT_TOKENS WARMUP_REQUESTS REQUEST_TIMEOUT_SECONDS RANDOM_SEED".split()
OPTIONAL_KEYS = "SERVER_SPLIT_MODE SERVER_TENSOR_SPLIT SERVER_MAIN_GPU SERVER_LOAD_MODE SERVER_STARTUP_TIMEOUT_SECONDS".split()


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    # hashlib.file_digest is unavailable in Ubuntu 22.04's Python 3.10.
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def model_path(model):
    if model.get("path"):
        path = Path(model["path"]).expanduser()
        if not path.is_absolute():
            raise RuntimeError(f"Manifest model path must be absolute: {path}")
    else:
        path = ROOT / "models" / model["filename"]
    return path.resolve()


def chat_template_path(model):
    value = model.get("chat_template_file")
    if not value:
        return None
    if not isinstance(value, str):
        raise RuntimeError("Manifest chat_template_file must be a path string")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Manifest chat template does not exist: {path}")
    return path


def chat_template_controls(model):
    path = chat_template_path(model)
    return {"SERVER_CHAT_TEMPLATE_FILE": str(path) if path else ""}


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, list) or not manifest:
        raise RuntimeError("Model manifest must be a nonempty list.")
    formats = set()
    for model in manifest:
        quant = model.get("quant") if isinstance(model, dict) else None
        if not isinstance(quant, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", quant):
            raise RuntimeError("Each manifest entry needs a nonempty quant label containing letters, digits, underscores, dots or hyphens.")
        if quant in formats:
            raise RuntimeError(f"Model manifest contains duplicate quant format: {quant}")
        formats.add(quant)
        if not model.get("filename") and not model.get("path"):
            raise RuntimeError(f"Manifest format {quant} requires filename or absolute path.")
        path = model_path(model)
        model.setdefault("filename", path.name)
        chat_template_path(model)
    return manifest


def verified_model(model):
    path = model_path(model)
    actual_bytes = path.stat().st_size
    if model.get("bytes") is not None and model["bytes"] != actual_bytes:
        raise RuntimeError(f"Model integrity check failed for {path.name}: expected {model['bytes']} bytes, got {actual_bytes}.")
    actual_sha256 = digest(path)
    if model.get("sha256") is not None and str(model["sha256"]).lower() != actual_sha256:
        raise RuntimeError(f"Model integrity check failed for {path.name}: expected SHA256 {model['sha256']}, got {actual_sha256}.")
    return model | {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def resolve_backend(requested, env):
    if requested in ("cuda", "rocm"):
        return requested
    if requested != "auto":
        raise RuntimeError(f"Unsupported accelerator backend: {requested}")
    available = [name for name, executable in (("cuda", "nvcc"), ("rocm", "hipconfig"))
                 if shutil.which(executable, path=env["PATH"])]
    if len(available) != 1:
        raise RuntimeError("Auto detection requires exactly one installed CUDA/ROCm toolkit; pass --backend cuda or --backend rocm.")
    return available[0]


def probe(argv, env, timeout=20):
    """Retain diagnostic failures; an installed tool does not establish access."""
    try:
        result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return {"command": argv, "status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode, "output": result.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "status": "unavailable", "error": str(exc)}


def monitoring_commands(backend, device, env):
    if backend == "cuda":
        query = ("timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.free,"
                 "power.draw,temperature.gpu,clocks.sm,clocks.mem,clocks_event_reasons.active,"
                 "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown")
        return {"identity": ["nvidia-smi", "-i", device, "--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total", "--format=csv,noheader"],
                "hardware": ["nvidia-smi", "-q", "-i", device],
                "processes": ["nvidia-smi", "-i", device, "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                "version": ["nvcc", "--version"],
                "telemetry": ["nvidia-smi", "-i", device, f"--query-gpu={query}", "--format=csv", "-lms", "200"]}
    if shutil.which("amd-smi", path=env["PATH"]):
        return {"identity": ["amd-smi", "list", "--gpu", device, "--json"],
                "hardware": ["amd-smi", "static", "--gpu", device, "--json"],
                "processes": ["amd-smi", "process", "--gpu", device, "--json"],
                "version": ["hipconfig", "--full"], "smi_version": ["amd-smi", "version"],
                "telemetry": ["amd-smi", "metric", "--gpu", device, "--usage", "--power", "--temperature", "--clock", "--mem-usage", "--json"]}
    return {"identity": ["rocm-smi", "-d", device, "--showuniqueid", "--showserial", "--showproductname", "--showdriverversion", "--json"],
            "hardware": ["rocm-smi", "-d", device, "--showallinfo", "--json"],
            "processes": ["rocm-smi", "--showpids", "--json"],
            "version": ["hipconfig", "--full"], "smi_version": ["rocm-smi", "--version"],
            "telemetry": ["rocm-smi", "-d", device, "--showuse", "--showmemuse", "--showmeminfo", "vram", "--showpower", "--showtemp", "--showclocks", "--json"]}


def permission_indicators():
    indicators = {"effective_uid": os.geteuid(), "groups": os.getgroups(),
                  "counter_access": "not_probed", "device_nodes": {}}
    for path in [Path("/dev/kfd"), *sorted(Path("/dev/dri").glob("renderD*")),
                 *sorted(Path("/dev").glob("nvidia[0-9]*"))]:
        indicators["device_nodes"][str(path)] = {"exists": path.exists(),
                                               "readable": os.access(path, os.R_OK),
                                               "writable": os.access(path, os.W_OK)}
    for name in ("/proc/driver/nvidia/params", "/proc/sys/kernel/perf_event_paranoid"):
        try:
            contents = Path(name).read_text()
            if name.endswith("/params"):
                contents = "\n".join(line for line in contents.splitlines() if "Profiling" in line)
            indicators[name] = contents.strip()
        except OSError as exc:
            indicators[name] = {"unavailable": str(exc)}
    return indicators


class Telemetry:
    """CUDA keeps its existing CSV; AMD samples retain vendor JSON and timestamps."""
    def __init__(self, backend, argv, env, cell):
        self.backend, self.argv, self.env, self.cell = backend, argv, env, cell
        self.process = self.thread = self.stream = None
        self.stop_event = threading.Event()
        self.status = {"backend": backend, "command": argv, "status": "not_started",
                       "sample_interval_seconds": 0.2 if backend == "cuda" else 1.0,
                       "counter_access": "not_probed"}

    def start(self):
        self.status["invocation_started_unix"] = time.time()
        if not shutil.which(self.argv[0], path=self.env["PATH"]):
            self.status.update(status="unavailable", reason=f"{self.argv[0]} not found")
            return
        self.stream = (self.cell / ("telemetry.csv" if self.backend == "cuda" else "telemetry.jsonl")).open("w")
        try:
            self.status["status"] = "started"
            if self.backend == "cuda":
                self.process = subprocess.Popen(self.argv, env=self.env, stdout=self.stream, stderr=subprocess.STDOUT)
            else:
                self.thread = threading.Thread(target=self.sample_amd, daemon=True)
                self.thread.start()
        except OSError as exc:
            self.status.update(status="unavailable", reason=str(exc))

    def sample_amd(self):
        while not self.stop_event.is_set():
            started = time.time()
            sample = probe(self.argv, self.env, timeout=5)
            sample.update(started_unix_seconds=started, finished_unix_seconds=time.time())
            self.stream.write(json.dumps(sample) + "\n")
            self.stream.flush()
            if sample["status"] != "ok":
                self.status.update(status="failed", reason="See telemetry.jsonl for diagnostic output")
                return
            self.status["samples"] = self.status.get("samples", 0) + 1
            self.stop_event.wait(max(0.0, 1.0 - (time.time() - started)))

    def stop(self):
        if self.process is not None:
            early_exit = self.process.poll()
            if early_exit is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.status["status"] = "stopped_after_workload"
            else:
                self.status.update(status="exited_during_workload", returncode=early_exit)
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=7)
            if self.status["status"] == "started":
                self.status["status"] = "stopped_after_workload"
        if self.stream is not None:
            self.stream.close()
        self.status.update(invocation_finished_unix=time.time(), interpretation=
            "Vendor SMI telemetry, not profiler hardware counters. Utilization percentages are not achieved bandwidth or occupancy. "
            "Crop to raw.json summary started_unix_seconds/finished_unix_seconds; AMD samples may span the boundary and have command-limited cadence.")
        write_json(self.cell / "telemetry-window.json", self.status)


def valid(path, requests, concurrency, reuse=False, repeat_prompt=False,
          prompt_tokens=2048, output_tokens=128, exact_prompt=False):
    try:
        data = json.loads(path.read_text())
        s = data["summary"]
        return (data["configuration"]["requests"] == requests
                and data["configuration"]["concurrency"] == concurrency
                and data["configuration"]["max_output_tokens"] == output_tokens
                and data["configuration"]["target_prompt_tokens"] == prompt_tokens
                and (not exact_prompt or (data["configuration"].get("exact_prompt_tokens", False)
                     and all(r.get("prompt_tokens") == prompt_tokens and r.get("prepared_prompt_tokens") == prompt_tokens
                             and r.get("evaluated_prompt_tokens") == prompt_tokens for r in data["requests"])))
                and data["configuration"].get("prompt_cache_requested", False) == reuse
                and data["configuration"].get("repeat_prompt", False) == repeat_prompt
                and (not repeat_prompt or (bool(data["configuration"].get("input_sha256"))
                     and all(r.get("input_sha256") == data["configuration"]["input_sha256"] for r in data["requests"])))
                and s["successful_requests"] == requests and s["failed_requests"] == 0
                and s["output_length_invalid_requests"] == 0
                and s["cache_counter_available_requests"] == requests
                and (all(r["cached_prompt_tokens"] > 0 for r in data["requests"]) if reuse else s["cached_prompt_tokens"] == 0)
                and s["max_client_inflight"] == concurrency
                and len(data["requests"]) == requests and all(r["ok"] for r in data["requests"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def context_arguments(args):
    def integers(text, allowed, name):
        try:
            values = [int(value) for value in text.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError(f"{name} requires comma-separated integers") from exc
        if len(set(values)) != len(values) or any(v not in allowed for v in values):
            raise ValueError(f"{name} requires unique values from {sorted(allowed)}")
        return sorted(values)
    concurrent = integers(args.concurrencies, {8, 16, 32, 64}, "--concurrencies")
    prompts = integers(args.prompt_lengths, {2048, 4096, 8192, 16384}, "--prompt-lengths")
    extension = integers(args.capacity_lengths, {32768, 65536}, "--capacity-lengths")
    if not concurrent or not prompts:
        raise ValueError("At least one concurrency and serving-grid prompt length is required")
    return concurrent, prompts, extension


def selected_context_cells(cells, selectors):
    """Select execution without changing the full study's resume identity."""
    if not selectors:
        return cells
    requested = set(selectors)
    available = {f"{cell['quant']}/c{cell['concurrency']}/p{cell['prompt_tokens']}" for cell in cells}
    if requested - available:
        raise ValueError(f"Unknown --only-cell selections: {sorted(requested - available)}")
    return [cell for cell in cells
            if f"{cell['quant']}/c{cell['concurrency']}/p{cell['prompt_tokens']}" in requested]


def context_controls(cfg, concurrency, prompt):
    sys.path.insert(0, str(ROOT / "benchmark"))
    from capacity import slot_capacity
    return cfg | {"SERVER_PARALLEL": str(concurrency), "SERVER_CTX_SIZE": str(concurrency * slot_capacity(prompt)),
                  "TARGET_PROMPT_TOKENS": str(prompt), "MAX_OUTPUT_TOKENS": "512", "WARMUP_REQUESTS": "0",
                  "CONCURRENCY_LEVELS": str(concurrency), "REQUESTS_PER_LEVEL": str(2 * concurrency),
                  "SERVER_CACHE_TYPE_K": "f16", "SERVER_CACHE_TYPE_V": "f16", "SERVER_NO_CONTEXT_SHIFT": "1",
                  "SERVER_FIT": "off", "SERVER_SPLIT_MODE": "layer", "SERVER_LOAD_MODE": "none", "SERVER_LOG_VERBOSITY": "4"}


class SwapGuard:
    def __init__(self, pid):
        self.pid, self.maximum = pid, 0
        self.errors = []
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        from capacity import process_swap_bytes
        while not self.event.is_set():
            try:
                self.maximum = max(self.maximum, process_swap_bytes(self.pid))
            except (OSError, RuntimeError) as exc:
                self.errors.append(str(exc))
                return
            self.event.wait(0.5)

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join(timeout=2)
        return {"max_server_swap_bytes": self.maximum, "sample_errors": self.errors,
                "sample_interval_seconds": 0.5, "valid": self.maximum == 0 and not self.errors}


def context_memory_summary(path, devices, headroom):
    import csv
    peaks = {str(d["index"]): 0 for d in devices}
    minimum_free = {str(d["index"]): None for d in devices}
    if path.exists():
        with path.open() as stream:
            for row in csv.DictReader(stream, skipinitialspace=True):
                key = row.get("index", "").strip()
                value = row.get("memory.used [MiB]", "")
                if key in peaks and re.match(r"^\d+(?:\.\d+)?", value):
                    peaks[key] = max(peaks[key], int(float(value.split()[0]) * 1024 ** 2))
                free = row.get("memory.free [MiB]", "")
                if key in minimum_free and re.match(r"^\d+(?:\.\d+)?", free):
                    amount = int(float(free.split()[0]) * 1024 ** 2)
                    minimum_free[key] = min(minimum_free[key], amount) if minimum_free[key] is not None else amount
    rows = [{"index": d["index"], "uuid": d["uuid"], "peak_used_bytes": peaks[str(d["index"])],
             "minimum_headroom_bytes": minimum_free[str(d["index"])],
             "samples_available": peaks[str(d["index"])] > 0 and minimum_free[str(d["index"])] is not None} for d in devices]
    return {"gpus": rows, "source": "nvidia-smi/NVML 200ms device-total sampling (not CUDA allocator accounting)",
            "valid": all(r["samples_available"] and r["minimum_headroom_bytes"] >= headroom for r in rows)}


def run_context_study(args, cfg, resolved, backend, devices, config_file):
    """One shared implementation for all model sizes and power-of-two cells."""
    sys.path.insert(0, str(ROOT / "benchmark"))
    from capacity import GIB, inspect_model, estimate, query_memory, validate_placement, classify_failure
    if backend != "cuda":
        raise RuntimeError("The context study currently requires the CUDA backend")
    required = {"LLAMA_CPP_REF": PIN, "SERVER_BATCH_SIZE": "2048", "SERVER_UBATCH_SIZE": "512",
                "GPU_LAYERS": "99", "FLASH_ATTN": "on"}
    changes = {k: {"expected": v, "received": cfg[k]} for k, v in required.items() if cfg[k] != v}
    if changes:
        raise RuntimeError(f"Context study controls differ: {changes}")
    concurrent, prompts, extension = context_arguments(args)
    extension_waves = getattr(args, "extension_request_waves", 1)
    split_text = cfg.get("SERVER_TENSOR_SPLIT") or ",".join("1" for _ in devices)
    split = [float(v) for v in split_text.split(",")]
    if len(split) != len(devices) or any(v <= 0 for v in split):
        raise RuntimeError("SERVER_TENSOR_SPLIT must have one positive weight per visible GPU")
    cfg.update(SERVER_TENSOR_SPLIT=split_text, SERVER_MAIN_GPU=cfg.get("SERVER_MAIN_GPU") or "0")
    manifest = load_manifest(args.manifest)
    if args.formats:
        requested = args.formats.split(",")
        unknown = set(requested) - {m["quant"] for m in manifest}
        if unknown:
            raise RuntimeError(f"Unknown formats: {sorted(unknown)}")
        manifest = [m for m in manifest if m["quant"] in requested]
    families = {m.get("model_family") for m in manifest}
    if len(families) != 1 or None in families:
        raise RuntimeError("Context manifest must describe one model family")
    family = next(iter(families))
    cpp_dir = Path(resolved.get("LLAMA_CPP_DIR", ROOT / "vendor/llama.cpp")).expanduser().resolve()
    build_dir = Path(resolved.get("LLAMA_BUILD_DIR", cpp_dir / "build-cuda")).expanduser().resolve()
    binary = Path(resolved.get("LLAMA_SERVER_BIN", build_dir / "bin/llama-server")).expanduser().resolve()
    runtime = {"LLAMA_CPP_DIR": str(cpp_dir), "LLAMA_BUILD_DIR": str(build_dir), "LLAMA_SERVER_BIN": str(binary)}
    output = (args.output_dir or ROOT / "results/context-study" / family).resolve()
    cells = [{"quant": model["quant"], "concurrency": c, "prompt_tokens": p,
              "experiment": "serving_grid" if p in prompts else "capacity_extension",
              "configuration": context_controls(cfg, c, p) | chat_template_controls(model)}
             for model in manifest for c in concurrent for p in prompts + extension]
    cells = selected_context_cells(cells, args.only_cell)
    if args.dry_run:
        print(json.dumps({"context_study": True, "model_family": family, "manifest": str(args.manifest),
                          "output": str(output), "binary": str(binary), "devices": devices,
                          "models": manifest, "cells": cells, "repetitions": args.repetitions,
                          "serving_requests_per_repetition": "2 * concurrency", "extension_requests_per_repetition": f"{extension_waves} * concurrency",
                          "discarded_warmup": "concurrency requests before the first measurement of each cell",
                          "capacity_screen": "Per-device GGUF weights + FP16 KV + 2 GiB headroom; observed buffers checked after startup",
                          "downloads": "Runner never downloads; absent FP16 is screened from manifest metadata"}, indent=2))
        return 0
    pidfile = ROOT / ".run/llama-server-cuda.pid"
    if pidfile.exists() and Path(f"/proc/{pidfile.read_text().strip()}").exists():
        raise RuntimeError("A managed CUDA server is already running")
    commit = subprocess.check_output(["git", "-C", str(cpp_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise RuntimeError(f"Expected pinned backend {PIN}; received {commit}")
    env = resolved | cfg | runtime | {"CUDA_VISIBLE_DEVICES": cfg["GPU_DEVICE"]}
    monitors = monitoring_commands(backend, cfg["GPU_DEVICE"], env)
    hardware = query_memory(devices, env)
    code_paths = [Path(__file__), ROOT / "benchmark/capacity.py", ROOT / "benchmark/load_test.py",
                  ROOT / "scripts/common.sh", ROOT / "scripts/03_start_server.sh", ROOT / "scripts/06_stop_server.sh",
                  ROOT / "prompts/topics.txt", config_file, ROOT / "config/experiment.env"]
    code_paths.extend(sorted({path for m in manifest if (path := chat_template_path(m)) is not None}))
    # One integrity pass per model; absent portable FP16 entries remain explicit.
    verified = [verified_model(m) if model_path(m).is_file() else m | {"path": str(model_path(m)), "available": False} for m in manifest]
    for model in verified:
        template = chat_template_path(model)
        if template:
            model.update(chat_template_file=str(template), chat_template_sha256=digest(template))
    metadata = {m["quant"]: inspect_model(model_path(m), m, cpp_dir, split) for m in manifest}
    identity = {"schema_version": 1, "context_study": True, "model_family": family, "models": verified,
                "configuration": cfg, "concurrencies": concurrent, "prompt_lengths": prompts, "capacity_lengths": extension,
                "repetitions": args.repetitions, "output_tokens": 512, "runtime_paths": runtime, "llama_cpp_commit": commit,
                "binary_sha256": digest(binary), "code_sha256": {str(p): digest(p) for p in code_paths},
                "devices": [{k: v for k, v in d.items() if k not in ("free_bytes", "used_bytes")} for d in hardware],
                "runtime_options": {k: v for k, v in env.items() if k.startswith(("GGML_", "CUDA_")) or k == "OMP_NUM_THREADS"},
                "headroom_bytes_per_gpu": 2 * GIB}
    if extension_waves != 1:
        identity["extension_request_waves"] = extension_waves
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"fingerprint": fingerprint, "cells": {}}
    if progress["fingerprint"] != fingerprint:
        raise RuntimeError("Resume fingerprint differs; use a new output directory for changed controls, code, binary, or models")
    write_json(output / "manifest.json", identity)
    write_json(output / "environment.json", {"utc": stamp(), "gpu_memory_before": hardware,
               "permissions": permission_indicators(), "probes": {k: probe(v, env) for k, v in monitors.items() if k != "telemetry"}})
    for quant, meta in metadata.items():
        write_json(output / quant / "tensor-inventory.json", meta)
    write_json(progress_path, progress)
    base_url = f"http://{cfg['SERVER_HOST']}:{cfg['SERVER_PORT']}"
    shared_log = ROOT / "logs/llama-server-cuda.log"

    def command(argv, logfile, execution_env, check=False):
        with (output / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps({"utc": stamp(), "argv": list(map(str, argv)), "config_file": execution_env["CONFIG_FILE"]}) + "\n")
        with logfile.open("w") as stream:
            return subprocess.run(argv, cwd=ROOT, env=execution_env, stdout=stream, stderr=subprocess.STDOUT, check=check)

    def load_command(alias, prompt, concurrency, count, repetition, path, prompts_output=None):
        return [sys.executable, str(ROOT / "benchmark/load_test.py"), "--base-url", base_url, "--model", alias,
                "--topics", str(ROOT / "prompts/topics.txt"), "--concurrency", str(concurrency), "--requests", str(count),
                "--target-prompt-tokens", str(prompt), "--max-output-tokens", "512", "--warmup-requests", "0",
                "--timeout", cfg["REQUEST_TIMEOUT_SECONDS"], "--seed", str(int(cfg["RANDOM_SEED"]) + repetition),
                "--repetition", str(repetition), "--exact-prompt-tokens", "--output", str(path)
                ] + (["--prompts-output", str(prompts_output)] if prompts_output else [])

    def check_raw(path, count, concurrency, prompt):
        return valid(path, count, concurrency, prompt_tokens=prompt, output_tokens=512, exact_prompt=True)

    for entry in cells:
        quant, concurrency, prompt = entry["quant"], entry["concurrency"], entry["prompt_tokens"]
        model = next(m for m in verified if m["quant"] == quant)
        meta = metadata[quant]
        key = f"{quant}/c{concurrency}/p{prompt}"
        folder = output / key
        folder.mkdir(parents=True, exist_ok=True)
        count = (2 if entry["experiment"] == "serving_grid" else extension_waves) * concurrency
        pending = [r for r in range(1, args.repetitions + 1) if not (
            progress["cells"].get(f"{key}/r{r}", {}).get("status") == "complete"
            and check_raw(folder / f"r{r}/raw.json", count, concurrency, prompt))]
        if not pending and (folder / "cell.json").exists():
            print(f"Skipping completed {key}", flush=True)
            continue
        controls = entry["configuration"] | runtime | {"MODEL_PATH": str(model_path(model)), "MODEL_FILENAME": model["filename"],
                    "MODEL_ALIAS": f"{family}-{quant}", "MODEL_URL": model.get("url", "")}
        cell_config = folder / "study.env"
        cell_config.write_text("# Exact context-study cell settings.\nsource " + shlex.quote(str(config_file)) + "\n"
                               + "".join(f"{k}={shlex.quote(str(v))}\n" for k, v in controls.items()))
        execution_env = env | controls | {"CONFIG_FILE": str(cell_config)}
        cell_doc = {"schema_version": 1, "model_family": family, "quantization": quant, "concurrency": concurrency,
                    "prompt_tokens": prompt, "output_tokens": 512, "experiment": entry["experiment"], "status": "pending",
                    "configuration": controls, "config_file": str(cell_config), "model": model, "fingerprint": fingerprint,
                    "repetitions": args.repetitions, "requests_per_repetition": count, "started_utc": stamp()}
        screen = estimate(meta, query_memory(devices, env), prompt, concurrency)
        if model.get("remote_only") and not args.formats and screen["status"] == "allocation_validation_required":
            screen.update(status="explicit_selection_required", reason="Portable remote-only format requires explicit --formats selection on suitable hardware")
        write_json(folder / "capacity-estimate.json", screen)
        if screen["status"] != "allocation_validation_required":
            cell_doc.update(status=screen["status"], reason=screen["reason"], finished_utc=stamp())
            write_json(folder / "cell.json", cell_doc)
            progress["cells"][key] = {"status": cell_doc["status"], "cell": str((folder / "cell.json").relative_to(output))}
            write_json(progress_path, progress)
            print(f"{key}: {screen['status']} — {screen['reason']}", flush=True)
            continue
        write_json(folder / "cell.json", cell_doc)
        try:
            startup = command(["bash", "scripts/03_start_server.sh"], folder / "start.log", execution_env)
            log = shared_log.read_text(errors="replace") if shared_log.exists() else ""
            if startup.returncode:
                status = classify_failure(log, startup=True)
                cell_doc.update(status=status, reason="Server startup failed; see server.log and start.log")
                if status != "observed_oom":
                    raise RuntimeError(f"{key}: {status}")
                continue
            placement = validate_placement(log, meta, concurrency, prompt)
            after_load = query_memory(devices, env)
            placement["gpu_memory_after_startup"] = after_load
            placement["headroom_valid"] = all(d["free_bytes"] >= 2 * GIB for d in after_load)
            write_json(folder / "placement.json", placement)
            if not placement["valid"]:
                cell_doc.update(status="placement_invalid", reason="; ".join(placement["errors"]))
                raise RuntimeError(f"{key}: {cell_doc['reason']}")
            if not placement["headroom_valid"]:
                cell_doc.update(status="observed_headroom_limit", reason="Startup allocations leave less than 2 GiB headroom on a GPU")
                continue
            pid = int(pidfile.read_text())
            warmup = folder / "warmup-discarded.json"
            warm_guard = SwapGuard(pid)
            warm_telemetry = Telemetry(backend, monitors["telemetry"], env, folder)
            warm_guard.start()
            warm_telemetry.start()
            try:
                result = command(load_command(controls["MODEL_ALIAS"], prompt, concurrency, concurrency, 0, warmup), folder / "warmup-discarded.log", execution_env)
            finally:
                warm_telemetry.stop()
                guard = warm_guard.stop()
                memory = context_memory_summary(folder / "telemetry.csv", hardware, 2 * GIB)
                write_json(folder / "warmup-resources.json", {"swap": guard, "memory": memory})
            if result.returncode or not check_raw(warmup, concurrency, concurrency, prompt):
                cell_doc.update(status=classify_failure(shared_log.read_text(errors="replace")), reason="Discarded concurrent warmup failed")
                if cell_doc["status"] == "observed_oom":
                    continue
                raise RuntimeError(f"{key}: warmup failed")
            if not guard["valid"] or not memory["valid"]:
                cell_doc.update(status="swap_rejected" if not guard["valid"] else "observed_headroom_limit",
                                reason="Warmup swap/headroom validation failed")
                continue
            for repetition in pending:
                rep_key = f"{key}/r{repetition}"
                cell = folder / f"r{repetition}"
                if cell.exists():
                    cell.rename(folder / f"r{repetition}-previous-{time.time_ns()}")
                cell.mkdir()
                progress["cells"][rep_key] = {"status": "running", "started_utc": stamp(), "raw": str((cell / "raw.json").relative_to(output))}
                write_json(progress_path, progress)
                telemetry = Telemetry(backend, monitors["telemetry"], env, cell)
                guard = SwapGuard(pid)
                offset = shared_log.stat().st_size
                telemetry.start()
                guard.start()
                try:
                    result = command(load_command(controls["MODEL_ALIAS"], prompt, concurrency, count, repetition, cell / "raw.json",
                                                  folder / "prompts.json" if repetition == 1 else None), cell / "client.log", execution_env)
                finally:
                    telemetry.stop()
                    swap = guard.stop()
                    with shared_log.open("rb") as stream:
                        stream.seek(offset)
                        (cell / "server.log").write_bytes(stream.read())
                memory = context_memory_summary(cell / "telemetry.csv", hardware, 2 * GIB)
                write_json(cell / "resources.json", {"swap": swap, "memory": memory})
                write_json(cell / "placement.json", placement)
                truncation = bool(re.search(r"truncated\s*=\s*1|context shift", (cell / "server.log").read_text(errors="replace"), re.I))
                accepted = result.returncode == 0 and check_raw(cell / "raw.json", count, concurrency, prompt) and swap["valid"] and memory["valid"] and not truncation
                if (cell / "raw.json").exists():
                    raw = json.loads((cell / "raw.json").read_text())
                    raw["study_configuration"] = cell_doc | {"repetition": repetition, "status": "complete" if accepted else "failed"}
                    write_json(cell / "raw.json", raw)
                status = "complete" if accepted else ("context_truncation_rejected" if truncation else "swap_rejected" if not swap["valid"] else "observed_headroom_limit" if not memory["valid"] else classify_failure((cell / "server.log").read_text(errors="replace")))
                progress["cells"][rep_key].update(status=status, finished_utc=stamp())
                write_json(progress_path, progress)
                if not accepted:
                    cell_doc.update(status=status, reason=f"Repetition {repetition} failed; see raw/client/resources/server artifacts")
                    if status in ("observed_oom", "observed_headroom_limit", "swap_rejected"):
                        break
                    raise RuntimeError(f"{rep_key}: {status}")
                print(f"Completed {rep_key}", flush=True)
            else:
                cell_doc.update(status="complete", reason="All requested repetitions passed token, placement, swap and headroom validation")
        finally:
            command(["bash", "scripts/06_stop_server.sh"], folder / "stop.log", execution_env)
            if shared_log.exists():
                shutil.copy2(shared_log, folder / "server.log")
            cell_doc["finished_utc"] = stamp()
            write_json(folder / "cell.json", cell_doc)
            progress["cells"][key] = {"status": cell_doc["status"], "cell": str((folder / "cell.json").relative_to(output))}
            write_json(progress_path, progress)
    write_json(output / "native-context-exclusions.json", {"input_tokens": 131072, "output_tokens": 512,
               "status": "native_context_unsupported", "reason": "Input plus generation exceeds native 131072 context without extrapolation"})
    print(f"Context study finished: {output}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cuda", "rocm"), default="auto",
                        help="Override config backend; auto uses the configured backend or installed toolkit")
    parser.add_argument("--config-file", type=Path, default=Path(os.environ.get("CONFIG_FILE", CONFIG)))
    parser.add_argument("--prefix-study", action="store_true", help="Compare c8 prefix reuse off/on using identical repeated prompts.")
    parser.add_argument("--context-study", action="store_true", help="Run the opt-in context/concurrency grid and power-of-two capacity extension.")
    parser.add_argument("--formats", default="", help="Comma-separated manifest quant labels; empty selects all.")
    parser.add_argument("--concurrencies", default="8,16,32,64")
    parser.add_argument("--prompt-lengths", default="2048,4096,8192,16384")
    parser.add_argument("--capacity-lengths", default="32768,65536", help="Comma-separated power-of-two extension inputs; empty disables extension.")
    parser.add_argument("--extension-request-waves", type=int, choices=(1, 2), default=1,
                        help="Measured request waves for capacity extensions; use 2 for a matched serving comparison. Part of resume identity.")
    parser.add_argument("--only-cell", action="append", default=[], metavar="FORMAT/cCLIENTS/pTOKENS",
                        help="Execute only this context-study cell; repeat to select more. Keeps the full grid's resume identity.")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/manifest.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.only_cell and not args.context_study:
        parser.error("--only-cell requires --context-study")
    if args.repetitions < 1 or args.requests < 8:
        parser.error("Use at least one repetition and eight requests (for concurrency 8).")
    config_file = args.config_file.expanduser().resolve()
    exported = subprocess.check_output(["bash", "-c", 'set -ae; source "$1"; env -0', "bash", str(config_file)])
    resolved = dict(item.decode().split("=", 1) for item in exported.split(b"\0") if item)
    toolkit_bins = [str(Path(resolved.get("ROCM_PATH", "/opt/rocm")) / "bin"), "/usr/local/cuda/bin"]
    resolved["PATH"] = os.pathsep.join([p for p in toolkit_bins if Path(p).is_dir()] + [resolved["PATH"]])
    backend = resolve_backend(args.backend if args.backend != "auto" else resolved.get("ACCELERATOR_BACKEND", "auto"), resolved)
    cfg = {key: resolved[key] for key in KEYS}
    cfg.update({key: resolved.get(key, "") for key in OPTIONAL_KEYS})
    cfg["ACCELERATOR_BACKEND"] = backend
    devices = [device.strip() for device in cfg["GPU_DEVICE"].split(",")]
    if any(not device for device in devices) or len(set(devices)) != len(devices):
        raise RuntimeError("GPU_DEVICE must contain unique, nonempty device selectors.")
    if backend != "cuda" and len(devices) > 1:
        raise RuntimeError("Comma-separated GPU_DEVICE selectors currently require --backend cuda.")
    cfg["GPU_DEVICE"] = ",".join(devices)
    cfg.update(REPETITIONS=str(args.repetitions), REQUESTS_PER_LEVEL=str(args.requests))
    if args.context_study:
        if args.prefix_study:
            parser.error("--context-study and --prefix-study are mutually exclusive")
        return run_context_study(args, cfg, resolved, backend, devices, config_file)
    fixed_controls = {"LLAMA_CPP_REF": PIN, "SERVER_PARALLEL": "8", "SERVER_CTX_SIZE": "32768",
                      "SERVER_BATCH_SIZE": "2048", "SERVER_UBATCH_SIZE": "512", "GPU_LAYERS": "99",
                      "FLASH_ATTN": "on", "TARGET_PROMPT_TOKENS": "2048", "MAX_OUTPUT_TOKENS": "128"}
    changed_controls = {key: {"expected": expected, "received": cfg[key]}
                        for key, expected in fixed_controls.items() if cfg[key] != expected}
    if changed_controls:
        raise RuntimeError(f"Config changes the fixed study controls: {changed_controls}")
    cfg.update(WARMUP_REQUESTS="0", CONCURRENCY_LEVELS="1 8")
    if args.prefix_study:
        cfg["CONCURRENCY_LEVELS"] = "8"
    args.output_dir = args.output_dir or ROOT / "results" / ("prefix-study" if args.prefix_study else f"{backend}-study")
    manifest = load_manifest(args.manifest)
    family = manifest[0].get("model_family", re.sub(r"([.-]" + re.escape(manifest[0]["quant"]) + r")?\.gguf$", "", manifest[0]["filename"]))
    cpp_dir = Path(resolved.get("LLAMA_CPP_DIR", ROOT / "vendor/llama.cpp")).expanduser().resolve()
    build_dir = Path(resolved.get("LLAMA_BUILD_DIR", cpp_dir / f"build-{backend}")).expanduser().resolve()
    binary = Path(resolved.get("LLAMA_SERVER_BIN", build_dir / "bin/llama-server")).expanduser().resolve()
    runtime_paths = {"LLAMA_CPP_DIR": str(cpp_dir), "LLAMA_BUILD_DIR": str(build_dir), "LLAMA_SERVER_BIN": str(binary)}
    monitors = monitoring_commands(backend, cfg["GPU_DEVICE"], resolved)
    plan = [(rep, manifest[(rep - 1 + index) % len(manifest)])
            for rep in range(1, args.repetitions + 1) for index in range(len(manifest))]
    def conditions(rep):
        if args.prefix_study:
            return [(8, reuse, "reuse-on" if reuse else "reuse-off") for reuse in ([True, False] if rep % 2 == 0 else [False, True])]
        return [(c, False, f"c{c}") for c in ([8, 1] if rep % 2 == 0 else [1, 8])]
    if args.dry_run:
        print(json.dumps({"configuration": cfg, "config_file": str(config_file), "backend": backend,
                          "prefix_study": args.prefix_study, "model_family": family, "manifest": str(args.manifest), "binary": str(binary), "output": str(args.output_dir),
                          "device_selectors": devices, "telemetry_command": monitors["telemetry"],
                          "loads": [{"repetition": r, "quant": m["quant"], "model": m["filename"], "model_path": str(model_path(m)),
                                     "discarded_warmup": ("Before each mode: 8 uncached identical requests; reuse-on adds 8 cached requests" if args.prefix_study else "8 requests at concurrency 8, 2048/128 tokens"),
                                     "condition_order": [name for _, _, name in conditions(r)],
                                     "concurrency_order": [c for c, _, _ in conditions(r)]} for r, m in plan]}, indent=2))
        return 0

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Source the user's config, then pin resolved controls so CLI overrides cannot
    # be reset when common.sh sources it again in the start/stop subprocesses.
    resolved_config = output / "study.env"
    env = resolved | cfg | runtime_paths | {"CONFIG_FILE": str(resolved_config)}
    env["CUDA_VISIBLE_DEVICES" if backend == "cuda" else "HIP_VISIBLE_DEVICES"] = cfg["GPU_DEVICE"]
    def command(argv, logfile, extra=None, check=True):
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"utc": stamp(), "argv": list(map(str, argv)), "environment": cfg | (extra or {})}) + "\n")
        with logfile.open("w") as log:
            return subprocess.run(argv, cwd=ROOT, env=env | (extra or {}), stdout=log, stderr=subprocess.STDOUT, check=check)

    commit = subprocess.check_output(["git", "-C", str(cpp_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise RuntimeError(f"Expected llama.cpp {PIN}; checkout is {commit}.")
    diagnostic_probes = {name: probe(argv, env) for name, argv in monitors.items() if name != "telemetry"}
    environment = {"backend": backend, "device_selectors": devices, "host": platform.node(), "os": platform.platform(),
                   "python": sys.version, "visibility": {name: env[name] for name in
                   ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES") if name in env},
                   "permissions": permission_indicators(), "probes": diagnostic_probes}
    runtime_options = {key: value for key, value in env.items()
                       if key.startswith(("GGML_", "ROCBLAS_", "HIPBLAS_", "HIPBLASLT_", "HSA_", "ROCR_"))
                       or key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS")}
    device_identity = diagnostic_probes["identity"]
    device_identity = {key: value for key, value in device_identity.items() if key != "error"}
    code_paths = [config_file, ROOT / "config/experiment.env", Path(__file__), ROOT / "benchmark/load_test.py",
                  ROOT / "scripts/common.sh", ROOT / "scripts/03_start_server.sh", ROOT / "scripts/06_stop_server.sh", ROOT / "prompts/topics.txt"]
    identity = {"configuration": cfg, "backend": backend, "host": platform.node(), "device_identity": device_identity,
                "runtime_paths": runtime_paths, "runtime_options": runtime_options, "config_file": str(config_file),
                "prefix_study": args.prefix_study, "model_family": family, "manifest": manifest, "manifest_path": str(args.manifest.resolve()), "llama_cpp_commit": commit,
                "binary_sha256": digest(binary), "kv_types": {"K": "f16", "V": "f16"},
                "code_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in code_paths},
                "models": [verified_model(model) for model in manifest]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"fingerprint": fingerprint, "cells": {}}
    if progress["fingerprint"] != fingerprint:
        raise RuntimeError("Resume fingerprint differs: use a new output directory for changed code, models or configuration.")
    resolved_config.write_text("# Resolved study configuration; generated by the runner.\n"
                              + f"source {shlex.quote(str(config_file))}\n"
                              + "".join(f"{key}={shlex.quote(value)}\n" for key, value in (cfg | runtime_paths).items()))
    write_json(output / "manifest.json", identity)
    write_json(output / "environment.json", environment)
    command([str(binary), "--version"], output / "binary-version.txt")
    write_json(progress_path, progress)
    base_url = f"http://{cfg['SERVER_HOST']}:{cfg['SERVER_PORT']}"
    shared_log = ROOT / f"logs/llama-server-{backend}.log"

    def snapshot(path):
        try:
            with urllib.request.urlopen(base_url + "/metrics", timeout=10) as response:
                path.write_bytes(response.read())
        except Exception as exc:
            path.write_text(f"# metrics unavailable: {exc}\n")

    def load_command(model, rep, concurrency, count, path, reuse=False):
        return [sys.executable, str(ROOT / "benchmark/load_test.py"), "--base-url", base_url,
                "--model", model, "--topics", str(ROOT / "prompts/topics.txt"), "--concurrency", str(concurrency),
                "--requests", str(count), "--target-prompt-tokens", "2048", "--max-output-tokens", "128",
                "--warmup-requests", "0", "--timeout", cfg["REQUEST_TIMEOUT_SECONDS"],
                "--seed", str(int(cfg["RANDOM_SEED"]) + rep), "--repetition", str(rep), "--output", str(path)
                ] + (["--repeat-prompt"] if args.prefix_study else []) + (["--prefix-reuse"] if reuse else [])

    for rep, model in plan:
        quant = model["quant"]
        alias = f"{family}-{quant}"
        folder = output / quant / f"r{rep}"
        cells = [(c, reuse, f"{quant}/r{rep}/{name}") for c, reuse, name in conditions(rep)]
        pending = [(c, reuse, key) for c, reuse, key in cells if not (progress["cells"].get(key, {}).get("status") == "complete"
                   and valid(output / progress["cells"][key]["raw"], args.requests, c, reuse, args.prefix_study))]
        if not pending:
            print(f"Skipping valid completed {quant} repetition {rep}", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        extra = {"MODEL_FILENAME": model["filename"], "MODEL_PATH": str(model_path(model)),
                 "MODEL_ALIAS": alias, "MODEL_URL": model.get("url", "")}
        pidfile = ROOT / f".run/llama-server-{backend}.pid"
        if pidfile.exists() and Path(f"/proc/{pidfile.read_text().strip()}").exists():
            raise RuntimeError("A managed server is already running; stop it before starting this study.")
        try:
            command(["bash", "scripts/03_start_server.sh"], folder / "start.log", extra)
            if not args.prefix_study:
                command(load_command(alias, rep, 8, 8, folder / "warmup-discarded.json"), folder / "warmup-discarded.log")
                if not valid(folder / "warmup-discarded.json", 8, 8):
                    raise RuntimeError("Full concurrency warmup failed validation.")
            for concurrency, reuse, key in pending:
                cell = output / key
                if cell.exists():
                    cell.rename(folder / f"{cell.name}-previous-{time.time_ns()}")
                cell.mkdir()
                progress["cells"][key] = {"status": "running", "started_utc": stamp(), "raw": str((cell / "raw.json").relative_to(output))}
                write_json(progress_path, progress)
                if args.prefix_study:
                    for warm_reuse in ([False, True] if reuse else [False]):
                        warm_name = "warmup-reuse" if warm_reuse else "warmup-prime"
                        command(load_command(alias, rep, 8, 8, cell / f"{warm_name}.json", warm_reuse), cell / f"{warm_name}.log")
                        if not valid(cell / f"{warm_name}.json", 8, 8, warm_reuse, True):
                            raise RuntimeError(f"Prefix {warm_name} failed validation: {cell}")
                snapshot(cell / "metrics-before.prom")
                offset = shared_log.stat().st_size
                telemetry = Telemetry(backend, monitors["telemetry"], env, cell)
                telemetry.start()
                try:
                    result = command(load_command(alias, rep, concurrency, args.requests, cell / "raw.json", reuse), cell / "client.log", check=False)
                finally:
                    telemetry.stop()
                    snapshot(cell / "metrics-after.prom")
                    with shared_log.open("rb") as stream:
                        stream.seek(offset)
                        (cell / "server.log").write_bytes(stream.read())
                    shutil.copy2(shared_log, folder / "server.log")
                accepted = result.returncode == 0 and valid(cell / "raw.json", args.requests, concurrency, reuse, args.prefix_study)
                if accepted and args.prefix_study:
                    paired = folder / ("reuse-off" if reuse else "reuse-on") / "raw.json"
                    if valid(paired, args.requests, 8, not reuse, True):
                        accepted = (json.loads(paired.read_text())["configuration"]["input_sha256"]
                                    == json.loads((cell / "raw.json").read_text())["configuration"]["input_sha256"])
                progress["cells"][key].update(status="complete" if accepted else "failed", finished_utc=stamp())
                write_json(progress_path, progress)
                if not accepted:
                    raise RuntimeError(f"Invalid cell {key}; inspect {cell} before resuming.")
                print(f"Completed {key}", flush=True)
        finally:
            command(["bash", "scripts/06_stop_server.sh"], folder / "stop.log", extra, check=False)
            if shared_log.exists():
                shutil.copy2(shared_log, folder / "server.log")
    print(f"Study complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
