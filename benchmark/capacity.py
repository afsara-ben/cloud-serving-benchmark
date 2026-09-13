"""Capacity arithmetic and placement evidence for the pinned CUDA layer split.

Estimates are lower bounds, never measured allocations. Input embeddings remain
on CPU by backend design; all transformer layers and the output layer must offload.
"""
from __future__ import annotations

import bisect
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2


def slot_capacity(prompt_tokens, output_tokens=512):
    # The pinned server flags truncation at prompt.n_tokens()+1 >= n_ctx.
    # Reserve a guard position beyond requested input/output before padding.
    return ((prompt_tokens + output_tokens + 1 + 255) // 256) * 256


def layer_devices(layers, split):
    if not split or any(value <= 0 for value in split):
        raise ValueError("Tensor split must contain positive weights.")
    total = sum(split)
    bounds, running = [], 0
    for value in split:
        running += value
        bounds.append(running / total)
    # llama-model.cpp get_layer_buft_list counts the output layer as well.
    return [bisect.bisect_right(bounds, layer / (layers + 1)) for layer in range(layers + 1)]


def inspect_model(path, manifest, cpp_dir, split):
    """Read only GGUF metadata/tensor descriptors (reader uses a read-only mmap)."""
    if not Path(path).is_file():
        required = ("parameters", "layers", "kv_heads", "head_dimension")
        if manifest.get("quant", "").upper() not in ("FP16", "F16") or not all(k in manifest for k in required):
            return {"available": False, "reason": "model_file_missing"}
        return {"available": False, "source": "manifest_fp16_lower_bound", "layers": manifest["layers"],
                "kv_heads": manifest["kv_heads"], "head_dimension": manifest["head_dimension"],
                "native_context": manifest.get("native_context", 131072),
                "minimum_total_weight_bytes": manifest["parameters"] * 2,
                "layer_devices": layer_devices(manifest["layers"], split)}
    sys.path.insert(0, str(Path(cpp_dir) / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(str(path), "r")
    def value(key, default=None):
        field = reader.fields.get(key)
        return field.contents() if field else default
    architecture = value("general.architecture")
    if architecture != "llama":
        raise ValueError(f"Capacity geometry is implemented for llama; received {architecture}.")
    layers = int(value("llama.block_count"))
    heads = int(value("llama.attention.head_count"))
    kv_heads = int(value("llama.attention.head_count_kv", heads))
    head_dimension = int(value("llama.attention.key_length", int(value("llama.embedding_length")) // heads))
    value_dimension = int(value("llama.attention.value_length", head_dimension))
    placement = layer_devices(layers, split)
    weights = [0] * len(split)
    cpu_bytes = 0
    tensor_inventory = []
    for tensor in reader.tensors:
        match = re.match(r"blk\.(\d+)\.", tensor.name)
        device = placement[int(match.group(1))] if match else (placement[-1] if tensor.name.startswith(("output.", "output_norm.")) else None)
        if device is None:
            cpu_bytes += tensor.n_bytes
        else:
            weights[device] += ((tensor.n_bytes + 255) // 256) * 256
        tensor_inventory.append({"name": tensor.name, "shape": [int(x) for x in tensor.shape],
                                 "type": tensor.tensor_type.name, "bytes": tensor.n_bytes, "device": device})
    # Tied output weights are duplicated to the output device by llama.cpp.
    if not any(t.name == "output.weight" for t in reader.tensors):
        embedding = next(t for t in reader.tensors if t.name == "token_embd.weight")
        weights[placement[-1]] += ((embedding.n_bytes + 255) // 256) * 256
    return {"available": True, "source": "gguf_tensor_inventory", "layers": layers,
            "kv_heads": kv_heads, "head_dimension": head_dimension, "value_dimension": value_dimension,
            "native_context": int(value("llama.context_length", 131072)), "layer_devices": placement,
            "weight_bytes_per_gpu": weights, "cpu_input_weight_bytes": cpu_bytes,
            "tensors": tensor_inventory}


def query_memory(devices, env=None):
    argv = ["nvidia-smi", "-i", ",".join(devices), "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]
    result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=20)
    rows = []
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        rows.append({"index": row[0], "uuid": row[1], "name": row[2], "total_bytes": int(row[3]) * MIB,
                     "free_bytes": int(row[4]) * MIB, "used_bytes": int(row[5]) * MIB})
    if len(rows) != len(devices):
        raise RuntimeError(f"Expected {len(devices)} GPUs, received {len(rows)} memory records")
    return rows


def estimate(metadata, devices, prompt_tokens, concurrency, output_tokens=512, headroom_bytes=2 * GIB):
    capacity = slot_capacity(prompt_tokens, output_tokens)
    result = {"status": "allocation_validation_required", "slot_capacity": capacity,
              "total_context": capacity * concurrency, "headroom_bytes_per_gpu": headroom_bytes,
              "source": metadata.get("source"), "working_buffers": "Not included in lower bound; validate measured allocation before requests."}
    if capacity > metadata.get("native_context", 131072):
        return result | {"status": "native_context_unsupported", "reason": "Input plus generation exceeds native context."}
    if "minimum_total_weight_bytes" in metadata:
        minimum = metadata["minimum_total_weight_bytes"]
        if minimum > sum(d["free_bytes"] - headroom_bytes for d in devices):
            return result | {"status": "capacity_estimated", "minimum_total_weight_bytes": minimum,
                             "reason": "FP16 weight lower bound alone exceeds aggregate available GPU memory with headroom; no download or load attempted."}
        return result | {"status": "model_missing", "reason": "Capacity lower bound passes, but explicit model acquisition is required."}
    if not metadata.get("available"):
        return result | {"status": "model_missing", "reason": metadata.get("reason")}
    kv_bytes_per_layer = concurrency * capacity * metadata["kv_heads"] * (metadata["head_dimension"] + metadata.get("value_dimension", metadata["head_dimension"])) * 2
    gpu_rows = []
    for index, device in enumerate(devices):
        layers = metadata["layer_devices"][:-1].count(index)
        kv = layers * kv_bytes_per_layer
        weight = metadata["weight_bytes_per_gpu"][index]
        gpu_rows.append({"device": device, "transformer_layers": layers, "weight_bytes": weight, "kv_bytes": kv,
                         "minimum_required_bytes": weight + kv + headroom_bytes,
                         "fits_lower_bound": weight + kv + headroom_bytes <= device["free_bytes"]})
    result["gpus"] = gpu_rows
    if not all(row["fits_lower_bound"] for row in gpu_rows):
        result.update(status="capacity_estimated", reason="Per-GPU weights plus FP16 KV plus reserved headroom exceed available VRAM, before working buffers.")
    return result


def validate_placement(log, metadata, concurrency, prompt_tokens, output_tokens=512):
    errors = []
    offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
    expected_layers = metadata["layers"] + 1
    if not offload or tuple(map(int, offload.groups())) != (expected_layers, expected_layers):
        errors.append("Full transformer/output layer GPU offload was not established.")
    capacity = slot_capacity(prompt_tokens, output_tokens)
    slots = re.search(r"n_seq_max\s*=\s*(\d+)", log)
    slot_ctx = re.search(r"n_ctx_(?:per_seq|seq)\s*=\s*(\d+)", log)
    if not slots or int(slots[1]) != concurrency:
        errors.append("Server active-slot count differs from requested concurrency or is unavailable.")
    if not slot_ctx or int(slot_ctx[1]) != capacity:
        errors.append("Server per-slot context differs from required padded capacity or is unavailable.")
    kv_buffers = re.findall(r"(CUDA\d+|CPU)\s+KV buffer size\s*=\s*([\d.]+) MiB", log)
    expected_devices = set(metadata["layer_devices"][:-1])
    if {int(name[4:]) for name, _ in kv_buffers if name.startswith("CUDA")} != expected_devices:
        errors.append("KV allocation evidence is incomplete for the expected GPUs.")
    if any(name == "CPU" and float(size) > 0 for name, size in kv_buffers):
        errors.append("CPU KV fallback detected.")
    if re.search(r"(?:type_k|type_v)\s*=\s*(?!f16\b)\S+", log):
        errors.append("Non-FP16 KV type detected.")
    kv_types = re.search(r"K \(([^)]+)\):[^\n]+V \(([^)]+)\):", log)
    if not kv_types or kv_types.groups() != ("f16", "f16"):
        errors.append("Positive evidence of FP16 K and V allocations is missing.")
    return {"valid": not errors, "errors": errors, "offload": list(map(int, offload.groups())) if offload else None,
            "slots": int(slots[1]) if slots else None, "slot_capacity": int(slot_ctx[1]) if slot_ctx else None,
            "kv_buffers_mib": dict(kv_buffers), "kv_types": list(kv_types.groups()) if kv_types else None,
            "cpu_input_embeddings": "Expected pinned-backend behavior; not transformer offload fallback."}


def process_swap_bytes(pid):
    text = Path(f"/proc/{pid}/status").read_text()
    match = re.search(r"^VmSwap:\s*(\d+)\s+kB", text, re.MULTILINE)
    if not match:
        raise RuntimeError("Server VmSwap measurement unavailable")
    return int(match[1]) * 1024


def classify_failure(log, startup=False):
    if re.search(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaMalloc.*failed|failed to allocate|unable to allocate", log, re.I):
        return "observed_oom"
    if re.search(r"illegal memory access|device-side assert|CUDA error|no CUDA-capable device|Xid", log, re.I):
        return "hardware_error"
    return "startup_error" if startup else "request_validation_failed"
