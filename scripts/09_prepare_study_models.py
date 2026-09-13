#!/usr/bin/env python3
"""Prepare pinned study GGUFs; FP16 is converted from original BF16 checkpoints.

Examples:
  python3 scripts/09_prepare_study_models.py --models 1b 8b --formats FP16
  python3 scripts/09_prepare_study_models.py --models 70b --formats IQ1_M
  python3 scripts/09_prepare_study_models.py --models 70b --formats FP16 --include-remote-only

The last command screens detected GPU memory before any model network request.
It cannot download 70B FP16 on the original two 48 GiB GPUs. The screen is a
weights-only lower bound; the benchmark must additionally screen KV and buffers.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "3f5e94d7c2ab2267fe39852051777fe30c1f49ef"
GIB = 1024 ** 3
SPECS = {
    "1b": {"model_family": "Llama-3.2-1B-Instruct", "repo": "meta-llama/Llama-3.2-1B-Instruct",
           "revision": "9213176726f574b556790deb65791e0c5aa438b6", "base": "manifest.json",
           "parameters": 1235814400, "layers": 16, "kv_heads": 8, "head_dimension": 64},
    "8b": {"model_family": "Llama-3.1-8B-Instruct", "repo": "meta-llama/Llama-3.1-8B-Instruct",
           "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "base": "manifest-8b.json",
           "parameters": 8030261248, "layers": 32, "kv_heads": 8, "head_dimension": 128},
    "70b": {"model_family": "Llama-3.3-70B-Instruct", "repo": "meta-llama/Llama-3.3-70B-Instruct",
            "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "base": "manifest-70b.json",
            "parameters": 70553706496, "layers": 80, "kv_heads": 8, "head_dimension": 128},
}
IQ70 = {
    "quant": "IQ1_M", "repo": "mradermacher/Llama-3.3-70B-Instruct-i1-GGUF",
    "revision": "3ebbcaafe45a639733d9deccb4010a6548db6384",
    "filename": "Llama-3.3-70B-Instruct.i1-IQ1_M.gguf", "bytes": 16751201664,
    "sha256": "6816693473ee28e52ad95dcc5ab92385f39abfee4384ab1e2d68d2c17f594f8c",
}
ADDITIONAL_FORMATS = {
    "1b": {"quant": "Q2_K", "repo": "mradermacher/Llama-3.2-1B-Instruct-i1-GGUF",
           "revision": "c61badb55d4d35ee6661a9ea178d9456404237b9",
           "filename": "Llama-3.2-1B-Instruct.i1-Q2_K.gguf", "bytes": 580874784,
           "sha256": "1eac0a34125c730c9ea0a7ed19c92fa7826f270c896ac078badb1f85270292f6"},
    "8b": {"quant": "Q2_K", "repo": "mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF",
           "revision": "15a6b2be4570af9886c5cde164ddc8f41b4d7520",
           "filename": "Meta-Llama-3.1-8B-Instruct.i1-Q2_K.gguf", "bytes": 3179132576,
           "sha256": "8e5021f2452946b02690956b7a4deff791734a47ed2915ab44f3331d889afbb4"},
    "70b": {"quant": "Q8_0", "repo": "mradermacher/Llama-3.3-70B-Instruct-GGUF",
            "revision": "16bef97e072fabd3c01d5f0feff0352841057fff",
            "filename": "Llama-3.3-70B-Instruct.Q8_0.gguf", "bytes": 74975054976,
            "parts": [
                {"filename": "Llama-3.3-70B-Instruct.Q8_0.gguf.part1of2", "bytes": 37580963840,
                 "sha256": "3f73661ab2ab170c127ed9a95552540474a47ee49899302c33c9279073fb6539"},
                {"filename": "Llama-3.3-70B-Instruct.Q8_0.gguf.part2of2", "bytes": 37394091136,
                 "sha256": "6e6dab056f07c8ad8f69834e03fd23b935c38777b68b1bff91eefda2adb9d334"}],
            "provenance": "Byte concatenation of SHA256-pinned upstream parts; assembled GGUF SHA256 recorded after preparation"},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def initial_manifest(model):
    spec = SPECS[model]
    common = {k: spec[k] for k in ("model_family", "parameters", "layers", "kv_heads", "head_dimension")}
    common.update(native_context=131072, model_size=model)
    if model == "8b":
        common["chat_template_file"] = "config/templates/llama-3.1-benchmark.jinja"
    entries = [{**common, **entry} for entry in json.loads((ROOT / "models" / spec["base"]).read_text())]
    fp16 = {**common, "quant": "FP16", "filename": spec["model_family"] + ".F16.gguf",
            "repo": spec["repo"], "revision": spec["revision"],
            "source_url": f"https://huggingface.co/{spec['repo']}/tree/{spec['revision']}",
            "provenance": "Original Meta BF16 safetensors converted to GGUF F16; no low-bit dequantization",
            "full_weight_bytes_estimate": 2 * spec["parameters"], "status": "not_prepared"}
    if model == "70b":
        fp16.update(remote_only=True, status="remote_only",
                    availability_note="Never download on original 2xRTX A6000. Explicit selection and GPU memory screening required on larger hardware.")
        entries.append({**common, **IQ70, "url": f"https://huggingface.co/{IQ70['repo']}/resolve/{IQ70['revision']}/{IQ70['filename']}"})
    addition = {**common, **ADDITIONAL_FORMATS[model]}
    if addition.get("parts"):
        addition["source_url"] = f"https://huggingface.co/{addition['repo']}/tree/{addition['revision']}"
    else:
        addition["url"] = f"https://huggingface.co/{addition['repo']}/resolve/{addition['revision']}/{addition['filename']}"
    entries.append(addition)
    return [fp16, *entries]


def memory_screen(entry, memory_bytes, headroom_gib=2):
    if not memory_bytes:
        raise RuntimeError("No GPU memory detected; cannot authorize this model download from a GPU-residency screen.")
    usable = sum(max(0, total - headroom_gib * GIB) for total in memory_bytes)
    required = entry["full_weight_bytes_estimate"]
    if required > usable:
        raise RuntimeError(f"Download skipped: FP16 weights alone require {required / GIB:.2f} GiB; "
                           f"selected GPUs provide {usable / GIB:.2f} GiB after headroom. "
                           "No 70B FP16 model files were requested.")
    return {"device_bytes": memory_bytes, "usable_bytes": usable, "weight_lower_bound_bytes": required,
            "interpretation": "Necessary weights-only screen; benchmark additionally checks per-GPU weights, KV and workspaces."}


def gpu_memory(devices):
    result = subprocess.check_output(["nvidia-smi", "-i", devices, "--query-gpu=memory.total",
                                      "--format=csv,noheader,nounits"], text=True)
    return [int(line.strip()) * 1024 ** 2 for line in result.splitlines() if line.strip()]


def get_token(token_file, repo):
    # Credentials are used only for the original first-party gated repository.
    if not repo.startswith("meta-llama/"):
        return False
    value = os.environ.get("HF_TOKEN")
    if value:
        return value
    if token_file.is_file():
        if token_file.stat().st_mode & 0o077:
            raise RuntimeError("Token file must not be readable by group or others.")
        return token_file.read_text().strip()
    return None


def fetch(repo, revision, filename, directory, token):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(5):
        try:
            return Path(hf_hub_download(repo, filename, revision=revision, local_dir=directory, token=token))
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in (408, 429, 500, 502, 503, 504) or attempt == 4:
                # Signed CDN URLs and credentials do not belong in acquisition logs.
                raise RuntimeError(f"Download failed for {repo}/{filename}: HTTP {status}") from None
            print(f"Retrying {filename} after HTTP {status} (attempt {attempt + 1}/5)", flush=True)
            time.sleep(min(10 * (attempt + 1), 60))


def tensor_inventory(path):
    sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(path)
    tensors = [{"name": tensor.name, "type": tensor.tensor_type.name,
                "shape": [int(x) for x in tensor.shape], "elements": int(tensor.n_elements),
                "bytes": int(tensor.n_bytes)} for tensor in reader.tensors]
    return {"tensor_count": len(tensors), "tensor_type_counts": dict(Counter(t["type"] for t in tensors)),
            "tensor_bytes": sum(t["bytes"] for t in tensors), "tensors": tensors}


def verify(path, entry):
    if not path.is_file():
        raise RuntimeError(f"Missing model: {path}")
    if entry.get("bytes") is not None and path.stat().st_size != entry["bytes"]:
        raise RuntimeError(f"Model size mismatch: {path}")
    actual = sha256(path)
    if entry.get("sha256") is not None and actual != entry["sha256"]:
        raise RuntimeError(f"Model SHA256 mismatch: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError(f"Model is not GGUF: {path}")
    return actual


def convert_fp16(entry, args, evidence):
    from huggingface_hub import HfApi
    repo, revision = entry["repo"], entry["revision"]
    token = get_token(args.token_file, repo)
    source_dir = args.source_dir / repo.replace("/", "--") / revision
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
    if info.sha != revision:
        raise RuntimeError("Source revision did not resolve to its pinned commit.")
    source_files = []
    source_tensor_types = Counter()
    for sibling in info.siblings:
        name = sibling.rfilename
        if "/" in name or not (name.endswith(".safetensors") or name.endswith(".json")
                               or name in ("tokenizer.model", "LICENSE", "README.md")):
            continue
        path = fetch(repo, revision, name, source_dir, token)
        actual = sha256(path)
        expected = sibling.lfs.sha256 if sibling.lfs else None
        if expected and actual != expected:
            raise RuntimeError(f"Source SHA256 mismatch: {name}")
        if not sibling.lfs:
            contents = path.read_bytes()
            blob_id = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
            if sibling.blob_id != blob_id:
                raise RuntimeError(f"Source Git blob hash mismatch: {name}")
        if name.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt", device="cpu") as tensors:
                for tensor_name in tensors.keys():
                    tensor = tensors.get_slice(tensor_name)
                    dtype = tensor.get_dtype()
                    source_tensor_types[dtype] += 1
                    if len(tensor.get_shape()) >= 2 and dtype not in ("BF16", "F16", "F32"):
                        raise RuntimeError(f"Original source matrix {tensor_name} has unsupported dtype {dtype}.")
        source_files.append({"filename": name, "bytes": path.stat().st_size, "sha256": actual,
                             "expected_lfs_sha256": expected, "git_blob_id": sibling.blob_id})
    evidence.update(source_repo=repo, source_revision=revision, source_files=source_files,
                    source_tensor_type_counts=dict(source_tensor_types), metadata_git_blobs_verified=True)
    write_json(args.log_dir / (entry["model_size"] + "-FP16-source.json"), evidence)
    converter = ROOT / "vendor/llama.cpp/convert_hf_to_gguf.py"
    revision_actual = subprocess.check_output(["git", "-C", str(converter.parent), "rev-parse", "HEAD"], text=True).strip()
    if revision_actual != PIN:
        raise RuntimeError(f"Converter commit must match study pin {PIN}, got {revision_actual}.")
    output = ROOT / "models" / entry["filename"]
    partial = output.with_suffix(".gguf.converting")
    command = [sys.executable, str(converter), str(source_dir), "--outtype", "f16",
               "--outfile", str(partial), "--use-temp-file"]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    # A local conversion does not need authentication, and never loads GPU tensors.
    environment.pop("HF_TOKEN", None)
    evidence.update(converter_commit=revision_actual, converter_sha256=sha256(converter), conversion_command=command)
    with (args.log_dir / (entry["model_size"] + "-FP16-conversion.log")).open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    inventory = tensor_inventory(partial)
    if any(t["type"] != "F16" for t in inventory["tensors"] if len(t["shape"]) >= 2):
        raise RuntimeError("FP16 conversion contains a non-F16 matrix tensor.")
    partial.replace(output)
    return output


def assemble_parts(entry, args, evidence):
    directory = args.source_dir / (entry["model_size"] + "-" + entry["quant"])
    parts = []
    for part in entry["parts"]:
        path = fetch(entry["repo"], entry["revision"], part["filename"], directory, False)
        if path.stat().st_size != part["bytes"] or sha256(path) != part["sha256"]:
            raise RuntimeError(f"Part integrity mismatch: {part['filename']}")
        parts.append(path)
    output = ROOT / "models" / entry["filename"]
    partial = output.with_suffix(".gguf.assembling")
    with partial.open("wb") as target:
        for path in parts:
            with path.open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024 ** 2)
    if partial.stat().st_size != entry["bytes"]:
        raise RuntimeError("Assembled GGUF byte count differs from pinned parts")
    with partial.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError("Concatenated parts do not begin with a GGUF header")
    evidence["assembly"] = {"method": "byte_concatenation_in_manifest_order", "parts": entry["parts"]}
    partial.replace(output)
    return output


def prepare(entry, args):
    path = Path(entry.get("path", ROOT / "models" / entry["filename"]))
    evidence_path = args.log_dir / f"{entry['model_size']}-{entry['quant']}-verified.json"
    evidence = json.loads(evidence_path.read_text()) if path.exists() and evidence_path.exists() else {}
    evidence.update(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), model=entry)
    if entry.get("remote_only") or (entry["model_size"] == "70b" and entry["quant"] == "FP16"):
        if not args.include_remote_only:
            print(f"Skipping {entry['model_size']} FP16: remote-only; explicit --include-remote-only required.", flush=True)
            return entry
        evidence["pre_download_memory_screen"] = memory_screen(entry, gpu_memory(args.devices))
    if not path.exists():
        if entry["quant"] == "FP16":
            path = convert_fp16(entry, args, evidence)
        elif entry.get("parts"):
            available = gpu_memory(args.devices)
            usable = sum(max(0, size - 2 * GIB) for size in available)
            if entry["bytes"] > usable:
                raise RuntimeError("Quantized model download skipped: weights alone exceed selected GPU capacity after headroom")
            evidence["pre_download_memory_screen"] = {"device_bytes": available, "usable_bytes": usable,
                "weight_file_bytes": entry["bytes"], "interpretation": "Necessary total-memory screen; runner checks actual per-device weights, KV and workspaces"}
            print(f"Downloading and assembling pinned {entry['model_size']} {entry['quant']} parts", flush=True)
            path = assemble_parts(entry, args, evidence)
        else:
            print(f"Downloading pinned {entry['model_size']} {entry['quant']}", flush=True)
            path = fetch(entry["repo"], entry["revision"], entry["filename"], path.parent,
                         get_token(args.token_file, entry["repo"]))
    digest = verify(path, entry)
    inventory = tensor_inventory(path)
    evidence.update(sha256=digest, bytes=path.stat().st_size, inventory=inventory,
                    verified_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write_json(evidence_path, evidence)
    print(f"Verified {entry['model_size']} {entry['quant']}: {path.name}, SHA256 {digest}", flush=True)
    try:
        verification_file = str(evidence_path.resolve().relative_to(ROOT))
    except ValueError:
        verification_file = str(evidence_path.resolve())
    return {**entry, "bytes": path.stat().st_size, "sha256": digest, "status": "ready",
            "tensor_type_counts": inventory["tensor_type_counts"], "tensor_bytes": inventory["tensor_bytes"],
            "verification_file": verification_file}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--formats", nargs="+", choices=("FP16", "Q8_0", "Q4_K_M", "Q2_K", "IQ1_M"))
    parser.add_argument("--manifest-only", action="store_true", help="Write initial manifests without model network requests")
    parser.add_argument("--include-remote-only", action="store_true")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".hf_token")
    parser.add_argument("--source-dir", type=Path, default=ROOT / ".run/model-sources")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results/model-preparation")
    args = parser.parse_args()
    if args.formats and not any(entry["quant"] in args.formats for model in args.models for entry in initial_manifest(model)):
        parser.error("None of the selected formats are included for the selected models.")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        manifest_path = ROOT / "models" / f"study-manifest-{model}.json"
        entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else initial_manifest(model)
        present = {entry["quant"] for entry in entries}
        entries.extend(entry for entry in initial_manifest(model) if entry["quant"] not in present)
        if not args.manifest_only:
            for index, entry in enumerate(entries):
                if args.formats and entry["quant"] not in args.formats:
                    continue
                entries[index] = prepare(entry, args)
                write_json(manifest_path, entries)
        write_json(manifest_path, entries)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
