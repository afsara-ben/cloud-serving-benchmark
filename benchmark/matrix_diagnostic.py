#!/usr/bin/env python3
"""Build and run isolated complete-matmul diagnostics without modifying vendor.

All GPU execution is explicit through ``run``. ``build`` only compiles; ``plan``
and ``inventory`` are read-only. Results retain commands and build/source hashes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / ".run/context-diagnostic-build"
SHAPES = {
    "ffn_gate_up": (28672, 8192),
    "ffn_down": (8192, 28672),
    "attn_q_output": (8192, 8192),
    "attn_k_v": (1024, 8192),
    "output": (128256, 8192),
}
TYPES = {"Q2_K": "q2_K", "Q3_K": "q3_K", "Q4_K": "q4_K", "Q5_K": "q5_K",
         "Q6_K": "q6_K", "Q8_0": "q8_0", "IQ1_M": "iq1_m", "FP16": "f16", "F16": "f16"}

NVTX_INSERT = r'''
#include <nvtx3/nvToolsExt.h>

// Private diagnostic annotation; no tensor values or dispatch choices change.
struct csb_operation_range {
    bool enabled;
    explicit csb_operation_range(ggml_tensor * dst, bool unknown_fusion = false) {
        static const bool annotate = std::getenv("CSB_NVTX_OPS") != nullptr;
        enabled = annotate;
        if (!enabled) return;
        const ggml_tensor * a = dst->src[0];
        const ggml_tensor * b = dst->src[1];
        char label[768];
        if (unknown_fusion) {
            std::snprintf(label, sizeof(label),
                          "csb_op:role=unknown_fusion:source_role=%s:type=unknown:fusion=unclassified:geometry=unknown:phase=unknown", dst->name);
        } else if (dst->op == GGML_OP_MUL_MAT && a && b) {
            std::snprintf(label, sizeof(label), "csb_op:role=%s:type=%s:m=%lld:n=%lld:k=%lld:phase=unknown",
                          a->name, ggml_type_name(a->type), (long long) a->ne[1],
                          (long long) b->ne[1], (long long) a->ne[0]);
        } else {
            std::snprintf(label, sizeof(label), "csb_op:role=%s:type=%s:op=%s:phase=unknown",
                          dst->name, a ? ggml_type_name(a->type) : "none", ggml_op_name(dst->op));
        }
        nvtxRangePushA(label);
    }
    ~csb_operation_range() { if (enabled) nvtxRangePop(); }
};

struct csb_fused_matrix_range {
    bool enabled;
    csb_fused_matrix_range(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * ids,
                          const ggml_cuda_mm_fusion_args_host & fusion) {
        static const bool annotate = std::getenv("CSB_NVTX_OPS") != nullptr;
        enabled = annotate;
        if (!enabled) return;
        const ggml_tensor * gate = fusion.gate;
        const char * kind = gate ? "gate_up_glu" : fusion.x_scale ? "matvec_scale" : "matvec_bias";
        const bool dense_same_geometry = !ids && (!gate || ggml_are_same_shape(a, gate));
        char geometry[256];
        if (dense_same_geometry) {
            std::snprintf(geometry, sizeof(geometry), "m=%lld:n=%lld:k=%lld:geometry=per_projection",
                          (long long) a->ne[1], (long long) b->ne[1], (long long) a->ne[0]);
        } else {
            std::snprintf(geometry, sizeof(geometry), "geometry=unknown");
        }
        char label[1536];
        std::snprintf(label, sizeof(label),
                      "csb_op:role=%s%s%s:type=%s%s%s:%s:fusion=%s:up_role=%s:gate_role=%s:gate_type=%s:matrix_count=%d:bias=%d:scale=%d:glu_op=%d:phase=unknown",
                      a->name, gate ? "+" : "", gate ? gate->name : "",
                      ggml_type_name(a->type), gate ? "+" : "", gate ? ggml_type_name(gate->type) : "",
                      geometry, kind, a->name, gate ? gate->name : "none", gate ? ggml_type_name(gate->type) : "none", gate ? 2 : 1,
                      int(fusion.x_bias != nullptr || fusion.gate_bias != nullptr),
                      int(fusion.x_scale != nullptr || fusion.gate_scale != nullptr),
                      gate ? int(fusion.glu_op) : -1);
        nvtxRangePushA(label);
    }
    ~csb_fused_matrix_range() { if (enabled) nvtxRangePop(); }
};
'''
EARLY_INSERT = '''
    // Controlled diagnostic: only Q2/Q3 at exactly eight columns on NVIDIA.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ne11 == 8 &&
        (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K)) {
        return false;
    }
'''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_if_changed(path, content):
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def execute(command, *, env=None, log=None, cwd=None):
    print(" ".join(map(str, command)), flush=True)
    if log:
        with Path(log).open("w") as handle:
            result = subprocess.run(list(map(str, command)), env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command exited {result.returncode}; see {log}")
        return result
    return subprocess.run(list(map(str, command)), env=env, cwd=cwd, check=True)


def gpu_snapshot(device):
    fields = "index,uuid,name,driver_version,clocks.current.sm,clocks.current.memory,memory.used,power.draw,temperature.gpu,pstate"
    result = subprocess.run(["nvidia-smi", "-i", device, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        return {"available": False, "diagnostic": result.stderr}
    records = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    return {"available": True, "devices": [dict(zip(fields.split(","), row)) for row in records],
            "units": "clocks MHz, memory MiB, power W, temperature C"}


def nvtx_include(explicit):
    candidates = [Path(explicit)] if explicit else []
    for prefix in [Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")), Path.home() / ".cuda-13.0"]:
        candidates += [prefix / "include", prefix / "targets/x86_64-linux/include"]
        candidates += list(prefix.glob("nsight-compute-*/host/target-linux-x64/nvtx/include"))
    candidates += [Path("/usr/include")]
    for path in candidates:
        if (path / "nvtx3/nvToolsExt.h").is_file():
            return path.resolve()
    raise RuntimeError("NVTX3 headers missing; pass --nvtx-include containing nvtx3/nvToolsExt.h")


def compile_harness(source, libraries, binary, nvtx, log):
    execute([os.environ.get("CXX", "c++"), "-O3", "-std=c++17", "-pthread", "-Wall", "-Wextra",
             f"-I{source / 'ggml/include'}", f"-I{nvtx}", ROOT / "benchmark/matrix_diagnostic.cpp",
             f"-L{libraries}", f"-Wl,-rpath,{libraries}", "-lggml", "-lggml-base", "-ldl", "-o", binary], log=log)


def annotated_cuda_source(original):
    signature = "static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {"
    loop = "                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);"
    fuse = "static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {"
    if any(original.count(marker) != 1 for marker in (signature, loop, fuse)):
        raise RuntimeError("Pinned CUDA annotation insertion points changed")
    annotated = original.replace(signature, NVTX_INSERT + "\n" + signature)
    annotated = annotated.replace(loop, "                csb_operation_range csb_range(node);\n" + loop)
    annotated = annotated.replace(fuse, fuse + "\n    csb_operation_range csb_unknown_fusion(cgraph->nodes[i], true);")
    pattern = r"^( +)(ggml_cuda_mul_mat_vec_[fq]\(\*cuda_ctx, src0, src1, ids, [^\n]+, &fusion_data\);)$"
    annotated, count = re.subn(pattern, r"\1csb_fused_matrix_range csb_fused_range(src0, src1, ids, fusion_data);\n\1\2",
                               annotated, flags=re.MULTILINE)
    if count != 9:
        raise RuntimeError(f"Expected nine pinned fused matvec call sites, received {count}")
    return annotated


def stage_instrumentation(args):
    """Stage source and a reviewable diff; do not compile or change existing binaries."""
    import difflib
    destination = args.build_root.resolve()
    provenance = json.loads((destination / "source.json").read_text())
    relative = "ggml/src/ggml-cuda/ggml-cuda.cu"
    original = subprocess.check_output(["git", "-C", provenance["vendor"], "show", f"{provenance['commit']}:{relative}"], text=True)
    updated = annotated_cuda_source(original)
    target = destination / "source" / relative
    write_if_changed(target, updated)
    difference = "".join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                           fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    diff_path = destination / "instrumentation-staged.diff"
    diff_path.write_text(difference)
    pending = {"status": "staged_not_built", "annotation_version": 2, "commit": provenance["commit"],
               "source_sha256": sha(target), "diff_sha256": sha(diff_path),
               "harness_sha256": sha(ROOT / "benchmark/matrix_diagnostic.cpp"),
               "wrapper_sha256": sha(Path(__file__)), "diagnostic_protocol_version": 2,
               "warmup_defaults": {"minimum_iterations": 3, "minimum_ms": 250.0,
                                   "position": "after_correctness_before_capture"},
               "build_command": [sys.executable, str(ROOT / "benchmark/matrix_diagnostic.py"), "build",
                                 "--build-root", str(destination), "--jobs", str(args.jobs), "--server"],
               "changes": "Graph-loop ranges cover fused and normal operations; fused matvec roles/types are explicit; diagnostic warmup runs for at least 250 ms and 3 operations immediately before capture after CPU correctness work"}
    (destination / "pending-instrumentation.json").write_text(json.dumps(pending, indent=2) + "\n")
    print(json.dumps(pending))


def build(args):
    destination = args.build_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    vendor = args.vendor.resolve()
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    nvtx = nvtx_include(args.nvtx_include)
    source = destination / "source"
    provenance = destination / "source.json"
    if not provenance.exists():
        if source.exists():
            raise RuntimeError("Unidentified private source exists; choose another --build-root")
        source.mkdir()
        archive = destination / "source.tar"
        with archive.open("wb") as handle:
            subprocess.run(["git", "-C", str(vendor), "archive", commit], stdout=handle, check=True)
        with tarfile.open(archive) as handle:
            handle.extractall(source, filter="data")
        archive.unlink()
        provenance.write_text(json.dumps({"vendor": str(vendor), "commit": commit,
                                         "harness_sha256": sha(ROOT / "benchmark/matrix_diagnostic.cpp")}, indent=2) + "\n")
    if json.loads(provenance.read_text())["commit"] != commit:
        raise RuntimeError("Private source belongs to another commit; choose another --build-root")
    cuda_file = source / "ggml/src/ggml-cuda/ggml-cuda.cu"
    mmvq_file = source / "ggml/src/ggml-cuda/mmvq.cu"
    originals = {}
    for relative in ["ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-cuda/mmvq.cu"]:
        originals[relative] = subprocess.check_output(["git", "-C", str(vendor), "show", f"{commit}:{relative}"], text=True)
    write_if_changed(cuda_file, annotated_cuda_source(originals["ggml/src/ggml-cuda/ggml-cuda.cu"]))
    baseline_mmvq = originals["ggml/src/ggml-cuda/mmvq.cu"]
    write_if_changed(mmvq_file, baseline_mmvq)
    cmake_build = destination / "cmake-build"
    command = ["cmake", "-S", str(source), "-B", str(cmake_build), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
               "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=ON",
               "-DLLAMA_BUILD_EXAMPLES=OFF", f"-DCMAKE_CUDA_FLAGS=-I{nvtx}"]
    if args.nvcc:
        command += [f"-DCMAKE_CUDA_COMPILER={Path(args.nvcc).resolve()}"]
    execute(command, log=destination / "configure.log")
    import difflib
    for variant in ["baseline", "early-mmq"]:
        if variant == "early-mmq":
            signature = "bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {"
            if baseline_mmvq.count(signature) != 1:
                raise RuntimeError("Pinned dispatch insertion point changed")
            write_if_changed(mmvq_file, baseline_mmvq.replace(signature, signature + EARLY_INSERT))
        execute(["cmake", "--build", cmake_build, "--parallel", str(args.jobs), "--target", "ggml",
                 *(["llama-server"] if args.server and variant == "baseline" else [])], log=destination / f"build-{variant}.log")
        output = destination / variant
        output.mkdir(exist_ok=True)
        for path in (cmake_build / "bin").iterdir():
            if path.is_file() and (".so" in path.name or (args.server and variant == "baseline" and path.name == "llama-server")):
                shutil.copy2(path, output / path.name, follow_symlinks=True)
        binary = output / "matrix-diagnostic"
        compile_harness(source, output, binary, nvtx, destination / f"harness-{variant}.log")
        difference = ""
        for relative, original in originals.items():
            difference += "".join(difflib.unified_diff(original.splitlines(True), (source / relative).read_text().splitlines(True),
                                                       fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        (output / "build.diff").write_text(difference)
        metadata = {"variant": variant, "commit": commit, "configure_command": command,
                    "annotation_version": 2,
                    "diagnostic_protocol_version": 2,
                    "warmup_defaults": {"minimum_iterations": 3, "minimum_ms": 250.0,
                                        "position": "after_correctness_before_capture"},
                    "embedded_version_note": "Archived source has no .git; the explicit commit and build diff are authoritative, not embedded version strings",
                    "harness_sha256": sha(ROOT / "benchmark/matrix_diagnostic.cpp"),
                    "build_diff_sha256": sha(output / "build.diff"),
                    "binary_sha256": sha(binary), "cuda_library_sha256": sha(output / "libggml-cuda.so"),
                    "annotation": "CSB_NVTX_OPS enables CPU launch ranges; disable CUDA graphs for full serving attribution",
                    "intervention": "Q2_K/Q3_K with exactly eight activation columns return false from should_use_mmvq on NVIDIA" if variant == "early-mmq" else "None; annotation only",
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (output / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "compiled", "root": str(destination)}))
    if args.server:
        annotate_server(args)
    (destination / "pending-instrumentation.json").unlink(missing_ok=True)


def annotate_server(args):
    """Add exact server-batch phase ranges without changing baseline CUDA libraries."""
    destination = args.build_root.resolve()
    source = destination / "source"
    provenance = json.loads((destination / "source.json").read_text())
    nvtx = nvtx_include(args.nvtx_include)
    relative = "tools/server/server-context.cpp"
    original = subprocess.check_output(["git", "-C", provenance["vendor"], "show", f"{provenance['commit']}:{relative}"], text=True)
    old = '''        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });'''
    new = '''        queue_tasks.yield_to_queue([&]() {
            static const bool csb_annotate = std::getenv("CSB_NVTX_OPS") != nullptr;
            if (csb_annotate) {
                int csb_prefill = 0;
                for (int i = off; i < off + batch_view.n_tokens; ++i) {
                    csb_prefill += batch.tokens[i].is_prompt;
                }
                const int csb_decode = batch_view.n_tokens - csb_prefill;
                const char * csb_phase = csb_prefill == 0 ? "decode" : csb_decode == 0 ? "prefill" : "mixed";
                char csb_label[256];
                std::snprintf(csb_label, sizeof(csb_label),
                              "csb_batch:phase=%s:prefill_tokens=%d:decode_tokens=%d:width=%d",
                              csb_phase, csb_prefill, csb_decode, batch_view.n_tokens);
                nvtxRangePushA(csb_label);
            }
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
            if (csb_annotate) nvtxRangePop();
        });'''
    if original.count(old) != 1:
        raise RuntimeError("Pinned server batch insertion point changed")
    updated = f'#include "{nvtx}/nvtx3/nvToolsExt.h"\n#include <cstdio>\n#include <cstdlib>\n' + original.replace(old, new)
    write_if_changed(source / relative, updated)
    execute(["cmake", "--build", destination / "cmake-build", "--parallel", str(args.jobs), "--target", "llama-server"],
            log=destination / "build-server-phase.log")
    baseline = destination / "baseline"
    for name in ["llama-server", "libllama-server-impl.so"]:
        shutil.copy2(destination / "cmake-build/bin" / name, baseline / name)
    import difflib
    difference = "".join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                           fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    (baseline / "server-phase.diff").write_text(difference)
    metadata_path = baseline / "build.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(server_phase_diff_sha256=sha(baseline / "server-phase.diff"),
                    server_sha256=sha(baseline / "llama-server"),
                    server_impl_sha256=sha(baseline / "libllama-server-impl.so"),
                    server_phase="Exact is_prompt counts for each llama_decode batch view; mixed batches explicitly labeled",
                    embedded_version_note="Archived source has no .git; use explicit commit and build diff rather than embedded version strings")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "phase_annotated_server_compiled", "binary": str(baseline / "llama-server"),
                      "required_environment": {"CSB_NVTX_OPS": "1", "GGML_CUDA_DISABLE_GRAPHS": "1", "LD_LIBRARY_PATH": str(baseline)}}))


def inventory(args):
    sys.path.insert(0, str(args.vendor / "gguf-py"))
    from gguf import GGUFReader
    import collections
    records = []
    for model in args.models:
        reader = GGUFReader(str(model), "r")
        counts = collections.Counter((tensor.name.split(".")[-2], tuple(int(x) for x in tensor.shape), tensor.tensor_type.name)
                                     for tensor in reader.tensors if len(tensor.shape) == 2 and
                                     (tensor.name.startswith("blk.") or tensor.name == "output.weight"))
        records.append({"model": str(model.resolve()), "operations": [
            {"role": role, "k": dimensions[0], "m": dimensions[1], "type": kind, "tensor_count": count}
            for (role, dimensions, kind), count in sorted(counts.items())]})
    print(json.dumps(records, indent=2))


def cases(args):
    selected_shapes = args.shapes.split(",")
    for variant in args.variants.split(","):
        if variant not in ("baseline", "early-mmq"):
            raise ValueError(f"Unknown build variant {variant}")
        for shape in selected_shapes:
            if shape not in SHAPES:
                raise ValueError(f"Unknown shape {shape}")
            m, k = SHAPES[shape]
            for kind in args.types.split(","):
                kind = kind.upper()
                if kind not in TYPES:
                    raise ValueError(f"Unsupported type {kind}")
                for width in map(int, args.widths.split(",")):
                    if width < 1:
                        raise ValueError("Activation widths must be positive")
                    if variant == "early-mmq" and (kind not in ("Q2_K", "Q3_K") or width != 8):
                        continue
                    yield {"variant": variant, "shape": shape, "m": m, "n": width, "k": k,
                           "type": kind, "ggml_type": TYPES[kind], "phase": args.phase}


def capture_groups(args):
    from profile_cuda import METRIC_GROUPS, combine_metric_groups
    if args.tool != "ncu":
        return [{"name": args.tool, "metric_groups": []}]
    requested = set(filter(None, (name.strip() for name in args.metric_groups.split(","))))
    if not requested or requested - METRIC_GROUPS.keys():
        raise ValueError("Select at least one known metric group: " + ",".join(METRIC_GROUPS))
    ordered = [name for name in METRIC_GROUPS if name in requested]
    selected = [[name] for name in ordered] if args.separate_metric_groups else [ordered]
    return [{"name": names[0] if len(names) == 1 else "combined", "metric_groups": names,
             **combine_metric_groups(names)} for names in selected]


def capture_folder(output, case, repetition, group, gpu):
    if not gpu or "," in gpu:
        raise ValueError("Each diagnostic process requires one GPU selector")
    label = group["name"]
    if label == "combined":
        label += "-" + "-".join(group["metric_groups"])
    return output / f"gpu{quote(gpu, safe='')}/{case['variant']}/{case['shape']}/{case['type']}/n{case['n']}/r{repetition}/{label}"


def validate_operation(operation, identity):
    errors = []
    correctness = operation.get("correctness", {})
    if correctness.get("passed") is not True:
        errors.append("Quantized CPU reference check did not pass")
    for value_key, tolerance_key in [("relative_l2", "relative_l2_tolerance"),
                                     ("max_error_over_reference_rms", "maximum_normalized_tolerance")]:
        value, tolerance = correctness.get(value_key), correctness.get(tolerance_key)
        if not all(isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
                   for number in (value, tolerance)) or not 0 <= value <= tolerance:
            errors.append(f"Reference error {value_key} is unavailable or exceeds tolerance")
    expected = {key: identity[key] for key in ["m", "n", "k", "seed", "phase"]}
    expected.update(type=identity["ggml_type"], role=identity["shape"], backend="CUDA0")
    if any(operation.get(key) != value for key, value in expected.items()):
        errors.append("Operation geometry, input identity or backend differs from the requested case")
    if (operation.get("schema") != 2 or operation.get("warmup") != identity["warmup_iterations"]
        or operation.get("warmup_ms") != identity["warmup_ms"]
        or operation.get("warmup_position") != "after_correctness_before_capture"):
        errors.append("Diagnostic warmup protocol differs from the requested case")
    count, elapsed = operation.get("warmup_actual_iterations"), operation.get("warmup_actual_ms")
    if not isinstance(count, int) or isinstance(count, bool) or count < identity["warmup_iterations"]:
        errors.append("Warmup did not execute the requested minimum operation count")
    if (not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed)
        or elapsed < identity["warmup_ms"]):
        errors.append("Warmup did not sustain operations for the requested minimum time")
    durations = operation.get("complete_operation_us", [])
    if len(durations) != identity["iterations"] or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                                                       or not math.isfinite(value) or value <= 0 for value in durations):
        errors.append("Complete-operation timing lacks the requested number of positive finite samples")
    return errors


def validate_matrix_counters(summary, group, iterations):
    from profile_cuda import validate_counter_capture
    validation = validate_counter_capture(summary, group["metrics"], group["sections"],
                                          required_launches=iterations, expected_devices=["0"])
    counts = collections.Counter((launch.get("kernel"), launch.get("grid"), launch.get("block"), launch.get("device"))
                                 for launch in summary.get("launches", []))
    cohorts = [{"kernel": key[0], "grid": key[1], "block": key[2], "device": key[3], "launches": count}
               for key, count in counts.items()]
    errors = list(validation.get("errors", []))
    incomplete = [row for row in cohorts if row["launches"] < iterations or row["launches"] % iterations]
    if incomplete:
        errors.append("Kernel signature counts do not cover every repeated complete operation")
    matrices = [launch for launch in summary.get("launches", [])
                if launch.get("class") in ("quantized_matvec", "quantized_matmul", "other_matrix_multiply")]
    if len(matrices) < iterations:
        errors.append("No matrix-kernel cohort covers the requested operation repetitions")
    return validation | {"valid": validation.get("valid") is True and not errors, "errors": errors,
                         "operation_iterations": iterations, "kernel_cohorts": cohorts,
                         "incomplete_kernel_cohorts": incomplete}


def run(args):
    from profile_cuda import summarize_ncu, summarize_sqlite, find_ncu_report
    if min(args.repetitions, args.iterations, args.profile_iterations, args.warmup_iterations) < 1:
        raise ValueError("Repetition and iteration counts must be positive")
    if not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        raise ValueError("Warmup milliseconds must be finite and nonnegative")
    groups = capture_groups(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.build_root.resolve() / "weight-cache" / sha(ROOT / "benchmark/matrix_diagnostic.cpp")[:16]
    cache.mkdir(parents=True, exist_ok=True)
    for case in cases(args):
        variant = args.build_root.resolve() / case["variant"]
        build_metadata = json.loads((variant / "build.json").read_text())
        if build_metadata.get("harness_sha256") != sha(ROOT / "benchmark/matrix_diagnostic.cpp"):
            raise RuntimeError("Diagnostic harness source changed; rebuild the private binaries before GPU execution")
        if sha(variant / "libggml-cuda.so") != build_metadata["cuda_library_sha256"]:
            raise RuntimeError("CUDA library differs from build provenance")
        if sha(variant / "matrix-diagnostic") != build_metadata["binary_sha256"]:
            raise RuntimeError("Diagnostic executable differs from build provenance")
        for repetition in range(args.repetitions):
            for group in groups:
                identity = dict(case, repetition=repetition, seed=args.seed, tool=args.tool, metric_group=group["name"],
                                metric_groups=group["metric_groups"],
                                iterations=args.iterations if args.tool == "timing" else args.profile_iterations,
                                warmup_iterations=args.warmup_iterations, warmup_ms=args.warmup_ms,
                                gpu=args.gpu, cache_control=args.cache_control, clock_control="none", build=build_metadata)
                folder = capture_folder(output, case, repetition, group, args.gpu)
                folder.mkdir(parents=True, exist_ok=True)
                metadata_file = folder / "run.json"
                if (folder / "complete.json").is_file():
                    existing = json.loads(metadata_file.read_text())["identity"]
                    if existing != identity:
                        raise RuntimeError(f"Resume identity differs: {folder}")
                    previous_operation = json.loads((folder / "operation.json").read_text())
                    errors = validate_operation(previous_operation, identity)
                    if args.tool == "ncu":
                        previous_summary = json.loads((folder / "counter_summary.json").read_text())
                        previous_validation = validate_matrix_counters(previous_summary, group, identity["iterations"])
                        if not previous_validation["valid"]:
                            errors += previous_validation["errors"]
                    if errors:
                        stale = json.loads(metadata_file.read_text())
                        stale.update(status="invalidated", errors=errors)
                        if args.tool == "ncu":
                            stale["counter_validation"] = previous_validation
                            (folder / "counter-validation.json").write_text(json.dumps(previous_validation, indent=2) + "\n")
                        metadata_file.write_text(json.dumps(stale, indent=2) + "\n")
                        (folder / "complete.json").replace(folder / "invalidated-complete.json")
                        raise RuntimeError(f"Completed capture failed resume validation: {folder}: {'; '.join(errors)}")
                    continue
                matrix = [str(variant / "matrix-diagnostic"), "--library-dir", str(variant), "--backend", "CUDA0",
                          "--type", case["ggml_type"], "--m", str(case["m"]), "--n", str(case["n"]), "--k", str(case["k"]),
                          "--role", case["shape"], "--phase", case["phase"], "--seed", str(args.seed), "--iterations", str(identity["iterations"]),
                          "--warmup", str(args.warmup_iterations), "--warmup-ms", str(args.warmup_ms),
                          "--weights-cache", str(cache / f"{case['type']}-m{case['m']}-k{case['k']}-seed{args.seed}.bin"),
                          "--output", str(folder / "operation.json")]
                env = os.environ | {"CUDA_VISIBLE_DEVICES": args.gpu, "GGML_CUDA_DISABLE_GRAPHS": "1",
                                    "LD_LIBRARY_PATH": str(variant) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
                env.pop("CSB_NVTX_OPS", None)  # The harness itself labels the complete operation.
                report = folder / "capture"
                if args.tool == "timing":
                    command = matrix
                elif args.tool == "nsys":
                    command = [args.nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
                               f"--output={report}", *matrix]
                else:
                    command = [args.ncu, "--profile-from-start", "off", "--replay-mode", "kernel",
                               "--clock-control", "none", "--cache-control", args.cache_control,
                               "--kernel-name-base", "demangled", "--rename-kernels", "off", "--metrics", group["metrics"], "--nvtx",
                               "--csv", "--page", "raw", "--print-units", "base", "--print-metric-instances", "details",
                               "--log-file", str(folder / "ncu.csv"),
                               "--export", str(report), "--force-overwrite"]
                    for section in group["sections"]:
                        command += ["--section", section]
                    command += matrix
                run_metadata = {"identity": identity, "command": command, "matrix_command": matrix,
                                "status": "running",
                                "environment": {k: env.get(k) for k in ["CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS", "LD_LIBRARY_PATH"]},
                                "timing_use": "Isolated diagnostic; never substitute for serving runtime", "gpu_before": gpu_snapshot(args.gpu)}
                metadata_file.write_text(json.dumps(run_metadata, indent=2) + "\n")
                try:
                    execute(command, env=env, log=folder / "execution.log")
                except RuntimeError as error:
                    run_metadata.update(status="failed", errors=[str(error)])
                    if args.tool == "ncu":
                        run_metadata["counter_validation"] = {"valid": False, "errors": [str(error)],
                                                               "diagnostics": ["execution.log", "ncu.csv"]}
                        (folder / "counter-validation.json").write_text(json.dumps(run_metadata["counter_validation"], indent=2) + "\n")
                    metadata_file.write_text(json.dumps(run_metadata, indent=2) + "\n")
                    raise
                run_metadata["gpu_after"] = gpu_snapshot(args.gpu)
                metadata_file.write_text(json.dumps(run_metadata, indent=2) + "\n")
                operation = json.loads((folder / "operation.json").read_text())
                errors = validate_operation(operation, identity)
                if errors:
                    run_metadata.update(status="failed", errors=errors)
                    metadata_file.write_text(json.dumps(run_metadata, indent=2) + "\n")
                    raise RuntimeError(f"Operation validation failed: {folder}: {'; '.join(errors)}")
                completion = {"correctness": operation["correctness"],
                              "warmup_actual_iterations": operation["warmup_actual_iterations"],
                              "warmup_actual_ms": operation["warmup_actual_ms"],
                              "median_operation_us": statistics.median(operation["complete_operation_us"])}
                if args.tool == "nsys":
                    execute([args.nsys, "export", "--type=sqlite", "--force-overwrite=true", f"--output={report}.sqlite", f"{report}.nsys-rep"],
                            env=env, log=folder / "export.log")
                    summarize_sqlite(report.with_suffix(".sqlite"), folder)
                elif args.tool == "ncu":
                    try:
                        if find_ncu_report(report) is None:
                            raise RuntimeError("Missing hardware counter report")
                        summary = summarize_ncu(folder / "ncu.csv", folder)
                        validation = validate_matrix_counters(summary, group, identity["iterations"])
                    except (OSError, ValueError, RuntimeError) as error:
                        validation = {"valid": False, "errors": [str(error)]}
                    (folder / "counter-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
                    run_metadata["counter_validation"] = validation
                    if not validation["valid"]:
                        run_metadata.update(status="failed", errors=validation["errors"])
                        metadata_file.write_text(json.dumps(run_metadata, indent=2) + "\n")
                        raise RuntimeError(f"Counter validation failed: {folder}: {'; '.join(validation['errors'])}")
                    completion["counter_validation"] = validation
                run_metadata["status"] = "complete"
                metadata_file.write_text(json.dumps(run_metadata, indent=2) + "\n")
                (folder / "complete.json").write_text(json.dumps(completion, indent=2) + "\n")
    summarize(output)


def summarize(output):
    rows = []
    for path in sorted(output.rglob("complete.json")):
        folder = path.parent
        metadata = json.loads((folder / "run.json").read_text())["identity"]
        complete = json.loads(path.read_text())
        rows.append({**{k: metadata[k] for k in ["variant", "shape", "type", "m", "n", "k", "phase", "repetition", "tool", "metric_group", "gpu"]},
                     "metric_groups": ";".join(metadata.get("metric_groups", [])),
                     "warmup_minimum_iterations": metadata.get("warmup_iterations"), "warmup_minimum_ms": metadata.get("warmup_ms"),
                     "warmup_actual_iterations": complete.get("warmup_actual_iterations"), "warmup_actual_ms": complete.get("warmup_actual_ms"),
                     "median_operation_us": complete["median_operation_us"], "reference_relative_l2": complete["correctness"]["relative_l2"],
                     "artifact": str(folder.relative_to(output))})
    if rows:
        with (output / "operations.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"completed_cases": len(rows), "summary": str(output / "operations.csv")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("build", help="Compile private baseline and early-MMQ libraries/harness; no GPU execution")
    make.add_argument("--build-root", type=Path, default=DEFAULT_BUILD)
    make.add_argument("--vendor", type=Path, default=ROOT / "vendor/llama.cpp")
    make.add_argument("--nvtx-include")
    make.add_argument("--nvcc", default=shutil.which("nvcc"))
    make.add_argument("--cuda-arch", default="86")
    make.add_argument("--jobs", type=int, default=4)
    make.add_argument("--server", action="store_true", help="Also compile the annotated baseline llama-server")
    annotate = sub.add_parser("annotate-server", help="Add exact prefill/decode/mixed batch NVTX ranges to an existing private server build")
    annotate.add_argument("--build-root", type=Path, default=DEFAULT_BUILD)
    annotate.add_argument("--nvtx-include")
    annotate.add_argument("--jobs", type=int, default=4)
    stage = sub.add_parser("stage-instrumentation", help="Stage private CUDA NVTX source changes and diff without compiling")
    stage.add_argument("--build-root", type=Path, default=DEFAULT_BUILD)
    stage.add_argument("--jobs", type=int, default=12, help="Deferred build command parallelism; this command performs no compilation")
    meta = sub.add_parser("inventory", help="Read matrix roles, exact dimensions and tensor types from GGUF headers")
    meta.add_argument("models", type=Path, nargs="+")
    meta.add_argument("--vendor", type=Path, default=ROOT / "vendor/llama.cpp")
    for command in ["plan", "run"]:
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--build-root", type=Path, default=DEFAULT_BUILD)
        command_parser.add_argument("--shapes", default="ffn_gate_up,ffn_down,attn_q_output,attn_k_v")
        command_parser.add_argument("--types", default="Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,IQ1_M,FP16")
        command_parser.add_argument("--widths", default="1,8,9,16,32,64")
        command_parser.add_argument("--variants", default="baseline,early-mmq")
        command_parser.add_argument("--phase", choices=["decode", "prefill", "unknown"], default="decode")
        command_parser.add_argument("--output", type=Path, default=ROOT / "results/cuda-context-study/diagnostics")
        command_parser.add_argument("--gpu", default="0")
        command_parser.add_argument("--repetitions", type=int, default=3)
        command_parser.add_argument("--iterations", type=int, default=20)
        command_parser.add_argument("--profile-iterations", type=int, default=5)
        command_parser.add_argument("--warmup-iterations", type=int, default=3,
                                    help="Minimum synchronized operations immediately before timing/capture")
        command_parser.add_argument("--warmup-ms", type=float, default=250.0,
                                    help="Minimum active warmup milliseconds after CPU correctness work; excluded from measured results")
        command_parser.add_argument("--seed", type=int, default=20260912)
        command_parser.add_argument("--tool", choices=["timing", "nsys", "ncu"], default="timing")
        command_parser.add_argument("--metric-groups", default="memory,instructions,occupancy,stalls,tensor")
        command_parser.add_argument("--separate-metric-groups", action="store_true",
                                    help="Capture groups separately when the installed profiler cannot collect their combined union")
        command_parser.add_argument("--cache-control", choices=["all", "none"], default="all")
        command_parser.add_argument("--ncu", default=shutil.which("ncu") or "ncu")
        command_parser.add_argument("--nsys", default=os.environ.get("NSYS_BIN", shutil.which("nsys") or "nsys"))
    summary = sub.add_parser("summarize")
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        build(args)
    elif args.action == "annotate-server":
        annotate_server(args)
    elif args.action == "stage-instrumentation":
        stage_instrumentation(args)
    elif args.action == "inventory":
        inventory(args)
    elif args.action == "plan":
        print(json.dumps(list(cases(args)), indent=2))
    elif args.action == "summarize":
        summarize(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
