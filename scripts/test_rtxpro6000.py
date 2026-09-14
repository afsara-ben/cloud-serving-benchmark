"""CPU-only RTX workflow checks using explicitly synthetic temporary evidence."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_a100_roofline as operators
import cuda_hardware
import export_rtxpro6000_results as exporter
import profile_a100_setting as profile
import profile_rtxpro6000 as pipeline
import report_context_study as context
import report_serving as serving
import rtxpro6000_publication as publication
import run_a100_serving as launcher
from context_protocol import fingerprint
from test_report_context_study import fixture, put

GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


def make_study(base, prompts=(2048, 32768), formats=None):
    root = base / "results/cuda-context-study-rtx-test"
    root.mkdir(parents=True)
    model_root = root.parent / (root.name + "-70b")
    model_root.mkdir()
    (root / "70b").symlink_to(Path("..") / model_root.name)
    scope = serving.scope_document(["0", "1"], "rtxpro6000")
    formats = list(formats or serving.FORMATS["70b"])
    scope["model_formats"] = {"70b": formats}
    put(root / "serving-scope.json", scope)
    folders = [fixture(root, p, q, "70b") for q in formats for p in prompts]
    models = [{"quant": q, "sha256": "synthetic-weights", "layers": 80,
               "model_family": "Llama-3.3-70B-Instruct"} for q in formats]
    manifest = {"context_study": True, "llama_cpp_commit": context.PIN, "binary_sha256": "synthetic-binary",
                "code_sha256": {"runner": "synthetic-code"}, "models": models, "repetitions": 1,
                "devices": [{"index": str(gpu), "name": GPU, "uuid": f"synthetic-{gpu}", "total_bytes": 96 * context.GIB} for gpu in (0, 1)],
                "concurrencies": scope["concurrencies"], "prompt_lengths": scope["prompt_lengths"],
                "capacity_lengths": scope["capacity_lengths"], "extension_request_waves": 2}
    put(model_root / "manifest.json", manifest)
    identity = fingerprint(manifest)
    progress = {"fingerprint": identity, "cells": {}}
    for folder in folders:
        cell = context.read_json(folder / "cell.json")
        cell.update(fingerprint=identity, repetitions=1, requests_per_repetition=16,
                    model_family="Llama-3.3-70B-Instruct", model=next(m for m in models if m["quant"] == cell["quantization"]))
        cell["configuration"]["GPU_DEVICE"] = "0,1"
        put(folder / "cell.json", cell)
        (folder / "study.env").write_text("# synthetic fixture; never executed\n")
        put(folder / "prompts.json", {})
        raw = context.read_json(folder / "r1/raw.json")
        raw["requests"] = [copy.deepcopy(raw["requests"][0]) for _ in range(16)]
        for i, record in enumerate(raw["requests"]):
            record.update(request_index=i, input_sha256=hashlib.sha256(f"synthetic-{cell['prompt_tokens']}-{i}".encode()).hexdigest())
        raw["configuration"]["requests"] = 16
        raw["summary"].update(successful_requests=16, cache_counter_available_requests=16, output_tokens_per_second=4096.)
        raw["study_configuration"] = cell | {"repetition": 1}
        put(folder / "r1/raw.json", raw)
        placement = context.read_json(folder / "r1/placement.json")
        placement["offload"] = [81, 81]
        put(folder / "r1/placement.json", placement)
        key = str(folder.relative_to(root / "70b"))
        progress["cells"][key] = {"status": "complete"}
        progress["cells"][key + "/r1"] = {"status": "complete"}
    put(model_root / "progress.json", progress)
    return root


def metric(value, unit=""):
    return {"value": value, "available": True, "unit": unit}


def sample(quant, phase, gpu, role="blk.*.ffn_gate.weight", index=0):
    total = 400 if quant == "Q4_K_M" else 500
    opcodes = {"FFMA": total / 4, "I2FP": total / 8, "IADD3": total * 5 / 8}
    tensor = "sm__ops_path_tensor_op_imma_src_int8_sparsity_off"
    tensor_type = {"IQ1_M": "iq1_m", "Q2_K": "q2_K", "Q4_K_M": "q4_K", "Q8_0": "q8_0"}[quant]
    return {"id": str(index), "device": str(gpu), "kernel": "synthetic_mul_mat_q", "grid": "(1,1,1)", "block": "(32,1,1)",
            "operation": {"role": role, "type": tensor_type, "m": "28672",
                          "n": "512" if phase == "prefill" else "8", "k": "8192", "phase": phase,
                          "operation_match": "nvtx_same_capture"},
            "metrics": {"gpu__time_duration.sum": metric(1000, "ns"),
                "dram__bytes_read.sum": metric(80, "byte"), "dram__bytes_write.sum": metric(20, "byte"),
                "sm__inst_executed.sum": metric(total, "inst"), "sass__inst_executed_per_opcode": metric(total),
                "sass__thread_inst_executed_true_per_opcode": metric(400),
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": metric(25, "%"),
                "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": metric(20, "%"),
                "sm__inst_executed.avg.per_cycle_active": metric(2),
                "sm__warps_active.avg.pct_of_peak_sustained_active": metric(35, "%"),
                "launch__registers_per_thread": metric(64), "launch__occupancy_limit_registers": metric(4),
                "sm__cycles_elapsed.avg.per_second": metric(1e9, "hz"), "dram__bytes.sum.per_second": metric(1e8),
                "dram__bytes.sum.peak_sustained": metric(100), "dram__cycles_elapsed.avg.per_second": metric(2e9, "hz"),
                tensor + ".sum.per_cycle_elapsed": metric(1), tensor + ".sum.peak_sustained": metric(4)},
            "metric_instances": {"sass__inst_executed_per_opcode": [metric(v) | {"instance": k} for k, v in opcodes.items()],
                "sass__thread_inst_executed_true_per_opcode": [metric(100) | {"instance": "FADD"},
                    metric(50) | {"instance": "FMUL"}, metric(250) | {"instance": "FFMA"}]}}


def make_captures(root, prompt=2048, minimal=False, formats=("Q4_K_M", "Q2_K")):
    output = root / "profiles" / f"c8-p{prompt}"
    output.mkdir(parents=True)
    jobs = pipeline.full_jobs(root, output, prompt, Path("synthetic-server"), "synthetic-ncu", formats)
    cells = [root / "70b" / q / "c8" / f"p{prompt}" for q in formats]
    sources = [{"path": str((p / "cell.json").resolve()), "sha256": publication.sha(p / "cell.json")} for p in cells]
    put(output / "paper-profile-plan.json", {"hardware": "rtxpro6000", "study_root": str(root),
        "prompt_tokens": prompt, "concurrency": 8, "output_tokens": 512, "source_cells": sources, "full_jobs": jobs,
        "formats": list(formats), "full_capture_count": 8, "operator_capture_count": 0 if minimal else 60,
        "operator_command": None if minimal else ["synthetic"],
        "analysis_scope": "sampled_gate_up_extrapolation" if minimal else "complete_operator_inventory",
        "operator_launch_cap_per_capture": 0 if minimal else 5})
    for job in jobs:
        directory = output / job["capture"]
        put(directory / "study-capture.json", {"exit_code": 0, "source_cell": str((Path(job["cell"]) / "cell.json").resolve()),
            "settings": {"PROFILE_DEVICES": str(job["gpu"]), "PROFILE_NVTX_INCLUDE":
                         f"regex:csb_batch:phase={job['phase']}:.*/*/{job['operation_regex']}"}})
        put(directory / "metadata.json", {"status": "captured", "counter_status": "collected", "server_code_commit": context.PIN,
            "gpu_identity_csv": "\n".join(f"{gpu}, synthetic-{gpu}, {GPU}, synthetic-pci, synthetic-driver" for gpu in (0, 1)),
            "warmup_requests": 8, "profiled_requests": 8, "server_build_provenance": {"status": "verified_adjacent_build",
                "build": {"configure_command": ["-DCMAKE_CUDA_ARCHITECTURES=120"]}}})
        put(directory / "counter_summary.json", {"launches": [sample(job["format"], job["phase"], job["gpu"],
            "blk.*.ffn_gate.weight" if i % 2 == 0 else "blk.*.ffn_up.weight", i) for i in range(5)]})
    if minimal:
        return output
    args = SimpleNamespace(cells=cells, output=output / "operators", hardware="rtxpro6000", phases=["prefill", "decode"],
                           operators=list(operators.OPERATORS), launch_count=5, diagnostic_server=Path("synthetic-server"), ncu="synthetic-ncu")
    plan = operators.make_plan(args)
    put(args.output / "roofline-plan.json", plan)
    for job in plan["jobs"]:
        directory = args.output / job["capture"]
        source = next(s for s in sources if f"/{job['format']}/" in s["path"])
        put(directory / "study-capture.json", {"exit_code": 0, "source_cell": source["path"], "settings": {
            "PROFILE_DEVICES": str(job["gpu"]), "PROFILE_NVTX_INCLUDE":
            f"regex:csb_batch:phase={job['phase']}:.*/*/{operators.operation_regex(job['operator'])}"}})
        put(directory / "metadata.json", {"status": "captured", "counter_status": "collected"})
        role = operators.OPERATORS[job["operator"]][1][0]
        put(directory / "counter_summary.json", {"launches": [sample(job["format"], job["phase"], job["gpu"], role)]})
    return output


class RTXWorkflowTests(unittest.TestCase):
    def test_two_format_shard_scope_builds_report_after_first_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory), prompts=(2048,), formats=("IQ1_M", "Q8_0"))
            audit = serving.build_report(root, plots=False)
            self.assertEqual(audit["scope"]["model_formats"], {"70b": ["IQ1_M", "Q8_0"]})
            self.assertEqual(audit["required_cells"], 30)
            self.assertEqual(audit["measured_cells"], 2)

    def test_git_export_splits_serving_and_profile_types_per_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "study"
            captures, output = root / "profiles/pair", Path(directory) / "exports"
            captures.mkdir(parents=True)
            serving_data = {"rows": [{"format": q, "source": str(root / q)} for q in ("Q4_K_M", "Q2_K")],
                "coverage": [{"format": q} for q in ("Q4_K_M", "Q2_K")],
                "audit": {"scope": {"model_formats": {"70b": ["Q4_K_M", "Q2_K"]}}},
                "devices": [{"name": GPU}], "sources": {str(root / "manifest.json"): "abc"}}
            profile_data = {"formats": ["Q4_K_M", "Q2_K"], "prompt_tokens": 2048,
                "analysis_scope": "sampled_gate_up_extrapolation", "layer_counts": [41, 39],
                "cases": [], "metric_rows": [], "instruction_rows": [], "tensor_samples": [],
                "extrapolation_rows": [], "operator_plan": {}, "operator_audit": {}, "operator_points": []}
            argv = ["export_rtxpro6000_results.py", "--study-root", str(root), "--captures", str(captures),
                    "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(exporter.publication, "serving_data", return_value=serving_data), \
                    patch.object(exporter.publication, "bottleneck_data", return_value=profile_data):
                exporter.main()
            index = publication.read(output / "index.json")
            self.assertEqual(len(index["files"]), 6)
            self.assertTrue((output / "Q2_K-profile-full-sections-p2048.json").is_file())
            exported = publication.read(output / "Q4_K_M-serving-r1.json")
            self.assertIn("study/Q4_K_M", json.dumps(exported))

    def test_missing_build_tools_stop_before_gpu_checks_or_build(self):
        def available(name, **kwargs):
            return None if name in ("ninja", "cmake") else "/synthetic/bin/" + name
        with patch.object(sys, "argv", ["run_rtxpro6000.py", "build"]), \
                patch.object(launcher.shutil, "which", side_effect=available), \
                patch.object(launcher, "check_devices") as gpu, patch.object(launcher, "run_command") as build:
            with self.assertRaisesRegex(ValueError, "Missing build prerequisites: cmake, ninja"):
                launcher.main(default_hardware="rtxpro6000")
            gpu.assert_not_called()
            build.assert_not_called()

    def test_preparation_lists_both_missing_local_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [{"quant": q, "filename": q + ".gguf"} for q in serving.FORMATS["70b"]]
            for entry in entries:
                if entry["quant"] in ("IQ1_M", "Q8_0"):
                    entry["repo"] = "synthetic-download-source"
            put(root / "models/study-manifest-70b.json", entries)
            with patch.object(launcher, "ROOT", root), self.assertRaises(ValueError) as caught:
                launcher.preparation_entries("70b", "rtxpro6000")
            message = str(caught.exception)
            self.assertIn(str(root / "models/Q2_K.gguf"), message)
            self.assertIn(str(root / "models/Q4_K_M.gguf"), message)
            self.assertIn("cannot download substitutes", message)

    def test_missing_ncu_explains_that_serving_can_proceed(self):
        with patch.object(sys, "argv", ["profile_rtxpro6000.py", "preflight", "--ncu", "/missing/ncu"]), \
                patch.object(pipeline, "check_devices"), patch.object(pipeline.shutil, "which", return_value=None), \
                patch.object(pipeline.subprocess, "run") as probe:
            with self.assertRaisesRegex(ValueError, "serving build/prepare/run can proceed without Nsight"):
                pipeline.main()
            probe.assert_not_called()

    def test_remote_preparation_relocates_models_without_rewriting_original_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            entries = [{"quant": q, "filename": q + ".gguf", "path": "/missing/original/" + q + ".gguf"}
                       for q in serving.FORMATS["70b"]]
            path = root / "models/study-manifest-70b.json"
            put(path, entries)
            original = path.read_bytes()
            for entry in entries:
                (root / "models" / entry["filename"]).write_bytes(b"synthetic")
            with patch.object(launcher, "ROOT", root):
                prepared = launcher.preparation_entries("70b", "rtxpro6000")
            self.assertTrue(all(Path(e["path"]).parent == (root / "models").resolve() for e in prepared))
            self.assertEqual({e["chat_template_file"] for e in prepared}, {"config/templates/llama-3.1-benchmark.jinja"})
            self.assertEqual(path.read_bytes(), original)

    def test_default_plan_and_32k_selection_do_no_gpu_work(self):
        for argv, count in ((["run_rtxpro6000.py", "plan"], 60),
                            (["run_rtxpro6000.py", "plan", "--formats", "Q4_K_M", "--concurrencies", "8", "--prompt-lengths", "32768"], 1)):
            stream = io.StringIO()
            with patch.object(sys, "argv", argv), patch.object(launcher, "check_devices") as probe, \
                    patch.object(launcher, "run_command") as execute, contextlib.redirect_stdout(stream):
                launcher.main(default_hardware="rtxpro6000")
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan["settings"], count)
            self.assertEqual(plan["cuda_architecture"], "120")
            self.assertEqual(plan["measured_requests_if_all_feasible"], 2 * plan["warmup_requests_if_all_feasible"])
            probe.assert_not_called(); execute.assert_not_called()
        command = launcher.study_command("study", "70b", ["Q4_K_M/c8/p32768"], "rtxpro6000")
        self.assertEqual(command[command.index("--extension-request-waves") + 1], "2")
        self.assertEqual(command[command.index("--capacity-lengths") + 1], "32768")

    def test_gpu_checks_and_rated_ceiling_editions(self):
        with patch.object(launcher.subprocess, "check_output", side_effect=[GPU + "\n" + GPU + "\n", "12.0, 97887\n12.0, 97887\n", ""]):
            launcher.check_devices(["0", "1"], {}, idle=True, hardware="rtxpro6000")
        with patch.object(launcher.subprocess, "check_output", return_value="NVIDIA RTX A6000\nNVIDIA RTX A6000\n"):
            with self.assertRaisesRegex(ValueError, "Expected"):
                launcher.check_devices(["0", "1"], {}, hardware="rtxpro6000")
        expected = [("Server Edition", 120, 1597), ("Workstation Edition", 125, 1792), ("Max-Q Workstation Edition", 110, 1792)]
        for edition, peak, bandwidth in expected:
            roof = cuda_hardware.rated_roof("NVIDIA RTX PRO 6000 Blackwell " + edition)
            self.assertEqual(roof["peak_fp32_flops_per_s"], peak * 1e12)
            self.assertEqual(roof["peak_dram_bytes_per_s"], bandwidth * 1e9)
        with self.assertRaisesRegex(ValueError, "edition missing"):
            cuda_hardware.rated_roof("NVIDIA RTX PRO 6000 Blackwell")

    def test_32k_report_accepts_two_waves_and_profiles_same_saved_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            audit = serving.build_report(root, plots=False)
            self.assertEqual(audit["required_cells"], 60)
            self.assertEqual(audit["measured_cells"], 8)
            rows = context.read_csv(root / "serving-report/runtime.csv")
            self.assertTrue(all(int(row["successful_requests"]) == 16 for row in rows))
            cell, devices = profile.load_cell(root / "70b/Q4_K_M/c8/p32768", "rtxpro6000")
            self.assertEqual(cell["prompt_tokens"], 32768)
            self.assertEqual(devices, ["0", "1"])
            raw_path = root / "70b/Q4_K_M/c8/p32768/r1/raw.json"
            raw = publication.read(raw_path)
            raw["requests"] = raw["requests"][:8]
            put(raw_path, raw)
            with self.assertRaisesRegex(ValueError, "independent validation"):
                profile.load_cell(root / "70b/Q4_K_M/c8/p32768", "rtxpro6000")

    def test_tensor_paths_use_blackwell_names_and_do_not_double_count_rollups(self):
        launch = sample("Q4_K_M", "decode", 0)
        launch["metrics"]["sm__ops_path_tensor_src_int8.sum.per_cycle_elapsed"] = metric(999)
        points = publication.tensor_coordinates(launch)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["arithmetic_path"], "op_imma_src_int8_sparsity_off")
        self.assertEqual(points[0]["work_ops_per_s"], 1e9)
        self.assertEqual(points[0]["arithmetic_intensity_ops_per_byte"], 10)
        self.assertEqual(points[0]["work_pct_of_peak"], 25)
        del launch["metrics"]["sm__ops_path_tensor_op_imma_src_int8_sparsity_off.sum.peak_sustained"]
        with self.assertRaises(KeyError):
            publication.tensor_coordinates(launch)

    def test_partial_opcode_inventory_is_rejected(self):
        launch = sample("Q2_K", "prefill", 0)
        self.assertEqual(sum(publication.opcode_counts(launch).values()), 500)
        launch["metric_instances"]["sass__inst_executed_per_opcode"].pop()
        with self.assertRaisesRegex(ValueError, "reconcile"):
            publication.opcode_counts(launch)

    def test_complete_profile_data_and_wrong_shape_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            captures = make_captures(root)
            data = publication.bottleneck_data(root, captures)
            self.assertEqual(len(data["cases"]), 8)
            self.assertEqual(len(data["operator_points"]), 60)
            path = captures / "full/Q2_K/decode/gpu0/counter_summary.json"
            summary = publication.read(path)
            summary["launches"][0]["operation"]["n"] = "1"
            put(path, summary)
            with self.assertRaisesRegex(ValueError, "Unmatched"):
                publication.bottleneck_data(root, captures)

    def test_minimal_profile_reuses_eight_captures_and_extrapolates_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            captures = make_captures(root, minimal=True)
            data = publication.bottleneck_data(root, captures)
            self.assertEqual(data["analysis_scope"], "sampled_gate_up_extrapolation")
            self.assertEqual(data["operator_audit"]["captures"], 8)
            self.assertEqual(len(data["operator_points"]), 8)
            self.assertEqual(data["layer_counts"], [41, 39])
            self.assertEqual(len(data["extrapolation_rows"]), 8)
            self.assertEqual(data["cases"][0]["projected_gate_up_launches"], 82)

    def test_missing_operator_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            captures = make_captures(root)
            (captures / "operators/captures/Q2_K/decode/gpu1/lm_head/counter_summary.json").unlink()
            with self.assertRaisesRegex(ValueError, "Operator captures are incomplete"):
                publication.bottleneck_data(root, captures)

    def test_profile_rejects_mismatched_source_prompts_and_capture_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            captures = make_captures(root)
            path = root / "70b/Q2_K/c8/p2048/r1/raw.json"
            raw = publication.read(path)
            original = path.read_bytes()
            raw["requests"][0]["input_sha256"] = "different"
            put(path, raw)
            with self.assertRaisesRegex(ValueError, "prompts differ"):
                publication.bottleneck_data(root, captures)
            path.write_bytes(original)
            path = captures / "full/Q2_K/decode/gpu0/metadata.json"
            metadata = publication.read(path)
            metadata["gpu_identity_csv"] = metadata["gpu_identity_csv"].replace("synthetic-0", "other-gpu")
            put(path, metadata)
            with self.assertRaisesRegex(ValueError, "GPU identities differ"):
                publication.bottleneck_data(root, captures)

    def test_publication_does_not_mix_prompt_fixtures_or_overwrite_historical_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            with self.assertRaisesRegex(ValueError, "unresolved"):
                publication.serving_data(root)
            publication.serving_data(root, allow_missing=True)
            path = root / "70b/Q2_K/c8/p2048/r1/raw.json"
            raw = publication.read(path)
            raw["requests"][0]["input_sha256"] = "different"
            put(path, raw)
            with self.assertRaisesRegex(ValueError, "prompts differ"):
                publication.serving_data(root, allow_missing=True)
            with self.assertRaisesRegex(ValueError, "historical"):
                publication.destination(ROOT / "paper/poster", root, "test")

    def test_complete_poster_combines_two_disjoint_git_export_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = (("a", ("IQ1_M", "Q8_0")), ("b", ("Q2_K", "Q4_K_M")))
            for shard, formats in pairs:
                target = root / shard
                target.mkdir()
                for quant in formats:
                    rows, coverage = [], []
                    for concurrency in (8, 16, 32):
                        for prompt in (2048, 4096, 8192, 16384, 32768):
                            rows.append({"format": quant, "concurrency": concurrency, "input_tokens": prompt,
                                "ttft_p95_s": prompt / 2048, "output_tok_s": 1000 / concurrency,
                                "tpot_p95_ms": 2, "thermal_limit_observed": False})
                            coverage.append({"format": quant, "concurrency": str(concurrency),
                                             "input_tokens": str(prompt), "status": "complete"})
                    put(target / f"{quant}-serving-r1.json", {"hardware": "rtxpro6000", "format": quant,
                        "repetitions": 1, "rows": rows, "coverage": coverage,
                        "devices": [{"name": GPU}, {"name": GPU}]})
                files = [{"path": path.name, "sha256": publication.sha(path)}
                         for path in target.glob("*-serving-r1.json")]
                put(target / "index.json", {"files": files})
            poster = publication.build_poster_exports([root / "a", root / "b"], root / "poster")
            self.assertTrue(poster.read_bytes().startswith(b"%PDF"))

    def test_both_pdf_builders_render_synthetic_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_study(Path(directory))
            captures = make_captures(root, 32768)
            poster = publication.build_poster(root, allow_missing=True)
            report = publication.build_bottlenecks(root, captures)
            self.assertTrue(poster.read_bytes().startswith(b"%PDF"))
            self.assertTrue(report.read_bytes().startswith(b"%PDF"))
            self.assertEqual(len(list((report.parent / "figures").glob("*.pdf"))), 5)
            self.assertEqual(len(list((poster.parent / "figures").glob("*.svg"))), 4)
            self.assertNotIn("A6000", (report.parent / "runtime-bottlenecks-report.md").read_text().split("This eight-column")[0])


if __name__ == "__main__":
    unittest.main()
