"""Offline checks for context-study profile selection and counter fidelity."""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import re
import sqlite3
import tempfile
import unittest
from unittest import mock

from profile_cuda import find_ncu_report, operation_from_nvtx, server_build_provenance, summarize_ncu, summarize_sqlite, validate_counter_capture
from profile_study import capture_identity, counter_matches, dominant_operations, ensure_operation_coverage, operation_nvtx_filter, run_capture, select_cells, study_cell_paths


class CounterTests(unittest.TestCase):
    def test_attention_node_names_pool_with_exact_nvtx_filter(self):
        operation = operation_from_nvtx([
            "csb_batch:phase=decode:prefill_tokens=0:decode_tokens=8:width=8",
            "csb_op:role=node_23:type=f32:op=FLASH_ATTN_EXT:phase=unknown"])
        self.assertEqual(operation["role"], "op.FLASH_ATTN_EXT")
        self.assertEqual(operation["tensor_name"], "node_23")
        inner = operation_nvtx_filter(operation).split("/*/")[1]
        self.assertRegex("csb_op:role=node_56:type=f32:op=FLASH_ATTN_EXT:phase=unknown", inner)
        self.assertIsNone(re.fullmatch(inner, "csb_op:role=node_56:type=f32:op=SOFT_MAX:phase=unknown"))

    def test_large_sass_instance_csv_field_preserves_all_instances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            source = root / "raw.csv"
            instances = "; ".join(f"SASS_{index:05d}: 1" for index in range(10000))
            self.assertGreater(len(instances), 131072)
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ID", "Kernel Name", "Device", "sass__inst_executed_per_opcode"])
                writer.writerow(["", "", "", "inst"])
                writer.writerow(["0", "mul_mat_q", "0", f"10000 ({instances})"])
            launch = summarize_ncu(source, root)["launches"][0]
            self.assertEqual(launch["metrics"]["sass__inst_executed_per_opcode"]["value"], 10000)
            self.assertEqual(len(launch["metric_instances"]["sass__inst_executed_per_opcode"]), 10000)

    def test_batch_phase_comes_from_tokens_not_activation_width(self):
        operation = operation_from_nvtx([
            "csb_batch:phase=prefill:prefill_tokens=64:decode_tokens=0:width=64",
            "csb_op:role=blk.12.attn_q.weight:type=q2_K:m=4096:n=64:k=8192:phase=unknown"])
        self.assertEqual(operation["phase"], "prefill")
        self.assertEqual(operation["role"], "blk.*.attn_q.weight")
        self.assertEqual(operation["tensor_name"], "blk.12.attn_q.weight")
        self.assertEqual(operation["n"], "64")

    def test_fused_roles_normalize_all_layer_names_and_keep_provenance(self):
        operation = operation_from_nvtx([
            "csb_batch:phase=decode:prefill_tokens=0:decode_tokens=1:width=1",
            "csb_op:role=blk.12.ffn_up.weight+blk.12.ffn_gate.weight:type=f16+f16:m=8192:n=1:k=2048:fusion=gate_up_glu:phase=unknown"])
        self.assertEqual(operation["role"], "blk.*.ffn_up.weight+blk.*.ffn_gate.weight")
        self.assertEqual(operation["fusion"], "gate_up_glu")
        self.assertIn("blk.12", operation["tensor_name"])

    def test_wide_instance_values_preserve_aggregate_opcodes_and_devices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            source = root / "raw.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ID", "Kernel Name", "Device", "Grid Size", "Block Size",
                                 "gpu__time_duration.sum", "sass__inst_executed_per_opcode", "dram__bytes_read.sum"])
                writer.writerow(["", "", "", "", "", "ns", "inst", "byte"])
                writer.writerow(["0", "mul_mat_q", "0", "(4, 1, 1)", "(32, 4, 1)", "1000",
                                 "120 (SHF: 80; IADD3: 40)", "N/A"])
                writer.writerow(["0", "mul_mat_q", "1", "(4, 1, 1)", "(32, 4, 1)", "2000",
                                 "110 (SHF: 70; IADD3: 40)", "50"])
            report = summarize_ncu(source, root)
            self.assertEqual(report["observed_cuda_devices"], ["0", "1"])
            self.assertEqual(report["sample_count"], 2)
            launch = report["launches"][0]
            self.assertEqual(launch["metrics"]["sass__inst_executed_per_opcode"]["value"], 120)
            self.assertEqual(launch["metric_instances"]["sass__inst_executed_per_opcode"][0]["instance"], "SHF")
            self.assertEqual(launch["metric_instances"]["sass__inst_executed_per_opcode"][0]["value"], 80)
            self.assertIsNone(launch["metrics"]["dram__bytes_read.sum"]["value"])
            self.assertTrue((root / "counter_metrics.csv").exists())

    def test_section_duplicates_and_conflicting_values_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            source = root / "raw.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["ID", "Kernel Name", "Device", "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Metric Instance"])
                writer.writerow(["0", "kernel", "0", "A", "metric__sum", "inst", "40", ""])
                writer.writerow(["0", "kernel", "0", "B", "metric__sum", "inst", "41", ""])
                writer.writerow(["0", "kernel", "0", "C", "metric__sum", "inst", "41", ""])
                writer.writerow(["0", "kernel", "0", "C", "units__sum", "inst", "41", ""])
                writer.writerow(["0", "kernel", "0", "C", "units__sum", "warp", "41", ""])
                writer.writerow(["0", "kernel", "0", "InstructionStats", "sass__per_opcode", "inst", "20", "SHF"])
            launch = summarize_ncu(source, root)["launches"][0]
            self.assertTrue(launch["metrics"]["metric__sum"]["ambiguous"])
            self.assertIsNone(launch["metrics"]["metric__sum"]["value"])
            self.assertEqual(len(launch["metrics"]["metric__sum"]["observations"]), 3)
            self.assertTrue(launch["metrics"]["units__sum"]["ambiguous"])
            self.assertEqual(launch["metric_instances"]["sass__per_opcode"][0]["value"], 20)

    def test_compressed_report_is_preferred_and_legacy_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            report = pathlib.Path(temp) / "profile"
            report.with_suffix(".ncu-rep").touch()
            self.assertEqual(find_ncu_report(report), report.with_suffix(".ncu-rep"))
            report.with_suffix(".ncu-repz").touch()
            self.assertEqual(find_ncu_report(report), report.with_suffix(".ncu-repz"))

    def test_section_opcode_instances_and_nvtx_are_required_when_requested(self):
        def metric(value):
            return {"value": value, "available": True}
        summary = {"launches": [{"id": "0", "device": "0", "metrics": {
            "gpu__time_duration.sum": metric(1), "smsp__inst_executed.sum": metric(100),
            "sass__inst_executed_per_opcode": metric(100)}}]}
        result = validate_counter_capture(summary, ["gpu__time_duration.sum"], ["InstructionStats"],
                                          required_launches=1, expected_devices=["0"], require_nvtx=True)
        self.assertFalse(result["valid"])
        self.assertFalse(result["section_coverage"]["InstructionStats"]["valid"])
        self.assertEqual(len(result["invalid_nvtx_launches"]), 1)
        launch = summary["launches"][0]
        launch["metric_instances"] = {"sass__inst_executed_per_opcode": [{"instance": "SHF", **metric(100)}]}
        launch["operation"] = {"operation_match": "nvtx_same_capture", "phase": "decode", "role": "blk.*.attn_q.weight",
                               "type": "q4_K", "m": "8192", "n": "8", "k": "8192"}
        self.assertTrue(validate_counter_capture(summary, ["gpu__time_duration.sum"], ["InstructionStats"],
                                                 required_launches=1, expected_devices=["0"], require_nvtx=True)["valid"])


class SelectionTests(unittest.TestCase):
    def test_shared_family_symlinks_and_canonical_roots_discover_once(self):
        with tempfile.TemporaryDirectory() as temp:
            results = pathlib.Path(temp)
            shared = results / "cuda-context-study"
            shared.mkdir()
            for family in ("1b", "8b", "70b"):
                canonical = results / ("cuda-context-study-" + family)
                folder = canonical / "Q4_K_M/c8/p2048"
                folder.mkdir(parents=True)
                (folder / "cell.json").write_text(json.dumps({"model_family": family, "quantization": "Q4_K_M",
                    "concurrency": 8, "prompt_tokens": 2048, "output_tokens": 512, "status": "complete"}))
                (shared / family).symlink_to(canonical, target_is_directory=True)
                self.assertEqual(len(select_cells(canonical)["selected"]), 1)
            # A general results-root traversal sees canonical directories and
            # the shared directory, but aliases must not duplicate any cell.
            self.assertEqual(len(study_cell_paths(results)), 3)
            selected = select_cells(shared)["selected"]
            self.assertEqual({cell["model_family"] for cell in selected}, {"1b", "8b", "70b"})
            self.assertEqual(len({cell["cell_file"] for cell in selected}), 3)

    def test_discovery_excludes_validation_profiles_archive_and_legacy_cuda_studies(self):
        with tempfile.TemporaryDirectory() as temp:
            results = pathlib.Path(temp)
            valid = results / "cuda-context-study-1b/Q4_K_M/c8/p2048"
            directories = [valid] + [valid.parents[2] / excluded / "Q4_K_M/c8/p65536"
                                     for excluded in ("validation", "profiles", "archive")]
            directories += [results / legacy / "Q4_K_M/c8/p65536" for legacy in ("cuda-study-1b", "prefix-study-1b")]
            for folder in directories:
                folder.mkdir(parents=True)
                (folder / "cell.json").write_text(json.dumps({"model_family": "1b", "quantization": "Q4_K_M",
                    "concurrency": 8, "prompt_tokens": int(folder.name[1:]), "output_tokens": 512, "status": "complete"}))
            self.assertEqual(study_cell_paths(results), [(valid / "cell.json").resolve()])
            self.assertEqual(select_cells(results)["joint_maxima"][0]["longest_jointly_feasible_prompt"], 2048)
            shared = results / "cuda-context-study"
            shared.mkdir()
            (shared / "1b").symlink_to(results / "cuda-study-1b", target_is_directory=True)
            self.assertEqual(study_cell_paths(shared), [])

    def test_custom_repeat_series_only_follows_its_exact_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            results = pathlib.Path(temp) / "results"
            shared = results / "cuda-context-study-repeat2"
            shared.mkdir(parents=True)
            own = results / "cuda-context-study-repeat2-1b"
            other_series = results / "cuda-context-study-1b"
            outside = pathlib.Path(temp) / "outside/cuda-context-study-repeat2-1b"
            for canonical in (own, other_series, outside):
                folder = canonical / "Q4_K_M/c8/p2048"
                folder.mkdir(parents=True)
                (folder / "cell.json").write_text(json.dumps({"model_family": "1b", "quantization": "Q4_K_M",
                    "concurrency": 8, "prompt_tokens": 2048, "output_tokens": 512, "status": "complete"}))
            alias = shared / "1b"
            alias.symlink_to(own, target_is_directory=True)
            expected = (own / "Q4_K_M/c8/p2048/cell.json").resolve()
            self.assertEqual(study_cell_paths(shared), [expected])
            self.assertEqual(len(select_cells(shared)["selected"]), 1)
            for rejected_target in (other_series, outside):
                alias.unlink()
                alias.symlink_to(rejected_target, target_is_directory=True)
                self.assertEqual(study_cell_paths(shared), [])

    def test_relocated_cell_uses_and_hashes_colocated_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            original = root / "old-run/Q4_K_M/c8/p2048/study.env"
            original.parent.mkdir(parents=True)
            original.write_text("SERVER_CTX_SIZE=1\n")
            moved = root / "cuda-context-study-1b/Q4_K_M/c8/p2048"
            moved.mkdir(parents=True)
            current = moved / "study.env"
            current.write_text("SERVER_CTX_SIZE=22528\n")
            (moved / "cell.json").write_text(json.dumps({"model_family": "1b", "quantization": "Q4_K_M",
                "concurrency": 8, "prompt_tokens": 2048, "output_tokens": 512, "status": "complete",
                "config_file": str(original)}))
            cell = select_cells(root / "cuda-context-study-1b")["selected"][0]
            self.assertEqual(cell["original_config_file"], str(original))
            self.assertEqual(cell["config_file"], str(current.resolve()))
            script = pathlib.Path(__file__).resolve().parents[1] / "scripts/07_profile_cuda.sh"
            identity = capture_identity(cell, {"PROFILE_TOOL": "nsys"}, script)
            original.write_text("SERVER_CTX_SIZE=2\n")
            self.assertEqual(identity, capture_identity(cell, {"PROFILE_TOOL": "nsys"}, script))
            current.write_text("SERVER_CTX_SIZE=45056\n")
            self.assertNotEqual(identity, capture_identity(cell, {"PROFILE_TOOL": "nsys"}, script))

    def test_same_capture_operation_matching_requires_matching_dimensions_and_phase(self):
        op = {"role": "blk.*.attn_q.weight", "type": "q2_K", "m": "8192", "n": "8", "k": "8192", "phase": "decode"}
        selected = [{"device": "1", "kernel": "mul_mat_q", "grid": "4x1x1", "block": "32x4x1", "calls": "1", **op}]
        summary = {"launches": [{"id": "0", "device": "1", "kernel": "mul_mat_q", "class": "quantized_matmul",
                                "grid": "(4, 1, 1)", "block": "(32, 4, 1)",
                                "operation": dict(op, operation_match="nvtx_same_capture")}]}
        self.assertTrue(counter_matches(summary, selected, "1")["matching_verified"])
        summary["launches"][0]["operation"]["n"] = "9"
        self.assertFalse(counter_matches(summary, selected, "1")["matching_verified"])

    def test_one_role_cannot_satisfy_other_selected_role_with_identical_grid(self):
        first = {"device": "0", "kernel": "mul_mat_q", "grid": "4x1x1", "block": "32x4x1", "calls": "12",
                 "role": "blk.*.attn_q.weight", "type": "q2_K", "m": "8192", "n": "8", "k": "8192", "phase": "decode"}
        second = dict(first, role="blk.*.attn_output.weight")
        summary = {"launches": [dict(first, id=str(index), operation=dict(first, operation_match="nvtx_same_capture")) for index in range(5)]}
        result = counter_matches(summary, [first, second], "0")
        self.assertFalse(result["matching_verified"])
        self.assertEqual([row["observed_launches"] for row in result["operation_coverage"]], [5, 0])
        self.assertEqual([row["required_launches"] for row in result["operation_coverage"]], [5, 5])

    def test_dequantization_contributes_to_dominant_matrix_cost(self):
        rows = [{"device": "0", "phase": "prefill", "class": kind, "kernel": kind, "total_ms": ms}
                for kind, ms in [("dequantization", 800), ("other_matrix_multiply", 200)]]
        selected = dominant_operations(rows)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["bucket_total_ms"] == 1000 for row in selected))

    def test_joint_maximum_excludes_unrunnable_format_and_deduplicates_16k(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for quant, lengths in {"Q2": [2048, 16384, 32768], "Q4": [2048, 16384], "FP16": []}.items():
                for p in sorted(set(lengths) | {2048, 16384, 32768}):
                    folder = root / quant / "c8" / f"p{p}"
                    folder.mkdir(parents=True)
                    (folder / "cell.json").write_text(json.dumps({
                        "model_family": "70b", "quantization": quant, "concurrency": 8,
                        "prompt_tokens": p, "output_tokens": 512,
                        "status": "complete" if p in lengths else "capacity_estimated"}))
            selected = select_cells(root)
            self.assertEqual(selected["joint_maxima"][0]["longest_jointly_feasible_prompt"], 16384)
            self.assertEqual(len(selected["selected"]), 4)
            self.assertEqual(selected["selected"][1]["endpoint_reasons"], ["16k", "joint_maximum"])
            self.assertTrue(all(row["quantization"] == "FP16" for row in selected["skipped"]))

    def test_dominance_uses_time_per_device_and_phase(self):
        rows = [{"device": str(d), "phase": phase, "class": "quantized_matmul", "kernel": str(ms), "total_ms": ms}
                for d in (0, 1) for phase in ("prefill", "decode") for ms in (91, 8, 1)]
        selected = dominant_operations(rows)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(row["total_ms"] == 91 for row in selected))

    def test_launch_grid_match_does_not_claim_matrix_shape_or_phase(self):
        ops = [{"device": "1", "kernel": "mul_mat_q", "grid": "4x1x1", "block": "32x4x1",
                "m": "4096", "n": "8", "k": "4096", "phase": "decode"}]
        summary = {"launches": [{"id": "0", "device": "1", "kernel": "mul_mat_q",
                                "grid": "(4, 1, 1)", "block": "(32, 4, 1)"}]}
        matched = counter_matches(summary, ops, "1")
        self.assertEqual(matched["matched_launch_shape_count"], 1)
        self.assertFalse(matched["matching_verified"])
        self.assertEqual(matched["launches"][0]["phase"], "unknown")


class ResumeTests(unittest.TestCase):
    def test_private_server_commit_requires_matching_build_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = pathlib.Path(temp) / ".run/context-diagnostic-build/baseline"
            folder.mkdir(parents=True)
            server, library = folder / "llama-server", folder / "libggml-cuda.so"
            server.write_bytes(b"server")
            library.write_bytes(b"library")
            manifest = {"commit": "1" * 40, "server_sha256": hashlib.sha256(b"server").hexdigest(),
                        "cuda_library_sha256": hashlib.sha256(b"library").hexdigest()}
            (folder / "build.json").write_text(json.dumps(manifest))
            with mock.patch("profile_cuda.subprocess.run") as git:
                self.assertEqual(server_build_provenance(server)["commit"], "1" * 40)
                library.write_bytes(b"changed library")
                self.assertIsNone(server_build_provenance(server)["commit"])
                (folder / "build.json").unlink()
                self.assertIsNone(server_build_provenance(server)["commit"])
                git.assert_not_called()

    def test_targeted_complete_source_replaces_partial_primary_without_summing(self):
        first = {"device": "0", "kernel": "mul_mat_q", "class": "quantized_matmul", "grid": "4x1x1", "block": "32x4x1",
                 "calls": "12", "role": "blk.*.attn_q.weight", "type": "q2_K", "m": "8192", "n": "8", "k": "8192", "phase": "decode"}
        second = dict(first, role="blk.*.attn_output.weight")
        def launch(operation, index):
            return dict(operation, id=str(index), operation=dict(operation, operation_match="nvtx_same_capture"))
        for target_count in (3, 5):
            with self.subTest(target_count=target_count), tempfile.TemporaryDirectory() as temp:
                primary = pathlib.Path(temp) / "capture1"
                primary.mkdir()
                (primary / "counter_summary.json").write_text(json.dumps({"launches":
                    [launch(first, i) for i in range(5)] + [launch(second, i + 5) for i in range(2)]}))
                def fake_capture(cell, destination, settings, **kwargs):
                    self.assertEqual(kwargs["required_operations"], [second])
                    self.assertIn("blk\\.[0-9]+\\.attn_output", settings["PROFILE_NVTX_INCLUDE"])
                    destination.mkdir(parents=True)
                    (destination / "counter_summary.json").write_text(json.dumps({"launches": [launch(second, i) for i in range(target_count)]}))
                    return destination
                with mock.patch("profile_study.run_capture", side_effect=fake_capture):
                    kwargs = dict(script=primary / "unused.sh", attempts=1, diagnostic_server=primary / "unused-server", requested_launches=5)
                    if target_count == 5:
                        result = ensure_operation_coverage({}, primary, [first, second], {"PROFILE_DEVICES": "0"},
                            {"capture": 1, "metric_group": "combined", "metric_groups": ["memory"]}, **kwargs)
                        self.assertEqual(result["status"], "complete")
                        self.assertEqual(result["operation_coverage"][1]["observed_launches"], 5)
                        self.assertTrue(result["operation_coverage"][1]["source_capture"].startswith("targeted/"))
                    else:
                        with self.assertRaisesRegex(RuntimeError, "incomplete"):
                            ensure_operation_coverage({}, primary, [first, second], {"PROFILE_DEVICES": "0"}, {"capture": 1}, **kwargs)
                        self.assertEqual(json.loads((primary / "coverage.json").read_text())["status"], "incomplete")

    def test_fixture_contents_change_capture_identity_at_same_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "study.env").write_text("SERVER_PARALLEL=8\n")
            (root / "prompts.json").write_text("[]")
            cell = {"config_file": str(root / "study.env"), "cell_file": str(root / "cell.json")}
            script = pathlib.Path(__file__).resolve().parents[1] / "scripts/07_profile_cuda.sh"
            before = capture_identity(cell, {"PROFILE_TOOL": "nsys"}, script)
            cell["endpoint_reasons"] = ["2k", "joint_maximum"]
            self.assertEqual(before, capture_identity(cell, {"PROFILE_TOOL": "nsys"}, script))
            (root / "prompts.json").write_text('[{"messages": []}]')
            self.assertNotEqual(before, capture_identity(cell, {"PROFILE_TOOL": "nsys"}, script))

    def test_counters_only_resolves_successful_trace_retry_without_launching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for name, code in [("trace", 2), ("trace-attempt2", 0)]:
                folder = root / name
                folder.mkdir()
                (folder / "study-capture.json").write_text(json.dumps({"identity": "same", "exit_code": code}))
                (folder / "metadata.json").write_text(json.dumps({"status": "captured"}))
                (folder / "kernel_summary.csv").write_text("kernel\n")
                (folder / "profile_summary.json").write_text("{}")
            with mock.patch("profile_study.capture_identity", return_value="same"), mock.patch("profile_study.subprocess.run") as execute:
                resolved = run_capture({"cell_file": str(root / "cell.json"), "config_file": "unused", "concurrency": 8},
                    root / "trace", {"PROFILE_TOOL": "nsys"}, execute=True, script=root / "unused.sh", attempts=1,
                    diagnostic_server=None, existing_only=True)
            self.assertEqual(resolved, root / "trace-attempt2")
            execute.assert_not_called()


class TimelineTests(unittest.TestCase):
    def test_nested_cpu_nvtx_ranges_attribute_asynchronous_kernel_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            database = root / "trace.sqlite"
            con = sqlite3.connect(database)
            con.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'mul_mat_q'),(2,'cudaLaunchKernel');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,
                  gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,
                  registersPerThread INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER,
                  localMemoryPerThread INTEGER,graphNodeId INTEGER,correlationId INTEGER,demangledName INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1000,2000,0,1,4,1,1,32,4,1,40,0,1024,0,0,9,1);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(20,25,8,9,2);
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
                INSERT INTO NVTX_EVENTS VALUES(0,90,8,'csb_batch:phase=decode:prefill_tokens=0:decode_tokens=8:width=8');
                INSERT INTO NVTX_EVENTS VALUES(10,30,8,'csb_op:role=blk.0.attn_q.weight:type=q4_K:m=8192:n=8:k=8192:phase=unknown');
            """)
            con.commit()
            con.close()
            summary = summarize_sqlite(database, root)
            self.assertEqual(summary["operation_attribution"]["phase_known_kernel_calls"], 1)
            with (root / "kernel_summary.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["phase"], "decode")
            self.assertEqual(row["n"], "8")
            self.assertEqual(row["role"], "blk.*.attn_q.weight")


if __name__ == "__main__":
    unittest.main()
