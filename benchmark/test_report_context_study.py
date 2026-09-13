"""Reporting acceptance checks using synthetic artifacts; never start a GPU job."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import report_context_study as report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, prompt=2048, quant="Q4_K_M", family="1b"):
    concurrency = 8
    experiment = "serving_grid" if prompt in report.PROMPTS else "capacity_extension"
    count = concurrency * (2 if experiment == "serving_grid" else 1)
    folder = root / family / quant / "c8" / f"p{prompt}"
    manifest = {"context_study": True, "llama_cpp_commit": report.PIN,
                "devices": [{"index": "0"}, {"index": "1"}], "binary_sha256": "binary",
                "code_sha256": {"runner": "code"}, "models": [{"quant": quant, "sha256": "weights"}]}
    put(root / family / "manifest.json", manifest)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cell = {"quantization": quant, "model_family": family, "concurrency": concurrency,
            "prompt_tokens": prompt, "output_tokens": 512, "experiment": experiment,
            "status": "complete", "fingerprint": fingerprint, "repetitions": 3,
            "requests_per_repetition": count}
    cell["configuration"] = {"LLAMA_CPP_REF": report.PIN, "SERVER_PARALLEL": "8",
        "SERVER_CTX_SIZE": str(8 * report.slot_capacity(prompt)), "SERVER_BATCH_SIZE": "2048",
        "SERVER_UBATCH_SIZE": "512", "MAX_OUTPUT_TOKENS": "512", "FLASH_ATTN": "on",
        "SERVER_SPLIT_MODE": "layer", "SERVER_TENSOR_SPLIT": "1,1", "SERVER_CACHE_TYPE_K": "f16",
        "SERVER_CACHE_TYPE_V": "f16", "SERVER_NO_CONTEXT_SHIFT": "1", "SERVER_FIT": "off"}
    put(folder / "cell.json", cell)
    key = f"{quant}/c8/p{prompt}"
    progress = {"fingerprint": fingerprint, "cells": {key: {"status": "complete"}}}
    placement = {"valid": True, "errors": [], "slots": concurrency, "slot_capacity": report.slot_capacity(prompt),
                 "headroom_valid": True, "offload": [17, 17], "kv_buffers_mib": {"CUDA0": "256.0", "CUDA1": "256.0"}}
    resources = {"swap": {"valid": True, "max_server_swap_bytes": 0, "sample_errors": []},
                 "memory": {"valid": True, "gpus": [{"index": str(gpu), "samples_available": True,
                    "peak_used_bytes": 3 * report.GIB, "minimum_headroom_bytes": 45 * report.GIB} for gpu in (0, 1)]}}
    def raw(repetition, requests):
        record = {"ok": True, "validation_errors": [], "prepared_prompt_tokens": prompt,
                  "prompt_tokens": prompt, "evaluated_prompt_tokens": prompt,
                  "completion_tokens": 512, "cached_prompt_tokens": 0, "fixed_output_length_valid": True,
                  "duration_seconds": 2., "ttft_seconds": .2, "tpot_seconds": 1.8 / 511,
                  "input_sha256": "a" * 64}
        return {"schema_version": 3, "repetition": repetition,
            "configuration": {"concurrency": concurrency, "requests": requests,
                "target_prompt_tokens": prompt, "max_output_tokens": 512, "exact_prompt_tokens": True,
                "prompt_cache_requested": False, "prefix_reuse": False, "repeat_prompt": False,
                "ignore_eos": True, "warmup_requests": 0},
            "summary": {"successful_requests": requests, "failed_requests": 0,
                "output_length_invalid_requests": 0, "cached_prompt_tokens": 0,
                "cache_counter_available_requests": requests, "max_client_inflight": concurrency,
                "wall_seconds": 2., "output_tokens_per_second": requests * 512 / 2,
                "started_unix_seconds": 1., "finished_unix_seconds": 3.},
            "requests": [copy.deepcopy(record) for _ in range(requests)],
            "study_configuration": cell | {"repetition": repetition}}
    put(folder / "warmup-discarded.json", raw(0, concurrency))
    put(folder / "warmup-resources.json", resources)
    for repetition in (1, 2, 3):
        target = folder / f"r{repetition}"
        put(target / "raw.json", raw(repetition, count))
        put(target / "placement.json", placement)
        put(target / "resources.json", resources)
        (target / "telemetry.csv").write_text(
            "timestamp,index,memory.used [MiB],power.draw [W],temperature.gpu,clocks.current.sm [MHz],clocks.current.memory [MHz],clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown\n"
            "1970/01/01 00:00:01.100,0,3072,250,65,1800,8000,Not Active,Not Active\n"
            "1970/01/01 00:00:01.100,1,4096,251,66,1800,8000,Not Active,Not Active\n"
            "1970/01/01 00:00:04.100,0,45000,250,65,1800,8000,Active,Not Active\n")
        log = []
        for burst in range(count // concurrency):
            log += [f"slot launch: id {i} | task {burst * concurrency + i} | processing task, is_child = 0\n" for i in range(concurrency)]
            log += [f"slot release: id {i} | task {burst * concurrency + i} | stop processing: n_tokens = 2559, truncated = 0\n" for i in range(concurrency)]
        (target / "server.log").write_text("".join(log))
        progress["cells"][f"{key}/r{repetition}"] = {"status": "complete"}
    put(root / family / "progress.json", progress)
    return folder


class ContextReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "context-study" / "run"
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def sibling_layout(self, name="cuda-context-study"):
        root = Path(self.temp.name) / "results" / name
        root.mkdir(parents=True)
        for family in report.FORMATS:
            target = root.parent / f"{name}-{family}"
            target.mkdir()
            (root / family).symlink_to(Path("..") / target.name, target_is_directory=True)
        return root

    def test_valid_cell_is_reported_but_partial_study_never_complete(self):
        fixture(self.root)
        audit = report.build_report(self.root, plots=False)
        rows = report.read_csv(self.root / "report/runtime.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["successful_requests"], "16")
        self.assertEqual(rows[0]["output_tokens_per_second_stddev"], "")
        self.assertEqual(float(rows[0]["output_tokens_per_second_mean"]), 4096.)
        self.assertEqual(float(rows[0]["gpu0_peak_vram_gib"]), 3.)
        self.assertEqual(float(rows[0]["gpu1_peak_vram_gib"]), 4.)
        self.assertEqual(rows[0]["gpu0_thermal_limit_observed"], "False")
        self.assertEqual(audit["status"], "incomplete")
        self.assertEqual(sum(audit["cell_status_counts"].values()), 360)

    def test_missing_repetition_never_publishes_partial_runtime(self):
        folder = fixture(self.root)
        (folder / "r1/raw.json").unlink()
        audit = report.build_report(self.root, plots=False)
        self.assertEqual(report.read_csv(self.root / "report/runtime.csv"), [])
        self.assertEqual(audit["cell_status_counts"]["incomplete"], 1)
        row = next(row for row in report.read_csv(self.root / "report/capacity.csv") if row["status"] == "incomplete")
        self.assertEqual(row["validated_repetitions"], "0")
        self.assertIn("r1", row["validation_errors"])

    def test_invalid_record_is_rejected_despite_success_summary(self):
        folder = fixture(self.root)
        path = folder / "r1/raw.json"
        raw = report.read_json(path)
        raw["requests"][3]["completion_tokens"] = 511
        put(path, raw)
        report.build_report(self.root, plots=False)
        self.assertEqual(report.read_csv(self.root / "report/runtime.csv"), [])

    def test_earlier_extra_runs_are_excluded_from_single_run_comparison(self):
        folder = fixture(self.root)
        (folder / "r2/raw.json").write_text('not valid JSON')
        report.build_report(self.root, plots=False)
        rows = report.read_csv(self.root / "report/runtime.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["successful_requests"], "16")
        self.assertEqual(rows[0]["repetitions"], "1")

    def test_malformed_raw_is_audit_failure_not_exception_or_completion(self):
        folder = fixture(self.root)
        for malformed in ("not json", '{"configuration": []}', '{"summary": "invalid"}'):
            with self.subTest(malformed=malformed):
                (folder / "r1/raw.json").write_text(malformed)
                audit = report.build_report(self.root, plots=False)
                self.assertEqual(audit["status"], "incomplete")
                self.assertEqual(report.read_csv(self.root / "report/runtime.csv"), [])

    def test_progress_and_fingerprint_must_agree(self):
        fixture(self.root)
        path = self.root / "1b/progress.json"
        progress = report.read_json(path)
        progress["fingerprint"] = "changed"
        put(path, progress)
        report.build_report(self.root, plots=False)
        self.assertEqual(report.read_csv(self.root / "report/runtime.csv"), [])

    def test_capacity_bursts_do_not_enter_serving_table(self):
        fixture(self.root, prompt=32768)
        report.build_report(self.root, plots=False)
        self.assertEqual(report.read_csv(self.root / "report/runtime.csv"), [])
        extension = report.read_csv(self.root / "report/capacity-extension.csv")
        self.assertEqual(len(extension), 1)
        self.assertEqual(extension[0]["successful_requests"], "8")

    def test_unavailable_thermal_flags_never_become_inactive(self):
        sw, hw = "clocks_event_reasons.sw_thermal_slowdown", "clocks_event_reasons.hw_thermal_slowdown"
        for rows in ([], [{sw: "[N/A]", hw: "[N/A]"}], [{sw: "Not Active", hw: "N/A"}],
                     [{sw: "Active", hw: "[N/A]"}], [{sw: "Not Active"}],
                     [{sw: "NotActive", hw: "NotActive"}, {sw: "[N/A]", hw: "NotActive"}]):
            with self.subTest(rows=rows):
                _, observed = report.thermal_observations(rows)
                self.assertIsNone(observed)
        flags, observed = report.thermal_observations([{sw: "Active", hw: "[N/A]"}])
        self.assertTrue(flags["sw"])
        self.assertIsNone(flags["hw"])
        self.assertIsNone(observed)
        self.assertFalse(report.thermal_observations([{sw: "Not Active", hw: "NotActive"}])[1])
        self.assertTrue(report.thermal_observations([{sw: "NotActive", hw: "Active"}])[1])

    def test_power_of_two_maxima_preserve_missing_formats_and_fp16_exclusion(self):
        capacity = []
        for family, formats in report.FORMATS.items():
            for quant in formats:
                for concurrency in report.CONCURRENCIES:
                    for prompt in report.PROMPTS + report.EXTENSION:
                        capacity.append({"model": family, "format": quant, "concurrency": concurrency, "input_tokens": prompt,
                            "status": "capacity_estimated" if (family, quant) == ("70b", "FP16") else "complete" if prompt <= 8192 else "observed_headroom_limit"})
        maximum, joint = report.maximum_inputs(capacity)
        row = next(row for row in joint if row["model"] == "70b" and row["concurrency"] == 8)
        self.assertEqual(row["capacity_excluded_formats"], "FP16")
        self.assertEqual(row["longest_jointly_validated_input_tokens"], 8192)
        for cell in capacity:
            if cell["model"] == "1b" and cell["format"] == "FP16":
                cell["status"] = "missing"
        _, joint = report.maximum_inputs(capacity)
        row = next(row for row in joint if row["model"] == "1b" and row["concurrency"] == 8)
        self.assertIsNone(row["longest_jointly_validated_input_tokens"])
        self.assertEqual(row["unresolved_formats"], "FP16")

    def test_counter_statistics_keep_different_shapes_devices_units_separate(self):
        for repetition in (1, 2, 3):
            target = self.root / "profiles/70b/Q2_K/c8/p2048/counters/dominant/gpu0/memory" / f"capture{repetition}"
            selection = {"matching_verified": True, "operation": {"phase": "decode", "role": "ffn", "m": 1024, "n": 8, "k": 2048}}
            put(target / "selection.json", selection)
            put(target / "metadata.json", {"status": "captured", "counter_status": "collected"})
            launches = [{"id": str(i), "kernel": "kernel", "device": "0", "grid": "32x1x1", "block": "32x2x1",
                        "metrics": {"dram__bytes_read.sum": {"value": 20, "unit": "byte", "available": True}}} for i in range(5)]
            launches += [{"id": "5", "kernel": "kernel", "device": "1", "grid": "64x1x1", "block": "32x2x1",
                          "metrics": {"dram__bytes_read.sum": {"value": 300, "unit": "Kbyte", "available": True}}}]
            launches += [{"id": "6", "kernel": "kernel", "device": "0", "grid": "32x1x1", "block": "32x2x1",
                          "metrics": {"dram__bytes_read.sum": {"value": None, "unit": "byte", "available": False, "raw": "n/a"}}}]
            put(target / "counter_summary.json", {"launches": launches})
        _, observations, summaries, _ = report.profile_tables(self.root)
        self.assertEqual(len(summaries), 2)
        row = next(row for row in summaries if row["device"] == "0")
        self.assertEqual(row["mean"], 20)
        self.assertEqual(row["independent_captures"], 1)
        self.assertEqual(row["available_launches"], 5)
        self.assertEqual(row["unavailable_launches"], 1)
        self.assertFalse(row["sampling_requirement_met"])  # No selected-operation coverage artifact.
        self.assertEqual(len(observations), 21)

    def test_selected_operation_uses_one_complete_source_not_sum_of_partials(self):
        operation = {"device": "0", "kernel": "kernel", "grid": "32x1x1", "block": "32x2x1",
                     "phase": "decode", "role": "blk.*.ffn_down.weight", "type": "Q2_K", "m": "1024", "n": "8", "k": "2048",
                     "fusion": "mul_mat_add"}
        identifier = report.operation_id(operation)
        for targeted_count in (5, 3):
            with self.subTest(targeted_count=targeted_count):
                for repetition in (1, 2, 3):
                    bundle = self.root / "profiles/70b/Q2_K/c8/p2048/counters/dominant/decode/gpu0/combined" / f"capture{repetition}"
                    for subpath, count, value in ((".", 2, 1), (f"targeted/{identifier}", targeted_count, 100)):
                        source = bundle / subpath
                        launches = [{"id": str(i), **operation, "metrics": {"dram__bytes_read.sum": {"value": value, "unit": "byte", "available": True}}} for i in range(count)]
                        coverage = {"operation_id": identifier, "operation": operation, "trace_available_launches": 100,
                                    "required_launches": 5, "observed_launches": count, "complete": count >= 5}
                        put(source / "metadata.json", {"status": "captured", "counter_status": "collected",
                                                      "counter_collection": {"metric_groups": ["memory"]}})
                        put(source / "counter_summary.json", {"launches": launches})
                        put(source / "selection.json", {"capture": repetition, "metric_group": "combined", "metric_groups": ["memory"],
                            "operation_coverage": [coverage], "launches": [{"launch_id": str(i), "matching_verified": True, "operation": operation} for i in range(count)]})
                    put(bundle / "coverage.json", {"status": "complete", "operation_coverage": [coverage | {
                        "complete": True, "observed_launches": 5, "source_capture": f"targeted/{identifier}"}]})
                _, raw, summaries, _ = report.profile_tables(self.root)
                accepted = [row for row in summaries if row["sampling_requirement_met"]]
                if targeted_count == 5:
                    self.assertEqual(len(accepted), 1)
                    self.assertEqual(accepted[0]["available_launches"], 5)
                    self.assertEqual(accepted[0]["mean"], 100)
                    self.assertTrue(any(not row["selected_for_summary"] for row in raw))
                else:
                    self.assertEqual(accepted, [])
                    self.assertTrue(all(row["status"] == "incomplete" for row in report.operation_coverage(self.root)[1]))
        # Fewer than five are accepted only with explicit positive trace availability.
        for path in report.profile_paths(self.root, "coverage.json"):
            document = report.read_json(path)
            document["operation_coverage"][0].update(trace_available_launches=2, required_launches=2, observed_launches=3)
            put(path, document)
            source = path.parent / document["operation_coverage"][0]["source_capture"] / "selection.json"
            selection = report.read_json(source)
            selection["operation_coverage"][0].update(trace_available_launches=2, required_launches=2, complete=True)
            put(source, selection)
        _, _, summaries, _ = report.profile_tables(self.root)
        limited = [row for row in summaries if row["sampling_requirement_met"]]
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["required_launches_per_capture"], "2")

    def test_server_batch_limit_is_not_reported_as_observed_width(self):
        path = self.root / "server.log"
        path.write_text("slot launch: id 0 | task 1 | processing task, is_child = 0\n"
                        "slot launch: id 1 | task 2 | processing task, is_child = 0\n"
                        "slot add: id 0 | task 1 | slot decode token, id=20, n_ctx = 2816\n"
                        "slot add: id 1 | task 2 | slot decode token, id=21, n_ctx = 2816\n"
                        "srv decode: n_batch (effective) = 2048, off = 0\n"
                        "slot release: id 0 | task 1 | stop processing: truncated = 0\n"
                        "slot release: id 1 | task 2 | stop processing: truncated = 0\n")
        evidence = report.server_concurrency(path)
        self.assertEqual(evidence["max_active_slots"], 2)
        self.assertTrue(evidence["task_events_complete"])
        self.assertEqual(evidence["decode_sequences_added_per_batch"], {2: 1})
        self.assertIsNone(evidence["matrix_activation_width"])

    def test_nested_family_profiles_and_verified_launch_metadata(self):
        target = self.root / "1b/profiles/Llama-3.2-1B-Instruct/Q4_K_M/c8/p2048/counters/dominant/decode/gpu0/combined/capture1"
        put(target / "metadata.json", {"status": "captured", "counter_status": "collected",
                                      "counter_collection": {"metric_groups": list(report.COUNTER_GROUPS)}})
        operation = {"phase": "decode", "role": "blk.*.ffn_down.weight", "type": "Q4_K", "m": 2048, "n": 8, "k": 8192}
        put(target / "selection.json", {"launches": [{"launch_id": "7", "matching_verified": True, "operation": operation}]})
        put(target / "counter_summary.json", {"launches": [{"id": "7", "kernel": "kernel", "device": "0", "grid": "32x1x1", "block": "32x2x1",
            "metrics": {"gpu__time_duration.sum": {"value": 400, "unit": "nsecond", "available": True}}}]})
        _, _, rows, _ = report.profile_tables(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "1b")
        self.assertEqual(rows[0]["phase"], "decode")
        self.assertEqual(rows[0]["operation_match"], "verified")
        self.assertEqual(rows[0]["metric_groups_covered"], ";".join(report.COUNTER_GROUPS))

    def test_fusion_identity_survives_trace_and_counter_reporting(self):
        target = self.root / "profiles/70b/Q2_K/c8/p2048/counters/decode/gpu0/combined/capture1"
        put(target / "metadata.json", {"status": "captured", "counter_status": "collected"})
        base = {"kernel": "kernel", "device": "0", "grid": "32x1x1", "block": "32x2x1",
                "phase": "decode", "role": "blk.*.ffn_gate.weight+blk.*.ffn_up.weight",
                "type": "Q2_K+Q2_K", "m": "1024", "n": "1", "k": "2048"}
        operations = [base | {"fusion": fusion} for fusion in ("gate_up_silu", "gate_up_gelu")]
        put(target / "selection.json", {"launches": [
            {"launch_id": str(i), "matching_verified": True, "operation": operation}
            for i, operation in enumerate(operations)]})
        put(target / "counter_summary.json", {"launches": [
            {"id": str(i), **operation, "metrics": {
                "gpu__time_duration.sum": {"value": 100 * (i + 1), "unit": "nsecond", "available": True}}}
            for i, operation in enumerate(operations)]})
        (target / "kernel_summary.csv").write_text(
            "kernel,class,device,grid,block,phase,role,type,m,n,k,fusion,calls,total_ms\n" +
            "\n".join(
                ",".join(str(operation[key]) for key in ("kernel",)) + ",quantized_matvec," +
                ",".join(str(operation[key]) for key in ("device", "grid", "block", "phase", "role", "type", "m", "n", "k", "fusion")) + ",5,1"
                for operation in operations) + "\n")
        traces, _, summaries, _ = report.profile_tables(self.root)
        self.assertEqual({row["fusion"] for row in traces}, {row["fusion"] for row in operations})
        self.assertEqual(len(summaries), 2)
        for operation, value in zip(operations, (100, 200)):
            row = next(row for row in summaries if row["fusion"] == operation["fusion"])
            self.assertEqual(row["operation_id"], report.operation_id(operation))
            self.assertEqual(row["mean"], value)

    def test_pending_selected_operation_has_unknown_source_not_a_crash(self):
        target = self.root / "profiles/1b/Q4_K_M/c8/p2048/counters/dominant/decode/gpu0/combined/capture1"
        operation = {"device": "0", "kernel": "kernel", "grid": "32x1x1", "block": "32x2x1",
                     "phase": "decode", "role": "ffn", "type": "Q4_K", "m": "1024", "n": "8", "k": "2048"}
        put(target / "coverage.json", {"status": "incomplete", "operation_coverage": [{
            "operation": operation, "operation_id": report.operation_id(operation), "trace_available_launches": 100,
            "required_launches": 5, "observed_launches": 0, "complete": False, "source_capture": None}]})
        accepted, rows = report.operation_coverage(self.root)
        self.assertEqual(accepted, {})
        self.assertEqual(rows[0]["status"], "incomplete")

    def test_manual_interpretation_review_does_not_survive_changed_evidence(self):
        source = self.root / "source.json"
        put(source, {"evidence": True})
        put(self.root / "report/interpretation-reviewed.json", {
            "status": "reviewed", "reviewer": "root", "note": "Reviewed source evidence",
            "reviewed_utc": "2000-01-01T00:00:00+00:00", "source_paths": ["source.json"]})
        result = report.interpretation_review(self.root)
        self.assertEqual(result["status"], "requires_review")
        self.assertIn("changed after review", ";".join(result["missing_or_invalid"]))

    def test_report_does_not_change_old_studies_or_measurements(self):
        old = self.root.parents[1] / "cuda-study-1b/raw.json"
        put(old, {"original": True})
        fixture(self.root)
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.parents[1].rglob("*") if path.is_file()}
        report.build_report(self.root, plots=False)
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == checksum for path, checksum in before.items()))
        with self.assertRaisesRegex(ValueError, "prior studies"):
            report.build_report(old.parent, plots=False)

    def test_sibling_layout_reports_one_copy_and_links_original_evidence(self):
        root = self.sibling_layout()
        folder = fixture(root)
        old = root.parent / "cuda-study-1b"
        put(old / "summary.json", {"old_study": True})
        put(old / "profiles/1b/Q8_0/c8/p2048/counter_summary.json", {"launches": []})
        before = {path: path.read_bytes() for path in root.parent.rglob("*") if path.is_file()}
        report.build_report(root, plots=False)
        rows = report.read_csv(root / "report/runtime.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence"], "../cuda-context-study-1b/Q4_K_M/c8/p2048")
        canonical = folder.resolve().parents[2]
        self.assertEqual(report.read_csv(canonical / "summary.csv")[0]["evidence"], "Q4_K_M/c8/p2048")
        summary = (canonical / "summary.md").read_text()
        self.assertIn("../cuda-context-study/report/index.md", summary)
        self.assertIn("[r1](Q4_K_M/c8/p2048/r1/raw.json)", summary)
        for family in ("8b", "70b"):
            pending = root.parent / f"cuda-context-study-{family}"
            self.assertIn("Status: **pending**", (pending / "summary.md").read_text())
            self.assertEqual(report.read_csv(pending / "summary.csv"), [])
            self.assertFalse((pending / "manifest.json").exists())
        self.assertEqual(report.profile_paths(root, "counter_summary.json"), [])
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
        self.assertFalse((old / "summary.md").exists())

    def test_sibling_capture_discovery_and_selected_coverage_use_canonical_identity(self):
        root = self.sibling_layout()
        operation = {"device": "0", "kernel": "kernel", "grid": "32x1x1", "block": "32x2x1",
                     "phase": "decode", "role": "ffn", "type": "Q4_K", "m": "1024", "n": "8", "k": "2048", "fusion": "none"}
        identifier = report.operation_id(operation)
        for repetition in (1, 2, 3):
            target = root / "1b/profiles/1b/Q4_K_M/c8/p2048/counters/decode/gpu0/combined" / f"capture{repetition}"
            coverage = {"operation_id": identifier, "operation": operation, "trace_available_launches": 100,
                        "required_launches": 5, "observed_launches": 5, "complete": True}
            put(target / "metadata.json", {"status": "captured", "counter_status": "collected",
                                          "counter_collection": {"metric_groups": ["memory"]}})
            put(target / "counter_summary.json", {"launches": [{"id": str(i), **operation,
                "metrics": {"dram__bytes_read.sum": {"value": 100, "unit": "byte", "available": True}}} for i in range(5)]})
            put(target / "selection.json", {"capture": repetition, "metric_group": "combined", "metric_groups": ["memory"],
                "operation_coverage": [coverage], "launches": [{"launch_id": str(i), "matching_verified": True, "operation": operation} for i in range(5)]})
            put(target / "coverage.json", {"status": "complete", "operation_coverage": [coverage | {"source_capture": "."}]})
        (root / "profiles").symlink_to(root / "1b/profiles", target_is_directory=True)
        self.assertEqual(len(report.profile_paths(root, "counter_summary.json")), 3)
        accepted, coverage = report.operation_coverage(root)
        self.assertEqual(len(accepted), 3)
        self.assertTrue(all(row["status"] == "complete" and row["source_capture"].startswith("../cuda-context-study-1b/") for row in coverage))
        _, observations, summaries, _ = report.profile_tables(root)
        self.assertEqual(len(observations), 15)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0]["sampling_requirement_met"])
        old = root.parent / "cuda-study-1b/capture1"
        put(old / "selection.json", report.read_json(target / "selection.json"))
        put(old / "metadata.json", report.read_json(target / "metadata.json"))
        put(old / "counter_summary.json", report.read_json(target / "counter_summary.json"))
        put(target / "coverage.json", {"status": "complete", "operation_coverage": [coverage[0] | {"operation": operation, "source_capture": str(old)}]})
        accepted, rows = report.operation_coverage(root)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(any("leaves the declared study roots" in row["reason"] for row in rows))

    def test_sibling_review_accepts_declared_evidence_and_rejects_old_study(self):
        root = self.sibling_layout()
        put(root / "1b/evidence.json", {"new_study": True})
        put(root.parent / "cuda-study-1b/evidence.json", {"old_study": True})
        review = {"status": "reviewed", "reviewer": "root", "note": "Reviewed measurement evidence",
                  "reviewed_utc": "2099-01-01T00:00:00+00:00"}
        for name in ("1b/evidence.json", "../cuda-context-study-1b/evidence.json"):
            put(root / "report/interpretation-reviewed.json", review | {"source_paths": [name]})
            self.assertEqual(report.interpretation_review(root)["status"], "complete")
        put(root / "report/interpretation-reviewed.json", review | {"source_paths": ["../cuda-study-1b/evidence.json"]})
        self.assertEqual(report.interpretation_review(root)["status"], "requires_review")

    def test_wrong_family_alias_is_rejected_before_report_writes(self):
        root = self.sibling_layout()
        old = root.parent / "cuda-study-1b"
        put(old / "manifest.json", {"old_study": True})
        (root / "1b").unlink()
        (root / "1b").symlink_to(old, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must point to canonical"):
            report.build_report(root, plots=False)
        self.assertFalse((root / "report").exists())
        self.assertFalse((old / "summary.md").exists())

    def test_custom_series_requires_explicit_family_layout(self):
        root = self.sibling_layout("cuda-context-study-repeat2")
        fixture(root)
        report.build_report(root, plots=False)
        rows = report.read_csv(root / "report/runtime.csv")
        self.assertEqual(rows[0]["evidence"], "../cuda-context-study-repeat2-1b/Q4_K_M/c8/p2048")
        unrelated = root.parent / "other-results"
        unrelated.mkdir()
        with self.assertRaisesRegex(ValueError, "prior studies"):
            report.build_report(unrelated, plots=False)


if __name__ == "__main__":
    unittest.main()
