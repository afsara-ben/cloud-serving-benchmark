"""Boundary and evidence checks for the context runner, with no CUDA access."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import capacity

spec = importlib.util.spec_from_file_location("context_runner", Path(__file__).resolve().parents[1] / "scripts/08_run_cuda_study.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CapacityTests(unittest.TestCase):
    def test_output_reservation_and_output_layer_affect_split(self):
        self.assertEqual(capacity.slot_capacity(2048), 2816)
        self.assertEqual(capacity.slot_capacity(65536), 66304)
        placement = capacity.layer_devices(80, [1, 1])
        self.assertEqual(placement[:-1].count(0), 41)
        self.assertEqual(placement[:-1].count(1), 39)
        self.assertEqual(placement[-1], 1)

    def test_per_device_limit_is_not_aggregate_fit(self):
        gpu = [{"index": str(i), "total_bytes": 48 * capacity.GIB, "free_bytes": 48 * capacity.GIB} for i in range(2)]
        metadata = {"available": True, "source": "test", "layers": 80, "kv_heads": 8, "head_dimension": 128,
                    "layer_devices": capacity.layer_devices(80, [1, 1]), "weight_bytes_per_gpu": [40 * capacity.GIB, 0]}
        result = capacity.estimate(metadata, gpu, 2048, 32)
        self.assertEqual(result["status"], "capacity_estimated")
        self.assertFalse(result["gpus"][0]["fits_lower_bound"])
        self.assertTrue(result["gpus"][1]["fits_lower_bound"])
        self.assertLess(sum(row["minimum_required_bytes"] for row in result["gpus"]), 96 * capacity.GIB)

    def test_absent_fp16_screen_needs_no_file_download_or_reader(self):
        model = {"quant": "FP16", "parameters": 70553706496, "layers": 80, "kv_heads": 8, "head_dimension": 128}
        metadata = capacity.inspect_model("/missing/70b.gguf", model, "/missing/backend", [1, 1])
        gpu = [{"free_bytes": 48 * capacity.GIB} for _ in range(2)]
        self.assertEqual(capacity.estimate(metadata, gpu, 2048, 8)["status"], "capacity_estimated")
        larger = [{"free_bytes": 192 * capacity.GIB} for _ in range(2)]
        self.assertEqual(capacity.estimate(metadata, larger, 2048, 8)["status"], "model_missing")

    def test_native_limit_includes_output(self):
        result = capacity.estimate({"native_context": 131072}, [], 131072, 8)
        self.assertEqual(result["status"], "native_context_unsupported")

    def test_placement_rejects_fallback_and_wrong_context(self):
        metadata = {"layers": 16, "layer_devices": capacity.layer_devices(16, [1, 1])}
        log = "offloaded 17/17 layers to GPU\nn_seq_max = 8\nn_ctx_seq = 2816\nCUDA0 KV buffer size = 10.00 MiB\nCUDA1 KV buffer size = 10.00 MiB\ntype_k = f16\ntype_v = f16\nK (f16): 10.00 MiB, V (f16): 10.00 MiB\n"
        self.assertTrue(capacity.validate_placement(log, metadata, 8, 2048)["valid"])
        for changed in (log.replace("17/17", "16/17"), log.replace("2816", "2048"), log + "CPU KV buffer size = 1.00 MiB\n", log.replace("type_k = f16", "type_k = q8_0")):
            self.assertFalse(capacity.validate_placement(changed, metadata, 8, 2048)["valid"])

    def test_failure_types_are_separate(self):
        self.assertEqual(capacity.classify_failure("cudaMalloc failed: out of memory", True), "observed_oom")
        self.assertEqual(capacity.classify_failure("CUDA error: illegal memory access"), "hardware_error")
        self.assertEqual(capacity.classify_failure("server failed", True), "startup_error")


class RunnerTests(unittest.TestCase):
    def test_optional_template_is_resolved_validated_and_hashable(self):
        self.assertEqual(runner.chat_template_controls({}), {"SERVER_CHAT_TEMPLATE_FILE": ""})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            template = root / "chat.jinja"
            template.write_text("{{ messages[0]['content'] }}")
            for value in (str(template), str(template.relative_to(root))):
                from unittest import mock
                with mock.patch.object(runner, "ROOT", root):
                    model = {"chat_template_file": value}
                    self.assertEqual(runner.chat_template_path(model), template)
                    self.assertEqual(runner.chat_template_controls(model)["SERVER_CHAT_TEMPLATE_FILE"], str(template))
                    self.assertEqual(len(runner.digest(runner.chat_template_path(model))), 64)
            template.unlink()
            with self.assertRaisesRegex(RuntimeError, "chat template does not exist"):
                runner.chat_template_controls({"chat_template_file": str(template)})

    def test_context_controls_preserve_legacy_configuration(self):
        legacy = {"SERVER_PARALLEL": "8", "SERVER_CTX_SIZE": "32768", "MAX_OUTPUT_TOKENS": "128"}
        result = runner.context_controls(legacy, 64, 16384)
        self.assertEqual(result["SERVER_CTX_SIZE"], str(64 * (16384 + 768)))
        self.assertEqual(result["SERVER_PARALLEL"], "64")
        self.assertEqual(result["SERVER_CACHE_TYPE_K"], "f16")
        self.assertEqual(result["SERVER_FIT"], "off")
        self.assertEqual(legacy["MAX_OUTPUT_TOKENS"], "128")

    def test_requested_grid_excludes_nonpowers_and_duplicate_cells(self):
        args = SimpleNamespace(concurrencies="8,64", prompt_lengths="2048,16384", capacity_lengths="")
        self.assertEqual(runner.context_arguments(args), ([8, 64], [2048, 16384], []))
        args.prompt_lengths = "4608"
        with self.assertRaises(ValueError):
            runner.context_arguments(args)

    def test_memory_sampling_rejects_missing_device_and_headroom_failure(self):
        devices = [{"index": str(i), "uuid": f"GPU-{i}", "total_bytes": 48 * capacity.GIB} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "telemetry.csv"
            path.write_text("timestamp, index, memory.used [MiB], memory.free [MiB]\n2026, 0, 1024, 47000\n2026, 1, 46000, 2400\n")
            self.assertTrue(runner.context_memory_summary(path, devices, 2 * capacity.GIB)["valid"])
            path.write_text("timestamp, index, memory.used [MiB], memory.free [MiB]\n2026, 0, 1024, 47000\n2026, 1, 46000, 1800\n")
            self.assertFalse(runner.context_memory_summary(path, devices, 2 * capacity.GIB)["valid"])
            path.write_text("timestamp, index, memory.used [MiB]\n2026, 0, 1024\n")
            self.assertFalse(runner.context_memory_summary(path, devices, 2 * capacity.GIB)["valid"])

    def test_resume_validates_actual_prompt_and_output_not_legacy_constants(self):
        record = {"ok": True, "prompt_tokens": 16384, "prepared_prompt_tokens": 16384, "evaluated_prompt_tokens": 16384, "cached_prompt_tokens": 0}
        data = {"configuration": {"requests": 8, "concurrency": 8, "max_output_tokens": 512,
                                  "target_prompt_tokens": 16384, "exact_prompt_tokens": True},
                "summary": {"successful_requests": 8, "failed_requests": 0, "output_length_invalid_requests": 0,
                            "cache_counter_available_requests": 8, "cached_prompt_tokens": 0, "max_client_inflight": 8},
                "requests": [record.copy() for _ in range(8)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps(data))
            self.assertTrue(runner.valid(path, 8, 8, prompt_tokens=16384, output_tokens=512, exact_prompt=True))
            self.assertFalse(runner.valid(path, 8, 8))
            data["requests"][0]["prompt_tokens"] = 16383
            path.write_text(json.dumps(data))
            self.assertFalse(runner.valid(path, 8, 8, prompt_tokens=16384, output_tokens=512, exact_prompt=True))


if __name__ == "__main__":
    unittest.main()
