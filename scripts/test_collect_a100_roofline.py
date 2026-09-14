"""CPU-only checks of roofline coordinates, operator coverage and attribution."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import collect_a100_roofline as roofline
import profile_a100_setting as profiler


def metric(value, unit=""):
    return {"value": value, "unit": unit, "available": True}


def launch():
    return {"id": "0", "device": "0", "kernel": "mul_mat_q", "grid": "(1,1,1)", "block": "(32,1,1)",
            "operation": {"role": "blk.*.ffn_gate.weight", "phase": "prefill", "type": "q4_K",
                          "m": "28672", "n": "512", "k": "8192", "operation_match": "nvtx_same_capture"},
            "metrics": {"gpu__time_duration.sum": metric(1000, "ns"),
                        "dram__bytes_read.sum": metric(80, "byte"),
                        "dram__bytes_write.sum": metric(20, "byte"), roofline.OPCODES: metric(400)},
            "metric_instances": {roofline.OPCODES: [metric(100) | {"instance": "FADD"},
                                                   metric(50) | {"instance": "FMUL"},
                                                   metric(250) | {"instance": "FFMA"}]}}


class RooflineTests(unittest.TestCase):
    def args(self, output, **changes):
        defaults = dict(cells=[Path("Q4"), Path("Q2")], phases=["prefill", "decode"],
                        operators=list(roofline.OPERATORS), launch_count=5, output=output,
                        diagnostic_server=Path("/test/llama-server"), ncu="ncu")
        return SimpleNamespace(**(defaults | changes))

    def cell(self, path):
        quant = {"Q4": "Q4_K_M", "Q2": "Q2_K", "IQ1": "IQ1_M", "Q8": "Q8_0"}[path.name]
        return {"quantization": quant, "model_family": "Llama-3.3-70B-Instruct", "model": {"layers": 80},
                "concurrency": 8, "prompt_tokens": 2048, "cell_file": str(path / "cell.json"),
                "configuration": {"SERVER_CACHE_TYPE_K": "f16", "SERVER_CACHE_TYPE_V": "f16",
                                  "SERVER_TENSOR_SPLIT": "1,1"}}, ["2", "5"]

    def plan(self, root, formats=("Q4", "Q2"), **changes):
        cells = [root / name for name in formats]
        for path in cells:
            path.mkdir(exist_ok=True)
            (path / "cell.json").write_text("{}")
        with patch.object(profiler, "load_cell", side_effect=self.cell):
            return roofline.make_plan(self.args(root, cells=cells, **changes))

    def test_coordinates_use_thread_counts_bytes_and_seconds(self):
        result = roofline.measured(launch())
        self.assertEqual(result["fp32_flops"], 650)
        self.assertEqual(result["dram_bytes"], 100)
        self.assertEqual(result["fp32_flops_per_byte"], 6.5)
        self.assertAlmostEqual(result["fp32_flops_per_s"], 650e6, delta=1e-5)
        for unit, duration in (("us", 1), ("ms", .001), ("s", .000001)):
            sample = launch()
            sample["metrics"]["gpu__time_duration.sum"] = metric(duration, unit)
            self.assertAlmostEqual(roofline.measured(sample)["duration_s"], 1e-6)

    def test_missing_or_partial_counters_are_not_zero_filled(self):
        sample = launch()
        sample["metric_instances"][roofline.OPCODES].pop()
        with self.assertRaisesRegex(ValueError, "inventory"):
            roofline.measured(sample)
        sample = launch()
        sample["metrics"]["dram__bytes_read.sum"]["available"] = False
        with self.assertRaises(ValueError):
            roofline.measured(sample)
        sample = launch()
        sample["metric_instances"][roofline.OPCODES] = [metric(400) | {"instance": "IADD3"}]
        self.assertEqual(roofline.measured(sample)["fp32_flops"], 0)

    def test_plan_targets_every_operator_and_only_the_output_gpu_for_lm_head(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
        self.assertEqual(len(plan["jobs"]), 60)
        self.assertEqual(plan["devices"], ["2", "5"])
        self.assertEqual({job["gpu"] for job in plan["jobs"] if job["operator"] == "lm_head"}, {1})
        for job in plan["jobs"]:
            command = job["command"]
            self.assertIn("--scalar-roofline", command)
            self.assertEqual(command[command.index("--operation-regex") + 1], roofline.operation_regex(job["operator"]))
        self.assertEqual(plan["peak_fp32_flops_per_s"], 19.5e12)  # Per GPU, never doubled.

    def test_four_format_plan_has_120_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory), formats=("IQ1", "Q2", "Q4", "Q8"))
        self.assertEqual(len(plan["jobs"]), 120)
        self.assertEqual({job["format"] for job in plan["jobs"]}, {"IQ1_M", "Q2_K", "Q4_K_M", "Q8_0"})

    def test_mixed_workloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.cell
            def mixed(path):
                cell, devices = original(path)
                cell["prompt_tokens"] = 4096 if path.name == "Q2" else 2048
                return cell, devices
            with patch.object(self, "cell", side_effect=mixed):
                with self.assertRaisesRegex(ValueError, "share concurrency"):
                    self.plan(Path(directory))

    def test_export_preserves_kernel_shapes_and_rejects_wrong_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root, phases=["prefill"], operators=["ffn_gate_up"])
            job = plan["jobs"][0]
            path = root / job["capture"]
            path.mkdir(parents=True)
            source = plan["source_cells"][0]
            roofline.write(path / "study-capture.json", {"exit_code": 0, "source_cell": source["path"],
                "settings": {"PROFILE_DEVICES": "0", "PROFILE_NVTX_INCLUDE":
                    "regex:csb_batch:phase=prefill:.*/*/" + roofline.operation_regex("ffn_gate_up")}})
            roofline.write(path / "metadata.json", {"status": "captured", "counter_status": "collected"})
            other_shape = copy.deepcopy(launch())
            other_shape["id"], other_shape["operation"]["n"] = "1", "256"
            roofline.write(path / "counter_summary.json", {"launches": [launch(), other_shape]})
            points, audit = roofline.export_values(root, plan)
            self.assertEqual(len(points), 2)
            self.assertEqual({point["n"] for point in points}, {"512", "256"})
            self.assertEqual(audit["status"], "incomplete")  # Other requested captures absent.
            self.assertEqual(audit["captured_launches"], 2)
            self.assertEqual(points[0]["physical_gpu"], "2")
            wrong = launch() | {"device": "1"}
            roofline.write(path / "counter_summary.json", {"launches": [wrong]})
            points, audit = roofline.export_values(root, plan)
            self.assertEqual(points, [])
            self.assertIn("differs", audit["captures"][0]["reason"])

    def test_plan_cli_has_no_gpu_probe_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.plan(root)
            output = root / "new-output"
            with patch.object(sys, "argv", ["collect_a100_roofline.py", "plan", "--cells", "Q4", "--output", str(output)]), \
                 patch.object(roofline, "make_plan", return_value=plan), \
                 patch.object(roofline.subprocess, "check_output") as probe, \
                 patch.object(roofline.subprocess, "run") as capture, contextlib.redirect_stdout(io.StringIO()):
                roofline.main()
            probe.assert_not_called()
            capture.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
