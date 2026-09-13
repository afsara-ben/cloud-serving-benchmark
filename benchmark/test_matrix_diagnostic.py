"""CPU-only checks for diagnostic capture identity and evidence gates."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matrix_diagnostic as matrix
import profile_cuda


def summary_for(groups, count=5):
    definition = profile_cuda.combine_metric_groups(groups)
    names = set(definition["metrics"].split(","))
    for section in definition["sections"]:
        names.update(profile_cuda.SECTION_REQUIREMENTS[section])
    return {"launches": [{"id": str(index), "device": "0", "kernel": "mul_mat_vec_q_test",
                           "class": "quantized_matvec", "grid": "(128, 1, 1)", "block": "(32, 1, 1)",
                           "metrics": {name: {"value": 1.0, "available": True} for name in names},
                           "metric_instances": {"sass__inst_executed_per_opcode": [
                               {"instance": "IADD3", "value": 1.0, "available": True}]}}
                         for index in range(count)]}


class CaptureValidationTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(tool="ncu", metric_groups="memory,instructions,occupancy,stalls,tensor",
                      separate_metric_groups=False, warmup_iterations=3, warmup_ms=250.0)
        return argparse.Namespace(**(values | overrides))

    def test_combined_default_avoids_five_repeated_captures(self):
        groups = matrix.capture_groups(self.args())
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]["metric_groups"]), set(profile_cuda.METRIC_GROUPS))
        metrics = groups[0]["metrics"].split(",")
        self.assertEqual(len(metrics), len(set(metrics)))
        self.assertEqual(len(matrix.capture_groups(self.args(separate_metric_groups=True))), 5)

    def test_gpu_paths_and_selected_metric_unions_do_not_collide(self):
        case = dict(variant="baseline", shape="ffn_gate_up", type="Q2_K", n=8)
        group = matrix.capture_groups(self.args())[0]
        first = matrix.capture_folder(Path("output"), case, 0, group, "0")
        second = matrix.capture_folder(Path("output"), case, 0, group, "1")
        self.assertNotEqual(first, second)
        self.assertIn("gpu1", second.parts)
        self.assertNotEqual(first, matrix.capture_folder(Path("output"), case, 0,
                            matrix.capture_groups(self.args(metric_groups="memory,instructions"))[0], "0"))
        with self.assertRaises(ValueError):
            matrix.capture_folder(Path("output"), case, 0, group, "0,1")

    def test_valid_full_counter_capture_passes(self):
        group = matrix.capture_groups(self.args())[0]
        self.assertTrue(matrix.validate_matrix_counters(summary_for(group["metric_groups"]), group, 5)["valid"])

    def test_nan_requested_metric_and_missing_opcode_instances_fail(self):
        group = matrix.capture_groups(self.args())[0]
        summary = summary_for(group["metric_groups"])
        summary["launches"][0]["metrics"]["dram__bytes_read.sum"]["value"] = float("nan")
        self.assertFalse(matrix.validate_matrix_counters(summary, group, 5)["valid"])
        summary = summary_for(group["metric_groups"])
        summary["launches"][0]["metric_instances"].clear()
        validation = matrix.validate_matrix_counters(summary, group, 5)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["section_coverage"]["InstructionStats"]["valid"])

    def test_total_launch_count_does_not_hide_incomplete_auxiliary_cohort(self):
        group = matrix.capture_groups(self.args(metric_groups="memory"))[0]
        summary = summary_for(group["metric_groups"])
        auxiliary = copy.deepcopy(summary["launches"][:4])
        for launch in auxiliary:
            launch.update(kernel="quantize_row_q8_1", **{"class": "activation_quantization"})
        summary["launches"] += auxiliary
        validation = matrix.validate_matrix_counters(summary, group, 5)
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["incomplete_kernel_cohorts"][0]["launches"], 4)

    def test_auxiliary_kernels_without_a_matrix_do_not_pass(self):
        group = matrix.capture_groups(self.args(metric_groups="memory"))[0]
        summary = summary_for(group["metric_groups"])
        for launch in summary["launches"]:
            launch["class"] = "activation_quantization"
        self.assertFalse(matrix.validate_matrix_counters(summary, group, 5)["valid"])

    def test_warmup_requires_both_duration_and_operation_count_after_correctness(self):
        identity = dict(m=28672, n=8, k=8192, seed=20260912, phase="decode", ggml_type="q2_K",
                        shape="ffn_gate_up", iterations=5, warmup_iterations=3, warmup_ms=250.0)
        operation = {"schema": 2, "m": 28672, "n": 8, "k": 8192, "seed": 20260912, "phase": "decode",
                     "type": "q2_K", "role": "ffn_gate_up", "backend": "CUDA0",
                     "warmup": 3, "warmup_ms": 250.0, "warmup_actual_iterations": 2000,
                     "warmup_actual_ms": 250.1, "warmup_position": "after_correctness_before_capture",
                     "correctness": {"passed": True, "relative_l2": .01, "relative_l2_tolerance": .03,
                                     "max_error_over_reference_rms": .02, "maximum_normalized_tolerance": .15},
                     "complete_operation_us": [10.0] * 5}
        self.assertEqual(matrix.validate_operation(operation, identity), [])
        for bad in [dict(warmup_actual_ms=249.9), dict(warmup_actual_iterations=2),
                    dict(warmup_position="before_correctness")]:
            self.assertTrue(matrix.validate_operation(operation | bad, identity))

    def test_run_marks_only_valid_capture_complete_and_reuses_it(self):
        self.exercise_run(valid=True)

    def test_invalid_counter_run_preserves_diagnostics_without_complete_marker(self):
        self.exercise_run(valid=False)

    def exercise_run(self, valid):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variant = root / "build/baseline"
            variant.mkdir(parents=True)
            (variant / "build.json").write_text(json.dumps({"cuda_library_sha256": "hash", "binary_sha256": "hash", "harness_sha256": "hash"}))
            args = self.args(output=root / "results", build_root=root / "build", gpu="1", repetitions=1,
                             iterations=20, profile_iterations=5, seed=20260912, cache_control="all", ncu="fake-ncu",
                             shapes="ffn_gate_up", types="Q2_K", widths="8", variants="baseline", phase="decode")
            group = matrix.capture_groups(args)[0]
            summary = summary_for(group["metric_groups"])
            if not valid:
                summary["launches"][0]["metrics"]["dram__bytes_read.sum"]["value"] = float("nan")

            def fake_execute(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                operation = {"schema": 2, "m": 28672, "n": 8, "k": 8192, "seed": args.seed, "phase": "decode",
                             "type": "q2_K", "role": "ffn_gate_up", "backend": "CUDA0",
                             "warmup": 3, "warmup_ms": 250.0, "warmup_actual_iterations": 2000,
                             "warmup_actual_ms": 250.1, "warmup_position": "after_correctness_before_capture",
                             "correctness": {"passed": True, "relative_l2": .01, "relative_l2_tolerance": .03,
                                             "max_error_over_reference_rms": .02, "maximum_normalized_tolerance": .15},
                             "complete_operation_us": [10.0] * 5}
                output.write_text(json.dumps(operation))
                output.with_name("capture.ncu-rep").write_text("fixture report")

            def fake_summarize(csv_path, folder):
                (folder / "counter_summary.json").write_text(json.dumps(summary))
                return summary

            with mock.patch.object(matrix, "sha", return_value="hash"), \
                 mock.patch.object(matrix, "gpu_snapshot", return_value={"available": True}), \
                 mock.patch.object(matrix, "execute", side_effect=fake_execute) as execute, \
                 mock.patch.object(profile_cuda, "summarize_ncu", side_effect=fake_summarize), \
                 contextlib.redirect_stdout(io.StringIO()):
                if valid:
                    matrix.run(args)
                    matrix.run(args)
                    self.assertEqual(execute.call_count, 1)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Counter validation failed"):
                        matrix.run(args)
            runs = list(args.output.rglob("run.json"))
            self.assertEqual(len(runs), 1)
            folder = runs[0].parent
            self.assertIn("gpu1", folder.parts)
            self.assertEqual((folder / "complete.json").exists(), valid)
            validation = json.loads((folder / "counter-validation.json").read_text())
            self.assertEqual(validation["valid"], valid)
            metadata = json.loads(runs[0].read_text())
            self.assertEqual(metadata["status"], "complete" if valid else "failed")
            self.assertEqual(len(metadata["identity"]["metric_groups"]), 5)


if __name__ == "__main__":
    unittest.main()
