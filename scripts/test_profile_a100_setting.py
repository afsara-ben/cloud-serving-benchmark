"""CPU-only tests of bounded single-setting profiling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_a100_setting as launcher


class ProfileSettingTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(action="counters", phase="decode", device=1, launch_count=5, launch_skip=0,
                      roofline=["tensor"], kernel_regex="mul_mat_q")
        return SimpleNamespace(**(values | changes))

    def fixture(self, directory, concurrency=8):
        cell = {"status": "complete", "concurrency": concurrency, "prompt_tokens": 2048, "output_tokens": 512,
                "model_family": "Llama-3.1-8B-Instruct", "quantization": "Q4_K_M",
                "configuration": {"GPU_DEVICE": "2,5", "LLAMA_CPP_REF": launcher.PIN}}
        (directory / "cell.json").write_text(json.dumps(cell))
        (directory / "study.env").write_text("# fixture")
        (directory / "prompts.json").write_text("[]")
        (directory / "r1").mkdir()
        (directory / "r1/raw.json").write_text("{}")
        return directory

    def test_counter_settings_bound_total_launches_to_one_phase_and_one_device(self):
        settings = launcher.capture_settings(self.args(), 2)
        self.assertEqual(settings["PROFILE_FILTER_MODE"], "global")
        self.assertEqual(settings["PROFILE_LAUNCH_COUNT"], "5")
        self.assertEqual(settings["PROFILE_DEVICES"], "1")
        self.assertEqual(settings["PROFILE_EXPECTED_DEVICES"], "1")
        self.assertEqual(settings["PROFILE_REQUIRE_NVTX"], "1")
        self.assertEqual(settings["PROFILE_NVTX_INCLUDE"], "regex:csb_batch:phase=decode:.*/")
        self.assertIn("SpeedOfLight_HierarchicalTensorRooflineChart", settings["PROFILE_SECTIONS"])
        with self.assertRaises(ValueError):
            launcher.capture_settings(self.args(device=1), 1)

    def test_trace_keeps_both_server_devices_visible(self):
        settings = launcher.capture_settings(self.args(action="trace"), 2)
        self.assertEqual(settings, {"PROFILE_TOOL": "nsys", "PROFILE_EXPECTED_DEVICES": "0,1"})

    def test_plan_reads_one_setting_without_gpu_probe_capture_or_output_writes(self):
        for concurrency in (8, 16, 32, 64):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as temporary:
                cell = self.fixture(Path(temporary), concurrency)
                output = io.StringIO()
                with patch.object(sys, "argv", ["profile_a100_setting.py", "plan", "--cell", str(cell), "--device", "1"]), \
                     patch.object(launcher, "run_capture") as capture, patch.object(launcher, "check_devices") as probe, \
                     contextlib.redirect_stdout(output):
                    launcher.main()
                plan = json.loads(output.getvalue())
                self.assertEqual(plan["counter_physical_gpu"], "5")
                self.assertEqual(plan["warmup_requests"], concurrency)
                self.assertEqual(plan["profiled_requests"], concurrency)
                self.assertFalse(Path(plan["output"]).exists())
                capture.assert_not_called()
                probe.assert_not_called()

    def test_missing_saved_requests_stop_before_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            cell = self.fixture(Path(temporary))
            (cell / "r1/raw.json").unlink()
            with self.assertRaisesRegex(ValueError, "completed serving cell"):
                launcher.load_cell(cell)

    def test_execution_schedules_exactly_one_capture_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell = self.fixture(root)
            extras = root / "extras/python"
            extras.mkdir(parents=True)
            (extras / "ncu_report.py").write_text("# fixture")
            with patch.object(sys, "argv", ["profile_a100_setting.py", "counters", "--cell", str(cell), "--device", "1"]), \
                 patch.object(launcher, "ROOT", root), patch.object(launcher, "validate_build"), \
                 patch.object(launcher, "validate_sections"), patch.object(launcher, "check_devices"), \
                 patch.object(launcher.shutil, "which", return_value=str(root / "ncu")), \
                 patch.dict(launcher.os.environ), patch.object(launcher, "run_capture", return_value=root / "captured") as capture, \
                 contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            capture.assert_called_once()
            self.assertEqual(capture.call_args.kwargs["attempts"], 1)
            self.assertEqual(capture.call_args.args[0]["cell_file"], str((cell / "cell.json").resolve()))
            self.assertEqual(capture.call_args.args[2]["PROFILE_DEVICES"], "1")

    def test_section_preflight_does_not_accept_roofline_name_as_base_section(self):
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(stdout="SpeedOfLight_RooflineChart\n")) as command:
            with self.assertRaisesRegex(ValueError, "lacks sections: SpeedOfLight"):
                launcher.validate_sections("ncu", ["SpeedOfLight"])
            self.assertEqual(command.call_args.args[0], ["ncu", "--config-file", "off", "--list-sections"])


if __name__ == "__main__":
    unittest.main()
