"""Offline checks of cell selection and stable resume identity."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


launcher = module("a100_launcher", "run_a100_serving.py")
runner = module("context_runner", "08_run_cuda_study.py")
import capacity


class A100LauncherTests(unittest.TestCase):
    def test_default_scope_and_individual_selection(self):
        args = SimpleNamespace(models=["70b", "8b", "1b"], formats=None, concurrencies="8,16,32,64",
                               prompt_lengths="2048,4096,8192,16384")
        selected = launcher.selection(args)
        self.assertEqual(list(selected), ["1b", "8b", "70b"])
        self.assertEqual(sum(map(len, selected.values())), 224)
        self.assertFalse(any(cell.startswith("FP16/") for cell in selected["70b"]))
        args.models, args.formats, args.concurrencies, args.prompt_lengths = ["8b"], "Q4_K_M", "64", "4096"
        self.assertEqual(launcher.selection(args), {"8b": ["Q4_K_M/c64/p4096"]})
        args.models, args.formats = ["1b", "8b", "70b"], "FP16"
        self.assertEqual(list(launcher.selection(args)), ["1b", "8b"])
        args.models = ["70b"]
        with self.assertRaisesRegex(ValueError, "70B FP16 is excluded"):
            launcher.selection(args)

    def test_plan_new_concurrencies_preserves_request_budget_and_full_runner_scope(self):
        for devices in ("0", "0,1"):
            with self.subTest(devices=devices):
                output = io.StringIO()
                with patch.object(sys, "argv", ["run_a100_serving.py", "plan", "--gpus", devices,
                                                "--models", "8b", "--formats", "Q4_K_M",
                                                "--concurrencies", "16,32", "--prompt-lengths", "4096"]), \
                     patch.object(launcher, "check_devices") as probe, \
                     patch.object(launcher, "run_command") as execute, contextlib.redirect_stdout(output):
                    launcher.main()
                plan = json.loads(output.getvalue())
                cells = ["Q4_K_M/c16/p4096", "Q4_K_M/c32/p4096"]
                self.assertEqual(plan["selected_cells"], {"8b": cells})
                self.assertEqual(plan["devices"], devices.split(","))
                self.assertEqual(plan["settings"], 2)
                self.assertEqual(plan["warmup_requests_if_all_feasible"], 48)
                self.assertEqual(plan["measured_requests_if_all_feasible"], 96)
                self.assertFalse(plan["inference_executed"])
                probe.assert_not_called()
                execute.assert_not_called()
                command = launcher.study_command("study", "8b", cells)
                self.assertEqual(command[command.index("--concurrencies") + 1], "8,16,32,64")
                self.assertEqual([command[i + 1] for i, arg in enumerate(command) if arg == "--only-cell"], cells)

    def test_invalid_selector_cannot_silently_run_full_grid(self):
        cells = [{"quant": "Q4_K_M", "concurrency": 8, "prompt_tokens": 2048}]
        with self.assertRaisesRegex(ValueError, "Unknown --only-cell"):
            runner.selected_context_cells(cells, ["Q4_K_M/c64/p2048"])

    def test_separate_cell_invocations_preserve_manifest_and_accumulate_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "study.env"
            config.write_text("# fixture")
            manifest = root / "models.json"
            manifest.write_text(json.dumps([{"quant": "Q4_K_M", "model_family": "1b", "filename": "test.gguf"}]))
            model = root / "models/test.gguf"
            model.parent.mkdir()
            model.write_bytes(b"GGUFfixture")
            args = SimpleNamespace(concurrencies="8,16,32,64", prompt_lengths="2048,4096,8192,16384", capacity_lengths="",
                                   formats="Q4_K_M", manifest=manifest, output_dir=root / "output", repetitions=1,
                                   dry_run=False, only_cell=["Q4_K_M/c16/p2048"])
            cfg = {"LLAMA_CPP_REF": runner.PIN, "SERVER_BATCH_SIZE": "2048", "SERVER_UBATCH_SIZE": "512",
                   "GPU_LAYERS": "99", "FLASH_ATTN": "on", "SERVER_TENSOR_SPLIT": "1", "SERVER_MAIN_GPU": "0",
                   "GPU_DEVICE": "0", "SERVER_HOST": "127.0.0.1", "SERVER_PORT": "8080"}
            hardware = [{"index": "0", "uuid": "test-gpu", "name": "A100", "total_bytes": 80 * capacity.GIB,
                         "free_bytes": 79 * capacity.GIB, "used_bytes": capacity.GIB}]
            with patch.object(runner, "ROOT", root), patch.object(runner, "digest", return_value="fixed-hash"), \
                 patch.object(runner, "verified_model", side_effect=lambda entry: entry | {"sha256": "weights"}), \
                 patch.object(runner, "monitoring_commands", return_value={}), \
                 patch.object(runner.subprocess, "check_output", return_value=runner.PIN), \
                 patch.object(runner.subprocess, "run") as inference, \
                 patch.object(capacity, "query_memory", return_value=hardware), \
                 patch.object(capacity, "inspect_model", return_value={}), \
                 patch.object(capacity, "estimate", return_value={"status": "capacity_estimated", "reason": "fixture"}) as estimate, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run_context_study(args, cfg.copy(), {}, "cuda", ["0"], config)
                first = json.loads((args.output_dir / "manifest.json").read_text())
                args.only_cell = ["Q4_K_M/c32/p4096"]
                runner.run_context_study(args, cfg.copy(), {}, "cuda", ["0"], config)
                second = json.loads((args.output_dir / "manifest.json").read_text())
                # A previously accepted first run must skip startup and warmup.
                progress_path = args.output_dir / "progress.json"
                progress = json.loads(progress_path.read_text())
                progress["cells"]["Q4_K_M/c16/p2048/r1"] = {"status": "complete"}
                progress_path.write_text(json.dumps(progress))
                args.only_cell = ["Q4_K_M/c16/p2048"]
                with patch.object(runner, "valid", return_value=True):
                    runner.run_context_study(args, cfg.copy(), {}, "cuda", ["0"], config)
                self.assertEqual(estimate.call_count, 2)
                inference.assert_not_called()
            self.assertEqual(first, second)
            progress = json.loads((args.output_dir / "progress.json").read_text())
            self.assertEqual(set(progress["cells"]), {"Q4_K_M/c16/p2048", "Q4_K_M/c16/p2048/r1", "Q4_K_M/c32/p4096"})


if __name__ == "__main__":
    unittest.main()
