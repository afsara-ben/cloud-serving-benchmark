"""Exercise shell orchestration without downloads, transfers, builds or GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("rtxpro6000_pipeline.sh")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rtx-shell-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts" / SCRIPT.name
        shutil.copy2(SCRIPT, self.script)
        self.venv = self.root / "venv"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin/activate").write_text("# Synthetic environment; no real activation.\n")
        self.calls = self.root / "calls.jsonl"
        self.env = dict(os.environ, CSB_VENV=str(self.venv), CSB_MODEL_DIR=str(self.root / "models"),
                        NCU_BIN="/synthetic/ncu", STUDY_NAME="cuda-context-study-rtx-shell-test", GPU_DEVICE="0,1",
                        CSB_TEST_CALLS=str(self.calls))
        self.env.pop("CUDA_HOME", None)
        self.env.pop("CSB_NVTX_INCLUDE", None)
        executable = self.venv / "bin/python3"
        executable.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
            "with open(os.environ['CSB_TEST_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.exit(42 if sys.argv[1:3] == ['scripts/run_rtxpro6000.py', 'build'] else 0)\n")
        executable.chmod(0o755)

    def invoke(self, *arguments):
        return subprocess.run([str(self.script), *arguments], cwd=self.root, env=self.env,
                              capture_output=True, text=True, timeout=15)

    def test_dry_run_covers_both_contexts_and_does_not_execute_or_write(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        result = self.invoke("all", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--prompt-lengths 32768", result.stdout)
        self.assertIn("--prompt-tokens 2048", result.stdout)
        self.assertIn("--prompt-tokens 32768", result.stdout)
        self.assertIn("--cuda-arch 120", result.stdout)
        self.assertNotIn("+ rsync", result.stdout)
        self.assertFalse(self.calls.exists())
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_failed_build_stops_before_editable_install_prepare_and_gpu_runs(self):
        (self.root / "models").mkdir()
        for name in ("Q2_K", "Q4_K_M"):
            (self.root / "models" / f"Llama-3.3-70B-Instruct-{name}.gguf").write_text("synthetic")
        # Provide command discovery only; all real work is intercepted by fake Python.
        tools = self.root / "tools"
        tools.mkdir()
        for command in ("nvcc", "git", "c++", "curl", "flock"):
            path = tools / command
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
        self.env["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        result = self.invoke("all")
        self.assertEqual(result.returncode, 42, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][:2], ["scripts/run_rtxpro6000.py", "build"])
        self.assertIn("Later stages were not run", result.stderr)
        self.assertFalse((self.root / "results").exists())

    def test_missing_models_stop_before_python_work(self):
        result = self.invoke("setup")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Q2_K.gguf", result.stderr)
        self.assertIn("Q4_K_M.gguf", result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "models").exists())

    def test_report_uses_only_saved_data_builders(self):
        result = self.invoke("report")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0][:2], ["scripts/run_rtxpro6000.py", "report"])
        self.assertEqual(calls[1][0], "paper/poster/build.py")
        self.assertTrue(all(call[0] == "paper/runtime-bottlenecks-report/build.py" for call in calls[2:]))
        self.assertIn("c8-p32768", calls[-1][-1])

    def test_transfer_preview_and_remote_shell_argument_validation(self):
        result = self.invoke("transfer-models", "--dry-run", "--remote-host", "user@cluster-login")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("+ rsync", result.stdout)
        self.assertIn("user@cluster-login:/scratch/cloud-serving-benchmark/models/", result.stdout)
        self.assertNotIn("scripts/run_rtxpro6000.py", result.stdout)
        self.assertFalse(self.calls.exists())
        result = self.invoke("transfer-models", "--dry-run", "--remote-dir", "/scratch/$(touch BAD)")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("+ ssh", result.stdout)
        self.assertFalse((self.root / "BAD").exists())

    def test_fetch_reads_remote_sources_without_requiring_local_models(self):
        result = self.invoke("fetch-models", "--dry-run", "--source-host", "user@source",
                             "--source-dir", "/data/models")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("user@source:/data/models/Llama-3.3-70B-Instruct-Q2_K.gguf", result.stdout)
        self.assertIn("user@source:/data/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf", result.stdout)
        self.assertIn(str(self.root / "models") + "/", result.stdout)
        self.assertLess(result.stdout.index("+ ssh"), result.stdout.index("+ rsync"))
        self.assertFalse((self.root / "models").exists())
        self.assertFalse(self.calls.exists())
        result = self.invoke("fetch-models", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --source-host", result.stderr)
        result = self.invoke("fetch-models", "--dry-run", "--source-host", "user@source",
                             "--source-dir", "/data/$(touch BAD)")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("+ ssh", result.stdout)

    def test_fetch_stops_after_failed_source_check(self):
        tools = self.root / "tools"
        tools.mkdir()
        for command, body in (("ssh", "exit 23"), ("rsync", "touch RSYNC_WAS_RUN")):
            path = tools / command
            path.write_text("#!/bin/sh\n" + body + "\n")
            path.chmod(0o755)
        self.env["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        result = self.invoke("fetch-models", "--source-host", "user@source")
        self.assertEqual(result.returncode, 23)
        self.assertFalse((self.root / "models").exists())
        self.assertFalse((self.root / "RSYNC_WAS_RUN").exists())

    def test_fetch_stores_files_in_selected_local_directory_and_links_checkout(self):
        tools = self.root / "tools"
        tools.mkdir()
        ssh = tools / "ssh"
        ssh.write_text("#!/bin/sh\nexit 0\n")
        ssh.chmod(0o755)
        rsync = tools / "rsync"
        rsync.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nimport sys\n"
                         "for source in sys.argv[sys.argv.index('--') + 1:-1]:\n"
                         "    (Path(sys.argv[-1]) / Path(source).name).write_text('synthetic GGUF')\n")
        rsync.chmod(0o755)
        self.env["PATH"] = str(tools) + os.pathsep + os.environ["PATH"]
        storage = self.root / "NVMe storage"
        result = self.invoke("fetch-models", "--source-host", "user@source", "--model-dir", str(storage))
        self.assertEqual(result.returncode, 0, result.stderr)
        for quant in ("Q2_K", "Q4_K_M"):
            name = f"Llama-3.3-70B-Instruct-{quant}.gguf"
            stored, linked = storage / name, self.root / "models" / name
            self.assertEqual(stored.read_text(), "synthetic GGUF")
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.resolve(), stored.resolve())
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
