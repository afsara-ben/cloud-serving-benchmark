"""Offline validation of one/two-GPU serving exports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_serving as serving
import report_context_study as context
from test_report_context_study import fixture, put


class ServingReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "results" / "cuda-context-study-a100-test"
        self.root.mkdir(parents=True)
        for model in serving.MODELS:
            target = self.root.parent / f"{self.root.name}-{model}"
            target.mkdir()
            (self.root / model).symlink_to(Path("..") / target.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_cell(self, devices):
        put(self.root / "serving-scope.json", serving.scope_document(devices))
        folder = fixture(self.root)
        manifest_path = self.root / "1b/manifest.json"
        manifest = context.read_json(manifest_path)
        manifest.update(devices=[{"index": gpu} for gpu in devices], concurrencies=list(serving.CONCURRENCIES),
                        prompt_lengths=list(serving.PROMPTS), capacity_lengths=[], repetitions=1,
                        models=[{"quant": quant, "sha256": "weights"} for quant in serving.FORMATS["1b"]])
        put(manifest_path, manifest)
        fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        progress_path = self.root / "1b/progress.json"
        progress = context.read_json(progress_path)
        progress["fingerprint"] = fingerprint
        put(progress_path, progress)
        cell = context.read_json(folder / "cell.json")
        cell.update(fingerprint=fingerprint, repetitions=1)
        cell["configuration"]["SERVER_TENSOR_SPLIT"] = ",".join("1" for _ in devices)
        put(folder / "cell.json", cell)
        raw = context.read_json(folder / "r1/raw.json")
        raw["study_configuration"] = cell | {"repetition": 1}
        put(folder / "r1/raw.json", raw)
        placement = context.read_json(folder / "r1/placement.json")
        placement["kv_buffers_mib"] = {f"CUDA{index}": "256.0" for index in range(len(devices))}
        put(folder / "r1/placement.json", placement)
        for path in (folder / "r1/resources.json", folder / "warmup-resources.json"):
            resources = context.read_json(path)
            resources["memory"]["gpus"] = [resources["memory"]["gpus"][0] | {"index": gpu} for gpu in devices]
            put(path, resources)
        (folder / "r1/telemetry.csv").write_text(
            "timestamp,index,memory.used [MiB]\n" + "".join(
                f"1970/01/01 00:00:01.100,{gpu},{3072 * (index + 1)}\n" for index, gpu in enumerate(devices))
            + f"1970/01/01 00:00:04.100,{devices[0]},70000\n")
        return folder

    def test_single_physical_gpu_index_is_reported_without_inventing_second_gpu(self):
        self.make_cell(["3"])
        audit = serving.build_report(self.root, plots=False)
        rows = context.read_csv(self.root / "serving-report/runtime.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["gpu3_peak_vram_gib"]), 3.)
        self.assertNotIn("gpu0_peak_vram_gib", rows[0])
        self.assertEqual(float(rows[0]["output_tokens_per_second_mean"]), 4096.)
        self.assertEqual(float(rows[0]["ttft_p95_ms"]), 200.)
        self.assertAlmostEqual(float(rows[0]["tpot_p95_ms"]), 1800 / 511)
        self.assertEqual(audit["required_cells"], 224)
        self.assertEqual(audit["measured_cells"], 1)
        self.assertFalse(any(row["model"] == "70b" and row["format"] == "FP16"
                             for row in context.read_csv(self.root / "serving-report/capacity.csv")))

    def test_two_gpu_memory_uses_physical_indices_and_plots_are_generated(self):
        self.make_cell(["2", "5"])
        audit = serving.build_report(self.root, plots=True)
        row = context.read_csv(self.root / "serving-report/runtime.csv")[0]
        self.assertEqual(float(row["gpu2_peak_vram_gib"]), 3.)
        self.assertEqual(float(row["gpu5_peak_vram_gib"]), 6.)
        self.assertEqual(len(audit["plots"]), 8)
        self.assertTrue(all((self.root / "serving-report" / path).is_file() for path in audit["plots"]))

    def test_plots_include_measurements_at_all_four_concurrencies(self):
        from matplotlib.figure import Figure
        rows = [{"model": "8b", "format": "Q4_K_M", "concurrency": concurrency,
                 "input_tokens": 4096, "output_tokens_per_second_mean": float(concurrency)}
                for concurrency in (8, 16, 32, 64)]
        throughputs = []

        def inspect_plot(figure, path, **kwargs):
            if path.name == "8b-throughput.png":
                throughputs.extend((axis.get_title(), axis.lines[0].get_ydata()[1]) for axis in figure.axes)

        with patch.object(Figure, "savefig", autospec=True, side_effect=inspect_plot):
            serving.plot_metrics(rows, self.root / "plots", ["0"])
        self.assertEqual(throughputs, [("8 clients", 8.), ("16 clients", 16.),
                                      ("32 clients", 32.), ("64 clients", 64.)])

    def test_resources_from_wrong_gpu_are_rejected(self):
        folder = self.make_cell(["3"])
        resources = context.read_json(folder / "r1/resources.json")
        resources["memory"]["gpus"][0]["index"] = "0"
        put(folder / "r1/resources.json", resources)
        serving.build_report(self.root, plots=False)
        self.assertEqual(context.read_csv(self.root / "serving-report/runtime.csv"), [])

    def test_missing_measured_memory_samples_are_rejected(self):
        folder = self.make_cell(["0"])
        (folder / "r1/telemetry.csv").write_text("timestamp,index,memory.used [MiB]\n1970/01/01 00:00:04.100,0,70000\n")
        serving.build_report(self.root, plots=False)
        self.assertEqual(context.read_csv(self.root / "serving-report/runtime.csv"), [])

    def test_original_two_gpu_report_still_rejects_single_gpu_layout(self):
        self.make_cell(["0"])
        context.build_report(self.root, plots=False)
        self.assertEqual(context.read_csv(self.root / "report/runtime.csv"), [])


if __name__ == "__main__":
    unittest.main()
