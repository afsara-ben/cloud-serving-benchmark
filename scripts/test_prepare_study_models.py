"""Offline tests of download safety and model provenance configuration."""
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).with_name("09_prepare_study_models.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ModelPreparationTests(unittest.TestCase):
    def test_all_model_sizes_include_five_requested_formats(self):
        for size in ("1b", "8b", "70b"):
            self.assertEqual({entry['quant'] for entry in prepare.initial_manifest(size)},
                             {"FP16", "Q8_0", "Q4_K_M", "Q2_K", "IQ1_M"})

    def test_multipart_assembly_preserves_bytes_and_rejects_changed_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            pieces = [b'GGUFfirst', b'second']
            paths = []
            entries = []
            for index, contents in enumerate(pieces):
                path = root / f'part{index}'
                path.write_bytes(contents)
                paths.append(path)
                entries.append({'filename': path.name, 'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()})
            entry = {'model_size': '70b', 'quant': 'Q8_0', 'repo': 'public/repo', 'revision': 'pinned',
                     'filename': 'assembled.gguf', 'bytes': sum(map(len, pieces)), 'parts': entries}
            args = SimpleNamespace(source_dir=root)
            with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'fetch', side_effect=paths):
                self.assertEqual(prepare.assemble_parts(entry, args, {}).read_bytes(), b''.join(pieces))
            paths[0].write_bytes(b'GGUFwrong')
            with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'fetch', side_effect=paths):
                with self.assertRaisesRegex(RuntimeError, 'Part integrity mismatch'):
                    prepare.assemble_parts(entry, args, {})

    def test_70b_fp16_rejected_on_original_hardware(self):
        entry = prepare.initial_manifest("70b")[0]
        with self.assertRaisesRegex(RuntimeError, "No 70B FP16 model files were requested"):
            prepare.memory_screen(entry, [48 * prepare.GIB, 48 * prepare.GIB])

    def test_70b_fp16_available_for_larger_hardware(self):
        entry = prepare.initial_manifest("70b")[0]
        screen = prepare.memory_screen(entry, [80 * prepare.GIB, 80 * prepare.GIB])
        self.assertGreater(screen["usable_bytes"], screen["weight_lower_bound_bytes"])

    def test_remote_only_guard_precedes_download(self):
        entry = prepare.initial_manifest("70b")[0]
        args = SimpleNamespace(include_remote_only=True, devices="0,1", log_dir=prepare.ROOT / "results/nonexistent-test")
        with patch.object(prepare, "gpu_memory", return_value=[48 * prepare.GIB] * 2), \
             patch.object(prepare, "convert_fp16") as conversion, patch.object(prepare, "fetch") as fetch:
            with self.assertRaises(RuntimeError):
                prepare.prepare(entry, args)
            conversion.assert_not_called()
            fetch.assert_not_called()

    def test_remote_only_default_does_not_probe_or_fetch(self):
        entry = prepare.initial_manifest("70b")[0]
        args = SimpleNamespace(include_remote_only=False, log_dir=prepare.ROOT / "results/nonexistent-test")
        with patch.object(prepare, "gpu_memory") as probe, patch.object(prepare, "fetch") as fetch:
            self.assertEqual(prepare.prepare(entry, args), entry)
            probe.assert_not_called()
            fetch.assert_not_called()

    def test_70b_fp16_guard_survives_missing_remote_only_metadata(self):
        entry = prepare.initial_manifest("70b")[0]
        del entry["remote_only"]
        args = SimpleNamespace(include_remote_only=False, log_dir=prepare.ROOT / "results/nonexistent-test")
        with patch.object(prepare, "convert_fp16") as conversion:
            self.assertEqual(prepare.prepare(entry, args), entry)
            conversion.assert_not_called()

    def test_full_precision_source_is_pinned_original_checkpoint(self):
        for model in ("1b", "8b", "70b"):
            entry = prepare.initial_manifest(model)[0]
            self.assertEqual(entry["quant"], "FP16")
            self.assertTrue(entry["repo"].startswith("meta-llama/"))
            self.assertRegex(entry["revision"], r"^[a-f0-9]{40}$")
            self.assertEqual(entry["native_context"], 131072)


if __name__ == "__main__":
    unittest.main()
