import copy
import hashlib
import unittest

from context_protocol import fingerprint, producer_manifest, transition_errors


class ContextProtocolTests(unittest.TestCase):
    def setUp(self):
        self.old_config = 'GPU_DEVICE="0,1"\nREPETITIONS="3"\n'
        self.new_config = self.old_config.replace('"3"', '"1"')
        self.old = {"repetitions": 3, "configuration": {"REPETITIONS": "3", "SERVER_BATCH_SIZE": "2048"},
                    "models": [{"quant": "Q4_K_M", "sha256": "original"}], "binary_sha256": "binary",
                    "code_sha256": {"/study/runner.py": "same", "/study/config/context-study.env":
                                    hashlib.sha256(self.old_config.encode()).hexdigest()}}
        self.new = copy.deepcopy(self.old)
        self.new["repetitions"] = 1
        self.new["configuration"]["REPETITIONS"] = "1"
        self.new["models"].append({"quant": "Q2_K", "sha256": "added"})
        self.new["code_sha256"]["/study/config/context-study.env"] = hashlib.sha256(self.new_config.encode()).hexdigest()

    def test_add_formats_and_reduce_runs_preserves_original_producer(self):
        self.assertEqual(transition_errors(self.old, self.new, self.old_config, self.new_config), [])
        history = {"current_fingerprint": fingerprint(self.new), "selected_repetition": 1,
                   "configuration_source": self.new_config,
                   "producers": {fingerprint(self.old): {"manifest": self.old, "configuration_source": self.old_config}}}
        self.assertEqual(producer_manifest(self.new, history, fingerprint(self.old)), self.old)
        history["producers"][fingerprint(self.old)]["manifest"] = dict(self.old, binary_sha256="forged")
        self.assertIsNone(producer_manifest(self.new, history, fingerprint(self.old)))

    def test_changed_workload_source_or_existing_weights_cannot_be_adopted(self):
        for kind in ("binary", "source", "weights", "batch", "removed_model"):
            candidate = copy.deepcopy(self.new)
            if kind == "binary": candidate["binary_sha256"] = "different"
            if kind == "source": candidate["code_sha256"]["/study/runner.py"] = "different"
            if kind == "weights": candidate["models"][0]["sha256"] = "different"
            if kind == "batch": candidate["configuration"]["SERVER_BATCH_SIZE"] = "512"
            if kind == "removed_model": candidate["models"].pop(0)
            with self.subTest(kind=kind):
                self.assertTrue(transition_errors(self.old, candidate, self.old_config, self.new_config))

    def test_configuration_hash_and_contents_both_must_match(self):
        self.assertTrue(transition_errors(self.old, self.new, self.old_config, self.new_config + '# changed\n'))
        candidate = copy.deepcopy(self.new)
        changed = self.new_config.replace('"0,1"', '"0"')
        candidate["code_sha256"]["/study/config/context-study.env"] = hashlib.sha256(changed.encode()).hexdigest()
        self.assertTrue(transition_errors(self.old, candidate, self.old_config, changed))


if __name__ == "__main__":
    unittest.main()
