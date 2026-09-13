"""Focused regression checks for benchmark measurement validity (no GPU needed)."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

import load_test
import summarize


def result() -> dict:
    return {
        "ok": True,
        "error": None,
        "duration_seconds": 1.0,
        "ttft_seconds": 0.3,
        "last_content_seconds": 0.9,
        "finish_reason": "length",
        "server_usage": {
            "prompt_tokens": 128,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "server_timings": {"prompt_n": 128, "predicted_n": 8, "prompt_ms": 250, "predicted_ms": 600},
    }


class StreamingMetricsTests(unittest.TestCase):
    def test_exact_input_validation_rejects_truncation_and_missing_accounting(self):
        for reported, evaluated in [(127, 127), (128, 127), (128, None)]:
            with self.subTest(reported=reported, evaluated=evaluated):
                record = result()
                record["server_usage"]["prompt_tokens"] = reported
                record["server_timings"]["prompt_n"] = evaluated
                load_test.finalize_record(record, 128, 8, expected_prompt_tokens=128)
                self.assertFalse(record["ok"])
        record = result()
        load_test.finalize_record(record, 128, 8, expected_prompt_tokens=128)
        self.assertTrue(record["ok"])

    def test_exact_prompt_rejects_target_below_template_size(self):
        tokenizer = mock.Mock()
        tokenizer.count_messages.return_value = 100
        with self.assertRaisesRegex(ValueError, "exceeding target"):
            load_test.make_messages(tokenizer, "GPU serving", 0, 64, exact=True)

    def test_prompt_preparation_matches_server_special_token_policy(self):
        tokenizer = load_test.ServerTokenizer("http://localhost", 1)
        with mock.patch.object(load_test, "post_json", side_effect=[{"prompt": "chat prompt"}, {"tokens": [1, 2, 3]}]) as post:
            self.assertEqual(tokenizer.count_messages([]), 3)
        self.assertEqual(post.call_args.args[1], {"content": "chat prompt", "add_special": True, "parse_special": True})

    def test_uses_server_tokens_and_preserves_final_usage_timings(self):
        events = [
            {"choices": [{"delta": {"content": "one visible word"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
            {"choices": [], "usage": result()["server_usage"], "timings": result()["server_timings"]},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body.encode())) as post:
            record = load_test.stream_completion("http://localhost", "model", [], 8, 1, 1)
        payload = json.loads(post.call_args.args[0].data)
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertFalse(payload["cache_prompt"])
        load_test.finalize_record(record, 127, 8)
        self.assertTrue(record["ok"])
        self.assertEqual(record["completion_tokens"], 8)
        self.assertEqual(record["prompt_tokens"], 128)
        self.assertEqual(record["prepared_prompt_tokens"], 127)
        self.assertEqual(record["server_timings"]["prompt_ms"], 250)
        self.assertEqual(record["completion_token_count_source"], "server_usage")

    def test_incomplete_stream_is_rejected_even_after_content(self):
        body = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
            record = load_test.stream_completion("http://localhost", "model", [], 8, 1, 1)
        self.assertFalse(record["ok"])
        self.assertIn("without [DONE]", record["error"])

    def test_short_output_cache_reuse_and_missing_counts_are_rejected(self):
        for change in ("short", "cached", "missing"):
            with self.subTest(change=change):
                record = result()
                if change == "short":
                    record["server_usage"]["completion_tokens"] = 7
                elif change == "cached":
                    record["server_usage"]["prompt_tokens_details"]["cached_tokens"] = 64
                else:
                    del record["server_usage"]["completion_tokens"]
                    del record["server_timings"]["predicted_n"]
                load_test.finalize_record(record, 128, 8)
                self.assertFalse(record["ok"])
                self.assertTrue(record["validation_errors"])

    def test_client_tpot_uses_server_token_count(self):
        record = result()
        load_test.finalize_record(record, 128, 8)
        self.assertAlmostEqual(record["tpot_seconds"], 0.1)
        self.assertAlmostEqual(record["content_tpot_seconds"], 0.6 / 7)

    def test_required_prefix_reuse_keeps_logical_and_evaluated_counts_distinct(self):
        record = result()
        record["server_usage"]["prompt_tokens_details"]["cached_tokens"] = 127
        record["server_timings"]["prompt_n"] = 1
        load_test.finalize_record(record, 128, 8, cache_policy="require")
        self.assertTrue(record["ok"])
        self.assertEqual(record["prompt_tokens"], 128)
        self.assertEqual(record["total_tokens"], 136)
        self.assertEqual(record["evaluated_prompt_tokens"], 1)

    def test_required_reuse_rejects_misses_and_inconsistent_accounting(self):
        for cached, evaluated in [(0, 128), (None, 128), (127, 128), (127, None)]:
            with self.subTest(cached=cached, evaluated=evaluated):
                record = result()
                record["server_usage"]["prompt_tokens_details"]["cached_tokens"] = cached
                record["server_timings"]["prompt_n"] = evaluated
                load_test.finalize_record(record, 128, 8, cache_policy="require")
                self.assertFalse(record["ok"])
                self.assertTrue(record["validation_errors"])

    def test_overlap_integrates_intervals_and_drain(self):
        intervals = [
            {"start_offset_seconds": 0.0, "end_offset_seconds": 2.0},
            {"start_offset_seconds": 1.0, "end_offset_seconds": 3.0},
        ]
        stats = load_test.interval_statistics(intervals)
        self.assertEqual(stats["max_client_inflight"], 2)
        self.assertAlmostEqual(stats["mean_client_inflight"], 4 / 3)
        self.assertAlmostEqual(stats["concurrent_client_wall_fraction"], 1 / 3)

    def test_full_closed_loop_run_preserves_counts_and_concurrency(self):
        def completion(*args, **kwargs):
            started_unix = time.time()
            started = time.perf_counter()
            time.sleep(0.025)
            record = result()
            record.update(_started=started, _finished=time.perf_counter())
            record["started_unix_seconds"] = started_unix
            record["duration_seconds"] = record["_finished"] - started
            record["ttft_seconds"] = record["duration_seconds"] / 2
            record["last_content_seconds"] = record["duration_seconds"]
            return record

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "topics.txt").write_text("GPU serving\n", encoding="utf-8")
            argv = [
                "load_test.py", "--base-url", "http://localhost", "--model", "mock-model",
                "--topics", str(root / "topics.txt"), "--concurrency", "2", "--requests", "4",
                "--target-prompt-tokens", "128", "--max-output-tokens", "8",
                "--warmup-requests", "0", "--output", str(root / "result.json"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(load_test, "make_messages", return_value=([], 128)), mock.patch.object(load_test, "stream_completion", side_effect=completion), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(load_test.main(), 0)
            document = json.loads((root / "result.json").read_text())
            self.assertEqual(document["schema_version"], 3)
            self.assertEqual(document["summary"]["successful_requests"], 4)
            self.assertEqual(document["summary"]["max_client_inflight"], 2)
            self.assertEqual(document["summary"]["mean_completion_tokens"], 8)
            self.assertEqual(document["summary"]["cached_prompt_tokens"], 0)
            self.assertAlmostEqual(document["summary"]["finished_unix_seconds"] - document["summary"]["started_unix_seconds"], document["summary"]["wall_seconds"], places=6)

            # Different prompt/output workloads must not be merged into one row.
            second = copy.deepcopy(document)
            second["configuration"]["target_prompt_tokens"] = 256
            (root / "second.json").write_text(json.dumps(second))
            repeated = copy.deepcopy(document)
            repeated["configuration"]["repeat_prompt"] = True
            (root / "repeated.json").write_text(json.dumps(repeated))
            reused = copy.deepcopy(repeated)
            reused["configuration"]["prefix_reuse"] = True
            (root / "reused.json").write_text(json.dumps(reused))
            argv = ["summarize.py", str(root), "--output-csv", str(root / "summary.csv"), "--output-markdown", str(root / "summary.md")]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(summarize.main(), 0)
            self.assertEqual(len((root / "summary.csv").read_text().splitlines()), 5)

    def test_repeat_workload_matches_inputs_and_counts_across_reuse_modes(self):
        def completion(*args, cache_prompt=False):
            started = time.perf_counter()
            unix = time.time()
            time.sleep(0.01)
            record = result()
            record.update(_started=started, _finished=time.perf_counter(), started_unix_seconds=unix)
            record["duration_seconds"] = record["_finished"] - started
            record["ttft_seconds"] = record["duration_seconds"] / 2
            record["last_content_seconds"] = record["duration_seconds"]
            if cache_prompt:
                record["server_usage"]["prompt_tokens_details"]["cached_tokens"] = 127
                record["server_timings"]["prompt_n"] = 1
            return record

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "topics.txt").write_text("GPU serving\n", encoding="utf-8")
            documents = []
            for reuse in (False, True):
                output = root / f"reuse-{reuse}.json"
                argv = [
                    "load_test.py", "--base-url", "http://localhost", "--model", "mock-model",
                    "--topics", str(root / "topics.txt"), "--concurrency", "2", "--requests", "4",
                    "--target-prompt-tokens", "128", "--max-output-tokens", "8",
                    "--warmup-requests", "0", "--repeat-prompt", "--output", str(output),
                ] + (["--prefix-reuse"] if reuse else [])
                with mock.patch.object(sys, "argv", argv), mock.patch.object(load_test, "make_messages", return_value=([{"role": "user", "content": "same input"}], 128)) as prepare, mock.patch.object(load_test, "stream_completion", side_effect=completion) as stream, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(load_test.main(), 0)
                self.assertEqual(prepare.call_count, 1)
                self.assertTrue(all(call.kwargs["cache_prompt"] == reuse for call in stream.call_args_list))
                document = json.loads(output.read_text())
                documents.append(document)
                self.assertEqual(document["summary"]["logical_prompt_tokens"], 512)
                self.assertEqual(document["summary"]["evaluated_prompt_tokens"], 4 if reuse else 512)
                self.assertEqual(document["summary"]["cache_hit_requests"], 4 if reuse else 0)
                self.assertEqual(len(set(r["input_sha256"] for r in document["requests"])), 1)
                self.assertEqual(document["configuration"]["input_sha256"], document["requests"][0]["input_sha256"])
            self.assertEqual(documents[0]["configuration"]["input_sha256"], documents[1]["configuration"]["input_sha256"])


if __name__ == "__main__":
    unittest.main()
