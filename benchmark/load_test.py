#!/usr/bin/env python3
"""Closed-loop concurrent load generator for llama-server.

The client uses the OpenAI-compatible streaming chat endpoint. Prompt lengths
are prepared with llama-server's own chat template and tokenizer before the
timed region. No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import pathlib
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "llama-cloud-serving-bench/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class ServerTokenizer:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        response = post_json(
            f"{self.base_url}/apply-template",
            {"messages": messages},
            self.timeout,
        )
        return str(response["prompt"])

    def count_text(self, text: str, add_special: bool = False) -> int:
        response = post_json(
            f"{self.base_url}/tokenize",
            {"content": text, "add_special": add_special, "parse_special": True},
            self.timeout,
        )
        return len(response["tokens"])

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        # Match handle_completions_impl: tokenize_input_prompts(..., true, true).
        # Some chat templates omit BOS; counting without add_special undercounts.
        return self.count_text(self.apply_template(messages), add_special=True)


def make_messages(
    tokenizer: ServerTokenizer,
    topic: str,
    request_index: int,
    target_tokens: int,
    exact: bool = False,
) -> tuple[list[dict[str, str]], int]:
    system_message = {
        "role": "system",
        "content": "You are a concise systems-analysis assistant. Follow the requested output format.",
    }
    prefix = (
        f"Benchmark request {request_index:04d}. The subject is: {topic}\n\n"
        "Use the following synthetic evidence as context. "
    )
    suffix = (
        "\n\nBased only on the supplied context, write a numbered technical explanation. "
        "Continue until the response reaches the configured output-token limit."
    )
    filler_unit = (
        f"Record {request_index:04d} describes request scheduling, prompt processing, "
        "decoder iteration, cache traffic, compute utilization, and service latency. "
    )
    filler = filler_unit * max(64, target_tokens // 4)

    def candidate(characters: int) -> tuple[list[dict[str, str]], int]:
        messages = [
            system_message,
            {"role": "user", "content": prefix + filler[:characters] + suffix},
        ]
        return messages, tokenizer.count_messages(messages)

    base_messages, base_count = candidate(0)
    if base_count >= target_tokens:
        if exact and base_count != target_tokens:
            raise ValueError(f"Chat template and instructions require {base_count} tokens, exceeding target {target_tokens}")
        return base_messages, base_count

    low = 0
    high = len(filler)
    best_messages = base_messages
    best_count = base_count
    best_characters = 0
    while low <= high:
        middle = (low + high) // 2
        messages, token_count = candidate(middle)
        if token_count <= target_tokens:
            best_messages = messages
            best_count = token_count
            best_characters = middle
            low = middle + 1
        else:
            high = middle - 1

    if not exact or best_count == target_tokens:
        return best_messages, best_count
    # Token counts can change non-monotonically at a word boundary. Inspect the
    # nearby boundary before adding a small, deterministic padding suffix.
    for characters in range(max(0, best_characters - 32), min(len(filler), best_characters + 32) + 1):
        messages, count = candidate(characters)
        if count == target_tokens:
            return messages, count
    gap = target_tokens - best_count
    for unit in (" x", " .", " 0", "\n"):
        for count in range(max(1, gap - 2), gap + 4):
            messages = [dict(message) for message in best_messages]
            messages[-1]["content"] += unit * count
            if tokenizer.count_messages(messages) == target_tokens:
                return messages, target_tokens
    raise ValueError(f"Could not prepare exactly {target_tokens} prompt tokens; nearest baseline was {best_count}")


def stream_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    timeout: float,
    seed: int,
    cache_prompt: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": cache_prompt,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "llama-cloud-serving-bench/1.0"},
        method="POST",
    )

    started_unix_seconds = time.time()
    started = time.perf_counter()
    first_token_at: float | None = None
    last_content_at: float | None = None
    chunks: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    done_received = False
    content_events = 0
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_received = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(f"Server stream error: {event['error']}")
                if isinstance(event.get("usage"), dict):
                    usage.update(event["usage"])
                if isinstance(event.get("timings"), dict):
                    timings.update(event["timings"])
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        last_content_at = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = last_content_at
                        chunks.append(str(content))
                        content_events += 1
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        error = f"HTTP {exc.code}: {body[:500]}"
    except Exception as exc:  # noqa: BLE001 - errors must be recorded per request
        error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    if error is None and not done_received:
        error = "Stream ended without [DONE]"
    if error is None and first_token_at is None:
        error = "No streamed content was received"
    return {
        "ok": error is None,
        "error": error,
        "_started": started,
        "_finished": finished,
        "started_unix_seconds": started_unix_seconds,
        "finished_unix_seconds": started_unix_seconds + (finished - started),
        "duration_seconds": finished - started,
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "last_content_seconds": None if last_content_at is None else last_content_at - started,
        "stream_content_events": content_events,
        "stream_done_received": done_received,
        "finish_reason": finish_reason,
        "server_usage": usage,
        "server_timings": timings,
        "output_characters": sum(len(chunk) for chunk in chunks),
    }


def finalize_record(
    record: dict[str, Any], prepared_prompt_tokens: int, expected_output_tokens: int,
    cache_policy: str = "forbid",
    expected_prompt_tokens: int | None = None,
) -> None:
    """Use generated-token counters, never retokenized visible output, for rates."""
    if cache_policy not in ("forbid", "require"):
        raise ValueError(f"Unknown cache policy: {cache_policy}")
    usage = record["server_usage"]
    timings = record["server_timings"]
    completion_tokens = usage.get("completion_tokens")
    record["completion_token_count_source"] = "server_usage"
    if completion_tokens is None:
        completion_tokens = timings.get("predicted_n")
        record["completion_token_count_source"] = "server_timings" if completion_tokens is not None else None
    prompt_tokens = usage.get("prompt_tokens", prepared_prompt_tokens)
    record["prompt_token_count_source"] = "server_usage" if "prompt_tokens" in usage else "prepared_tokenizer"
    record["prepared_prompt_tokens"] = prepared_prompt_tokens
    record["prompt_tokens"] = prompt_tokens
    record["completion_tokens"] = completion_tokens
    record["total_tokens"] = None if completion_tokens is None else prompt_tokens + completion_tokens
    details = usage.get("prompt_tokens_details") or {}
    record["cached_prompt_tokens"] = details.get("cached_tokens", timings.get("cache_n"))
    record["server_prompt_tokens_evaluated"] = timings.get("prompt_n")
    record["evaluated_prompt_tokens"] = timings.get("prompt_n")

    issues = []
    if expected_prompt_tokens is not None:
        if prepared_prompt_tokens != expected_prompt_tokens or prompt_tokens != expected_prompt_tokens:
            issues.append(f"Expected {expected_prompt_tokens} input tokens; prepared {prepared_prompt_tokens}, server reported {prompt_tokens}")
        cached_count = record["cached_prompt_tokens"]
        evaluated_count = record["evaluated_prompt_tokens"]
        if not isinstance(cached_count, int) or not isinstance(evaluated_count, int):
            issues.append("Exact prompt validation requires evaluated and cached token counts")
        elif evaluated_count + cached_count != expected_prompt_tokens:
            issues.append("Evaluated plus cached tokens disagree with the exact input target")
    if not isinstance(completion_tokens, int) or completion_tokens < 0:
        issues.append("Missing or invalid server completion-token count")
    elif completion_tokens != expected_output_tokens:
        issues.append(f"Expected {expected_output_tokens} output tokens; server reported {completion_tokens}")
    if "predicted_n" in timings and completion_tokens is not None and timings["predicted_n"] != completion_tokens:
        issues.append("Server usage and timing completion-token counts disagree")
    if record.get("finish_reason") != "length":
        issues.append(f"Expected finish_reason=length; received {record.get('finish_reason')!r}")
    cached = record["cached_prompt_tokens"]
    evaluated = record["evaluated_prompt_tokens"]
    if cache_policy == "forbid" and cached is not None and cached > 0:
        issues.append(f"Server reused {record['cached_prompt_tokens']} cached prompt tokens")
    if cache_policy == "require":
        if not isinstance(cached, int) or cached <= 0:
            issues.append("Prefix reuse required but no positive server cache count was reported")
        if not isinstance(evaluated, int) or evaluated < 0:
            issues.append("Prefix reuse requires a valid server evaluated-prompt count")
        elif isinstance(cached, int) and evaluated + cached != prompt_tokens:
            issues.append("Evaluated plus cached prompt tokens disagree with logical input-token count")
    record["validation_errors"] = issues
    record["fixed_output_length_valid"] = completion_tokens == expected_output_tokens
    if issues:
        record["ok"] = False
        record["error"] = "; ".join(([record["error"]] if record.get("error") else []) + issues)

    ttft = record.get("ttft_seconds")
    last_content = record.get("last_content_seconds")
    if ttft is not None and isinstance(completion_tokens, int) and completion_tokens > 1:
        # Standard client TPOT includes the small final stream/usage delivery tail.
        record["tpot_seconds"] = (record["duration_seconds"] - ttft) / (completion_tokens - 1)
        record["content_tpot_seconds"] = None if last_content is None else (last_content - ttft) / (completion_tokens - 1)
    else:
        record["tpot_seconds"] = None
        record["content_tpot_seconds"] = None


def interval_statistics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Measure client in-flight requests; this alone does not prove GPU batching."""
    events = []
    for record in records:
        events.extend(((record["start_offset_seconds"], 1), (record["end_offset_seconds"], -1)))
    active = maximum = 0
    area = 0.0
    concurrent_seconds = 0.0
    previous = 0.0
    for timestamp, change in sorted(events):
        duration = timestamp - previous
        area += active * duration
        if active > 1:
            concurrent_seconds += duration
        active += change
        maximum = max(maximum, active)
        previous = timestamp
    return {
        "max_client_inflight": maximum,
        "mean_client_inflight": area / previous if previous else 0.0,
        "concurrent_client_wall_fraction": concurrent_seconds / previous if previous else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--topics", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, required=True)
    parser.add_argument("--exact-prompt-tokens", action="store_true", help="Require the prepared and server-reported input counts to match the target exactly")
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--repeat-prompt", action="store_true", help="Use the first prepared prompt for every request (best-case reuse workload)")
    parser.add_argument("--prefix-reuse", action="store_true", help="Request prefix reuse and require a positive cache hit for each measured request; warm all slots first")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--prompts-output", type=pathlib.Path, help="Save prepared messages and their exact counts/hashes before timing")
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must both be positive")
    if args.requests < args.concurrency:
        parser.error("requests must be at least as large as concurrency")
    if args.target_prompt_tokens < 1 or args.max_output_tokens < 1 or args.timeout <= 0 or args.warmup_requests < 0:
        parser.error("token lengths and timeout must be positive; warmup requests must be nonnegative")

    topics = [line.strip() for line in args.topics.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not topics:
        parser.error("topics file is empty")

    tokenizer = ServerTokenizer(args.base_url, args.timeout)
    prepared: list[dict[str, Any]] = []
    precision = "exactly" if args.exact_prompt_tokens else "approximately"
    print(f"Preparing {args.requests} prompts at {precision} {args.target_prompt_tokens} tokens...", flush=True)
    for index in range(args.requests):
        if args.repeat_prompt and prepared:
            prepared.append(prepared[0])
            continue
        messages, prompt_tokens = make_messages(
            tokenizer,
            topics[index % len(topics)],
            index,
            args.target_prompt_tokens,
            **({"exact": True} if args.exact_prompt_tokens else {}),
        )
        input_sha256 = hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        prepared.append({"messages": messages, "prompt_tokens": prompt_tokens, "input_sha256": input_sha256})

    if args.prompts_output is not None:
        args.prompts_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompts_output.write_text(json.dumps(prepared, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for warmup_index in range(args.warmup_requests):
        warmup = stream_completion(
            args.base_url,
            args.model,
            prepared[warmup_index % len(prepared)]["messages"],
            min(16, args.max_output_tokens),
            args.timeout,
            args.seed + 100_000 + warmup_index,
            cache_prompt=args.prefix_reuse,
        )
        if not warmup["ok"]:
            print(f"Warmup failed: {warmup['error']}", file=sys.stderr)
            return 2

    records: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    first_wave = threading.Barrier(args.concurrency)

    def run_one(index: int) -> tuple[int, dict[str, Any]]:
        if index < args.concurrency:
            first_wave.wait(timeout=args.timeout)
        record = stream_completion(
            args.base_url,
            args.model,
            prepared[index]["messages"],
            args.max_output_tokens,
            args.timeout,
            args.seed + index,
            cache_prompt=args.prefix_reuse,
        )
        return index, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, index) for index in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            index, record = future.result()
            record["request_index"] = index
            record["input_sha256"] = prepared[index]["input_sha256"]
            finalize_record(record, prepared[index]["prompt_tokens"], args.max_output_tokens,
                            cache_policy="require" if args.prefix_reuse else "forbid",
                            expected_prompt_tokens=args.target_prompt_tokens if args.exact_prompt_tokens else None)
            records.append(record)

    driver_wall_seconds = time.perf_counter() - wall_started
    first_record = min(records, key=lambda record: record["_started"])
    first_request_started = first_record["_started"]
    wall_seconds = max(record["_finished"] for record in records) - first_request_started
    records.sort(key=lambda item: item["request_index"])

    for record in records:
        record["start_offset_seconds"] = record.pop("_started") - first_request_started
        record["end_offset_seconds"] = record.pop("_finished") - first_request_started

    successful = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    ttft = [record["ttft_seconds"] for record in successful if record.get("ttft_seconds") is not None]
    tpot = [record["tpot_seconds"] for record in successful if record.get("tpot_seconds") is not None]
    e2e = [record["duration_seconds"] for record in successful]
    prompt_tokens_total = sum(record["prompt_tokens"] for record in successful)
    completion_tokens_total = sum(record["completion_tokens"] for record in successful)
    evaluated_counts = [record["evaluated_prompt_tokens"] for record in successful]
    evaluated_prompt_tokens_total = sum(evaluated_counts) if all(isinstance(count, int) for count in evaluated_counts) else None

    summary = {
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_seconds": wall_seconds,
        "started_unix_seconds": first_record["started_unix_seconds"],
        "finished_unix_seconds": first_record["started_unix_seconds"] + wall_seconds,
        "driver_wall_seconds": driver_wall_seconds,
        **interval_statistics(records),
        "output_length_invalid_requests": sum(not record["fixed_output_length_valid"] for record in records),
        "cache_counter_available_requests": sum(record["cached_prompt_tokens"] is not None for record in records),
        "cached_prompt_tokens": sum(record["cached_prompt_tokens"] or 0 for record in records),
        "cache_hit_requests": sum(isinstance(record["cached_prompt_tokens"], int) and record["cached_prompt_tokens"] > 0 for record in records),
        "logical_prompt_tokens": prompt_tokens_total,
        "evaluated_prompt_tokens": evaluated_prompt_tokens_total,
        "evaluated_prompt_tokens_per_second": evaluated_prompt_tokens_total / wall_seconds if wall_seconds and evaluated_prompt_tokens_total is not None else None,
        "requests_per_second": len(successful) / wall_seconds if wall_seconds else None,
        "prompt_tokens_per_second": prompt_tokens_total / wall_seconds if wall_seconds else None,
        "output_tokens_per_second": completion_tokens_total / wall_seconds if wall_seconds else None,
        "total_tokens_per_second": (prompt_tokens_total + completion_tokens_total) / wall_seconds if wall_seconds else None,
        "ttft_p50_ms": None if not ttft else percentile(ttft, 0.50) * 1000.0,
        "ttft_p95_ms": None if not ttft else percentile(ttft, 0.95) * 1000.0,
        "tpot_p50_ms": None if not tpot else percentile(tpot, 0.50) * 1000.0,
        "tpot_p95_ms": None if not tpot else percentile(tpot, 0.95) * 1000.0,
        "e2e_p50_ms": None if not e2e else percentile(e2e, 0.50) * 1000.0,
        "e2e_p95_ms": None if not e2e else percentile(e2e, 0.95) * 1000.0,
        "mean_prompt_tokens": statistics.fmean(record["prompt_tokens"] for record in successful) if successful else None,
        "mean_completion_tokens": statistics.fmean(record["completion_tokens"] for record in successful) if successful else None,
    }

    document = {
        "schema_version": 3,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "repetition": args.repetition,
        "configuration": {
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "target_prompt_tokens": args.target_prompt_tokens,
            "exact_prompt_tokens": args.exact_prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
            "closed_loop": True,
            "prompt_cache_requested": args.prefix_reuse,
            "repeat_prompt": args.repeat_prompt,
            "prefix_reuse": args.prefix_reuse,
            "cache_policy": "require" if args.prefix_reuse else "forbid",
            "input_sha256": prepared[0]["input_sha256"] if args.repeat_prompt else None,
            "input_sha256_definition": "SHA256 of messages JSON, sorted keys, compact separators, UTF-8 without ASCII escaping",
            "ignore_eos": True,
            "throughput_window": "first HTTP request start through last stream completion (includes drain)",
            "ttft_definition": "HTTP request start to first nonempty streamed content event",
            "tpot_definition": "(HTTP stream duration - TTFT) / (server completion tokens - 1)",
            "concurrency_definition": "closed-loop HTTP requests in flight; GPU batching requires server/profile evidence",
            "prompt_token_definition": "prompt_tokens and total_tokens include cached logical inputs; evaluated_prompt_tokens is server timing.prompt_n",
        },
        "summary": summary,
        "requests": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if failed:
        print(f"{len(failed)} request(s) failed; details are in {args.output}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
