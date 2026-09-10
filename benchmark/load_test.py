#!/usr/bin/env python3
"""Closed-loop concurrent load generator for llama-server.

The client uses the OpenAI-compatible streaming chat endpoint. Prompt lengths
are prepared with llama-server's own chat template and tokenizer before the
timed region. No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pathlib
import statistics
import sys
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

    def count_text(self, text: str) -> int:
        response = post_json(
            f"{self.base_url}/tokenize",
            {"content": text, "add_special": False, "parse_special": True},
            self.timeout,
        )
        return len(response["tokens"])

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return self.count_text(self.apply_template(messages))


def make_messages(
    tokenizer: ServerTokenizer,
    topic: str,
    request_index: int,
    target_tokens: int,
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
        return base_messages, base_count

    low = 0
    high = len(filler)
    best_messages = base_messages
    best_count = base_count
    while low <= high:
        middle = (low + high) // 2
        messages, token_count = candidate(middle)
        if token_count <= target_tokens:
            best_messages = messages
            best_count = token_count
            low = middle + 1
        else:
            high = middle - 1

    return best_messages, best_count


def stream_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    timeout: float,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "llama-cloud-serving-bench/1.0"},
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []
    finish_reason: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(str(content))
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "error": f"HTTP {exc.code}: {body[:500]}",
            "duration_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001 - errors must be recorded per request
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": time.perf_counter() - started,
        }

    finished = time.perf_counter()
    text = "".join(chunks)
    if first_token_at is None and text:
        first_token_at = finished
    return {
        "ok": first_token_at is not None,
        "error": None if first_token_at is not None else "No streamed content token was received",
        "duration_seconds": finished - started,
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "finish_reason": finish_reason,
        "_text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--topics", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--target-prompt-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must both be positive")
    if args.requests < args.concurrency:
        parser.error("requests must be at least as large as concurrency")

    topics = [line.strip() for line in args.topics.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not topics:
        parser.error("topics file is empty")

    tokenizer = ServerTokenizer(args.base_url, args.timeout)
    prepared: list[dict[str, Any]] = []
    print(f"Preparing {args.requests} prompts at approximately {args.target_prompt_tokens} tokens...", flush=True)
    for index in range(args.requests):
        messages, prompt_tokens = make_messages(
            tokenizer,
            topics[index % len(topics)],
            index,
            args.target_prompt_tokens,
        )
        prepared.append({"messages": messages, "prompt_tokens": prompt_tokens})

    for warmup_index in range(args.warmup_requests):
        warmup = stream_completion(
            args.base_url,
            args.model,
            prepared[warmup_index % len(prepared)]["messages"],
            min(16, args.max_output_tokens),
            args.timeout,
            args.seed + 100_000 + warmup_index,
        )
        if not warmup["ok"]:
            print(f"Warmup failed: {warmup['error']}", file=sys.stderr)
            return 2

    records: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    def run_one(index: int) -> tuple[int, dict[str, Any]]:
        record = stream_completion(
            args.base_url,
            args.model,
            prepared[index]["messages"],
            args.max_output_tokens,
            args.timeout,
            args.seed + index,
        )
        return index, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, index) for index in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            index, record = future.result()
            record["request_index"] = index
            record["prompt_tokens"] = prepared[index]["prompt_tokens"]
            records.append(record)

    wall_seconds = time.perf_counter() - wall_started
    records.sort(key=lambda item: item["request_index"])

    for record in records:
        generated_text = record.pop("_text", "")
        completion_tokens = tokenizer.count_text(generated_text) if generated_text else 0
        record["completion_tokens"] = completion_tokens
        record["total_tokens"] = record["prompt_tokens"] + completion_tokens
        record["output_characters"] = len(generated_text)
        if record.get("ttft_seconds") is not None and completion_tokens > 1:
            record["tpot_seconds"] = max(
                0.0,
                (record["duration_seconds"] - record["ttft_seconds"]) / (completion_tokens - 1),
            )
        else:
            record["tpot_seconds"] = None

    successful = [record for record in records if record["ok"]]
    failed = [record for record in records if not record["ok"]]
    ttft = [record["ttft_seconds"] for record in successful if record.get("ttft_seconds") is not None]
    tpot = [record["tpot_seconds"] for record in successful if record.get("tpot_seconds") is not None]
    e2e = [record["duration_seconds"] for record in successful]
    prompt_tokens_total = sum(record["prompt_tokens"] for record in successful)
    completion_tokens_total = sum(record["completion_tokens"] for record in successful)

    summary = {
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_seconds": wall_seconds,
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
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "repetition": args.repetition,
        "configuration": {
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "target_prompt_tokens": args.target_prompt_tokens,
            "max_output_tokens": args.max_output_tokens,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
            "closed_loop": True,
            "prompt_cache_requested": False,
            "ignore_eos": True,
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
