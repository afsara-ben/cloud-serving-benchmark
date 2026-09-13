#!/usr/bin/env python3
"""Combine repeated load-test JSON files into CSV and readable Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any


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


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sample_stddev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=pathlib.Path)
    parser.add_argument("--output-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output-markdown", type=pathlib.Path, required=True)
    args = parser.parse_args()

    documents = []
    paths = set(args.raw_dir.glob("*.json")) | set(args.raw_dir.rglob("raw.json"))
    for path in sorted(paths):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not all(key in document for key in ("configuration", "requests", "summary")):
            continue
        document["_path"] = str(path)
        documents.append(document)
    if not documents:
        parser.error(f"no JSON result files found in {args.raw_dir}")

    grouped: dict[tuple[str, int, int, int, int, bool, bool], list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        config = document["configuration"]
        key = (
            document["model"], int(config["concurrency"]),
            int(config["target_prompt_tokens"]), int(config["max_output_tokens"]),
            int(document.get("schema_version", 1)),
            bool(config.get("repeat_prompt", False)),
            bool(config.get("prefix_reuse", config.get("prompt_cache_requested", False))),
        )
        grouped[key].append(document)

    rows: list[dict[str, Any]] = []
    for (model, concurrency, prompt_length, output_length, schema_version, repeat_prompt, prefix_reuse), runs in sorted(grouped.items()):
        successful_records = [
            record
            for run in runs
            for record in run["requests"]
            if record.get("ok")
        ]
        ttft_ms = numeric([record.get("ttft_seconds") for record in successful_records])
        ttft_ms = [value * 1000.0 for value in ttft_ms]
        tpot_ms = numeric([record.get("tpot_seconds") for record in successful_records])
        tpot_ms = [value * 1000.0 for value in tpot_ms]
        e2e_ms = [float(record["duration_seconds"]) * 1000.0 for record in successful_records]

        request_rates = numeric([run["summary"].get("requests_per_second") for run in runs])
        output_rates = numeric([run["summary"].get("output_tokens_per_second") for run in runs])
        total_rates = numeric([run["summary"].get("total_tokens_per_second") for run in runs])
        overlaps = numeric([run["summary"].get("max_client_inflight") for run in runs])

        rows.append(
            {
                "model": model,
                "concurrency": concurrency,
                "target_prompt_tokens": prompt_length,
                "max_output_tokens": output_length,
                "schema_version": schema_version,
                "repeat_prompt": repeat_prompt,
                "prefix_reuse": prefix_reuse,
                "repetitions": len(runs),
                "successful_requests": len(successful_records),
                "failed_requests": sum(int(run["summary"]["failed_requests"]) for run in runs),
                "requests_per_second_mean": mean(request_rates),
                "requests_per_second_stddev": sample_stddev(request_rates),
                "output_tokens_per_second_mean": mean(output_rates),
                "output_tokens_per_second_stddev": sample_stddev(output_rates),
                "total_tokens_per_second_mean": mean(total_rates),
                "ttft_p50_ms": percentile(ttft_ms, 0.50),
                "ttft_p95_ms": percentile(ttft_ms, 0.95),
                "tpot_p50_ms": percentile(tpot_ms, 0.50),
                "tpot_p95_ms": percentile(tpot_ms, 0.95),
                "e2e_p50_ms": percentile(e2e_ms, 0.50),
                "e2e_p95_ms": percentile(e2e_ms, 0.95),
                "mean_prompt_tokens": mean([float(record["prompt_tokens"]) for record in successful_records]),
                "mean_completion_tokens": mean([float(record["completion_tokens"]) for record in successful_records]),
                "mean_evaluated_prompt_tokens": mean(numeric([record.get("evaluated_prompt_tokens", record.get("server_prompt_tokens_evaluated")) for record in successful_records])),
                "evaluated_prompt_tokens_per_second_mean": mean(numeric([run["summary"].get("evaluated_prompt_tokens_per_second") for run in runs])),
                "max_client_inflight_min_across_runs": min(overlaps) if overlaps else None,
                "mean_client_inflight": mean(numeric([run["summary"].get("mean_client_inflight") for run in runs])),
                "output_length_invalid_requests": sum(int(run["summary"].get("output_length_invalid_requests", 0)) for run in runs),
                "cache_counter_available_requests": sum(int(run["summary"].get("cache_counter_available_requests", 0)) for run in runs),
                "cached_prompt_tokens": sum(int(run["summary"].get("cached_prompt_tokens", 0)) for run in runs),
                "cache_hit_requests": sum(int(run["summary"].get("cache_hit_requests", 0)) for run in runs),
                "server_prompt_p50_ms": percentile(numeric([record.get("server_timings", {}).get("prompt_ms") for record in successful_records]), 0.5),
                "server_decode_p50_ms": percentile(numeric([record.get("server_timings", {}).get("predicted_ms") for record in successful_records]), 0.5),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cloud serving experiment summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "Rates are means across repetitions; latency percentiles pool successful requests. "
        "The timed interval runs from first HTTP request start through last stream completion and includes batch drain. "
        "Client overlap demonstrates concurrent requests; server traces establish GPU batching. "
        "Server prompt/decode timings are elapsed per-request intervals and overlap across concurrent requests. "
        "Logical input counts include cached tokens; evaluated prompt counts measure the work remaining after reuse.",
        "",
        "| Model | Input / reuse | Prompt / output | Clients | Runs | Valid / failed | Output tok/s (SD) | TTFT p50 / p95 ms | TPOT p50 / p95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {'repeated' if row['repeat_prompt'] else 'varied'} / {'on' if row['prefix_reuse'] else 'off'} "
            f"| {row['target_prompt_tokens']} / {row['max_output_tokens']} "
            f"| {row['concurrency']} | {row['repetitions']} "
            f"| {row['successful_requests']} / {row['failed_requests']} "
            f"| {fmt(row['output_tokens_per_second_mean'])} ({fmt(row['output_tokens_per_second_stddev'])}) "
            f"| {fmt(row['ttft_p50_ms'])} / {fmt(row['ttft_p95_ms'])} "
            f"| {fmt(row['tpot_p50_ms'])} / {fmt(row['tpot_p95_ms'])} |"
        )
    lines.extend(["", "The CSV retains end-to-end latency, server phase timings, cache counters, and observed client overlap.", ""])
    if any(row["schema_version"] < 2 for row in rows):
        lines.extend(["Warning: schema v1 results use retokenized text counts and lack cache/overlap validation. They are kept separate from v2 results.", ""])

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
