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
    for path in sorted(args.raw_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["_path"] = str(path)
        documents.append(document)
    if not documents:
        parser.error(f"no JSON result files found in {args.raw_dir}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        key = (document["model"], int(document["configuration"]["concurrency"]))
        grouped[key].append(document)

    rows: list[dict[str, Any]] = []
    for (model, concurrency), runs in sorted(grouped.items()):
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

        rows.append(
            {
                "model": model,
                "concurrency": concurrency,
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
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['model']} at concurrency {row['concurrency']}",
                "",
                f"- Repetitions: {row['repetitions']}",
                f"- Successful requests: {row['successful_requests']} (failed: {row['failed_requests']})",
                f"- Mean request rate: {fmt(row['requests_per_second_mean'])} req/s (run-to-run SD {fmt(row['requests_per_second_stddev'])})",
                f"- Mean output throughput: {fmt(row['output_tokens_per_second_mean'])} tokens/s (run-to-run SD {fmt(row['output_tokens_per_second_stddev'])})",
                f"- Mean total-token throughput: {fmt(row['total_tokens_per_second_mean'])} tokens/s",
                f"- TTFT: p50 {fmt(row['ttft_p50_ms'])} ms; p95 {fmt(row['ttft_p95_ms'])} ms",
                f"- TPOT: p50 {fmt(row['tpot_p50_ms'])} ms; p95 {fmt(row['tpot_p95_ms'])} ms",
                f"- End-to-end latency: p50 {fmt(row['e2e_p50_ms'])} ms; p95 {fmt(row['e2e_p95_ms'])} ms",
                f"- Mean measured prompt/completion length: {fmt(row['mean_prompt_tokens'], 1)} / {fmt(row['mean_completion_tokens'], 1)} tokens",
                "",
            ]
        )

    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
