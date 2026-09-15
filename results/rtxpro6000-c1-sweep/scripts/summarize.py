#!/usr/bin/env python3
"""Summarize the concurrency-1 sweep into a CSV and a Markdown table."""
import csv
import json
import os
from pathlib import Path

OUT = Path(os.environ.get("OUT", "/tmp/c1_sweep/results"))
FORMATS = ["IQ1_M", "Q2_K", "Q4_K_M", "Q8_0"]
PROMPTS = [2048, 4096, 8192, 16384]

FIELDS = ["quant", "concurrency", "prompt_tokens", "output_tokens", "status",
          "wall_seconds", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms",
          "e2e_p50_ms", "output_tokens_per_second", "evaluated_prompt_tokens_per_second",
          "total_tokens_per_second", "requests"]

rows = []
for quant in FORMATS:
    for prompt in PROMPTS:
        cell = OUT / quant / f"p{prompt}"
        status_file = cell / "status"
        status = status_file.read_text().strip() if status_file.exists() else "not_run"
        row = {"quant": quant, "concurrency": 1, "prompt_tokens": prompt,
               "output_tokens": 512, "status": status}
        raw = cell / "r1" / "raw.json"
        if raw.exists():
            s = json.load(raw.open())["summary"]
            row.update(wall_seconds=round(s["wall_seconds"], 3),
                       ttft_p50_ms=round(s["ttft_p50_ms"], 1),
                       ttft_p95_ms=round(s["ttft_p95_ms"], 1),
                       tpot_p50_ms=round(s["tpot_p50_ms"], 3),
                       tpot_p95_ms=round(s["tpot_p95_ms"], 3),
                       e2e_p50_ms=round(s["e2e_p50_ms"], 1),
                       output_tokens_per_second=round(s["output_tokens_per_second"], 3),
                       evaluated_prompt_tokens_per_second=round(s["evaluated_prompt_tokens_per_second"], 2),
                       total_tokens_per_second=round(s["total_tokens_per_second"], 3),
                       requests=s["successful_requests"])
            # Guard: these cells are only meaningful at inflight 1 with exact tokens.
            if (s["max_client_inflight"] != 1 or s["failed_requests"]
                    or s["mean_prompt_tokens"] != prompt or s["mean_completion_tokens"] != 512):
                row["status"] = "INVALID"
        rows.append(row)

csv_path = OUT.parent / "c1_sweep.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

def table(metric, label, precision=1):
    lines = [f"**{label}**", "",
             "| Format | " + " | ".join(f"{p//1024}K" for p in PROMPTS) + " |",
             "|---|" + "---|" * len(PROMPTS)]
    for quant in FORMATS:
        cells = []
        for prompt in PROMPTS:
            row = next(r for r in rows if r["quant"] == quant and r["prompt_tokens"] == prompt)
            value = row.get(metric)
            cells.append("—" if value is None else f"{value:,.{precision}f}")
        lines.append(f"| {quant} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"

done = sum(1 for r in rows if r["status"] == "ok")
report = [f"# Concurrency-1 context sweep (Llama-3.3-70B, 2x RTX PRO 6000)", "",
          f"{done}/{len(rows)} cells measured. 1 warmup + 2 measured requests per cell, "
          f"512 output tokens, FP16 KV, layer split across GPUs 0,1.", "",
          table("tpot_p50_ms", "TPOT p50 (ms/token) — lower is better", 2), "",
          table("output_tokens_per_second", "Decode throughput (tok/s)", 2), "",
          table("ttft_p50_ms", "TTFT p50 (ms)", 0), "",
          table("evaluated_prompt_tokens_per_second", "Prefill rate (tok/s)", 1), ""]
report_path = OUT.parent / "c1_sweep.md"
report_path.write_text("\n".join(report))
print("\n".join(report))
print(f"\nCSV:  {csv_path}\nMD:   {report_path}")
