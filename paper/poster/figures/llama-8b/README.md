Llama 3.1 8B F1 and F3

Rebuild with `python paper/poster/build_8b.py`. Outputs are separate from 70B.

F1: TTFT p95 at concurrency = 8; percentages relative to Q4_K_M.

F3: throughput versus concurrency (8, 16, 32), with separate quantization panels and context-length line styles (2K, 4K, 8K, 16K). Point labels show raw tok/s and percentage changes relative to concurrency = 8. Panels have individually labeled y-axis scales.

No missing values. All 36 settings were checked against raw summaries: 1344 successful requests. Exact values and source hashes are in data.json.
