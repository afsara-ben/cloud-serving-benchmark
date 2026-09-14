# Llama 70B inference coverage

Aggregate output tokens/s. One validated run per setting, 512 output tokens per request, FP16 KV and no prompt reuse.

| Format | Concurrency | 2K | 4K | 8K | 16K |
|---|---:|---:|---:|---:|---:|
| IQ1_M | 8 | 49.42 | 36.64 | 23.94 | 13.67 |
| IQ1_M | 16 | 30.09 | 24.89 | 18.24 | Not measured |
| IQ1_M | 32 | Not measured | Not measured | Not measured | Not measured |
| Q2_K | 8 | 38.48 | 30.64 | 21.23 | 12.87 |
| Q2_K | 16 | 77.56 | 49.95 | 28.65 | VRAM estimate |
| Q2_K | 32 | 87.50 | 53.90 | VRAM estimate | VRAM estimate |
| Q4_K_M | 8 | 45.92 | 36.72 | 25.66 | 15.54 |
| Q4_K_M | 16 | 83.54 | 56.14 | 33.41 | VRAM estimate |
| Q4_K_M | 32 | 104.60 | 65.01 | VRAM estimate | VRAM estimate |
| Q8_0 | 8 | Not measured | Not measured | Not measured | Not measured |
| Q8_0 | 16 | Not measured | Not measured | Not measured | Not measured |
| Q8_0 | 32 | Not measured | Not measured | Not measured | Not measured |

Per-setting statuses and evidence paths: [coverage.csv](coverage.csv).
Not measured means no validated result exists. Some of these settings are expected to exceed VRAM, but their per-cell capacity screen had not run when the queue was paused.
VRAM estimate means a saved per-cell capacity exclusion exists; it is not an observed OOM or a throughput measurement.
Baseline and continuation are separate sessions on the same two RTX A6000 GPUs. Excluded settings have no throughput value.
