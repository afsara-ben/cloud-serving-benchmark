# Concurrency-1 context sweep (RTX PRO 6000, Llama-3.3-70B)

Single-request latency for the four quantized formats across 2K-16K input, measured
2026-09-15 on two RTX PRO 6000 Blackwell 96GB GPUs (driver 580.95.05).

This is an **exploratory side experiment, not part of the `cuda-context-study-rtxpro6000-*-r1`
series.** It is kept separate because concurrency 1 lies outside the paper scope, which
admits only C8/16/32 (`benchmark/report_serving.py` `CONCURRENCIES`, and the allowed set in
`scripts/08_run_cuda_study.py`). Nothing here was produced by `run_rtxpro6000.py`, and these
cells must not be merged into that series' reports or completion audits.

## What was measured

4 formats (IQ1_M, Q2_K, Q4_K_M, Q8_0) x 4 inputs (2048, 4096, 8192, 16384) = 16 cells.
Per cell: a fresh `llama-server`, 1 discarded warmup request, then 2 measured requests at
concurrency 1, exactly 512 output tokens each.

Server settings match the study protocol: FP16 KV, `--split-mode layer --tensor-split 1,1`
across GPUs 0,1, `--no-context-shift`, `--no-cache-prompt`, prompt reuse off, the shared
`config/templates/llama-3.1-benchmark.jinja` template, and `RANDOM_SEED=20260912`. The only
deliberate departures are `--parallel 1` and `n_ctx = slot_capacity(prompt)` for a single slot
(2816 / 4864 / 8960 / 17152).

All 16 cells passed the same client-side checks the harness applies: exact 2048/4096/8192/16384
input tokens, exact 512 output tokens, `max_client_inflight == 1`, zero failed requests and zero
cached prompt tokens.

## Caveats

- **No GPU telemetry.** The sampler that fills the `gpu*_peak_vram_gib`, power, clock and
  thermal-limit columns of the series' `runtime.csv` was not run, so `c1_sweep.csv` has only the
  client-side columns. It is deliberately *not* schema-compatible with that file.
- **Two measured requests per cell, one repetition.** Enough to separate the formats, which are
  far apart; not enough to characterize run-to-run variation. The p95 columns describe two
  requests and should not be read as tail latency.
- **`output_tokens_per_second` is an aggregate**, generated tokens over the whole HTTP window,
  so it includes prefill and falls with context even where TPOT is flat. The steady-state decode
  rate is `1000 / tpot_p50_ms`.

## Findings

At batch 1 decode is weight-bandwidth-bound: TPOT is nearly flat across context (+13% from 2K to
16K for IQ1_M, +5% for Q8_0) and tracks weight size, with IQ1_M about 3.1x faster per token than
Q8_0. Prefill inverts that ordering -- Q8_0 reaches first token fastest (5.65s vs IQ1_M's 9.00s
at 16K) because the low-bit formats pay dequantization cost in the compute-bound phase. The
inversion is clearer here than at C8/16/32.

See `c1_sweep.md` for the tables and `c1_sweep.csv` for all 16 rows.

## Files

- `c1_sweep.csv`, `c1_sweep.md` -- summary over the 16 cells.
- `cells/<FORMAT>/p<INPUT>/raw.json` -- per-request client timings, as written by
  `benchmark/load_test.py`.
- `cells/<FORMAT>/p<INPUT>/study.env` -- the exact server configuration for that cell.
- `scripts/run_c1_sweep.sh`, `scripts/summarize.py` -- the driver and summarizer.

Verbose `server.log` (log-verbosity 4), `client.log` and the generated `prompts.json` were not
retained.

## Reproducing

`run_c1_sweep.sh` drives `scripts/03_start_server.sh` and `benchmark/load_test.py` directly,
bypassing the study runner's concurrency restriction. It is resumable: a cell whose
`r1/raw.json` already exists is skipped. `ROOT`, `OUT` and `CUDA_HOME` can be overridden.

```bash
OUT=/tmp/c1_sweep/results bash results/rtxpro6000-c1-sweep/scripts/run_c1_sweep.sh
OUT=/tmp/c1_sweep/results python3 results/rtxpro6000-c1-sweep/scripts/summarize.py
```

Note that the pinned `llama-server` is built against CUDA 12.8.1
(`vendor/llama.cpp/build-cuda/CMakeCache.txt`) while the system default is CUDA 13. Without that
toolkit's `lib64` on `LD_LIBRARY_PATH` the server dies at startup with
`libcudart.so.12: cannot open shared object file`; the driver script exports it.
