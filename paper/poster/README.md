# Runtime Cost poster figures — Llama 70B

The two main exports use **Llama-3.3-70B-Instruct only**, on two RTX A6000 GPUs:

- **Merged, two panels:** [PDF](runtime-cost-merged.pdf) · [PNG](runtime-cost-merged.png) · [editable SVG](runtime-cost-merged.svg) · [values and provenance](merged-data.json).
- **Poster, four panels:** [PDF](runtime-cost-poster.pdf) · [PNG](runtime-cost-poster.png) · [editable SVG](runtime-cost-poster.svg) · [values and provenance](data.json).
- **Merged throughput panel alone:** [PDF](figures/f2-f4-throughput-context.pdf) · [PNG](figures/f2-f4-throughput-context.png) · [SVG](figures/f2-f4-throughput-context.svg).
- **Companion capacity matrix:** [PDF](figures/context-capacity.pdf) · [PNG](figures/context-capacity.png) · [SVG](figures/context-capacity.svg).
- **Requested inference coverage:** [throughput table](coverage.md) · [statuses and evidence paths](coverage.csv).
- **Measurement audit:** [validation results](measurement-audit.json) · [reproducible checker](audit.py).

The requested grid is IQ1_M / Q2_K / Q4_K_M / Q8_0 × 8 / 16 / 32 concurrent clients × 2K / 4K / 8K / 16K input tokens. K = 1,024. Missing and capacity-excluded settings have no throughput point; they are never plotted as zero or interpolated. By default, the builder requires a validated run or documented capacity exclusion for every setting. `--allow-missing` creates an explicitly incomplete snapshot from the available measurements.

The current exports are a snapshot taken after pausing the queue on September 14: **25 validated measurements, six recorded capacity exclusions, and 17 settings without a saved result**. IQ1_M has no C32 measurements, and Q8_0 has no measurements. Of the 17 settings without a result, five feasible measurements remain queued; the other 12 are expected to exceed VRAM but had not reached their per-cell capacity screen. The coverage table preserves this distinction. No inference was launched to generate these exports.

## Merged layout

The first panel compares IQ1_M, Q2_K and Q4_K_M time to first token (TTFT p95) at eight clients using grouped bars. All four input lengths are shown, with latency in seconds above each bar. Above each group, orange and purple labels give Q2_K and IQ1_M percentage increases relative to Q4_K_M: `100 × (TTFT_format / TTFT_Q4_K_M − 1)`. Q4_K_M is the reference.

The second panel contains only 70B throughput measurements. Colors identify the four requested quantization formats; circle/solid, triangle/dashed and square/dotted series identify 8, 16 and 32 concurrent clients. Prompt ticks have log2 spacing; throughput uses a linear axis. Lines stop where a measurement is missing or capacity-excluded. The notes report the available IQ1_M concurrency sequence, Q4_K_M concurrency gains, IQ1_M throughput relative to Q4_K_M, and missing Q8_0 data. Q8_0 appears in the format key but has no plotted points.

## Four-panel layout

| Panel | View | Setting |
|---|---|---|
| F1. Lower bits ≠ lower latency | Grouped TTFT bars with raw values and percentage increases relative to Q4_K_M | IQ1_M, Q2_K and Q4_K_M, eight clients, all four input lengths |
| F2. More clients can lower throughput | Throughput versus concurrent clients; color identifies quantization and line/marker identifies input length | 2K/4K/8K/16K input, C8/C16/C32 where measured; IQ1_M, Q2_K and Q4_K_M available; Q8_0 missing |
| F3. Throughput at 8 and 16 clients | Raw throughput bars with numeric labels and percentage changes above each pair; hatched bars identify C8 and solid bars C16 | IQ1_M, Q2_K and Q4_K_M at 2K/4K/8K/16K; 16K has C8 only |
| F4. Context changes quantization gains | Horizontal dumbbells with signed percentage changes | IQ1_M versus Q4_K_M, eight clients, all four input lengths |

F3 shows measured aggregate output throughput in tokens/s, with one decimal place on each bar. Each measured format/context pair has a hatched C8 bar and a solid C16 bar. At 16K, the three measured C8 bars are shown, while missing C16 slots are marked N/A and have no percentage label. The bracket above each pair reports `100 × (throughput_C16 / throughput_C8 − 1)`, including negative changes for IQ1_M. At 16K, Q2_K and Q4_K_M C16 are capacity-excluded, and IQ1_M C16 is not measured; these unavailable pairs are explicitly labeled. The merged layout's separate F3 text note still reports the Q4_K_M percentage gains. F4 uses `100 × (throughput_IQ1_M / throughput_Q4_K_M − 1)` at eight clients. Its signed labels report the observed difference; near-zero differences do not establish a statistically significant winner. The reference in F4 is Q4_K_M; FP16 70B weights exceed this setup's capacity.

The companion capacity matrix shows the largest **validated** input within the requested 2K–16K grid, by format and concurrency. It is bounded by the tested grid and does not claim a model's native context limit. All main and companion plots use 70B data only.

## Measurements and provenance

Every measured setting uses the same pinned, unmodified llama.cpp server binary (`3f5e94d7c2ab2267fe39852051777fe30c1f49ef`), two RTX A6000 GPUs, layer splitting, FP16 KV, flash attention, 512 generated tokens per request and no prompt reuse. A discarded C-request warmup precedes one measured run containing 2C requests. Report validation checks exact input/output counts, concurrent requests, full transformer/output GPU placement, zero server swap and at least 2 GiB headroom on each GPU. No profiler is active during the serving runs.

The inputs combine two separately recorded sessions:

1. [Baseline report](../../results/cuda-context-study/report/runtime.csv) and [70B artifacts](../../results/cuda-context-study-70b/summary.md), collected September 13.
2. [September 14 continuation](../../results/context-study/poster-70b-20260914/report/runtime.csv), with [launch command and selected cells](../../results/context-study/poster-70b-20260914/launch.json), [runner log](../../results/context-study/poster-70b-20260914/runner.log) and raw artifacts under its `70b/` directory.

The continuation uses a separate directory because the runner source changed after the baseline. It does not rewrite the baseline's identity or measurements. Both sessions retain their own model, binary, source and configuration hashes. Each plotted point records its source CSV and raw artifact directory in the corresponding data JSON.

These are single-run observations under automatic clocks. Thermal limiting occurred in some settings, and separate measurement sessions add a timing-comparison limitation. No run-to-run uncertainty or causal hardware claim is inferred from the plots.

## Rebuild

From the repository root, to reproduce the current snapshot:

```sh
python3 benchmark/report_context_study.py --study-root results/context-study/poster-70b-20260914 --no-plots
python3 paper/poster/build.py --layout merged --allow-missing
python3 paper/poster/build.py --layout four --allow-missing
python3 paper/poster/audit.py --allow-missing
```

On this workspace use `/home/hys4qm/anaconda3/bin/python3`. The report command independently validates saved runs. The poster builder reads the validated baseline and continuation CSVs, verifies requested coverage and writes only this poster directory.

After the remaining experiments and capacity screens finish, omit `--allow-missing` to enforce complete resolution of the grid.

The audit checks retained request counts, exact prompt/output lengths, attained concurrency, zero prompt-cache reuse, matching inputs across formats, and agreement between both posters' values and the raw measurements. The report's broader full-study status also includes other model sizes, formats, concurrencies and profiling tasks; the poster coverage check is limited to the 48 settings requested here.

The unrelated [two-line 8B version](runtime-cost-two-line.pdf) and [explicitly illustrative 64K extrapolation](illustrative/f1-with-64k-extrapolation.pdf) remain separate historical exports. Neither contributes 8B points or synthetic values to the two main 70B figures.
