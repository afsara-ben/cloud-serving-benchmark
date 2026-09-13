# Two-GPU inference study: approved execution plan

Originally recorded as future work; execution authorized on 12 September 2026.
This file preserves the agreed scope before implementation and experiments.
Amended on 13 September 2026: one measured run per case, and all five formats
for every model size when the fixed two-GPU memory requirements can be met.

## Shared configuration

| Model | Weight formats |
|---|---|
| Llama 3.2 1B Instruct | FP16, Q8_0, Q4_K_M, Q2_K, IQ1_M |
| Llama 3.1 8B Instruct | FP16, Q8_0, Q4_K_M, Q2_K, IQ1_M |
| Llama 3.3 70B Instruct | FP16, Q8_0, Q4_K_M, Q2_K, IQ1_M |

- One model instance split across the two RTX A6000 GPUs, fixed layer split 1,1.
- FP16 KV for all weight formats and exactly 512 generated tokens per request.
- Input lengths count tokenized chat templates and must be powers of two.
- Fully offload transformer layers and KV; reject capacity fallback, server swap,
  truncation, context shifting, or silently reduced active-slot counts.
- Reserve input + 512 + 1 positions per slot, applying the backend's 256-token
  allocation padding; slots equal requested concurrency. The extra guard position
  avoids the pinned server's context-limit stop condition at exactly input + output,
  established in the initial 16k validation; actual input/output counts stay fixed.
  Allow 2 GiB headroom
  per GPU after weights, KV, and working buffers.
- Hold the pinned llama.cpp backend, FlashAttention, logical/physical token
  batches 2048/512, GPU placement, and automatic clock/power policy fixed.
- True FP16 weights come from original full-precision checkpoints. Pin source
  revisions, model hashes, and exact tensor inventories for new files.
- Match tokenizers and rendered inputs across each model's formats. Use the same
  explicit 8B chat template: the original FP16 template otherwise adds dates
  absent from the existing quantized models' template.
- Do not download or attempt 70B FP16 on this device: its approximately
  131.4 GiB of weights alone exceeds both GPUs. Keep it explicitly selectable
  on other hardware, screened before download and validated before execution.
- Extend a shared runner, model manifests, and common profiling/analysis tools;
  avoid duplicate per-model scripts and preserve previous commands/results.

## Numbered experiments

1. **Serving matrix and capacity screening.** For every model/format, test input
   2048/4096/8192/16384 at 8/16/32/64 clients. One discarded concurrent warmup,
   then one measured run of two requests per client, with prefix reuse off.
   Record throughput, TTFT, TPOT, end-to-end latency, failures, peak per-GPU VRAM,
   actual active slots/decode widths, clocks, temperature, and thermal limiting.
   Distinguish predicted capacity limits from observed allocation/request failures.

2. **Largest supported power-of-two input.** Reuse experiment 1, extending only
   where feasible to 32768 and 65536 tokens. Validate each new input length with
   one complete concurrent burst and report the largest passing power of two
   for the fixed layout. Do not test intermediate lengths such as 4.6k.
   131072 input plus 512 output exceeds the native 131072 context; record it as
   unsupported without extrapolation.

3. **Targeted profiling.** For each model, format, and concurrency, profile 2048,
   16384, and the longest jointly feasible input across runnable formats.
   Deduplicate matching endpoints; explicitly skip unsupported cells. Separate
   prefill and decode using actual phase/matrix metadata. Collect traces and
   counters for dominant matched operations on both GPUs: instruction counts
   and mix, IPC, DRAM traffic/bandwidth, L1/L2 behavior, occupancy, registers/shared
   memory, spills, scheduler stalls, tensor activity, transfers, and idle time.
   Use one independent capture with five matched launches where available;
   supplement only missing operation samples. Kernel samples within a capture
   are distinct from repeating the whole experiment.
   Collect ordinary serving timing, timeline traces, and hardware counters in
   separate runs; replay/instrumentation timing is not serving throughput.
   Each diagnostic capture uses one C-request burst after discarded warmup;
   interpret its phase fractions and batch widths separately from the 2C-request
   main serving workload, comparing matched operations across formats.

4. **Q2/Q4 causal check.** Test dominant matched 70B matrix shapes at activation
   widths 1/8/9/16/32/64, plus one isolated earlier-MMQ-dispatch intervention at
   width 8 for Q2/Q3. Preserve the pinned binary for headline serving results.
   Use IQ1_M's measured path as an additional comparison. Attribute mechanisms
   only when counters and controlled tests support them; retain unresolved
   explanations explicitly rather than infer causes from model size alone.
   Run each diagnostic case once; retain the short internal kernel timing loop.

## Validation and deliverables

Validate exact tokens, full GPU placement, actual concurrency, capacity arithmetic,
resume identities, matching model inputs, per-device counter coverage, and CSV
parsing. Preserve unavailable/invalid counters and their diagnostics.
Keep validation reporting to brief pass/fail updates.
Reuse existing validated first runs. Retain earlier extra repetitions as prior
protocol evidence, exclude them from the amended comparisons, and do not report
run-to-run standard deviations from a single run. Prioritize findings that explain
unexpected rankings or changes with context/concurrency using measured mechanisms.

Store the new study separately with organized runtime measurements, capacity
results, timeline traces, hardware counters, diagnostic builds, and raw evidence.
Write one compact findings report and update FINDINGS.md with capacity tables,
largest supported input lengths, serving results, and evidence-backed low-level
interpretation. Include reproducible commands and a requirement-by-requirement
completion audit. Do not treat skipped 70B FP16 capacity cells as measured results.

## Terminology

Activation width is the number of token vectors in a matrix operation. During
decode, width 8 normally means eight active sequences producing their next token;
during prefill it denotes prompt tokens processed together. Width 9 probes the
pinned Q2/Q4 dispatch transition above eight. It is not a context length.

Active KV normally resides in GPU memory in latency-sensitive serving; optional
CPU and storage tiers support offloading/reuse. There is no universal production
input/output length. This study uses a controlled workload, not a production
arrival distribution. References: [TensorRT-LLM KV cache](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html),
[vLLM offloading](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/),
[serving benchmark controls](https://docs.vllm.ai/en/latest/cli/bench/serve/).
