# A100 scalar FP32 operator rooflines

`scripts/collect_a100_roofline.py` collects the values used by the expanded
`roofline-explained.pdf`, then exports CSVs and an A100 figure. It accepts
completed 70B IQ1_M, Q2_K, Q4_K_M and Q8_0 serving cells at one common input
length and concurrency. Select a different set of cells and output directory
for each workload. No missing serving measurements are run automatically.

Required per-kernel evidence:

| Value | Nsight Compute source |
|---|---|
| Predicated-on FADD, FMUL and FFMA thread counts | `sass__thread_inst_executed_true_per_opcode` instances |
| Kernel duration | `gpu__time_duration.sum` |
| DRAM bytes read and written | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Operator, phase, shape and device | Same-capture NVTX labels and launch metadata |

`F = FADD + FMUL + 2*FFMA`, `B = read_bytes + write_bytes`.
The plotted coordinates are `x = F/B` and `y = F/duration_seconds`.
Compute each coordinate per launch, then take medians within the same
format/phase/GPU/role/type/shape/kernel/launch configuration and capture.
Zero scalar work remains zero; missing or invalid counters are unavailable.

For **A100 SXM 80GB**, the rated ceilings are **19.5 TFLOP/s scalar FP32** and
**2,039 GB/s DRAM bandwidth**, per GPU. The roof is
`min(19.5e12, x * 2039e9)` FLOP/s. Do not double these ceilings for two GPUs.
[NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf).
Tensor Core, FP16, integer and special-function work are outside this scalar
view; these coordinates alone cannot establish a mixed-arithmetic kernel's
bottleneck. FlashAttention includes fused softmax.

## Commands on the A100 host

First obtain completed serving measurements using [the serving workflow](A100_SERVING.md).
Install the full Nsight Compute package, including its Python Report Interface,
and enable counter access for your account. Build the diagnostic server once:

```bash
python3 benchmark/matrix_diagnostic.py build --server --cuda-arch 80 \
  --build-root .run/a100-profile-build
```

For the four-format study from the serving commands, at C8/P2K:

```bash
model_root=results/cuda-context-study-a100-merged-70b
roofline_output=results/a100-roofline-c8-p2048

# Preview commands and capture count; no GPU work or output writes.
python3 scripts/collect_a100_roofline.py plan \
  --cells "$model_root"/{IQ1_M,Q2_K,Q4_K_M,Q8_0}/c8/p2048 \
  --output "$roofline_output"

# Collect serially, export values, and generate the PDF.
python3 scripts/collect_a100_roofline.py run \
  --cells "$model_root"/{IQ1_M,Q2_K,Q4_K_M,Q8_0}/c8/p2048 \
  --output "$roofline_output"

# Re-export and plot saved captures only, including a partial interrupted run.
python3 scripts/collect_a100_roofline.py report --output "$roofline_output"
```

Operators: FFN gate/up, FFN down, Q/K/V projections, attention output,
FlashAttention and LM head. Each operator/phase/GPU has its own capture so an
early global launch limit cannot consume the entire budget before later
operators. Both server GPUs remain visible, while one GPU's counters are
collected. The LM head is captured only on its output-layer GPU.

The four-format example schedules **120 captures**, each capped at five
matching kernel launches, with C discarded warmup requests and C profiled
requests. Nsight may replay each sampled launch. Captures do not guarantee
five samples of every kernel shape; CSVs retain the actual sample count.
Use `--operators ffn_gate_up` or `--phases decode` for an explicit subset.
These options and `--launch-count` are recorded in `roofline-plan.json`.

Failed captures stop the run and preserve partial CSVs; no automatic retry is
scheduled. Completed matching captures can resume. A changed plan or retry of
a failed capture requires a fresh output directory. Missing operators and
zero-FP32 groups remain visible in the audit without invented plot positions.

## Outputs

- `roofline-samples.csv`: measured counts, seconds, bytes, coordinates and source hashes per launch.
- `roofline-points.csv`: medians per operator/configuration, with actual sample counts.
- `roofline-validation.json`: capture coverage, unavailable values and formulas.
- `roofline-explained.pdf`: one page per GPU, with format columns and phase rows.
- `roofline-explained-gpu*.png` / `.svg`: individual page exports.
- `captures/`: original Nsight reports, instruction instances, metadata and logs.

The original A6000 results and figures are not inputs to collection.
These measurements require separate profiling runs and are not serving
latency or throughput. A 70B/64K cell excluded by FP16-KV capacity cannot be
profiled; it must stay excluded.
