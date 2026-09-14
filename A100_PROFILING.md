# One-setting A100 hardware profiles

Use `scripts/profile_a100_setting.py` after measuring the desired setting with
the [serving launcher](A100_SERVING.md). It profiles one saved setting at a time,
reusing that cell's model, input messages, token counts, slots and GPU split.

For the scalar FP32 operator figure, use the [roofline collector](A100_ROOFLINE.md).
It targets operators individually and exports the counts, coordinates and PDF.

The default counter collection requests these full Nsight Compute sections in
one capture:

| Sections | Evidence |
|---|---|
| `SpeedOfLight` | Compute and memory utilization |
| `ComputeWorkloadAnalysis`, `InstructionStats` | Pipeline utilization, IPC and SASS instruction mix |
| `WarpStateStats`, `SchedulerStats`, `Occupancy`, `LaunchStats` | Stalls, scheduler readiness, occupancy and launch resources |
| `MemoryWorkloadAnalysis` | Memory utilization and cache behavior, supplemented by explicit DRAM bytes, global load/store requests/sectors and L2 read/write sectors |

Add `--roofline half`, `single`, `double`, `tensor` or `overview` for corresponding
roofline sections; multiple selections are supported. Choose based on actual
kernel arithmetic: a Q4 weight file does not imply an FP4 compute roofline, and
floating-point rooflines do not account for all integer unpacking work.

## Once on the A100 host

Use a complete Nsight Compute installation (`ncu`, including its `extras/python`
Report Interface). Nsight Systems (`nsys`) is needed only for the optional
timeline. Counter access must be enabled for your account.

Check section availability, then build the annotated server for A100 SM80.
Neither command runs inference:

```bash
python3 scripts/profile_a100_setting.py sections --roofline tensor
python3 benchmark/matrix_diagnostic.py build --server --cuda-arch 80 \
  --build-root .run/a100-profile-build
```

The diagnostic server labels actual batch phases and operations. These
captures enable `CSB_NVTX_OPS=1` and disable CUDA graphs for attribution;
ordinary serving continues to use its original binary/settings.

## Capture one setting

This example assumes 8B Q4_K_M / C8 / 2k already passed serving validation:

```bash
cell_dir=results/cuda-context-study-a100-2gpu-8b/Q4_K_M/c8/p2048

# Preview the capture and request budget, without GPU work or output writes:
python3 scripts/profile_a100_setting.py plan --cell "$cell_dir" \
  --phase decode --device 0 --launch-count 5 --roofline tensor

# Collect the full sections above and the Tensor Core roofline:
python3 scripts/profile_a100_setting.py counters --cell "$cell_dir" \
  --phase decode --device 0 --launch-count 5 --roofline tensor
```

Use `--phase prefill` for a separate prefill capture, or `--device 1` for the
second GPU's counters. Both server GPUs stay visible; this selects only which
GPU's kernels are profiled. For one GPU, use the saved `1gpu` path and device 0.
Device indices are logical positions in the saved `GPU_DEVICE` list; logical
1 means physical GPU 5 when that list is `2,5`.

Each command uses C discarded warmup requests and one C-request profiled burst,
at the saved input length and 512 output tokens: 8 + 8 requests at C8,
16 + 16 at C16, 32 + 32 at C32, or 64 + 64 at C64.
Only the selected phase's matching kernels are collected, but
HTTP requests still perform their complete prefill/decode. No serving rerun,
endpoint sweep, automatic timeline, supplemental capture or retry is scheduled.
Compatible completed captures are reused.

`--launch-count 5` caps the **total** matching launches on the selected GPU,
not five per kernel or shape. The default regex `mul_mat|gemm|gemv|mma` samples
matrix paths. Refine it with `--kernel-regex` and `--launch-skip`; it does not
cover every operator. An optional timeline helps identify dominant kernels:

```bash
python3 scripts/profile_a100_setting.py trace --cell "$cell_dir"
```

The timeline adds C warmup + C profiled requests. Inspect its `kernel_summary.csv`
and choose a kernel regex for counters, including attention kernels if desired.
Prefill/decode attribution uses same-capture NVTX labels, not kernel names.

## Read the outputs

The command prints the capture directory, normally beneath the cell's
`profiles/`. Settings determine its name. Use `--output` for a fresh explicit
destination after changing profiler/build identity or to retry a failed capture.

- `profile.ncu-rep` or `profile.ncu-repz`: open in Nsight Compute UI for section
  charts, SASS instruction mix and requested rooflines.
- `ncu_raw.csv`: all exported raw counters and native units.
- `counter_summary.json`: per-launch counters and operation/phase attribution.
- `counter_summary.csv`: compact counters; `counter_metrics.csv`: all scalar
  observations and detailed metric instances, including SASS opcodes.
- `metadata.json`: exact commands, outcomes, section validation and missing data.
- `study-capture.json`: source-cell/build identity and completion status.

Validation checks essential section evidence, including matching work/peak
metrics for rooflines. Missing, nonfinite or ambiguous values remain unavailable
and fail validation. Optional chart metrics still depend on GPU/tool support.

Nsight may replay sampled kernels several times; its memory backup can be
expensive for 70B. The launch cap bounds sampled launches, not internal replay
passes. Profile timing and per-kernel utilization are separate from ordinary
TTFT/TPOT/throughput and whole-server utilization. See NVIDIA's
[section/replay guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metric-collection)
and [CLI filters](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#profile).
