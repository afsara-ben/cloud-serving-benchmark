# MI300X inference and kernel evaluation

This is the separate AMD entry point. Defaults: **one MI300X (device 0)**,
Llama 3.3 **70B**, **IQ1_M,
Q2_K, Q4_K_M, Q8_0**, **2K/4K/8K/16K/64K input tokens**, **8/16/32 concurrent
requests**, FP16 KV, 512 generated tokens, one repetition. Every serving
repetition sends `2 × concurrency` requests after a concurrent warmup burst.
`--concurrency 64` is also supported. 1B and 8B use `--model-size`.

## Host requirements and model files

Use Linux with the MI300X ROCm driver/runtime, HIP compiler, hipBLAS, rocBLAS,
CMake >= 3.21, Ninja, Git, Python >= 3.10 with venv, and sufficient RAM/disk.
ROCm/HIP >= 6.1 is required by the pinned llama.cpp source. For **profiling**, also
install a compatible `rocprofv3`, `rocprofv3-avail`, and ROCprofiler-SDK ROCTx
development headers/library under `/opt/rocm` (override `--rocm`). The counter
probe tests capability instead of trusting a ROCm version string: some older
SDKs gate traces but still count warmup kernels. Such SDKs fail `doctor`.
GPU access and performance-counter permissions must already be configured.
Use full MI300X devices (304 CUs, approximately 192 GiB); partitioned devices are
rejected so full-device roof ceilings cannot be applied to a partition.
Use otherwise idle GPUs; these scripts do not change clocks or power limits.

The existing 70B Q2_K and Q4_K_M manifest entries are **local user-provided
files**, with no pinned upstream download source. Copy those exact GGUFs to
`models/mi300x/` on the AMD machine, or pass `--model-dir /path/to/files`.
Existing manifest paths and repository `models/` files are also reused read-only.
`prepare` downloads pinned IQ1_M/Q8_0 sources and verifies sizes and SHA256;
Q8_0 parts are verified before byte concatenation. It does not substitute a
different quantization recipe. A custom `--manifest` uses the same list-of-models
schema as `models/study-manifest-70b.json`, including size and SHA256.
Hugging Face downloads use its normal cache/authentication (`HF_TOKEN` if needed).
Allow disk space for the download cache **and** assembled model files.

## End to end

To reproduce **AMD versions of the four-panel serving poster and the runtime
bottlenecks report**, use this entry point (one MI300X by default):

```bash
bash scripts/mi300x_paper_pipeline.sh

# Match the current poster's 2K–16K grid; omit CONTEXTS to include 64K too.
CONTEXTS="2048 4096 8192 16384" OUTPUT=results/mi300x/paper-1gpu \
    bash scripts/mi300x_paper_pipeline.sh
```

This builds HIP, verifies/prepares models, measures all four formats at C8/C16/C32,
then profiles **Q2_K/Q4_K_M, 2K, C8 on the selected GPU**. The diagnostic stage
captures a complete pure-prefill batch and a complete C8 decode batch in separate
runs for each counter group. It collects every dispatch in those batches, so a
kernel-name launch cap cannot discard later operators or another selected GPU. Warmup and all
other batches remain outside the capture. These synchronized diagnostic runs
never contribute to serving timings.

The outputs are under `OUTPUT/publication/paper1/`:

| File | Purpose |
|---|---|
| `runtime-cost-poster.pdf` | F1 TTFT, F2 concurrency throughput, F3 C8/C16 gains, F4 IQ1/Q4 gaps |
| `runtime-cost-merged.pdf` | TTFT and context-throughput panels |
| `runtime-bottlenecks-report.pdf` | Prefill/decode counters, instruction attribution, INT8/F16 rooflines, other operators, scratch/residency, selected GPUs and methods |
| `poster-data.json`, `coverage.csv` | Revalidated request measurements, derived plot values and missing/capacity-excluded cells |
| `comparison-data.csv`, `instruction-attribution-data.csv` | Q4/Q2 values, signed differences and extra-instruction shares |
| `kernel-medians.csv`, `selected-kernels.csv` | Five-launch medians and every selected dispatch with its capture identity |
| `operator-roofline-data.csv`, `operator-coverage.csv` | Per-configuration coordinates and trace-to-counter coverage, including zero-HBM cases |
| `validation.json`, `evidence.json`, `definitions.json` | Completion status, hashes, metric formulas, units and primary sources |

The paper stage rechecks raw request counts, server token/cache counters, stream
completion, attained concurrency, prompt hashes across formats, GPU placement,
zero host swap, and agreement with saved summaries. Captured gate/up kernels
must match **M=28,672, K=8,192, N=512 prefill / N=8 decode**, actual Q4_K/Q2_K
tensor types, and fusion scope across formats/GPUs. By default each unfused case
uses three gate plus two up launches; fused gate/up uses five whole fused kernels
and is never pooled with unfused launches. Kernel/grid/workgroup configurations
must match across the separate metric passes within each format/GPU.

The command exits nonzero if required evidence is missing, while preserving a
clearly labeled incomplete PDF and all available values. A counter can be
unavailable on an older installed SDK; it is never substituted with zero. An
observed zero-HBM dispatch has no finite HBM roofline point and is retained in
coverage as such. Capacity exclusions are valid resolved serving cells.

Useful overrides: `DEVICES="0 1"` if two GPUs are available, `PROFILE_TAG=paper2` for fresh
captures, `CAPTURE_BATCHES=2` to collect more phase batches, `SAMPLES=5`,
`BOTTLENECK_CONTEXT=2048`, and `BOTTLENECK_CONCURRENCY=8`. The selected bottleneck
context/concurrency must also be present in the serving sweep. Use a new output
directory after this protocol update; the previous source-hash checks correctly
reject mixing old/new serving protocols. No existing CUDA report is overwritten.

To collect diagnostics or rebuild figures after serving has finished:

```bash
PY=.run/mi300x/venv/bin/python
OUT=results/mi300x/paper-70b
$PY scripts/run_mi300x.py setup --diagnostic --devices 0
$PY scripts/run_mi300x.py bottlenecks --output "$OUT" --devices 0 \
    --formats Q2_K Q4_K_M --contexts 2048 --concurrency 8 --profile-tag paper2
$PY scripts/run_mi300x.py paper --output "$OUT" --devices 0 --profile-tag paper2
# Add --no-plots for CPU-only CSV/JSON validation without Matplotlib.
# Add --allow-incomplete only for an explicitly incomplete snapshot.
```

`paper` takes the same `--formats`, `--contexts`, `--concurrency`,
`--repetitions` and `--output-tokens` used for serving. The detailed publication
geometry is currently specific to 70B. The general `report` action supports
the other model sizes.

### AMD measurement definitions

* **Instruction attribution:** `SQ_INSTS`, F32 FMA/MAD and `SQ_INSTS_VALU_CVT`
  are collected together. CVT counts all VALU type conversions; it is not a
  renamed NVIDIA I2FP opcode count. The selected per-launch categories reconcile
  to total issued instructions. The extra-instruction share is reported only
  when Q2 has a positive net instruction increase.
* **IPC:** `sum(SQ_INSTS)/sum(SQ_BUSY_CU_CYCLES)`, following ROCm Compute
  Profiler's gfx942 convention.
* **Pipeline activity:** VALU uses
  `100*sum(SQ_ACTIVE_INST_VALU)/(CU_NUM*max(GRBM_GUI_ACTIVE))`; MFMA uses
  `100*sum(SQ_VALU_MFMA_BUSY_CYCLES)/(4*CU_NUM*max(GRBM_GUI_ACTIVE))`.
* **Occupancy:** accumulated resident wavefronts divided by active CU cycles,
  normalized to 32 wave64 slots/CU. Short-kernel occupancy can be inaccurate.
* **Registers:** both VGPR and accumulator VGPR metadata are required. The
  report shows the rounded combined allocation, a vector-register-only
  waves/SIMD bound and a workgroups/CU upper bound. These are not CUDA blocks/SM
  and do not include LDS, SGPR, scheduling or other residency limits.
* **Scratch:** TA buffer read/write wavefront counts measure spill/stack
  instructions, alongside scratch allocation bytes/work-item. They do not
  measure spill-only bytes or establish occupancy's causal runtime contribution.
* **Matrix rooflines:** INT8 uses `512*SQ_INSTS_VALU_MFMA_MOPS_I8`; F16 uses its
  separate counter. Both follow hardware issued-work/full-EXEC conventions.
  The INT8 ceiling is 2614.9 TOP/s per MI300X; override `--peak-int8-tops` for
  a measured ceiling. HBM traffic and arithmetic operands always come from the
  same dispatch/pass. FP32 remains explicitly labeled as a wave64 proxy.

These definitions follow the [AMD pipeline metric reference](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.2.0/conceptual/pipeline-metrics.html)
and [MI300 counter reference](https://rocm.docs.amd.com/en/docs-6.3.1/conceptual/gpu-arch/mi300-mi200-performance-counters.html).
`doctor` checks hardware counter compatibility and actually runs each group on
a tiny kernel before loading models. Custom gfx942 reductions ensure scalar
values across counter instances. Its installed catalog, definitions, extra
counter file, PCI-to-profiler device binding and probe results are copied into
`profiles/PROFILE_TAG/profiler-info/` by the bottlenecks stage.

The resulting AMD comparisons can differ from the CUDA findings. Figure titles
and explanations do not assume Q2 is slower, memory-bound or conversion-bound.

### General inference and exploratory profiling

From the repository root, on the MI300X host:

```bash
# One MI300X; creates an isolated Python venv and builds its own HIP binaries.
bash scripts/mi300x_pipeline.sh

# Two MI300X GPUs, with a separate output directory.
DEVICES="0 1" OUTPUT=results/mi300x/70b-2gpu bash scripts/mi300x_pipeline.sh

# Small first run: one format, 2K input, eight requests in flight, one repetition.
FORMATS=Q2_K CONTEXTS=2048 CONCURRENCY=8 REPETITIONS=1 \
OUTPUT=results/mi300x/smoke bash scripts/mi300x_pipeline.sh
```

The launcher runs setup → model preparation → plan → inference → serving reports
→ diagnostic build → profiler doctor → trace/counter captures → final reports.
By default, kernel profiling selects **Q2_K/Q4_K_M, 2K, concurrency 8,
repetition 1**. Each selected cell gets one uncapped trace and eleven separately
gated counter captures. `FORMATS`, `CONTEXTS`, and `CONCURRENCY` overrides also
become the profiling selection unless `PROFILE_FORMATS`, `PROFILE_CONTEXTS`, or
`PROFILE_CONCURRENCY` are supplied. For the entire kernel sweep:

```bash
DEVICES="0" OUTPUT=results/mi300x/full \
PROFILE_FORMATS="IQ1_M Q2_K Q4_K_M Q8_0" \
PROFILE_CONTEXTS="2048 4096 8192 16384 65536" \
PROFILE_CONCURRENCY="8 16 32" bash scripts/mi300x_pipeline.sh
```

Full profiling repeats model loading and a request burst for every counter group;
it can take substantially longer and produce much larger files than serving.
Use `PROFILE=0` for inference only. `JOBS`, `ROCM_PATH`, `MODEL_DIR`, `MODEL_SIZE`,
`MANIFEST`, `OUTPUT_TOKENS`, `REPETITIONS`, `PROFILE_TAG`, and `LAUNCH_COUNT` are
additional launcher overrides. Do not start a second copy against the same GPUs.

## Individual stages

```bash
python3 -m venv .run/mi300x/venv
.run/mi300x/venv/bin/pip install -r scripts/requirements-mi300x.txt
PY=.run/mi300x/venv/bin/python

$PY scripts/run_mi300x.py setup
$PY scripts/run_mi300x.py prepare
$PY scripts/run_mi300x.py plan --devices 0
$PY scripts/run_mi300x.py serve --devices 0 --resume

$PY scripts/run_mi300x.py setup --diagnostic
$PY scripts/run_mi300x.py doctor --devices 0
$PY scripts/run_mi300x.py profile --devices 0 \
    --formats Q2_K Q4_K_M --contexts 2048 --concurrency 8
$PY scripts/run_mi300x.py report
```

Keep `--output` consistent across serve/profile/report, and `--devices` consistent
between serving and profiling. Default inference/profile ports are 18080/18081;
override `--port` if occupied. `--split 1 1` is the default equal **layer** split
for two GPUs. It is not a claim of equal per-GPU allocations. A fresh
`--profile-tag rerun2` preserves prior captures. `--cells /path/to/cell.json ...`
selects specific completed AMD serving cells. CUDA cells are rejected.

`plan` and `report --no-plots` run without any GPU installation. `serve --resume`
only keeps completed cells whose configuration, binary hashes, and GPU identity
match, including the Python protocol source hashes. Failed, excluded, interrupted, or changed cells require a new output
directory; they are never silently replaced. Counter groups missing from the
installed catalog are recorded as unavailable; `profile` returns a nonzero exit
status for any missing/failed capture, while preserving completed serving data.

## Values and figures

All artifacts live under the selected AMD output directory:

| Artifact | Contents |
|---|---|
| `cells/.../cell.json` | Configuration, model and binary hashes, actual HIP devices, capacity screen, ROCm allocation evidence, status |
| `cells/.../requests.json`, `prompts.json` | Per-request streamed timing, exact-token/cache checks, prompt hashes; warmup is separate |
| `profiles/.../{trace,group}/rocprof/` | Raw kernel/API/ROCTx/memory-copy traces, counter CSVs and agent inventory |
| `profiles/.../kernels.json` | Dispatch identity, operator, phase, shape, grid/workgroup, VGPR/SGPR/LDS/scratch, counters, attribution coverage |
| `reports/serving.csv` | Per-run p95 TTFT/TPOT, output tokens/s, concurrency and failures/exclusions |
| `reports/runtime-cost-merged.pdf` | TTFT and TPOT versus input length; one page per model size/concurrency |
| `reports/throughput-context.pdf` | Output throughput versus input length |
| `reports/kernel-values.csv` | Instruction mix, memory traffic, L2 hits/misses, occupancy and dispatch resources |
| `reports/roofline-values.csv` | Same-dispatch operations, HBM bytes, duration, operations/byte and operations/s |
| `reports/roofline-explained.pdf` | Separate FP32 issued-wave proxy, F16 MFMA and INT8 MFMA panels; pages by workload and phase |
| `reports/capture-coverage.csv`, `definitions.json` | Missing counters, attribution coverage, units, formulas and roof sources |

Counter groups are checked with `rocprofv3-avail pmc-check` before model loading:

| Group | Requested values |
|---|---|
| instructions | SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_SALU, SQ_INSTS_MFMA |
| memory | FETCH_SIZE, WRITE_SIZE |
| cache | TCC_HIT, TCC_MISS |
| occupancy | MeanOccupancyPerActiveCU |
| scalar_fp32 | SQ_INSTS_VALU_ADD_F32, MUL_F32, FMA_F32, FETCH_SIZE, WRITE_SIZE |
| mfma_f16 | SQ_INSTS_VALU_MFMA_MOPS_F16, FETCH_SIZE, WRITE_SIZE |
| mfma_i8 | SQ_INSTS_VALU_MFMA_MOPS_I8, FETCH_SIZE, WRITE_SIZE |
| attribution | SQ_INSTS, SQ_INSTS_VALU_FMA_F32, SQ_INSTS_VALU_CVT |
| ipc | SQ_INSTS, SQ_BUSY_CU_CYCLES |
| activity | Custom gfx942 VALU/MFMA busy percentages |
| scratch | TA_BUFFER_READ_WAVEFRONTS_sum, TA_BUFFER_WRITE_WAVEFRONTS_sum |

`FETCH_SIZE + WRITE_SIZE` is converted from KiB to bytes. MFMA F16 work is
`512 × SQ_INSTS_VALU_MFMA_MOPS_F16`. The FP32 panel uses
`64 × (ADD_F32 + MUL_F32 + 2 × FMA_F32)`: **an issued-wave proxy assuming all 64
lanes active**, not the predicated-on NVIDIA SASS FLOP count in the CUDA figure.
It excludes MFMA/transcendentals. The two arithmetic domains remain separate.
An absent counter is unavailable, not zero. Counter operands are never joined
across passes, agents or dispatches; duplicate dimension rows are rejected from
derived coordinates. Definitions follow AMD's
[MI300 counter reference](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html).

Default roof ceilings are **per GPU**, theoretical: 163.4 TFLOP/s vector FP32,
1307.4 TFLOP/s dense F16 MFMA, 5.3 TB/s HBM, from the
[MI300X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf).
Override `--peak-fp32-tflops`, `--peak-mfma-tflops`, and `--bandwidth-tb-s` with
measured ceilings when available. Never double a per-dispatch roof for two GPUs.

ROCTx labels identify actual tensor roles/types and server `is_prompt` batches
(prefill/decode/mixed). Attribution follows HIP API correlation IDs because GPU
execution is asynchronous. Unclassified fusions remain explicit. Profiler agent
IDs remain raw IDs with the saved agent inventory; they are not assumed to be
HIP device indices. Counter collection defaults to the first eight captured
dispatches **per kernel name**. This does not guarantee every layer/operator/
phase is covered. Inspect the coverage and raw rows; raise `--launch-count`,
adjust `--launch-skip`, or select `--kernel-regex` and repeat with a new tag.
No profiler timing is used in serving figures. Kernel duration sums can overlap.

## Optional AMD full kernel analysis and calibrated roofline

Install AMD's `rocprof-compute` for its full counter analysis and built-in
roofline microbenchmarks. Run one physical GPU at a time:

```bash
$PY scripts/run_mi300x.py compute --devices 0 --formats Q2_K Q4_K_M \
    --contexts 2048 --profile-tag compute1
```

This runs the HIP `llama-bench` with synthetic 2K prefill + 512 decode tokens,
full layer offload and FP16 KV, then `rocprof-compute analyze`. Reports and
vendor-generated roofline data live under `compute/compute1/`. The default kernel
substring is `mul_mat`; use `--kernel-regex-compute flash_attn` for attention.
Despite the option name, the Compute Profiler filter is a **substring**.
Its replayed benchmark includes initialization and uses a single synthetic
sequence; compare its kernel metrics separately from HTTP serving results.
The optional tool supplies architecture-specific analysis and measured roof
ceilings through its [profile/analyze workflow](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/profile/mode.html).

## Isolation and capacity

AMD uses `.run/mi300x/{source,build,diagnostic-source,diagnostic-build,venv}`,
its own lock, process groups, ports, and results marker. Child environments
remove inherited GPU selection, profiler preloads and tuning switches before
setting `HIP_VISIBLE_DEVICES`. No shell startup files are changed. Inference
does not need ROCTx or any profiler. No CUDA launcher, source checkout, build,
manifest, or figure is modified. Upstream llama.cpp stores shared HIP kernel
source in `ggml-cuda/`; the AMD builds explicitly use `GGML_CUDA=OFF`,
`GGML_HIP=ON`, `gfx942`. Only the diagnostic build has annotations and HIP graphs
disabled. The source commit matches the existing benchmark pin.

One MI300X cannot cover the entire requested grid with FP16 KV. The CPU-only
`python3 scripts/run_mi300x.py plan --devices 0` command lists every cell's weight
file size and KV allocation. For 70B with 512 output tokens and padded slots:

| Input tokens | Concurrency | FP16 KV alone (GiB) | Single-MI300X implication |
|---|---:|---:|---|
| 16K | 8 / 16 | 41.9 / 83.8 | Leaves room for each format's weights; runtime allocation still checked |
| 16K | 32 | 167.5 | Q4/Q8 exceed capacity; Q2 is also very tight before buffers/headroom |
| 64K | 8 | 161.9 | Q4/Q8 exceed capacity; IQ1/Q2 require runtime allocation checks |
| 64K | 16 / 32 | 323.8 / 647.5 | Excluded for every format on one GPU |

The manifest's weight files are approximately 15.6 / 24.6 / 39.6 / 69.8 GiB
for IQ1_M / Q2_K / Q4_K_M / Q8_0. File sizes are not exact GPU allocations;
compute buffers and the default 2 GiB headroom also need space. The Q2/Q4
2K/C8 bottleneck comparison has a much smaller KV allocation (6.9 GiB).
Actual GGUF tensor placement,
current free memory, reserved headroom, and server allocation logs determine
feasibility. Excluded cases remain exclusions in CSV; there is no CPU KV fallback,
automatic context shrink, or fabricated latency. Full transformer/output offload,
exact prompt/output counts, no prefix-cache reuse and no process swap are checked.

The implementation has offline tests; MI300X hardware execution must be validated
on the target host with the small first-run command above.
