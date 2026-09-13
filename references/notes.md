# Supporting reference and method notes

The concise findings are in [FINDINGS.md](../FINDINGS.md); experiment status is in [README_LOG.md](../README_LOG.md).

## Remote reproduction update

Use [execute.sh](../execute.sh) and the remote section of [README](../README.md)
for one-line A100/MI300X experiments. The old CUDA-named runner/profile files now
support both backends; names were retained to avoid additional wrapper scripts.
Model manifests now include the measured GGUF sizes and SHA256 digests.

The local host rebooted into driver 595.91.07 with counter policy enabled.
System libraries now match; the old local library repair is conditional on the
old driver. Counter access now works on GPU0 with NCU 2026.3.0 after pausing the
DCGM exporter on `tripod-worker`. A real 1B Q4_K one-column matvec capture records
7,979,008 executed instructions and 34,592 ns in
[profile.ncu-repz](../tmp/ncu-1b-20260912T173016Z-N8hkRy/profile/profile.ncu-repz).
The wrapper initially missed NCU's new `.ncu-repz` suffix. Permission-denied and
resource-busy captures remain historical evidence. Exporter restoration is
pending; [state.json](../tmp/dcgm-counter-access/state.json) records the saved
setting and `paused` status. The [README](../README.md#run-nsight-compute-on-this-host)
has the sudo-assisted restore command.

The [temporary 1B matrix](../tmp/run_1b_hardware_metrics.py) collects real HTTP
inference with 2,048 input and 128 output tokens: matrix/dequantization selection
at eight clients, one/eight-column matvec selection at one/eight clients, and
eight-column matvec with warm prefixes at eight clients. All **23 captures
completed and validated, with 166 kernel samples**. They distinguish the mixed tensor types in Q8_0/Q4_K_M/IQ1_M and
sample two launches per matching launch configuration. Each directory under
[`results/cuda-study-1b/hardware-counters`](../results/cuda-study-1b/hardware-counters) retains request metadata,
raw NCU reports and counter summaries; `progress.json` records the validated
captures and `status: complete`.
The capture gate covers the entire HTTP burst. `prefill`, `decode` and `reuse`
are selector labels, not separate phase timing gates; selected matvec launches
can include final-prefill tail work.
NCU uses kernel replay, cold caches (`--cache-control all`) and unchanged clocks
(`--clock-control none`). Measured bandwidth, cache hit rates, occupancy, IPC,
tensor activity and stalls describe these selected launches, not whole-model
averages or unprofiled serving latency. Hardware counters for 8B/70B remain
unmeasured.

Remote checks run a real CUDA/HIP sentinel before counter experiments, excluding
two outside launches and collecting the inside launch. On MI300X, that launch
has 64 waves; leaked outside launches have four each. Gated AMD counters require
ROCm 7.14+ / SDK 1.3.2+, since earlier selected-region implementations only gated
traces. Counter groups are checked against the installed catalog before the model
is loaded; each group uses a separate warmed HTTP run. These behaviors were
checked against the [AMD release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html),
[ROCTx guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofiler-sdk-roctx.html)
and [counter discovery guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-avail.html).

CUDA output keeps replay/cache/clock controls, kernel shape and units. AMD output
keeps per-dispatch raw counters, VGPR/SGPR/LDS resources and native formulas.
AMD IPC and coalescing conventions are labelled separately from NVIDIA SM IPC
and load sectors/request. Memory read/write passes remain separate; values from
different dispatches are never combined to invent a single bandwidth measurement.
Profiler timings remain diagnostic, separate from unprofiled serving results.


## Sources and testable claims

Read on 2026-09-12. This is the research note for the CUDA serving experiment; it records source claims separately from proposed tests.

### Access and provenance

- The user subsequently supplied [references/Tapia.pdf](Tapia.pdf), a two-page project paper titled *Understanding Large Language Model inference on Cloud: Datacenter GPUs: A Quantization perspective*. Both pages were read, and page 2's charts and counter table were visually inspected. It was absent during the initial repository search; that earlier access gap is now resolved.
- The requested [ACM paper, DOI 10.1145/3771563](https://doi.org/10.1145/3771563) has an accessible abstract, but its PDF returned 403. Its authors' [lab publication page](https://thexsel.github.io/papers.html) links the accessible [24-page author preprint PDF](https://arxiv.org/pdf/2508.08531), also available as [HTML](https://arxiv.org/html/2508.08531v1). That preprint was read, including its numbered findings and profiling discussion. It is August 2025 v1, titled *Profiling Large Language Model Inference on Apple Silicon: A Quantization Perspective*. The December 2025 ACM publication is titled *Benchmarking and Characterization of Large Language Model Inference on Apple Silicon* and lists 26 pages. Exact final-version changes were not verified.
- The [January 10 Medium article](https://medium.com/@afsara.benazir/what-rocm-profiling-revealed-about-quantized-llms-on-amds-fastest-gpu-c0edfab9624f) was readable in full through web retrieval. The [January 3 Medium article](https://medium.com/@afsara.benazir/profiling-quantized-llms-on-amd-mi300x-with-rocm-what-the-hardware-is-really-telling-us-ac3cc33ebcff) returned an empty shell/direct-access 403, but its full textual discussion was available in the search index. Embedded plot pixels and underlying raw measurements were not independently verified.

### What the paper actually claims

The preprint studies single-request llama.cpp inference on Apple GPUs and RTX A6000 configurations, using 8B–405B models. Its relevant findings are: FFN/matrix operations dominate; higher precision can win in prefill; fewer weight bits do not guarantee lower latency; dequantization and codebook work can outweigh memory savings; and some aggressively quantized Apple kernels become arithmetic limited. It distinguishes prefill GEMM from single-request decode GEMV. Crucially, Figure 6 and page 19 describe a different CUDA result: lower-bit models, including IQ1_M, perform well on A6000. Its Apple memory-capacity and cost advantages concern particular workstation configurations and offloading conditions. These are scoped observations, not universal hardware rankings. See findings 1, 2, 4–9 and §§4.3, 5.5 in the [preprint](https://arxiv.org/pdf/2508.08531).

### What the MI300X articles add

The [January 3 article](https://medium.com/@afsara.benazir/profiling-quantized-llms-on-amd-mi300x-with-rocm-what-the-hardware-is-really-telling-us-ac3cc33ebcff) uses ROCm 7.0, llama.cpp HIP and a single MI300X. For 70B prefill with token batch 128, Q8_0 is fastest; IQ1_M is reported at 3.01 ms per input token, 2.64× slower. IQ1_M versus Q4_K_M has reported vL1D coalescing of 66.84% versus 77.62% and IPC of 0.65 versus 0.77. The interpretation emphasizes unpacking, codebook loads and instruction efficiency. Batch sizes 128/256 in this prefill comparison are not demonstrated to be independent concurrent requests. Prefill latency per input token must also not be relabeled as decode TPOT.

The [January 10 article](https://medium.com/@afsara.benazir/what-rocm-profiling-revealed-about-quantized-llms-on-amds-fastest-gpu-c0edfab9624f) profiles 70B Q4_K_M/IQ1_M with a 2048-token prompt and one output token. It reports low matrix utilization and more scalar/control overhead for IQ1_M, interpreting the run as limited by utilization/instructions. Its later memory-bound roofline example is labeled **1B Q4_0**, a different configuration. The text changes scope ambiguously, so the apparent contradiction should be resolved by retaining model, precision, phase and workload labels with each measurement. Neither section establishes a bottleneck for concurrent CUDA serving.

### What the uploaded project PDF establishes

[Tapia.pdf](Tapia.pdf) extends the MI300X prefill analysis into a cloud research question. Its plots vary prompt length (2048/4096) and token batch (128/256), while its counter table compares IQ1_M with Q4_K_M: coalescing 66.84%/77.62%, cache bandwidth 8290.67/6465.68 GB/s, L1 bandwidth utilization 10.15%/7.91%, and IPC 0.65/0.77. The prose reports IQ1_M's 2.64× slowdown relative to Q8_0 and attributes it to instruction overhead and weak matrix-engine use. Its plan calls for more explicit pipeline/instruction attribution.

These are single-accelerator prefill results. The PDF does not document simultaneous HTTP clients, arrival timing, per-request TTFT/TPOT, or a mixed prefill/decode serving trace. The new experiment tests that missing serving dimension. Also, the table's several-TB/s **L1 cache** traffic is not HBM bandwidth. Without measured DRAM traffic and limiting-stall evidence, low cache/fabric utilization does not prove that all memory effects have been excluded. These are interpretation limits, not a rejection of the observed latency ranking.

### Minimal experiments and interpretation rules

| Question | Controlled CUDA test | Evidence needed for a conclusion |
|---|---|---|
| Does the precision ranking survive serving? | Same checkpoint, Q8_0/Q4_K_M/IQ1_M; concurrency 1 and 8; fixed prompts, output counts, server slots and token batches. | Repeated aggregate output tokens/s plus TTFT, TPOT and end-to-end latency. A smaller model that is reliably slower is a counterexample to monotonic speedup. |
| Does concurrency change the limiting work? | Compare the same format at both loads. | Kernel timings/types and measured batch sizes; separate improved system throughput from slower individual requests. |
| Does IQ1 unpacking explain a slowdown? | Compare dominant quantized kernels at matched shapes. | Instruction mix, issue efficiency, memory stalls, occupancy and tensor-pipeline activity. Timing/name evidence alone supports a hypothesis, not causal attribution. |
| Is memory bandwidth saturated? | Profile dominant prefill and decode kernels separately. | Measured DRAM bytes/time, percent of peak, cache traffic and stall reasons. Low FLOP utilization or low achieved bandwidth alone is insufficient. |
| Do longer contexts add serving cost? | Optional second prompt length while holding output count fixed. | TTFT and TPOT changes; report cache/attention evidence where available. |

If profiling only a short trace, report how its request mix relates to the main benchmark. Kernel names can reveal dispatch changes, but shape metadata is stronger evidence of batching. End-to-end serving throughput includes both phases and HTTP/scheduling work; it cannot be substituted for isolated GEMM throughput.

For a fixed dense layer with `W` weights, decode batch `B` and stored weight bytes `s`, an idealized weight-only intensity is `2WB/(Ws) = 2B/s` operations per weight byte. This is an analytical illustration of batch reuse, **not a measured roofline**: metadata, dequantization, activation/KV traffic and cache reuse change actual intensity. Integer instructions must not be added to FLOPs and compared against an FP/tensor ceiling without a corresponding mixed-instruction model.

### CUDA profiling correspondence

ROCm terminology has approximate CUDA counterparts, not numerically interchangeable counters. Use NVIDIA's [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): `SpeedOfLight` for compute/memory utilization; `ComputeWorkloadAnalysis` and `InstructionStats` for pipeline/IPC/instruction mix; `WarpStateStats`, `SchedulerStats` and `Occupancy` for execution readiness; and `MemoryWorkloadAnalysis` for DRAM/cache traffic and transaction behavior. Nsight offers precision-specific roofline sections. Metric availability depends on the GPU/tool version. Profiling may replay kernels and materially perturb execution, so collect ordinary request latency separately. NVIDIA warns that a stall is consequential when schedulers cannot issue; high occupancy by itself does not establish good performance.

If hardware counter access fails, preserve the failure and mark counter-dependent claims **unresolved**. GPU utilization sampling, file size, static assembly and kernel durations remain useful observations but do not recreate measured bandwidth, cache hit rates, occupancy or a hardware roofline. Apple-versus-cloud economics, MI300X-specific rankings, model accuracy and large-model offloading are outside a small, fully resident CUDA experiment.


## Reading the CUDA kernel traces

Source: llama.cpp commit `3f5e94d7c2ab2267fe39852051777fe30c1f49ef`. These are source-backed expectations for RTX A6000 (`sm86`), not measured utilization. The build uses `CMAKE_CUDA_ARCHITECTURES=86`, with `GGML_CUDA_FORCE_CUBLAS` and `GGML_CUDA_FORCE_MMQ` both off. Quantization names describe mixed tensor recipes: an IQ1_M or Q4_K_M file can launch kernels for other tensor types.

For ordinary quantized weight matrices with F32 activations/output, dispatch checks MMVQ before MMQ. On this GPU, **MMVQ accepts 1–8 activation columns**. Above eight, Q4_K and Q8_0 support MMQ; IQ1_M does not. The A6000 takes the supported-type integer-MMA branch before the older-GPU `ne11 < 64` heuristic, so that 64-column limit does **not** apply here. [Dispatch](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/ggml-cuda.cu#L1815), [MMVQ threshold](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mmvq.cu#L318), [MMQ support and selection](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mmq.cu#L266).

| Weight tensor | Decode, 1–8 columns | Typical prefill, 512-token microbatch |
| --- | --- | --- |
| Q8_0 (`ggml_type` 8) | MMVQ; Q8_1 activation quantization and integer DP4A dot products | MMQ; integer tensor operations |
| Q4_K (`ggml_type` 12) | MMVQ; packed-weight extraction/scales and DP4A | MMQ; weight unpacking into integer tiles and integer tensor operations |
| IQ1_M (`ggml_type` 29) | MMVQ; codebook lookup, bit extraction, scale/delta reconstruction and DP4A | Separate weight dequantization to FP16, followed by dense cuBLAS GEMM |

The IQ1_M prefill fallback selects FP16 by default on this GPU, converts both operands as needed, and requests `CUBLAS_GEMM_DEFAULT_TENSOR_OP`; an explicit precision/environment override can change this. Q4/Q8 MMQ instead selects integer-MMA helpers, whose Ampere implementation contains `mma.sync...s32.s8.s8.s32`. DP4A in MMVQ is a packed integer dot product, not a tensor-core MMA instruction. [cuBLAS conversion and call](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/ggml-cuda.cu#L1408), [MMQ helpers](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mmq.cuh#L742), [integer MMA](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mma.cuh#L915).

Read a demangled `mul_mat_vec_q<type, ncols_dst, has_fusion, small_k, halve_iters>` literally. For example, **if observed during decode**, `<29, 8, false, false, false>` establishes eight activation columns in that IQ1_M matrix operation. Combined with server slot timing, it supports actual batching of eight requests. It does not imply eight throughout the run; arrivals and drain change the count. The booleans select fused operations and launch/loop variants, not utilization. In `mul_mat_q<type, J, fallback>`, `J` is a tile-column parameter, **not** the request count. [MMVQ template and column loop](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mmvq.cu#L583), [MMQ template](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mmq.cuh#L945).

IQ1_M decode explicitly loads `iq1s_grid_gpu`, extracts packed fields, reconstructs scales, and performs extra integer work. The MMVQ column loop invokes this dot-product routine for each column. This supplies a plausible mechanism to investigate when batching changes quantization rankings, but **does not establish an unpacking, integer-pipeline, or memory bottleneck**. Nsight Systems can establish kernel paths, counts, durations and overlap; static instructions do not measure their utilization. The subsequent 1B NCU captures provide measured GDDR6 traffic, occupancy, tensor activity and stalls for selected kernels; they do not extend those measurements to 8B/70B. [IQ1_M dot product](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/vecdotq.cuh#L1286).


## CUDA environment diagnosis and recovery

Historical diagnosis, 2026-09-12 UTC, before the reboot and exporter pause described above. This host has two NVIDIA RTX A6000 GPUs. GPU execution, NVML telemetry, and Nsight Systems CUDA tracing worked with the project-local environment below, while Nsight Compute returned `ERR_NVGPUCTRPERM`. Counter access was subsequently restored for the 1B captures.

### Use the working environment

```bash
cd /home/hys4qm/cloud-serving-benchmark
source .run/cuda-recovery/env.sh
nvidia-smi
python3 .run/cuda-recovery/probe_cuda.py
```

The original repair used `.run/cuda-recovery/lib` in `LD_LIBRARY_PATH`; its three symbolic links select NVIDIA's `595.71.05` CUDA driver, NVML, and PTX JIT compiler libraries. That local repair changed no system libraries, driver modules, permissions, or persistent shell configuration. The system CUDA driver API also initialized successfully before recovery; the observed mismatch specifically broke NVML/nvidia-smi. The current `env.sh` only applies those links when the loaded driver needs them; the post-reboot 595.91.07 system libraries match the loaded module.

### Evidence

| Check | Result | Raw evidence |
|---|---|---|
| Loaded kernel module | 595.71.05 | `.run/cuda-recovery/before.json` |
| System userspace libraries | 595.91.07 | `.run/cuda-recovery/before.json` |
| Original `nvidia-smi` | Exit 18, driver/library mismatch | `.run/cuda-recovery/before.json` |
| Original `cuInit(0)` | `CUDA_SUCCESS` | `.run/cuda-recovery/before.json` |
| Matching local libraries | NVML and CUDA initialization succeed | `.run/cuda-recovery/after-nvidia-smi.txt`, `after-cuda.json` |
| Minimal CUDA kernel | Wrote and copied back 42.0 | `.run/cuda-recovery/nsys-probe.txt` |
| Nsight Compute | `ERR_NVGPUCTRPERM` | `.run/cuda-recovery/ncu-user.txt` |
| Driver profiling policy | `RmProfilingAdminOnly: 1` | `.run/cuda-recovery/kernel-params.txt` |
| Passwordless sudo | Unavailable: password required | `.run/cuda-recovery/sudo-permissions.txt` |
| Nsight Systems | Captured one kernel and one device-to-host copy | `.run/cuda-recovery/nsys-stats.csv`; initial raw probe removed, canonical serving traces retained in `results/` |

The tiny kernel measured 1,760 ns in its diagnostic trace; this is only a profiler functionality test, not an LLM performance result.

### Hardware snapshot

Each GPU reports compute capability 8.6, 84 SMs, 6 MiB L2, 384-bit GDDR6, 8,001 MHz maximum memory clock, 2,100 MHz maximum SM clock, 49,140 MiB reported total memory, and a 300 W power limit. The memory clock/bus-width specification implies approximately 768 GB/s theoretical bandwidth; it is not a measured achieved bandwidth. CUDA reports slightly less allocatable memory than NVML's total. GPU0 drives a display; GPU1 does not. The GPUs have an NV4 connection according to `nvidia-smi topo -m`.

At the first recovered NVML snapshot both GPUs were idle (0% utilization), with no listed GPU processes. GPU0 used 33 MiB and drew about 34 W; GPU1 used 1 MiB and drew about 40 W. These values are a snapshot, not workload measurements. Raw data: `.run/cuda-recovery/gpu-specs.csv`, `topology.txt`.

RTX A6000 is an Ampere workstation GPU. Results on this host should be described as CUDA cloud-style serving experiments on A6000, without presenting them as MI300X or datacenter-GPU measurements.

### Recovery provenance and reproduction

The matching userspace libraries came from the [official NVIDIA 595.71.05 download directory](https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05/). The no-compat32 runfile was downloaded with its SHA-256 file, verified successfully using `sha256sum -c`, and extracted using `--extract-only`; the installer was not run. NVIDIA describes this package in its [package selection documentation](https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05/README/selectdriver.html).

```bash
cd .run/cuda-recovery
curl -fLO https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05/NVIDIA-Linux-x86_64-595.71.05-no-compat32.run
curl -fLO https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05/NVIDIA-Linux-x86_64-595.71.05-no-compat32.run.sha256sum
sha256sum -c NVIDIA-Linux-x86_64-595.71.05-no-compat32.run.sha256sum
sh NVIDIA-Linux-x86_64-595.71.05-no-compat32.run --extract-only --target driver-595.71.05
mkdir -p lib
ln -s ../driver-595.71.05/libcuda.so.595.71.05 lib/libcuda.so.1
ln -s ../driver-595.71.05/libnvidia-ml.so.595.71.05 lib/libnvidia-ml.so.1
ln -s ../driver-595.71.05/libnvidia-ptxjitcompiler.so.595.71.05 lib/libnvidia-ptxjitcompiler.so.1
```

The existing symlinks need not be recreated. The recovery files are under the repository's ignored `.run/` directory. The local driver files must match the actually loaded kernel version; if the host is rebooted into a different driver, diagnose the versions again.

### Profiler commands and limits

```bash
source .run/cuda-recovery/env.sh
nvcc -arch=sm_86 -O2 .run/cuda-recovery/probe_kernel.cu -o .run/cuda-recovery/probe_kernel
ncu --target-processes all --set basic --launch-count 1 .run/cuda-recovery/probe_kernel
## Historical result before counter policy/exporter changes: ERR_NVGPUCTRPERM.

```

Use the serving wrapper below for new Nsight Systems captures; it restricts
inherited environment metadata as well as selecting the measured request range.

Nsight Systems provides GPU kernel times, kernel counts, launch dimensions, and CUDA copy activity without hardware counters. NVML provides sampled utilization, memory allocation, clocks, and power. Neither source supplies achieved occupancy, L2 hit rate, achieved DRAM bandwidth, or tensor-pipeline utilization; the subsequent NCU captures measure those directly for selected 1B kernels. NVIDIA documents the original restriction at [ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM).


## CUDA profiling for the serving experiment

`scripts/07_profile_cuda.sh` profiles the same HTTP server settings as the
performance experiment. It first warms one concurrent burst, starts capture,
sends one synchronized burst of 1 or 8 requests, and stops capture after all
responses complete. Baseline prompt reuse and idle-slot RAM caching stay disabled
(`--no-cache-prompt --cache-ram 0 --no-cache-idle-slots`). CUDA graph node tracing
preserves the server's graph execution while exposing individual kernels.

The small `profile_gate.cpp` preload library waits for files from the HTTP
driver and calls `cudaProfilerStart/Stop` in the server process. This excludes
model loading, tokenizer preparation, and warmup. The gate was checked with a
CUDA probe: one warmup launch was excluded and exactly three capture launches
appeared in the trace.

### Run

Stop other experiments on the selected GPU before profiling. Use the same
model, server configuration, and workload as the performance runs:

```bash
source .run/cuda-recovery/env.sh  # This machine's local driver-library repair.
export CONFIG_FILE="$PWD/config/cuda-study.env"
export NSYS_BIN=/home/hys4qm/.cuda-13.0/nsight-compute-2025.3.1/host/target-linux-x64/nsys
export MODEL_PATH="$PWD/models/Llama-3.2-1B-Instruct.i1-Q4_K_M.gguf"
export MODEL_ALIAS=llama-3.2-1b-q4-k-m
PROFILE_CONCURRENCY=8 PROFILE_OUTPUT="$PWD/results/example-q4-c8-trace" \
  bash scripts/07_profile_cuda.sh
```

Each output directory must be new. Change the model and concurrency to repeat
the small matrix. The profiler uses port 8081 by default; `PROFILE_PORT` changes
it. This profiles one GPU selected with `GPU_DEVICE`, even if the machine has
multiple GPUs.

For matched prefix traces, set `PROFILE_REPEAT_PROMPT=1` and capture separate
new directories with `PROFILE_PREFIX_REUSE=0` and `PROFILE_PREFIX_REUSE=1`.
Both use eight identical 2,048-token prompts and eight generated responses of
128 tokens. The driver primes the eight slots before capture and adds a cached
warmup for reuse-on. Request `cache_prompt=true` overrides the server's default;
idle-slot RAM caching stays off. Metadata verifies matching input hashes and
2,047 cached plus one evaluated token per cached request. These are warm per-slot
KV hits, not a measurement of arbitrary cross-request global prefix sharing.

For a short hardware-counter capture:

```bash
PROFILE_TOOL=ncu PROFILE_CONCURRENCY=8 \
PROFILE_OUTPUT="$PWD/results/example-q4-c8-counters" \
  bash scripts/07_profile_cuda.sh
```

This samples four matching quantized matrix kernels. `PROFILE_KERNEL_REGEX`,
`PROFILE_LAUNCH_COUNT`, and `PROFILE_LAUNCH_SKIP` refine that sample after reading
the trace. For example, `mul_mat_vec_q` selects the small-batch vector path;
`mul_mat_q` selects the matrix path. A kernel's function name is evidence of its
implementation; it is not an exact request-phase marker when the server mixes
prompt and decode work in an internal batch.

### What can be compared with the ROCm articles

| Article metric class | CUDA measurement | Availability here |
|---|---|---|
| Kernel time and hot functions | Nsight Systems CUDA kernel durations and time shares | Available |
| Launch shape and resource pressure | Grid/block dimensions, registers/thread, shared bytes/block | Available; these are not achieved occupancy |
| Explicit device transfers | CUDA memcpy bytes, direction, duration | Available; these exclude memory traffic inside kernels |
| GPU memory traffic and bandwidth | NCU `dram__bytes_read.sum`, `dram__bytes_write.sum`, `dram__throughput.avg.pct_of_peak_sustained_elapsed` | Available in selected 1B NCU captures |
| L1/L2 cache hit rates | NCU `l1tex__t_sector_hit_rate.pct`, `lts__t_sector_hit_rate.pct` | Available in selected 1B NCU captures |
| Achieved occupancy | NCU `sm__warps_active.avg.pct_of_peak_sustained_active` | Available in selected 1B NCU captures |
| Compute/Tensor Core activity | NCU SM throughput and tensor-pipeline active cycles | Available in selected 1B NCU captures |
| IPC and scheduling stalls | NCU `sm__inst_executed.avg.per_cycle_active`, `smsp__warp_issue_stalled_*_per_warp_active.pct` | Available in selected 1B NCU captures; active/elapsed denominators retained |
| AMD scalar/vector instruction mix | NVIDIA SASS instruction mix would require a separate NCU collection | No universal AMD SALU/VALU-to-CUDA equivalence; not measured |

The default NCU metrics were checked against this tool's GA102 metric catalog.
The original `RmProfilingAdminOnly=1` restriction and `ERR_NVGPUCTRPERM` probe
are historical permission failures, not workload results. After the policy was
enabled, resource-busy errors persisted until the exporter was paused; selected
1B captures now contain finite hardware counters. See
[NVIDIA's counter-permission documentation](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

### Read the output

- `metadata.json`: exact command, environment, token targets, request outcomes,
  and capture status; `prompts.json` stores the actual prepared prompts.
- `profile.nsys-rep` and `profile.sqlite`: original trace and exported data.
- `profile_summary.md` / `.json`: compact function-class summary.
- `kernel_summary.csv`: timings grouped by function **and launch shape**, with
  register/shared-memory resources; `kernels.csv` retains each kernel event.
- `memory_copies.csv` / `cuda_api.csv`: explicit transfers and host CUDA calls.
- `server-profiler.log` / `ncu.csv`: profiler and server diagnostics, including
  unavailable counters or other failures.
- `profile.ncu-rep` / `profile.ncu-repz` and `counter_summary.csv` / `.json`:
  raw NCU report and per-launch counters with units and derived DRAM bytes/time.

Use the unprofiled experiment for throughput, TTFT, TPOT, and latency. Nsight
Compute replay can change execution, while tracing also has overhead. A kernel
time share has total kernel time as its denominator; overlapping kernels count
separately. The union of kernel intervals is reported separately. Neither this
timeline-active percentage nor `nvidia-smi` utilization measures achieved
occupancy or Tensor Core utilization. The trace cannot establish saturated
DRAM bandwidth, cache efficiency, or executed dequantization instruction counts.
See the [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
and [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

## Control discovered during profiling

The pinned server enables idle-slot RAM caching by default even when a request
sets `cache_prompt=false`. Original 1B traces included 224 state-copy operations,
499 MB of D2H data, before the first inference kernel. Explicitly disabling
`--cache-ram` and `--cache-idle-slots` eliminated them. Preliminary performance
runs are archived separately and excluded from the report. The corrected c8
trace still transfers 525,336,576 bytes to the host: eight requests times 128 output
tokens times 128,256 vocabulary entries times four bytes/logit. This is explicit
CUDA copy traffic for CPU sampling, not measured kernel memory bandwidth.

Nsight archives inherit environment metadata. The profiling runner passes an
explicit runtime environment allowlist; initial archives were removed and
captures regenerated. Avoid passing unrelated environment variables to profilers.

## Final matched prefix trace evidence

All 12 prefix traces passed independent checks: matching input hashes, eight
valid 2,048/128-token requests, correct cache accounting, clean capture gates,
allowlisted environment metadata and zero pre-inference device-to-host copies.
Reuse-on always cached 2,047 tokens and evaluated one. Both modes copied exactly
525,336,576 bytes of logits to the host. Paths:
`results/prefix-study-{1b,8b}/profiles/{quant}-reuse-{off,on}`.

| Size | Format | Sum of kernel time, off → on (ms) |
|---|---|---:|
| 1B | Q8_0 | 1,186.67 → 551.71 |
| 1B | Q4_K_M | 1,181.17 → 562.56 |
| 1B | IQ1_M | 1,248.04 → 520.51 |
| 8B | Q8_0 | 5,817.54 → 2,353.10 |
| 8B | Q4_K_M | 6,007.40 → 2,428.75 |
| 8B | IQ1_M | 6,305.39 → 2,094.76 |

With reuse, large-prompt MMQ and standalone dequantization/cuBLAS GEMM launches
disappear. In the 8B IQ1_M off trace, dequantization takes 791.24 ms (12.55% of
kernel time) and FP16 GEMM takes 1,993.35 ms (31.61%). Its matrix-vector time
changes from 1,523.83 to 1,477.56 ms, while attention changes from 668.64 to
470.03 ms. Across all formats/sizes, the fraction of eight-column MMVQ launches
rises from about 88% to 98.5%; ramp/drain accounts for other widths. Each entry is
one diagnostic capture, with instrumentation overhead and no confidence interval.

## 70B models and two-GPU interpretation

The following are metadata facts and source-derived expectations, not measured
70B profiling results. Both local GGUFs identify Llama 3.3 70B Instruct with
70.554 billion parameters. Q4_K_M mixes Q4_K/Q5_K/Q6_K/F32 tensors
(39.600 GiB; 4.82 payload bits/parameter); Q2_K mixes Q2_K/Q3_K/Q5_K/Q6_K/F32
(24.564 GiB; 2.99 bits/parameter). Neither is a uniform bit-width model.
Embedded metadata names Meta's Llama 3.1 70B as the base model but supplies no
conversion repository or revision. The manifest records local file hashes.

With all layers offloaded, `--split-mode layer --tensor-split 1,1` assigns
41 transformer layers to GPU 0 and 39 plus the output head to GPU 1; the input
embeddings remain on CPU. F16 KV at 32,768 total context tokens occupies
10 GiB, split 5.125/4.875 GiB by layer ownership. See
[pinned placement code](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/src/llama-model.cpp#L1494).

On A6000, both Q2_K and Q4_K support fused quantized MMQ prefill using INT8
tensor-core instructions; actual 1–8-column decode uses DP4A MMVQ. Q2_K therefore
differs from IQ1_M's standalone dequantization/FP16 GEMM prefill path. Mixed
tensors dispatch individually, and HTTP concurrency need not equal matrix width.
See [dispatch](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/ggml-cuda.cu#L1860).

Per-GPU interval unions and their overlap establish simultaneous kernel activity,
not scaling or bandwidth saturation. One F32 hidden-state boundary tensor is
32 KiB/token; default CPU sampling downloads 501 KiB of logits/output token.
These component sizes do not describe all transfers.

`GGML_CUDA_P2P` is unset: explicit peer-access setup is skipped, but
[cross-GPU copies still use cudaMemcpyPeerAsync](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/ggml-cuda.cu#L2515).
CUDA can stage transfers through host memory; neither the API name nor NV4
topology establishes NVLink usage. See [NVIDIA's transfer documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#peer-to-peer-memory-transfers).
Two workstation GPUs and two quantization recipes do not establish datacenter
behavior or multi-GPU speedup.
