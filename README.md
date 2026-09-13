# Cloud Serving Benchmark

This project turns the poster's single-process prompt benchmark into a small,
request-level serving experiment. It starts an OpenAI-compatible `llama-server`
with continuous batching and sends real concurrent streaming requests. The same
project supports both NVIDIA CUDA and AMD ROCm GPUs.

The motivating poster is included at [references/Tapia.pdf](references/Tapia.pdf).
Read [FINDINGS.md](FINDINGS.md) for findings and [README_LOG.md](README_LOG.md) for
experiment history and resumption instructions. Supporting source and profiler
details are consolidated in [references/notes.md](references/notes.md).

The three-page 70B paper is available as [PDF](paper/llama70b.pdf) and
[editable Markdown](paper/llama70b.md), with separate Runtime Cost and Runtime
Bottlenecks sections, two figures, and two tables. [Build instructions](paper/README.md).
The [Runtime Cost poster figures](paper/poster/README.md) provide four-panel and
two-line layouts with editable exports; the optional 64K extrapolation is labeled
separately from the measured figures.

For TTFT, TPOT, generated tokens/s and GPU memory on **one or two A100s**, use
[the A100 serving launcher](A100_SERVING.md). It supports individual settings or
serial 1B/8B/70B runs at 8/16/32/64 clients and 2k/4k/8k/16k inputs, with one measured
run, no profiling, no long-context extensions, and no 70B FP16.
For the full Nsight section suite and optional precision-specific rooflines,
use [one-setting A100 profiling](A100_PROFILING.md) after the serving measurement.

## Context and concurrency study

The concise numbered plan is [FUTURE_WORK.md](FUTURE_WORK.md). It covers all three
model sizes, IQ1_M/Q2_K/Q4_K_M/Q8_0 and feasible FP16 weights, 8/16/32/64 clients, 2k/4k/8k/16k inputs,
and exactly 512 output tokens per request. Capacity extensions use 32k and 64k
inputs. Every format uses FP16 KV and the same two-GPU layer split.
Each setting uses one measured run. Existing first runs are reused; earlier
extra repetitions remain as provenance and are excluded from current comparisons.

Use the shared commands below after building the pinned CUDA server. Results go
to `results/cuda-context-study-{1b,8b,70b}/`, separate from the earlier
`results/cuda-study-*` experiments. The shared overview, reports, and controlled
diagnostics are in `results/cuda-context-study/`; its model links point to the
same data, without making copies. Each model folder has a summary and preserves
the hierarchy `<format>/c<clients>/p<prompt tokens>/r<repetition>/`.

```bash
export GPU_DEVICE=0,1
bash execute.sh study-setup cuda 1b
bash execute.sh study-setup cuda 8b
bash execute.sh study-setup cuda 70b
bash execute.sh study-plan cuda 1b
bash execute.sh study cuda 1b
bash execute.sh study cuda 8b
bash execute.sh study cuda 70b
```

For an independent repeat series, set the same `STUDY_NAME` for every command,
for example `export STUDY_NAME=cuda-context-study-repeat2`. `RUN_TAG` continues
to name the earlier baseline/prefix experiments and does not change these paths.

`study-plan` only prints the planned cells. `study` measures feasible cells and
records capacity exclusions separately. It resumes completed repetitions only
when model, binary, code, GPU identity, and settings still match. Run commands
sequentially on idle GPUs. The original `baseline`/`prefix` commands below retain
their earlier 2k-input/128-output workload.

The preparation helper uses original full-precision checkpoints for FP16, and
verifies downloaded GGUFs by hash. The existing 70B Q4/Q2 paths are recorded in
[the study manifest](models/study-manifest-70b.json); adjust those paths on another
host. **70B FP16 is excluded on the two A6000s before download or load.** On larger
hardware, explicitly select it with
`bash execute.sh study-setup cuda 70b --formats FP16 --include-remote-only`, then
`bash execute.sh study cuda 70b --formats FP16` in a fresh series. Memory screening
still applies.

All 8B formats use one explicit chat template so that the FP16 checkpoint's
additional date text cannot change the comparison inputs. Prepared prompts use
the server tokenizer and must match the requested token count exactly.

Profiling uses separate runs because tracing and counter replay affect timing.
`study-profile` selects 2k, 16k, and the longest jointly feasible input for each
model/format/concurrency, with duplicate endpoints removed. Raw serving requests,
per-GPU memory/clock telemetry, capacity evidence, and profiles stay separate.
Validation updates are brief; detailed artifacts document actual measurements.
Each profile captures one burst of C requests at concurrency C; the main serving
grid measures 2C requests. Kernel comparisons use matched operations and actual
activation widths. Profile phase fractions describe the diagnostic burst;
headline serving rates come from the separate unprofiled runs.

Build the private diagnostic binaries for your GPU architecture, then profile
completed serving cells. For example, use `--cuda-arch 86` for these A6000s:

```bash
python3 benchmark/matrix_diagnostic.py build --server --cuda-arch 86
bash execute.sh study-profile cuda 1b --diagnostic-server .run/context-diagnostic-build/baseline/llama-server
bash execute.sh study-profile cuda 8b --diagnostic-server .run/context-diagnostic-build/baseline/llama-server
bash execute.sh study-profile cuda 70b --diagnostic-server .run/context-diagnostic-build/baseline/llama-server
```

The private server annotates actual phases and matrix dimensions. Profiling
disables CUDA graphs for attribution; headline serving retains the pinned
unmodified binary. Compatible hardware metrics share a capture; prefill and decode
use separate filters, with one capture per phase and GPU and targeted supplements
only for missing operation samples. Full profiling follows the inference sweep.
The same diagnostic
tool's `run` command performs experiment 4; `--tool timing`, `nsys`, and `ncu`
keep ordinary operation timings, traces, and counters separate. Pass
`--repetitions 1` for each diagnostic case.

Counter replay saves GPU-memory state and may use host RAM for backup copies.
The reported serving VRAM footprint comes from unprofiled runs; profiler backup
storage is separate from the model's active GPU weights and KV cache.
[NVIDIA kernel-replay documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#kernel-replay).

Generate the organized tables, plots, and completion audit with:

```bash
python3 benchmark/report_context_study.py --study-root "results/${STUDY_NAME:-cuda-context-study}"
```

The report keeps unmeasured or excluded cells visible and only summarizes fully
completed serving cells. Hardware interpretation requires a separate evidence
review before the study can be marked complete.

## Run on a remote A100 or MI300X

Copy this repository to the server, including `models/*.json`, then use
[execute.sh](execute.sh). It contains **separate one-line commands** for every
experiment; running it without arguments prints the commands and runs nothing.
The 1B/8B commands use one GPU and the same pinned models/workload as the local study.
The CUDA 70B commands use two GPUs with an equal layer split.

```bash
bash execute.sh setup cuda 1b
bash execute.sh doctor cuda
bash execute.sh baseline cuda 1b
bash execute.sh prefix cuda 1b
bash execute.sh trace cuda 1b
bash execute.sh counters cuda 1b prefill
bash execute.sh counters cuda 1b decode
bash execute.sh counters cuda 1b reuse
```

Run those lines individually. Use `rocm` instead of `cuda` for MI300X, and `8b`
instead of `1b` after the 1B experiments. `setup` builds the pinned server and
downloads/verifies that size's three GGUFs. Download diagnostics append to
`logs/download.log`; build diagnostics append to `logs/build.log`.

Prerequisites: Linux, Python 3.10+, Git, CMake 3.21+, Ninja, g++, curl and flock;
a working driver/toolkit; and an idle GPU allocation. On Ubuntu the ordinary
build tools can be installed with `sudo apt-get install build-essential cmake
ninja-build git curl python3 util-linux`. Install a newer CMake separately if the
image supplies an older version. Toolkits/drivers are provided by the server image,
not installed or replaced by `execute.sh`.

- **A100:** CUDA compiler `nvcc`, Nsight Systems `nsys`, and Nsight Compute `ncu`.
  Put them on PATH or set `CUDA_HOME`, `NSYS_BIN` and `NCU_BIN`. The build detects
  the local GPU architecture; `CUDA_ARCHITECTURES=80` explicitly targets A100.
- **MI300X:** ROCm with `hipcc`, `hipconfig`, `rocminfo`, `rocprofv3`,
  `rocprofv3-avail` and ROCTx
  development headers/library. The build targets `gfx942`. Gated counter capture
  requires **ROCm 7.14+ / ROCprofiler-SDK 1.3.2+**; older tools can accept the
  region flag without actually excluding warmup counters. The real sentinel probe
  checks this behavior. [AMD release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html).

`doctor` compiles and profiles a tiny kernel and requires finite counters from
only the intended region. Every `counters` command repeats this check, so a policy
setting alone cannot produce a false success. It reports whether tools are missing,
device access is denied, a monitor such as DCGM owns the counters, or collection
otherwise failed. AMD requires access to `/dev/kfd` and its GPU render device;
containers must expose those devices. **The script does not grant itself sudo,
change driver policy, reboot, or stop monitoring services.** It prints the necessary
administrator action when access is blocked. [NVIDIA permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM),
[counter resource conflicts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq),
[AMD device access](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html).

`baseline` compares concurrency 1/8 with reuse disabled. `prefix` compares reuse
off/on at concurrency 8 with identical warmed prompts. Both retain three runs of
16 requests per setting. `trace` captures kernel timelines at concurrency 1/8.
Counter commands select matrix kernels associated with prefill, matvec kernels
at 1/8 activation columns, or matvec kernels with prefix reuse. The capture gate
covers the whole HTTP burst; phase labels select kernel families and column
counts, so matvec samples can include final-prefill tail work. CUDA retains
per-launch values and units for bandwidth, cache
hits, occupancy, IPC, tensor activity and load transactions. AMD collects compatible
groups in separate runs for instructions, IPC, memory traffic, cache behavior,
occupancy and coalescing. Reports remain separate from ordinary serving timing;
counter collection/replay changes execution, and metrics differ between vendors.
CUDA samples two matching launches per grid/block/shared-memory configuration;
AMD samples the first eight matching launches per kernel inside the measured
region. Compare matching shapes in the saved rows, not an unweighted average of
different matrix sizes. Each AMD counter group gets its own warmed HTTP run.
Mixed recipes also sample Q3_K within Q2_K, Q6_K within Q4_K_M, and Q5_K/IQ2_XXS
matvec within IQ1_M. These selected kernels do not cover every tensor type.

Results are under `results/remote-{cuda,rocm}/HOSTNAME/`. Each study has `summary.md`,
`summary.csv`, raw requests and reproducibility metadata; profiles have raw reports
and compact counter summaries. `GPU_DEVICE=1` selects a different GPU.
`RUN_TAG=repeat2` creates a separate experiment series. Benchmark commands resume
only matching completed cells; profile commands require new output directories.
Use a new `RUN_TAG` for repeated profiles or changed code/settings. To inspect both
benchmark plans without touching a GPU: `bash execute.sh plan rocm 8b`.

The portable CUDA runner has been exercised on this A6000. A100 and MI300X execution
must be validated on those machines; no remote measurements are claimed here.

## Llama 70B across two CUDA GPUs

`models/manifest-70b.json` records the existing local Llama 3.3 70B Instruct
Q4_K_M and Q2_K files by absolute path, byte size and SHA256. Their upstream
conversion revisions are unknown. On another host, update the paths to copies of
these same files. The 70B setup verifies local files; it cannot download them.

```bash
GPU_DEVICE=0,1 bash execute.sh baseline cuda 70b
GPU_DEVICE=0,1 bash execute.sh prefix cuda 70b
GPU_DEVICE=0,1 bash execute.sh trace cuda 70b
GPU_DEVICE=0,1 bash execute.sh trace cuda 70b reuse
```

The [two-GPU config](config/multigpu-study.env) uses an equal layer split and
streaming file loading (`--load-mode none`) to limit host-memory pressure, and
the same workload as 1B/8B: eight server slots, 32,768 total FP16 KV positions,
2,048 input tokens and 128 generated tokens per request. Q4's GPU-resident weights
plus the configured KV require about 49.0 GiB before temporary buffers; both GPUs
provide memory for the fully offloaded model. Q2 is a mixed roughly three-bit
recipe, distinct from IQ1_M. Comparing both formats on two GPUs does not measure
speedup over one GPU. Local experiment state is in [README_LOG.md](README_LOG.md).
The completed study contains 384 measured requests and six two-GPU kernel traces:
[baseline](results/cuda-study-70b/summary.md),
[prefix reuse](results/prefix-study-70b/summary.md),
[validation and trace analysis](results/70b-audit.json).

## Run Nsight Compute on this host

DCGM is NVIDIA's GPU monitoring software. `dcgmi` is its command-line control
tool; `dcgm-exporter` publishes GPU metrics for Prometheus. Nsight Compute (`ncu`)
collects detailed kernel counters. DCGM profiling can reserve the resource NCU
needs. Our exporter is the `gpu-operator/nvidia-dcgm-exporter` DaemonSet on
kind node `tripod-worker`. A host
`dcgmi profile --pause` only helps if it can reach the engine collecting counters.

Counter access is now working on GPU0 with NCU 2026.3.0 after pausing the exporter
on `tripod-worker`. The initial real 1B inference capture contains **7,979,008
executed instructions** and **34,592 ns** for one Q4_K one-column matvec kernel; its raw
report is [profile.ncu-repz](tmp/ncu-1b-20260912T173016Z-N8hkRy/profile/profile.ncu-repz).
NCU 2026.3 uses `.ncu-repz`; the wrapper initially misclassified that valid report
as missing. Earlier permission and resource-unavailable failures remain historical
evidence in [README_LOG.md](README_LOG.md).

The exporter is currently **paused**, with its original node label saved in
[state.json](tmp/dcgm-counter-access/state.json). For future captures, the temporary
helper prompts for sudo and saves that setting:

```bash
python3 /home/hys4qm/cloud-serving-benchmark/tmp/dcgm_counter_access.py pause
```

After it reports `PAUSED`, prepare the local installation (which selects NCU
2026.3.0), then run each check separately:

```bash
cd /home/hys4qm/cloud-serving-benchmark
source .run/cuda-recovery/env.sh
export NCU_BIN="$(command -v ncu)"
export RUN_TAG="ncu-$(date -u +%Y%m%dT%H%M%SZ)"
GPU_DEVICE=0 bash execute.sh doctor cuda
GPU_DEVICE=1 bash execute.sh doctor cuda
```

GPU0 is used for the 1B captures. Both checks must pass before a future two-GPU
70B collection. These commands start the server, send HTTP requests and invoke NCU:

```bash
GPU_DEVICE=0 bash execute.sh counters cuda 1b prefill
GPU_DEVICE=0 bash execute.sh counters cuda 1b decode
GPU_DEVICE=0 bash execute.sh counters cuda 1b reuse
```

The `prefill` selector samples matrix/dequantization kernels at eight clients;
`decode` selects one/eight-column matvec kernels; `reuse` selects eight-column
matvec with warm prefixes. These labels do not create separate phase timing
gates, and selected matvec launches can include final-prefill tail work. Results are under
`results/remote-cuda/$RUN_TAG/counters-1b/`, including `.ncu-rep` or `.ncu-repz`
files and `counter_summary.csv`.

The temporary 1B diagnostic matrix uses [run_1b_hardware_metrics.py](tmp/run_1b_hardware_metrics.py):

```bash
python3 tmp/run_1b_hardware_metrics.py --output results/cuda-study-1b/hardware-counters
```

It validates completed captures before resuming. Its [progress.json](results/cuda-study-1b/hardware-counters/progress.json)
records completion and its per-capture directories retain raw reports, request
metadata and counter summaries. All **23 captures completed and validated, with
166 kernel samples**. They cover mixed tensor
types in Q8_0/Q4_K_M/IQ1_M with 2,048 input and 128 output tokens, two launches
per matching launch configuration, cold replay caches and unchanged GPU clocks.
These are selected kernel diagnostics; replay timings are separate from serving
benchmarks. Hardware counters for 8B and 70B remain unmeasured.

Exporter restoration is pending and needs interactive sudo. Restore the saved
setting after profiling and verify the exporter returns:

```bash
python3 /home/hys4qm/cloud-serving-benchmark/tmp/dcgm_counter_access.py resume
python3 /home/hys4qm/cloud-serving-benchmark/tmp/dcgm_counter_access.py status
```

See
[NVIDIA's pause/resume reference](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgmi/dcgmi-profile.html)
and [exporter reference](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgm-exporter.html).

## Reproduce this CUDA study

The study compares Q8_0, Q4_K_M, and IQ1_M on one RTX A6000: Llama 3.2 1B
first, then Llama 3.1 8B. Both use the same fixed
[study configuration](config/cuda-study.env). Model revisions are pinned in
`models/manifest.json` and `models/manifest-8b.json`.

On this prepared machine:

```bash
cd /home/hys4qm/cloud-serving-benchmark
source .run/cuda-recovery/env.sh
python3 scripts/08_run_cuda_study.py --output-dir results/repeat-baseline-1b
python3 scripts/08_run_cuda_study.py --manifest models/manifest-8b.json --output-dir results/repeat-baseline-8b
```

The runner skips completed valid cells and refuses to mix changed code, models,
or configuration into an existing study. `--dry-run` prints the plan. For a fresh
repeat, use new output directories. It rotates model order, records GPU telemetry
and server metrics, and excludes a full eight-request warmup after each model
load. Each cell contains 16 requests, repeated three times. First-request
transition overhead and final request drain remain part of the measured workload.

The reported baseline data are in `results/cuda-study-{1b,8b}`. Their exact code
is saved in `results/baseline-code.tar.gz`; optional prefix controls were added
after those runs, so the commands above use fresh directories.

For the matched prefix-reuse comparison, run:

```bash
python3 scripts/08_run_cuda_study.py --prefix-study --output-dir results/repeat-prefix-1b
python3 scripts/08_run_cuda_study.py --prefix-study --manifest models/manifest-8b.json --output-dir results/repeat-prefix-8b
```

This compares reuse off/on at eight clients with the same repeated 2,048-token
prompt. Slots are primed before timing; reuse-on also gets a discarded cached
warmup. It measures the best case for a warm per-slot GPU KV prefix, with idle-slot
RAM caching disabled in both modes. The reported data are in
`results/prefix-study-{1b,8b}`. Per-request hashes and cached/evaluated token counts
verify that inputs match and reuse occurred. Throughput counts generated tokens.

Summarize either study with the existing analysis tool:

```bash
python3 benchmark/summarize.py results/cuda-study-1b --output-csv results/cuda-study-1b/summary.csv --output-markdown results/cuda-study-1b/summary.md
```

Profiling uses `scripts/07_profile_cuda.sh`; the matching commands and counter
mapping are in [the method notes](references/notes.md#cuda-profiling-for-the-serving-experiment).
Kernel tracing works here. Counter access was restored after enabling the driver
policy and pausing the DCGM exporter. Real 1B inference captures now contain
bandwidth, L1/L2 hit rates, achieved occupancy, IPC, tensor activity and stall
metrics; see [FINDINGS.md](FINDINGS.md) for the measured scope and interpretation.
The original denial remains historical evidence, and 8B/70B hardware counters
remain unmeasured.

`references/` holds the PDFs and notes; `models/` holds GGUFs and manifests;
`results/` holds the baseline and prefix studies and their profiles. `logs/download.log` combines
download diagnostics, while the build and study drivers have one log each.

The starter configuration uses Llama 3.2 1B Instruct Q4_K_M on one GPU:

- one fixed workload: approximately 2,048 prompt tokens and 128 output tokens;
- two request loads: one and eight concurrent clients;
- three repetitions per load;
- prompt reuse and idle-slot RAM caching disabled;
- the same server-side logical and physical token-batch sizes in every run.

The result includes request rate, aggregate token throughput, time to first token
(TTFT), time per output token (TPOT), and end-to-end latency at p50 and p95.

## Requirements

Run this project on an Ubuntu/Debian GPU cloud image with either ROCm or the CUDA
Toolkit already installed. The installation script adds ordinary build tools,
but deliberately does not install or replace the GPU toolkit because the correct
driver and toolkit versions depend on the provider's image.

With `ACCELERATOR_BACKEND="auto"`, the scripts choose ROCm when `hipconfig` is
available or CUDA when `nvcc` is available. If both are installed, set the
backend explicitly in `config/experiment.env`:

```bash
ACCELERATOR_BACKEND="rocm"
```

or:

```bash
ACCELERATOR_BACKEND="cuda"
```

The ROCm default targets MI300X (`gfx942`). CUDA defaults to CMake's locally
detected NVIDIA architecture; `CUDA_ARCHITECTURES` can be set when a portable
binary is required. ROCm and CUDA use separate build directories, so switching
does not reuse an incompatible CMake cache.

## Quick start

```bash
cd cloud-serving-benchmark
./scripts/00_install_prereqs.sh
./scripts/01_build_llama_cpp.sh
./scripts/02_download_model.sh
./scripts/03_start_server.sh
./scripts/04_smoke_test.sh
./scripts/05_run_experiment.sh
./scripts/06_stop_server.sh
```

Alternatively, the complete sequence—including stopping the server on exit—is:

```bash
./scripts/run_all.sh
```

The experiment writes a timestamped directory under `results/`. Read
`summary.md` for the compact results and use `summary.csv` for plots. Raw
per-request measurements, the exact configuration, the resolved llama.cpp
commit, model checksum, GPU-toolkit information, and server metrics are retained in
the same result directory.

## What “concurrency” means

`CONCURRENCY_LEVELS="1 8"` means one or eight simultaneous HTTP clients. This
is distinct from `SERVER_BATCH_SIZE`, which is llama.cpp's logical maximum
number of tokens processed in an internal batch. The server is kept at eight
slots for both loads, and only client concurrency changes.

The client is a closed-loop load generator: each client begins another request
when its preceding request completes. This is a controlled serving experiment,
not a full model of a production arrival distribution.

## Configuration

Edit `config/experiment.env`, or copy it and run with a different file:

```bash
CONFIG_FILE=/path/to/another.env ./scripts/03_start_server.sh
CONFIG_FILE=/path/to/another.env ./scripts/05_run_experiment.sh
```

Important controls include:

- `CONCURRENCY_LEVELS`: simultaneous request counts;
- `REQUESTS_PER_LEVEL`: total requests in each repetition;
- `TARGET_PROMPT_TOKENS` and `MAX_OUTPUT_TOKENS`: workload shape;
- `SERVER_PARALLEL`: number of server slots; it must be at least the largest
  tested concurrency;
- `SERVER_BATCH_SIZE` and `SERVER_UBATCH_SIZE`: internal token batching, held
  fixed across experiments;
- `LLAMA_CPP_REF`: branch, tag, or commit. Each run records the resolved commit;
- `ACCELERATOR_BACKEND`: `auto`, `rocm`, or `cuda`;
- `ROCM_GPU_TARGETS` and `CUDA_ARCHITECTURES`: optional architecture controls.

The benchmark uses llama-server's own `/apply-template` and `/tokenize`
endpoints before timing to prepare comparable prompt lengths. Streaming output
is requested so TTFT can be measured. `ignore_eos` is enabled so each request
performs the configured amount of decode work; this makes the workload
synthetic but controlled.

## Other models and backends

The original scripts below `scripts/00`–`06` still support individual CUDA or
ROCm runs through `config/experiment.env`. Change `MODEL_URL`, `MODEL_FILENAME`,
and `MODEL_ALIAS` together and keep serving/workload settings fixed when comparing
formats. The study runner reuses these scripts instead of adding per-model scripts.

The added Llama 3.3 70B study uses Q4_K_M and Q2_K on two RTX A6000s. It tests
concurrent serving of a larger model; different quantization recipes and hardware
still prevent direct replication of the poster's 70B MI300X comparison.

## Methodological scope

This setup fixes the three central limitations in the poster:

1. request concurrency is explicitly separated from internal token batching;
2. requests pass through an actual continuously batched HTTP serving path;
3. every request includes both prompt processing and autoregressive decoding.

These are fixed-concurrency experiments: one GPU for 1B/8B and two GPUs for 70B.
They do not represent a complete production or multi-tenant cloud deployment.

## Upstream references

- [llama.cpp HIP build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
- [llama.cpp CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda)
- [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama-server benchmark example](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/README.md)
