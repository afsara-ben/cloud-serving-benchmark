# Cloud Serving Benchmark

This project turns the poster's single-process prompt benchmark into a small,
request-level serving experiment. It starts an OpenAI-compatible `llama-server`
with continuous batching and sends real concurrent streaming requests. The same
project supports both NVIDIA CUDA and AMD ROCm GPUs.

The motivating poster is included at [paper/Tapia.pdf](paper/Tapia.pdf).

The starter configuration uses Llama 3.2 1B Instruct Q4_K_M on one GPU:

- one fixed workload: approximately 2,048 prompt tokens and 128 output tokens;
- two request loads: one and eight concurrent clients;
- three repetitions per load;
- prompt caching disabled to avoid cross-request cache reuse;
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

## Adding quantization variants

Start with the included Q4_K_M model. To compare formats later, create one
configuration file per model and change `EXPERIMENT_NAME`, `MODEL_URL`,
`MODEL_FILENAME`, and `MODEL_ALIAS`. Keep every server and workload setting
unchanged. Q8_0 and IQ1_M are the most useful next variants because they match
the poster's baseline and aggressive-compression endpoint.

Do not present the 1B measurements as evidence for the 70B model. Use them to
validate the server workflow and analysis scripts, then repeat the same small
matrix with the selected 70B variants.

## Methodological scope

This setup fixes the three central limitations in the poster:

1. request concurrency is explicitly separated from internal token batching;
2. requests pass through an actual continuously batched HTTP serving path;
3. every request includes both prompt processing and autoregressive decoding.

It remains a single-GPU, fixed-concurrency experiment. State that scope rather
than claiming a complete production or multi-tenant cloud deployment.

## Upstream references

- [llama.cpp HIP build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
- [llama.cpp CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda)
- [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama-server benchmark example](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/README.md)
