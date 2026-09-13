# A100 serving measurements

For a separate kernel/hardware-counter capture of one completed setting, use
[the A100 profiling commands](A100_PROFILING.md).

Use `scripts/run_a100_serving.py` on one or two A100 SXM 80 GB GPUs. It runs
Llama 1B, 8B, then 70B serially, and accepts individual model/format/client/input
selections. The full scope is 224 settings:

- 1B and 8B: IQ1_M, Q2_K, Q4_K_M, Q8_0, FP16.
- 70B: IQ1_M, Q2_K, Q4_K_M, Q8_0. No 70B FP16 preparation or inference.
- 8, 16, 32 and 64 concurrent clients; 2,048, 4,096, 8,192 and 16,384 input tokens.
- Exactly 512 output tokens per request, FP16 KV, prompt reuse off.
- One measured run per setting. C discarded warmup requests, then 2C measured
  requests: 8 + 16 at C8, 16 + 32 at C16, 32 + 64 at C32, and 64 + 128 at C64.

TTFT, TPOT, aggregate generated tokens/s and per-GPU peak memory are collected
together. No profiler, prefix experiment, capacity extension, or matrix
diagnostic is invoked. Capacity checks skip settings that cannot fit with the
required GPU headroom. Memory is device-total SMI sampling during measured HTTP
requests, in GiB; sampled peaks can miss short allocation spikes.

## Prepare the destination machine

Copy the repository source, including `benchmark/`, `scripts/`, `execute.sh`,
`config/`, `prompts/` and `models/*.json`, to the A100 host. Linux, Python 3.10+,
a working NVIDIA driver and CUDA toolkit (`nvcc` and `nvidia-smi`), Git, CMake,
Ninja, a C++ compiler, curl and flock are required. Nsight is not required.

The existing 70B Q2_K and Q4_K_M GGUFs have hashes but no recorded download
source. Copy those exact files to the destination checkout's `models/`:

```text
Llama-3.3-70B-Instruct-Q2_K.gguf
Llama-3.3-70B-Instruct-Q4_K_M.gguf
```

Other formats can be prepared from the pinned sources in the manifests.
Copying already prepared GGUFs into `models/` avoids downloading them again.
FP16 conversion uses original Meta checkpoints; it requires access to those
repositories via `HF_TOKEN` or a private `~/.hf_token` file. Preparation runs
on CPU and does not execute inference.

From the destination checkout, create/activate a Python environment and build
the pinned CUDA server for SM80. Use `--gpus 0` throughout for one GPU, or
`--gpus 0,1` for two:

```bash
python3 -m venv .venv-a100-serving
source .venv-a100-serving/bin/activate
python3 scripts/run_a100_serving.py build --gpus 0,1
python3 -m pip install -r vendor/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python3 -m pip install huggingface_hub matplotlib
python3 scripts/run_a100_serving.py prepare --gpus 0,1
```

`prepare --models 1b` prepares only that model size. Preparation always includes
that size's complete format set so that later cell selections have a stable
resume identity. Separate `a100-serving-manifest-*.json` files preserve the
original study manifests. Prepare each model size once before its first run.
Build failures and missing dependencies stop before inference.

## Run selected settings

The default action is `plan`; it prints the selected settings and request budget
without touching GPUs, downloading models or writing results.

```bash
# Preview exactly one setting:
python3 scripts/run_a100_serving.py plan --gpus 0,1 \
  --models 8b --formats Q4_K_M --concurrencies 64 --prompt-lengths 4096

# Run exactly that setting:
python3 scripts/run_a100_serving.py run --gpus 0,1 \
  --models 8b --formats Q4_K_M --concurrencies 64 --prompt-lengths 4096

# Run the new client counts for one model, format and input length:
python3 scripts/run_a100_serving.py run --gpus 0,1 \
  --models 8b --formats Q4_K_M --concurrencies 16,32 --prompt-lengths 4096

# Run one model and format at all four client counts and input lengths:
python3 scripts/run_a100_serving.py run --gpus 0,1 --models 70b --formats Q2_K

# Run only one model, with all of its formats and settings:
python3 scripts/run_a100_serving.py run --gpus 0,1 --models 1b

# Run the entire 224-setting scope, serially in model order 1B, 8B, 70B:
python3 scripts/run_a100_serving.py run --gpus 0,1
```

`--models` accepts space-separated sizes. `--formats`, `--concurrencies`, and
`--prompt-lengths` accept comma-separated values; omitted selectors use the
full applicable set. Selecting multiple values runs their Cartesian product.
Run commands serially on otherwise idle GPUs.

Results accumulate in `results/cuda-context-study-a100-1gpu/` or
`results/cuda-context-study-a100-2gpu/`, with sibling directories per model.
Individual-cell execution uses `--only-cell` in the underlying runner, retaining
the full model scope in its fingerprint. Completed valid runs are reused when
later commands overlap. A genuinely independent repeat or a hardware/code/model
change requires a fresh `--study-name`, used consistently for `run` and `report`.
Use the updated code in a fresh destination series; old series created before
the cell-selector change retain their old code fingerprint.
Existing series limited to 8/64 clients also require a fresh `--study-name`
for the expanded 8/16/32/64 grid; their scope and run fingerprints differ.

## Read the results

Every run refreshes `serving-report/index.md`, `runtime.csv`, `capacity.csv`,
`completion-audit.json`, and PNG/SVG plots for the available model sizes.
The audit tracks these 224 settings only and requires no profiling. Partial
selections remain visibly incomplete until the rest are measured or excluded.

```bash
# Refresh reports from existing files, with no inference:
python3 scripts/run_a100_serving.py report --gpus 0,1
```

Add `--no-plots` to `run` or `report` for CSV/Markdown output only. GPU memory
columns use physical indices, e.g. `gpu0_peak_vram_gib`. Latency columns are
`ttft_p50_ms`, `ttft_p95_ms`, `tpot_p50_ms`, `tpot_p95_ms`; generated throughput is
`output_tokens_per_second_mean`. Individual request records are in each
`<format>/c<clients>/p<input tokens>/r1/raw.json`.
