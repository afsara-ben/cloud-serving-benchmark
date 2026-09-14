# Llama 70B figures on two RTX PRO 6000 96 GB GPUs

The RTX workflow runs the experiments for the four-panel Runtime Cost poster
and Runtime Bottlenecks report, including 32K inputs. It writes new figures
under the selected results series. The existing A6000 PDFs remain historical
artifacts.

The scripts have CPU validation and synthetic PDF rendering coverage. Actual
SM120 compilation, model capacity and Nsight captures must pass the pilots on
the destination machine before a full campaign is warranted.

## Experiment scope

| Measurement | Requested settings |
|---|---|
| Unprofiled serving | Llama 3.3 70B; IQ1_M, Q2_K, Q4_K_M, Q8_0 |
| Concurrent clients | 8, 16, 32 |
| Input tokens | 2,048; 4,096; 8,192; 16,384; 32,768 |
| Output tokens | Exactly 512 per request |
| Serving sampling | C discarded warmup requests, then 2C measured requests; one repetition |
| Placement | Two GPUs, layer split 1:1, all layers/output on GPUs; FP16 KV; flash attention |
| Bottleneck comparisons | Q2_K vs Q4_K_M, C8, separately at 2K and 32K |

There are **60 serving cells**. A complete grid requires a validated measurement
or a recorded capacity exclusion for each cell. The same two-wave request
protocol applies to 32K. Here, 32K means 32,768 **input** tokens, plus 512 output
tokens; it is not a 32,768-token total slot limit. No 70B FP16 or 64K run is
included.

Each bottleneck workload needs **68 serial captures**: eight full-section
gate/up captures (two formats × two phases × two GPUs) and 60 targeted scalar
operator captures. Each capture has eight discarded warmup requests and eight
profiled requests, with a cap of five matching launches. The two contexts
therefore require 136 captures. Nsight replay makes these substantially more
expensive than 136 ordinary inference runs. Timings under the profiler are
separate from the serving measurements.

## 1. Prepare the remote checkout and tools

Copy the updated source, including the new scripts and benchmark modules, to
the remote checkout. Required directories are `benchmark/`, `scripts/`,
`config/`, `prompts/`, `paper/`, and `models/*.json`, plus `execute.sh`.
Do not copy local compiled binaries as the remote build. The build command
clones the pinned llama.cpp source if `vendor/llama.cpp` is absent.

The host needs Linux, Python 3.10+, Git, CMake, Ninja, a C++ compiler, curl,
flock, a working NVIDIA driver, and a CUDA toolkit that supports SM120
(CUDA 12.8 or newer). Install the **full Nsight Compute package**, including
its `extras/python/ncu_report*` interface, and NVTX3 headers. The counter probe
below tests the installed driver/profiler combination and counter permissions.
Nsight Systems is only needed for an optional trace, not this campaign.

Copy the exact existing Q2 and Q4 GGUFs into the remote checkout's `models/`:

```text
Llama-3.3-70B-Instruct-Q2_K.gguf
Llama-3.3-70B-Instruct-Q4_K_M.gguf
```

Their download sources were not recorded; `prepare` cannot fetch replacements.
It checks their SHA256 hashes against `models/study-manifest-70b.json`. IQ1_M
and Q8_0 can be downloaded from the pinned sources. All four weights total
about 150 GiB; Q8 preparation also retains about 70 GiB of download parts.
Allow additional space for builds and raw profiler reports (300 GB free is a
useful starting point, not a bound on the eventual capture size).

Run the following in one remote shell, on an otherwise idle GPU allocation.
Use `tmux` or an equivalent persistent session for the long stages. Change the
checkout path if necessary. Ensure the desired `nvcc` and `ncu` are on `PATH`.

```bash
cd ~/cloud-serving-benchmark
set -euo pipefail
python3 -m venv .venv-rtxpro6000
source .venv-rtxpro6000/bin/activate
python3 -m pip install -r scripts/requirements-rtxpro6000.txt

command -v nvcc ncu
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export NCU_BIN="$(readlink -f "$(command -v ncu)")"
export STUDY_NAME=cuda-context-study-rtxpro6000-r1
study_root="results/$STUDY_NAME"

nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap --format=csv
nvcc --version
"$NCU_BIN" --version

# Tiny real counter probe on each GPU; no Llama model is loaded.
python3 scripts/profile_rtxpro6000.py preflight --gpus 0,1

# Unmodified serving server, pinned revision, compiled for SM120.
python3 scripts/run_rtxpro6000.py build --gpus 0,1
python3 -m pip install -e vendor/llama.cpp/gguf-py
python3 scripts/run_rtxpro6000.py prepare --gpus 0,1
```

Preparation creates `models/rtxpro6000-serving-manifest-70b.json` with portable
model paths. All four formats explicitly use the same Llama 3 role template,
`config/templates/llama-3.1-benchmark.jinja`; the template is compatible with
Llama 3.3 and its hash is recorded. Input hashes are checked across formats.
The original study manifest is preserved.

If the counter probe fails, resolve its reported installation or driver
permission problem before profiling. Unprofiled serving can be run independently
of Nsight. The scripts do not change system counter permissions.

## 2. Serving pilots, sweep and Runtime Cost poster

```bash
# Read-only: prints the complete 60-cell grid and request budget.
python3 scripts/run_rtxpro6000.py plan --study-name "$STUDY_NAME" --gpus 0,1

# Small serving pilot, then an early 32K capacity/performance check.
python3 scripts/run_rtxpro6000.py run --study-name "$STUDY_NAME" --gpus 0,1 \
  --formats Q4_K_M --concurrencies 8 --prompt-lengths 2048
python3 scripts/run_rtxpro6000.py run --study-name "$STUDY_NAME" --gpus 0,1 \
  --formats Q4_K_M --concurrencies 8 --prompt-lengths 32768

# Full sweep; reuses completed pilot cells with the same experiment identity.
python3 scripts/run_rtxpro6000.py run --study-name "$STUDY_NAME" --gpus 0,1

# Revalidate saved data, then render the four-panel poster and individual figures.
python3 scripts/run_rtxpro6000.py report --study-name "$STUDY_NAME" --gpus 0,1
python3 paper/poster/build.py --study-root "$study_root"
```

The main outputs are:

```text
results/$STUDY_NAME/serving-report/index.md
results/$STUDY_NAME/serving-report/runtime.csv
results/$STUDY_NAME/serving-report/capacity.csv
results/$STUDY_NAME/serving-report/completion-audit.json
results/$STUDY_NAME/paper/poster/runtime-cost-poster.pdf
results/$STUDY_NAME/paper/poster/figures/              # PDF, PNG, SVG
```

`--formats`, `--concurrencies` and `--prompt-lengths` accept comma-separated
subsets. The full scope stays fixed so a pilot and subsequent sweep can share
one series. Always pass the same `--study-name`; exporting `STUDY_NAME` alone
does not select the launcher argument. For an explicitly incomplete poster,
add `--allow-missing` to the poster build command.

### Expected 32K capacity

These are **lower bounds**, not measured capacity: tensor placement plus FP16
KV for padded slots, plus the required 2 GiB headroom per GPU. Actual CUDA,
compute and graph allocations can make a setting infeasible.

| Format | C8 / 32K GiB, GPU0 / GPU1 | C16 / 32K GiB, GPU0 / GPU1 |
|---|---:|---:|
| IQ1_M | 51.56 / 49.59 | 93.52 / 89.50 |
| Q2_K | 55.97 / 54.14 | 97.93 / 94.05 |
| Q4_K_M | 63.53 / 61.39 | 105.49 / 101.30 |
| Q8_0 | 78.68 / 75.98 | 120.64 / 115.89 |

C8/32K is a reasonable target for all four formats. IQ1_M C16/32K is tight;
the other C16/32K settings and every C32/32K setting exceed the memory budget
already at this lower bound. The runner performs capacity screening and
retains exclusions in the audit. It also requires exact token counts, attained
concurrency, full GPU placement, zero server swap and measured GPU headroom.
Neither an exclusion nor an unfinished setting becomes a zero-valued plot point.

## 3. Diagnostic build and Runtime Bottlenecks captures

Run after the serving cells have completed. Both Q4_K_M and Q2_K at C8 and the
selected context must pass independent serving validation before even the
single-capture pilot starts. Captures reuse the saved physical GPU selection;
both GPUs stay visible, with one logical GPU's counters selected at a time.

```bash
# Separate annotated server and libraries; the serving build remains the source
# of the unprofiled timing measurements.
python3 benchmark/matrix_diagnostic.py build --server --cuda-arch 120 \
  --build-root .run/rtxpro6000-profile-build

# 2K baseline: preview, one full-section pilot, then all 68 captures + report.
python3 scripts/profile_rtxpro6000.py plan --study-root "$study_root" --prompt-tokens 2048
python3 scripts/profile_rtxpro6000.py pilot --study-root "$study_root" --prompt-tokens 2048
python3 scripts/profile_rtxpro6000.py run --study-root "$study_root" --prompt-tokens 2048

# 32K comparison: independent captures and report, after the baseline succeeds.
python3 scripts/profile_rtxpro6000.py plan --study-root "$study_root" --prompt-tokens 32768
python3 scripts/profile_rtxpro6000.py pilot --study-root "$study_root" --prompt-tokens 32768
python3 scripts/profile_rtxpro6000.py run --study-root "$study_root" --prompt-tokens 32768
```

If NVTX headers are outside the detected CUDA/system directories, append
`--nvtx-include /path/to/include` to the diagnostic build; that directory must
contain `nvtx3/nvToolsExt.h`.

`pilot` collects one Q4 prefill/GPU0 full-section capture; `run` reuses it and
continues the campaign. Each full comparison requires five matched gate/up
launches at M=28672, K=8192, N=512 for prefill or N=8 for decode. Kernel names,
NVTX attribution, opcode inventories and GPU identities are validated from
the new captures. CUDA graphs are disabled for attribution. No A6000 kernel
name or performance conclusion is imposed on Blackwell.

The report reads Blackwell Tensor Core arithmetic paths from the exported
metrics. Scalar FP32 roofs use **per-GPU** rated values for the saved edition:
[Server](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
120 TFLOP/s / 1597 GB/s,
[Workstation](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
125 / 1792, or
[Max-Q Workstation](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/)
110 / 1792 (specifications checked September 14, 2026). Unrecognized editions
stop the scalar roofline stage rather than receiving another GPU's ceilings.

```text
results/$STUDY_NAME/profiles/c8-p2048/                 # raw 2K captures + plans
results/$STUDY_NAME/profiles/c8-p32768/                # raw 32K captures + plans
results/$STUDY_NAME/paper/runtime-bottlenecks-c8-p2048/runtime-bottlenecks-report.pdf
results/$STUDY_NAME/paper/runtime-bottlenecks-c8-p32768/runtime-bottlenecks-report.pdf
```

Each report directory also contains five standalone figures in PDF/PNG/SVG,
Markdown, metrics/roofline CSVs, and JSON evidence with source hashes. The PDF
describes the actual measured changes without assuming Q2 is slower than Q4.

## 4. Rebuild figures or resume

These commands only read saved measurements and regenerate reports:

```bash
python3 paper/poster/build.py --study-root "$study_root"
python3 paper/runtime-bottlenecks-report/build.py --study-root "$study_root" \
  --captures "$study_root/profiles/c8-p2048"
python3 paper/runtime-bottlenecks-report/build.py --study-root "$study_root" \
  --captures "$study_root/profiles/c8-p32768"
```

The existing paper builders retain their historical behavior when called
without `--study-root`; always supply it for the RTX series. Complete report
builds reject missing required captures or mismatched evidence. No GPU work is
started to fill gaps during figure generation.

Successful cells/captures can be resumed using the same command and unchanged
identity. Code, models, hardware or protocol changes require a fresh serving
`--study-name`. Failed profile captures are preserved and are not retried
automatically: diagnose the failure, then choose a fresh capture directory
with `--output "$study_root/profiles/c8-p2048-retry1"` (or the 32K equivalent),
using that same output for both pilot and run. Supply that directory to the
report builder's `--captures` argument. The usual PDF output path remains
associated with the selected study and context.
