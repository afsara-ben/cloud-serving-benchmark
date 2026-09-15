# Llama 70B figures on two RTX PRO 6000 96 GB GPUs

The RTX workflow runs the experiments for the four-panel Runtime Cost poster
and Runtime Bottlenecks report, including 32K inputs. It writes new figures
under the selected results series. The existing A6000 PDFs remain historical
artifacts.

## Executable workflow

All commands below are also packaged in
[`scripts/rtxpro6000_pipeline.sh`](scripts/rtxpro6000_pipeline.sh), which is
executable and stops at the first failing command. Sync the updated checkout
to the remote host before running it.

### Two-server split (recommended reduced workflow)

Use separate study roots so the two servers never update the same manifest or
progress file. Both launchers keep one serving repetition (`r1`), run the same
C8/C16/C32 and 2K/4K/8K/16K/32K grid for their two formats, and profile C8/2K
with eight full-section captures. Each capture samples five gate/up launches.
The report scales the per-launch medians over the 41/39 transformer-layer
placement and reuses those captures for the sampled gate/up scalar roofline.
There are no 60-capture scalar-operator campaigns and no 32K profile campaigns.

On **`l2x01`** (IQ1_M and Q8_0):

```bash
cd /scratch/cloud-serving-benchmark
./scripts/rtxpro6000_server_iq1_q8.sh all
```

On **`l2x02`** (Q2_K and Q4_K_M):

```bash
cd /scratch/cloud-serving-benchmark
./scripts/rtxpro6000_server_q2_q4.sh all
```

If the GGUFs are outside the checkout, append `--model-dir /absolute/nvme/path`
to the applicable command. `l2x01` selects only `IQ1_M,Q8_0`; `l2x02` selects
only `Q4_K_M,Q2_K`. Setup prepares only that pair. The study roots are:

```text
results/cuda-context-study-rtxpro6000-iq1-q8-r1/
results/cuda-context-study-rtxpro6000-q2-q4-r1/
```

Each run writes small, commit-ready files separately by quantization and
profile type:

```text
results/rtxpro6000-exports/cuda-context-study-rtxpro6000-iq1-q8-r1/
  IQ1_M-serving-r1.json
  IQ1_M-profile-full-sections-p2048.json
  IQ1_M-profile-scalar-roofline-p2048.json
  Q8_0-...
results/rtxpro6000-exports/cuda-context-study-rtxpro6000-q2-q4-r1/
  Q4_K_M-...
  Q2_K-...
  Q4_K_M-Q2_K-runtime-bottlenecks-p2048.pdf
```

These export directories are intentionally not ignored by Git. Raw serving and
Nsight directories remain ignored because they can be large. Commit an export
on each server with:

```bash
git add results/rtxpro6000-exports/
git commit -m "Add RTX PRO 6000 benchmark shard"
git push
```

After both export directories are present in one checkout, build the complete
60-cell poster without the raw server trees:

```bash
python3 paper/poster/build.py --serving-exports \
  results/rtxpro6000-exports/cuda-context-study-rtxpro6000-iq1-q8-r1 \
  results/rtxpro6000-exports/cuda-context-study-rtxpro6000-q2-q4-r1 \
  --output results/rtxpro6000-exports/combined-poster
```

Copy each server's selected GGUF pair into its checkout's `models/` directory,
or keep the files elsewhere on that server's NVMe and pass the absolute
directory with `--model-dir`. `/scratch` is backed by the root NVMe filesystem
on `l2x02`, so `/scratch/cloud-serving-benchmark/models` is already NVMe-backed.
The wrapper creates checkout-local symlinks when `--model-dir` points elsewhere;
it does not duplicate the weight files.

Then **on the remote GPU server**, run:

```bash
cd /scratch/cloud-serving-benchmark
./scripts/rtxpro6000_pipeline.sh all --dry-run  # Preview; no changes or GPU work.
./scripts/rtxpro6000_pipeline.sh all            # Execute the complete workflow.
```

For the shortest single-server workflow that still creates the complete Runtime
Cost poster with 32K serving data and a reduced-scope 2K bottleneck report, use:

```bash
./scripts/rtxpro6000_pipeline.sh all --fast
```

`--fast` omits the optional 32K profiling campaign and all 60 targeted scalar
operator captures. The eight full-section captures retain five gate/up launches
and supply the counters used by every reduced report figure. The builder clearly
labels layer-extrapolated values and the limited gate/up operator scope. The
serving protocol stays at `r1` and its 60-cell poster grid is unchanged.

`all` checks the local Q2/Q4 files, creates/activates `.venv-rtxpro6000`, installs
dependencies including Ninja and CMake, builds the pinned SM120 server, prepares
models, runs the serving pilots/sweep/poster, locates or installs full Nsight,
probes counters, builds the annotated server, and runs the profiling
pilots/campaigns/reports at 2K and 32K. GPU runs are serial. If Q2/Q4 are already
elsewhere on the remote filesystem, pass `--model-dir /path/to/GGUFs`; setup
links them into `models/` and preparation verifies their hashes.

Individual actions avoid repeating earlier stages:

```bash
./scripts/rtxpro6000_pipeline.sh setup
./scripts/rtxpro6000_pipeline.sh serving
./scripts/rtxpro6000_pipeline.sh nsight
./scripts/rtxpro6000_pipeline.sh profile
./scripts/rtxpro6000_pipeline.sh report
```

Use `--study-name YOUR_SERIES` consistently for independent experiments; the
default is `cuda-context-study-rtxpro6000-r1` (or the exported `STUDY_NAME`).
Use `--gpus 2,3` for different physical indices. Export `NCU_BIN` to select an
existing profiler, and `CUDA_HOME` if the compiler requires a site-specific
toolkit path. Drivers, CUDA itself and system counter permissions remain host
prerequisites. Run `./scripts/rtxpro6000_pipeline.sh --help` for all options.

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
| Serving sampling | C discarded warmup requests, then 2C measured requests inside one repetition (`r1`) |
| Placement | Two GPUs, layer split 1:1, all layers/output on GPUs; FP16 KV; flash attention |
| Bottleneck comparisons | Q2_K vs Q4_K_M, C8, separately at 2K and 32K |

There are **60 serving cells**. A complete grid requires a validated measurement
or a recorded capacity exclusion for each cell. The same two-wave request
protocol applies to 32K. Here, 32K means 32,768 **input** tokens, plus 512 output
tokens; it is not a 32,768-token total slot limit. No 70B FP16 or 64K run is
included.

The complete bottleneck workload needs **68 serial captures**: eight full-section
gate/up captures (two formats × two phases × two GPUs) and 60 targeted scalar
operator captures. Each capture has eight discarded warmup requests and eight
profiled requests, with a cap of five matching launches. The two contexts
therefore require 136 captures. Nsight replay makes these substantially more
expensive than 136 ordinary inference runs. Timings under the profiler are
separate from the serving measurements.

The recommended two-server launchers use the reduced image scope: **8 captures
per server**, at 2K only. The servers run concurrently, and each processes 30
serving cells. The five sampled gate/up launches per capture are not independent
experiment repetitions; the serving data remains `r1`.

## 1. Prepare the remote checkout and tools

Copy the updated source, including the new scripts and benchmark modules, to
the remote checkout. Required directories are `benchmark/`, `scripts/`,
`config/`, `prompts/`, `paper/`, and `models/*.json`, plus `execute.sh`.
Do not copy local compiled binaries as the remote build. The build command
clones the pinned llama.cpp source if `vendor/llama.cpp` is absent.

The host needs Linux, Python 3.10+, Git, CMake, Ninja, a C++ compiler, curl,
flock, a working NVIDIA driver, and a CUDA toolkit that supports SM120
(CUDA 12.8 or newer). The requirements file installs CMake and Ninja inside the
Python environment, without sudo. Load site compiler/CUDA modules if `c++` or
`nvcc` is not on `PATH`. Install the **full Nsight Compute package**, including
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
checkout path if necessary (for example `/scratch/cloud-serving-benchmark`).
Ensure the desired `nvcc` is on `PATH`; Nsight is configured in step 3.

```bash
cd ~/cloud-serving-benchmark
set -euo pipefail
python3 -m venv .venv-rtxpro6000
source .venv-rtxpro6000/bin/activate
python3 -m pip install -r scripts/requirements-rtxpro6000.txt

command -v git cmake ninja c++ nvcc curl flock
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export STUDY_NAME=cuda-context-study-rtxpro6000-r1
study_root="results/$STUDY_NAME"

nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap --format=csv
nvcc --version
# Unmodified serving server, pinned revision, compiled for SM120.
python3 scripts/run_rtxpro6000.py build --gpus 0,1 &&
python3 -m pip install -e vendor/llama.cpp/gguf-py &&
python3 scripts/run_rtxpro6000.py prepare --gpus 0,1
```

Preparation creates `models/rtxpro6000-serving-manifest-70b.json` with portable
model paths. All four formats explicitly use the same Llama 3 role template,
`config/templates/llama-3.1-benchmark.jinja`; the template is compatible with
Llama 3.3 and its hash is recorded. Input hashes are checked across formats.
The original study manifest is preserved.

If `ninja` was missing in an earlier setup, rerun the requirements installation.
The vendor checkout is created by `build`, after its prerequisite checks. An
editable `gguf-py` installation attempted after a failed build will fail because
the project directory has not been created; retry it only after a successful
build. The `&&` chain above prevents those dependent steps from running after
a failure, even if the shell does not have `set -e` enabled.

The exact Q2 and Q4 originals exist in `/data/home/hys4qm/` on the source host.
For the `/scratch` remote checkout, first run `mkdir -p models` in that checkout.
Then, **from the original source host** (where `l2x02` is SSH-reachable):

```bash
rsync -avP \
  /data/home/hys4qm/Llama-3.3-70B-Instruct-Q2_K.gguf \
  /data/home/hys4qm/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
  hys4qm@l2x02:/scratch/cloud-serving-benchmark/models/
```

If the compute node is not directly reachable, use the cluster transfer/login
host that provides access to the same destination filesystem. If these files
are already on the remote host, copy or link them into `models/` instead.

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

Nsight is needed for this stage only. If a full installation already exists,
load the site's Nsight Compute module or set `NCU_BIN` to its executable, e.g.
`/opt/nvidia/nsight-compute/VERSION/ncu`. Check common install locations with:

```bash
find /opt/nvidia /usr/local "$HOME/.local/opt" -name ncu -type f 2>/dev/null || true
```

If no installation is available, NVIDIA provides a full
[Linux x86_64 redistribution archive](https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/).
On an x86_64 server, unpack it into your account without sudo:

```bash
(
  set -euo pipefail
  test "$(uname -m)" = x86_64
  csb_ncu_archive=nsight_compute-linux-x86_64-2026.3.0.13-archive
  csb_ncu_prefix="$HOME/.local/opt/nvidia"
  mkdir -p "$csb_ncu_prefix"
  curl -fL --retry 3 -C - \
    "https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/$csb_ncu_archive.tar.xz" \
    -o "$csb_ncu_prefix/$csb_ncu_archive.tar.xz"
  tar -xJf "$csb_ncu_prefix/$csb_ncu_archive.tar.xz" -C "$csb_ncu_prefix"
  "$csb_ncu_prefix/$csb_ncu_archive/ncu" --version
)
export NCU_BIN="$HOME/.local/opt/nvidia/nsight_compute-linux-x86_64-2026.3.0.13-archive/ncu"
```

The archive is about 380 MiB compressed and includes the Python Report
Interface. This installs the profiler in your account; it does not update the
driver or change counter permissions. If the following counter probe reports a
driver or permission failure, resolve that reported problem before profiling.

Run after the serving cells have completed. Both selected formats at C8 and the
selected context must pass independent serving validation before even the
single-capture pilot starts. The default pair is Q4_K_M/Q2_K. Captures reuse the saved physical GPU selection;
both GPUs stay visible, with one logical GPU's counters selected at a time.

```bash
# Tiny real counter probe on each GPU; no Llama model is loaded.
python3 scripts/profile_rtxpro6000.py preflight --gpus 0,1

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

### Only Q2_K and Q4_K_M are a comparable pair

`bottleneck_data` requires the two compared formats to sample the same gate/up
kernel families in **both** phases, and rejects any other pairing with
`Compared formats sampled different gate/up kernel configurations`. Measured
2K/C8 captures on two RTX PRO 6000 Blackwell Server Edition GPUs
(September 15, 2026) give:

| Format | Prefill | Decode |
|---|---|---|
| Q2_K | `mul_mat_q` | `mul_mat_q` + `mul_mat_q_stream_k_fixup` |
| Q4_K_M | `mul_mat_q` | `mul_mat_q` + `mul_mat_q_stream_k_fixup` |
| Q8_0 | `mul_mat_q` | `mul_mat_vec_q` |
| IQ1_M | `cutlass::Kernel2` | `mul_mat_vec_q` |

Of the six possible pairs only **Q2_K + Q4_K_M** matches in both phases, so the
`Q2_K vs Q4_K_M` scope above is the only bottleneck comparison these four
formats support. Each remaining format is isolated by a different phase: IQ1_M
alone dispatches CUTLASS for prefill, and Q8_0 is the only `mul_mat_q` prefill
format that decodes through `mul_mat_vec_q` instead of the stream-k matmul.
Re-pairing the two servers therefore cannot yield a second comparable pair; it
is not a sharding choice but a dispatch property of the quantization formats.

Consequences for the two-server split:

- The `q2-q4` shard is the Runtime Bottlenecks report. Build it on that server.
- The `iq1-q8` shard contributes serving cells and per-format captures only. Its
  `run` and `export` stages fail at the pairing check, so it produces no
  `IQ1_M-Q8_0-runtime-bottlenecks-*.pdf`. An export predating this check may
  still contain one; that PDF is not reproducible and should be discarded.
- IQ1_M and Q8_0 do match in decode (`mul_mat_vec_q`). A decode-only comparison
  would be valid but is not implemented; the check is all-or-nothing per pair.

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
