# Experiment and decision log

Read this file first when resuming. It records actions, evidence, decisions and
remaining work.

## Objective and authorized scope

**Datacenter GPUs: A Quantization perspective.** Test which observations from
the on-device paper and MI300X articles transfer to concurrent CUDA HTTP serving.
Read the supplied PDF, collect available low-level measurements, and produce one
concise report, `FINDINGS.md`, with reproducible evidence.

The user requested Llama 1B first, then authorized Llama 8B on the same single GPU.
They also requested matched comparisons with and without prefix reuse. Multi-GPU
experiments are now explicitly authorized for Llama 70B. Reuse the existing scripts
and keep artifacts organized.

## Current state — 2026-09-12 UTC, counters blocked awaiting terminal action

- **Incomplete goal:** release the occupied profiling resource, collect actual IPC,
  global-memory access and bandwidth counters, and explain the inference results.
  The user will run needed sudo commands. A guarded node-only pause/resume helper
  is ready at `tmp/dcgm_counter_access.py`; an asynchronous terminal request was
  sent. Read `tmp/dcgm-counter-access/state.json` when it exists and verify the
  exporter process exits before retrying. No pause has yet been observed.
  The same sudo blocker has persisted across three goal turns; the goal is
  blocked until the user runs the terminal command below. No job is running.

- **1B, 8B and two-GPU 70B studies are complete:** 1,536 retained requests and
  30 valid Nsight Systems traces. [FINDINGS.md](FINDINGS.md) contains the report.
  All managed GPU jobs ended; both GPUs returned to 0% utilization.
- **70B baseline:** 12 cells / 192 requests passed. Q4_K_M c1/c8 means are
  12.03/27.23 output tok/s; Q2_K means are 13.24/22.38. Three repetitions.
  Canonical summaries: `results/cuda-study-70b/summary.{csv,md}`.
- **70B prefix:** another 12 cells / 192 requests passed. Q4 off/on means are
  27.32/57.14 tok/s; Q2 means are 22.39/49.90. Each cache hit reused 2,047
  tokens and evaluated one. Results: `results/prefix-study-70b/`.
- **70B low-level analysis:** six traces passed all 132 audit checks. Q2's
  matrix-vector advantage at c1 reverses at c8 and persists with prefix reuse.
  Both cached traces show sequential layer execution across the two GPUs.
  Request, telemetry, copy and kernel evidence is in `results/70b-audit.json`.
- Hardware execution counters remain unavailable. Driver 595.91.07 works and
  `RmProfilingAdminOnly: 0` is confirmed, but NCU reports a busy resource on both
  GPUs. DCGM is active; ownership was not directly established. No monitoring,
  permissions or host memory settings were changed. Kernel tracing/NVML work.
- GPU 0 experienced thermal clock limiting during the broader 70B study windows;
  cell-specific effects are unresolved. Treat results as this host's observations.
- Portable `execute.sh` supports separate baseline/prefix/trace/counter commands
  for CUDA/ROCm, plus two-GPU CUDA 70B. A100/MI300X runs are not claimed. Its
  48-request CUDA workflow check is separate from the scientific comparisons.
- Original cache-confounded runs in `results/preliminary-cache-enabled/` and
  70B startup failures in `results/diagnostics/` are excluded. References remain
  in `references/`; study/download logs are consolidated. Read E18–E20 below
  before reproducing 70B; do not restart completed canonical studies.

## Remote handoff

- Copy the repository (including `models/*.json`) to the GPU server. Toolkits and
  profiling tools must be installed there; prerequisites are in `README.md`.
- Begin with `bash execute.sh setup cuda 1b`, then `bash execute.sh doctor cuda`.
  Use `rocm` for MI300X. Follow the separate baseline/prefix/trace/counter lines
  in `execute.sh`, then repeat with `8b`. For two-GPU CUDA 70B use the README's
  70B commands and verified local files from `models/manifest-70b.json`.
- Counter access is never assumed from a driver flag: a real GPU probe must
  succeed, and AMD selected-region gating must exclude both outside launches.
  MI300X gated counters require ROCm 7.14+ / SDK 1.3.2+; counter availability and
  group compatibility are checked before loading the model. Host permission or
  monitoring changes require the administrator; none are automated.
- Remote results use `results/remote-BACKEND/HOSTNAME`; use `RUN_TAG=repeat2` for
  repeated profiles or changed code. Per-backend/size study logs append; all
  download output remains in `logs/download.log`.

## Fixed controls and interpretation

- RTX A6000, 48 GB, Ampere workstation GPUs with GDDR6: GPU 0 for 1B/8B and
  GPUs 0,1 for 70B. Describe these as cloud-style HTTP serving measurements,
  not MI300X hardware replication. The two GPUs have NV4 topology.
- llama.cpp commit: `3f5e94d7c2ab2267fe39852051777fe30c1f49ef`; project-local
  CUDA 13.0 Release build, architecture 86, in `vendor/llama.cpp/build-cuda`.
- Llama 3.2 1B Instruct and Llama 3.1 8B Instruct, each Q8_0/Q4_K_M/IQ1_M.
  Files, publisher revisions and URLs: `models/manifest.json` and `manifest-8b.json`.
  Study manifests retain exact model/binary/source hashes and resolved settings.
  Llama 3.3 70B Instruct Q4_K_M/Q2_K uses `models/manifest-70b.json`; local
  file hashes are pinned, but upstream converter revisions are unknown.
- Each measured request has 2,048 logical prompt tokens and exactly 128 outputs;
  16 requests per cell, three repetitions. Eight slots, context 32,768, logical
  batch 2,048, physical microbatch 512, FlashAttention on, FP16 K/V, full offload.
- Baselines use concurrency 1 and 8, no prompt reuse, and a discarded full c8
  warmup after loading. Model order rotates; concurrency order alternates by repetition.
- Both studies disable idle-slot RAM serialization: `--cache-ram 0
  --no-cache-idle-slots`. Request `cache_prompt` selects GPU-slot prefix reuse.
- Serving measurements are unprofiled. Matched traces are separate because tracing
  and replay affect execution. Client overlap alone does not prove GPU batching.
- Quantization labels describe mixed tensor recipes, not uniform 8/4/1-bit weights.
  The sizes also differ in model generation; this is not isolated parameter scaling.

## Resume and reproduction

First inspect active work and each study's `progress.json`. This study is complete;
do not restart it merely because this log is read. Preserve unrelated jobs and
avoid overlapping any future GPU performance/profiling experiments.

```bash
cd /home/hys4qm/cloud-serving-benchmark
source .run/cuda-recovery/env.sh
pgrep -af '08_run_cuda_study|07_profile_cuda|llama-server|nsys'
cat results/prefix-study-1b/progress.json
cat results/prefix-study-8b/progress.json
```

Use the existing runner for new experiments or resuming matching prefix studies:

```bash
python3 scripts/08_run_cuda_study.py --prefix-study --output-dir results/prefix-study-1b
python3 scripts/08_run_cuda_study.py --prefix-study --manifest models/manifest-8b.json --output-dir results/prefix-study-8b
```

The runner skips only complete, validated cells with matching fingerprints.
`--dry-run`, `--repetitions` and `--requests` support inspection or smaller smoke runs;
use a separate output directory when settings change. Baseline commands omit
`--prefix-study` and use `results/cuda-study-{1b,8b}`.

**Baseline source snapshot:** `results/baseline-code.tar.gz` was saved before adding
optional prefix flags. Completed baseline fingerprints refer to that older code.
For an exact baseline rerun, use the snapshot in an isolated checkout; otherwise
choose a new output directory. Do not overwrite canonical results to bypass a mismatch.

Profiling uses the existing `scripts/07_profile_cuda.sh`; reproduction commands and
output definitions are in `references/notes.md`. Prefix captures set
`PROFILE_REPEAT_PROMPT=1` and `PROFILE_PREFIX_REUSE=0` or `1`, with c8.

## Experiment chronology

### E00–E03 — Environment, sources and model scope

- Initial `nvidia-smi` failed: kernel driver 595.71.05 versus NVML 595.91.07.
  Extracted matching libraries from an official NVIDIA installer into
  `.run/cuda-recovery`; changed no system libraries or driver modules. Its `env.sh`
  selects those libraries. CUDA test kernel returned its expected value.
- Nsight Compute returned `ERR_NVGPUCTRPERM`; driver policy is
  `RmProfilingAdminOnly: 1`, and passwordless sudo is unavailable. Nsight Systems
  CUDA tracing succeeded. Diagnostic evidence is retained under `.run/cuda-recovery`.
- Read both Medium articles, the second through indexed text, and the accessible
  24-page author preprint at `https://arxiv.org/pdf/2508.08531`. The exact 26-page
  ACM final PDF returned 403; do not claim its final-version changes were checked.
- Read both pages of user-supplied `references/Tapia.pdf` and inspected its figures.
  It measures MI300X 70B prefill, not documented concurrent HTTP serving. Preserve
  the paper's favorable A6000 results separately from its Apple dequantization costs.
- The January 10 article's memory-bound roofline is labeled 1B Q4_0, distinct from
  its preceding 70B Q4/IQ comparison. Keep those configurations separate.
- Initially selected 8B; stopped our downloads when the user requested 1B first.
  Later authorization allowed 8B after 1B. All inference stayed on one GPU.

### E04–E08 — Build, client validation and initial experiments

- Built the pinned CUDA source; one clean compilation interruption changed only
  build parallelism. Compiler/configuration output is retained in `logs/build.log`.
- Q4_K_M 1B streaming smoke completed successfully. Found a tokenizer mismatch:
  requesting 128 input tokens produced 129 because preparation omitted BOS.
  Fixed `/tokenize` with `add_special=true` to match the serving path.
- Added authoritative server token counts, fixed-output checks, cache observations,
  first-wave synchronization, exact request timestamps and measured client overlap.
  Updated stop handling for terminated zombie processes and nested result summaries.
- Completed initial 1B performance/traces and began 8B. Those performance rates
  were later superseded by E10; retain their archive only as diagnostic history.
- Consolidated download output into `logs/download.log` and build output into
  `logs/build.log`. Model provenance remains in manifests, not progress-bar logs.
- `results/model-inventory.json` records mixed GGUF tensors and effective file
  bits/parameter: 1B Q8/Q4/IQ1 = 8.55/5.23/2.68; 8B = 8.51/4.90/2.15.

### E09–E11 — Trace hygiene and hidden cache confound

- Initial Nsight archives inherited a download credential variable. No artifacts
  were published. Removed those archives, restricted capture/export environments
  to required runtime variables, and regenerated clean traces; no credential
  values are recorded in this log. Supporting notes moved to `references/notes.md`.
- Trace audit found 499 MB of D2H copying before the first measured 1B kernel:
  224 transfers from saving idle-slot KV state. `cache_prompt=false` and zero
  reused tokens did not disable the server's default 8 GiB RAM cache.
- Added `--cache-ram 0 --no-cache-idle-slots` to serving and profiling wrappers.
  Interrupted partial 8B cleanly, archived the affected runs under
  `results/preliminary-cache-enabled/`, then reran corrected 1B followed by 8B.
- Corrected Q4_K_M c8 validation showed zero copies before its first kernel.
  Remaining 525,336,576 D2H bytes match 8 requests × 128 outputs × 128,256
  vocabulary entries × 4-byte logits for CPU sampling; this is not DRAM bandwidth.
- Clean archives passed the runtime-environment allowlist audit. Canonical accepted
  runs are identified by study `progress.json`, not accumulated console-log history.

### E12–E14 — Corrected baselines complete

- Both sizes passed all 18 cells, token/cache checks and server-metric deltas;
  288 requests per size. Both cache-disabling flags were present in all model loads.
- Corrected 1B mean output tokens/s, c1 → c8: Q8 281.15 → 686.53;
  Q4 343.41 → 674.13; IQ1 348.59 → 642.33. Q8/Q4 differ by only 1.84% at c8.
- Corrected 8B: Q8 59.85 → 163.33; Q4 81.65 → 158.29;
  IQ1 93.05 → 150.67. Each size has three repetitions; variability is in its CSV.
- All 12 corrected baseline Nsight Systems traces completed. Kernel templates
  establish one-column versus eight-column MMVQ work, with ramp/drain variation.
  Q4/Q8 prefill uses MMQ; IQ1_M includes dequantization plus FP16 cuBLAS.
- An actual-serving NCU attempt again recorded `counter_status=permission_denied`.
  Low-level instruction/cache/bandwidth explanations therefore remain hypotheses.
- Archived baseline code before extending optional prefix support. No GPU jobs
  overlapped, and no multi-GPU inference was performed.

### E15–E16 — Matched prefix reuse studies

- Extended the existing client, summary and runner with `--repeat-prompt` and
  `--prefix-reuse`/`--prefix-study`; no additional per-model scripts. Ten client
  regression tests and runner dry-run/cache-validation checks passed.
- Compare identical repeated 2,048-token prompts at c8, all formats and both sizes,
  16 requests per mode and three repetitions. Input hashes verify matched messages.
  Mode order alternates by repetition while format order rotates.
- Before each mode, eight uncached requests prime all slots; reuse-on additionally
  receives a discarded cached warmup. Idle-slot RAM serialization stays disabled.
- Off requires zero cache hits. On requires positive hits on every request and
  logical = cached + evaluated prompt counts: observed 2,047 cached plus one
  evaluated token, with 128 newly generated outputs. Both studies passed all cells.
- This is a best-case warmed GPU-slot reuse experiment, not a production hit-rate
  model or general global-prefix deduplication benchmark. Distinguish logical
  input throughput from actually evaluated prompt tokens when interpreting speedup.
- Prefix performance finished before the 12 matched off/on c8 traces began.

### E17 — Final validation and report

- All 576 prefix-study requests passed independent checks of hashes, exact token
  counts, cache policy, server counters and eight-slot overlap. Together with the
  corrected baselines, 72 cells / 1,152 requests enter the report.
- All 24 traces passed capture and environment checks. Matched reuse-on captures
  have no large-prompt MMQ, standalone dequantization or cuBLAS GEMM launches.
  IQ1_M kernel time falls 1,248 to 521 ms at 1B and 6,305 to 2,095 ms at 8B;
  matrix-vector time changes little. These are diagnostic single captures.
- Prefix reuse changes IQ1_M throughput from 638 to 1,339 tok/s at 1B and
  151 to 429 tok/s at 8B, making it fastest in this best-case cache regime.
- Wrote and independently checked `FINDINGS.md` against all four summaries and
  the source/trace evidence. Ten unit tests, shell syntax and whitespace checks
  pass. References and download logs are consolidated; no extra report or plot
  scripts were added. All managed GPU processes have exited.

## Final reporting boundaries

Remote-workflow validation: 48 real CUDA requests across all three formats and
c1/c8, one eight-request Nsight trace, real detection of busy hardware counters,
CUDA/ROCm dry runs, mocked ROCm execution/resume/telemetry, temporary download
and build fixtures, counter CSV/gating-version tests, shell/Python syntax and ten
client regression tests. The late model-integrity and AMD trace-shape fixes have
focused checks. No A100/MI300X execution or successful hardware-counter capture
was claimed. The two new workflow files are `execute.sh` and
`config/remote-study.env`; existing runners/profilers were extended.

Use corrected baseline and matched prefix results only. Report output throughput,
TTFT/TPOT and per-request latency together; these finite closed-loop experiments
include scheduling and drain, not sustained arrival-rate capacity or SLO guarantees.

Nsight Systems measures kernel paths, timing, launch resources and explicit copies.
It does not recreate IPC, achieved occupancy, L1/L2 cache hit rates, DRAM bandwidth,
coalescing, tensor-pipeline utilization or roofline counters on this host. Static
DP4A/MMA instructions are not measured utilization. Preserve unresolved findings.

### E18 — Llama 70B baseline on two GPUs (complete)

- Used existing local Llama 3.3 70B Instruct Q4_K_M (39.600 GiB) and Q2_K
  (24.564 GiB), without downloading/copying models. Absolute paths and SHA256
  are in `models/manifest-70b.json`; converter revisions are unknown.
- Reused the runner/client/profiler with `config/multigpu-study.env`, GPUs 0,1,
  `--split-mode layer --tensor-split 1,1`, eight slots and the original 2048/128
  workload. Logs verify 81/81 layers offloaded, 41 transformer layers on GPU 0,
  39 plus output head on GPU 1, and FP16 KV of 5248/4992 MiB. Embeddings stay
  on CPU. No single-GPU scaling control was run.
- First Q4 load exceeded 180 seconds before any measurements. Increased only
  the 70B startup allowance to 600 seconds. Before accepting results, changed
  both formats to `--load-mode none`: ordinary RAM is constrained by 84 GiB
  reserved unused hugepages, while streaming loading uses small upload buffers.
  Streaming Q4 loaded in about 118 seconds. Memory pressure plausibly affected
  mmap loading, but loader causality was not isolated. Host settings were unchanged.
  Failed/interrupted attempts remain excluded under `results/diagnostics/`.
- All 12 cells / 192 requests passed, three repetitions of 16 requests. Q4 c1/c8
  = 12.029 ± 0.010 / 27.229 ± 0.091 output tok/s; Q2 = 13.244 ± 0.006 /
  22.379 ± 0.018. Q2 wins c1 by 10.1%; Q4 wins c8 by 21.7%. Combined NVML
  VRAM allocation is about 50.3/35.5 GiB for Q4/Q2. At this KV capacity Q4
  exceeds one 48-GB GPU; this does not imply it cannot fit with a smaller context.
- Reproduction after sourcing `.run/cuda-recovery/env.sh`:
  `python3 -u scripts/08_run_cuda_study.py --backend cuda --config-file config/multigpu-study.env --manifest models/manifest-70b.json --output-dir results/cuda-study-70b --repetitions 3 --requests 16`
  Console output appends to `logs/study-cuda-70b.log`. Completed cells resume only
  with matching fingerprints; use a new output directory for changed code/settings.
- Existing tests passed: ten client regressions, shell/Python syntax, whitespace,
  runner mocks and CUDA/ROCm dry runs. Test discovery uses
  `python3 -m unittest discover -s benchmark -p 'test_*.py'`. No new experiment
  or test scripts were added; portable `execute.sh` also supports two-GPU 70B.

### E19 — 70B prefix comparison (complete)

- Used the E18 command with `--prefix-study --output-dir results/prefix-study-70b`.
  All 12 cells / 192 requests passed. Identical input hashes match every format,
  mode and repetition. Every reuse-on request cached 2,047 tokens and evaluated
  one; reuse-off cached zero. Slot priming/warmup was excluded from timing.
- Q4 off/on means = 27.3249/57.1378 tok/s; Q2 = 22.3910/49.9036.
  TTFT p95 falls 20.029 s→237.0 ms for Q4 and 25.791 s→235.7 ms for Q2.
  Reuse improves throughput 2.09×/2.23×, but Q4 still wins by 14.5%.
- Across baseline plus prefix, all 384 retained requests passed independent
  input/output/cache, server concurrency/counter, telemetry and fingerprint
  audits in `results/70b-audit.json`. Profiling requests are excluded.
- NVML cumulative software thermal-clock-limiting counters grew by 275.02 s
  during the baseline window and 165.56 s during prefix on GPU 0; GPU 1 had
  zero growth. These elapsed windows include loading/warmup/idle time and do
  not locate effects in individual cells. Default 300-W limits/automatic clocks
  stayed unchanged. Evidence: environment, telemetry and `clock-events.log`.

### E20 — 70B two-GPU timelines (complete)

- All six Nsight Systems captures passed 132 audit checks. Each format has c1/c8
  uncached and c8 reuse-on; performance runs ended before profiling began, and
  GPU jobs did not overlap. Captures exclude loading/warmup, include both GPUs,
  and retain clean allowlisted environments. No detailed NCU counters succeeded.
- Reused `scripts/07_profile_cuda.sh` with `CONFIG_FILE=config/multigpu-study.env`,
  `GPU_DEVICE=0,1`, `PROFILE_TOOL=nsys`, `PROFILE_REPEAT_PROMPT=1`, model fields
  from the manifest and `PROFILE_CONCURRENCY=1` or `8`. Cached mode additionally
  sets `PROFILE_PREFIX_REUSE=1`. Exact commands are in each metadata.json.
  Local NSYS_BIN: `/home/hys4qm/.cuda-13.0/nsight-compute-2025.3.1/host/target-linux-x64/nsys`.
- Uncached paths: `results/cuda-study-70b/profiles/{quant}-c{1,8}`. Cached paths:
  `results/prefix-study-70b/profiles/{quant}-reuse-on`. The repeated-prompt c8
  uncached capture also supplies the matched prefix-off trace; no duplicate was
  needed. OFF/ON hashes, token counts, KV placement and gating all passed.
- Both formats use fused MMQ prefill, unlike IQ1_M's separate dequantization
  path. Q2 c1 matrix-vector kernel sum is 5.970 s versus Q4 7.699 s; at c8,
  Q2 is slower, 18.864 s versus 16.155 s. Q2 uncached c8 MMQ also takes 32.7%
  longer. With reuse, MMQ disappears and matrix-vector sums are 18.401/15.780 s
  for Q2/Q4. Almost 99% of cached matrix-vector calls process eight columns.
  These sums explain kernel cost changes; they are not HTTP throughput.
- Both cached traces have zero simultaneous two-GPU kernel activity. Uncached c1
  traces overlap during prefill, then execute the GPU layer partitions in
  sequence during decode. The layer split provides capacity; no one-GPU control
  exists to measure scaling. It does not establish a bandwidth/occupancy limit.
- Peer calls carry 71,270,400 logical bytes at c1, 570,163,200 at uncached c8,
  and 33,554,432 at cached c8. Each c8 trace separately downloads 525,336,576
  bytes of logits for CPU sampling. Correlated D2H/H2D-labelled device-memory
  rows represent one peer call: count its bytes once. The old PTOP-only tally
  `peer_copy_bytes=0` does not mean zero GPU-to-GPU traffic. Physical routing
  is unresolved; no host-staging or NVLink bandwidth claim is supported.
- Updated `FINDINGS.md` and this log. All six traces and 384 requests passed
  independent audit; managed servers/profilers exited and both GPUs are idle.
  Do not rerun completed studies on resumption; inspect `progress.json`, profile
  metadata and live processes before any newly authorized experiments.

### E21 — Counter-resource resolution guidance (read-only)

- Rechecked the running exporter: root PID 11381, nested Kubernetes pod UID
  `997a89be-c162-4e46-a247-e8778019d584`, inside Docker container
  `fdad19f60d4a74210fc7b8601bfbf04d5538a249a4f869234de3b7bda24990b0`.
  The accessible host K3s cluster does not contain that pod. Docker access is
  denied, sudo requires a password, and no host `dcgmi` is installed.
- User identified that Docker container as `tripod-worker`, a kind Kubernetes
  node (`kindest/node:v1.35.0`). No local kubeconfig for this cluster is available;
  next inspect its exporter pod and controller through the kind control plane.
- NVIDIA documents `dcgmi profile --pause/--resume` for a reachable host engine.
  An embedded exporter requires control of its actual engine or a graceful stop
  through its owning workload; freezing a container does not release counters.
  Need privileged identification of that workload before exact stop/restore
  commands. No service, pod, driver or permission changes were made.
- After collection is stopped, run existing `execute.sh doctor cuda` separately
  with GPU_DEVICE=0 and 1. Success would implicate DCGM; a continued busy error
  needs further investigation. Restore monitoring after profiling.
- Added requested local Nsight Compute bash instructions and DCGM terminology
  to the existing README. Verified installed NCU 2025.3.1 and existing wrapper
  paths/phases; no new scripts or GPU counter captures. A privileged read of the
  kind exporter pod/controller remains needed to give targeted stop/restore steps.

### E22 — Isolated 1B counter diagnostic and profiler update (complete)

- User requested a local 1B test with scripts/artifacts in `tmp/`. Added only
  `tmp/check_1b_counters.sh`: pinned Q4_K_M, GPU 0, 128 input/32 output tokens,
  one warmup then one profiled request, configured to sample one matrix-vector launch.
- First attempt failed before inference because NCU 2025.3.1 rejects
  `--print-metric-name name` with `--page raw`. Removed that option from the
  existing profiler and portable probe/export commands; NVIDIA's shipped report
  verified raw CSV already contains canonical metric names. No study data changed.
- Retry `tmp/ncu-1b-20260912T161147Z-IFaE3a/` completed the warmup (128/32,
  zero cached tokens), then NCU failed with `resource_unavailable`, exit 9.
  No counters collected; server exited and GPU is idle. `dcgmi` is absent;
  root exporter PID 11381 remains active. This diagnostic is excluded from all
  scientific request/trace totals.
- User supplied controller information: `gpu-operator/nvidia-dcgm-exporter`,
  pod `nvidia-dcgm-exporter-s4r8m` on kind node `tripod-worker`. NVIDIA source
  supports the node-only `nvidia.com/gpu.deploy.dcgm-exporter=false` label;
  preserve/restore its previous value. No monitoring changes were made.
- User then requested reinstalling/updating Nsight Compute to test whether the
  version causes this failure. Installed NVIDIA's current 2026.3.0 release at
  `/data/home/hys4qm/.local/opt/nvidia/nsight-compute/2026.3.0`, beside 2025.3.1.
  Source URL, installer SHA256 and version are in `tmp/nsight-compute-install/`;
  one combined install log. The local environment helper now prefers this NCU;
  the CUDA 13.0 compiler/runtime and driver 595.91.07 are unchanged.
- New-version retry `tmp/ncu-1b-20260912T161648Z-IVaeT1/` again completed the
  exact 128/32-token warmup, then failed with resource unavailable/exit 9.
  NCU's detail log reports failure to obtain the counter availability image.
  Updating the profiler did not resolve this failure. Both versions have zero
  successful counter samples; both managed servers exited, GPUs idle. DCGM
  ownership remains unproven until a controlled stop/retest can be performed.

### E23 — Release counter resource and collect hardware explanations (blocked)

- Revalidated idle GPUs, active root exporter PID 11381, and unavailable
  noninteractive sudo. Prepared `tmp/dcgm_counter_access.py pause|resume|status`.
  It validates the exporter selector, preserves the original node label and
  resource UIDs, changes only `nvidia.com/gpu.deploy.dcgm-exporter`, and waits
  for the exporter pod to disappear. Eight CPU-only lifecycle/mock scenarios
  passed. The user was asked to run `pause` in their terminal; no monitoring
  change is assumed until actual state/process evidence confirms it.
- After pause: rerun the temporary 1B counter check, then collect counter samples
  through existing `execute.sh` for 1B, 8B and two-GPU 70B, prefill/decode/reuse.
  Preserve all unprofiled benchmarks. Use matching kernel shapes, explicit cold
  replay caches (`PROFILE_CACHE_CONTROL=all`) and unchanged automatic clocks;
  profile timings do not represent HTTP performance or production cache state.
- Reviewed metrics against the installed GA102 catalog. Extended the existing
  profiler with elapsed-cycle IPC, instruction count and global-load bytes per
  sector/max-rate, in addition to existing active IPC, DRAM, cache and occupancy
  counters. Sectors/request is not a universal coalescing percentage.
- Extended existing selectors for mixed recipes: Q3_K in Q2_K, Q6_K in Q4_K_M,
  and Q5_K/IQ2_XXS decode in IQ1_M. The selected types account for about 93–100%
  of uncached c8 quantized-matvec time in saved traces; this is planned type
  coverage, not counter evidence or whole-model coverage. All 62 generated CUDA
  selectors matched saved kernel names with correct enum IDs. Wide/long CSV
  fixtures verified new IPC, instruction, sector and bandwidth mappings and
  missing-value handling; fixtures removed. No new GPU runs or scripts.
  Audit actual sampled devices after collection; per-launch-config is not per-GPU.
- Rechecked the installation and old/new diagnostic evidence: NCU 2026.3.0 is
  selected by the local environment helper; upgrading did not resolve access.
  Exporter PID 11381 remains active and no pause state exists. README now has
  the exact terminal pause/resume commands; actual collection awaits that step.
- Restore the saved exporter label with the helper's `resume` after profiling,
  then verify monitoring returns. Goal remains incomplete until real counter
  data, interpretations and restoration are verified.
- Blocked audit: previous turn made progress by correcting selectors and testing
  CSV mappings. This continuation reconfirmed exporter PID 11381, no pause state,
  no helper/profiler/inference process, idle GPUs and `sudo: a password is required`.
  This is an external-action blocker, not a live-job wait. After three consecutive
  goal turns with this same condition, stop automatic retries. User terminal:
  `python3 /home/hys4qm/cloud-serving-benchmark/tmp/dcgm_counter_access.py pause`.
  On resumption, verify the exporter exited, then run
  `bash tmp/check_1b_counters.sh` before the prepared counter matrix. Restore with
  the helper's `resume` after testing, including if access still fails.

### E24 — Real 1B hardware values collected and audited (measurements complete)

- On resumption, `tmp/dcgm-counter-access/state.json` recorded `paused` at
  17:29:47 UTC, the exporter was absent and both GPUs were idle. The initial
  128/32-token 1B test then collected one real Q4_K matrix-vector launch:
  7,979,008 executed warp instructions and 34,592 ns. It was incorrectly marked
  `no_report` because NCU 2026.3 defaults to `.ncu-repz`. Fixed
  `benchmark/profile_cuda.py` to locate/import both report extensions, and
  `execute.sh doctor` to use the shared wide/long CSV parser. The original raw
  evidence stays unchanged; recovery is under `tmp/ncu-report-recovery-3tgme5zb`.
- Rerun `tmp/ncu-1b-20260912T173155Z-J8mnW3` passed with `counter_success=true`.
  The real `RUN_TAG=hardware-resolution-20260912 GPU_DEVICE=0 bash execute.sh
  doctor cuda 1b` also passed after collection. Legacy report/CSV compatibility
  was checked against saved real reports; syntax and whitespace checks passed.
- Reproduction at the current result location: `source .run/cuda-recovery/env.sh`,
  then `python3 tmp/run_1b_hardware_metrics.py --output results/cuda-study-1b/hardware-counters`.
  All 23 captures completed: 166 kernel samples, 28 requested metrics per sample,
  4,648 finite requested values. Q8/Q4/IQ1 include mixed secondary tensor types,
  prefill selections, c1/c8 matvec and c8 prefix reuse. All 142 profiled, 142
  warmup and 48 priming requests passed exact 2048/128-token and cache checks.
  All three model hashes match the manifest. Existing serving studies and their
  1,536-request/30-timeline totals were preserved; diagnostic requests are separate.
- Used cold replay caches, automatic clocks, two launches per matching launch
  configuration and the original eight-slot server settings. Column selectors
  identify the small-batch path but can include final-prefill work because the
  temporal gate covers the full warmed HTTP burst. Grid equality does not prove
  equal matrix dimensions. Results are selected kernel diagnostics, not
  whole-model averages, HTTP timing or AMD-equivalent coalescing/IPC values.
- `tmp/summarize_1b_hardware_metrics.py` produces `hardware-metrics.csv`,
  `hardware-summary.csv/json` and `README.md` in the capture directory. The
  independent audit cross-checks all raw values, hashes, requests/cache/concurrency,
  gates, selections, derived metrics and all 3,904 aggregate metric rows.
  Evidence: `results/cuda-study-1b/hardware-counters/independent-audit.json`.
- One short dequantization sample has a nonphysical 101.64% L2 ratio; raw retained,
  flagged and excluded from physical percentage summaries. A targeted repeat via
  `tmp/check_1b_l2_precision.py` reduced replay passes from 12 to three but still
  returned 100.32% on one short launch. Both precision limits are documented;
  88/166 main launches are under 20 microseconds. The main FINDINGS table uses
  the longer dequantization shape, whose values are physically bounded.
- Updated `FINDINGS.md`, README and source notes with actual 1B bandwidth,
  cache, occupancy, IPC, tensor, scheduler and load-access values. Preserved
  concurrently added context-length/quantization explanations in FINDINGS.
  8B/70B hardware counters and the separately documented long-context sweep
  are outside these completed 1B diagnostics and remain unmeasured.
- All profiling has stopped. Exporter restoration was requested after the last
  GPU check because noninteractive sudo still requires a password. Monitoring
  restoration is pending; verify the saved state and exporter process before
  declaring it restored. Do not rerun the completed captures on resumption.

### E25 — Store hardware measurements with the 1B results

- Moved the hardware report, per-launch values, summaries, all 23 native captures
  and their audit to `results/cuda-study-1b/hardware-counters/`. The reduced-metric
  precision capture is in its `l2-precision/` subfolder.
- Updated FINDINGS, README, notes, current reproduction commands and script
  destinations. Diagnostic scripts remain in `tmp/`; collected results now go
  under `results/`. Regenerated the report without running GPU workloads.
- Verified all 369 moved files: 367 are byte-identical, including every capture,
  numerical summary and original audit. Only the README navigation and progress
  directory pointers changed. `relocation-audit.json` records hashes and maps
  historical acquisition paths to current locations. All local report links pass.

[2026-09-12 20:05:37 UTC] Context study: plan saved; both GPUs passed counter probes; 8 concurrent 16,384-input/512-output requests validated; 1B FP16 ready; 8B FP16 conversion and 70B IQ1_M download active; full-runner calibration running.
[2026-09-12 20:06:45 UTC] Context study: runner validation passed; validation reporting will remain brief; detailed artifacts are reserved for measured inference and profiling experiments.
[2026-09-12 20:10:34 UTC] Context study: validation updates kept to pass/fail; detailed artifacts retained for benchmark measurements and hardware profiles.
[2026-09-12 20:11:31 UTC] Context study: 70B IQ1_M ready; 8B FP16 conversion finishing; private diagnostic build completing before throughput measurements.
[2026-09-12 20:14:05 UTC] Context study: required model files ready; checking 8B FP16 chat-template compatibility; profiling combines compatible counters while retaining prefill/decode coverage.
[2026-09-12 20:16:11 UTC] Context study: common 8B chat template applied to prevent date-text input differences; validation passed; diagnostic build nearly finished.
[2026-09-12 20:17:40 UTC] Context study: diagnostic build complete; starting measured serving sweep in order 1B, 8B, 70B with 512 output tokens per request.
[2026-09-12 20:19:46 UTC] Context study: valid measurements collected for 1B FP16 at eight clients and 2,048 input/512 output tokens; larger prompts and concurrency settings continuing.
[2026-09-12 20:22:06 UTC] Context study: initial review found missing thermal-throttling flags; adding them and restarting sweep, retaining initial measurements separately.
[2026-09-12 20:24:02 UTC] Context study: corrected sweep running with thermal-throttling flags confirmed on both GPUs; serving measurements precede hardware profiling.
[2026-09-12 20:25:21 UTC] Context study: first complete corrected setting has 48 successful requests and full thermal telemetry; organized results index linked from FINDINGS.md.
[2026-09-12 20:26:34 UTC] Context study: serving sweep remains active; completed 1B FP16 settings are being reported while profiling coverage checks are finalized.
[2026-09-12 20:26:48 UTC] Context study: 1B FP16 at eight clients completed 2k, 4k, and 8k inputs; moving to 16k and larger capacity tests.
[2026-09-12 20:28:44 UTC] Context study: 1B FP16 C8/P16k completed all 48 requests with exactly 512 output tokens and no failures, swap, or truncation; 32k/64k capacity tests next.
[2026-09-12 20:30:21 UTC] Context study: 32k capacity test progressing; profiling checks require selected-operation coverage including IQ1_M dequantization.
[2026-09-12 20:32:13 UTC] Context study: 1B FP16 C8/P32k completed successfully; testing 64k next; other formats and concurrency levels remain pending.
[2026-09-12 20:35:05 UTC] Context study: 1B FP16 C8/P64k passed all 24 requests across three capacity runs with 512 output tokens and GPU-resident model/KV; proceeding to 16 clients.
[2026-09-12 20:36:20 UTC] Context study: serving sweep continues at 16 clients; private profiling labels being corrected for fused kernels; serving binary unchanged.
[2026-09-12 20:37:51 UTC] Context study: 1B FP16 P2k throughput rose 945→1,284 tok/s from C8→C16; p95 TPOT rose 8.3→12.2 ms; larger settings continue.
[2026-09-12 20:44:32 UTC] Context study: 1B FP16 C16/P16k completed; results tables refreshed while 32k/64k capacity tests continue.
[2026-09-12 20:46:48 UTC] Context study: 1B FP16 C16/P32k is on its final repetition; P64k follows; completed settings have no failures.
[2026-09-12 20:48:54 UTC] Context study: 1B FP16 C16/P32k passed; C16/P64k loaded with full GPU placement and required headroom; request measurements underway.
[2026-09-12 20:52:24 UTC] Context study: first 1B FP16 C16/P64k measured burst passed; two repetitions remain before capacity acceptance.
[2026-09-12 20:54:40 UTC] Context study: 1B FP16 C16/P64k final repetition running; first two passed token-count, GPU-placement, and memory checks.
[2026-09-12 20:57:38 UTC] Context study: 1B FP16 P64k passed all three bursts at C16; 64k now validated at C8 and C16; sweep moved to C32.
[2026-09-12 20:59:50 UTC] Context study: 1B FP16 C32/P2k completed; P4k final repetition running; 13 settings have passed.
[2026-09-12 21:02:12 UTC] Context study: 1B FP16 C32/P4k passed; 14 settings completed; C32/P8k running.
[2026-09-12 21:03:33 UTC] Context study: 1B FP16 C32/P8k final repetition running; first two passed; P16k follows.
[2026-09-12 21:05:29 UTC] Context study: 1B FP16 C32/P8k passed; 15 settings complete; C32/P16k starting.
[2026-09-12 21:10:50 UTC] Results organization: labeled cuda-study-1b as the earlier single-GPU 2k/128-token C1/C8 experiment; current two-GPU 512-token results remain in context-study/fp16kv-512-20260912; repetitions and summaries explained.
[2026-09-12 21:12:25 UTC] Context study: 1B FP16 C32/P16k passed all three repetitions; 16 settings complete; C32 capacity extensions are underway; results index refreshed.
[2026-09-12 21:13:40 UTC] Context study: serving driver remains live on 1B FP16 C32 capacity tests; profiling waits for the unprofiled serving sweep to finish.
[2026-09-12 21:14:11 UTC] Context study: first 1B FP16 C32/P32k burst passed; two repetitions remain; read-only 70B trace review is narrowing the later Q2/Q4 diagnostics.
[2026-09-12 21:16:23 UTC] Context study: 1B FP16 C32/P32k final repetition running; reporter preserves fused-operation identity; 16 focused checks passed, with no detailed validation log saved.
[2026-09-12 21:17:55 UTC] Context study: 1B FP16 C32/P32k passed; 17 settings complete; C32/P64k loaded fully on GPUs with 9.3/17.2 GiB free; request validation remains pending.
[2026-09-12 21:19:33 UTC] Context study: C32/P64k warmup continues; earlier 70B traces suggest two FFN shapes for concise causal diagnostics, subject to confirmation from new-study traces.
[2026-09-12 21:20:46 UTC] Context study: live serving driver checked; 1B FP16 C32/P64k remains underway before the C64 sweep.
[2026-09-12 21:21:20 UTC] Context study: live logs confirm C32/P64k warmup is advancing through prefill and decode; GPU memory headroom remains valid; thermal and clock telemetry are being recorded.
[2026-09-12 21:24:18 UTC] Organization update: preparing cuda-context-study-1b/8b/70b paths separate from old cuda-study folders; new workloads will receive fresh timeline/counter captures, while serving telemetry is recorded inline.
[2026-09-12 21:27:21 UTC] Results renamed: physical data now in cuda-context-study-1b, with separate 8b/70b folders and shared cuda-context-study overview; temporary old-path link preserves the live run; C32/P64k first burst passed.
[2026-09-12 21:29:55 UTC] New canonical model folders and shared overview are in place with one physical copy; relocated profile selection works; C32/P64k two bursts passed and final repetition is running.
[2026-09-12 21:31:41 UTC] Folder migration preserved serving fingerprints and live results; per-model summary.md/summary.csv generation is being finalized alongside the shared report.
[2026-09-12 21:33:49 UTC] New model summaries and shared report are ready; 1B has 12 serving plus 5 capacity-extension results, 8B/70B are explicitly pending; fresh hardware profiles will accompany the new study; validation passed without detailed logs.
[2026-09-12 21:34:08 UTC] Context study: 1B FP16 C32/P64k passed all three 512-output bursts with full GPU placement; 18 settings complete; canonical summaries refreshed and C64 sweep follows.
[2026-09-12 21:35:34 UTC] Context study: live driver checked after folder migration; 18 1B settings complete and C64 serving sweep continues; fresh profiling remains queued.
[2026-09-12 21:37:36 UTC] Context study: 1B FP16 C64/P2k first two repetitions passed; final repetition running; profiler/custom-series and diagnostic output paths now match the canonical layout; focused validation passed.
[2026-09-12 21:39:01 UTC] Context study: 19 settings complete; 1B FP16 P2k C32→C64 throughput rose 1498→1591 tok/s (+6.2%) while p95 TPOT rose 20.8→39.3 ms; C64 larger-prompt sweep continues.
[2026-09-12 21:40:34 UTC] Context study: 1B FP16 C64/P4k first repetition passed; two remain; live sweep continues with 19 complete settings.
[2026-09-12 21:41:38 UTC] Context study: live C64/P4k run checked; second repetition passed and final repetition is underway; summaries await full-cell completion.
[2026-09-12 21:43:24 UTC] Context study: 1B FP16 C64/P4k passed all three repetitions; 20 settings complete; canonical summaries refreshed while C64/P8k starts.
[2026-09-12 21:45:23 UTC] Context study: live C64/P8k run checked; serving continues on both GPUs with 20 settings complete; detailed profiling remains queued.
[2026-09-12 21:45:35 UTC] Context study: C64/P8k warmup finished and first measured repetition is running; full GPU placement and startup headroom passed.
[2026-09-12 21:46:49 UTC] Context study: 1B FP16 C64/P8k first measured repetition passed; two remain before inclusion in summaries.
[2026-09-12 21:47:51 UTC] Context study: live C64/P8k driver checked; measured repetitions continue, with completed results passing request and memory checks.
[2026-09-12 21:49:06 UTC] Context study: 1B FP16 C64/P8k second repetition passed; final repetition is running before summary inclusion.
[2026-09-12 21:50:42 UTC] Context study: 1B FP16 C64/P8k passed all three repetitions; 21 settings complete; summaries refreshed as the C64/P16k test starts.
[2026-09-12 21:51:32 UTC] Context study: C64/P16k loaded all layers and FP16 KV on GPUs with 27.1/31.0 GiB free; warmup and request repetitions remain underway.
[2026-09-12 21:52:40 UTC] Context study: live C64/P16k driver checked; final main-grid FP16 prompt length is underway before C64 capacity extensions.
[2026-09-12 21:52:56 UTC] Context study: C64/P16k warmup passed; first measured repetition is running; summary inclusion requires all three repetitions.
[2026-09-12 21:55:07 UTC] Context study: serving driver remains live; latest recorded step FP16/c64/p16384/r1 is running; hardware profiling stays queued.
[2026-09-12 21:55:51 UTC] Context study: 1B FP16 C64/P16k first measured repetition passed; two repetitions remain before full-setting acceptance.
[2026-09-12 21:57:56 UTC] Context study: live serving driver checked; FP16/c64/p16384/r2 is running; remaining C64 capacity lengths follow full-setting completion.
[2026-09-12 21:59:25 UTC] Context study: 1B FP16 C64/P16k second repetition passed; final repetition is running before C64 capacity extensions.
[2026-09-12 22:00:43 UTC] Context study: live driver checked; FP16/c64/p16384/r3 is running; main 1B FP16 grid awaits its final repetition before C64 capacity tests.
[2026-09-12 22:02:02 UTC] Context study: final C64/P16k repetition remains active; first two passed; no new complete setting is added yet.
[2026-09-12 22:03:12 UTC] Context study: all 16 main 1B FP16 settings passed (2k/4k/8k/16k × C8/16/32/64); 22 total settings complete including capacity extensions; C64/P32k follows.
[2026-09-12 22:05:40 UTC] Context study: live C64/P32k capacity test checked; three successful measured bursts are required before accepting the input length.
[2026-09-12 22:06:18 UTC] Context study: C64/P32k warmup continues after full-GPU placement and startup headroom passed; measured capacity bursts remain pending.
[2026-09-12 22:08:22 UTC] Context study: live serving driver checked; FP16/c64/p32768/r1 is running; C64/P32k capacity acceptance still requires completed measured bursts.
[2026-09-12 22:09:06 UTC] Context study: C64/P32k warmup passed memory and no-swap checks; first measured concurrent burst is running.
[2026-09-12 22:10:04 UTC] Context study: first 1B FP16 C64/P32k measured burst passed; two more are required for capacity acceptance.
[2026-09-12 22:11:07 UTC] Context study: live driver checked; FP16/c64/p32768/r2 is running; first C64/P32k burst passed and remaining bursts continue.
[2026-09-12 22:13:03 UTC] Context study: second C64/P32k burst remains active; awaiting its completed request and memory checks.
[2026-09-12 22:14:07 UTC] Context study: second 1B FP16 C64/P32k burst passed; final burst is running before capacity acceptance.
[2026-09-12 22:16:07 UTC] Context study: live driver checked; FP16/c64/p32768/r3 is running; C64/P64k screening and Q8 follow the final 32k burst.
[2026-09-12 22:17:45 UTC] Context study: 1B FP16 capacity sweep complete (P64k at C8/16/32, P32k at C64); C64/P64k excluded by VRAM estimate before launch; Q8 C8/P2k passed all three repetitions.
[2026-09-12 22:19:40 UTC] Context study: Q8 C8/P4k and P8k passed all three repetitions; 26 settings complete; matched-format summaries are being refreshed.
[2026-09-12 22:20:35 UTC] Context study: C8 Q8 vs FP16 throughput is 1111 vs 945 tok/s at P2k and 639 vs 577 at P8k; measured advantage narrows with input length; hardware explanation awaits fresh profiles.
[2026-09-12 22:21:43 UTC] Context study: Q8 C8/P16k passed all three repetitions; all four C8 main prompt lengths complete for Q8; 27 settings complete and summaries refreshed.
[2026-09-12 22:23:08 UTC] Context study: Q8 C8/P32k passed all three capacity bursts; 28 settings complete; capacity table refreshed as C8/P64k starts.
[2026-09-12 22:24:33 UTC] Context study: live Q8 capacity sweep checked; latest recorded step Q8_0/c8/p65536/r1 is running; burst completion and memory checks remain under observation.
[2026-09-12 22:24:48 UTC] Context study: Q8 C8/P64k warmup finished; first measured burst is running, with three passes required for table inclusion.
[2026-09-12 22:25:42 UTC] Context study: first Q8 C8/P64k measured burst passed; two further bursts remain for capacity acceptance.
[2026-09-12 22:26:50 UTC] Context study: second Q8 C8/P64k burst passed; latest recorded step Q8_0/c8/p65536/r3 is running; C16 sweep follows resolution.
[2026-09-12 22:27:54 UTC] Context study: Q8 C8/P64k passed all three bursts; 29 settings complete; latest Q8_0/c16/p2048/r3 is running; summaries refreshed.
[2026-09-12 22:29:29 UTC] Context study: Q8 C16 sweep remains live; 30 settings complete; latest Q8_0/c16/p4096/r3 is running; fresh hardware profiling remains queued.
[2026-09-12 22:30:32 UTC] Context study: Q8 C16/P4k passed all three repetitions; 31 settings complete; latest Q8_0/c16/p8192/r2 is running; summaries refreshed.
[2026-09-12 22:32:00 UTC] Context study: Q8 C16/P8k passed all three repetitions; 32 settings complete; latest Q8_0/c16/p8192 is complete; summaries refreshed.
[2026-09-12 22:33:20 UTC] Context study: Q8 C16/P16k first repetition passed; remaining repetitions continue after the C16/P8k summary update.
[2026-09-12 22:34:25 UTC] Context study: second Q8 C16/P16k repetition passed; latest Q8_0/c16/p16384/r3 is running; longer capacity tests follow full-setting completion.
[2026-09-12 22:35:28 UTC] Context study: Q8 C16/P16k passed all three repetitions; 33 settings complete; C16 capacity extensions follow; summaries refreshed.
[2026-09-12 22:36:45 UTC] Context study: live Q8 C16 capacity run checked; latest Q8_0/c16/p32768/r1 is running; three measured bursts are required for acceptance.
[2026-09-12 22:36:58 UTC] Context study: Q8 C16/P32k warmup finished; first measured concurrent burst is running.
[2026-09-12 22:37:54 UTC] Context study: first Q8 C16/P32k measured burst passed; capacity validation continues.
[2026-09-12 22:39:03 UTC] Context study: Q8 C16/P32k passed all three bursts; 34 settings complete; latest Q8_0/c16/p32768 is complete; capacity summaries refreshed.
[2026-09-12 22:40:55 UTC] Context study: live Q8 C16/P64k test checked; latest recorded step Q8_0/c16/p32768 is complete; burst and memory checks continue.
[2026-09-12 22:41:16 UTC] Context study: Q8 C16/P64k warmup remains active; measured bursts await completed warmup request and memory checks.
[2026-09-12 22:43:12 UTC] Context study: first Q8 C16/P64k measured burst passed; latest Q8_0/c16/p65536/r2 is running; remaining capacity bursts continue.
[2026-09-12 22:45:30 UTC] Context study: second Q8 C16/P64k burst passed; latest Q8_0/c16/p65536/r3 is running; capacity validation continues.
[2026-09-12 22:47:34 UTC] Context study: Q8 C16/P64k passed all three bursts; 35 settings complete; latest Q8_0/c32/p2048/r1 is running; capacity summaries refreshed.
[2026-09-12 22:49:33 UTC] Context study: Q8 C32/P2k passed all three repetitions; 36 settings complete; latest Q8_0/c32/p4096/r1 is running; summaries refreshed.
[2026-09-12 22:50:48 UTC] Context study: first Q8 C32/P4k repetition passed; remaining repetitions continue after the C32/P2k summary update.
[2026-09-12 22:52:09 UTC] Context study: Q8 C32/P4k passed all three repetitions; 37 settings complete; latest Q8_0/c32/p4096 is complete; summaries refreshed.
[2026-09-12 22:53:33 UTC] Context study: first Q8 C32/P8k repetition passed; latest Q8_0/c32/p8192/r2 is running; remaining repetitions continue.
[2026-09-12 22:54:40 UTC] Context study: second Q8 C32/P8k repetition passed; final repetition continues before summary inclusion.
[2026-09-12 22:55:51 UTC] Context study: Q8 C32/P8k passed all three repetitions; 38 settings complete; latest Q8_0/c32/p8192 is complete; summaries refreshed.
[2026-09-12 22:57:36 UTC] Context study: live Q8 C32/P16k run checked; latest Q8_0/c32/p16384/r1 is running; awaiting completed repetitions.
[2026-09-12 22:57:46 UTC] Context study: Q8 C32/P16k warmup finished; first measured repetition is running.
[2026-09-12 22:58:37 UTC] Context study: first Q8 C32/P16k repetition passed; remaining repetitions continue.
[2026-09-12 22:59:52 UTC] Context study: live Q8 C32/P16k run checked; latest Q8_0/c32/p16384/r2 is running; full-setting acceptance awaits remaining repetitions.
[2026-09-12 23:00:51 UTC] Context study: second Q8 C32/P16k repetition passed; final repetition is running before summary inclusion.
[2026-09-12 23:01:55 UTC] Context study: live Q8 C32/P16k run checked; latest Q8_0/c32/p16384/r3 is running; capacity extensions follow full-setting completion.
[2026-09-12 23:03:10 UTC] Context study: Q8 C32/P16k passed all three repetitions; 39 settings complete; C32 capacity extensions follow; summaries refreshed.
[2026-09-12 23:04:34 UTC] Context study: live Q8 C32/P32k capacity test checked; latest Q8_0/c32/p32768/r1 is running; three measured bursts are required for acceptance.
[2026-09-12 23:04:44 UTC] Context study: Q8 C32/P32k warmup finished; first measured concurrent burst is running.
[2026-09-12 23:06:37 UTC] Context study: first Q8 C32/P32k measured burst passed; latest Q8_0/c32/p32768/r2 is running; remaining capacity bursts continue.
[2026-09-12 23:07:35 UTC] Context study: second Q8 C32/P32k measured burst passed; final burst continues before capacity acceptance.
[2026-09-12 23:08:46 UTC] Context study: live Q8 C32/P32k run checked; latest Q8_0/c32/p32768/r3 is running; final acceptance precedes C32/P64k.
[2026-09-12 23:09:48 UTC] Context study: Q8 C32/P32k passed all three bursts; 40 settings complete; C32/P64k follows; capacity summaries refreshed.
[2026-09-12 23:11:22 UTC] Context study: live Q8 C32/P64k capacity test checked; latest recorded step Q8_0/c32/p32768 is complete; warmup and measured bursts continue.
[2026-09-12 23:11:36 UTC] Context study: Q8 C32/P64k warmup remains active after startup placement/headroom passed; measured bursts are pending.
[2026-09-12 23:13:32 UTC] Context study: live Q8 C32/P64k test checked; latest Q8_0/c32/p65536/r1 is running; awaiting completed capacity bursts.
[2026-09-12 23:13:42 UTC] Context study: Q8 C32/P64k warmup passed memory and no-swap checks; first measured burst is running.
[2026-09-12 23:15:32 UTC] Context study: live Q8 C32/P64k driver checked; latest Q8_0/c32/p65536/r1 is running; completed request checks remain pending.
[2026-09-12 23:16:35 UTC] Context study: first Q8 C32/P64k burst remains active; awaiting completed result before accepting the repetition.
[2026-09-12 23:17:40 UTC] Context study: first Q8 C32/P64k measured burst passed; latest Q8_0/c32/p65536/r2 is running; remaining capacity bursts continue.
[2026-09-12 23:19:42 UTC] Context study: live Q8 C32/P64k driver checked; latest Q8_0/c32/p65536/r2 is running; completed request and memory checks remain pending for the active burst.
[2026-09-12 23:20:42 UTC] Context study: second Q8 C32/P64k burst remains active; first burst passed and the next completed result is pending.
[2026-09-12 23:21:47 UTC] Context study: second Q8 C32/P64k measured burst passed; latest Q8_0/c32/p65536/r3 is running; final capacity burst continues.
[2026-09-12 23:23:53 UTC] Context study: live Q8 C32/P64k driver checked; latest Q8_0/c32/p65536/r3 is running; final acceptance will complete Q8 capacity tests at C32.
[2026-09-12 23:24:54 UTC] Context study: final Q8 C32/P64k burst remains active; first two passed and the capacity-table update awaits the last result.
[2026-09-12 23:26:03 UTC] Context study: Q8 C32/P64k passed all three bursts; 41 settings complete; latest Q8_0/c32/p65536 is complete; capacity summaries refreshed.
[2026-09-12 23:27:33 UTC] Context study: first Q8 C64/P2k repetition passed; latest Q8_0/c64/p2048/r2 is running; remaining repetitions continue.
[2026-09-12 23:28:31 UTC] Context study: second Q8 C64/P2k repetition passed; final repetition continues before summary inclusion.
[2026-09-12 23:33:35 UTC] Context study: fresh matched Nsight timelines and hardware counters will be collected for new workloads; inline GPU telemetry accompanies serving, and old cuda-study captures remain reference evidence.
[2026-09-12 23:35:06 UTC] Context study: 1B Q8 at 64 clients and 4k input completed all three repetitions; serving continues before fresh hardware profiling.
[2026-09-12 23:36:28 UTC] Context study: monitoring the live serving sweep; refresh summaries at completed settings, with hardware profiling queued afterward.
[2026-09-12 23:37:46 UTC] Context study: 1B Q8 is testing 64 clients at 8k; completed matched results show its FP16 throughput advantage narrowing with longer prompts and higher concurrency, with hardware explanation pending.
[2026-09-12 23:38:54 UTC] Context study: monitoring remaining 1B Q8 repetitions at 64 clients and 8k before the 16k setting.
[2026-09-12 23:39:58 UTC] Context study: 1B Q8/64 clients/8k second repetition remains active; server VmSwap is zero, and the setting awaits all three repetitions before summary inclusion.
[2026-09-12 23:40:52 UTC] Context study: 1B Q8 at 64 clients and 8k has completed two repetitions; the serving sweep remains live.
[2026-09-12 23:41:11 UTC] Context study: checking the final 1B Q8/64 clients/8k repetition before refreshing the summary.
[2026-09-12 23:42:20 UTC] Context study: 1B Q8 at 64 clients and 8k passed all three repetitions; refreshing summary before the 16k setting.
[2026-09-12 23:42:40 UTC] Context study: summary now includes 1B Q8/64 clients/8k at 800.09 output tok/s across 384 successful requests; no sampled thermal limiting on either GPU.
[2026-09-12 23:42:59 UTC] Context study: checking 1B Q8 at 64 clients and 16k while the serving sweep continues uninterrupted; separate profiles will match the completed workloads.
[2026-09-12 23:44:05 UTC] Context study: 1B Q8/64 clients/16k warmup finished and first measured repetition is starting; exact 512-output and no-truncation checks remain enforced.
[2026-09-12 23:45:24 UTC] Context study: monitoring the first 1B Q8/64 clients/16k repetition and GPU-residency checks.
[2026-09-12 23:46:29 UTC] Context study: active 1B Q8/64 clients/16k server confirms 17/17 layers on GPU, FP16 KV allocations of 18.84 GiB and 14.66 GiB, and zero server swap; first repetition remains in progress.
[2026-09-12 23:46:50 UTC] Context study: continuing to monitor 1B Q8/64 clients/16k; reported numbers will update only after completed validation.
[2026-09-12 23:47:51 UTC] Context study: first 1B Q8/64 clients/16k repetition passed in about 212 seconds; two remain, approximately seven minutes at that pace, followed by larger power-of-two capacity checks.
[2026-09-12 23:48:48 UTC] Context study: 1B Q8/64 clients/16k second repetition remains active; no newly completed setting for summary inclusion.
[2026-09-12 23:49:47 UTC] Context study: ongoing 1B Q8/64 clients/16k repetition processes 128 requests, each with 16384 input and 512 output tokens.
[2026-09-12 23:50:58 UTC] Context study: active 1B Q8/64 clients/16k sample uses 20342/16162 MiB VRAM with 28198/32380 MiB free; neither GPU reports software or hardware thermal limiting.
[2026-09-12 23:51:55 UTC] Context study: second 1B Q8/64 clients/16k repetition passed; final repetition is running before completion of the Q8 main serving grid.
[2026-09-12 23:53:00 UTC] Context study: final 1B Q8/64 clients/16k repetition remains active; comparison tables will refresh on completion before the 32k capacity test.
[2026-09-12 23:53:56 UTC] Context study: final 1B Q8/64 clients/16k repetition is still running without a reported failure.
[2026-09-12 23:54:53 UTC] Context study: all 16 main 1B Q8 serving settings passed; refreshing tables and plots while remaining capacity checks proceed.
[2026-09-12 23:56:15 UTC] Context study: completed 1B Q8/FP16 at 64 clients and 16k are close (465.95/468.45 output tok/s); checking recorded thermal limiting and correcting plot input-axis spacing.
[2026-09-12 23:58:16 UTC] Context study: plot axes corrected and visually checked; GPU0 software thermal limiting affected 5.97%/11.26% of measured FP16/Q8 samples at 64 clients and 16k, so the 0.53% throughput gap is treated as near parity under differing thermal exposure.
[2026-09-12 23:58:41 UTC] Context study: FINDINGS.md now includes a clearly marked interim 1B FP16/Q8 comparison with measured thermal-exposure evidence; complete hardware interpretation remains pending.
[2026-09-12 23:59:05 UTC] Context study: monitoring 1B Q8 capacity bursts at 64 clients and 32k to validate sustained operation with model and KV on both GPUs.
[2026-09-13 00:00:15 UTC] Context study: active 1B Q8/64 clients/32k has 17/17 layers and KV on GPU, with 9702/17980 MiB free in the current sample; first capacity burst continues.
[2026-09-13 00:01:25 UTC] Context study: 32 main serving and 13 capacity settings completed for 1B FP16/Q8; remaining Q8 capacity bursts are active before Q4 and IQ1.
[2026-09-13 00:02:20 UTC] Context study: first 1B Q8/64 clients/32k capacity burst passed; second burst is running, with three required before declaring the setting validated.
[2026-09-13 00:02:40 UTC] Context study: following remaining 1B Q8/64 clients/32k bursts before updating maximum-input results and proceeding to Q4.
[2026-09-13 00:03:56 UTC] Context study: second 1B Q8/64 clients/32k burst remains active; logged FP16 KV allocations total 65.5 GiB across both GPUs.
[2026-09-13 00:04:52 UTC] Context study: second 1B Q8/64 clients/32k capacity burst passed; one final burst remains before validating this setting.
[2026-09-13 00:05:57 UTC] Context study: final 1B Q8/64 clients/32k burst continues; capacity measurements remain separate from main serving comparisons.
[2026-09-13 00:06:57 UTC] Context study: runner remains active on the final 1B Q8/64 clients/32k burst; no new completed burst yet.
[2026-09-13 00:08:13 UTC] Context study: server log shows 44/64 requests completed in the final 1B Q8/64 clients/32k burst; waiting for the remainder and runner validation.
[2026-09-13 00:09:17 UTC] Context study: 1B Q8 capacity resolved at 64k for 8/16/32 clients and 32k for 64 clients; 64-client/64k excluded by per-GPU memory estimate; Q4/8 clients/2k passed.
[2026-09-13 00:10:05 UTC] Context study: FINDINGS.md now records validated 1B FP16/Q8 maximum inputs (64k at 8/16/32 clients, 32k at 64) and the 129.5 GiB FP16 KV estimate excluding 64-client/64k.
[2026-09-13 00:10:35 UTC] Context study: validated FP16/Q8 context limits are in FINDINGS.md; Q4 passed 2k and 4k at eight clients and is measuring 8k.
[2026-09-13 00:11:05 UTC] Context study: checking the next completed Q4 settings for inclusion in the current comparison table.
[2026-09-13 00:12:10 UTC] Context study: 1B Q4/8 clients/8k passed all repetitions; first 16k repetition passed with two remaining.
[2026-09-13 00:13:43 UTC] Context study: 1B Q4 passed all four main prompt lengths at eight clients; it uses less VRAM but has 1.5–2% lower output throughput than Q8, with no sampled thermal limiting in either format at these settings; kernel explanation awaits fresh profiles.
[2026-09-13 00:14:39 UTC] Context study: first two 1B Q4/8 clients/32k capacity bursts passed; final burst is running.
[2026-09-13 00:15:35 UTC] Context study: 1B Q4/8 clients/32k passed all three capacity bursts; refreshing summaries as the runner proceeds to 64k.
[2026-09-13 00:16:00 UTC] Context study: monitoring 1B Q4/8 clients/64k capacity validation before higher-concurrency settings.
[2026-09-13 00:17:04 UTC] Context study: first 1B Q4/8 clients/64k burst is active; exact 512-output and memory checks remain required for passing status.
[2026-09-13 00:17:59 UTC] Context study: first 1B Q4/8 clients/64k capacity burst passed; second is running with three total required.
[2026-09-13 00:18:57 UTC] Context study: second 1B Q4/8 clients/64k capacity burst passed; final burst is running before the 16-client sweep.
[2026-09-13 00:20:01 UTC] Context study: 1B Q4 validated 64k at eight clients, completing all six lengths at that concurrency; 16-client sweep started and capacity table is refreshing.
[2026-09-13 00:20:25 UTC] Context study: 1B Q4 also passed 16 clients at 2k; 53 settings are now complete, with remaining serving/capacity and Nsight profiling still pending.
[2026-09-13 00:20:59 UTC] Context study: monitoring Q4 at 16 clients and updating summaries as settings complete.
[2026-09-13 00:22:12 UTC] Context study: 1B Q4/16 clients/4k passed and summary refreshed; 8k measurements are active.
[2026-09-13 00:23:26 UTC] Context study: first two 1B Q4/16 clients/8k repetitions passed; final repetition is active before 16k.
[2026-09-13 00:24:46 UTC] Context study: 1B Q4/16 clients/8k passed; 16k is active, and summary now covers 55 completed settings.
[2026-09-13 00:25:06 UTC] Context study: monitoring 1B Q4/16 clients/16k repetitions before refreshing the completed-setting results.
[2026-09-13 00:26:21 UTC] Context study: first 1B Q4/16 clients/16k repetition passed; second is active and third remains required.
[2026-09-13 00:27:31 UTC] Context study: 1B Q4/16 clients/16k passed all three repetitions; main grids for 8 and 16 clients are complete, followed by 16-client capacity extensions.
[2026-09-13 00:27:58 UTC] Context study: completed 1B Q4/16 clients/16k measures 409.48 output tok/s with 37.15 ms p95 time per output token; summary contains 56 completed settings.
[2026-09-13 00:28:16 UTC] Context study: monitoring 1B Q4 capacity extensions at 16 clients and adding only fully validated lengths to the capacity table.
[2026-09-13 00:29:22 UTC] Context study: first 1B Q4/16 clients/32k capacity burst passed; second is active.
[2026-09-13 00:30:28 UTC] Context study: second 1B Q4/16 clients/32k capacity burst passed; final burst is active before 64k.
[2026-09-13 00:31:33 UTC] Context study: 1B Q4/16 clients/32k passed all three capacity bursts; summary refreshed to 57 completed settings as the runner proceeds to 64k.
[2026-09-13 00:31:51 UTC] Context study: checking 1B Q4/16 clients/64k capacity testing and live runner progress.
[2026-09-13 00:33:09 UTC] Context study: 1B Q4/16 clients/64k warmup finished; three measured bursts of 16 requests are next.
[2026-09-13 00:34:12 UTC] Context study: first measured 1B Q4/16 clients/64k burst is active; no new completed setting yet.
[2026-09-13 00:34:30 UTC] Context study: monitoring 1B Q4/16 clients/64k bursts for completed validation.
[2026-09-13 00:35:57 UTC] Context study: first 1B Q4/16 clients/64k capacity burst passed in about 121 seconds; second is running.
[2026-09-13 00:37:05 UTC] Context study: second 1B Q4/16 clients/64k burst remains active; no additional completed setting yet.
[2026-09-13 00:38:09 UTC] Context study: second 1B Q4/16 clients/64k capacity burst passed; final burst is active before the 32-client sweep.
[2026-09-13 00:39:17 UTC] Context study: final 1B Q4/16 clients/64k burst remains active; maximum-input table will update after passing validation.
[2026-09-13 00:40:24 UTC] Context study: 1B Q4 validated 64k at 16 clients; 32-client sweep started and report refreshed to 58 completed settings.
[2026-09-13 00:40:46 UTC] Context study: monitoring the 1B Q4 32-client sweep for newly completed settings.
[2026-09-13 00:41:53 UTC] Context study: 1B Q4/32 clients/2k passed and summary refreshed to 59 completed settings; 4k is active.
[2026-09-13 00:43:04 UTC] Context study: first 1B Q4/32 clients/4k repetition passed; remaining repetitions continue.
[2026-09-13 00:44:04 UTC] Context study: 1B Q4/32 clients/4k passed all three repetitions; summary refreshed to 60 completed settings as the runner proceeds to 8k.
[2026-09-13 00:44:43 UTC] Context study: checking 1B Q4/32 clients/8k serving measurements; separate Nsight captures remain pending.
[2026-09-13 00:45:49 UTC] Context study: first 1B Q4/32 clients/8k repetition passed; second is active.
[2026-09-13 00:46:56 UTC] Context study: second 1B Q4/32 clients/8k repetition passed; final repetition is active before 16k.
[2026-09-13 00:48:04 UTC] Context study: 1B Q4/32 clients/8k passed all three repetitions; summary refreshed to 61 completed settings before the 16k test.
[2026-09-13 00:48:22 UTC] Context study: checking 1B Q4/32 clients/16k while the serving sweep continues uninterrupted.
[2026-09-13 00:49:42 UTC] Context study: first 1B Q4/32 clients/16k repetition remains active; no new completed setting for summary inclusion yet.
[2026-09-13 00:50:43 UTC] Context study: first 1B Q4/32 clients/16k repetition passed in about 112 seconds; second is running.
[2026-09-13 00:51:04 UTC] Context study: monitoring remaining 1B Q4/32 clients/16k repetitions before the next summary refresh.
[2026-09-13 00:52:15 UTC] Context study: second 1B Q4/32 clients/16k repetition remains active; no new completed setting yet.
[2026-09-13 00:53:15 UTC] Context study: second 1B Q4/32 clients/16k repetition passed; final repetition is active before capacity extensions.
[2026-09-13 00:54:22 UTC] Context study: 1B Q4 main serving grid passed at 32 clients; refreshing report before 32k/64k capacity extensions.
[2026-09-13 00:54:34 UTC] Context study: summary refreshed to 62 completed settings; Q4 main grids are complete at 8/16/32 clients, with capacity extensions and 64-client tests still ahead.
[2026-09-13 00:54:54 UTC] Context study: checking 1B Q4/32 clients/32k capacity testing as the scheduled sweep continues.
[2026-09-13 00:56:21 UTC] Context study: 1B Q4/32 clients/32k warmup finished with zero server swap; measured capacity bursts are next.
[2026-09-13 00:57:23 UTC] Context study: first measured 1B Q4/32 clients/32k capacity burst is active; full validation remains pending.
[2026-09-13 00:57:41 UTC] Context study: checking 1B Q4/32 clients/32k burst completion; capacity remains pending until all three pass.
[2026-09-13 00:58:44 UTC] Context study: first 1B Q4/32 clients/32k capacity burst passed; second is active.
[2026-09-13 00:59:52 UTC] Context study: second 1B Q4/32 clients/32k capacity burst passed; final burst is active before 64k.
[2026-09-13 01:01:00 UTC] Context study: final 1B Q4/32 clients/32k burst remains active; capacity table will update after runner validation.
[2026-09-13 01:02:01 UTC] Context study: 1B Q4/32 clients/32k passed all three capacity bursts; summary refreshed to 63 completed settings as the runner proceeds to 64k.
[2026-09-13 01:02:28 UTC] Context study: checking 1B Q4/32 clients/64k, the final input length for this concurrency before 64-client tests.
[2026-09-13 01:04:05 UTC] Context study: 1B Q4/32 clients/64k warmup is still processing prompts; measured bursts have not started yet.
[2026-09-13 01:05:29 UTC] Context study: checking startup of measured 1B Q4/32 clients/64k bursts and live runner state.
[2026-09-13 01:07:17 UTC] Context study: first measured 1B Q4/32 clients/64k burst is active with 17/17 layers and KV on GPU, 10244/18436 MiB currently free, and zero server swap.
[2026-09-13 01:08:32 UTC] Context study: continuing to monitor 1B Q4/32 clients/64k capacity bursts for the first completed result.
[2026-09-13 01:09:47 UTC] Context study: first 1B Q4/32 clients/64k burst remains active; each burst processes 2097152 input tokens across 32 requests.
[2026-09-13 01:10:49 UTC] Context study: first 1B Q4/32 clients/64k capacity burst passed in about 249 seconds; second is running with two bursts still required.
[2026-09-13 01:11:07 UTC] Context study: monitoring remaining 1B Q4/32 clients/64k bursts; the observed pace implies several more minutes before capacity validation completes.
[2026-09-13 01:13:08 UTC] Context study: continuing verified waits on the live runner for the next completed 1B Q4/32 clients/64k burst.
[2026-09-13 06:17:02 UTC] Context study: prior observation handle is unavailable after interruption; checking OS processes and recorded progress before taking action.
[2026-09-13 06:18:39 UTC] Context study: old runner stopped after interruption and GPUs are idle; saved Q4/32 clients/64k passed plus Q4/64 clients/2k r1; auditing resume checkpoint while completing the deferred private profiling build.
[2026-09-13 06:21:20 UTC] Context study: resume checkpoint validation PASS; serving source/binary unchanged and saved Q4/64 clients/2k r1 passes token/resource/placement checks; bounded real profiling check is queued before serving resumes.
[2026-09-13 06:23:44 UTC] Context study: v2 profiling build PASS and bounded real Nsight capture started; serving resume checks PASS; preparing a persistent restart after profiling calibration ends.
[2026-09-13 06:24:36 UTC] Status: 64 completed 1B settings plus 2 capacity exclusions; FP16/Q8 finished, Q4 64-client tests partial, IQ1/8B/70B serving pending; first fresh Nsight timeline passed, counter check underway before serving resume; remaining matched profiles, Q2/Q4 diagnostics and final FINDINGS analysis still required.
[2026-09-13 06:26:30 UTC] Hardware status: first fresh matched kernel timeline completed on both GPUs; checking the active hardware-counter capture.
[2026-09-13 06:27:45 UTC] Hardware status: kernel timing collection started; first trace retained, attention grouping corrected, replacement 1B Q4/8 clients/2k timeline is active; no hardware-counter captures completed yet.
[2026-09-13 06:30:14 UTC] Context study: background-service restart is prepared to survive status-session disconnection; waiting for bounded profiler calibration to release both GPUs before launch.
[2026-09-13 06:33:10 UTC] Hardware status: first prefill/GPU0 counter capture PASS with 20 matched launches and all requested metrics/sections; saved report parsed after reader fix without rerunning GPU work; remaining phase/device captures follow.
[2026-09-13 06:38:23 UTC] Context study: ordinary serving sweep remains paused while isolated Nsight calibration profiles its own inference workload; replayed profiler timings are excluded from headline latency/throughput, and no ordinary serving runner was present in the live process check.
[2026-09-13 06:39:15 UTC] Context study: isolated calibration finished with valid prefill/decode counters on both GPUs; confirmed no compute processes before resuming canonical serving sweep as cloud-serving-context-study.service, preserving completed Q4 C64 P2k r1; full profiler endpoint still needs operation supplements and captures 2–3.
[2026-09-13 06:40:57 UTC] Context study: service startup failed before model load because libcudart.so.13 was outside its library path; corrected the resume wrapper to use the same CUDA runtime directory as the original shell, with completed benchmark repetitions preserved.
[2026-09-13 06:42:43 UTC] Context study: clarified that inference is incomplete (1B FP16/Q8 complete, Q4 partial, IQ1_M pending; 8B/70B pending) and hardware evidence currently covers only the 1B Q4 C8 P2k calibration; final findings must explain measured mechanisms and ranking changes, with Q2/Q4 causality still unresolved.
[2026-09-13 06:43:16 UTC] Context study: CUDA library-path correction succeeded; original pinned server PID 289342 is running inference on both GPUs with zero server swap, and completed Q4 C64 P2k r1 remains unchanged while pending repetitions resume.
[2026-09-13 06:51:13 UTC] Context study: user amended scope to one measured run/capture per case and IQ1_M/Q2_K/Q4_K_M/Q8_0/feasible FP16 for all three sizes; stopped serving after Q4 C64 P4k r1 completed, preserving original results while updating manifests and report selection.
[2026-09-13 06:58:10 UTC] Context study: full timeline/counter profiling follows all inference/capacity runs; Q2_K for 1B/8B is now verified, 70B Q8 parts are downloading after the weight-capacity screen, and 66 existing first runs were retained under a verified compatible protocol amendment with no raw-result changes.
[2026-09-13 07:07:47 UTC] Context study: one-run reporting and compatible-result reuse checks pass (33 tests); Q2_K tokenizer checks pass for 1B/8B, public plans select all five formats with one run, and refreshed reports contain 46 serving settings plus 20 capacity extensions using r1 only with no run-to-run error bars.
[2026-09-13 07:11:16 UTC] Context study: clarified execution order as all inference/capacity runs first, then selected timeline/counter runs; 70B Q8 download completed and verified-part assembly is active, with inference intentionally paused until model preparation finishes.
[2026-09-13 07:18:44 UTC] Context study: original profiler identities reproduced using the explicitly pinned Nsight executable paths; all five saved timeline/counter captures remain reusable after the one-run amendment, with original producer identities retained and no GPU reruns.
[2026-09-13 07:21:33 UTC] Context study: 70B Q8 assembly continues in live preparation process 292005; inference remains paused during disk-intensive preparation, and the existing profiler calibration is preserved for the later counter sweep.
[2026-09-13 07:26:32 UTC] Context study: 70B Q8 is verified (SHA256 15c72f81ed7ab6d0b98b1261dd91fff22e5d9be97d34810c8afdc8da8113b6af) with matching Q4 tokenizer/template; removed redundant verified download parts and resumed the one-run serving sweep after confirming idle GPUs.
[2026-09-13 07:30:14 UTC] Context study: resumed service accepted the amended manifest fingerprint and skipped preserved first runs; pending 1B Q4 cases are continuing with one run per setting, followed by IQ1_M/Q2_K and the 8B/70B sweeps.
[2026-09-13 07:34:44 UTC] Context study: Q4 C64 P8k completed with one validated run of 128 requests (8192 input and 512 output tokens each); Q4 C64 P16k is running, bringing completed settings to 67 while reports refresh.
[2026-09-13 07:43:23 UTC] Context study: hardware counters will follow the full inference/capacity sweep in separate matching diagnostic reruns with ordinary inference stopped; 1B Q4 C64 P16k completed (68 total settings), and C64 P32k is running.
[2026-09-13 07:44:09 UTC] Context study: 1B Q4 sweep completed, including C64 P32k; FP16/Q8/Q4 now provide 69 measured settings and three predicted C64 P64k capacity exclusions; IQ1_M is next, and no hardware profiler is running.
[2026-09-13 07:47:31 UTC] Context study: IQ1_M passed the 2k–16k C8 settings; FINDINGS.md now includes completed 1B Q4 throughput/capacity and measured thermal exposure, with kernel explanations still pending.
[2026-09-13 07:50:19 UTC] Context study: 1B IQ1_M passed C8 through 64k and is running C16; 70B dry-plan validation PASS for all five formats, one measured run and 512 output tokens, with FP16 excluded on this device.
[2026-09-13 07:53:26 UTC] Context study: 1B IQ1_M completed all four main input lengths at C16 without failures; capacity extensions are next, followed by C32.
[2026-09-13 07:54:58 UTC] Context study: IQ1_M C16 P32k is running; refreshing the report with completed IQ1_M settings, while pending runs remain excluded.
[2026-09-13 07:56:31 UTC] Context study: IQ1_M C16 P32k passed; report refreshed to 80 completed settings; C16 P64k is preparing.
[2026-09-13 07:57:44 UTC] Context study: IQ1_M C16 P64k measured run started; C32 follows after this capacity check.
[2026-09-13 07:59:24 UTC] Context study: IQ1_M C16 P64k requests are finishing with 512 output tokens and no truncation in completed server records; full-burst acceptance remains pending.
[2026-09-13 08:01:17 UTC] Context study: 1B IQ1_M passed C16 through 64k and C32 P2k; remaining IQ1_M C32/C64 settings precede Q2_K.
[2026-09-13 08:02:17 UTC] Context study: IQ1_M C32 P4k passed and P8k is preparing; no new failed settings or capacity exclusions.
[2026-09-13 08:04:54 UTC] Context study: FINDINGS.md records the 1B P2k C8→C16 reversal (IQ1_M −9.98%, Q4 +24.10% output throughput); neither GPU recorded thermal limiting in these four measured windows, and the kernel cause remains pending.
[2026-09-13 08:05:20 UTC] Context study: 84 settings completed; IQ1_M C32 P16k is running, and the scaling-reversal evidence is linked in FINDINGS.md.
[2026-09-13 08:06:23 UTC] Context study: live service and server log confirm IQ1_M C32 P16k is advancing through generation; its single measured run remains in progress.
[2026-09-13 08:08:10 UTC] Context study: IQ1_M completed C32 main settings through 16k (85 settings overall); refreshing plots while C32 capacity extensions proceed.
[2026-09-13 08:09:15 UTC] Context study: plots refreshed and inspected at 85 completed 1B settings; IQ1_M C32 P32k is running, while 8B/70B remain queued.
[2026-09-13 08:10:50 UTC] Context study: IQ1_M C32 P32k passed (86 completed settings); C32 P64k is preparing, and the capacity tables are being refreshed.
[2026-09-13 08:13:30 UTC] Context study: live server logs confirm IQ1_M C32 P64k warmup is advancing through prompt processing and generation; the measured burst has not started.
[2026-09-13 08:14:55 UTC] Context study: IQ1_M C32 P64k warmup telemetry shows about 10.1/18.1 GiB free on GPUs 0/1 and zero server swap; measured capacity acceptance remains pending.
[2026-09-13 08:16:51 UTC] Context study: IQ1_M C32 P64k measured burst is running; GPU process isolation PASS, with only benchmark server PID 310407 on both GPUs.
[2026-09-13 08:17:47 UTC] Context study: IQ1_M C32 P64k remains active; acceptance awaits all 32 exact-512-output requests and the memory checks.
[2026-09-13 08:19:40 UTC] Context study: IQ1_M C32 P64k server logs show interleaved prompt processing and generation; later profiles will separate prefill, decode and mixed batches using actual metadata.
[2026-09-13 08:20:22 UTC] Context study: IQ1_M passed C32 P64k (87 completed settings), establishing 64k at C8/C16/C32; C64 settings follow.
[2026-09-13 08:23:42 UTC] Context study: IQ1_M C64 P2k/P4k passed (89 completed settings); refreshing tables and plots while P8k starts.
[2026-09-13 08:25:19 UTC] Context study: IQ1_M C64 P8k measured run is underway; P16k and the P32k capacity check remain before Q2_K.
[2026-09-13 08:28:18 UTC] Context study: IQ1_M C64 P8k passed (90 settings); method docs now explicitly distinguish C-request diagnostic bursts from 2C-request serving runs and record actual per-GPU placement.
[2026-09-13 08:28:52 UTC] Context study: 90 settings completed; IQ1_M C64 P16k is preparing, with profiling-versus-serving workload distinctions documented.
[2026-09-13 08:30:41 UTC] Context study: IQ1_M C64 P16k measured run started; P32k capacity validation and the P64k memory screen remain for this format.
[2026-09-13 08:31:49 UTC] Context study: IQ1_M C64 P16k remains active; the report row awaits complete request and memory records.
[2026-09-13 08:34:32 UTC] Context study: IQ1_M completed all 16 main 1B inference settings (91 completed settings overall); final C64 P32k capacity check is next, and report plots are refreshing.
[2026-09-13 08:36:50 UTC] Context study: FINDINGS.md now includes IQ1_M C64 P16k at 420.07 output tok/s and 134/778 (17.22%) GPU0 software thermal-limit samples; the kernel interpretation remains pending.
[2026-09-13 08:40:38 UTC] Context study: report generator now includes per-model/format inference completion and separate hardware profiling status; IQ1_M final C64 P32k run remains active.
[2026-09-13 08:42:26 UTC] Context study: IQ1_M inference/capacity sweep complete; Q2_K passed C8 P2k/P4k and is running P8k; report validation PASS (22 checks), with inference and profiling status separated.
[2026-09-13 08:46:17 UTC] Context study: Q2_K passed all C8 main prompts and P32k (97 completed settings); P64k is running, and the report is refreshing with Q2_K data.
[2026-09-13 08:48:19 UTC] Context study: Q2_K passed C8 through 64k and C16 P2k/P4k (100 completed settings); 8B/70B serving remains queued.
[2026-09-13 08:49:01 UTC] Context study: live Q2_K sweep has passed C16 P8k and is measuring P16k (101 completed settings); profiling remains queued until all three models finish inference.
[2026-09-13 08:50:25 UTC] Context study: Q2_K completed C16 main inputs through 16k (102 settings); reviewing matched Q2/Q4 serving rates while capacity extensions proceed.
[2026-09-13 08:53:07 UTC] Context study: FINDINGS.md records Q2/Q4 ranking reversals with concurrency and prompt length (P2k: −7.42% at C8, +3.20% at C16; C16 P4k: −4.52%); no thermal limiting was recorded in these runs, and kernel attribution remains pending.
[2026-09-13 08:56:07 UTC] Context study: Q2_K C16 P64k is measuring; refreshing tables and plots with completed C16 results and the observed Q2/Q4 ranking change.
[2026-09-13 08:59:06 UTC] Context study: Q2_K passed C16 through 64k and C32 P2k/P4k (106 settings); C32 P8k is running.
[2026-09-13 09:01:47 UTC] Context study: Q2_K C32 P8k passed (107 settings), and P16k is measuring; C32 capacity extensions and C64 settings remain for 1B.
[2026-09-13 09:05:14 UTC] Context study: Q2_K completed main inputs at C8/C16/C32 (108 settings); report refresh underway, with C32 capacity extensions and C64 settings remaining for 1B.
[2026-09-13 09:09:19 UTC] Context study: Q2_K C32 P32k passed (109 settings); P64k is preparing before C64, and the report/FINDINGS.md retain measured ranking reversals with kernel attribution pending.
[2026-09-13 09:11:15 UTC] Context study: live service is preparing Q2_K C32 P64k; 109 settings are complete, and the remaining 1B cases precede the automatic move to 8B.
[2026-09-13 09:12:29 UTC] Context study: Q2_K C32 P64k measured burst has started; the C64 sweep is the final inference work for 1B afterward.
[2026-09-13 09:13:51 UTC] Context study: live Q2_K C32 P64k measurement retains about 10.1/18.1 GiB free on GPUs 0/1 and zero server swap; full-burst acceptance remains pending.
[2026-09-13 09:22:36 UTC] Context study: all five 1B formats passed 64k at C8/C16/C32; Q2_K C64 P2k/P4k passed (112 settings), P8k is running, and the report is refreshing.
[2026-09-13 09:23:22 UTC] Context study: Q2_K C64 P8k passed (113 settings); P16k, P32k capacity validation and the P64k screen remain before 1B finishes.
[2026-09-13 09:24:30 UTC] Context study: Q2_K C64 P16k discarded warmup is advancing through prompt processing and generation; the measured run follows.
[2026-09-13 09:25:49 UTC] Context study: the last main 1B setting, Q2_K C64 P16k, is measuring; final Q2_K capacity checks precede the move to 8B.
[2026-09-13 09:27:02 UTC] Context study: Q2_K C64 P16k remains active; GPU process isolation PASS, with only benchmark server PID 328679 on both GPUs.
[2026-09-13 09:31:41 UTC] Context study: all 80 main 1B inference settings completed (114 measured settings including capacity extensions); Q2_K C64 P32k is preparing before the final memory screen and move to 8B.
[2026-09-13 09:32:45 UTC] Context study: FINDINGS.md includes all five completed 1B main formats, including Q2_K C64 P16k at 417.69 tok/s and 19.28% GPU0 thermal-limit samples; final C64 P32k burst has started.
[2026-09-13 09:34:30 UTC] Context study: final 1B capacity burst remains active at C64 P32k with 512 outputs/request; capacity-burst timing remains separate from main inference comparisons.
[2026-09-13 09:35:50 UTC] Context study: one measured capacity case remains for 1B; Q2_K C64 P32k is live, with final acceptance still pending.
[2026-09-13 09:38:22 UTC] Context study: 1B inference/capacity complete across all five formats (80 main + 35 capacity measurements, five predicted exclusions); maxima are 64k at C8/C16/C32 and 32k at C64; driver advanced to 8B, with full profiling still queued.
[2026-09-13 09:42:36 UTC] Context study: first 8B FP16 measurement passed at C8 P2k with 512 outputs/request; P4k is running, and the report is refreshing alongside completed 1B results.
[2026-09-13 09:45:21 UTC] Context study: 1B inference/capacity remains complete; 8B FP16 C8 P2k/P4k/P8k passed and P16k is running; 70B inference and full hardware profiling remain pending.
[2026-09-13 09:50:52 UTC] Context study: clarified execution order—all inference/capacity sweeps first, then separate profiler reruns of selected settings on otherwise idle GPUs; 1B inference complete, 8B has five completed FP16/c8 settings through 32k, 70B and the main hardware-counter sweep pending; earlier 1B Q4 profiling was calibration during an inference pause.
[2026-09-13 09:52:36 UTC] Context study: refreshed report with 8B FP16/c8 passes at 2k, 4k, 8k, 16k and 32k; 64k allocation and per-GPU reserve checks passed, discarded warmup running, request-level capacity validation pending; hardware profiling remains queued after inference.
[2026-09-13 09:54:04 UTC] Context study: 8B FP16/c8/p65536 discarded warmup completed and its single measured capacity burst is running; no hardware profiler is scheduled during serving.
[2026-09-13 09:55:31 UTC] Context study: live 8B FP16/c8/64k measured run continues with interleaved prompt processing and token generation; actual matrix widths remain pending fresh profiling.
[2026-09-13 09:57:51 UTC] Context study: 8B FP16/c8 capacity sweep complete through 65,536 input + 512 output tokens; all eight 64k requests passed, sampled server swap zero, minimum GPU headroom 5.34/9.26 GiB; report and FINDINGS.md updated, 16-client serving underway.
[2026-09-13 09:59:42 UTC] Context study: 8B FP16/c16/4k measurement running; saved completed-1B profiling selection with 60 unique settings across all five formats (2k/16k/64k at c8/c16/c32, 2k/16k/32k at c64); selection only, no GPU profiling executed.
[2026-09-13 10:01:17 UTC] Context study: 8B FP16/c16 passed 2k and 4k, single 8k measurement running; refreshing runtime/capacity report and plots, main hardware profiling still pending inference completion.
[2026-09-13 10:02:41 UTC] Context study: 8B FP16/c16/8k completed; report and plots refreshed through the preceding completed checkpoint, with the new 8k result available in its r1 directory; persistent inference service remains active.
[2026-09-13 10:04:37 UTC] Context study: 8B FP16/c16/16k single measured run underway after discarded warmup; GPU process check found only the serving process on both devices.
[2026-09-13 10:05:53 UTC] Context study: live 8B FP16/c16/16k measured workload progressing, with new requests admitted as earlier requests finish; publication awaits completed-run validation.
[2026-09-13 10:08:00 UTC] Context study: 8B FP16/c16 main grid complete through 16k; final run passed all 32 requests at exact 16,384+512 tokens, 100.32 output tok/s, zero sampled server swap and 22.13/24.09 GiB minimum GPU headroom; 32k capacity extension warming up, report refresh running.
[2026-09-13 10:08:28 UTC] Context study: report and plots refreshed with all 8B FP16 main settings at c8/c16; 1B inference complete, remaining serving/capacity sweeps and fresh hardware analysis still pending.
[2026-09-13 10:10:00 UTC] Context study: 8B FP16/c16/32k startup placement/headroom checks passed with 5.10/9.06 GiB GPU free memory; discarded warmup continues, measured capacity result pending.
[2026-09-13 10:11:11 UTC] Context study: 8B FP16/c16/32k discarded warmup completed; single 16-request measured capacity burst running, service active and hardware profiling queued after serving.
[2026-09-13 10:12:18 UTC] Context study: verified live service and advancing request processing for 8B FP16/c16/32k; capacity result remains pending final request completion and validation.
[2026-09-13 10:14:07 UTC] Context study: 8B FP16/c16 capacity sweep complete at 32k+512 with all 16 requests passing, zero sampled server swap and 5.07/9.03 GiB minimum GPU headroom; 64k excluded before launch because KV alone needs 129.5 GiB; FINDINGS.md capacity table updated, c32 serving underway.
[2026-09-13 10:16:41 UTC] Context study: 8B FP16/c32/2k passed all 64 requests at 471.42 output tok/s with valid placement, memory and swap checks; c32/4k warmup underway, next report refresh follows its completed measurement.
[2026-09-13 10:17:44 UTC] Context study: 8B FP16/c32/4k measured run advancing through the second request wave; observed completions show 512 output tokens without truncation, full-run validation pending.
[2026-09-13 10:19:26 UTC] Context study: 8B FP16/c32/4k passed all 64 requests with exact 4,096+512 tokens at 328.23 output tok/s, zero sampled server swap and valid GPU placement/headroom; c32/8k warmup underway, report/plots refresh running.
[2026-09-13 10:20:05 UTC] Context study: report and plots refreshed through 8B FP16/c32/4k; c32/8k single measured run has started and the persistent inference service remains active; remaining serving and hardware analysis pending.
[2026-09-13 10:21:48 UTC] Context study: verified live 8B FP16/c32/8k measured run progressing through 64 requests; complete-run publication and hardware analysis remain pending.
[2026-09-13 10:23:58 UTC] Context study: 8B FP16/c32/8k passed all 64 requests at exact 8,192+512 tokens and 199.34 output tok/s, with zero sampled server swap and 21.36/23.42 GiB minimum GPU headroom; c32/16k startup checks passed, report refresh running.
[2026-09-13 10:24:57 UTC] Context study: report and plots refreshed through 8B FP16/c32/8k; persistent service active on the 16k setting, remaining serving/capacity sweeps and hardware analysis pending.
[2026-09-13 10:26:19 UTC] Context study: 8B FP16/c32/16k warmup finished and single 64-request measured run started; startup placement/headroom checks passed with 4.36/8.42 GiB GPU free memory, full-run validation pending.
[2026-09-13 10:28:11 UTC] Context study: verified active service measuring 8B FP16/c32/16k; 1B inference complete, remaining 8B formats and 70B serving not yet complete, hardware profiling queued after serving.
[2026-09-13 10:29:21 UTC] Context study: active 8B FP16/c32/16k measured run continues admitting later requests as slots free up; full 64-request result pending.
[2026-09-13 10:31:35 UTC] Context study: live 8B FP16/c32/16k server log shows 42 of 64 measured requests finished, excluding 32 discarded warmup requests; all measured requests admitted, remaining outputs in progress.
[2026-09-13 10:33:30 UTC] Context study: 8B FP16/c32 capacity sweep complete at 16k+512, all 64 main-run requests passed at 108.96 output tok/s with zero sampled server swap and 4.33/8.39 GiB GPU headroom; 32k/64k excluded before launch because KV alone needs 131/259 GiB; FINDINGS.md updated.
[2026-09-13 10:34:03 UTC] Context study: report/plots refreshed with 8B FP16 validated maxima of 64k/32k/16k at c8/c16/c32; c64/2k single measured run active, remaining 8B formats, 70B serving and hardware analysis pending.
[2026-09-13 10:36:31 UTC] Context study: 8B FP16/c64/2k passed all 128 requests with exact 2,048+512 tokens at 570.20 output tok/s, zero sampled server swap and 28.29/29.54 GiB minimum GPU headroom; c64/4k warmup underway, report refresh running.
[2026-09-13 10:37:17 UTC] Context study: report and plots refreshed through 8B FP16/c64/2k; service continues on the 4k setting, remaining serving/capacity sweeps and hardware analysis pending.
[2026-09-13 10:38:51 UTC] Context study: verified active 8B FP16/c64/4k run with advancing generation counters; all 128 measured requests must finish before publication.
[2026-09-13 10:39:40 UTC] Context study: 8B FP16/c64/4k single measured run progressing through its second request wave; completed 128-request result pending.
[2026-09-13 10:41:02 UTC] Context study: 8B FP16/c64/4k passed all 128 requests at exact 4,096+512 tokens and 378.77 output tok/s with zero sampled server swap; c64/8k startup placement passed with 2.80/7.05 GiB free, sustained 2 GiB reserve validation pending.
[2026-09-13 10:41:55 UTC] Context study: report and plots refreshed through 8B FP16/c64/4k; service active on the 8k setting, remaining 8B formats, 70B serving and hardware analysis pending.
[2026-09-13 10:43:27 UTC] Context study: 8B FP16/c64/8k warmup completed and single 128-request measured run started; sustained 2 GiB per-GPU reserve and exact-output validation remain active.
[2026-09-13 10:44:47 UTC] Context study: live 8B FP16/c64/8k measurement progressing; discarded warmup retained at least 2.77 GiB GPU0 headroom, full measured-run capacity validation pending.
[2026-09-13 10:45:54 UTC] Context study: active 8B FP16/c64/8k measurement continues generating its first request wave; hardware captures remain queued after serving completion.
[2026-09-13 10:46:58 UTC] Context study: live 8B FP16/c64/8k log shows 35 of 128 measured requests finished and 98 admitted, excluding 64 discarded warmup requests; full-run validation pending.
[2026-09-13 10:48:26 UTC] Context study: all 128 measured requests admitted for 8B FP16/c64/8k, 66 finished in the live server log; remaining outputs and final capacity validation pending.
[2026-09-13 10:50:15 UTC] Context study: 8B FP16 inference/capacity complete with 18 measurements and six predicted memory exclusions; final c64/8k run passed all 128 requests at 222.72 output tok/s, zero sampled server swap and 2.77/7.01 GiB minimum GPU headroom; FP16 maxima 64k/32k/16k/8k at c8/c16/c32/c64, FINDINGS.md updated, Q8_0 serving started.
[2026-09-13 10:51:35 UTC] Context study: FP16/8B completion report refreshed; Q8_0/c8 passed 2k and 4k, input matching to FP16 and exact 512-output checks passed; refreshing new Q8 rows, remaining serving and hardware analysis pending.
[2026-09-13 10:52:51 UTC] Context study: report/plots refreshed with complete 8B FP16 results and first Q8 measurements; live Q8_0/c8 sweep has now passed 2k, 4k and 8k, remaining 8B formats, 70B serving and hardware analysis pending.
[2026-09-13 10:54:45 UTC] Context study: active 8B Q8_0/c8/16k measured run generating responses; throughput and memory publication await all 16 measured requests and final validation.
[2026-09-13 10:56:50 UTC] Context study: 8B Q8_0/c8 main grid complete through 16k; final run passed 16 requests at 96.79 output tok/s, exact 512 outputs, inputs matching FP16 and zero sampled server swap; 32k capacity extension underway, report refresh running.
[2026-09-13 10:57:21 UTC] Context study: report/plots refreshed with all 8B Q8_0/c8 main settings through 16k; persistent service continues the capacity extension, remaining serving and hardware analysis pending.
[2026-09-13 10:59:51 UTC] Context study: 8B Q8_0/c8/32k capacity burst passed all eight requests at 48.20 output tok/s with exact 512 outputs, inputs matching FP16 and zero sampled server swap; 64k startup checks passed with 8.61/12.60 GiB free, warmup underway, capacity report refreshing.
[2026-09-13 11:00:52 UTC] Context study: capacity report refreshed through 8B Q8_0/c8/32k; persistent service continues the 64k setting, remaining serving sweeps and hardware analysis pending.
[2026-09-13 11:02:13 UTC] Context study: 8B Q8_0/c8/64k warmup completed and single eight-request measured burst started; final token, placement, swap and memory validation pending.
[2026-09-13 11:03:26 UTC] Context study: verified active 8B Q8_0/c8/64k measurement with advancing prompt processing and generation; publication awaits full burst completion.
[2026-09-13 11:04:27 UTC] Context study: live 8B Q8_0/c8/64k measurement is the final check for its eight-client sweep; 16-client serving follows after completion and validation.
[2026-09-13 11:06:50 UTC] Context study: 8B Q8_0/c8 sweep complete through 64k+512; all eight capacity requests passed with matched FP16 inputs, zero sampled server swap and 8.60/12.59 GiB minimum GPU headroom; FINDINGS.md updated, 16-client serving underway.
[2026-09-13 11:07:25 UTC] Context study: capacity report refreshed with completed 8B Q8_0/c8 maximum of 64k; Q8_0/c16/2k has passed and its 4k measurement is active, remaining serving and hardware analysis pending.
[2026-09-13 11:09:20 UTC] Context study: 8B Q8_0/c16 passed 2k and 4k at 424.52/302.80 output tok/s; both 32-request runs have matched FP16 inputs, exact 512 outputs, valid GPU placement/headroom and zero sampled server swap; report refresh running.
[2026-09-13 11:09:50 UTC] Context study: report and plots refreshed through 8B Q8_0/c16/4k; c16/8k single measured run active, remaining serving/capacity sweeps and hardware analysis pending.
[2026-09-13 11:11:47 UTC] Context study: 8B Q8_0/c16/8k passed all 32 requests at 189.28 output tok/s with matched FP16 inputs, exact 512 outputs and zero sampled server swap; 16k warmup underway, report refresh running.
[2026-09-13 11:12:33 UTC] Context study: report and plots refreshed through 8B Q8_0/c16/8k; persistent service continues the 16k setting, remaining serving/capacity sweeps and hardware analysis pending.
[2026-09-13 11:14:21 UTC] Context study: verified active 8B Q8_0/c16/16k single measured run progressing through its second request wave; completed 32-request result pending.
[2026-09-13 11:16:01 UTC] Context study: 8B Q8_0/c16 main grid complete through 16k; final run passed all 32 requests at 103.66 output tok/s with matched FP16 inputs, exact 512 outputs and zero sampled server swap; 32k capacity extension underway, report refresh running.
[2026-09-13 11:17:12 UTC] Context study: report and plots refreshed with all 8B Q8_0/c16 main settings through 16k; persistent service continues the 32k capacity extension, remaining serving and hardware analysis pending.
[2026-09-13 11:18:41 UTC] Context study: 8B Q8_0/c16/32k warmup completed and single 16-request measured capacity burst started; final token, swap and GPU headroom validation pending.
[2026-09-13 11:19:31 UTC] Context study: verified active 8B Q8_0/c16/32k capacity burst with advancing prompt and generation counters; all 16 requests must finish before capacity publication.
[2026-09-13 11:20:49 UTC] Context study: 8B Q8_0/c16/32k burst still generating outputs; 64k capacity screening and 32-client serving follow its completed-run validation.
[2026-09-13 11:23:02 UTC] Context study: 8B Q8_0/c16 capacity sweep complete at 32k+512 with all 16 requests passing, zero sampled server swap and 8.33/12.37 GiB GPU headroom; 64k excluded before launch because KV alone needs 129.5 GiB; FINDINGS.md updated, 32-client serving underway.
[2026-09-13 11:24:04 UTC] Context study: capacity report refreshed with Q8_0/8B maxima of 64k at c8 and 32k at c16; Q8_0/c32/2k has passed and 4k measurement is active, remaining serving and hardware analysis pending.
[2026-09-13 11:27:18 UTC] Context study: 8B Q8_0/c32 passed 2k and 4k at 543.84/359.12 output tok/s, both 64-request runs with matched FP16 inputs, exact 512 outputs and zero sampled server swap; report refresh running, 8k setting underway.
[2026-09-13 11:27:44 UTC] Context study: report and plots refreshed through 8B Q8_0/c32/4k; c32/8k single measured run active, remaining serving and hardware analysis pending.
[2026-09-13 11:29:28 UTC] Context study: verified active 8B Q8_0/c32/8k measurement progressing through its second request wave; publication awaits the complete 64-request result.
[2026-09-13 11:31:38 UTC] Context study: 8B Q8_0/c32/8k passed all 64 requests at 209.54 output tok/s with matched FP16 inputs, exact 512 outputs and zero sampled server swap; 16k warmup underway, report refresh running.
[2026-09-13 11:32:42 UTC] Context study: report and plots refreshed through 8B Q8_0/c32/8k; persistent service continues the 16k setting, remaining serving/capacity sweeps and hardware analysis pending.
[2026-09-13 11:34:36 UTC] Context study: verified active 8B Q8_0/c32/16k single measured run with advancing request processing; publication awaits all 64 requests and final validation.
[2026-09-13 11:56:46 UTC] Context study: confirmed serving only, no active hardware profiler; all 1B and 8B FP16/Q8_0 serving sweeps complete, 8B Q4_K_M underway; selected profiler reruns follow all serving sweeps on otherwise idle GPUs.
[2026-09-13 11:58:03 UTC] Context study: report/plots refreshed and FINDINGS.md updated for complete 8B Q8_0 serving/capacity sweep (18 measurements, six capacity exclusions); largest tested inputs match FP16 at all four concurrencies despite 3.26–3.34 GiB more sampled headroom per GPU.
[2026-09-13 11:59:55 UTC] Context study: 8B Q4_K_M/c8 passed 2k, 4k and 8k at 342.02, 257.64 and 168.75 output tok/s with matched inputs and exact 512 outputs; 16k single measured run active.
[2026-09-13 12:00:51 UTC] Context study: verified live 8B Q4_K_M/c8/16k measurement advancing through response generation; report publication awaits the full 16-request result.
[2026-09-13 12:02:02 UTC] Context study: 8B Q4_K_M/c8/16k passed all 16 requests at 96.65 output tok/s with exact 512 outputs and zero sampled server swap; four main lengths complete at c8, capacity testing active and report refresh running.
[2026-09-13 12:02:50 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c8/16k; c8/32k single capacity measurement is live, remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:04:28 UTC] Context study: 8B Q4_K_M/c8/32k capacity burst passed eight requests at 47.91 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 64k setting started, capacity report refresh running.
[2026-09-13 12:05:34 UTC] Context study: capacity report refreshed through 8B Q4_K_M/c8/32k; persistent service verified live in 64k discarded warmup, with hardware profiling still pending completion of all serving sweeps.
[2026-09-13 12:06:59 UTC] Context study: verified active 8B Q4_K_M/c8/64k discarded warmup with advancing prompt processing; single measured run and publication remain pending.
[2026-09-13 12:07:55 UTC] Context study: 8B Q4_K_M/c8/64k warmup passed; single measured eight-request capacity burst started at 12:07:41 UTC with 65,536 input tokens and 512 requested outputs per request.
[2026-09-13 12:08:55 UTC] Context study: verified live 8B Q4_K_M/c8/64k single capacity measurement with advancing prefill; complete runtime result remains pending.
[2026-09-13 12:10:02 UTC] Context study: persistent service remains active in 8B Q4_K_M/c8/64k measurement, with later requests reaching prefill; final throughput and sampled headroom await completion.
[2026-09-13 12:10:56 UTC] Context study: 8B Q4_K_M/c8/64k single capacity measurement verified live in response generation; publication awaits all eight requests and final validation.
[2026-09-13 12:12:31 UTC] Context study: 8B Q4_K_M/c8/64k passed eight requests at 20.71 output tok/s with exact 512 outputs, matched inputs, zero sampled server swap and 10.20/14.11 GiB minimum GPU headroom; c8 sweep complete and c16 underway.
[2026-09-13 12:12:46 UTC] Context study: capacity report refreshed through the completed 8B Q4_K_M/c8/64k result; persistent serving service remains active at c16, with later formats/model and hardware analysis pending.
[2026-09-13 12:14:29 UTC] Context study: 8B Q4_K_M/c16 passed 2k and 4k at 491.74 and 332.96 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 8k setting underway and report/plot refresh running.
[2026-09-13 12:15:13 UTC] Context study: report and runtime plots refreshed through 8B Q4_K_M/c16/4k; c16/8k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:16:54 UTC] Context study: 8B Q4_K_M/c16/8k passed 32 requests at 199.21 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 16k setting underway and report/plot refresh running.
[2026-09-13 12:17:39 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c16/8k; persistent service verified active in 16k discarded warmup, with remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:18:55 UTC] Context study: verified live 8B Q4_K_M/c16/16k single measured run after successful warmup; 32-request workload advancing, publication awaits completion.
[2026-09-13 12:19:58 UTC] Context study: persistent service verified active with advancing 8B Q4_K_M/c16/16k request processing; final 32-request measurement remains pending.
[2026-09-13 12:21:30 UTC] Context study: 8B Q4_K_M/c16/16k passed 32 requests at 105.60 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; c16 main lengths complete, 32k capacity setting underway and report/plot refresh running.
[2026-09-13 12:22:13 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c16/16k; persistent serving service verified live in 32k discarded warmup, with remaining serving and hardware analysis pending.
[2026-09-13 12:23:48 UTC] Context study: verified active 8B Q4_K_M/c16/32k discarded warmup in response generation; single measured capacity run remains pending.
[2026-09-13 12:24:44 UTC] Context study: 8B Q4_K_M/c16/32k warmup passed and single measured 16-request capacity burst is live with advancing prefill; final result pending.
[2026-09-13 12:25:39 UTC] Context study: verified live 8B Q4_K_M/c16/32k measurement progressing through later prompts; publication awaits all 16 requests and final validation.
[2026-09-13 12:26:37 UTC] Context study: 8B Q4_K_M/c16/32k single capacity measurement verified live in response generation; full 16-request result remains pending.
[2026-09-13 12:29:53 UTC] Context study: 8B Q4_K_M/c16/32k passed 16 requests at 50.12 output tok/s with exact 512 outputs and zero sampled server swap; 64k excluded at 129.5 GiB KV alone, capacity report refreshed and FINDINGS.md consolidated, c32 underway.
[2026-09-13 12:31:02 UTC] Context study: 8B Q4_K_M/c32/2k passed at 569.54 output tok/s with matched inputs and exact 512 outputs; report/plots refreshed, c32/4k single measurement verified active, remaining serving and hardware analysis pending.
[2026-09-13 12:32:47 UTC] Context study: 8B Q4_K_M/c32/4k passed 64 requests at 368.98 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 8k warmup underway and report/plot refresh running.
[2026-09-13 12:33:33 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c32/4k; c32/8k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:34:48 UTC] Context study: verified active 8B Q4_K_M/c32/8k single measurement with advancing request processing; publication awaits the complete 64-request result.
[2026-09-13 12:36:25 UTC] Context study: 8B Q4_K_M/c32/8k passed 64 requests at 210.57 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 16k setting underway and report/plot refresh running.
[2026-09-13 12:37:13 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c32/8k; persistent service verified live in 16k discarded warmup, with remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:38:39 UTC] Context study: verified active 8B Q4_K_M/c32/16k discarded warmup completing responses; the single measured 64-request run remains pending.
[2026-09-13 12:39:36 UTC] Context study: 8B Q4_K_M/c32/16k warmup passed; single measured 64-request run verified live with advancing prefill, final result pending.
[2026-09-13 12:40:35 UTC] Context study: verified active 8B Q4_K_M/c32/16k single measurement with ongoing prefill and response generation; publication awaits all 64 requests and final validation.
[2026-09-13 12:41:37 UTC] Context study: persistent service verified live in 8B Q4_K_M/c32/16k response generation; complete 64-request runtime result remains pending.
[2026-09-13 12:42:40 UTC] Context study: verified active 8B Q4_K_M/c32/16k single measurement with later requests entering prefill; final validation and publication remain pending.
[2026-09-13 12:43:39 UTC] Context study: 8B Q4_K_M/c32/16k single measurement verified live with later requests generating responses; complete runtime metrics remain pending.
[2026-09-13 12:45:35 UTC] Context study: 8B Q4_K_M/c32/16k passed 64 requests at 110.47 output tok/s with exact 512 outputs, matched inputs and zero sampled server swap; 32k/64k excluded at 131/259 GiB KV alone, FINDINGS.md updated and c64 underway.
[2026-09-13 12:45:51 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c32/16k; findings record the 16k maximum at c32 with 9.20/13.24 GiB sampled headroom, and the persistent service continues c64 before later formats/model and hardware profiling.
[2026-09-13 12:47:23 UTC] Context study: GPU process check showed only the owned inference server on both GPUs; 8B Q4_K_M/c64/2k single measurement verified live in response generation, final result pending.
[2026-09-13 12:48:44 UTC] Context study: 8B Q4_K_M/c64/2k passed 128 requests at 617.78 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 4k setting underway and report/plot refresh running.
[2026-09-13 12:49:29 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c64/2k; c64/4k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:50:56 UTC] Context study: verified active 8B Q4_K_M/c64/4k single measurement in response generation; publication awaits the complete 128-request result.
[2026-09-13 12:51:57 UTC] Context study: 8B Q4_K_M/c64/4k single measurement verified live with later requests generating responses; complete runtime metrics and validation remain pending.
[2026-09-13 12:53:14 UTC] Context study: 8B Q4_K_M/c64/4k passed 128 requests at 393.80 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 8k setting underway and report/plot refresh running.
[2026-09-13 12:54:09 UTC] Context study: report and plots refreshed through 8B Q4_K_M/c64/4k; persistent service verified active in 8k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 12:55:27 UTC] Context study: verified active 8B Q4_K_M/c64/8k discarded warmup completing responses; single measured 128-request run remains pending.
[2026-09-13 12:56:26 UTC] Context study: 8B Q4_K_M/c64/8k warmup passed; single measured 128-request run verified live with advancing prefill, final result pending.
[2026-09-13 12:57:25 UTC] Context study: verified active 8B Q4_K_M/c64/8k single measurement with ongoing prefill and response generation; publication awaits all 128 requests and final validation.
[2026-09-13 12:58:24 UTC] Context study: 8B Q4_K_M/c64/8k single measurement verified live with later requests entering prefill; complete runtime metrics remain pending.
[2026-09-13 12:59:26 UTC] Context study: persistent service verified active with advancing later-request processing in 8B Q4_K_M/c64/8k; publication awaits the complete 128-request result.
[2026-09-13 13:00:35 UTC] Context study: 8B Q4_K_M/c64/8k single measurement verified live with later requests generating responses; final validation and publication remain pending.
[2026-09-13 13:02:33 UTC] Context study: 8B Q4_K_M serving/capacity sweep complete with 18 measurements and six exclusions; final c64/8k passed at 225.05 output tok/s with exact 512 outputs and zero sampled server swap, FINDINGS.md updated, IQ1_M serving underway.
[2026-09-13 13:03:00 UTC] Context study: report and plots refreshed with complete 8B FP16, Q8_0 and Q4_K_M sweeps; IQ1_M serving remains active, Q2_K/70B and fresh hardware profiling/analysis remain pending.
[2026-09-13 13:04:46 UTC] Context study: 8B IQ1_M/c8 passed 2k, 4k and 8k at 353.21, 257.32 and 163.92 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 16k setting underway and report/plot refresh running.
[2026-09-13 13:05:27 UTC] Context study: report and plots refreshed through 8B IQ1_M/c8/8k; c8/16k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:07:23 UTC] Context study: 8B IQ1_M/c8/16k passed 16 requests at 92.09 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; c8 main lengths complete, 32k capacity setting underway and report/plot refresh running.
[2026-09-13 13:08:07 UTC] Context study: report and plots refreshed through 8B IQ1_M/c8/16k; persistent service verified active in 32k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:09:26 UTC] Context study: verified active 8B IQ1_M/c8/32k single capacity measurement with advancing prefill; publication awaits all eight requests and final validation.
[2026-09-13 13:10:38 UTC] Context study: 8B IQ1_M/c8/32k capacity burst passed eight requests at 45.38 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 64k setting underway and capacity report refresh running.
[2026-09-13 13:11:18 UTC] Context study: capacity report refreshed through 8B IQ1_M/c8/32k; persistent service verified live in 64k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:12:41 UTC] Context study: verified active 8B IQ1_M/c8/64k discarded warmup with advancing prefill and response generation; single measured capacity run remains pending.
[2026-09-13 13:13:47 UTC] Context study: 8B IQ1_M/c8/64k discarded warmup verified live while finishing responses; single measured eight-request capacity burst remains pending.
[2026-09-13 13:14:44 UTC] Context study: 8B IQ1_M/c8/64k warmup passed; single measured eight-request capacity burst verified live with advancing prefill, final result pending.
[2026-09-13 13:15:42 UTC] Context study: verified active 8B IQ1_M/c8/64k single capacity measurement with advancing later-prompt processing; publication awaits all eight requests and final validation.
[2026-09-13 13:16:43 UTC] Context study: 8B IQ1_M/c8/64k single capacity measurement verified live with ongoing prefill and response generation; complete result remains pending.
[2026-09-13 13:18:57 UTC] Context study: 8B IQ1_M/c8/64k passed eight requests at 19.69 output tok/s with matched inputs, exact 512 outputs, zero sampled server swap and 11.32/15.19 GiB minimum GPU headroom; c8 sweep complete, FINDINGS.md updated and c16 underway.
[2026-09-13 13:20:52 UTC] Context study: 8B IQ1_M/c16/2k passed 32 requests at 242.94 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; observed throughput below c8's 353.21 tok/s, hardware cause pending, report/plot refresh running.
[2026-09-13 13:22:56 UTC] Context study: FINDINGS.md records the 8B 2k concurrency reversal: Q4_K_M gains 43.77% from c8 to c16 while IQ1_M loses 31.22%; all four measured windows had no sampled GPU thermal-limit flags, with kernel cause and repeatability unproven.
[2026-09-13 13:24:17 UTC] Context study: 8B IQ1_M/c16/4k passed 32 requests at 194.54 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; c16/8k measurement active and report/plot refresh running.
[2026-09-13 13:25:24 UTC] Context study: report and plots refreshed through 8B IQ1_M/c16/4k; c16/8k single measurement verified live with later requests generating responses, remaining serving and hardware analysis pending.
[2026-09-13 13:26:28 UTC] Context study: 8B IQ1_M/c16/8k passed 32 requests at 136.81 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 16k setting underway, with the next plot refresh planned after its completion.
[2026-09-13 13:27:27 UTC] Context study: 8B IQ1_M/c16/16k warmup passed; single measured 32-request run verified active, with report/plot refresh pending completion.
[2026-09-13 13:28:32 UTC] Context study: verified live 8B IQ1_M/c16/16k single measurement in response generation; complete 32-request metrics and validation remain pending.
[2026-09-13 13:29:32 UTC] Context study: 8B IQ1_M/c16/16k single measurement verified live with later requests entering prefill; publication awaits all 32 requests and final validation.
[2026-09-13 13:30:35 UTC] Context study: verified active 8B IQ1_M/c16/16k single measurement with later requests generating responses; full result and report refresh remain pending.
[2026-09-13 13:32:09 UTC] Context study: 8B IQ1_M/c16/16k passed 32 requests at 83.21 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; c16 main lengths complete, 32k capacity setting underway and report/plot refresh running.
[2026-09-13 13:32:57 UTC] Context study: report and plots refreshed through 8B IQ1_M/c16/16k, including the 8k result; persistent service verified active in 32k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:34:23 UTC] Context study: 8B IQ1_M/c16/32k warmup passed; single measured 16-request capacity burst verified active, final result pending.
[2026-09-13 13:35:42 UTC] Context study: verified active 8B IQ1_M/c16/32k single capacity measurement with advancing prefill and response generation; publication awaits all 16 requests and final validation.
[2026-09-13 13:36:35 UTC] Context study: 8B IQ1_M/c16/32k single capacity measurement verified live with later prompts advancing; complete result and capacity report update remain pending.
[2026-09-13 13:37:26 UTC] Context study: verified live 8B IQ1_M/c16/32k single capacity measurement finishing response generation; final validation and subsequent 64k feasibility check remain pending.
[2026-09-13 13:39:00 UTC] Context study: 8B IQ1_M/c16/32k passed 16 requests at 43.32 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 64k excluded at 129.5 GiB KV alone, c16 sweep complete, FINDINGS.md updated and c32 underway.
[2026-09-13 13:39:31 UTC] Context study: capacity report refreshed through completed 8B IQ1_M/c16/32k and its 64k exclusion; findings updated with the 32k maximum and 11.05/14.96 GiB sampled headroom, persistent service continues c32 before remaining serving and hardware analysis.
[2026-09-13 13:41:30 UTC] Context study: 8B IQ1_M/c32/2k passed 64 requests at 350.42 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 4k setting underway and report/plot refresh running.
[2026-09-13 13:42:18 UTC] Context study: report and plots refreshed through 8B IQ1_M/c32/2k; c32/4k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:43:38 UTC] Context study: verified active 8B IQ1_M/c32/4k single measurement with later requests generating responses; publication awaits the complete 64-request result.
[2026-09-13 13:44:43 UTC] Context study: 8B IQ1_M/c32/4k passed 64 requests at 256.11 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 8k setting underway and report/plot refresh running.
[2026-09-13 13:45:35 UTC] Context study: report and plots refreshed through 8B IQ1_M/c32/4k; persistent service verified active in 8k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:46:52 UTC] Context study: verified active 8B IQ1_M/c32/8k single measurement after successful warmup, with advancing prefill; complete 64-request result remains pending.
[2026-09-13 13:47:47 UTC] Context study: 8B IQ1_M/c32/8k single measurement verified live with later requests entering prefill as earlier responses finish; publication awaits all 64 requests and final validation.
[2026-09-13 13:48:40 UTC] Context study: verified live 8B IQ1_M/c32/8k single measurement with later requests reaching response generation; complete runtime result remains pending.
[2026-09-13 13:49:59 UTC] Context study: 8B IQ1_M/c32/8k passed 64 requests at 163.70 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 16k setting underway and report/plot refresh running.
[2026-09-13 13:50:52 UTC] Context study: report and plots refreshed through 8B IQ1_M/c32/8k; persistent service verified active in 16k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 13:52:23 UTC] Context study: verified active 8B IQ1_M/c32/16k discarded warmup in response generation; single measured 64-request run remains pending.
[2026-09-13 13:53:45 UTC] Context study: 8B IQ1_M/c32/16k warmup passed; single measured 64-request run verified active, final result and capacity checks pending.
[2026-09-13 13:54:40 UTC] Context study: verified active 8B IQ1_M/c32/16k single measurement with advancing prefill; publication awaits the complete 64-request result.
[2026-09-13 13:55:31 UTC] Context study: 8B IQ1_M/c32/16k single measurement verified live with ongoing prefill and response generation; final metrics and validation remain pending.
[2026-09-13 13:56:42 UTC] Context study: verified live 8B IQ1_M/c32/16k single measurement in response generation; complete 64-request result remains pending.
[2026-09-13 13:58:10 UTC] Context study: 8B IQ1_M/c32/16k single measurement verified live with later requests entering prefill; full result and subsequent capacity checks remain pending.
[2026-09-13 13:59:03 UTC] Context study: verified active 8B IQ1_M/c32/16k single measurement with later requests generating responses; final validation and publication remain pending.
[2026-09-13 14:01:03 UTC] Context study: 8B IQ1_M/c32/16k passed 64 requests at 92.99 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 32k/64k excluded at 131/259 GiB KV alone, FINDINGS.md updated and c64 underway.
[2026-09-13 14:01:32 UTC] Context study: report and plots refreshed through 8B IQ1_M/c32/16k and both capacity exclusions; findings record the 16k maximum with 10.31/14.32 GiB sampled headroom, persistent service continues c64 before remaining serving and hardware analysis.
[2026-09-13 14:03:20 UTC] Context study: verified active 8B IQ1_M/c64/2k single measurement with later requests generating responses; publication awaits the complete 128-request result.
[2026-09-13 14:05:28 UTC] Context study: 8B IQ1_M/c64/2k passed 128 requests at 455.21 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 4k setting underway and report/plot refresh running.
[2026-09-13 14:06:17 UTC] Context study: report and plots refreshed through 8B IQ1_M/c64/2k; c64/4k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 14:07:48 UTC] Context study: verified live 8B IQ1_M/c64/4k single measurement in response generation; publication awaits all 128 requests and final validation.
[2026-09-13 14:08:51 UTC] Context study: 8B IQ1_M/c64/4k single measurement verified live with later requests entering prefill; complete runtime metrics remain pending.
[2026-09-13 14:09:51 UTC] Context study: verified active 8B IQ1_M/c64/4k single measurement with later requests finishing response generation; final validation and publication remain pending.
[2026-09-13 14:11:10 UTC] Context study: 8B IQ1_M/c64/4k passed 128 requests at 309.70 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 8k setting underway and report/plot refresh running.
[2026-09-13 14:12:01 UTC] Context study: report and plots refreshed through 8B IQ1_M/c64/4k; persistent service verified active in 8k discarded warmup, remaining serving sweeps and hardware analysis pending.
[2026-09-13 14:13:27 UTC] Context study: 8B IQ1_M/c64/8k warmup passed; single measured 128-request run verified active, final result and capacity checks pending.
[2026-09-13 14:14:19 UTC] Context study: verified active 8B IQ1_M/c64/8k single measurement with advancing prefill; publication awaits all 128 requests and final validation.
[2026-09-13 14:15:11 UTC] Context study: 8B IQ1_M/c64/8k single measurement verified live with ongoing prefill and response generation; complete runtime result remains pending.
[2026-09-13 14:16:11 UTC] Context study: verified active 8B IQ1_M/c64/8k single measurement in response generation; complete 128-request result remains pending.
[2026-09-13 14:17:07 UTC] Context study: 8B IQ1_M/c64/8k single measurement verified live with later requests entering prefill as earlier responses finish; final validation and publication remain pending.
[2026-09-13 14:18:15 UTC] Context study: verified live 8B IQ1_M/c64/8k single measurement with advancing later-request processing; complete runtime result remains pending.
[2026-09-13 14:19:10 UTC] Context study: 8B IQ1_M/c64/8k single measurement verified live with later requests generating responses; final validation and larger-prompt capacity checks remain pending.
[2026-09-13 14:22:08 UTC] Context study: 8B IQ1_M sweep complete with 18 measurements and six exclusions; final c64/8k passed at 187.35 output tok/s with exact 512 outputs and zero sampled server swap, report/plots refreshed and FINDINGS.md consolidated across four completed formats; Q2_K serving underway.
[2026-09-13 14:24:10 UTC] Context study: 8B Q2_K/c8 passed 2k, 4k and 8k at 295.95, 221.50 and 145.97 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; 16k setting underway and report/plot refresh starting.
[2026-09-13 14:25:05 UTC] Context study: report and plots refreshed through 8B Q2_K/c8/8k; c8/16k single measured run verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 14:27:19 UTC] Context study: 8B Q2_K/c8/16k passed 16 requests at 84.26 output tok/s with matched inputs, exact 512 outputs and zero sampled server swap; c8 main lengths complete, 32k capacity setting underway and report/plot refresh running.
[2026-09-13 14:28:10 UTC] Context study: report and plots refreshed through 8B Q2_K/c8/16k; c8/32k single measured capacity burst verified active, remaining serving sweeps and hardware analysis pending.
[2026-09-13 14:30:24 UTC] Context study: verified live 8B Q2_K/c8/32k single capacity measurement finishing responses; complete eight-request result and validation remain pending.
[2026-09-13 14:37:51 UTC] Context study: Confirmed sequencing: finish all serving/capacity sweeps first, then profile selected identical settings separately on otherwise idle GPUs; current 8B Q2_K/c8/64k inference active, 1B inference complete, 70B inference and full hardware-counter sweep pending.
[2026-09-13 14:38:57 UTC] Context study: 8B Q2_K/c8 passed 32k and 64k capacity bursts at 42.44 and 18.76 output tok/s with exact 512 outputs, matched inputs, full GPU placement and zero sampled server swap; 64k retained 11.00/14.82 GiB GPU headroom. Report refreshed through 32k; c16 serving underway, full profiling remains after all serving.
[2026-09-13 14:40:36 UTC] Context study: Added 8B Q2_K 2k concurrency comparison to FINDINGS.md: 295.95 to 468.37 output tok/s from c8 to c16 (+58.26%), narrowing its Q4 deficit from 13.47% to 4.75%; both Q2 runs passed exact tokens/input/placement/swap checks and had no sampled thermal-limit flags. Report and plots refreshing.
[2026-09-13 14:41:52 UTC] Context study: Report and plots refreshed with 8B Q2_K/c8/64k and c16/2k results; c16/4k completed and c16/8k measured run verified active. Remaining 8B and 70B serving precedes hardware profiling.
[2026-09-13 14:42:58 UTC] Context study: Verified live 8B Q2_K/c16/8k measured run finishing responses; 16k and 32k capacity settings follow. Q2_K/c16/4k passed at 308.49 output tok/s; no profiler is active.
[2026-09-13 14:44:17 UTC] Context study: 8B Q2_K/c16/8k completed and passed exact-token, matched-input, placement, memory and swap checks; 81 total 8B measurements plus 24 capacity exclusions are complete. Updated Q2 concurrency comparison is in FINDINGS.md; full hardware profiling remains pending.
[2026-09-13 14:45:02 UTC] Context study: Verified 8B Q2_K/c16/16k single measured run active under the serving service; report and plots refreshing through c16/8k. Profiling waits for completion of all serving/capacity sweeps.
[2026-09-13 14:45:55 UTC] Context study: Report and plots refreshed through 8B Q2_K/c16/8k (179.04 output tok/s); c16/16k measured run verified active. New 8B total remains 81 completed measurements plus 24 capacity exclusions.
[2026-09-13 14:46:55 UTC] Context study: Verified live serving service continues the 8B Q2_K/c16/16k 32-request measurement; final metrics pending, then c16/32k capacity testing.
[2026-09-13 14:49:28 UTC] Context study: 8B Q2_K/c16/16k passed 32 matched requests with exactly 512 output tokens at 93.59 output tok/s, full GPU placement and zero sampled swap; GPU0 software thermal limiting was observed. Report/plots refreshing; c16/32k capacity warmup active.
[2026-09-13 14:50:12 UTC] Context study: Report and plots refreshed through 8B Q2_K/c16/16k; c16 main prompt lengths complete and 32k capacity warmup verified active. Serving service remains live; 70B inference and full hardware profiling are pending.
[2026-09-13 14:51:05 UTC] Context study: GPU isolation check found only the study's llama-server PID 408828 on both A6000s; no NCU/Nsys profiling process. 8B Q2_K/c16/32k capacity warmup remains active under the serving service.
[2026-09-13 14:52:08 UTC] Context study: 8B Q2_K/c16/32k warmup finished; the single measured 16-request capacity burst is verified active since 14:51:14 UTC. Both GPUs remain assigned to the study inference server; counters are pending.
[2026-09-13 14:53:17 UTC] Context study: Verified live 8B Q2_K/c16/32k measurement continues prompt processing and response generation; complete result remains pending and no restart or profiling was initiated.
[2026-09-13 14:54:27 UTC] Context study: 8B Q2_K/c16/32k single capacity burst verified live and generating remaining outputs; complete 16-request result and final checks remain pending.
[2026-09-13 14:55:44 UTC] Context study: 8B Q2_K/c16/32k passed at 44.75 output tok/s with 16 matched requests, exact 512 outputs, full GPU placement, zero sampled server swap and 10.72/14.60 GiB GPU headroom; c16/64k excluded by capacity screen. Capacity report refreshing; c32 serving underway.
[2026-09-13 14:56:38 UTC] Context study: Capacity report refreshed through 8B Q2_K/c16/32k; c16/64k excluded because FP16 KV alone requires 129.5 GiB across both GPUs. The c32/2k measured run is verified active and finishing responses.
[2026-09-13 14:57:52 UTC] Context study: 8B Q2_K/c32/2k passed 64 matched requests with exact 512 outputs at 515.23 output tok/s, full GPU placement, zero sampled swap and no sampled thermal-limit flags. Report/plots refreshing; c32/4k warmup underway.
[2026-09-13 14:59:19 UTC] Context study: Report and plots refreshed through 8B Q2_K/c32/2k; c32/4k single measured run verified active. Six 8B measurements and remaining capacity screens are outstanding; 70B serving and full profiling remain pending.
[2026-09-13 15:00:37 UTC] Context study: 8B Q2_K/c32/4k passed 64 matched requests with exact 512 outputs at 327.45 output tok/s, full GPU placement and zero sampled swap; GPU0 software thermal limiting observed. Report/plots refreshing; c32/8k warmup active.
[2026-09-13 15:01:44 UTC] Context study: Report and plots refreshed through 8B Q2_K/c32/4k; c32/8k single measured run verified active since 15:01:03 UTC. Five 8B measurements and remaining capacity screens are outstanding; 70B serving and full profiling remain pending.
[2026-09-13 15:02:51 UTC] Context study: Verified live 8B Q2_K/c32/8k single measured run continues; final 64-request result is pending, with c32/16k scheduled next. Full hardware profiling remains after all serving.
[2026-09-13 15:03:42 UTC] Context study: 8B Q2_K/c32/8k measured workload verified active; complete 64-request result remains pending. Published report and plots currently cover c32/2k and c32/4k.
[2026-09-13 15:05:23 UTC] Context study: 8B Q2_K/c32/8k passed 64 matched requests with exact 512 outputs at 185.42 output tok/s, full GPU placement and zero sampled swap; GPU0 software thermal limiting observed. Report/plots refreshing; c32/16k preparation underway.
[2026-09-13 15:05:38 UTC] Context study: Report and plots refreshed through 8B Q2_K/c32/8k; c32/16k warmup verified active. Four 8B measurements plus remaining capacity screens are outstanding; 70B inference and full hardware profiling remain pending.
[2026-09-13 15:07:02 UTC] Context study: Read-only profiling selector check confirmed 2k/16k/joint-maximum endpoint deduplication across runnable formats and single-capture wrapper default; 8B selection will be finalized after all its serving cells resolve. Q2_K/c32/16k warmup remains active.
[2026-09-13 15:08:36 UTC] Context study: 8B Q2_K/c32/16k warmup completed; the single measured 64-request run is verified active since 15:07:38 UTC. Final inference metrics and capacity validation remain pending.
[2026-09-13 15:10:37 UTC] Context study: Verified active 8B Q2_K/c32/16k measurement continues; the other four 8B formats have completed this setting. Q2's final matched result is still pending before publishing the five-format comparison.
[2026-09-13 15:11:39 UTC] Context study: Verified live 8B Q2_K/c32/16k run has advanced to later requests in its 64-request workload; final timing, memory and capacity checks remain pending.
[2026-09-13 15:13:14 UTC] Context study: Verified live 8B Q2_K/c32/16k run continues processing its final group of requests; the complete measurement remains pending, with the serving service healthy and active.
[2026-09-13 15:14:07 UTC] Context study: Verified 8B Q2_K/c32/16k measurement finishing responses; acceptance would leave three c64 measurements plus remaining capacity screens before completing 8B serving.
[2026-09-13 15:15:15 UTC] Context study: 8B Q2_K/c32/16k passed 64 matched requests with exact 512 outputs at 96.66 output tok/s, full GPU placement, zero sampled swap and 9.99/13.96 GiB headroom; c32/32k and 64k excluded by FP16 KV requirements of 131/259 GiB. Report/plots refreshing; c64 serving underway.
[2026-09-13 15:16:26 UTC] Context study: All five 8B formats now have validated maxima of 64k/c8, 32k/c16 and 16k/c32; Q2_K/c64 testing remains. The c64/2k single measurement is verified active since 15:15:15 UTC; report refresh still finishing.
[2026-09-13 15:17:23 UTC] Context study: Report and plots refreshed through 8B Q2_K/c32/16k and both c32 capacity exclusions; c64/2k measured run verified active. Three 8B measurements plus remaining capacity screens, all 70B serving and full hardware profiling remain pending.
[2026-09-13 15:18:44 UTC] Context study: 8B Q2_K/c64/2k passed 128 matched requests with exact 512 outputs at 554.01 output tok/s, full GPU placement and zero sampled swap; GPU0 software thermal limiting observed. Report/plots refreshing; only c64/4k and 8k measurements plus capacity screens remain for 8B.
[2026-09-13 15:19:34 UTC] Context study: Report and plots refreshed through 8B Q2_K/c64/2k; c64/4k warmup verified active and finishing responses. Two 8B measurements plus capacity screens, all 70B serving and full profiling remain outstanding.
[2026-09-13 15:20:31 UTC] Context study: Verified active 8B Q2_K/c64/4k single measured run since 15:19:25 UTC; final 128-request result remains pending. After c64/8k and capacity screens, consolidate the complete 8B capacity table and finalize profiling selection.
[2026-09-13 15:22:02 UTC] Context study: Verified 8B Q2_K/c64/4k measurement has advanced to later requests in the 128-request workload; final result remains pending and c64/8k has not started.
[2026-09-13 15:23:12 UTC] Context study: Verified live 8B Q2_K/c64/4k measurement generating remaining outputs; complete result and acceptance checks remain pending.
[2026-09-13 15:24:49 UTC] Context study: 8B Q2_K/c64/4k passed 128 matched requests with exact 512 outputs at 345.85 output tok/s, full GPU placement and zero sampled swap; GPU0 software thermal limiting observed. Report/plots refreshing, old/new study labels clarified in FINDINGS.md, and only c64/8k plus capacity screens remain for 8B.
[2026-09-13 15:26:29 UTC] Context study: Report and plots refreshed through 8B Q2_K/c64/4k; the final 8B c64/8k setting is verified active in warmup. FINDINGS.md now explicitly distinguishes the earlier study's workload and GPU layout; 70B serving and full profiling remain pending.
[2026-09-13 15:28:09 UTC] Context study: Final 8B Q2_K/c64/8k single measured run verified active since 15:26:21 UTC; GPU isolation check lists only study llama-server PID 417712 on both A6000s. Complete 128-request result remains pending.
[2026-09-13 15:29:35 UTC] Context study: Verified live final 8B Q2_K/c64/8k measurement continues its 128-request workload; complete result, final capacity screens and 8B profiling selection remain pending.
[2026-09-13 15:31:07 UTC] Context study: Verified final 8B Q2_K/c64/8k measurement progressing through later requests; the driver advances to 70B after this result and remaining capacity screens. Full profiling remains after all serving.
[2026-09-13 15:32:34 UTC] Context study: Verified final 8B Q2_K/c64/8k measurement finishing responses; complete result and memory checks remain pending before declaring 8B inference complete.
[2026-09-13 15:36:57 UTC] Context study: 8B inference complete for all five formats: 90 validated measurements and 30 capacity exclusions; final Q2_K/c64/8k passed at 195.83 output tok/s with 8.42/12.58 GiB headroom and zero sampled swap. FINDINGS.md includes all five capacity columns; 50 unique profiling endpoints and five capacity skips selected without GPU captures. 70B driver preparing models since 15:32:34 UTC.
[2026-09-13 15:38:43 UTC] Context study: Report and plots now publish all 1B and 8B results: 155 main measurements and 50 capacity-extension measurements; 8B throughput plot visually checked. 70B initialization is reading existing Q8_0 model data before request measurements; full profiling and final kernel-based analysis remain pending.
[2026-09-13 15:41:53 UTC] Context study: 70B initialization completed its model-file checks; all 24 FP16 settings excluded before download/load because weights alone exceed available aggregate GPU memory with reserve. Q4_K_M is loading for c8/2k; report refreshing the FP16 capacity exclusions.
[2026-09-13 15:44:43 UTC] Context study: 70B Q4_K_M/c8/2k warmup completed and single measured run is active since 15:42:59 UTC; report refreshed with all 24 FP16 settings excluded before download/load. No completed 70B throughput is published yet.
[2026-09-13 15:48:37 UTC] Context study: First 70B measurement passed: Q4_K_M/c8/2k, 16 exact-512-output requests, 45.92 output tok/s, full GPU placement, zero sampled swap and 23.67/23.97 GiB GPU headroom. Report/plots refreshing; c8/4k model startup underway.
[2026-09-13 15:50:12 UTC] Context study: Report and plots refreshed with the first validated 70B Q4_K_M/c8/2k measurement and 24 FP16 exclusions; Q4_K_M/c8/4k single measured run verified active since 15:49:26 UTC. Remaining 70B serving and full hardware analysis are pending.
[2026-09-13 15:52:40 UTC] Context study: 70B Q4_K_M/c8/2k measured-window monitoring recorded GPU0 software thermal limiting; GPU1 and both hardware thermal flags remained inactive. The c8/4k single measured run remains active, with no completed peer-format input comparison yet.
[2026-09-13 15:53:50 UTC] Context study: 70B Q4_K_M/c8/4k passed 16 exact-512-output requests at 36.72 output tok/s with full GPU placement, zero sampled swap and 21.09/21.52 GiB headroom; GPU0 software thermal limiting recorded. Report/plots refreshing; c8/8k model startup underway.
[2026-09-13 15:57:00 UTC] Context study: Report and plots refreshed through 70B Q4_K_M/c8/4k; c8/8k warmup verified active. Two 70B measurements are complete, 24 FP16 settings are excluded, and the remaining 70B serving and hardware profiling are pending.
[2026-09-13 15:57:31 UTC] Context study: Verified 70B Q4_K_M/c8/8k warmup finishing responses; the single measured run follows. Published 70B results remain c8/2k and c8/4k, with full profiling pending.
[2026-09-13 15:58:26 UTC] Context study: 70B Q4_K_M/c8/8k warmup completed; the single measured 16-request run is verified active since 15:57:31 UTC. Final runtime and resource checks remain pending.
[2026-09-13 16:00:20 UTC] Context study: Verified live 70B Q4_K_M/c8/8k single measured workload continues; full 16-request result remains pending before report publication.
[2026-09-13 16:01:45 UTC] Context study: Verified 70B Q4_K_M/c8/8k measured workload has advanced to later requests; complete 16-request result remains pending. Published 70B measurements remain c8/2k and c8/4k.
[2026-09-13 16:02:34 UTC] Context study: Verified live 70B Q4_K_M/c8/8k batch generating remaining outputs; complete result and final runtime/resource acceptance remain pending.
[2026-09-13 16:05:11 UTC] Context study: 70B Q4_K_M/c8/8k passed 16 exact-512-output requests at 25.66 output tok/s with full GPU placement, zero sampled swap and 15.95/16.63 GiB headroom; GPU0 software thermal limiting recorded. Report/plots refreshing; c8/16k model startup underway.
[2026-09-13 16:06:01 UTC] Context study: Report and plots refreshed through 70B Q4_K_M/c8/8k; c8/16k warmup verified active. Three 70B measurements and 24 FP16 exclusions are recorded; remaining serving and full hardware profiling are pending.
[2026-09-13 16:06:50 UTC] Context study: Verified 70B Q4_K_M/c8/16k warmup active; GPU isolation check lists only study llama-server PID 426830 on both A6000s. Complete 16k measurement remains pending.
[2026-09-13 16:11:46 UTC] Context study: Added measured KV-allocation explanation to FINDINGS.md: 32/128/320 KiB per token position for 1B/8B/70B, verified against GPU buffers at 2k and 4k; calculated c64 2k-to-4k KV increase is 4/16/40 GiB. 70B Q4_K_M/c8/16k measured run remains active.
[2026-09-13 16:13:57 UTC] Context study: Verified live 70B Q4_K_M/c8/16k measurement progressing into later requests of its 16-request workload; complete runtime and resource results remain pending.
[2026-09-13 16:14:48 UTC] Context study: 70B Q4_K_M/c8/16k single measurement verified active and processing later requests; final 16-request result is still pending.
[2026-09-13 16:16:22 UTC] Context study: Verified live 70B Q4_K_M/c8/16k measurement processing its later prompts; complete runtime result and subsequent 32k/64k capacity screens remain pending.
[2026-09-13 16:17:16 UTC] Context study: Verified 70B Q4_K_M/c8/16k measurement generating remaining responses; final acceptance and maximum-input recording await the complete 16-request result.
[2026-09-13 16:19:30 UTC] Context study: 70B Q4_K_M/c8/16k passed 16 exact-512-output requests at 15.54 output tok/s with full GPU placement, zero sampled swap and 5.67/6.85 GiB headroom; GPU0 software thermal limiting recorded. Report/plots refreshing with larger-length capacity screens.
[2026-09-13 16:34:33 UTC] Context study: Serving sweeps precede separate Nsight captures on idle GPUs; 1B/8B are complete, 70B paused after 63.6 MiB host swap in discarded warmup. Archived two unaccepted attempts and preparing service-only zero-swap enforcement; accepted results retained.
[2026-09-13 16:35:21 UTC] Context study: Resumed serving driver with service MemorySwapMax=0; verified effective cgroup swap limit and current swap are both zero. Completed cells are reused; failed warmup and interrupted 70B attempts remain archived.
[2026-09-13 16:38:08 UTC] Context study: Resume reused all completed 1B cases; 8B model identity checks are in progress. Study cgroup swap remains zero, with no GPU profiler active.
[2026-09-13 16:39:53 UTC] Context study: All accepted 1B and 8B measurements were reused; 70B model hash checks are progressing before retrying the two archived unaccepted cases. Effective study swap limit/current usage remain 0/0.
[2026-09-13 16:41:08 UTC] Context study: 70B preflight has read 73.45 GiB while validating model files; no new inference result yet. Study cgroup swap remains zero.
[2026-09-13 16:42:22 UTC] Context study: 70B restart remains in model verification with zero study cgroup swap; inference resumes after checks, and hardware counters remain queued after all serving sweeps.
[2026-09-13 16:43:35 UTC] Context study: 70B model verification is progressing; four accepted 70B measurements remain intact and study cgroup swap remains zero.
[2026-09-13 16:44:41 UTC] Context study: 70B startup verification continues without new failures; zero swap is enforced, and confirmation during the first resumed inference case is pending.
[2026-09-13 16:45:58 UTC] Context study: 70B model identity checks passed and Q4_K_M/c16/2k retry is loading; verified the server inherits the service zero-swap cgroup and currently has VmSwap=0. All accepted cases were skipped.
[2026-09-13 16:46:57 UTC] Context study: 70B Q4_K_M/c16/2k retry reached discarded warmup; verified only the study server is using both GPUs, with no concurrent profiler.
[2026-09-13 16:48:07 UTC] Context study: 70B Q4_K_M/c16/2k discarded warmup is nearing completion with zero cgroup swap and no out-of-memory events; the single measured run awaits warmup acceptance.
[2026-09-13 16:48:59 UTC] Context study: 70B Q4_K_M/c16/2k retry passed discarded warmup with zero sampled server swap and 20.14/20.61 GiB GPU headroom; its single measured r1 is active. Hardware profiling remains deferred until all serving sweeps finish.
[2026-09-13 16:51:18 UTC] Context study: Verified live 70B Q4_K_M/c16/2k r1 is generating later responses; the complete 32-request result is pending, and the archived interrupted attempt remains excluded.
[2026-09-13 16:52:25 UTC] Context study: 70B Q4_K_M/c16/2k passed 32 exact-512-output requests at 83.54 output tok/s with zero sampled swap and 20.14/20.61 GiB GPU headroom; GPU0 software thermal limiting recorded. Report refresh started; 4k case follows.
[2026-09-13 16:54:00 UTC] Context study: Report/plots refreshed successfully with five accepted 70B measurements; Q4_K_M/c16/4k is in discarded warmup. Overall serving, hardware profiling, and final interpretation remain incomplete.
[2026-09-13 16:56:28 UTC] Context study: Prepared sequential profiling continuation launcher with one capture and five matched launches; it requires completed serving/capacity/input checks, idle GPUs, and a zero-swap service before starting. Shell syntax passed; no profiling was launched.
[2026-09-13 16:57:29 UTC] Context study: 70B Q4_K_M/c16/4k passed warmup with zero server swap and more than 15 GiB free per GPU; its single measured r1 is active. Profiling continuation launcher is prepared but not started.
[2026-09-13 16:59:10 UTC] Context study: CPU-only profiling preflight passed: all five existing calibration capture identities match and launcher rejects an active serving service. 70B Q4_K_M/c16/4k remains live with zero cgroup swap; no profiler started.
[2026-09-13 17:00:21 UTC] Context study: Verified 70B Q4_K_M/c16/4k measurement is generating later responses. Profiling reuse and serving-isolation checks passed; the profiler remains unstarted.
[2026-09-13 17:01:31 UTC] Context study: 70B Q4_K_M/c16/4k passed 32 exact-512-output requests at 56.14 output tok/s with zero swap and 15.01/15.73 GiB headroom; GPU0 software thermal limiting recorded. Both archived recovery cases now have accepted r1 results; report refresh started.
[2026-09-13 17:03:05 UTC] Context study: Report and plots refreshed with six validated 70B measurements. Verified live Q4_K_M/c16/8k server with zero cgroup swap; hardware profiling remains queued behind serving.
[2026-09-13 17:04:58 UTC] Context study: Corrected stale FINDINGS.md text that called the extension unrun and described a 128-output follow-up; it now states the approved power-of-two inputs/512 outputs and separates completed 1B/8B sweeps from pending 70B/profiling work.
[2026-09-13 17:06:22 UTC] Context study: FINDINGS.md now reflects the current extension protocol/status. Verified live 70B Q4_K_M/c16/8k warmup generating remaining outputs with zero cgroup swap.
[2026-09-13 17:08:21 UTC] Context study: 70B Q4_K_M/c16/8k passed discarded warmup with zero server swap and 4.74/5.96 GiB GPU headroom; single measured r1 is active and final throughput is pending.
[2026-09-13 17:09:41 UTC] Context study: Verified live 70B Q4_K_M/c16/8k measurement generating responses with zero cgroup swap; complete 32-request result remains pending.
[2026-09-13 17:10:42 UTC] Context study: 70B Q4_K_M/c16/8k remains live and is finishing responses without new failures or cgroup swap; throughput will be accepted only after the full request set completes.
[2026-09-13 17:12:30 UTC] Context study: Verified 70B Q4_K_M/c16/8k measurement progressing through later requests with zero swap; both GPUs list only the owned study server and no external compute process.
[2026-09-13 17:13:28 UTC] Context study: Verified exclusive use of both GPUs by the live 70B Q4_K_M/c16/8k server; later prompts are processing and final r1 result remains pending.
[2026-09-13 17:14:25 UTC] Context study: 70B Q4_K_M/c16/8k later requests are generating remaining outputs; live checks show zero cgroup swap and no competing GPU process.
[2026-09-13 17:16:06 UTC] Context study: 70B Q4_K_M/c16/8k passed 32 exact-512-output requests at 33.41 output tok/s, zero swap, and 4.74/5.96 GiB headroom; GPU0 software thermal limiting recorded. Larger c16 inputs were capacity-excluded; c32/2k server is now live.
[2026-09-13 17:17:26 UTC] Context study: Report updated: 70B Q4_K_M/c16 maximum validated input is 8k; 16k requires at least 64.49/62.30 GiB per GPU before working buffers versus ~47.4 GiB available. Q4_K_M/c32/2k warmup is underway.
[2026-09-13 17:20:59 UTC] Context study: Verified pinned split logic and loaded KV bytes agree on 9/7, 17/15, and 41/39 transformer layers on GPU0/GPU1 for 1B/8B/70B; output-layer counting explains unequal KV under split 1,1. The 70B c32/2k measurement is active.
[2026-09-13 17:21:57 UTC] Context study: Added source- and allocation-backed FINDINGS.md explanation: output-layer counting under split 1,1 creates unequal transformer/KV placement; 8B/c64/8k has 4.375 GiB more KV on GPU0. Live 70B Q4_K_M/c32/2k measurement continues.
[2026-09-13 17:22:59 UTC] Context study: Verified live 70B Q4_K_M/c32/2k r1 processing later requests; both GPUs remain exclusive to the study and cgroup swap remains zero.
[2026-09-13 17:24:17 UTC] Context study: 70B Q4_K_M/c32/2k later responses are generating; live checks show no competing GPU process or swap. Final 64-request result remains pending.
[2026-09-13 17:25:40 UTC] Context study: 70B Q4_K_M/c32/2k passed 64 exact-512-output requests at 104.60 output tok/s with zero swap and 13.09/13.91 GiB headroom; GPU0 software thermal limiting recorded. c32/4k server is live; report refresh started.
[2026-09-13 17:26:46 UTC] Context study: Verified live 70B Q4_K_M/c32/4k server after loading, with exclusive GPU use and zero cgroup swap; report refresh remains active and no profiler has started.
[2026-09-13 17:27:33 UTC] Context study: Report/plots refreshed with eight validated 70B measurements; 32-client/2k result appears once. FINDINGS.md now explains unequal KV allocation under the fixed 1,1 split; c32/4k testing continues.
[2026-09-13 17:29:43 UTC] Context study: 70B Q4_K_M/c32/4k warmup is generating outputs after full transformer/FP16-KV placement passed; both GPUs remain exclusive to the study with zero cgroup swap.
[2026-09-13 17:31:04 UTC] Context study: 70B Q4_K_M/c32/4k passed warmup with zero server swap and 2.83/4.15 GiB headroom; the single measured run of 64 requests has started, retaining the required per-GPU reserve.
[2026-09-13 17:32:02 UTC] Context study: Verified live 70B Q4_K_M/c32/4k r1 processing prompts normally; both GPUs remain exclusive and study cgroup swap is zero.
[2026-09-13 17:33:26 UTC] Context study: 70B Q4_K_M/c32/4k single measurement remains active without resource failures; waiting for all 64 requests before acceptance.
[2026-09-13 17:34:20 UTC] Context study: Verified live 70B Q4_K_M/c32/4k run generating outputs, with exclusive use of both GPUs and zero study cgroup swap.
[2026-09-13 17:35:41 UTC] Context study: 70B Q4_K_M/c32/4k measurement has advanced to later requests; acceptance awaits the complete 64-request output and resource checks.
[2026-09-13 17:36:42 UTC] Context study: Verified live 70B Q4_K_M/c32/4k run progressing through later prompts with exclusive GPU use and zero swap; final result remains pending.
[2026-09-13 17:38:05 UTC] Context study: 70B Q4_K_M/c32/4k final queued prompts have started; verified live server, zero cgroup swap, and no competing GPU workload.
[2026-09-13 17:39:06 UTC] Context study: 70B Q4_K_M/c32/4k remaining responses are nearing 512 output tokens; final acceptance and Q4_K_M capacity consolidation are next, followed by Q2_K.
[2026-09-13 17:41:35 UTC] Context study: 70B Q4_K_M sweep completed with nine validated measurements and 15 capacity exclusions; c32/4k passed at 65.01 output tok/s with exact outputs and zero swap. Q2_K/c8/2k server is live; consolidating capacity findings.
[2026-09-13 17:43:50 UTC] Context study: Published complete 70B Q4_K_M capacity table and explanation: equal 131072 input tokens across slots need 41.875–55 GiB KV as per-request output/padding reservations grow. Q2_K/c8/2k single measured run is active; report remains overall incomplete.
[2026-09-13 17:46:15 UTC] Context study: 70B Q2_K/c8/2k completed and the runner advanced to 4k; checking exact outputs, resources, and matched inputs against the Q4_K_M baseline before publishing the comparison.
[2026-09-13 17:47:53 UTC] Context study: 70B Q2_K/c8/2k passed 16 exact-512-output requests with inputs matched to Q4_K_M: 38.48 vs 45.92 output tok/s (16.2% lower), zero swap, and GPU0 software thermal limiting. Refreshing the report; kernel attribution remains pending.
[2026-09-13 17:49:23 UTC] Context study: Published matched 70B/c8/2k Q2_K–Q4_K_M comparison: Q2 saves 7.56/7.25 GiB at individual GPU peaks but has 16.2% lower throughput, 29.2% higher p95 TTFT, and 19.2% higher p95 TPOT; thermal limits disclosed and kernel attribution pending. Q2_K/c8/4k is measuring.
[2026-09-13 17:51:18 UTC] Context study: Verified live 70B Q2_K/c8/4k measurement processing later requests with exclusive GPU use and zero cgroup swap.
[2026-09-13 17:52:44 UTC] Context study: 70B Q2_K/c8/4k later responses are approaching completion; exact-token/resource validation and report refresh will follow the complete result.
[2026-09-13 17:53:50 UTC] Context study: 70B Q2_K/c8/4k passed 16 exact-512-output requests with matched Q4_K_M inputs: 30.64 vs 36.72 output tok/s, zero swap, and 28.65/28.77 GiB headroom; GPU0 software thermal limiting recorded. Report refreshing; 8k case started.
[2026-09-13 17:55:25 UTC] Context study: Report/plots refreshed with 11 validated 70B measurements; Q2_K/c8/4k throughput is 16.6% below matched Q4_K_M. Verified live 8k warmup with zero cgroup swap and no competing GPU process.
[2026-09-13 17:57:04 UTC] Context study: 70B Q2_K/c8/8k passed warmup with zero server swap and over 23 GiB free on each GPU; its single measured run is active with exclusive GPU use.
[2026-09-13 17:58:31 UTC] Context study: Verified live 70B Q2_K/c8/8k r1 progressing into output generation with zero cgroup swap and no competing GPU process.
[2026-09-13 17:59:29 UTC] Context study: 70B Q2_K/c8/8k has advanced to later requests without resource failures; complete 16-request result remains pending.
[2026-09-13 18:01:01 UTC] Context study: Verified live 70B Q2_K/c8/8k measurement progressing through later prompts and remaining responses; exclusive GPU use and zero cgroup swap continue.
[2026-09-13 18:01:59 UTC] Context study: 70B Q2_K/c8/8k later requests are generating remaining outputs; final throughput is pending with zero swap and exclusive GPU use verified.
[2026-09-13 18:03:45 UTC] Context study: 70B Q2_K/c8/8k passed 16 exact-512-output requests with matched Q4_K_M inputs: 21.23 vs 25.66 output tok/s, zero swap, and 23.51/23.88 GiB headroom. Both GPUs recorded software thermal limiting; report refreshing as 16k starts.
[2026-09-13 18:05:38 UTC] Context study: Report/plots refreshed with 12 validated 70B measurements, including Q2_K/c8/8k and software thermal limiting on both GPUs. Verified live 16k warmup with zero cgroup swap and exclusive GPU use.
[2026-09-13 18:07:28 UTC] Context study: Verified live 70B Q2_K/c8/16k warmup generating outputs after prompt processing, with exclusive GPU use and zero cgroup swap.
[2026-09-13 18:08:52 UTC] Context study: 70B Q2_K/c8/16k passed discarded warmup with zero server swap and over 13 GiB free on each GPU; its single measured 16-request run has started.
[2026-09-13 18:10:22 UTC] Context study: Verified live 70B Q2_K/c8/16k measurement processing prompts normally with exclusive GPU use and zero cgroup swap.
[2026-09-13 18:11:20 UTC] Context study: 70B Q2_K/c8/16k remains live and progresses through prompt processing without resource failures; throughput acceptance awaits the full request set.
[2026-09-13 18:12:20 UTC] Context study: Verified live 70B Q2_K/c8/16k run generating outputs, with no competing GPU process and zero cgroup swap.
[2026-09-13 18:13:48 UTC] Context study: 70B Q2_K/c8/16k responses are nearing completion within the active run; full 16-request result remains pending, with zero swap and exclusive GPU use verified.
[2026-09-13 18:15:09 UTC] Context study: Verified live 70B Q2_K/c8/16k measurement processing later requests with zero cgroup swap and exclusive GPU use.
[2026-09-13 18:16:06 UTC] Context study: 70B Q2_K/c8/16k remains live while processing later prompts; final measurement and subsequent larger-input capacity screens are pending.
[2026-09-13 18:17:54 UTC] Context study: 70B Q2_K/c8/16k final prompts finished prefill; remaining outputs are generating with zero cgroup swap and exclusive GPU use.
[2026-09-13 18:26:05 UTC] Context study: Status: 295/360 serving/capacity settings resolved (219 measured, 76 capacity-excluded); 65 remain, all 70B; Q2_K c16/p4096 active, Q4_K_M complete, IQ1_M/Q8_0 queued; full hardware profiling and controlled kernel diagnostics follow serving.
[2026-09-13 18:27:19 UTC] Context study: Paused at user request: serving service stopped and both GPUs idle; completed results preserved (295/360 settings resolved: 219 measured, 76 capacity-excluded), 65 settings remain; Q2_K 70B c16/p4096 interrupted before acceptance; full profiling and controlled diagnostics not started.
[2026-09-13 18:33:33 UTC] Context study: User authorized targeted 70B Q2_K/Q4_K_M hardware investigation while broader serving stays paused; canonical c8/p2048, 512-output traces started in isolated zero-swap service; existing inference reused; concise scope in results/cuda-context-study/diagnostics/70b-q2-q4-priority-plan.md.
[2026-09-13 18:35:41 UTC] Context study: Targeted Q4_K_M trace has fully loaded on both GPUs and reached warmup; 200ms clock/temperature/thermal telemetry is recording, and phase/tensor/shape attribution will drive matched counter selection; broader serving remains stopped.
[2026-09-13 18:36:58 UTC] Context study: Targeted Q4_K_M instrumented burst is progressing with both GPUs dedicated and zero service swap; prefill/decode attribution remains separate; no broader serving benchmarks resumed.
[2026-09-13 18:38:11 UTC] Context study: Q4_K_M targeted request burst finished; Nsight timeline export is in progress, with Q2_K trace queued next; hardware-counter attribution remains pending and no causal findings have been claimed.
[2026-09-13 18:39:24 UTC] Context study: Q4_K_M c8/p2048 timeline captured successfully; Q2_K timeline is loading next, while measured Q4 phase/tensor/shape dominance is being checked for counter targeting.
[2026-09-13 18:41:04 UTC] Context study: Q2_K trace burst is generating responses; Q4 trace shows separate gate/up matrix kernels at decode width 8 and fused gate/up in width-1 tails, which will be distinguished in matched-operation analysis.
[2026-09-13 18:42:04 UTC] Context study: Q2_K warmup completed and the recorded eight-request trace burst started; hardware counters will follow trace export; the remaining serving sweep stays paused.
[2026-09-13 18:43:27 UTC] Context study: Q2_K recorded trace burst is nearing the 512-token output limit; the profiler service is live and swap remains zero.
[2026-09-13 18:44:39 UTC] Context study: Both targeted trace bursts finished; Q2_K summary parsing remains active; Q4 matrix time is dominated by feed-forward gate/up and down projections, which will guide matched counter comparisons.
[2026-09-13 18:46:41 UTC] Context study: Both targeted 70B traces validated at 8 requests x 2048 input + 512 output on both GPUs; Q2 matched gate/up kernels are slower at widths 512 and 8; isolated Nsight Compute counter service started, broader inference still paused.
[2026-09-13 18:48:08 UTC] Context study: First hardware-counter capture is Q4_K_M prefill on GPU0; controlled selection saved from both fresh traces: gate/up Q2 versus Q4 and down Q3 versus Q4/Q6, with widths 1/8/9 plus matched prefill width512; broader sweep remains paused.
[2026-09-13 18:49:59 UTC] Context study: Q4_K_M first prefill/GPU0 counter run is live through warmup/capture; clock and thermal telemetry are recording alongside it for interpretation, with zero swap and no serving sweep resumed.
[2026-09-13 18:51:23 UTC] Context study: First Q4_K_M prefill/GPU0 counter run has reached decode; counter report finalization and sample checks will follow completion of the request burst.
[2026-09-13 18:54:02 UTC] Context study: First Q4_K_M prefill/GPU0 hardware report collected 20 launches; exact operation coverage is being completed using targeted missing samples, preserving the accepted capture and all original serving timings.
[2026-09-13 18:56:46 UTC] Context study: Primary-capture ordering is ready and identity-compatible with the accepted Q4 report; allowing the current exact-role capture to finish before switching to paired Q2/Q4 phase/GPU captures; full endpoint coverage remains explicitly incomplete.
[2026-09-13 18:58:36 UTC] Context study: Switched at an accepted capture boundary to paired primary Q2/Q4 captures; Q4 prefill/GPU0 primary and completed attention-Q supplement preserved, no partial capture to discard; new zero-swap primary counter service started, broader serving paused.
[2026-09-13 19:00:41 UTC] Context study: Q2_K matching prefill/GPU0 primary counter capture is running; controlled trace-selected matrix commands are prepared for execution after primary captures; accepted Q4 captures are reused.
[2026-09-13 19:02:21 UTC] Context study: Primary hardware captures: 1/8 complete; Q2_K prefill/GPU0 capture underway; no external GPU workload detected and service swap remains zero.
[2026-09-13 19:03:24 UTC] Context study: Q2_K first primary counter burst is still generating responses; paired counter analysis awaits report finalization, with existing inference results unchanged.
[2026-09-13 19:05:01 UTC] Context study: Primary hardware captures: 2/8 complete; first Q2/Q4 prefill GPU0 comparison available, while Q4_K_M decode/GPU0 capture starts.
[2026-09-13 19:05:34 UTC] Context study: First paired prefill/GPU0 result, pooled 5 same-shape gate/up launches per format: Q2 reads 30.0% fewer DRAM bytes yet executes 69.6% more instructions and takes 48.5% longer; L2 hit rate is higher, occupancy ~16.7% in both, no measured local-memory spills; decode/GPU1 confirmation pending.
[2026-09-13 19:08:48 UTC] Context study: FINDINGS.md now records the measured GPU0 prefill mechanism: Q2 uses 16-weight scale/offset groups versus Q4's 32 and executes more partial-product/correction instructions; first paired counter table added with explicit scope, replay caveat, and pending decode/GPU1/controlled checks.
[2026-09-13 19:09:27 UTC] Context study: Measured prefill explanation and table are now in FINDINGS.md; Q4_K_M decode/GPU0 counter collection remains active, and decode attribution is still provisional pending its Q2 comparison.
[2026-09-13 19:10:57 UTC] Context study: Q4_K_M decode/GPU0 counter collection is progressing with profiling overhead; replay/instrumented durations remain diagnostic and do not replace completed unprofiled inference throughput.
[2026-09-13 19:12:23 UTC] Context study: Q4_K_M decode/GPU0 capture is nearing completion; paired prefill counter bursts showed no sampled software/hardware thermal limiting, while automatic SM clocks varied, so attribution also uses instruction counts and traffic.
