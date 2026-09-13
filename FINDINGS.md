# Datacenter GPUs: A Quantization Perspective

CUDA serving study · 12 September 2026

The [context/concurrency extension](results/cuda-context-study/report/index.md)
uses the [approved plan](FUTURE_WORK.md), with 512 generated tokens
and two GPUs for every model size. Its report separates completed measurements,
capacity exclusions, and pending profiling. The interim extension results below
cover completed settings; findings from the earlier 2k-input/128-output study follow.
The extension is **not complete**; current inference and profiling status appears
in the linked report. The section labeled **Earlier 2k-input/128-output study**
uses its original workload and GPU layout.
The amended extension compares IQ1_M, Q2_K, Q4_K_M, Q8_0 and feasible FP16 for
all three sizes, using **one measured run per setting**. Existing validated first
runs are reused; previous extra repetitions are excluded from these comparisons.
The broader serving sweep is paused. The requested [70B Q2/Q4 hardware investigation](results/cuda-context-study/diagnostics/q2-q4-priority/README.md)
is complete for the selected 2k-input/eight-client operations on both GPUs,
including paired counters and isolated width/dispatch controls.

**FP16 KV has a different per-token footprint in each model.** Summed across
both GPUs, each allocated token position in one request consumes
[32 KiB for 1B](results/cuda-context-study-1b/FP16/c8/p2048/r1/placement.json),
[128 KiB for 8B](results/cuda-context-study-8b/FP16/c8/p2048/r1/placement.json),
and [320 KiB for 70B](results/cuda-context-study-70b/Q4_K_M/c8/p2048/r1/placement.json).
These values are verified against the loaded GPU KV buffers and allocated slot
counts. Total KV equals clients × padded positions per slot × bytes per position;
weight quantization leaves this FP16 KV term unchanged. Under this allocation
rule, the **calculated** KV increase from 2k to 4k input at 64 clients is
**4 GiB, 16 GiB, and 40 GiB**, respectively, before weights and working buffers.

**`--tensor-split 1,1` still gives GPU0 more KV.** The pinned backend divides
the transformer layers plus the output layer between devices; the output layer
has weights but no KV cache. Its [assignment rule](vendor/llama.cpp/src/llama-model.cpp#L1494)
gives **9/7, 17/15, and 41/39 transformer layers on GPU0/GPU1** for 1B, 8B,
and 70B. The loaded KV buffers agree with those counts. At 8B FP16, 64 clients,
and 8k input, [KV occupies 37.1875 / 32.8125 GiB](results/cuda-context-study-8b/FP16/c64/p8192/r1/placement.json):
GPU0 carries **4.375 GiB more KV**, before weight and working-buffer differences.
The measured minimum headroom is [2.77 / 7.01 GiB](results/cuda-context-study-8b/FP16/c64/p8192/r1/resources.json).
Capacity therefore needs a per-device check for this layout; pooled free VRAM
does not establish that every device has enough room.

**1B inference and capacity testing is complete:** all five formats have completed the
80 main measurements and 35 feasible capacity-extension measurements; five
predicted capacity exclusions finish the 1B grid. Hardware profiling remains
incomplete. At 64 clients and 16,384 input tokens, the selected first runs show:

| Format | Output tok/s | Versus FP16 | GPU0 software thermal limiting, active samples |
|---|---:|---:|---:|
| FP16 | 469.43 | — | 49/696 (7.04%) |
| Q8_0 | 466.49 | −0.63% | 70/701 (9.99%) |
| Q4_K_M | 458.24 | −2.38% | 96/713 (13.46%) |
| Q2_K | 417.69 | −11.02% | 151/783 (19.28%) |
| IQ1_M | 420.07 | −10.51% | 134/778 (17.22%) |

Thermal samples cover only the measured HTTP request windows. GPU1's thermal
flags and GPU0's hardware thermal flag stayed inactive. These runs experienced
different thermal exposure. One run does not establish repeatability, an intrinsic
format advantage, or a thermal penalty.
Fresh comparative kernel profiling remains incomplete; the first
[1B Q4 calibration](results/cuda-context-study-1b/profiles/Llama-3.2-1B-Instruct/Q4_K_M/c8/p2048/)
has captured prefill and decode on both GPUs, but still needs samples of selected
operations before that endpoint is complete. See the [current summary](results/cuda-context-study-1b/summary.md)
and `r1`'s `raw.json`/`telemetry.csv` in the
[FP16 setting](results/cuda-context-study-1b/FP16/c64/p16384/),
[Q8 setting](results/cuda-context-study-1b/Q8_0/c64/p16384/),
[Q4 setting](results/cuda-context-study-1b/Q4_K_M/c64/p16384/),
[Q2 setting](results/cuda-context-study-1b/Q2_K/c64/p16384/) and
[IQ1_M setting](results/cuda-context-study-1b/IQ1_M/c64/p16384/).

For **all five 1B formats**, the largest validated power-of-two input is
**65,536 tokens at 8/16/32 clients**, and **32,768 tokens at 64 clients**, with
512 output tokens per request and the required per-GPU memory reserve.
At 64 clients, the 65,536-input setting was excluded before launch: padded FP16
KV alone requires **129.5 GiB** across the two GPUs, before weights and working
buffers. See the [maximum-input table](results/cuda-context-study/report/max-inputs.csv)
and [Q8 capacity estimate](results/cuda-context-study-1b/Q8_0/c64/p65536/capacity-estimate.json).
The [IQ1_M](results/cuda-context-study-1b/IQ1_M/c64/p32768/r1/) and
[Q2_K](results/cuda-context-study-1b/Q2_K/c64/p32768/r1/) 64-client, 32k runs
each completed all 64 requests with exactly 512 output tokens.

**Early extension scaling anomaly, 1B at 2,048 input tokens:** increasing
concurrency from 8 to 16 reduced IQ1_M's aggregate output throughput in these
runs, while Q2 and Q4 increased:

| Format | 8 clients, output tok/s | 16 clients, output tok/s | Change |
|---|---:|---:|---:|
| Q4_K_M | 1,092.40 | 1,355.66 | +24.10% |
| Q2_K | 1,011.31 | 1,399.11 | +38.35% |
| IQ1_M | 1,117.03 | 1,005.60 | −9.98% |

All six runs completed two requests per client with exactly 512 output tokens.
Neither GPU recorded a thermal-limit flag in their measured request windows.
This identifies a scaling reversal to explain; fresh profiles must establish the
actual activation widths and kernel costs before assigning its cause. Evidence:
`r1/raw.json` and `r1/telemetry.csv` under the Q4
[8-client](results/cuda-context-study-1b/Q4_K_M/c8/p2048/) and
[16-client](results/cuda-context-study-1b/Q4_K_M/c16/p2048/) settings, Q2
[8-client](results/cuda-context-study-1b/Q2_K/c8/p2048/) and
[16-client](results/cuda-context-study-1b/Q2_K/c16/p2048/) settings, and IQ1_M
[8-client](results/cuda-context-study-1b/IQ1_M/c8/p2048/) and
[16-client](results/cuda-context-study-1b/IQ1_M/c16/p2048/) settings.

Q2 moved from **7.42% below Q4 at eight clients to 3.20% above it at 16 clients**
with 2k input. That advantage did not persist at 4k input: at 16 clients,
[Q2](results/cuda-context-study-1b/Q2_K/c16/p4096/r1/) measured **1,042.72 tok/s**
versus [Q4](results/cuda-context-study-1b/Q4_K_M/c16/p4096/r1/) at **1,092.10 tok/s**,
a **4.52% Q2 deficit**. Neither 4k run recorded a thermal-limit flag. These are
single-run ranking changes; the kernel cause remains to be measured.

**8B inference and capacity testing is complete for all five formats:** each
has 15 main measurements, three capacity-extension measurements, and six
predicted capacity exclusions: 90 measurements and 30 exclusions in total.

The completed sweeps have the same largest validated power-of-two inputs.
Headroom below is the minimum sampled free memory on **GPU0 / GPU1, in GiB**;
each value links to its measured run.

| Clients | Maximum input tokens | Padded FP16 KV, both GPUs | FP16 headroom | Q8_0 headroom | Q4_K_M headroom | Q2_K headroom | IQ1_M headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 65,536 | 64.75 GiB | [5.34 / 9.26](results/cuda-context-study-8b/FP16/c8/p65536/r1/) | [8.60 / 12.59](results/cuda-context-study-8b/Q8_0/c8/p65536/r1/) | [10.20 / 14.11](results/cuda-context-study-8b/Q4_K_M/c8/p65536/r1/) | [11.00 / 14.82](results/cuda-context-study-8b/Q2_K/c8/p65536/r1/) | [11.32 / 15.19](results/cuda-context-study-8b/IQ1_M/c8/p65536/r1/) |
| 16 | 32,768 | 65.50 GiB | [5.07 / 9.03](results/cuda-context-study-8b/FP16/c16/p32768/r1/) | [8.33 / 12.37](results/cuda-context-study-8b/Q8_0/c16/p32768/r1/) | [9.93 / 13.88](results/cuda-context-study-8b/Q4_K_M/c16/p32768/r1/) | [10.72 / 14.60](results/cuda-context-study-8b/Q2_K/c16/p32768/r1/) | [11.05 / 14.96](results/cuda-context-study-8b/IQ1_M/c16/p32768/r1/) |
| 32 | 16,384 | 67.00 GiB | [4.33 / 8.39](results/cuda-context-study-8b/FP16/c32/p16384/r1/) | [7.59 / 11.72](results/cuda-context-study-8b/Q8_0/c32/p16384/r1/) | [9.20 / 13.24](results/cuda-context-study-8b/Q4_K_M/c32/p16384/r1/) | [9.99 / 13.96](results/cuda-context-study-8b/Q2_K/c32/p16384/r1/) | [10.31 / 14.32](results/cuda-context-study-8b/IQ1_M/c32/p16384/r1/) |
| 64 | 8,192 | 70.00 GiB | [2.77 / 7.01](results/cuda-context-study-8b/FP16/c64/p8192/r1/) | [6.03 / 10.35](results/cuda-context-study-8b/Q8_0/c64/p8192/r1/) | [7.63 / 11.87](results/cuda-context-study-8b/Q4_K_M/c64/p8192/r1/) | [8.42 / 12.58](results/cuda-context-study-8b/Q2_K/c64/p8192/r1/) | [8.75 / 12.94](results/cuda-context-study-8b/IQ1_M/c64/p8192/r1/) |

At eight and 16 clients, these capacity runs used one request per client.
At 32 and 64 clients, the maxima reuse the main runs of two requests per
client. Every linked result has exactly 512 output tokens per request,
matching rendered inputs across formats, full transformer/KV GPU placement,
and zero sampled server swap.

Q8_0 leaves **3.26–3.34 GiB more headroom per GPU** than FP16 at these maxima;
Q4_K_M leaves **4.85–4.87 GiB more** across all four concurrencies.
Q2_K leaves **5.57–5.66 GiB more**; IQ1_M leaves **5.93–5.99 GiB more**.
These savings do not increase the maximum power-of-two input length.
At 16, 32, and 64 clients, the unchanged FP16 KV cache alone blocks the next
length. At 16 clients,
[64k was excluded for Q4_K_M](results/cuda-context-study-8b/Q4_K_M/c16/p65536/capacity-estimate.json)
as well as FP16, Q8_0, Q2_K, and IQ1_M: KV alone requires **129.5 GiB** before weights and
working buffers. At 32 clients, [32k input](results/cuda-context-study-8b/Q4_K_M/c32/p32768/capacity-estimate.json)
requires **131 GiB** of KV, and 64k requires **259 GiB**. At 64 clients,
[16k input](results/cuda-context-study-8b/Q4_K_M/c64/p16384/capacity-estimate.json)
requires **134 GiB**; 32k and 64k require **262 GiB** and **518 GiB**.
At eight clients, the next input length, 131,072, leaves no room for the
512 outputs within the model's 131,072-token context limit.
Remaining 70B inference and hardware profiling are tracked in the
[completion table](results/cuda-context-study/report/index.md).

**8B IQ1_M loses aggregate throughput when concurrency increases from eight
to 16 clients at 2,048 input tokens.** Q4_K_M and Q2_K move in the opposite direction:

| Format | Eight clients, output tok/s | 16 clients, output tok/s | Change |
|---|---:|---:|---:|
| Q4_K_M | [342.02](results/cuda-context-study-8b/Q4_K_M/c8/p2048/r1/) | [491.74](results/cuda-context-study-8b/Q4_K_M/c16/p2048/r1/) | +43.77% |
| Q2_K | [295.95](results/cuda-context-study-8b/Q2_K/c8/p2048/r1/) | [468.37](results/cuda-context-study-8b/Q2_K/c16/p2048/r1/) | +58.26% |
| IQ1_M | [353.21](results/cuda-context-study-8b/IQ1_M/c8/p2048/r1/) | [242.94](results/cuda-context-study-8b/IQ1_M/c16/p2048/r1/) | −31.22% |

IQ1_M changes from **3.27% above Q4_K_M** to **50.60% below it**. Q2_K's deficit
against Q4_K_M narrows from **13.47% to 4.75%**. Each cell is one measured run
of two requests per client, with matched rendered inputs and exactly 512 output
tokens per request. Neither GPU recorded an active software or hardware
thermal-limit flag in these six measured HTTP windows (120, 166, 138, 174, 115,
and 336 samples per GPU in table order). These single-run scaling differences
still require the planned profiling to establish actual matrix widths, kernel
paths, and causes. They do not establish run-to-run repeatability.

**70B Q4_K_M serving and capacity testing is complete:** nine measured settings
and 15 predicted capacity exclusions resolve its grid. The other 70B quantized
formats remain incomplete and paused. The targeted Q2/Q4 hardware investigation
below is complete; the broader profiling grid remains pending.

| Clients | Largest validated input | FP16 KV, both GPUs | Minimum sampled headroom, GPU0 / GPU1 | Output tok/s at that input |
|---|---:|---:|---:|---:|
| 8 | [16,384](results/cuda-context-study-70b/Q4_K_M/c8/p16384/r1/) | 41.875 GiB | 5.67 / 6.85 GiB | 15.54 |
| 16 | [8,192](results/cuda-context-study-70b/Q4_K_M/c16/p8192/r1/) | 43.75 GiB | 4.74 / 5.96 GiB | 33.41 |
| 32 | [4,096](results/cuda-context-study-70b/Q4_K_M/c32/p4096/r1/) | 47.5 GiB | 2.83 / 4.15 GiB | 65.01 |
| 64 | [2k and every larger tested input excluded](results/cuda-context-study-70b/Q4_K_M/c64/p2048/capacity-estimate.json) | 55 GiB required at 2k, calculated | — | — |

**Equal input tokens across slots do not imply equal KV capacity.** The four
configurations 8×16k, 16×8k, 32×4k, and 64×2k each specify 131,072 input tokens
across active slots. Every slot also reserves space for 512 outputs plus the guard
and allocation padding. Here the padded slot size is `input + 768`, so the
additional KV reservation grows from **1.875 GiB at eight clients to 15 GiB at
64 clients**, on top of the same 40 GiB input component. The first three KV totals
are verified from loaded buffers; the 64-client case was screened before launch.
At 64 clients and 2k input, GPU weights (39.04 GiB), KV (55 GiB), and the two
2 GiB reserves already require **98.04 GiB**, before working buffers, versus
**94.77 GiB available**. This explains the capacity exclusion despite an unchanged
input-token total. Each measured maximum used two requests per client, exact
512-token outputs, full transformer/KV GPU placement, and zero sampled swap.
GPU0 recorded software thermal limiting in all three measurements; throughput
values are single-run observations.

**70B Q2_K saves VRAM but loses throughput at eight clients and 2k input.**
The first matched extension runs, with 512 output tokens per request, show:

| Format | Output tok/s | TTFT p95 | TPOT p95 | Sampled peak device VRAM, GPU0 / GPU1 |
|---|---:|---:|---:|---:|
| [Q4_K_M](results/cuda-context-study-70b/Q4_K_M/c8/p2048/r1/) | 45.92 | 19.58 s | 168.48 ms | 23.74 / 23.44 GiB |
| [Q2_K](results/cuda-context-study-70b/Q2_K/c8/p2048/r1/) | 38.48 | 25.30 s | 200.84 ms | 16.18 / 16.19 GiB |

Q2_K's output throughput is **16.2% lower**, despite saving **7.56 / 7.25 GiB**
at the respective device peaks. Its p95 TTFT and TPOT are **29.2% and 19.2%
higher**. Both runs completed all 16 requests with identical rendered inputs,
exact outputs, full transformer/KV placement, and zero sampled swap. GPU0 recorded
software thermal limiting in both; neither GPU recorded hardware thermal limiting,
and GPU1 recorded no software thermal limiting. These are single-run observations;
the hardware follow-up below identifies the extra matrix-kernel work.

**70B hardware cause: Q2's smaller memory traffic is outweighed by extra instruction work.**
The completed paired captures compare real-model gate/up projections at
`M=28672, K=8192`: pure prefill uses `N=512` and pure decode `N=8`.
Each format/device/phase contributes five launches from one capture, pooling
three gate and two up projections with identical unfused geometry. Width means
token vectors in one matrix operation; 512 here is the prefill processing chunk,
within a 2,048-token input.

| Phase | GPU | Q4 → Q2 median kernel duration | Q2 instructions | Q2 DRAM reads | Q2 duration |
|---|---:|---:|---:|---:|---:|
| Prefill | 0 | 2.264 → 3.363 ms | +69.6% | −30.0% | +48.5% |
| Prefill | 1 | 2.246 → 3.323 ms | +69.6% | −30.1% | +48.0% |
| Decode | 0 | 318.24 → 402.30 µs | +35.2% | −41.6% | +26.4% |
| Decode | 1 | 316.77 → 398.53 µs | +35.2% | −41.6% | +25.8% |

The pinned [quantization layouts](vendor/llama.cpp/ggml/src/ggml-common.h#L293)
use scale/offset groups of **16 weights for Q2 versus 32 for Q4**. The NVIDIA
[Q2 matrix path](vendor/llama.cpp/ggml/src/ggml-cuda/mmq-vec-dot.cuh#L751)
performs smaller partial products and additional offset corrections; the
[Q4 path](vendor/llama.cpp/ggml/src/ggml-cuda/mmq-vec-dot.cuh#L363) handles larger
groups. Prefill counters show **1.875× floating-point multiply-adds** and
**2.25× integer-to-float conversions**; decode shows **1.8× and 2×**, respectively.
This locates substantial overhead in scaling/conversion/correction inside the
quantized kernels. Both formats handle quantization inside their prefill/decode
kernels. Their prefill path avoids the separate FP16 weight reconstruction
observed for IQ1_M tensors.

**Lower L2 hit rate does not explain the decode penalty.** Q2's L1 hit rate is
higher and absolute DRAM traffic is lower. On GPU0, long-scoreboard stalls fall
from **0.75 to 0.07 cycles per issued instruction**, while instruction-fetch
stalls remain **0.02** for both formats. Decode registers rise **125 → 151** per
thread, reducing the register-limited block count **8 → 6** and achieved occupancy
from about **32% to 24%** on both GPUs. Neither kernel records local-memory spills.
Prefill occupancy is essentially unchanged, about **16.7%**. Register pressure is
measured, but its separate contribution to latency is not isolated; occupancy
alone cannot explain performance. [Metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#sets-and-sections).

**A dispatch change isolates a large avoidable cost at eight columns.** Stock
A6000 dispatch uses the quantized matrix-vector kernel (MMVQ) through width 8 and
switches to the quantized matrix-matrix kernel (MMQ) at width 9. In isolated,
unprofiled gate/up tests, Q2 beats Q4 at width 1, loses at width 8, and wins again
at width 9. Changing only Q2/Q3's width-8 dispatch to MMQ gives:

| Controlled operation | GPU0 stock → earlier MMQ | GPU1 stock → earlier MMQ | Total-path instructions | DRAM reads |
|---|---:|---:|---:|---:|
| Gate/up, Q2_K | 485.70 → 214.29 µs | 485.26 → 211.92 µs | −78.3% | +1.1% |
| Down, Q3_K | 542.55 → 297.72 µs | 534.38 → 296.25 µs | −69.9% | +1.4% |

Thus the Q2 gate/up operation takes **56% less time (2.27–2.29× speedup)** with
almost unchanged DRAM reads. The faster MMQ kernel actually has lower occupancy,
which strengthens the explanation based on dispatch and instruction work.
The Q2_K model uses Q3_K down projections; Q4_K_M uses a Q4_K/Q6_K mixture,
so those types were tested separately. These controls use private diagnostic
binaries; the serving binary and existing inference measurements are unchanged.
[Full measurements, width plot, and raw counter links](results/cuda-context-study/diagnostics/q2-q4-priority/README.md).

**Scope:** eight primary server captures and 44 isolated cases, each with one
timing run and one counter capture, completed. Isolated timing is the median of
20 internal synchronized operations; each counter case captures five operations.
Experiments ran serially with swap disabled and no competing GPU workload.
Counters use kernel replay with cache/clock control set to `none`; their timings
do not replace unprofiled serving throughput. The width-1 harness is an unfused
single-matrix reference, whereas serving can fuse gate/up at width 1. These
results establish kernel mechanisms, without assigning exact shares of the
whole-model slowdown or reproducing the earlier 128-output aggregate percentages.
Full endpoint profiling remains deferred with the broader study.

**Earlier 2k-input/128-output study:** the following results use the original
workload and hardware layout described below.

**Smaller weights do not guarantee higher serving throughput.** For 1B and 8B, IQ1_M
wins with one client, loses with eight uncached clients, and wins again with
prefix reuse. For 70B, Q2_K wins with one client, but Q4_K_M wins with eight
clients in both cache modes.

Llama 3.2 1B and Llama 3.1 8B Instruct used **one RTX A6000 48 GB**;
Llama 3.3 70B Instruct used **two**, with `--split-mode layer --tensor-split 1,1`. This is
cloud-style HTTP serving on workstation GPUs. Pinned llama.cpp `3f5e94d7c2ab`,
CUDA 13.0, full transformer offload, eight slots, 32,768 total context capacity
(4,096 tokens per slot, including input and output),
FP16 KV, FlashAttention, and token batch/microbatch 2048/512 were fixed;
idle-slot RAM caching was disabled. Every request had 2,048 input and exactly
128 generated tokens. Three repetitions of 16 requests per setting produced
**1,536 validated measured requests**, excluding warmup. Client and server
records verify concurrency.

**Earlier-study context coverage:** these are **2,048-input-token** experiments,
with 2,176 total input-plus-output tokens per request. Allocating a 4,096-token
slot does not constitute a 4k-length workload. A 4k–128k length sweep was not part
of these earlier measurements; the new extension linked above is separate and
still incomplete. The rankings below do not establish performance over that
requested range. The one-client runs also kept eight slots and the same 4,096-token
per-request limit; one client did not receive the full 32,768-token pool.

**Without prefix reuse**, aggregate output tokens/s were:

| Model | Format | One client | Eight clients |
|---|---|---:|---:|
| 1B | Q8_0 | 281.15 ± 1.35 | 686.53 ± 10.92 |
| 1B | Q4_K_M | 343.41 ± 1.59 | 674.13 ± 4.76 |
| 1B | IQ1_M | 348.59 ± 0.77 | 642.33 ± 2.97 |
| 8B | Q8_0 | 59.85 ± 0.15 | 163.33 ± 1.01 |
| 8B | Q4_K_M | 81.65 ± 0.03 | 158.29 ± 0.35 |
| 8B | IQ1_M | 93.05 ± 0.05 | 150.67 ± 0.52 |
| 70B, two GPUs | Q4_K_M | 12.03 ± 0.01 | 27.23 ± 0.09 |
| 70B, two GPUs | Q2_K | 13.24 ± 0.01 | 22.38 ± 0.02 |

Values are means ± sample standard deviation; latency percentiles pool 48 requests
per setting. The 1B Q8/Q4 eight-client difference is only 1.8%; treat them as close.
For 70B, Q2 wins by **10.1%** at one client, while
Q4 wins by **21.7%** at eight. Q2 reduces combined allocated VRAM from about
**50.3 to 35.5 GiB**, but does not maximize concurrent throughput.

**IQ1_M is not universally fastest.** Among the tested formats, its 1B/8B
throughput ranks first with one uncached client and last with eight uncached
clients; with eight clients and warm prefix reuse it ranks first again. For
example, at 8B and eight uncached clients, Q8_0 produces **163.33 tok/s**, Q4_K_M
**158.29 tok/s**, and IQ1_M **150.67 tok/s**. “Fewer bits need not mean faster”
means reducing weight precision can either help or hurt, depending on the
workload and kernel implementation. It does not mean lower precision is always
faster at higher concurrency. IQ1_M was not tested at 70B, and Q2_K was not
tested at 1B/8B; these results cannot rank IQ1_M against Q2_K for the same model.

Concurrency increases aggregate throughput while slowing individual responses.
For 70B Q2, median time per output token rises from **51.53 to 282.26 ms**;
Q4 rises from **64.71 to 234.81 ms**. Eight-client p95 time to first token reaches
20.14 seconds for Q4 and 25.83 seconds for Q2, including scheduling.

**With prefix reuse**, eight clients sent identical prompts in both modes.
Every reuse-on request used **2,047 cached tokens and evaluated one token**;
reuse-off evaluated all 2,048. Primed slots provide best-case GPU KV hits;
production hit rates and global prefix deduplication were not tested.

| Model | Format | Reuse off, tok/s | Reuse on, tok/s | Gain | TTFT p95, off → on |
|---|---|---:|---:|---:|---:|
| 1B | Q8_0 | 684 | 1267 | 1.85× | 711 → 30 ms |
| 1B | Q4_K_M | 672 | 1262 | 1.88× | 733 → 22 ms |
| 1B | IQ1_M | 638 | 1339 | 2.10× | 842 → 27 ms |
| 8B | Q8_0 | 164 | 387 | 2.36× | 3607 → 44 ms |
| 8B | Q4_K_M | 159 | 373 | 2.35× | 3748 → 39 ms |
| 8B | IQ1_M | 151 | 429 | 2.84× | 4364 → 40 ms |
| 70B, two GPUs | Q4_K_M | 27.32 | 57.14 | 2.09× | 20029 → 237 ms |
| 70B, two GPUs | Q2_K | 22.39 | 49.90 | 2.23× | 25791 → 236 ms |

Throughput counts generated tokens. Reuse reverses IQ1_M's ranking at 1B/8B,
but Q4 remains **14.5% faster** than Q2 at 70B. Removing most prefill does not
make the smallest recipe universally fastest.

**What the 30 kernel traces explain.** Nsight Systems captures exclude loading
and warmup. Times below sum kernel durations; they are diagnostic, not HTTP
latency or throughput. Six captures cover 70B on both GPUs.

- **All these formats require quantization handling.** Q2_K/Q4_K must unpack
  their low-bit values; Q8_0 already stores 8-bit integers; all must account for
  their scales and other applicable metadata. In this pinned A6000 build, their
  large-prompt MMQ kernels combine that work with integer matrix multiplication,
  without first materializing a full FP16 weight matrix in device memory.
  IQ1_M tensors have no supported MMQ path here: a separate kernel reconstructs
  FP16 weights into a temporary buffer, which a subsequent cuBLAS GEMM reads.
  This is a difference in **where and how quantization is handled**, not an
  absence of dequantization cost in Q2/Q4/Q8. The separate stages occupy
  **31.0% of 1B and 44.0% of 8B** baseline uncached eight-request kernel time.
  [Pinned MMQ support](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/mmq.cu#L266),
  [FP16 conversion and cuBLAS call](https://github.com/ggml-org/llama.cpp/blob/3f5e94d7c2ab2267fe39852051777fe30c1f49ef/ggml/src/ggml-cuda/ggml-cuda.cu#L1408).
- At the observed 1–8-column decode widths, **all four tensor types, including
  IQ1_M, use MMVQ**, with quantization handling inside the matrix-vector kernel.
  These kernels use DP4A; their gains do not demonstrate tensor-core acceleration.
  Format names identify mixed tensor recipes, so dispatch occurs per tensor.
- Reuse removes that prompt work. At 8B, IQ1_M summed kernel time falls
  **6,305 → 2,095 ms**, while matrix-vector time changes little,
  **1,524 → 1,478 ms**. The expensive prefill path explains its uncached penalty.
- **70B uses a different kernel path:** both Q2 and Q4 use fused quantized matrix
  multiplication for prefill. Q2's matrix-vector kernels take **22.5% less time**
  than Q4's with one request, but **16.8% more** with eight. Its eight-request
  prefill matrix kernels also take **32.7% more** time. These are observed kernel
  costs from the earlier workload. The new 512-output hardware follow-up above
  identifies extra scaling/conversion instructions on both GPUs. Its dispatch
  control cuts isolated Q2 width-8 operation time by 56% at nearly unchanged
  DRAM traffic; this is an operation-level result, not a serving speedup.
- With 70B prefix reuse, large-prompt matrix multiplication disappears, yet Q2's
  matrix-vector kernels still take **18.40 s versus Q4's 15.78 s**, or **16.6%
  longer**. Almost 99% of their matrix-vector calls process eight columns.
  This is a batched-decode disadvantage, not IQ1_M's separate dequantization path.
- **Two GPUs provide capacity, but decode runs their layer partitions in
  sequence.** Both cached 70B traces show **zero simultaneous GPU kernel activity**;
  uncached single-request traces show overlap during prompt processing. This describes our
  layer split, not the scaling of other parallel strategies.
- At eight requests, 70B peer-copy calls transfer **570.2 MB** of logical
  activations without reuse and **33.6 MB** with reuse. Both also copy **525.3 MB**
  of logits to CPU sampling. Paired activity records are counted once per peer
  call. The physical transfer route is unresolved; these are not DRAM bandwidth
  or NVLink utilization measurements.

Q2_K is a mixed roughly three-bit recipe, distinct from IQ1_M; Q4_K_M averages
4.82 payload bits/parameter. The 70B converter revisions are unknown.

**Why Q2_K throughput is below Q4_K_M at eight clients:** the traces locate the
extra time in both matrix paths. Uncached prefill MMQ totals **34.08 s for Q2
versus 25.67 s for Q4**; MMVQ totals **18.86 s versus 16.15 s**. Even with most
prefill removed, Q2 remains slower, as the prefix results above show. Smaller
stored weights therefore do not compensate for the observed kernel costs
under this workload. The newer paired prefill counters above now identify extra
scaling/correction instruction work in the matched feed-forward projections.
The completed isolated width tests reproduce a Q2 advantage at one column and
disadvantage at eight. Switching Q2 to MMQ at eight cuts instructions by 78.3%
and operation time by about 56%, with nearly unchanged DRAM reads. This identifies
a dispatch-dependent arithmetic cost; the unfused control does not quantify the
earlier single-client fused aggregate.
The single-client result goes the other way: Q2's shorter matrix-vector time
outweighs its slower prefill and gives higher end-to-end throughput.
[Trace comparisons](results/70b-audit.json).

**1B hardware counters: measured.** With the DCGM exporter paused, Nsight Compute
2026.3.0 collected **166 kernel samples across 23 captures**, requesting 28 metrics
per sample. Q8_0, Q4_K_M and IQ1_M cover prefill, one/eight-column matrix-vector
work and prefix reuse, including secondary tensor types. All **142 profiled HTTP
requests** passed the 2,048-input/128-output checks; reuse requests cached 2,047
tokens. These diagnostics are separate from the 1,536 serving measurements above.
The wrapper now recognizes both `.ncu-repz` and `.ncu-rep` reports and both raw
CSV layouts; a valid new-format report was initially misclassified as missing.

The following are unweighted medians for **selected kernel launches at eight
clients**, with sample counts. They are not whole-model averages or comparisons
of identical matrix dimensions. Replay flushed hardware caches between passes;
GPU clocks remained automatic. Prefix reuse refers to model KV reuse, separately
from this hardware-cache policy.

| Selected 1B kernel | Samples | DRAM GB/s | L1 hit % | L2 hit % | Occupancy % | SM IPC active / elapsed | Tensor active % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 prefill MMQ | 2 | 327.33 | 54.76 | 80.12 | 17.24 | 1.12 / 0.94 | 24.70 |
| Q4_K prefill MMQ | 2 | 214.88 | 37.86 | 85.19 | 16.80 | 1.29 / 1.06 | 21.22 |
| IQ1_M dequantization | 2 | 600.43 | 83.15 | 97.37 | 29.18 | 0.93 / 0.90 | 0.00 |
| IQ1_M FP16 GEMM | 4 | 435.76 | 0.56 | 89.12 | 15.29 | 0.64 / 0.56 | 76.75 |
| Q8_0 eight-column MMVQ | 2 | 599.22 | 93.24 | 26.05 | 38.59 | 1.98 / 1.87 | 0.00 |
| Q4_K eight-column MMVQ | 2 | 381.01 | 95.37 | 32.27 | 28.45 | 2.49 / 2.33 | 0.00 |
| IQ1_M eight-column MMVQ | 2 | 147.09 | 99.01 | 51.00 | 21.06 | 1.90 / 1.79 | 0.00 |

DRAM GB/s is measured read-plus-write bytes divided by kernel nanoseconds.
Cache hits count sectors; IPC counts warp instructions per SM cycle. Occupancy
and tensor activity use active-cycle denominators. MMQ rows select tile 128;
dequantization selects grid 65,536; GEMM selects the main `h1688` kernel at
grid (64,4,2); MMVQ selects grid (4096,1,1). The capture gate covers the warmed
HTTP burst: column selectors can include final-prefill work as well as decode.
Exact specializations, launch shapes, ranges, elapsed tensor activity, clocks,
one-client/reuse results and all mixed types are in the
[hardware report](results/cuda-study-1b/hardware-counters/README.md) and
[per-launch values](results/cuda-study-1b/hardware-counters/hardware-metrics.csv).

The selected IQ1_M dequantization samples spend **60.22% of active-warp states**
waiting on long-scoreboard dependencies, with **23.15% issue-active cycles**.
Its separate FP16 GEMM nevertheless reaches **76.75% tensor activity**: the
prefill penalty does not imply that its GEMM fails to use tensor cores.
Every sampled one/eight-column MMVQ has zero tensor activity. Global-load
bytes per sector in the three tabled MMVQ groups are **9.19 / 9.33 / 3.91**
(Q8/Q4/IQ1; reported maximum 32). These access-pattern measurements are not the
AMD poster's coalescing percentage, nor proof of a whole-model bottleneck.

One short dequantization launch reported an impossible **101.64% L2 hit rate**;
the raw value is retained and flagged, excluded from bounded-percentage summaries
and absent from the selected table. A [reduced-metric retry](results/cuda-study-1b/hardware-counters/l2-precision/precision-validation.json)
reduced replay passes from 12 to three but still reported 100.32% on one short
launch. This precision limit remains: 88 of the main 166 launches are under
20 microseconds. See NVIDIA's [range and precision guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision).
The [independent audit](results/cuda-study-1b/hardware-counters/independent-audit.json) verifies
all requested raw values, model hashes, request lengths/cache behavior and gates.
These measurements resolve counter access and provide 1B hardware values.
Those 1B measurements alone do not establish hardware bottlenecks for 8B or 70B.

**Which source findings transfer?**

- **Reducing weight bits does not guarantee faster inference: supported for
  this 2k-input workload.** The counterexamples are IQ1_M losing to Q4/Q8 at
  eight uncached clients, and 70B Q2_K losing to Q4_K_M at eight clients in both
  reuse modes. Low-bit formats also win in other settings, including the
  single-client runs. This supports a workload-dependent ranking, not a rule
  that either fewer or more bits always wins. Our 1B/8B single-client results
  agree with the paper's separate favorable A6000 low-bit observation.
  [Author preprint](https://arxiv.org/pdf/2508.08531).
- **Q8's prefill advantage: directionally supported at 1B/8B.** Q8 has the lowest
  single-client TTFT, while IQ1 loses at uncached concurrency eight. Our 70B
  Q4/Q2 experiment does not replicate the poster's MI300X Q8/IQ1 comparison.
  Its internal token batches 128/256 are distinct from HTTP concurrency.
  [Supplied poster](references/Tapia.pdf), [January 3 article](https://medium.com/@afsara.benazir/profiling-quantized-llms-on-amd-mi300x-with-rocm-what-the-hardware-is-really-telling-us-ac3cc33ebcff).
- **IPC, cache, bandwidth and execution activity: now measured for selected 1B
  kernels.** The values above replace the previous access blocker. NVIDIA IPC
  and load-sector metrics do not directly reproduce AMD's IPC/coalescing values;
  no whole-workload roofline or 8B/70B counter conclusion follows from these
  1B measurements. The separate 70B investigation appears above. January 10's
  roofline concerns 1B Q4_0, separately from its 70B discussion.
  [Article](https://medium.com/@afsara.benazir/what-rocm-profiling-revealed-about-quantized-llms-on-amds-fastest-gpu-c0edfab9624f).

**Context scope:** the earlier results in this section use 2,048 input and
128 output tokens. The long-context extension follows the [approved plan](FUTURE_WORK.md):
power-of-two **input** lengths, 512 outputs, and 8/16/32/64 clients, with capacity
limits checked for each setting. Its 1B and 8B serving/capacity sweeps are complete;
remaining 70B inference and broader profiling are paused. The targeted 70B
Q2/Q4 counters and width/dispatch diagnostics above are complete. The [current report](results/cuda-context-study/report/index.md)
records their separate completion states. A 131,072-token input plus 512 outputs
exceeds the models' native 131,072-token context and is recorded as unsupported.

These earlier runs use short closed-loop bursts over loopback, automatic clocks and default
300 W GPU limits. GPU 0 accrued software thermal clock limiting during broader
70B study windows, including loading and warmup; snapshots cannot assign its
effect to individual settings. Small repetition variance does not remove this
limitation. Accuracy, realistic arrivals and long contexts were not tested.
Different model versions and GPU counts prevent controlled cross-size or scaling
conclusions. The inaccessible ACM final PDF was replaced by its author preprint.

Reproduction: [README](README.md). Results: [1B baseline](results/cuda-study-1b/summary.csv),
[8B baseline](results/cuda-study-8b/summary.csv),
[70B baseline](results/cuda-study-70b/summary.csv),
[1B reuse](results/prefix-study-1b/summary.csv),
[8B reuse](results/prefix-study-8b/summary.csv),
[70B reuse](results/prefix-study-70b/summary.csv),
[70B audit](results/70b-audit.json).
Source/metric details: [notes](references/notes.md).
Experiment decisions and resumption state: [README_LOG](README_LOG.md).
