#!/usr/bin/env bash
# Run ONE line at a time on the remote GPU server. No experiment runs by default.
# NVIDIA A100 (CUDA):
#   bash execute.sh setup cuda 1b
#   bash execute.sh doctor cuda
#   bash execute.sh baseline cuda 1b
#   bash execute.sh prefix cuda 1b
#   bash execute.sh trace cuda 1b
#   bash execute.sh counters cuda 1b prefill
#   bash execute.sh counters cuda 1b decode
#   bash execute.sh counters cuda 1b reuse
#   bash execute.sh setup cuda 8b
#   bash execute.sh baseline cuda 8b
#   bash execute.sh prefix cuda 8b
#   bash execute.sh trace cuda 8b
#   bash execute.sh counters cuda 8b prefill
#   bash execute.sh counters cuda 8b decode
#   bash execute.sh counters cuda 8b reuse
# AMD MI300X: use the SAME individual lines with rocm instead of cuda, e.g.:
#   bash execute.sh setup rocm 1b
#   bash execute.sh doctor rocm
#   bash execute.sh baseline rocm 1b
#   bash execute.sh prefix rocm 1b
#   bash execute.sh trace rocm 1b
#   bash execute.sh counters rocm 1b prefill
#   bash execute.sh counters rocm 1b decode
#   bash execute.sh counters rocm 1b reuse
#   bash execute.sh setup rocm 8b
#   bash execute.sh baseline rocm 8b
#   bash execute.sh prefix rocm 8b
#   bash execute.sh trace rocm 8b
#   bash execute.sh counters rocm 8b prefill
#   bash execute.sh counters rocm 8b decode
#   bash execute.sh counters rocm 8b reuse
# Existing local Llama 70B models, split across CUDA GPUs 0 and 1:
#   bash execute.sh setup cuda 70b
#   bash execute.sh doctor cuda 70b  (counter probe checks GPU 0 only)
#   bash execute.sh baseline cuda 70b
#   bash execute.sh prefix cuda 70b
#   bash execute.sh trace cuda 70b
#   bash execute.sh trace cuda 70b reuse  (cached c8 timeline; also accepts 1b/8b)
#   bash execute.sh counters cuda 70b prefill
#   bash execute.sh counters cuda 70b decode
#   bash execute.sh counters cuda 70b reuse
#   bash execute.sh plan cuda 70b  (no GPU, build or downloads)
# Context/concurrency study, 512 output tokens and FP16 KV:
#   bash execute.sh study-setup cuda 1b
#   bash execute.sh study-setup cuda 8b
#   bash execute.sh study-setup cuda 70b
#   bash execute.sh study-plan cuda 1b
#   bash execute.sh study cuda 1b
#   bash execute.sh study cuda 8b
#   bash execute.sh study cuda 70b
#   bash execute.sh study-profile cuda 1b
#   bash execute.sh study-profile cuda 8b
#   bash execute.sh study-profile cuda 70b
# Context results use results/cuda-context-study-{1b,8b,70b}, with shared links
# and reports in results/cuda-context-study. Set STUDY_NAME for an independent
# repeat series, e.g. STUDY_NAME=cuda-context-study-repeat2. RUN_TAG continues to
# name legacy experiments and does not affect these context-study commands.
# Additional arguments are passed to the shared runner. GPU_DEVICE defaults to 0,1.
# On larger hardware only, explicitly prepare the 70B FP16 reference:
#   GPU_DEVICE=0,1 bash execute.sh study-setup cuda 70b --formats FP16 --include-remote-only
# The original `setup cuda 70b` command verifies models/manifest-70b.json paths.
# The context study's `study-setup` also prepares its additional selected formats.
# Optional: GPU_DEVICE=2,3 RUN_TAG=two-gpus bash execute.sh baseline cuda 70b
# Optional: RUN_TAG=repeat2 GPU_DEVICE=0 bash execute.sh baseline cuda 1b
# Optional: bash execute.sh plan rocm 8b  (no GPU, build or downloads)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || fail "Missing $1. Install the toolkit/dependencies described in README.md."; }
usage() {
  awk '/^#/ {if (NR>1) {sub(/^# ?/, ""); print}; next} {exit}' "$ROOT/execute.sh"
  echo 'Usage: bash execute.sh {setup|doctor|baseline|prefix|trace|counters|plan} {cuda|rocm} [1b|8b|70b] [prefill|decode|reuse]'
  echo 'Context study: bash execute.sh {study-setup|study-plan|study|study-profile} cuda {1b|8b|70b} [runner arguments]'
  echo 'Trace: omit the phase (or use decode) for uncached c1/c8; use reuse for a cached c8 timeline on either backend.'
  echo 'Counters require a successful real counter probe. No sudo, driver changes or monitoring-service stops are automated.'
}
action="${1:-help}"
if [[ "$action" == help || "$action" == --help || "$action" == -h ]]; then usage; exit 0; fi
case "$action" in
  study-setup|study-plan|study|study-profile)
    study_backend="${2:-}"
    study_size="${3:-}"
    [[ "$study_backend" == cuda ]] || fail 'The context study currently requires CUDA.'
    [[ "$study_size" == 1b || "$study_size" == 8b || "$study_size" == 70b ]] || fail 'Choose study model size 1b, 8b or 70b.'
    shift 3
    export ACCELERATOR_BACKEND=cuda GPU_DEVICE="${GPU_DEVICE:-0,1}"
    export CONFIG_FILE="$ROOT/config/context-study.env"
    study_name="${STUDY_NAME:-cuda-context-study}"
    [[ "$study_name" =~ ^[A-Za-z0-9_.-]+$ && "$study_name" != . && "$study_name" != .. ]] || fail 'STUDY_NAME must be a simple directory name.'
    study_manifest="$ROOT/models/study-manifest-$study_size.json"
    study_shared="$ROOT/results/$study_name"
    study_output="$ROOT/results/$study_name-$study_size"
    need python3
    if [[ "$action" == study-setup ]]; then
      exec python3 scripts/09_prepare_study_models.py --models "$study_size" --devices "$GPU_DEVICE" "$@"
    fi
    if [[ "$action" == study-plan ]]; then
      exec python3 scripts/08_run_cuda_study.py --context-study --dry-run --backend cuda --config-file "$CONFIG_FILE" --manifest "$study_manifest" --output-dir "$study_output" --repetitions "${STUDY_REPETITIONS:-1}" "$@"
    fi
    mkdir -p "$ROOT/.run" "$ROOT/logs"
    need flock
    exec 9>"$ROOT/.run/remote-experiment.lock"
    flock -n 9 || fail 'Another experiment or profiling command is active in this checkout.'
    if [[ "$action" == study ]]; then
      mkdir -p "$study_shared"
      study_link="$study_shared/$study_size"
      study_target="../$study_name-$study_size"
      if [[ -L "$study_link" ]]; then
        [[ "$(readlink -- "$study_link")" == "$study_target" ]] || fail "Conflicting study link: $study_link (expected $study_target)."
      elif [[ -e "$study_link" ]]; then
        fail "Conflicting study path: $study_link; expected a relative link to $study_target."
      else
        ln -s -- "$study_target" "$study_link"
      fi
      exec python3 scripts/08_run_cuda_study.py --context-study --backend cuda --config-file "$CONFIG_FILE" --manifest "$study_manifest" --output-dir "$study_output" --repetitions "${STUDY_REPETITIONS:-1}" "$@"
    fi
    exec python3 benchmark/profile_study.py --study-root "$study_output" --stage all --execute --captures "${STUDY_CAPTURES:-1}" "$@"
    ;;
esac
backend="${2:-}"; size="${3:-1b}"; phase="${4:-decode}"
[[ "$backend" == cuda || "$backend" == rocm ]] || fail 'Choose backend cuda or rocm.'
[[ "$size" == 1b || "$size" == 8b || "$size" == 70b ]] || fail 'Choose model size 1b, 8b or 70b.'
[[ "$size" != 70b || "$backend" == cuda ]] || fail 'The local 70B experiment currently uses two CUDA GPUs.'
case "$action" in setup|doctor|baseline|prefix|trace|counters|plan) ;; *) usage; exit 2;; esac
[[ $# -le 4 ]] || fail 'Too many arguments.'
export ACCELERATOR_BACKEND="$backend" CONFIG_FILE="$ROOT/config/remote-study.env"
default_gpu=0
if [[ "$size" == 70b ]]; then
  export CONFIG_FILE="$ROOT/config/multigpu-study.env"
  default_gpu=0,1
fi
export GPU_DEVICE="${GPU_DEVICE:-$default_gpu}"
[[ "$GPU_DEVICE" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'GPU_DEVICE must contain physical GPU indices, such as 0 or 0,1.'
[[ "$backend" == cuda || "$GPU_DEVICE" != *,* ]] || fail 'ROCm experiments currently select one GPU.'
for toolkit in "${CUDA_HOME:-/usr/local/cuda}" "${ROCM_PATH:-/opt/rocm}"; do
  [[ ! -d "$toolkit/bin" ]] || export PATH="$toolkit/bin:$PATH"
done
need python3
python3 -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10 or newer is required"'
python3 - "$GPU_DEVICE" "$size" <<'PY'
import sys
devices = [int(value) for value in sys.argv[1].split(',')]
assert len(devices) == len(set(devices)), 'GPU_DEVICE indices must be unique'
assert sys.argv[2] != '70b' or len(devices) == 2, 'The 70B config requires exactly two CUDA GPUs, for example GPU_DEVICE=0,1'
PY
manifest="$ROOT/models/manifest.json"
[[ "$size" != 8b ]] || manifest="$ROOT/models/manifest-8b.json"
[[ "$size" != 70b ]] || manifest="$ROOT/models/manifest-70b.json"
tag="${RUN_TAG:-$(hostname -s)}"
[[ "$tag" =~ ^[A-Za-z0-9_.-]+$ && "$tag" != . && "$tag" != .. ]] || fail 'RUN_TAG must contain only letters, digits, dots, underscores or hyphens.'
result_root="$ROOT/results/remote-$backend/$tag"
if [[ "$action" == plan ]]; then
  python3 scripts/08_run_cuda_study.py --dry-run --backend "$backend" --config-file "$CONFIG_FILE" --manifest "$manifest" --output-dir "$result_root/baseline-$size"
  python3 scripts/08_run_cuda_study.py --dry-run --prefix-study --backend "$backend" --config-file "$CONFIG_FILE" --manifest "$manifest" --output-dir "$result_root/prefix-$size"
  exit 0
fi
mkdir -p "$ROOT/.run" "$ROOT/logs" "$result_root"
need flock
exec 9>"$ROOT/.run/remote-experiment.lock"
flock -n 9 || fail 'Another execute.sh command is active in this checkout. Run experiments sequentially.'
source "$ROOT/scripts/common.sh"
rocm_groups=('SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_MFMA' 'SQ_INSTS,SQ_BUSY_CU_CYCLES' 'FETCH_SIZE' 'WRITE_SIZE' 'TCC_HIT,TCC_MISS' 'MeanOccupancyPerActiveCU' 'TA_TOTAL_WAVEFRONTS_sum,TCP_TOTAL_ACCESSES_sum')

device_check() {
  if [[ "$backend" == cuda ]]; then
    need nvcc; need nvidia-smi
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
      local capabilities
      capabilities="$(nvidia-smi -i "$GPU_DEVICE" --query-gpu=compute_cap --format=csv,noheader)"
      CUDA_ARCHITECTURES="$(python3 - "$capabilities" "${GPU_DEVICE//[^,]/}" <<'PY'
import re,sys
values = sys.argv[1].splitlines()
assert len(values) == len(sys.argv[2]) + 1, 'Architecture query did not return every selected GPU'
assert all(re.fullmatch(r'\d+\.\d+', value.strip()) for value in values), 'Cannot detect CUDA architectures; set CUDA_ARCHITECTURES explicitly'
print(';'.join(dict.fromkeys(value.strip().replace('.', '') for value in values)))
PY
      )"
      export CUDA_ARCHITECTURES
    fi
    if [[ -n "$(nvidia-smi -i "$GPU_DEVICE" --query-compute-apps=pid --format=csv,noheader)" ]]; then
      fail 'A selected GPU already has a compute process. Use an idle allocation.'
    fi
  else
    need hipcc; need hipconfig; need rocminfo
    [[ -r /dev/kfd && -w /dev/kfd ]] || fail 'Need read/write /dev/kfd access. Ask the host administrator for render/video device access; reconnect after group changes.'
    local accessible=0 node
    for node in /dev/dri/renderD*; do
      [[ ! -r "$node" || ! -w "$node" ]] || accessible=1
    done
    [[ "$accessible" == 1 ]] || fail 'No accessible GPU render node. In a container pass /dev/kfd and the allocated /dev/dri render nodes.'
    rocminfo > "$result_root/rocminfo.txt"
    awk '/gfx942|Marketing Name:/' "$result_root/rocminfo.txt"
  fi
}

doctor() {
  device_check
  local probe_dir="$ROOT/.run/remote-preflight-$backend" profiler probe_gpu="${GPU_DEVICE%%,*}"
  if [[ "$GPU_DEVICE" == *,* ]]; then
    echo "Counter preflight probes physical GPU $probe_gpu only; model experiments select GPUs $GPU_DEVICE."
  fi
  mkdir -p "$probe_dir"
  # A single kernel with different launch sizes detects leaks across region gates.
  cat > "$probe_dir/probe.cpp" <<'CPP'
#ifdef CSB_HIP
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#define MALLOC hipMalloc
#define SYNC hipDeviceSynchronize
#define FREE hipFree
#define START() roctxProfilerResume(0)
#define STOP() roctxProfilerPause(0)
#else
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#define MALLOC cudaMalloc
#define SYNC cudaDeviceSynchronize
#define FREE cudaFree
#define START() cudaProfilerStart()
#define STOP() cudaProfilerStop()
#endif
#include <cstdio>
__global__ void csb_counter_probe(float *p) { p[blockIdx.x*blockDim.x+threadIdx.x] = float(threadIdx.x)+1.0f; }
int main() {
  float *p=nullptr;
  if (MALLOC((void **)&p,4096*sizeof(float))) return 2;
  csb_counter_probe<<<1,256>>>(p); if (SYNC()) return 3;
  START(); csb_counter_probe<<<16,256>>>(p); if (SYNC()) return 4; STOP();
  csb_counter_probe<<<1,256>>>(p); if (SYNC()) return 5;
  FREE(p); std::puts("Counter probe completed"); return 0;
}
CPP
  if [[ "$backend" == cuda ]]; then
    profiler="${NCU_BIN:-$(command -v ncu || true)}"
    [[ -x "$profiler" ]] || fail 'Nsight Compute ncu is missing. Install it or set NCU_BIN=/absolute/path/to/ncu.'
    nvcc -x cu -O2 "$probe_dir/probe.cpp" -o "$probe_dir/probe"
  else
    profiler="${ROCPROFV3_BIN:-$(command -v rocprofv3 || true)}"
    [[ -x "$profiler" ]] || fail 'rocprofv3 is missing. Install ROCprofiler-SDK (ROCm 7.14+ for gated counters).'
    local hip_root
    hip_root="$(hipconfig -R)"
    hipcc -DCSB_HIP -O2 "$probe_dir/probe.cpp" -I"$hip_root/include" -L"$hip_root/lib" -Wl,-rpath,"$hip_root/lib" -lrocprofiler-sdk-roctx -o "$probe_dir/probe"
  fi
  # A clean child environment avoids embedding tokens/SSH/editor settings in reports.
  python3 - "$backend" "$profiler" "$probe_dir" "$probe_gpu" "$result_root" "$GPU_DEVICE" <<'PY'
import csv, io, json, math, os, pathlib, subprocess, sys, time
backend, profiler, directory, gpu, result, requested_devices = sys.argv[1:]
p=pathlib.Path(directory); out=p/str(time.time_ns()); out.mkdir()
env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','USER','LOGNAME','LANG','LC_ALL','LD_LIBRARY_PATH'}}
env['CUDA_VISIBLE_DEVICES' if backend=='cuda' else 'HIP_VISIBLE_DEVICES']=gpu
if backend=='cuda':
 cmd=[profiler,'--profile-from-start','off','--clock-control','none','--cache-control','none','--csv','--page','raw','--print-units','base','--metrics','gpu__time_duration.sum,sm__inst_executed.sum','--log-file',str(out/'counters.csv'),'--export',str(out/'probe'),str(p/'probe')]
else:
 cmd=[profiler,'--selected-regions','--marker-trace','--kernel-trace','--output-format','csv','--output-directory',str(out),'--output-file','probe','--pmc','SQ_WAVES','--',str(p/'probe')]
with (out/'profiler.log').open('w') as log:
 run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
report=None; export_cmd=None; export_returncode=None
if backend=='cuda':
 sys.path.insert(0,str(pathlib.Path.cwd()/'benchmark'))
 from profile_cuda import find_ncu_report, summarize_ncu
 report=find_ncu_report(out/'probe')
 if report is not None:
  export_cmd=[profiler,'--import',str(report),'--page','raw','--csv','--print-units','base']
  with (out/'ncu_raw.csv').open('w') as raw, (out/'export.log').open('w') as log:
   export_returncode=subprocess.run(export_cmd,env=env,stdout=raw,stderr=log,timeout=120).returncode
text='\n'.join(f.read_text(errors='replace') for f in out.rglob('*') if f.suffix in {'.csv','.log'})
status='profiler_failed'
if 'ERR_NVGPUCTRPERM' in text: status='permission_denied'
elif 'driver resource was unavailable' in text: status='resource_busy'
rows=[]
for f in out.rglob('*.csv'):
 lines=f.read_text(errors='replace').splitlines()
 for i,line in enumerate(lines):
  if 'Metric Name' in line or 'Counter_Name' in line:
   rows.extend(csv.DictReader(io.StringIO('\n'.join(lines[i:])))); break
def num(value):
 try: return float(str(value).replace(',',''))
 except (ValueError,TypeError): return float('nan')
parse_error=None
if backend=='cuda':
 passed=False
 if export_returncode==0:
  try:
   summary=summarize_ncu(out/'ncu_raw.csv',out)
   metrics=summary['launches'][0]['metrics']
   values=[metrics.get(name,{}) for name in ('sm__inst_executed.sum','gpu__time_duration.sum')]
   passed=summary['sample_count']==1 and all(m.get('available') and m['value']>0 for m in values)
  except ValueError as error:
   parse_error=str(error)
else:
 values=[r for r in rows if r.get('Counter_Name')=='SQ_WAVES']
 # MI300X uses 64-lane waves: inside=16*256/64=64; outside=4.
 passed=len(values)==1 and num(values[0].get('Counter_Value'))==64
passed=passed and run.returncode==0
if passed: status='passed'
record={'status':status,'backend':backend,'device':gpu,'requested_devices':requested_devices.split(','),'probe_scope':'first selected physical GPU only','command':cmd,'returncode':run.returncode,'evidence':str(out),'region_validation':'One included launch; outside launches excluded' if passed else 'failed','created_unix':time.time()}
if backend=='cuda': record.update(counter_report=str(report) if report else None,counter_export_command=export_cmd,counter_export_returncode=export_returncode,counter_parse_error=parse_error)
(pathlib.Path(result)/'counter-access.json').write_text(json.dumps(record,indent=2)+'\n')
if not passed:
 print(f'Counter preflight FAILED: {status}. Evidence: {out}',file=sys.stderr)
 if status=='permission_denied': print('Host admin: enable NVIDIA non-admin counters (NVreg_RestrictProfilingToAdminUsers=0), or grant approved profiler privileges.',file=sys.stderr)
 elif status=='resource_busy': print('A monitor such as DCGM owns the counters. Ask its owner to pause profiling collection, then resume it after these experiments. No monitoring service was changed.',file=sys.stderr)
 elif backend=='rocm': print('Check AMD device permissions, supported SQ_WAVES counter, and ROCm 7.14+ selected-region counter support. Only 64 waves from the inside region should be captured.',file=sys.stderr)
 else: print(text[-1800:],file=sys.stderr)
 sys.exit(2)
print('Counter access and selected-region probe: PASS')
PY
}

model_rows() {
  python3 - "$manifest" "$ROOT" <<'PY'
import json,pathlib,re,sys
models=json.load(open(sys.argv[1]))
assert isinstance(models,list) and models, 'Model manifest must be a nonempty list'
formats=set()
for m in models:
 quant=m.get('quant','')
 assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',quant) and quant not in formats, 'Invalid or duplicate quant format'
 formats.add(quant)
 assert isinstance(m.get('bytes'),int) and m['bytes']>4 and re.fullmatch(r'[0-9a-fA-F]{64}',m.get('sha256','')), 'Use a pinned manifest with bytes/SHA256 fields'
 assert m.get('filename') or m.get('path'), 'Each model requires filename or absolute path'
 filename=m.get('filename') or pathlib.Path(m['path']).name
 assert pathlib.Path(filename).name==filename, 'filename must be a basename; use path for absolute files'
 path=pathlib.Path(m['path']).expanduser() if m.get('path') else pathlib.Path(sys.argv[2])/'models'/filename
 assert path.is_absolute(), 'Manifest model path must be absolute'
 fields=[quant,filename,str(path),m.get('url',''),str(m['bytes']),m['sha256']]
 assert all(all(char not in str(v) for char in ['\x1f','\n','\r']) for v in fields), 'Invalid manifest field'
 # A non-whitespace delimiter retains an empty optional URL in Bash read.
 print('\x1f'.join(fields))
PY
}

if [[ "$action" != doctor ]]; then
  model_table="$(model_rows)"
  [[ -n "$model_table" ]] || fail 'Model manifest is empty.'
fi

if [[ "$action" == setup ]]; then
  for tool in git cmake ninja g++ python3; do need "$tool"; done
  [[ "$size" == 70b ]] || need curl
  python3 - "$(cmake --version)" <<'PY'
import re,sys
v=re.search(r'(\d+)\.(\d+)',sys.argv[1])
assert v and tuple(map(int,v.groups())) >= (3,21), 'Install CMake 3.21 or newer'
PY
  echo 'Checking local model prerequisites; details append to logs/download.log'
  python3 - "$manifest" "$ROOT" "$size" >> logs/download.log 2>&1 <<'PY' || fail 'Local model verification failed; inspect logs/download.log. No download was attempted.'
import hashlib,json,pathlib,sys
for model in json.load(open(sys.argv[1])):
 if sys.argv[3] != '70b' and model.get('url'): continue
 path=pathlib.Path(model['path']).expanduser() if model.get('path') else pathlib.Path(sys.argv[2])/'models'/model['filename']
 if not path.is_file():
  raise SystemExit(f'Missing local model: {path}. Supply the original file or update the manifest with a verified local path; download provenance is unknown.')
 if path.stat().st_size != model['bytes']:
  raise SystemExit(f'Size mismatch: {path}')
 with path.open('rb') as stream:
  if stream.read(4) != b'GGUF': raise SystemExit(f'Invalid GGUF header: {path}')
  stream.seek(0); digest=hashlib.sha256()
  for block in iter(lambda: stream.read(8*1024*1024),b''): digest.update(block)
 if digest.hexdigest() != model['sha256'].lower(): raise SystemExit(f'SHA256 mismatch: {path}')
 print(f'Verified existing local model: {path} ({model["bytes"]} bytes)',flush=True)
PY
  device_check
  echo 'Building the pinned server; details append to logs/build.log'
  bash scripts/01_build_llama_cpp.sh >> logs/build.log 2>&1 || fail 'Build failed; inspect logs/build.log.'
  if [[ "$size" != 70b ]]; then
    while IFS=$'\x1f' read -r quant filename path url bytes sha; do
      echo "Preparing $size $quant"
      [[ -n "$url" ]] || continue  # Files without URLs were verified above.
      MODEL_FILENAME="$filename" MODEL_PATH="$path" MODEL_URL="$url" MODEL_SIZE_BYTES="$bytes" MODEL_SHA256="$sha" bash scripts/02_download_model.sh >> logs/download.log 2>&1 || fail 'Download verification failed; inspect logs/download.log.'
    done <<< "$model_table"
  fi
  echo "Setup complete. Next: bash execute.sh doctor $backend $size"
  exit 0
fi
if [[ "$action" == doctor ]]; then doctor; exit 0; fi
if [[ "$action" == baseline || "$action" == prefix ]]; then
  device_check
  flags=(); [[ "$action" != prefix ]] || flags+=(--prefix-study)
  out="$result_root/$action-$size"
  python3 scripts/08_run_cuda_study.py --backend "$backend" --config-file "$CONFIG_FILE" --manifest "$manifest" --output-dir "$out" --repetitions "${STUDY_REPETITIONS:-3}" --requests "${STUDY_REQUESTS:-16}" "${flags[@]}" 2>&1 | tee -a "logs/study-$backend-$size.log"
  python3 benchmark/summarize.py "$out" --output-csv "$out/summary.csv" --output-markdown "$out/summary.md"
  echo "Results: $out/summary.md"
  exit 0
fi

[[ "$action" != counters || "$phase" == prefill || "$phase" == decode || "$phase" == reuse ]] || fail 'Counter phase must be prefill, decode or reuse.'
[[ "$action" != trace || "$phase" == decode || "$phase" == reuse ]] || fail 'Trace phase must be decode (uncached c1/c8, the default) or reuse (cached c8).'
if [[ "$action" == counters ]]; then doctor; else device_check; fi
if [[ "$action" == counters && "$backend" == rocm ]]; then
  need rocprofv3-avail
  catalog="$result_root/rocm-counter-catalog"
  mkdir -p "$catalog"
  rocprofv3-avail list --pmc -d "$GPU_DEVICE" > "$catalog/list.txt" 2>&1 || fail "Cannot enumerate AMD counters; inspect $catalog/list.txt."
  rocprofv3-avail info --pmc -d "$GPU_DEVICE" > "$catalog/definitions.txt" 2>&1 || fail "Cannot read AMD counter definitions; inspect $catalog/definitions.txt."
  : > "$catalog/group-checks.log"
  for group in "${rocm_groups[@]}"; do
    IFS=, read -r -a metrics <<< "$group"
    printf 'Checking %s\n' "$group" >> "$catalog/group-checks.log"
    rocprofv3-avail pmc-check -d "$GPU_DEVICE" "${metrics[@]}" >> "$catalog/group-checks.log" 2>&1 || fail "Unsupported/incompatible AMD counter group: $group. Inspect $catalog/group-checks.log before loading a model."
  done
fi
while IFS=$'\x1f' read -r quant filename path url bytes sha; do
  export MODEL_FILENAME="$filename" MODEL_PATH="$path" MODEL_ALIAS="llama-$size-$quant"
  require_file "$MODEL_PATH"; require_file "$LLAMA_SERVER_BIN"
  columns=(1 8); [[ "$action" != counters || "$phase" == decode ]] || columns=(8)
  [[ "$action" != trace || "$phase" != reuse ]] || columns=(8)
  for c in "${columns[@]}"; do
    export PROFILE_CONCURRENCY="$c" PROFILE_REPEAT_PROMPT=1 PROFILE_PREFIX_REUSE=0
    if [[ "$action" == trace ]]; then
      export PROFILE_TOOL=nsys; [[ "$backend" != rocm ]] || export PROFILE_TOOL=rocprof-trace
      export PROFILE_OUTPUT="$result_root/traces-$size/$quant-c$c"
      if [[ "$phase" == reuse ]]; then
        export PROFILE_PREFIX_REUSE=1
        export PROFILE_OUTPUT="$result_root/traces-$size/$quant-reuse-on"
      fi
      bash scripts/07_profile_cuda.sh 2>&1 | tee -a "logs/study-$backend-$size.log"
      continue
    fi
    export PROFILE_TOOL=ncu; [[ "$backend" != rocm ]] || export PROFILE_TOOL=rocprof-counters
    [[ "$phase" != reuse ]] || export PROFILE_PREFIX_REUSE=1
    # Pinned ggml/include/ggml.h enum IDs; Q2_K supports MMQ and MMVQ.
    case "$quant" in
      Q8_0) type=8;; Q2_K) type=10;; Q4_K_M) type=12;; IQ1_M) type=29;;
      *) fail "No validated counter kernel selector for $quant. Inspect its trace before adding a selector.";;
    esac
    # Match the enum value immediately after its type, not later template integers.
    enum_regex="\\(?ggml_type\\)? *$type"
    filters=("mul_mat_vec_q<$enum_regex, *(\\(int\\))?$c,")
    labels=(matvec)
    if [[ "$phase" == prefill ]]; then
      filters=("mul_mat_q<$enum_regex,"); labels=(matrix)
      if [[ "$quant" == IQ1_M ]]; then
        filters=('dequantize_block_iq1_m' 'gemm|Gemm'); labels=(dequant gemm)
        [[ "$backend" != rocm ]] || filters[1]='gemm|Gemm|Cijk_'
      fi
    fi
    # Mixed recipes contain substantial secondary tensor types in the saved
    # CUDA traces. Separate captures prevent per-launch-config limits from
    # hiding them; these remain selected samples, not whole-model counters.
    extra_types=(); extra_labels=()
    case "$quant" in
      Q2_K) extra_types=(11); extra_labels=(q3k);;
      Q4_K_M) extra_types=(14); extra_labels=(q6k);;
      IQ1_M)
        if [[ "$phase" != prefill ]]; then
          extra_types=(13 16); extra_labels=(q5k iq2xxs)
        fi;;
    esac
    for e in "${!extra_types[@]}"; do
      extra_enum="\\(?ggml_type\\)? *${extra_types[$e]}"
      if [[ "$phase" == prefill ]]; then
        filters+=("mul_mat_q<$extra_enum,")
        labels+=("matrix-${extra_labels[$e]}")
      else
        filters+=("mul_mat_vec_q<$extra_enum, *(\\(int\\))?$c,")
        labels+=("matvec-${extra_labels[$e]}")
      fi
    done
    for i in "${!filters[@]}"; do
      export PROFILE_KERNEL_REGEX="${filters[$i]}" PROFILE_LAUNCH_COUNT=2 PROFILE_FILTER_MODE=per-launch-config
      export PROFILE_KERNEL_ITERATION_RANGE='[1-8]'
      groups=(default); group_labels=(counters)
      if [[ "$backend" == rocm ]]; then
        groups=("${rocm_groups[@]}")
        group_labels=(instructions ipc memory-read memory-write l2 occupancy coalescing)
      fi
      for g in "${!groups[@]}"; do
        [[ "$backend" != rocm ]] || export PROFILE_METRICS="${groups[$g]}"
        export PROFILE_OUTPUT="$result_root/counters-$size/$phase/$quant-c$c-${labels[$i]}-${group_labels[$g]}"
        bash scripts/07_profile_cuda.sh 2>&1 | tee -a "logs/study-$backend-$size.log"
      done
    done
  done
done <<< "$model_table"
echo "Completed $action $backend $size. Results: $result_root"
