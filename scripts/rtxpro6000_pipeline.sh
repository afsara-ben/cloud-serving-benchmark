#!/usr/bin/env bash
# Remote Llama 70B workflow. See RTX_PRO_6000.md for protocol and prerequisites.
set -Eeuo pipefail

csb_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
csb_action=all
csb_dry_run=0
csb_stage=arguments
csb_study="${STUDY_NAME:-cuda-context-study-rtxpro6000-r1}"
csb_gpus="${GPU_DEVICE:-0,1}"
csb_model_dir="${CSB_MODEL_DIR:-$HOME}"
csb_remote_host="${CSB_REMOTE_HOST:-hys4qm@l2x02}"
csb_remote_dir="${CSB_REMOTE_DIR:-/scratch/cloud-serving-benchmark}"
csb_venv="${CSB_VENV:-$csb_root/.venv-rtxpro6000}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'csb_status=$?; printf "ERROR: stage %s failed at line %s (exit %s). Later stages were not run.\n" "$csb_stage" "$LINENO" "$csb_status" >&2; exit "$csb_status"' ERR

usage() {
  cat <<'HELP'
Usage: ./scripts/rtxpro6000_pipeline.sh [ACTION] [OPTIONS]

Actions (default: all):
  all              Setup, serving pilots/sweep/poster, profiling pilots/reports.
  setup            Check Q2/Q4 files, install Python/build dependencies, build
                   SM120 llama.cpp, install gguf-py, prepare all four models.
  serving          Run 2K/32K pilots, the 60-cell sweep and the poster builder.
  nsight           Locate/install full Nsight Compute and probe both GPUs.
  profile          Nsight preflight, diagnostic build, 2K/32K captures and PDFs.
  report           Rebuild the serving report, poster and both bottleneck PDFs.
  transfer-models  Run ON THE ORIGINAL MACHINE: rsync exact Q2/Q4 files to remote.

Options:
  --dry-run        Print commands; no installs, transfers, writes or GPU jobs.
  --study-name ID  Results series (default: STUDY_NAME or cuda-context-study-rtxpro6000-r1).
  --gpus IDS       Two physical indices (default: GPU_DEVICE or 0,1).
  --model-dir DIR  Local directory holding Q2/Q4 GGUFs (default: CSB_MODEL_DIR or HOME).
  --remote-host H  Model-transfer destination (default: CSB_REMOTE_HOST or hys4qm@l2x02).
  --remote-dir DIR Remote checkout (default: CSB_REMOTE_DIR or /scratch/cloud-serving-benchmark).
  -h, --help       Show this help without doing work.

Environment: CSB_VENV overrides the Python environment directory; NCU_BIN selects
an existing full Nsight installation. CUDA_HOME selects CUDA; CSB_NVTX_INCLUDE
can point to a directory containing nvtx3/nvToolsExt.h.

Run on an idle allocation with two RTX PRO 6000 96GB GPUs and CUDA >=12.8.
No sudo or driver changes are performed. Missing Q2/Q4 files stop setup before
installation/build/downloads; use transfer-models or --model-dir to supply them.
Successful measurements resume with unchanged identity. Failed captures remain
preserved; see RTX_PRO_6000.md for retry commands using a fresh capture directory.
HELP
}

while (($#)); do
  case "$1" in
    all|setup|serving|nsight|profile|report|transfer-models) csb_action="$1"; shift ;;
    --dry-run) csb_dry_run=1; shift ;;
    --study-name|--gpus|--model-dir|--remote-host|--remote-dir)
      (($# >= 2)) && [[ -n "$2" ]] || die "Missing value for $1"
      case "$1" in
        --study-name) csb_study="$2" ;;
        --gpus) csb_gpus="$2" ;;
        --model-dir) csb_model_dir="$2" ;;
        --remote-host) csb_remote_host="$2" ;;
        --remote-dir) csb_remote_dir="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ "$csb_study" =~ ^[A-Za-z0-9_.-]+$ && "$csb_study" != . && "$csb_study" != .. ]] || die 'Invalid study name'
[[ "$csb_gpus" =~ ^[0-9]+,[0-9]+$ && "${csb_gpus%,*}" != "${csb_gpus#*,}" ]] || die 'Select two distinct GPU indices, e.g. --gpus 0,1'
# Resolve user-supplied relative paths before changing working directory.
[[ "$csb_model_dir" = /* ]] || csb_model_dir="$PWD/$csb_model_dir"
[[ "$csb_venv" = /* ]] || csb_venv="$PWD/$csb_venv"
cd -- "$csb_root"
csb_python="$csb_venv/bin/python3"
csb_study_root="$csb_root/results/$csb_study"
csb_models=(Llama-3.3-70B-Instruct-Q2_K.gguf Llama-3.3-70B-Instruct-Q4_K_M.gguf)

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if (( ! csb_dry_run )); then "$@"; fi
}

require_tools() {
  local tool
  (( csb_dry_run )) && return 0
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || die "Missing command: $tool. Load the site's CUDA/compiler module or install this prerequisite."
  done
}

activate_environment() {
  if (( csb_dry_run )); then
    printf '+ source %q\n' "$csb_venv/bin/activate"
  else
    [[ -f "$csb_venv/bin/activate" ]] || die 'Python environment missing; run this script with setup first.'
    # shellcheck source=/dev/null
    source "$csb_venv/bin/activate"
  fi
}

configure_cuda() {
  if [[ -n "${CUDA_HOME:-}" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
  require_tools nvcc
  if (( ! csb_dry_run )) && [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")"
    export CUDA_HOME
  fi
}

ensure_models() {
  local name missing=0
  for name in "${csb_models[@]}"; do
    if [[ ! -s "$csb_root/models/$name" && ! -s "$csb_model_dir/$name" ]] && (( ! csb_dry_run )); then
      printf 'Missing GGUF: %s\n' "$csb_root/models/$name" >&2
      missing=1
    fi
  done
  (( missing == 0 )) || die 'Run transfer-models on the original machine, or use --model-dir /path/to/existing/GGUFs. No setup jobs were started.'
  run mkdir -p "$csb_root/models"
  for name in "${csb_models[@]}"; do
    if [[ ! -s "$csb_root/models/$name" && -s "$csb_model_dir/$name" ]]; then
      [[ ! -e "$csb_root/models/$name" && ! -L "$csb_root/models/$name" ]] || die "Existing empty file or broken link: models/$name; repair it before setup."
      run ln -s "$csb_model_dir/$name" "$csb_root/models/$name"
    fi
    run test -s "$csb_root/models/$name"
  done
}

setup() {
  csb_stage=setup
  ensure_models
  configure_cuda
  require_tools python3 git c++ curl flock readlink
  if [[ ! -f "$csb_venv/bin/activate" ]]; then run python3 -m venv "$csb_venv"; fi
  activate_environment
  # Keep build tools explicit so this wrapper also repairs an older remote
  # requirements file that omitted Ninja/CMake.
  run "$csb_python" -m pip install ninja 'cmake>=3.28' -r scripts/requirements-rtxpro6000.txt
  run "$csb_python" scripts/run_rtxpro6000.py build --gpus "$csb_gpus"
  run "$csb_python" -m pip install -e vendor/llama.cpp/gguf-py
  run "$csb_python" scripts/run_rtxpro6000.py prepare --gpus "$csb_gpus"
}

nsight() {
  csb_stage=nsight
  configure_cuda
  local archive=nsight_compute-linux-x86_64-2026.3.0.13-archive
  local prefix="$HOME/.local/opt/nvidia" candidate
  if [[ -z "${NCU_BIN:-}" ]]; then
    NCU_BIN="$(command -v ncu || true)"
    if [[ -z "$NCU_BIN" ]]; then
      for candidate in "$prefix/$archive/ncu" "$prefix"/nsight-compute/*/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda*/nsight-compute*/ncu; do
        if [[ -f "$candidate" && -x "$candidate" ]]; then NCU_BIN="$candidate"; break; fi
      done
    fi
    if [[ -z "$NCU_BIN" ]]; then
      require_tools curl tar xz uname
      if (( ! csb_dry_run )); then
        [[ "$(uname -m)" = x86_64 ]] || die 'Automatic Nsight install supports x86_64 only; set NCU_BIN to a full installation for this CPU.'
      fi
      run mkdir -p "$prefix"
      run curl -fL --retry 3 -C - \
        "https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/$archive.tar.xz" \
        -o "$prefix/$archive.tar.xz"
      run tar -xJf "$prefix/$archive.tar.xz" -C "$prefix"
      NCU_BIN="$prefix/$archive/ncu"
    fi
  fi
  export NCU_BIN
  run "$NCU_BIN" --version
  run "$csb_python" scripts/profile_rtxpro6000.py preflight --gpus "$csb_gpus" --ncu "$NCU_BIN"
}

serving() {
  csb_stage=serving
  run "$csb_python" scripts/run_rtxpro6000.py plan --study-name "$csb_study" --gpus "$csb_gpus"
  local prompt
  for prompt in 2048 32768; do
    run "$csb_python" scripts/run_rtxpro6000.py run --study-name "$csb_study" --gpus "$csb_gpus" \
      --formats Q4_K_M --concurrencies 8 --prompt-lengths "$prompt"
  done
  run "$csb_python" scripts/run_rtxpro6000.py run --study-name "$csb_study" --gpus "$csb_gpus"
  run "$csb_python" scripts/run_rtxpro6000.py report --study-name "$csb_study" --gpus "$csb_gpus"
  run "$csb_python" paper/poster/build.py --study-root "$csb_study_root"
}

profile() {
  nsight
  csb_stage=profile
  local build=("$csb_python" benchmark/matrix_diagnostic.py build --server --cuda-arch 120 --build-root .run/rtxpro6000-profile-build)
  if [[ -n "${CSB_NVTX_INCLUDE:-}" ]]; then build+=(--nvtx-include "$CSB_NVTX_INCLUDE"); fi
  run "${build[@]}"
  local prompt action
  for prompt in 2048 32768; do
    for action in plan pilot run; do
      run "$csb_python" scripts/profile_rtxpro6000.py "$action" --study-root "$csb_study_root" \
        --prompt-tokens "$prompt" --ncu "$NCU_BIN"
    done
  done
}

report() {
  csb_stage=report
  run "$csb_python" scripts/run_rtxpro6000.py report --study-name "$csb_study" --gpus "$csb_gpus"
  run "$csb_python" paper/poster/build.py --study-root "$csb_study_root"
  local prompt
  for prompt in 2048 32768; do
    run "$csb_python" paper/runtime-bottlenecks-report/build.py --study-root "$csb_study_root" \
      --captures "$csb_study_root/profiles/c8-p$prompt"
  done
}

transfer_models() {
  csb_stage=transfer-models
  require_tools rsync ssh
  # These restrictions make the remote shell's mkdir argument unambiguous.
  [[ "$csb_remote_host" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]] || die 'Use a plain SSH host or user@host (configure jump hosts in SSH config).'
  [[ "$csb_remote_dir" =~ ^/[A-Za-z0-9_./-]+$ ]] || die 'Remote checkout must be an absolute path without spaces or shell syntax.'
  local name sources=()
  for name in "${csb_models[@]}"; do
    if (( ! csb_dry_run )); then [[ -s "$csb_model_dir/$name" ]] || die "Missing source GGUF: $csb_model_dir/$name"; fi
    sources+=("$csb_model_dir/$name")
  done
  run ssh -- "$csb_remote_host" "mkdir -p -- '$csb_remote_dir/models'"
  run rsync -avP -- "${sources[@]}" "$csb_remote_host:$csb_remote_dir/models/"
}

printf 'RTX PRO 6000 workflow: %s; study: %s; GPUs: %s; dry-run: %s\n' "$csb_action" "$csb_study" "$csb_gpus" "$csb_dry_run"
case "$csb_action" in
  all) setup; serving; profile ;;
  setup) setup ;;
  serving|nsight|profile|report) activate_environment; "$csb_action" ;;
  transfer-models) transfer_models ;;
esac
if (( csb_dry_run )); then
  printf 'Dry run complete. No commands were executed.\n'
else
  printf 'Completed: %s.\n' "$csb_action"
  if [[ "$csb_action" =~ ^(all|serving|profile|report)$ ]]; then
    printf 'Figure outputs: %s/paper/\n' "$csb_study_root"
  fi
fi
