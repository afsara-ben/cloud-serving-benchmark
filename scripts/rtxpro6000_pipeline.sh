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
csb_model_dir_explicit=0
if [[ -n "${CSB_MODEL_DIR:-}" ]]; then csb_model_dir_explicit=1; fi
csb_remote_host="${CSB_REMOTE_HOST:-hys4qm@l2x02}"
csb_remote_dir="${CSB_REMOTE_DIR:-/scratch/cloud-serving-benchmark}"
csb_source_host="${CSB_SOURCE_HOST:-}"
csb_source_dir="${CSB_SOURCE_DIR:-/data/home/hys4qm}"
csb_venv="${CSB_VENV:-$csb_root/.venv-rtxpro6000}"
csb_fast=0
csb_minimal_profile=0
csb_serving_formats="${CSB_SERVING_FORMATS:-IQ1_M,Q2_K,Q4_K_M,Q8_0}"
csb_profile_formats="${CSB_PROFILE_FORMATS:-Q4_K_M,Q2_K}"
csb_profile_contexts="${CSB_PROFILE_CONTEXTS:-2048,32768}"
csb_operator_launch_count="${CSB_OPERATOR_LAUNCH_COUNT:-5}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'csb_status=$?; printf "ERROR: stage %s failed at line %s (exit %s). Later stages were not run.\n" "$csb_stage" "$LINENO" "$csb_status" >&2; exit "$csb_status"' ERR

usage() {
  cat <<'HELP'
Usage: ./scripts/rtxpro6000_pipeline.sh [ACTION] [OPTIONS]

Actions (default: all):
  all              Setup, selected serving sweep, reduced/full profiles and exports.
  setup            Check selected local files, install/build, and prepare only
                   the selected model formats.
  serving          Run 2K/32K pilots and the selected r1 serving sweep.
  nsight           Locate/install full Nsight Compute and probe both GPUs.
  profile          Nsight preflight, diagnostic build, selected captures/PDF/export.
  report           Rebuild the saved-data reports and available complete poster.
  export           Validate and write one Git-friendly JSON per format/profile type.
  transfer-models  Run ON THE ORIGINAL MACHINE: rsync selected GGUFs to remote.
  fetch-models     Run ON THE GPU SERVER: pull selected GGUFs over SSH.

Options:
  --dry-run        Print commands; no installs, transfers, writes or GPU jobs.
  --fast           Complete selected serving cells including 32K; keep r=1;
                   profile only 2K with 8 gate/up captures and layer extrapolation.
                   Skips the 60 targeted operator captures.
  --profile-contexts CSV  Bottleneck contexts (default: 2048,32768).
  --formats CSV    Serving formats. Also selects the two profile formats for a
                   two-format shard (default serving: all four).
  --profile-formats CSV  Exactly two formats to profile (default: Q4_K_M,Q2_K).
  --operator-launch-count N  Scalar launches/capture, 1-5 (default: 5).
  --study-name ID  Results series (default: STUDY_NAME or cuda-context-study-rtxpro6000-r1).
  --gpus IDS       Two physical indices (default: GPU_DEVICE or 0,1).
  --model-dir DIR  Local GGUF directory, including NVMe; fetch-models stores files
                   here (fetch default: checkout/models; otherwise CSB_MODEL_DIR or HOME).
  --remote-host H  Model-transfer destination (default: CSB_REMOTE_HOST or hys4qm@l2x02).
  --remote-dir DIR Remote checkout (default: CSB_REMOTE_DIR or /scratch/cloud-serving-benchmark).
  --source-host H  fetch-models source SSH host, e.g. hys4qm@xsel02 (or CSB_SOURCE_HOST).
  --source-dir DIR fetch-models source directory (default: CSB_SOURCE_DIR or /data/home/hys4qm).
  -h, --help       Show this help without doing work.

Environment: CSB_VENV overrides the Python environment directory; NCU_BIN selects
an existing full Nsight installation. CUDA_HOME selects CUDA; CSB_NVTX_INCLUDE
can point to a directory containing nvtx3/nvToolsExt.h.

Run on an idle allocation with two RTX PRO 6000 96GB GPUs and CUDA >=12.8.
No sudo or driver changes are performed. A missing selected Q2/Q4 file stops
setup before installation/build/downloads; use transfer-models or --model-dir.
Successful measurements resume with unchanged identity. Failed captures remain
preserved; see RTX_PRO_6000.md for retry commands using a fresh capture directory.
HELP
}

while (($#)); do
  case "$1" in
    all|setup|serving|nsight|profile|report|export|transfer-models|fetch-models) csb_action="$1"; shift ;;
    --dry-run) csb_dry_run=1; shift ;;
    --fast) csb_fast=1; shift ;;
    --study-name|--gpus|--model-dir|--remote-host|--remote-dir|--source-host|--source-dir|--profile-contexts|--operator-launch-count|--formats|--profile-formats)
      (($# >= 2)) && [[ -n "$2" ]] || die "Missing value for $1"
      case "$1" in
        --study-name) csb_study="$2" ;;
        --gpus) csb_gpus="$2" ;;
        --model-dir) csb_model_dir="$2"; csb_model_dir_explicit=1 ;;
        --remote-host) csb_remote_host="$2" ;;
        --remote-dir) csb_remote_dir="$2" ;;
        --source-host) csb_source_host="$2" ;;
        --source-dir) csb_source_dir="$2" ;;
        --profile-contexts) csb_profile_contexts="$2" ;;
        --operator-launch-count) csb_operator_launch_count="$2" ;;
        --formats) csb_serving_formats="$2"; csb_profile_formats="$2" ;;
        --profile-formats) csb_profile_formats="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

if (( csb_fast )); then
  csb_profile_contexts=2048
  csb_minimal_profile=1
fi

[[ "$csb_study" =~ ^[A-Za-z0-9_.-]+$ && "$csb_study" != . && "$csb_study" != .. ]] || die 'Invalid study name'
[[ "$csb_gpus" =~ ^[0-9]+,[0-9]+$ && "${csb_gpus%,*}" != "${csb_gpus#*,}" ]] || die 'Select two distinct GPU indices, e.g. --gpus 0,1'
[[ "$csb_operator_launch_count" =~ ^[1-5]$ ]] || die '--operator-launch-count must be 1 through 5'
[[ "$csb_serving_formats" =~ ^(IQ1_M|Q2_K|Q4_K_M|Q8_0)(,(IQ1_M|Q2_K|Q4_K_M|Q8_0))*$ ]] || die 'Invalid --formats list'
IFS=, read -r -a csb_format_check <<< "$csb_serving_formats"
[[ "$(printf '%s\n' "${csb_format_check[@]}" | sort -u | wc -l)" -eq "${#csb_format_check[@]}" ]] || die '--formats cannot contain duplicates'
[[ "$csb_profile_formats" =~ ^(IQ1_M|Q2_K|Q4_K_M|Q8_0),(IQ1_M|Q2_K|Q4_K_M|Q8_0)$ ]] || die '--profile-formats requires exactly two formats'
[[ "${csb_profile_formats%,*}" != "${csb_profile_formats#*,}" ]] || die '--profile-formats cannot repeat a format'
while IFS= read -r csb_format; do
  [[ ",$csb_serving_formats," == *",$csb_format,"* ]] || die "Profile format $csb_format is absent from --formats"
done < <(printf '%s\n' "${csb_profile_formats//,/$'\n'}")
[[ "$csb_profile_contexts" =~ ^(2048|4096|8192|16384|32768)(,(2048|4096|8192|16384|32768))*$ ]] || die 'Invalid --profile-contexts list'
IFS=, read -r -a csb_profile_prompts <<< "$csb_profile_contexts"
[[ "$(printf '%s\n' "${csb_profile_prompts[@]}" | sort -u | wc -l)" -eq "${#csb_profile_prompts[@]}" ]] || die '--profile-contexts cannot contain duplicates'
# Resolve user-supplied relative paths before changing working directory.
if [[ "$csb_action" = fetch-models ]] && (( ! csb_model_dir_explicit )); then
  csb_model_dir="$csb_root/models"
fi
[[ "$csb_model_dir" = /* ]] || csb_model_dir="$PWD/$csb_model_dir"
[[ "$csb_venv" = /* ]] || csb_venv="$PWD/$csb_venv"
cd -- "$csb_root"
csb_python="$csb_venv/bin/python3"
csb_study_root="$csb_root/results/$csb_study"
declare -A csb_model_names=(
  [IQ1_M]=Llama-3.3-70B-Instruct.i1-IQ1_M.gguf
  [Q2_K]=Llama-3.3-70B-Instruct-Q2_K.gguf
  [Q4_K_M]=Llama-3.3-70B-Instruct-Q4_K_M.gguf
  [Q8_0]=Llama-3.3-70B-Instruct.Q8_0.gguf
)
IFS=, read -r -a csb_selected_formats <<< "$csb_serving_formats"
csb_models=()
for csb_format in "${csb_selected_formats[@]}"; do csb_models+=("${csb_model_names[$csb_format]}"); done
csb_profile_tag="${csb_profile_formats//,/-}"
csb_profile_tag="${csb_profile_tag,,}"

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
  local name quant missing=0
  local -a selected
  IFS=, read -r -a selected <<< "$csb_serving_formats"
  for quant in "${selected[@]}"; do
    name="${csb_model_names[$quant]}"
    [[ "$quant" == Q2_K || "$quant" == Q4_K_M ]] || continue
    if [[ ! -s "$csb_root/models/$name" && ! -s "$csb_model_dir/$name" ]] && (( ! csb_dry_run )); then
      printf 'Missing GGUF: %s\n' "$csb_root/models/$name" >&2
      missing=1
    fi
  done
  (( missing == 0 )) || die 'On this GPU server, run fetch-models --source-host USER@SOURCE_HOST; alternatively run transfer-models on the source machine or use --model-dir for existing local GGUFs. No setup jobs were started.'
  run mkdir -p "$csb_root/models"
  for quant in "${selected[@]}"; do
    name="${csb_model_names[$quant]}"
    if [[ ! -s "$csb_root/models/$name" && "$csb_model_dir" != "$csb_root/models" ]] &&
       { [[ -s "$csb_model_dir/$name" ]] || (( csb_dry_run )); }; then
      [[ ! -e "$csb_root/models/$name" && ! -L "$csb_root/models/$name" ]] || die "Existing empty file or broken link: models/$name; repair it before setup."
      run ln -s "$csb_model_dir/$name" "$csb_root/models/$name"
    fi
    if [[ "$quant" == Q2_K || "$quant" == Q4_K_M || -s "$csb_root/models/$name" || -s "$csb_model_dir/$name" ]]; then
      run test -s "$csb_root/models/$name"
    fi
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
  run "$csb_python" scripts/run_rtxpro6000.py prepare --gpus "$csb_gpus" --formats "$csb_serving_formats"
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
  run "$csb_python" scripts/run_rtxpro6000.py plan --study-name "$csb_study" --gpus "$csb_gpus" --formats "$csb_serving_formats"
  local pilot_format="${csb_selected_formats[0]}"
  if [[ ",$csb_serving_formats," == *,Q4_K_M,* ]]; then pilot_format=Q4_K_M; fi
  local prompt
  for prompt in 2048 32768; do
    run "$csb_python" scripts/run_rtxpro6000.py run --study-name "$csb_study" --gpus "$csb_gpus" \
      --formats "$pilot_format" --concurrencies 8 --prompt-lengths "$prompt"
  done
  run "$csb_python" scripts/run_rtxpro6000.py run --study-name "$csb_study" --gpus "$csb_gpus" --formats "$csb_serving_formats"
  run "$csb_python" scripts/run_rtxpro6000.py report --study-name "$csb_study" --gpus "$csb_gpus" --formats "$csb_serving_formats"
  if [[ "$csb_serving_formats" == IQ1_M,Q2_K,Q4_K_M,Q8_0 ]]; then
    run "$csb_python" paper/poster/build.py --study-root "$csb_study_root"
  fi
}

profile() {
  nsight
  csb_stage=profile
  local build=("$csb_python" benchmark/matrix_diagnostic.py build --server --cuda-arch 120 --build-root .run/rtxpro6000-profile-build)
  if [[ -n "${CSB_NVTX_INCLUDE:-}" ]]; then build+=(--nvtx-include "$CSB_NVTX_INCLUDE"); fi
  run "${build[@]}"
  local prompt action capture_root
  local profile_options=()
  if (( csb_minimal_profile )); then profile_options=(--minimal-images --stage full); fi
  for prompt in "${csb_profile_prompts[@]}"; do
    capture_root="$csb_study_root/profiles/$csb_profile_tag-c8-p$prompt"
    if (( csb_minimal_profile )); then
      capture_root+="-minimal"
    elif [[ "$csb_operator_launch_count" != 5 ]]; then
      capture_root+="-op$csb_operator_launch_count"
    fi
    for action in plan pilot run; do
      run "$csb_python" scripts/profile_rtxpro6000.py "$action" --study-root "$csb_study_root" \
        --prompt-tokens "$prompt" --operator-launch-count "$csb_operator_launch_count" \
        --formats "$csb_profile_formats" --output "$capture_root" --ncu "$NCU_BIN" "${profile_options[@]}"
    done
  done
  export_results
}

export_results() {
  csb_stage=export
  local prompt capture_root
  for prompt in "${csb_profile_prompts[@]}"; do
    capture_root="$csb_study_root/profiles/$csb_profile_tag-c8-p$prompt"
    if (( csb_minimal_profile )); then
      capture_root+="-minimal"
    elif [[ "$csb_operator_launch_count" != 5 ]]; then
      capture_root+="-op$csb_operator_launch_count"
    fi
    run "$csb_python" scripts/export_rtxpro6000_results.py --study-root "$csb_study_root" \
      --captures "$capture_root"
  done
}

report() {
  csb_stage=report
  run "$csb_python" scripts/run_rtxpro6000.py report --study-name "$csb_study" --gpus "$csb_gpus" --formats "$csb_serving_formats"
  if [[ "$csb_serving_formats" == IQ1_M,Q2_K,Q4_K_M,Q8_0 ]]; then
    run "$csb_python" paper/poster/build.py --study-root "$csb_study_root"
  fi
  local prompt capture_root
  for prompt in "${csb_profile_prompts[@]}"; do
    capture_root="$csb_study_root/profiles/$csb_profile_tag-c8-p$prompt"
    if (( csb_minimal_profile )); then
      capture_root+="-minimal"
    elif [[ "$csb_operator_launch_count" != 5 ]]; then
      capture_root+="-op$csb_operator_launch_count"
    fi
    run "$csb_python" paper/runtime-bottlenecks-report/build.py --study-root "$csb_study_root" \
      --captures "$capture_root"
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
    if (( ! csb_dry_run )); then
      [[ -s "$csb_model_dir/$name" ]] || die "Missing source GGUF on this machine: $csb_model_dir/$name. transfer-models pushes LOCAL files. On the GPU server, use fetch-models --source-host USER@SOURCE_HOST --source-dir /path/on/source."
    fi
    sources+=("$csb_model_dir/$name")
  done
  run ssh -- "$csb_remote_host" "mkdir -p -- '$csb_remote_dir/models'"
  run rsync -avP -- "${sources[@]}" "$csb_remote_host:$csb_remote_dir/models/"
}

fetch_models() {
  csb_stage=fetch-models
  require_tools rsync ssh
  [[ -n "$csb_source_host" ]] || die 'fetch-models requires --source-host USER@SOURCE_HOST (the machine containing the original GGUFs).'
  [[ "$csb_source_host" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]] || die 'Use a plain source SSH host or user@host (configure jump hosts in SSH config).'
  [[ "$csb_source_dir" =~ ^/[A-Za-z0-9_./-]+$ ]] || die 'Source directory must be an absolute path without spaces or shell syntax.'
  # Check both files on their actual host before transferring either file.
  run ssh -- "$csb_source_host" "test -s '$csb_source_dir/${csb_models[0]}' && test -s '$csb_source_dir/${csb_models[1]}'"
  run mkdir -p "$csb_model_dir"
  local name sources=()
  for name in "${csb_models[@]}"; do sources+=("$csb_source_host:$csb_source_dir/$name"); done
  run rsync -avP -- "${sources[@]}" "$csb_model_dir/"
  # Keep the large files in the selected NVMe directory; checkout-local names
  # are symlinks when storage is outside the checkout.
  ensure_models
  printf 'GGUF storage directory: %s\n' "$csb_model_dir"
}

printf 'RTX PRO 6000 workflow: %s; study: %s; GPUs: %s; formats: %s; profile formats: %s; serving repetitions: 1; profile contexts: %s; minimal profile: %s; operator launches: %s; dry-run: %s\n' \
  "$csb_action" "$csb_study" "$csb_gpus" "$csb_serving_formats" "$csb_profile_formats" "$csb_profile_contexts" "$csb_minimal_profile" "$csb_operator_launch_count" "$csb_dry_run"
case "$csb_action" in
  all) setup; serving; profile ;;
  setup) setup ;;
  serving|nsight|profile|report) activate_environment; "$csb_action" ;;
  export) activate_environment; export_results ;;
  transfer-models) transfer_models ;;
  fetch-models) fetch_models ;;
esac
if (( csb_dry_run )); then
  printf 'Dry run complete. No commands were executed.\n'
else
  printf 'Completed: %s.\n' "$csb_action"
  if [[ "$csb_action" =~ ^(all|serving|profile|report)$ ]]; then
    printf 'Figure outputs: %s/paper/\n' "$csb_study_root"
  fi
fi
