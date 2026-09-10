#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config/experiment.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$CONFIG_FILE"
set +a

if [[ -d /opt/rocm/bin ]]; then
  export PATH="/opt/rocm/bin:$PATH"
fi
if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

resolve_accelerator_backend() {
  local requested="${ACCELERATOR_BACKEND:-auto}"
  requested="$(printf '%s' "$requested" | tr '[:upper:]' '[:lower:]')"
  case "$requested" in
    rocm|cuda)
      printf '%s\n' "$requested"
      ;;
    auto)
      local has_rocm=0
      local has_cuda=0
      command -v hipconfig >/dev/null 2>&1 && has_rocm=1
      command -v nvcc >/dev/null 2>&1 && has_cuda=1
      if (( has_rocm == 1 && has_cuda == 1 )); then
        echo "Both ROCm and CUDA toolkits were found. Set ACCELERATOR_BACKEND explicitly." >&2
        return 1
      elif (( has_rocm == 1 )); then
        printf 'rocm\n'
      elif (( has_cuda == 1 )); then
        printf 'cuda\n'
      else
        echo "No ROCm or CUDA compiler was detected. Install a GPU toolkit or set ACCELERATOR_BACKEND explicitly." >&2
        return 1
      fi
      ;;
    *)
      echo "ACCELERATOR_BACKEND must be auto, rocm, or cuda; got: $requested" >&2
      return 1
      ;;
  esac
}

ACCELERATOR_BACKEND_RESOLVED="$(resolve_accelerator_backend)" || exit 1
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_ROOT/vendor/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_CPP_DIR/build-$ACCELERATOR_BACKEND_RESOLVED}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_BUILD_DIR/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/$MODEL_FILENAME}"
SERVER_URL="http://$SERVER_HOST:$SERVER_PORT"
RUN_STATE_DIR="$PROJECT_ROOT/.run"
SERVER_PID_FILE="$RUN_STATE_DIR/llama-server-$ACCELERATOR_BACKEND_RESOLVED.pid"
SERVER_LOG_FILE="$PROJECT_ROOT/logs/llama-server-$ACCELERATOR_BACKEND_RESOLVED.log"
