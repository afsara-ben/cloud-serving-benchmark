#!/usr/bin/env bash
# Concurrency-1 context sweep: 4 quantized Llama-3.3-70B formats x 2K-16K inputs.
# Reuses the study's own server launcher (03_start_server.sh) and load generator
# (benchmark/load_test.py) so numbers are comparable to the C8/16/32 series.
# Writes only to /tmp; the paper's results/ series is untouched.
set -Eeuo pipefail

ROOT="${ROOT:-/scratch/cloud-serving-benchmark}"
# The pinned llama-server was built against CUDA 12.8.1 (see build-cuda/CMakeCache.txt).
# The system default is CUDA 13, so libcudart.so.12 must come from the build toolkit.
CUDA_HOME="${CUDA_HOME:-/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT="${OUT:-/tmp/c1_sweep/results}"
SEED=20260912
FORMATS=(IQ1_M Q2_K Q4_K_M Q8_0)
PROMPTS=(2048 4096 8192 16384)

declare -A GGUF=(
  [IQ1_M]=Llama-3.3-70B-Instruct.i1-IQ1_M.gguf
  [Q2_K]=Llama-3.3-70B-Instruct-Q2_K.gguf
  [Q4_K_M]=Llama-3.3-70B-Instruct-Q4_K_M.gguf
  [Q8_0]=Llama-3.3-70B-Instruct.Q8_0.gguf
)

slot_capacity() { python3 -c "print((($1 + 512 + 1 + 255)//256)*256)"; }

current_env=""
cleanup() {
  if [[ -n "$current_env" ]]; then
    CONFIG_FILE="$current_env" bash "$ROOT/scripts/06_stop_server.sh" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$OUT"
echo "=== concurrency-1 sweep started $(date -u +%FT%TZ) ==="

for quant in "${FORMATS[@]}"; do
  for prompt in "${PROMPTS[@]}"; do
    cell="$OUT/$quant/p$prompt"
    if [[ -f "$cell/r1/raw.json" ]]; then
      echo "[skip] $quant/c1/p$prompt already measured"
      continue
    fi
    mkdir -p "$cell/r1"
    ctx=$(slot_capacity "$prompt")
    alias="Llama-3.3-70B-Instruct-$quant"
    model="$ROOT/models/${GGUF[$quant]}"

    cat > "$cell/study.env" <<ENVEOF
# Concurrency-1 cell: $quant / c1 / p$prompt
source $ROOT/config/context-study.env
EXPERIMENT_NAME=two-gpu-context-quantization-study
ACCELERATOR_BACKEND=cuda
GPU_DEVICE=0,1
SERVER_HOST=127.0.0.1
SERVER_PORT=8080
SERVER_PARALLEL=1
SERVER_CTX_SIZE=$ctx
SERVER_BATCH_SIZE=2048
SERVER_UBATCH_SIZE=512
GPU_LAYERS=99
FLASH_ATTN=on
CONCURRENCY_LEVELS=1
REQUESTS_PER_LEVEL=2
REPETITIONS=1
TARGET_PROMPT_TOKENS=$prompt
MAX_OUTPUT_TOKENS=512
WARMUP_REQUESTS=0
REQUEST_TIMEOUT_SECONDS=7200
RANDOM_SEED=$SEED
SERVER_SPLIT_MODE=layer
SERVER_TENSOR_SPLIT=1,1
SERVER_MAIN_GPU=0
SERVER_LOAD_MODE=none
SERVER_STARTUP_TIMEOUT_SECONDS=600
SERVER_CACHE_TYPE_K=f16
SERVER_CACHE_TYPE_V=f16
SERVER_NO_CONTEXT_SHIFT=1
SERVER_FIT=off
SERVER_LOG_VERBOSITY=4
SERVER_CHAT_TEMPLATE_FILE=$ROOT/config/templates/llama-3.1-benchmark.jinja
LLAMA_CPP_DIR=$ROOT/vendor/llama.cpp
LLAMA_BUILD_DIR=$ROOT/vendor/llama.cpp/build-cuda
LLAMA_SERVER_BIN=$ROOT/vendor/llama.cpp/build-cuda/bin/llama-server
MODEL_PATH=$model
MODEL_FILENAME=${GGUF[$quant]}
MODEL_ALIAS=$alias
ENVEOF

    echo "--- [$(date -u +%T)] $quant/c1/p$prompt (n_ctx=$ctx)"
    current_env="$cell/study.env"
    if ! CONFIG_FILE="$cell/study.env" bash "$ROOT/scripts/03_start_server.sh" > "$cell/start.log" 2>&1; then
      echo "    SERVER START FAILED; see $cell/start.log"
      echo "server_start_failed" > "$cell/status"
      current_env=""
      continue
    fi

    status=ok
    # 1 discarded warmup request, then 2 measured requests (matches C-warmup / 2C-measured).
    if ! python3 "$ROOT/benchmark/load_test.py" --base-url http://127.0.0.1:8080 --model "$alias" \
        --topics "$ROOT/prompts/topics.txt" --concurrency 1 --requests 1 \
        --target-prompt-tokens "$prompt" --max-output-tokens 512 --warmup-requests 0 \
        --timeout 7200 --seed "$SEED" --repetition 0 --exact-prompt-tokens \
        --output "$cell/warmup-discarded.json" > "$cell/warmup.log" 2>&1; then
      status=warmup_failed
    fi

    if [[ "$status" == ok ]]; then
      if ! python3 "$ROOT/benchmark/load_test.py" --base-url http://127.0.0.1:8080 --model "$alias" \
          --topics "$ROOT/prompts/topics.txt" --concurrency 1 --requests 2 \
          --target-prompt-tokens "$prompt" --max-output-tokens 512 --warmup-requests 0 \
          --timeout 7200 --seed "$((SEED + 1))" --repetition 1 --exact-prompt-tokens \
          --output "$cell/r1/raw.json" --prompts-output "$cell/prompts.json" \
          > "$cell/client.log" 2>&1; then
        status=measurement_failed
      fi
    fi

    cp -f "$ROOT/logs/llama-server-cuda.log" "$cell/server.log" 2>/dev/null || true
    CONFIG_FILE="$cell/study.env" bash "$ROOT/scripts/06_stop_server.sh" > "$cell/stop.log" 2>&1 || true
    current_env=""
    echo "$status" > "$cell/status"
    echo "    $status"
  done
done

echo "=== sweep finished $(date -u +%FT%TZ) ==="
