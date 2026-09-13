#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

LLAMA_SERVER_BIN="${PROFILE_SERVER_BIN:-$LLAMA_SERVER_BIN}"
require_file "$LLAMA_SERVER_BIN"
require_file "$MODEL_PATH"
if [[ "$ACCELERATOR_BACKEND_RESOLVED" == rocm ]]; then
  PROFILE_TOOL="${PROFILE_TOOL:-rocprof-trace}"
else
  PROFILE_TOOL="${PROFILE_TOOL:-nsys}"
fi
PROFILE_CONCURRENCY="${PROFILE_CONCURRENCY:-8}"
PROFILE_PORT="${PROFILE_PORT:-8081}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-$PROJECT_ROOT/results/profile-$MODEL_ALIAS-c$PROFILE_CONCURRENCY-$PROFILE_TOOL-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$PROJECT_ROOT/.run"
profile_workload_args=()
profile_server_args=()
[[ "${SERVER_NO_CONTEXT_SHIFT:-0}" != 1 ]] || profile_server_args+=(--no-context-shift)
[[ -z "${SERVER_CACHE_TYPE_K:-}" ]] || profile_server_args+=(--cache-type-k "$SERVER_CACHE_TYPE_K")
[[ -z "${SERVER_CACHE_TYPE_V:-}" ]] || profile_server_args+=(--cache-type-v "$SERVER_CACHE_TYPE_V")
[[ -z "${SERVER_FIT:-}" ]] || profile_server_args+=(--fit "$SERVER_FIT")
[[ -z "${SERVER_CHAT_TEMPLATE_FILE:-}" ]] || profile_server_args+=(--chat-template-file "$SERVER_CHAT_TEMPLATE_FILE")
[[ -z "${SERVER_SPLIT_MODE:-}" ]] || profile_server_args+=(--split-mode "$SERVER_SPLIT_MODE")
[[ -z "${SERVER_TENSOR_SPLIT:-}" ]] || profile_server_args+=(--tensor-split "$SERVER_TENSOR_SPLIT")
[[ -z "${SERVER_MAIN_GPU:-}" ]] || profile_server_args+=(--main-gpu "$SERVER_MAIN_GPU")
[[ -z "${SERVER_LOAD_MODE:-}" ]] || profile_server_args+=(--load-mode "$SERVER_LOAD_MODE")
if [[ "$ACCELERATOR_BACKEND_RESOLVED" == rocm ]]; then
  [[ "$PROFILE_TOOL" == rocprof-trace || "$PROFILE_TOOL" == rocprof-counters ]] || die "ROCm requires rocprof-trace or rocprof-counters."
  require_command hipcc
  ROCM_PATH="${ROCM_PATH:-$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)}"
  require_file "$ROCM_PATH/include/rocprofiler-sdk-roctx/roctx.h"
  hipcc -std=c++17 -shared -fPIC -pthread -DCSB_USE_ROCM \
    -I "$ROCM_PATH/include" "$PROJECT_ROOT/benchmark/profile_gate.cpp" \
    -L "$ROCM_PATH/lib" -Wl,-rpath,"$ROCM_PATH/lib" -lrocprofiler-sdk-roctx \
    -o "$PROJECT_ROOT/.run/profile_gate-rocm.so"
  gate_library="$PROJECT_ROOT/.run/profile_gate-rocm.so"
  PROFILE_BIN="${ROCPROFV3_BIN:-$(command -v rocprofv3 || true)}"
  visibility_env=("HIP_VISIBLE_DEVICES=$GPU_DEVICE")
  profile_workload_args+=(--rocm-root "$ROCM_PATH")
else
  [[ "$PROFILE_TOOL" == nsys || "$PROFILE_TOOL" == ncu ]] || die "CUDA requires nsys or ncu."
  require_command nvcc
  CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
  CUDA_TOOLKIT_LIB="$CUDA_TOOLKIT_ROOT/targets/x86_64-linux/lib"
  g++ -std=c++17 -shared -fPIC -pthread \
    -I "$CUDA_TOOLKIT_ROOT/targets/x86_64-linux/include" "$PROJECT_ROOT/benchmark/profile_gate.cpp" \
    -L "$CUDA_TOOLKIT_LIB" -Wl,-rpath,"$CUDA_TOOLKIT_LIB" -lcudart \
    -o "$PROJECT_ROOT/.run/profile_gate.so"
  gate_library="$PROJECT_ROOT/.run/profile_gate.so"
  visibility_env=("CUDA_VISIBLE_DEVICES=$GPU_DEVICE")
  if [[ "$PROFILE_TOOL" == nsys ]]; then
    PROFILE_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
  else
    PROFILE_BIN="${NCU_BIN:-$(command -v ncu || true)}"
  fi
fi
[[ -n "$PROFILE_BIN" ]] || die "Set NSYS_BIN, NCU_BIN or ROCPROFV3_BIN to the requested profiler executable."
[[ "${PROFILE_REPEAT_PROMPT:-0}" != 1 ]] || profile_workload_args+=(--repeat-prompt)
[[ "${PROFILE_PREFIX_REUSE:-0}" != 1 ]] || profile_workload_args+=(--prefix-reuse)
[[ -z "${PROFILE_METRICS:-}" ]] || profile_workload_args+=(--metrics "$PROFILE_METRICS")
[[ -z "${PROFILE_METRIC_GROUP:-}" ]] || profile_workload_args+=(--metric-group "$PROFILE_METRIC_GROUP")
if [[ -n "${PROFILE_SECTIONS:-}" ]]; then
  IFS=',' read -r -a profile_sections <<< "$PROFILE_SECTIONS"
  for profile_section in "${profile_sections[@]}"; do
    profile_workload_args+=(--section "$profile_section")
  done
fi
[[ -z "${PROFILE_DEVICES:-}" ]] || profile_workload_args+=(--devices "$PROFILE_DEVICES")
[[ -z "${PROFILE_EXPECTED_DEVICES:-}" ]] || profile_workload_args+=(--expected-devices "$PROFILE_EXPECTED_DEVICES")
[[ -z "${PROFILE_NVTX_INCLUDE:-}" ]] || profile_workload_args+=(--nvtx-include "$PROFILE_NVTX_INCLUDE")
[[ "${PROFILE_REQUIRE_NVTX:-0}" != 1 ]] || profile_workload_args+=(--require-nvtx)
[[ -z "${PROFILE_OPERATION_METADATA:-}" ]] || profile_workload_args+=(--operation-metadata "$PROFILE_OPERATION_METADATA")
[[ -z "${PROFILE_PROMPT_FIXTURE:-}" ]] || profile_workload_args+=(--prompt-fixture "$PROFILE_PROMPT_FIXTURE")
[[ "${EXACT_PROMPT_TOKENS:-0}" != 1 ]] || profile_workload_args+=(--exact-prompt-tokens)
[[ -z "${PROFILE_KERNEL_ITERATION_RANGE:-}" ]] || profile_workload_args+=(--kernel-iteration-range "$PROFILE_KERNEL_ITERATION_RANGE")
env "${visibility_env[@]}" python3 "$PROJECT_ROOT/benchmark/profile_cuda.py" \
  --tool "$PROFILE_TOOL" --profiler "$PROFILE_BIN" \
  --gate-library "$gate_library" \
  --base-url "http://$SERVER_HOST:$PROFILE_PORT" --model "$MODEL_ALIAS" \
  --concurrency "$PROFILE_CONCURRENCY" \
  --prompt-tokens "$TARGET_PROMPT_TOKENS" --output-tokens "$MAX_OUTPUT_TOKENS" \
  --topics "$PROJECT_ROOT/prompts/topics.txt" --seed "$RANDOM_SEED" \
  "${profile_workload_args[@]}" \
  --timeout "$REQUEST_TIMEOUT_SECONDS" --output "$PROFILE_OUTPUT" \
  --kernel-regex "${PROFILE_KERNEL_REGEX:-mul_mat_q|mul_mat_vec_q}" \
  --launch-count "${PROFILE_LAUNCH_COUNT:-4}" --launch-skip "${PROFILE_LAUNCH_SKIP:-0}" \
  --cache-control "${PROFILE_CACHE_CONTROL:-none}" --clock-control "${PROFILE_CLOCK_CONTROL:-none}" \
  --filter-mode "${PROFILE_FILTER_MODE:-global}" \
  -- "$LLAMA_SERVER_BIN" --model "$MODEL_PATH" --alias "$MODEL_ALIAS" \
  --host "$SERVER_HOST" --port "$PROFILE_PORT" --n-gpu-layers "$GPU_LAYERS" \
  --parallel "$SERVER_PARALLEL" --ctx-size "$SERVER_CTX_SIZE" \
  --batch-size "$SERVER_BATCH_SIZE" --ubatch-size "$SERVER_UBATCH_SIZE" \
  "${profile_server_args[@]}" \
  --flash-attn "$FLASH_ATTN" --cont-batching --no-cache-prompt \
  --cache-ram 0 --no-cache-idle-slots --metrics --jinja --log-verbosity 4
