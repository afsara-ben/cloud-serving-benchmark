#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command curl
require_file "$LLAMA_SERVER_BIN"
require_file "$MODEL_PATH"
mkdir -p "$RUN_STATE_DIR" "$PROJECT_ROOT/logs"

if [[ -f "$SERVER_PID_FILE" ]]; then
  existing_pid="$(<"$SERVER_PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "llama-server is already running with PID $existing_pid"
    exit 0
  fi
  rm -f "$SERVER_PID_FILE"
fi

if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
  die "A server is already responding at $SERVER_URL but is not managed by this project."
fi

server_args=(
  --model "$MODEL_PATH"
  --alias "$MODEL_ALIAS"
  --host "$SERVER_HOST"
  --port "$SERVER_PORT"
  --n-gpu-layers "$GPU_LAYERS"
  --parallel "$SERVER_PARALLEL"
  --ctx-size "$SERVER_CTX_SIZE"
  --batch-size "$SERVER_BATCH_SIZE"
  --ubatch-size "$SERVER_UBATCH_SIZE"
  --flash-attn "$FLASH_ATTN"
  --cont-batching
  --no-cache-prompt
  --cache-ram 0
  --no-cache-idle-slots
  --metrics
  --jinja
)
if [[ -n "${SERVER_SPLIT_MODE:-}" ]]; then
  server_args+=(--split-mode "$SERVER_SPLIT_MODE")
fi
if [[ -n "${SERVER_TENSOR_SPLIT:-}" ]]; then
  server_args+=(--tensor-split "$SERVER_TENSOR_SPLIT")
fi
if [[ -n "${SERVER_MAIN_GPU:-}" ]]; then
  server_args+=(--main-gpu "$SERVER_MAIN_GPU")
fi
if [[ -n "${SERVER_LOAD_MODE:-}" ]]; then
  server_args+=(--load-mode "$SERVER_LOAD_MODE")
fi
if [[ -n "${SERVER_CACHE_TYPE_K:-}" ]]; then
  server_args+=(--cache-type-k "$SERVER_CACHE_TYPE_K")
fi
if [[ -n "${SERVER_CACHE_TYPE_V:-}" ]]; then
  server_args+=(--cache-type-v "$SERVER_CACHE_TYPE_V")
fi
if [[ "${SERVER_NO_CONTEXT_SHIFT:-0}" == "1" ]]; then
  server_args+=(--no-context-shift)
fi
if [[ -n "${SERVER_FIT:-}" ]]; then
  server_args+=(--fit "$SERVER_FIT")
fi
if [[ -n "${SERVER_LOG_VERBOSITY:-}" ]]; then
  server_args+=(--log-verbosity "$SERVER_LOG_VERBOSITY")
fi
if [[ -n "${SERVER_CHAT_TEMPLATE_FILE:-}" ]]; then
  require_file "$SERVER_CHAT_TEMPLATE_FILE"
  server_args+=(--chat-template-file "$SERVER_CHAT_TEMPLATE_FILE")
fi

: > "$SERVER_LOG_FILE"
{
  echo "Resolved llama.cpp commit: $(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
  printf 'Command:'
  printf ' %q' "$LLAMA_SERVER_BIN" "${server_args[@]}"
  printf '\n'
} >> "$SERVER_LOG_FILE"

case "$ACCELERATOR_BACKEND_RESOLVED" in
  rocm)
    visibility_env=("HIP_VISIBLE_DEVICES=$GPU_DEVICE")
    ;;
  cuda)
    visibility_env=("CUDA_VISIBLE_DEVICES=$GPU_DEVICE")
    ;;
esac

nohup env "${visibility_env[@]}" "$LLAMA_SERVER_BIN" "${server_args[@]}" \
  >> "$SERVER_LOG_FILE" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$SERVER_PID_FILE"

echo "Starting llama-server with PID $server_pid"
startup_timeout="${SERVER_STARTUP_TIMEOUT_SECONDS:-180}"
[[ "$startup_timeout" =~ ^[1-9][0-9]*$ ]] || die "SERVER_STARTUP_TIMEOUT_SECONDS must be a positive integer."
for _ in $(seq 1 "$startup_timeout"); do
  if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
    echo "Server is healthy at $SERVER_URL"
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 80 "$SERVER_LOG_FILE" >&2 || true
    die "llama-server exited before becoming healthy."
  fi
  sleep 1
done

tail -n 80 "$SERVER_LOG_FILE" >&2 || true
die "llama-server did not become healthy within $startup_timeout seconds."
