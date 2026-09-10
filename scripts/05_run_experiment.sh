#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command curl
require_command python3
require_file "$MODEL_PATH"

curl -fsS "$SERVER_URL/health" >/dev/null || die "llama-server is not healthy at $SERVER_URL"

read -r -a concurrency_levels <<< "$CONCURRENCY_LEVELS"
for concurrency in "${concurrency_levels[@]}"; do
  if (( concurrency > SERVER_PARALLEL )); then
    die "Concurrency $concurrency exceeds SERVER_PARALLEL=$SERVER_PARALLEL"
  fi
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$PROJECT_ROOT/results/$run_id"
raw_dir="$run_dir/raw"
mkdir -p "$raw_dir"
cp "$CONFIG_FILE" "$run_dir/experiment.env"

{
  echo "experiment_name=$EXPERIMENT_NAME"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "accelerator_backend=$ACCELERATOR_BACKEND_RESOLVED"
  echo "llama_cpp_commit=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
  echo "model_path=$MODEL_PATH"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "model_sha256=$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
  fi
  if [[ "$ACCELERATOR_BACKEND_RESOLVED" == "rocm" ]]; then
    echo "rocm_path=$(hipconfig -R 2>/dev/null || true)"
    echo "rocm_version=$(hipconfig --version 2>/dev/null | head -n 1 || true)"
    echo "gpu_targets=$(rocminfo 2>/dev/null | awk '/Name:/ && /gfx/ {print $2}' | sort -u | paste -sd, - || true)"
  else
    echo "cuda_version=$(nvcc --version 2>/dev/null | tail -n 1 || true)"
    echo "gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd';' - || true)"
  fi
} > "$run_dir/metadata.txt"

curl -fsS "$SERVER_URL/metrics" > "$run_dir/metrics-before.prom" || true

for repetition in $(seq 1 "$REPETITIONS"); do
  for concurrency in "${concurrency_levels[@]}"; do
    output_file="$raw_dir/c${concurrency}-r${repetition}.json"
    echo "Running concurrency=$concurrency repetition=$repetition"
    python3 "$PROJECT_ROOT/benchmark/load_test.py" \
      --base-url "$SERVER_URL" \
      --model "$MODEL_ALIAS" \
      --topics "$PROJECT_ROOT/prompts/topics.txt" \
      --concurrency "$concurrency" \
      --requests "$REQUESTS_PER_LEVEL" \
      --target-prompt-tokens "$TARGET_PROMPT_TOKENS" \
      --max-output-tokens "$MAX_OUTPUT_TOKENS" \
      --warmup-requests "$WARMUP_REQUESTS" \
      --timeout "$REQUEST_TIMEOUT_SECONDS" \
      --seed "$((RANDOM_SEED + repetition))" \
      --repetition "$repetition" \
      --output "$output_file"
    sleep 2
  done
done

curl -fsS "$SERVER_URL/metrics" > "$run_dir/metrics-after.prom" || true

python3 "$PROJECT_ROOT/benchmark/summarize.py" \
  "$raw_dir" \
  --output-csv "$run_dir/summary.csv" \
  --output-markdown "$run_dir/summary.md"

echo "Experiment complete: $run_dir"
echo "Summary: $run_dir/summary.md"
