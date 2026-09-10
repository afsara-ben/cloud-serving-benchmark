#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command python3
mkdir -p "$RUN_STATE_DIR"

python3 "$PROJECT_ROOT/benchmark/load_test.py" \
  --base-url "$SERVER_URL" \
  --model "$MODEL_ALIAS" \
  --topics "$PROJECT_ROOT/prompts/topics.txt" \
  --concurrency 1 \
  --requests 1 \
  --target-prompt-tokens 128 \
  --max-output-tokens 16 \
  --warmup-requests 0 \
  --timeout "$REQUEST_TIMEOUT_SECONDS" \
  --seed "$RANDOM_SEED" \
  --output "$RUN_STATE_DIR/smoke-test.json"

echo "Smoke test passed."
