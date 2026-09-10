#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command curl
mkdir -p "$PROJECT_ROOT/models"

if [[ -s "$MODEL_PATH" ]]; then
  echo "Model already exists: $MODEL_PATH"
  exit 0
fi

curl_args=(
  --fail
  --location
  --retry 5
  --retry-delay 2
  --continue-at -
  --output "$MODEL_PATH"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  curl_args+=(--header "Authorization: Bearer $HF_TOKEN")
fi

echo "Downloading $MODEL_FILENAME"
curl "${curl_args[@]}" "$MODEL_URL"
require_file "$MODEL_PATH"
echo "Downloaded model to $MODEL_PATH"
