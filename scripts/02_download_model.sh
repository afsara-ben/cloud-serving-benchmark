#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command curl
require_command python3
mkdir -p "$(dirname "$MODEL_PATH")" "$RUN_STATE_DIR"

# Keep authentication out of process arguments. Curl does not forward this
# Authorization header to other hosts when following download redirects.
curl_model() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    [[ "$HF_TOKEN" != *$'\n'* && "$HF_TOKEN" != *$'\r'* ]] || die "HF_TOKEN contains an invalid newline."
    local token_escaped="${HF_TOKEN//\\/\\\\}"
    token_escaped="${token_escaped//\"/\\\"}"
    printf 'header = "Authorization: Bearer %s"\n' "$token_escaped" | curl --config - "$@"
  else
    curl "$@"
  fi
}

expected_size="${MODEL_SIZE_BYTES:-}"
expected_sha="${MODEL_SHA256:-}"
headers_path="$(mktemp "$RUN_STATE_DIR/download-headers.XXXXXX")"
trap 'rm -f "$headers_path"' EXIT

if [[ -z "$expected_size" ]]; then
  curl_model --fail --location --silent --show-error --retry 5 --retry-delay 2 \
    --head --output "$headers_path" "$MODEL_URL"
  mapfile -t remote_info < <(python3 - "$headers_path" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
blocks = re.split(r"\n\s*\n", text.strip())
headers = [{k.lower(): v.strip() for k, v in re.findall(r"^([^:\n]+):\s*(.*)$", block, re.M)}
           for block in blocks]
linked_size = next((h["x-linked-size"] for h in headers if "x-linked-size" in h), "")
final_size = headers[-1].get("content-length", "") if headers else ""
linked_etag = next((h["x-linked-etag"].strip('"') for h in headers if "x-linked-etag" in h), "")
print(linked_size or final_size)
print(linked_etag if re.fullmatch(r"[0-9a-fA-F]{64}", linked_etag) else "")
PY
  )
  expected_size="${remote_info[0]:-}"
  expected_sha="${expected_sha:-${remote_info[1]:-}}"
fi

[[ "$expected_size" =~ ^[0-9]+$ && "$expected_size" -gt 4 ]] || \
  die "Could not determine the model's size. Supply MODEL_SIZE_BYTES from the pinned model metadata."
[[ -z "$expected_sha" || "$expected_sha" =~ ^[0-9a-fA-F]{64}$ ]] || die "MODEL_SHA256 must contain 64 hexadecimal characters."

verify_model() {
  python3 - "$1" "$expected_size" "$expected_sha" <<'PY'
import hashlib
import sys
from pathlib import Path

path, expected_size, expected_sha = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3].lower()
if not path.is_file() or path.stat().st_size != expected_size:
    sys.exit(1)
with path.open("rb") as stream:
    if stream.read(4) != b"GGUF":
        sys.exit(1)
    if expected_sha:
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha:
            sys.exit(1)
PY
}

partial_path="$MODEL_PATH.part"
if [[ -e "$MODEL_PATH" ]]; then
  if verify_model "$MODEL_PATH"; then
    echo "Verified existing model: $MODEL_PATH ($expected_size bytes)"
    exit 0
  fi
  # Recover a partial file left by the original downloader without ever treating
  # a nonempty file as a completed model. Preserve suspicious/full-size files.
  if [[ ! -e "$partial_path" ]] && python3 - "$MODEL_PATH" "$expected_size" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
sys.exit(0 if path.is_file() and path.stat().st_size < int(sys.argv[2]) else 1)
PY
  then
    mv "$MODEL_PATH" "$partial_path"
  else
    die "Existing model failed size, GGUF, or SHA256 verification: $MODEL_PATH. Move it aside before retrying."
  fi
fi

if [[ -e "$partial_path" ]] && verify_model "$partial_path"; then
  mv "$partial_path" "$MODEL_PATH"
  echo "Verified completed partial download: $MODEL_PATH"
  exit 0
fi

echo "Downloading $MODEL_FILENAME"
curl_model --fail --location --silent --show-error --retry 5 --retry-delay 2 \
  --continue-at - --output "$partial_path" "$MODEL_URL"
verify_model "$partial_path" || die "Downloaded file failed size, GGUF, or SHA256 verification; retained $partial_path for inspection."
mv "$partial_path" "$MODEL_PATH"
echo "Verified download: $MODEL_PATH ($expected_size bytes)"
