#!/usr/bin/env bash
# Run this shard on l2x01: IQ1_M and Q8_0, r1 serving and reduced profiling.
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$root/scripts/rtxpro6000_pipeline.sh" \
  --formats IQ1_M,Q8_0 \
  --study-name cuda-context-study-rtxpro6000-iq1-q8-r1 \
  --fast "$@"
