#!/usr/bin/env bash
# Run this shard on l2x02: Q2_K and Q4_K_M, r1 serving and reduced profiling.
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$root/scripts/rtxpro6000_pipeline.sh" \
  --formats Q4_K_M,Q2_K \
  --study-name cuda-context-study-rtxpro6000-q2-q4-r1 \
  --fast "$@"
