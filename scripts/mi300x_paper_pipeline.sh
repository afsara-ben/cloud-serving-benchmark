#!/usr/bin/env bash
# Complete AMD-only pipeline for the runtime poster and bottlenecks report.
set -euo pipefail
csb_amd_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$csb_amd_root"
export DEVICES="${DEVICES:-0}"
export OUTPUT="${OUTPUT:-$csb_amd_root/results/mi300x/paper-70b}"
export MODEL_SIZE="${MODEL_SIZE:-70b}"
export PROFILE_TAG="${PROFILE_TAG:-paper1}"
csb_amd_status=0
# This stage creates its own venv, builds HIP, prepares models and runs serving.
PROFILE=0 bash scripts/mi300x_pipeline.sh || csb_amd_status=$?
csb_amd_python="$csb_amd_root/.run/mi300x/venv/bin/python"
read -r -a csb_amd_devices <<< "$DEVICES"
read -r -a csb_amd_formats <<< "${FORMATS:-IQ1_M Q2_K Q4_K_M Q8_0}"
read -r -a csb_amd_contexts <<< "${CONTEXTS:-2048 4096 8192 16384 65536}"
read -r -a csb_amd_concurrency <<< "${CONCURRENCY:-8 16 32}"
csb_amd_base=(scripts/run_mi300x.py --output "$OUTPUT" --devices "${csb_amd_devices[@]}"
    --model-size "$MODEL_SIZE" --rocm "${ROCM_PATH:-/opt/rocm}" --jobs "${JOBS:-8}"
    --profile-tag "$PROFILE_TAG" --output-tokens "${OUTPUT_TOKENS:-512}")
"$csb_amd_python" "${csb_amd_base[@]}" setup --diagnostic
"$csb_amd_python" "${csb_amd_base[@]}" bottlenecks --formats Q2_K Q4_K_M \
    --contexts "${BOTTLENECK_CONTEXT:-2048}" --concurrency "${BOTTLENECK_CONCURRENCY:-8}" \
    --capture-batches "${CAPTURE_BATCHES:-1}" || csb_amd_status=$?
"$csb_amd_python" "${csb_amd_base[@]}" report
"$csb_amd_python" "${csb_amd_base[@]}" paper --formats "${csb_amd_formats[@]}" \
    --contexts "${csb_amd_contexts[@]}" --concurrency "${csb_amd_concurrency[@]}" \
    --repetitions "${REPETITIONS:-1}" --samples "${SAMPLES:-5}" \
    --bottleneck-context "${BOTTLENECK_CONTEXT:-2048}" \
    --bottleneck-concurrency "${BOTTLENECK_CONCURRENCY:-8}" || csb_amd_status=$?
exit "$csb_amd_status"
