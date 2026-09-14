#!/usr/bin/env bash
# MI300X-only end-to-end launcher. Does not source common.sh or execute.sh.
set -euo pipefail
csb_mi300x_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$csb_mi300x_root"
csb_mi300x_venv="$csb_mi300x_root/.run/mi300x/venv"
if [[ ! -x "$csb_mi300x_venv/bin/python" ]]; then
    "${PYTHON:-python3}" -m venv "$csb_mi300x_venv"
fi
csb_mi300x_python="$csb_mi300x_venv/bin/python"
"$csb_mi300x_python" -m pip install -r scripts/requirements-mi300x.txt

read -r -a csb_mi300x_devices <<< "${DEVICES:-0}"
read -r -a csb_mi300x_formats <<< "${FORMATS:-IQ1_M Q2_K Q4_K_M Q8_0}"
read -r -a csb_mi300x_contexts <<< "${CONTEXTS:-2048 4096 8192 16384 65536}"
read -r -a csb_mi300x_concurrency <<< "${CONCURRENCY:-8 16 32}"
csb_mi300x_base=(scripts/run_mi300x.py --rocm "${ROCM_PATH:-/opt/rocm}"
    --devices "${csb_mi300x_devices[@]}" --model-size "${MODEL_SIZE:-70b}"
    --model-dir "${MODEL_DIR:-$csb_mi300x_root/models/mi300x}"
    --output "${OUTPUT:-$csb_mi300x_root/results/mi300x/70b-context}"
    --jobs "${JOBS:-8}" --output-tokens "${OUTPUT_TOKENS:-512}")
if [[ -n "${MANIFEST:-}" ]]; then csb_mi300x_base+=(--manifest "$MANIFEST"); fi
csb_mi300x_sweep=(--formats "${csb_mi300x_formats[@]}" --contexts "${csb_mi300x_contexts[@]}"
    --concurrency "${csb_mi300x_concurrency[@]}" --repetitions "${REPETITIONS:-1}")
"$csb_mi300x_python" "${csb_mi300x_base[@]}" setup
"$csb_mi300x_python" "${csb_mi300x_base[@]}" prepare --formats "${csb_mi300x_formats[@]}"
"$csb_mi300x_python" "${csb_mi300x_base[@]}" plan "${csb_mi300x_sweep[@]}"
csb_mi300x_status=0
"$csb_mi300x_python" "${csb_mi300x_base[@]}" serve "${csb_mi300x_sweep[@]}" --resume || csb_mi300x_status=$?
# Always export surviving serving results, even if a later profiling stage fails.
"$csb_mi300x_python" "${csb_mi300x_base[@]}" report
if [[ "${PROFILE:-1}" == 1 ]]; then
    read -r -a csb_mi300x_profile_formats <<< "${PROFILE_FORMATS:-${FORMATS:-Q2_K Q4_K_M}}"
    read -r -a csb_mi300x_profile_contexts <<< "${PROFILE_CONTEXTS:-${CONTEXTS:-2048}}"
    read -r -a csb_mi300x_profile_concurrency <<< "${PROFILE_CONCURRENCY:-${CONCURRENCY:-8}}"
    "$csb_mi300x_python" "${csb_mi300x_base[@]}" setup --diagnostic
    "$csb_mi300x_python" "${csb_mi300x_base[@]}" profile --formats "${csb_mi300x_profile_formats[@]}" \
        --contexts "${csb_mi300x_profile_contexts[@]}" --concurrency "${csb_mi300x_profile_concurrency[@]}" \
        --profile-tag "${PROFILE_TAG:-run1}" --launch-count "${LAUNCH_COUNT:-8}" || csb_mi300x_status=$?
    "$csb_mi300x_python" "${csb_mi300x_base[@]}" report
fi
exit "$csb_mi300x_status"
