#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command git
require_command cmake

mkdir -p "$PROJECT_ROOT/vendor"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

git -C "$LLAMA_CPP_DIR" fetch --tags origin "$LLAMA_CPP_REF"
git -C "$LLAMA_CPP_DIR" checkout --detach FETCH_HEAD

resolved_commit="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "Building llama.cpp commit $resolved_commit with $ACCELERATOR_BACKEND_RESOLVED"

cmake_args=(
  -S "$LLAMA_CPP_DIR"
  -B "$LLAMA_BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
)

case "$ACCELERATOR_BACKEND_RESOLVED" in
  rocm)
    require_command hipconfig
    cmake_args+=(
      -DGGML_HIP=ON
      -DGPU_TARGETS="$ROCM_GPU_TARGETS"
    )
    HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
      cmake "${cmake_args[@]}"
    ;;
  cuda)
    require_command nvcc
    cmake_args+=(-DGGML_CUDA=ON)
    if [[ -n "${CUDA_ARCHITECTURES:-}" ]]; then
      cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES")
    fi
    cmake "${cmake_args[@]}"
    ;;
esac

cmake --build "$LLAMA_BUILD_DIR" --config Release -j "$BUILD_JOBS" \
  --target llama-server llama-bench llama-batched-bench

require_file "$LLAMA_SERVER_BIN"
"$LLAMA_SERVER_BIN" --version
