#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

require_command git
require_command cmake
require_command ninja

mkdir -p "$PROJECT_ROOT/vendor"

if [[ ! -e "$LLAMA_CPP_DIR" ]]; then
  git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

git -C "$LLAMA_CPP_DIR" rev-parse --git-dir >/dev/null 2>&1 || \
  die "$LLAMA_CPP_DIR exists but is not a Git checkout. Choose another LLAMA_CPP_DIR."
[[ -z "$(git -C "$LLAMA_CPP_DIR" status --porcelain --untracked-files=normal)" ]] || \
  die "$LLAMA_CPP_DIR has local changes. Commit or stash them before building; no files were changed."

# Exact commits already present locally need no network access. Branches and
# tags are fetched so an old local branch cannot silently select the wrong code.
if [[ "$LLAMA_CPP_REF" =~ ^[0-9a-fA-F]{40}$ ]] && \
   git -C "$LLAMA_CPP_DIR" cat-file -e "$LLAMA_CPP_REF^{commit}" 2>/dev/null; then
  requested_commit="$(git -C "$LLAMA_CPP_DIR" rev-parse "$LLAMA_CPP_REF^{commit}")"
else
  git -C "$LLAMA_CPP_DIR" fetch origin "$LLAMA_CPP_REF"
  requested_commit="$(git -C "$LLAMA_CPP_DIR" rev-parse --verify 'FETCH_HEAD^{commit}')"
fi
if [[ "$LLAMA_CPP_REF" =~ ^[0-9a-fA-F]{40}$ ]]; then
  [[ "$requested_commit" == "${LLAMA_CPP_REF,,}" ]] || die "Fetched commit does not match LLAMA_CPP_REF."
fi
git -C "$LLAMA_CPP_DIR" checkout --detach "$requested_commit"

resolved_commit="$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
echo "Building llama.cpp commit $resolved_commit with $ACCELERATOR_BACKEND_RESOLVED"

cmake_args=(
  -S "$LLAMA_CPP_DIR"
  -B "$LLAMA_BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF
  -DGGML_CUDA=OFF
  -DGGML_HIP=OFF
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
    cmake_args+=(
      -DGGML_CUDA=ON
      "-DCMAKE_CUDA_COMPILER=$(command -v nvcc)"
      "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-native}"
    )
    cmake "${cmake_args[@]}"
    ;;
esac

cmake --build "$LLAMA_BUILD_DIR" --config Release -j "$BUILD_JOBS" \
  --target llama-server llama-bench llama-batched-bench

require_file "$LLAMA_SERVER_BIN"
"$LLAMA_SERVER_BIN" --version
