#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

if command -v apt-get >/dev/null 2>&1; then
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    apt_prefix=()
  elif command -v sudo >/dev/null 2>&1; then
    apt_prefix=(sudo)
  else
    die "apt-get is available, but root access or sudo is required to install packages."
  fi

  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y \
    build-essential ca-certificates cmake curl git jq libcurl4-openssl-dev \
    ninja-build pkg-config python3 python3-venv
else
  die "This installer targets Ubuntu/Debian AMD cloud images. Install Git, CMake, Ninja, a C++ compiler, curl, jq, and Python 3 manually."
fi

case "$ACCELERATOR_BACKEND_RESOLVED" in
  rocm)
    require_command hipconfig
    require_command rocminfo
    echo "Backend: ROCm"
    echo "ROCm installation: $(hipconfig -R)"
    echo "Detected AMD GPU targets:"
    rocminfo | awk '/Name:/ && /gfx/ {print $2}' | sort -u
    ;;
  cuda)
    require_command nvcc
    require_command nvidia-smi
    echo "Backend: CUDA"
    nvcc --version
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    ;;
esac
echo "Prerequisites are ready."
