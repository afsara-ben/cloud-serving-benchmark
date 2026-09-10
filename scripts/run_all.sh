#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  "$script_dir/06_stop_server.sh" || true
}
trap cleanup EXIT INT TERM

"$script_dir/00_install_prereqs.sh"
"$script_dir/01_build_llama_cpp.sh"
"$script_dir/02_download_model.sh"
"$script_dir/03_start_server.sh"
"$script_dir/04_smoke_test.sh"
"$script_dir/05_run_experiment.sh"
