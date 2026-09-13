#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

if [[ ! -f "$SERVER_PID_FILE" ]]; then
  echo "No managed llama-server PID file was found."
  exit 0
fi

server_pid="$(<"$SERVER_PID_FILE")"
server_state="$(ps -p "$server_pid" -o stat= 2>/dev/null || true)"
if ! kill -0 "$server_pid" 2>/dev/null || [[ "$server_state" == Z* ]]; then
  rm -f "$SERVER_PID_FILE"
  echo "The recorded server process is no longer running."
  exit 0
fi

server_command="$(ps -p "$server_pid" -o command= 2>/dev/null || true)"
if [[ "$server_command" != *"llama-server"* ]]; then
  die "PID $server_pid does not appear to be llama-server; refusing to stop it."
fi

kill "$server_pid"
for _ in $(seq 1 30); do
  server_state="$(ps -p "$server_pid" -o stat= 2>/dev/null || true)"
  if ! kill -0 "$server_pid" 2>/dev/null || [[ "$server_state" == Z* ]]; then
    rm -f "$SERVER_PID_FILE"
    echo "llama-server stopped."
    exit 0
  fi
  sleep 1
done

die "llama-server did not stop within 30 seconds; inspect PID $server_pid manually."
