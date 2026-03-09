#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${1:-${PROJECT_DIR}/.env.testnet}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

RUNTIME_DIR="${RUNTIME_DIR:-${PROJECT_DIR}/.runtime/public-testnet}"
PID_DIR="${RUNTIME_DIR}/pids"

if [[ ! -d "$PID_DIR" ]]; then
  echo "[stop] no pid directory at ${PID_DIR}; nothing to stop."
  exit 0
fi

stopped=0
for pid_file in "${PID_DIR}"/*.pid; do
  [[ -f "$pid_file" ]] || continue
  name="$(basename "$pid_file" .pid)"
  pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    continue
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "[stop] stopping ${name} (pid=${pid})"
    kill "$pid" >/dev/null 2>&1 || true
    stopped=$((stopped + 1))
  else
    echo "[stop] ${name} already stopped (pid=${pid})"
  fi
  rm -f "$pid_file"
done

echo "[stop] stopped ${stopped} service(s)."
