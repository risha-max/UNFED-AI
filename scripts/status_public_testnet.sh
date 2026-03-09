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
LOG_DIR="${RUNTIME_DIR}/logs"

echo "[status] runtime_dir=${RUNTIME_DIR}"
echo "[status] pid_dir=${PID_DIR}"
echo "[status] log_dir=${LOG_DIR}"

if [[ ! -d "$PID_DIR" ]]; then
  echo "[status] no pid directory yet."
  exit 0
fi

for pid_file in "${PID_DIR}"/*.pid; do
  [[ -f "$pid_file" ]] || continue
  name="$(basename "$pid_file" .pid)"
  pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "[status] ${name}: running (pid=${pid})"
  else
    echo "[status] ${name}: not running"
  fi
done
