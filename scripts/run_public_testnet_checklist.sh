#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${1:-${PROJECT_DIR}/.env.testnet}"

if [ ! -f "$ENV_FILE" ]; then
  echo "[checklist] missing env profile: $ENV_FILE"
  echo "[checklist] copy .env.public-testnet.example -> .env.testnet and retry"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

echo "[checklist] using env profile: $ENV_FILE"
echo "[checklist] running static web/node preflight checks..."

"$PYTHON_BIN" "${PROJECT_DIR}/scripts/testnet_preflight.py" web \
  --host "${WEB_HOST:-0.0.0.0}"

"$PYTHON_BIN" "${PROJECT_DIR}/scripts/testnet_preflight.py" node \
  --advertise "${NODE_ADVERTISE_ADDR:-}" \
  --tls-cert "${NODE_TLS_CERT:-}" \
  --tls-key "${NODE_TLS_KEY:-}"

echo "[checklist] running runtime API readiness checks..."
"$PYTHON_BIN" "${PROJECT_DIR}/scripts/testnet_preflight.py" runtime \
  --web-url "${WEB_URL:-http://127.0.0.1:8080}"

echo "[checklist] GO/NO-GO checklist finished."
