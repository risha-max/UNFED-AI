#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env.testnet"
REUSE_RUNNING=0
SKIP_CHECKLIST=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --reuse-running)
      REUSE_RUNNING=1
      shift
      ;;
    --skip-checklist)
      SKIP_CHECKLIST=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "[start] unknown argument: $1"
      echo "usage: $0 [--env-file PATH] [--reuse-running] [--skip-checklist] [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ "$DRY_RUN" == "1" && "$SKIP_CHECKLIST" == "0" ]]; then
  SKIP_CHECKLIST=1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[start] missing env file: $ENV_FILE"
  echo "[start] copy .env.public-testnet.example to .env.testnet first"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

RUNTIME_DIR="${RUNTIME_DIR:-${PROJECT_DIR}/.runtime/public-testnet}"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

REGISTRY_HOST="${REGISTRY_HOST:-127.0.0.1}"
REGISTRY_PORT="${REGISTRY_PORT:-50050}"
REGISTRY_ADDR="${REGISTRY_HOST}:${REGISTRY_PORT}"
REGISTRY_CLUSTER_CONFIG="${REGISTRY_CLUSTER_CONFIG:-cluster_config.json}"
REGISTRY_NO_CHAIN="${REGISTRY_NO_CHAIN:-0}"

DAEMON_HOST="${DAEMON_HOST:-127.0.0.1}"
DAEMON_PORT="${DAEMON_PORT:-50070}"
DAEMON_ADVERTISE_ADDR="${DAEMON_ADVERTISE_ADDR:-}"
DAEMON_ETH_ADDRESS="${DAEMON_ETH_ADDRESS:-}"
DAEMON_DB="${DAEMON_DB:-${RUNTIME_DIR}/daemon-chain.db}"

NODE_ENABLED="${NODE_ENABLED:-1}"
NODE_HOST="${NODE_HOST:-127.0.0.1}"
NODE_PORT="${NODE_PORT:-50051}"
NODE_ADVERTISE_ADDR="${NODE_ADVERTISE_ADDR:-}"
NODE_SHARD_INDEX="${NODE_SHARD_INDEX:-0}"
NODE_SHARDS_DIR="${NODE_SHARDS_DIR:-shards}"
NODE_ETH_ADDRESS="${NODE_ETH_ADDRESS:-}"
NODE_WILLING_TO_MPC="${NODE_WILLING_TO_MPC:-1}"
MOCK_NODE_ENABLED="${MOCK_NODE_ENABLED:-0}"
MOCK_NODE_MODEL_ID="${MOCK_NODE_MODEL_ID:-mock/testnet-model}"
MOCK_NODE_ADDRESS="${MOCK_NODE_ADDRESS:-mock-node:59999}"

WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8080}"
WEB_URL="${WEB_URL:-http://127.0.0.1:${WEB_PORT}}"

STARTED_PIDS=()

cleanup_on_error() {
  if [[ ${#STARTED_PIDS[@]} -eq 0 ]]; then
    return
  fi
  echo "[start] startup failed; stopping newly started services..."
  for pid in "${STARTED_PIDS[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup_on_error ERR

port_in_use() {
  local port="$1"
  local out
  out="$(ss -ltnH "( sport = :${port} )" || true)"
  [[ -n "$out" ]]
}

start_service() {
  local name="$1"
  local port="$2"
  local cmd="$3"
  local pid_file="${PID_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$port" -gt 0 ]] && port_in_use "$port"; then
      echo "[dry-run] port ${port} already in use (${name})"
    fi
    echo "[dry-run] $cmd"
    return
  fi

  if [[ "$port" -gt 0 ]] && port_in_use "$port"; then
    if [[ "$REUSE_RUNNING" == "1" ]]; then
      echo "[start] reusing existing ${name} on port ${port}"
      return
    fi
    echo "[start] port ${port} already in use (${name})"
    echo "[start] rerun with --reuse-running to keep existing services"
    exit 1
  fi

  if [[ "$port" -gt 0 ]]; then
    echo "[start] launching ${name} on port ${port}"
  else
    echo "[start] launching ${name}"
  fi

  nohup bash -lc "cd '$PROJECT_DIR' && source .venv/bin/activate && $cmd" >"$log_file" 2>&1 &
  local pid=$!
  STARTED_PIDS+=("$pid")
  echo "$pid" >"$pid_file"
}

wait_http_ready() {
  local name="$1"
  local url="$2"
  local timeout_s="${3:-45}"
  local started=0

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] wait ${name} url=${url}"
    return
  fi

  while (( started < timeout_s )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[start] ${name} ready: ${url}"
      return
    fi
    sleep 1
    started=$((started + 1))
  done
  echo "[start] timeout waiting for ${name}: ${url}"
  return 1
}

if [[ "$REGISTRY_NO_CHAIN" != "1" && ! -f "$PROJECT_DIR/$REGISTRY_CLUSTER_CONFIG" ]]; then
  echo "[start] missing registry cluster config: $PROJECT_DIR/$REGISTRY_CLUSTER_CONFIG"
  echo "[start] set REGISTRY_NO_CHAIN=1 or create the cluster config file"
  exit 1
fi

REGISTRY_CMD="python -m network.registry_server --port ${REGISTRY_PORT}"
if [[ "$REGISTRY_NO_CHAIN" == "1" ]]; then
  REGISTRY_CMD+=" --no-chain"
else
  REGISTRY_CMD+=" --cluster-config ${REGISTRY_CLUSTER_CONFIG}"
fi

DAEMON_CMD="python -m network.daemon_node --host ${DAEMON_HOST} --port ${DAEMON_PORT} --registry ${REGISTRY_ADDR} --db ${DAEMON_DB}"
if [[ -n "$DAEMON_ADVERTISE_ADDR" ]]; then
  DAEMON_CMD+=" --advertise ${DAEMON_ADVERTISE_ADDR}"
fi
if [[ -n "$DAEMON_ETH_ADDRESS" ]]; then
  DAEMON_CMD+=" --eth-address ${DAEMON_ETH_ADDRESS}"
fi

NODE_CMD="python -m node.server --host ${NODE_HOST} --port ${NODE_PORT} --registry ${REGISTRY_ADDR} --shard-index ${NODE_SHARD_INDEX} --shards-dir ${NODE_SHARDS_DIR}"
if [[ -n "$NODE_ADVERTISE_ADDR" ]]; then
  NODE_CMD+=" --advertise ${NODE_ADVERTISE_ADDR}"
fi
if [[ -n "$NODE_ETH_ADDRESS" ]]; then
  NODE_CMD+=" --eth-address ${NODE_ETH_ADDRESS}"
fi
if [[ "$NODE_WILLING_TO_MPC" == "1" ]]; then
  NODE_CMD+=" --willing-to-mpc"
fi

WEB_CMD="python -m web.server --host ${WEB_HOST} --port ${WEB_PORT} --registry ${REGISTRY_ADDR}"

echo "[start] env file: $ENV_FILE"
echo "[start] runtime dir: $RUNTIME_DIR"

start_service "registry" "$REGISTRY_PORT" "$REGISTRY_CMD"
sleep 2
start_service "daemon" "$DAEMON_PORT" "$DAEMON_CMD"
sleep 2

if [[ "$NODE_ENABLED" == "1" ]]; then
  start_service "node" "$NODE_PORT" "$NODE_CMD"
  sleep 2
else
  echo "[start] NODE_ENABLED=0, skipping node startup"
fi

if [[ "$MOCK_NODE_ENABLED" == "1" ]]; then
  MOCK_CMD="python scripts/mock_serving_node.py --registry ${REGISTRY_ADDR} --model-id ${MOCK_NODE_MODEL_ID} --address ${MOCK_NODE_ADDRESS}"
  start_service "mock_node" "0" "$MOCK_CMD"
  sleep 1
fi

start_service "web" "$WEB_PORT" "$WEB_CMD"
wait_http_ready "web" "${WEB_URL}/api/client/auth/mode" 60

echo "[start] services are up."
echo "[start] logs: $LOG_DIR"
echo "[start] pids: $PID_DIR"

if [[ "$SKIP_CHECKLIST" == "1" ]]; then
  echo "[start] checklist skipped (--skip-checklist)"
  trap - ERR
  exit 0
fi

trap - ERR
echo "[start] running public testnet checklist..."
"${PROJECT_DIR}/scripts/run_public_testnet_checklist.sh" "$ENV_FILE"

echo "[start] launch + checklist complete."
