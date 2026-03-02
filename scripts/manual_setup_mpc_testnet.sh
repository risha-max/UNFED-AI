#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DRY_RUN=0
YES_REUSE_PORTS=0
TERMINAL_CMD="${UNFED_TERMINAL_CMD:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes-reuse-ports)
      YES_REUSE_PORTS=1
      shift
      ;;
    *)
      echo "[setup] Unknown argument: $1"
      echo "Usage: $0 [--dry-run] [--yes-reuse-ports]"
      exit 2
      ;;
  esac
done

_require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[setup] Missing prerequisite command: $cmd"
    exit 1
  fi
}

_pick_terminal_cmd() {
  if [[ -n "$TERMINAL_CMD" ]]; then
    return
  fi
  if command -v gnome-terminal >/dev/null 2>&1; then
    TERMINAL_CMD="gnome-terminal"
    return
  fi
  if command -v konsole >/dev/null 2>&1; then
    TERMINAL_CMD="konsole"
    return
  fi
  if command -v xterm >/dev/null 2>&1; then
    TERMINAL_CMD="xterm"
    return
  fi
  echo "[setup] Missing terminal emulator. Install gnome-terminal/konsole/xterm or set UNFED_TERMINAL_CMD."
  exit 1
}

if [[ "${UNFED_SKIP_PREREQ:-0}" != "1" ]]; then
  _pick_terminal_cmd
  _require_cmd "$TERMINAL_CMD"
  _require_cmd cast
  _require_cmd anvil
  _require_cmd ss
else
  _pick_terminal_cmd
fi
_require_cmd bash
_require_cmd python

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[setup] Missing .venv. Create it first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# Auto-load operator credentials from deployed.env if available.
if [[ -f "deployed.env" ]]; then
  if [[ -z "${OPERATOR_PRIVATE_KEY:-}" || -z "${OPERATOR_ADDRESS:-}" ]]; then
    # shellcheck disable=SC1091
    source deployed.env
  fi
fi

if [[ -z "${OPERATOR_PRIVATE_KEY:-}" || -z "${OPERATOR_ADDRESS:-}" ]]; then
  echo "[setup] OPERATOR_PRIVATE_KEY and OPERATOR_ADDRESS are required."
  echo "[setup] Export them manually or put them in deployed.env."
  exit 1
fi

PORTS=(8545 50050 50070 50051 50052 50053 50054 8080)
USED_PORTS=()
for p in "${PORTS[@]}"; do
  if ss -ltn "( sport = :$p )" | sed -n '2p' | grep -q .; then
    USED_PORTS+=("$p")
  fi
done
if [[ ${#USED_PORTS[@]} -gt 0 && "$YES_REUSE_PORTS" -ne 1 ]]; then
  echo "[setup] Ports already in use: ${USED_PORTS[*]}"
  echo "[setup] Re-run with --yes-reuse-ports to continue without killing running services."
  exit 1
fi

_sleep() {
  local seconds="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] sleep $seconds"
  else
    sleep "$seconds"
  fi
}

_open_tab() {
  local title="$1"
  local cmd="$2"
  local bootstrap="cd '$PROJECT_DIR'; source .venv/bin/activate; $cmd; exec bash"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] terminal=$TERMINAL_CMD title=\"$title\" cmd=\"$bootstrap\""
    return
  fi
  if [[ "$TERMINAL_CMD" == "gnome-terminal" ]]; then
    gnome-terminal --title="$title" -- bash -lc "$bootstrap"
    return
  fi
  if [[ "$TERMINAL_CMD" == "konsole" ]]; then
    konsole --new-tab -p "tabtitle=$title" --hold -e bash -lc "$bootstrap"
    return
  fi
  if [[ "$TERMINAL_CMD" == "xterm" ]]; then
    xterm -T "$title" -hold -e bash -lc "$bootstrap"
    return
  fi
  "$TERMINAL_CMD" -e bash -lc "$bootstrap"
}

_ensure_cluster_config_cmd='if [[ ! -f cluster_config.json ]]; then source deployed.env; cat > cluster_config.json <<EOF
{
  "name": "UNFED Local MPC Testnet",
  "chain_rpc_url": "'"${CHAIN_RPC_URL:-http://localhost:8545}"'",
  "escrow_contract_address": "'"${ESCROW_ADDRESS:-0x0000000000000000000000000000000000000000}"'",
  "staking_token_address": "'"${TOKEN_ADDRESS:-0x0000000000000000000000000000000000000000}"'",
  "operator_private_key": "'"${OPERATOR_PRIVATE_KEY}"'"
}
EOF
fi'

FUND_STAKE_CMD='set -euo pipefail; source deployed.env; export CHAIN_RPC_URL TOKEN_ADDRESS ESCROW_ADDRESS; MNEMONIC="${MNEMONIC:-test test test test test test test test test test test junk}"; STAKE_WEI="${STAKE_WEI:-100000000000000000000}"; FUND_WEI="${FUND_WEI:-200000000000000000000}"; OP_KEY="$(cast wallet private-key "$MNEMONIC" 0)"; for i in 1 2 3; do KEY="$(cast wallet private-key "$MNEMONIC" "$i")"; ADDR="$(cast wallet address --private-key "$KEY")"; BAL="$(cast call "$TOKEN_ADDRESS" "balanceOf(address)(uint256)" "$ADDR" --rpc-url "$CHAIN_RPC_URL")"; if [[ "$BAL" -lt "$FUND_WEI" ]]; then cast send "$TOKEN_ADDRESS" "transfer(address,uint256)" "$ADDR" "$FUND_WEI" --rpc-url "$CHAIN_RPC_URL" --private-key "$OP_KEY" >/dev/null; fi; done; for i in 0 1 2 3; do KEY="$(cast wallet private-key "$MNEMONIC" "$i")"; ADDR="$(cast wallet address --private-key "$KEY")"; STAKED="$(cast call "$ESCROW_ADDRESS" "stakedBalance(address)(uint256)" "$ADDR" --rpc-url "$CHAIN_RPC_URL")"; if [[ "$STAKED" -lt "$STAKE_WEI" ]]; then cast send "$TOKEN_ADDRESS" "approve(address,uint256)" "$ESCROW_ADDRESS" "$STAKE_WEI" --rpc-url "$CHAIN_RPC_URL" --private-key "$KEY" >/dev/null; cast send "$ESCROW_ADDRESS" "stake(uint256)" "$STAKE_WEI" --rpc-url "$CHAIN_RPC_URL" --private-key "$KEY" >/dev/null; fi; echo "[stake] $ADDR => $(cast call "$ESCROW_ADDRESS" "stakedBalance(address)(uint256)" "$ADDR" --rpc-url "$CHAIN_RPC_URL")"; done'

echo "[setup] Starting manual MPC testnet orchestration..."
echo "[setup] Order: chain -> funding/staking -> registry -> daemon -> nodes -> web"

_open_tab "UNFED Chain Deploy" "export OPERATOR_PRIVATE_KEY='$OPERATOR_PRIVATE_KEY'; export OPERATOR_ADDRESS='$OPERATOR_ADDRESS'; ./scripts/start_local_chain.sh"
_sleep 4
_open_tab "UNFED Fund + Stake" "$FUND_STAKE_CMD"
_sleep 4
_open_tab "UNFED Registry" "$_ensure_cluster_config_cmd; python -m network.registry_server --port 50050 --cluster-config cluster_config.json"
_sleep 2
_open_tab "UNFED Daemon" "python -m network.daemon_node --port 50070 --registry localhost:50050"
_sleep 2
_open_tab "UNFED Node 0 (MPC)" "python -m node.server --shard-index 0 --port 50051 --registry localhost:50050 --shards-dir shards --eth-address 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 --willing-to-mpc"
_open_tab "UNFED Node 1" "python -m node.server --shard-index 1 --port 50052 --registry localhost:50050 --shards-dir shards --eth-address 0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
_open_tab "UNFED Node 2" "python -m node.server --shard-index 2 --port 50053 --registry localhost:50050 --shards-dir shards --eth-address 0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
_open_tab "UNFED Node 3" "python -m node.server --shard-index 3 --port 50054 --registry localhost:50050 --shards-dir shards --eth-address 0x90F79bf6EB2c4f870365E785982E1f101E93b906"
_sleep 2
_open_tab "UNFED Web" "python -m web.server --port 8080 --registry localhost:50050"

echo "[setup] Done."
echo "[setup] Dashboard: http://localhost:8080"
echo "[setup] Registry gRPC: localhost:50050"
echo "[setup] Daemon gRPC: localhost:50070"
