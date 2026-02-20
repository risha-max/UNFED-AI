#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# UNFED AI — Start SmolVLM-256M network with ON-CHAIN STAKING
#
# Boots:
#   0. Anvil local chain + deploy contracts
#   1. Fund & stake node Ethereum accounts
#   2. Registry server with escrow config  (port 50050)
#   3. Vision shard 0                      (port 50060)
#   4. MPC Node B (shard 0, role B)        (port 50063)
#   5. MPC Node A (shard 0, role A)        (port 50061)
#   6. Text shard 1                        (port 50062)
#   7. Web dashboard                       (port 8080)
#
# Each node stakes UNFED tokens on-chain before registering.
# The registry gates registration via the escrow contract.
#
# Usage:
#   ./start_smolvlm_staked.sh          # start everything
#   ./start_smolvlm_staked.sh --stop   # kill all processes
#
# All paths, ports, and amounts are configurable via environment variables.
# Override any default by exporting before running, e.g.:
#
#   export UNFED_SHARDS_DIR=/data/my_shards
#   export UNFED_WEB_PORT=9090
#   ./start_smolvlm_staked.sh
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

export PATH="$HOME/.foundry/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ─── Configurable parameters (override via env) ───
SHARDS_DIR="${UNFED_SHARDS_DIR:-shards_smolvlm}"
MODEL_PATH="${UNFED_MODEL_PATH:-}"  # used only in error messages
REGISTRY_PORT="${UNFED_REGISTRY_PORT:-50050}"
VISION_PORT="${UNFED_VISION_PORT:-50060}"
MPC_A_PORT="${UNFED_MPC_A_PORT:-50061}"
TEXT1_PORT="${UNFED_TEXT1_PORT:-50062}"
MPC_B_PORT="${UNFED_MPC_B_PORT:-50063}"
WEB_PORT="${UNFED_WEB_PORT:-8080}"
LOG_DIR="${UNFED_LOG_DIR:-$PROJECT_DIR/logs}"
RPC="${UNFED_RPC_URL:-http://localhost:8545}"
STAKE_AMOUNT="${UNFED_STAKE_AMOUNT:-200}"       # UNFED tokens per node
CLIENT_DEPOSIT="${UNFED_CLIENT_DEPOSIT:-1000}"

# Anvil default accounts (deterministic) — override for custom devnet
OPERATOR_ADDR="${UNFED_OPERATOR_ADDR:-0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266}"
OPERATOR_KEY="${UNFED_OPERATOR_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}"

VISION_ADDR="${UNFED_VISION_ADDR:-0x70997970C51812dc3A010C7d01b50e0d17dc79C8}"
VISION_KEY="${UNFED_VISION_KEY:-0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d}"

MPC_A_ADDR="${UNFED_MPC_A_ADDR:-0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC}"
MPC_A_KEY="${UNFED_MPC_A_KEY:-0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a}"

TEXT1_ADDR="${UNFED_TEXT1_ADDR:-0x90F79bf6EB2c4f870365E785982E1f101E93b906}"
TEXT1_KEY="${UNFED_TEXT1_KEY:-0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6}"

CLIENT_ADDR="${UNFED_CLIENT_ADDR:-0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65}"
CLIENT_KEY="${UNFED_CLIENT_KEY:-0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ─── Stop mode ───
if [[ "$1" == "--stop" || "$1" == "stop" ]]; then
    echo -e "${RED}Stopping UNFED AI processes...${NC}"
    pkill -f "network.registry_server" 2>/dev/null && echo "  Stopped registry" || true
    pkill -f "node.server" 2>/dev/null && echo "  Stopped nodes" || true
    pkill -f "network.mpc_shard0" 2>/dev/null && echo "  Stopped MPC nodes" || true
    pkill -f "web.server" 2>/dev/null && echo "  Stopped web server" || true
    kill $(cat /tmp/anvil.pid 2>/dev/null) 2>/dev/null && echo "  Stopped Anvil" || true
    echo -e "${GREEN}Done.${NC}"
    exit 0
fi

# ─── Checks ───
if [[ ! -d "$SHARDS_DIR" ]]; then
    echo -e "${RED}Error: $SHARDS_DIR not found.${NC}"
    if [[ -n "$MODEL_PATH" ]]; then
        echo "Run: python -m tools.splitter $MODEL_PATH"
    else
        echo "Run: python -m tools.splitter <path-to-model>"
        echo "Or set UNFED_SHARDS_DIR to point to an existing shards directory."
    fi
    exit 1
fi
if ! command -v cast &>/dev/null; then
    echo -e "${RED}Error: Foundry 'cast' not found. Install: curl -L https://foundry.paradigm.xyz | bash && foundryup${NC}"
    exit 1
fi

# Activate venv if needed
if [[ -z "$VIRTUAL_ENV" && -d ".venv" ]]; then
    source .venv/bin/activate
fi

mkdir -p "$LOG_DIR"

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  UNFED AI — SmolVLM-256M + On-Chain Staking       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Blockchain Setup
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 1] Blockchain Setup${NC}"
echo -e "─────────────────────────────────────────────"

# Kill existing Anvil if running
kill $(cat /tmp/anvil.pid 2>/dev/null) 2>/dev/null || true
sleep 1

# Start Anvil + deploy contracts
echo -e "  ${BLUE}Starting Anvil + deploying contracts...${NC}"
bash scripts/start_local_chain.sh > "$LOG_DIR/chain.log" 2>&1
source contracts/deployed.env

TOKEN=$TOKEN_ADDRESS
ESCROW=$ESCROW_ADDRESS

echo -e "  ${GREEN}Chain:${NC}   $RPC"
echo -e "  ${GREEN}Token:${NC}   $TOKEN"
echo -e "  ${GREEN}Escrow:${NC}  $ESCROW"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Fund & Stake Nodes
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 2] Fund & Stake Nodes${NC}"
echo -e "─────────────────────────────────────────────"

WEI_STAKE=$(cast --to-wei $STAKE_AMOUNT)
WEI_CLIENT=$(cast --to-wei $CLIENT_DEPOSIT)

fund_and_stake() {
    local name="$1" addr="$2" key="$3" amount="$4"
    local wei=$(cast --to-wei "$amount")

    # Transfer tokens from operator
    cast send "$TOKEN" "transfer(address,uint256)" "$addr" "$wei" \
        --private-key "$OPERATOR_KEY" --rpc-url "$RPC" > /dev/null 2>&1

    # Approve escrow to spend tokens
    cast send "$TOKEN" "approve(address,uint256)" "$ESCROW" "$wei" \
        --private-key "$key" --rpc-url "$RPC" > /dev/null 2>&1

    # Stake
    cast send "$ESCROW" "stake(uint256)" "$wei" \
        --private-key "$key" --rpc-url "$RPC" > /dev/null 2>&1

    # Verify
    local staked=$(cast call "$ESCROW" "stakedBalance(address)(uint256)" "$addr" --rpc-url "$RPC")
    local eligible=$(cast call "$ESCROW" "isEligible(address)(bool)" "$addr" --rpc-url "$RPC")
    echo -e "  ${GREEN}$name${NC} (${addr:0:10}...): staked ${YELLOW}$amount UNFED${NC}, eligible=$eligible"
}

fund_and_stake "Vision Node" "$VISION_ADDR" "$VISION_KEY" "$STAKE_AMOUNT"
fund_and_stake "MPC Node A " "$MPC_A_ADDR"  "$MPC_A_KEY"  "$STAKE_AMOUNT"
fund_and_stake "Text Shard1" "$TEXT1_ADDR"   "$TEXT1_KEY"  "$STAKE_AMOUNT"

# Client deposit
echo -e "  ${BLUE}Client deposits $CLIENT_DEPOSIT UNFED to escrow...${NC}"
cast send "$TOKEN" "transfer(address,uint256)" "$CLIENT_ADDR" "$WEI_CLIENT" \
    --private-key "$OPERATOR_KEY" --rpc-url "$RPC" > /dev/null 2>&1
cast send "$TOKEN" "approve(address,uint256)" "$ESCROW" "$WEI_CLIENT" \
    --private-key "$CLIENT_KEY" --rpc-url "$RPC" > /dev/null 2>&1
cast send "$ESCROW" "deposit(uint256)" "$WEI_CLIENT" \
    --private-key "$CLIENT_KEY" --rpc-url "$RPC" > /dev/null 2>&1
echo -e "  ${GREEN}Client${NC}     (${CLIENT_ADDR:0:10}...): deposited ${YELLOW}$CLIENT_DEPOSIT UNFED${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Cluster Config
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 3] Create Cluster Config${NC}"
echo -e "─────────────────────────────────────────────"

CLUSTER_CONFIG="$PROJECT_DIR/contracts/cluster_config.json"
cat > "$CLUSTER_CONFIG" <<EOJSON
{
    "cluster_id": "",
    "name": "SmolVLM-Staked-Cluster",
    "description": "SmolVLM-256M with on-chain staking via Anvil",
    "operator": "$OPERATOR_ADDR",
    "public_endpoint": "localhost:$REGISTRY_PORT",
    "default_min_stake": 100.0,
    "default_slash_fraction": 0.5,
    "default_reward_scheme": "proportional",
    "default_base_rate": 0.001,
    "default_price_per_input_token": 0.0001,
    "default_price_per_output_token": 0.001,
    "chain_rpc_url": "$RPC",
    "escrow_contract_address": "$ESCROW",
    "staking_token_address": "$TOKEN",
    "operator_private_key": "$OPERATOR_KEY",
    "cooldown_seconds": 300,
    "challenge_window_seconds": 60
}
EOJSON
echo -e "  ${GREEN}Written:${NC} $CLUSTER_CONFIG"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Start Network
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 4] Start Network Components${NC}"
echo -e "─────────────────────────────────────────────"

# 1. Registry
echo -e "  ${BLUE}[1/6]${NC} Registry (port $REGISTRY_PORT, escrow-gated)..."
python -m network.registry_server \
    --port $REGISTRY_PORT \
    --cluster-config "$CLUSTER_CONFIG" \
    > "$LOG_DIR/registry.log" 2>&1 &
REGISTRY_PID=$!
sleep 3
if ! kill -0 $REGISTRY_PID 2>/dev/null; then
    echo -e "  ${RED}Registry failed! Check $LOG_DIR/registry.log${NC}"
    cat "$LOG_DIR/registry.log"
    exit 1
fi
echo -e "  ${GREEN}PID: $REGISTRY_PID${NC}"

# 2. Vision shard
echo -e "  ${BLUE}[2/6]${NC} Vision shard 0 (port $VISION_PORT)..."
python -m node.server \
    --shard-index 0 \
    --port $VISION_PORT \
    --model-type smolvlm_vision \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    --eth-address "$VISION_ADDR" \
    > "$LOG_DIR/vision_0.log" 2>&1 &
VISION_PID=$!
sleep 4
if ! kill -0 $VISION_PID 2>/dev/null; then
    echo -e "  ${RED}Vision node failed! Check $LOG_DIR/vision_0.log${NC}"
    cat "$LOG_DIR/vision_0.log"
    exit 1
fi
echo -e "  ${GREEN}PID: $VISION_PID${NC}"

# 3. MPC Node B (start first, waits for A)
echo -e "  ${BLUE}[3/6]${NC} MPC Node B (port $MPC_B_PORT, role B)..."
python -m network.mpc_shard0 \
    --role B \
    --port $MPC_B_PORT \
    --peer "localhost:$MPC_A_PORT" \
    --shards-dir "$SHARDS_DIR" \
    > "$LOG_DIR/mpc_b.log" 2>&1 &
MPC_B_PID=$!
sleep 2
echo -e "  ${GREEN}PID: $MPC_B_PID${NC}"

# 4. MPC Node A (entry point, registers with registry)
echo -e "  ${BLUE}[4/6]${NC} MPC Node A (port $MPC_A_PORT, role A, registers)..."
python -m network.mpc_shard0 \
    --role A \
    --port $MPC_A_PORT \
    --peer "localhost:$MPC_B_PORT" \
    --advertise "localhost:$MPC_A_PORT" \
    --registry "localhost:$REGISTRY_PORT" \
    --shards-dir "$SHARDS_DIR" \
    --eth-address "$MPC_A_ADDR" \
    > "$LOG_DIR/mpc_a.log" 2>&1 &
MPC_A_PID=$!
sleep 5
if ! kill -0 $MPC_A_PID 2>/dev/null; then
    echo -e "  ${RED}MPC Node A failed! Check $LOG_DIR/mpc_a.log${NC}"
    cat "$LOG_DIR/mpc_a.log"
    exit 1
fi
echo -e "  ${GREEN}PID: $MPC_A_PID${NC}"

# 5. Text shard 1
echo -e "  ${BLUE}[5/6]${NC} Text shard 1 (port $TEXT1_PORT)..."
python -m node.server \
    --shard-index 1 \
    --port $TEXT1_PORT \
    --model-type smolvlm \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    --eth-address "$TEXT1_ADDR" \
    > "$LOG_DIR/text_1.log" 2>&1 &
TEXT1_PID=$!
sleep 4
if ! kill -0 $TEXT1_PID 2>/dev/null; then
    echo -e "  ${RED}Text shard 1 failed! Check $LOG_DIR/text_1.log${NC}"
    cat "$LOG_DIR/text_1.log"
    exit 1
fi
echo -e "  ${GREEN}PID: $TEXT1_PID${NC}"

# 6. Web dashboard
echo -e "  ${BLUE}[6/6]${NC} Web dashboard (port $WEB_PORT)..."
python -m web.server \
    --port $WEB_PORT \
    --registry "localhost:$REGISTRY_PORT" \
    > "$LOG_DIR/web.log" 2>&1 &
WEB_PID=$!
sleep 2
echo -e "  ${GREEN}PID: $WEB_PID${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Network is UP — On-Chain Staking Enforced!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}Blockchain:${NC}"
echo -e "    Anvil:            $RPC"
echo -e "    Token (UNFED):    $TOKEN"
echo -e "    Escrow:           $ESCROW"
echo ""
echo -e "  ${YELLOW}Staked Nodes:${NC}"
echo -e "    Vision shard 0:   localhost:$VISION_PORT  ($VISION_ADDR)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
echo -e "    MPC shard 0 (A):  localhost:$MPC_A_PORT  ($MPC_A_ADDR)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
echo -e "    MPC shard 0 (B):  localhost:$MPC_B_PORT  (internal, no stake needed)"
echo -e "    Text shard 1:     localhost:$TEXT1_PORT  ($TEXT1_ADDR)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
echo ""
echo -e "  ${YELLOW}Services:${NC}"
echo -e "    Registry:         localhost:$REGISTRY_PORT  (escrow-gated)"
echo -e "    Web dashboard:    ${CYAN}http://localhost:$WEB_PORT${NC}"
echo ""
echo -e "  ${YELLOW}Client:${NC}"
echo -e "    Address:          $CLIENT_ADDR"
echo -e "    Escrow balance:   ${GREEN}$CLIENT_DEPOSIT UNFED${NC}"
echo ""
echo -e "  Logs: $LOG_DIR/"
echo -e "  Stop: ${RED}./start_smolvlm_staked.sh --stop${NC}"
echo ""

# Save PIDs
cat > "$LOG_DIR/pids_staked.txt" <<EOF
anvil=$(cat /tmp/anvil.pid)
registry=$REGISTRY_PID
vision_0=$VISION_PID
mpc_b=$MPC_B_PID
mpc_a=$MPC_A_PID
text_1=$TEXT1_PID
web=$WEB_PID
EOF

echo "Waiting for processes (Ctrl+C to stop all)..."
echo ""

cleanup() {
    echo ""
    echo -e "${RED}Shutting down...${NC}"
    kill $WEB_PID $TEXT1_PID $MPC_A_PID $MPC_B_PID $VISION_PID $REGISTRY_PID 2>/dev/null
    kill $(cat /tmp/anvil.pid 2>/dev/null) 2>/dev/null
    wait 2>/dev/null
    echo -e "${GREEN}All processes stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

tail -f "$LOG_DIR/registry.log" "$LOG_DIR/mpc_a.log" "$LOG_DIR/web.log" &
TAIL_PID=$!
wait $REGISTRY_PID $VISION_PID $MPC_A_PID $MPC_B_PID $TEXT1_PID $WEB_PID 2>/dev/null
kill $TAIL_PID 2>/dev/null
