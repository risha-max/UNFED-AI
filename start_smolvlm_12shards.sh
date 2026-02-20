#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# UNFED AI — Start SmolVLM-256M network with 12 SHARDS + On-Chain Staking
#
# Layout: 2 vision shards (6 layers each) + 10 text shards (3 layers each)
#
# Boots:
#   0. Anvil local chain (20 accounts) + deploy contracts
#   1. Fund & stake all node Ethereum accounts
#   2. Registry server with escrow config          (port 50050)
#   3. Vision shard 0                              (port 50060)
#   4. Vision shard 1                              (port 50061)
#   5. MPC Node B (text shard 0, role B)           (port 50071)
#   6. MPC Node A (text shard 0, role A)           (port 50070)
#   7. Text shards 1-9                             (ports 50072-50080)
#   8. Web dashboard                               (port 8080)
#
# Total: 14 compute nodes + registry + web = 16 processes
#
# Usage:
#   ./start_smolvlm_12shards.sh          # start everything
#   ./start_smolvlm_12shards.sh --stop   # kill all processes
# ═══════════════════════════════════════════════════════════════════════════

set -e

export PATH="$HOME/.foundry/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ─── Configurable parameters ───
SHARDS_DIR="${UNFED_SHARDS_DIR:-shards_smolvlm}"
REGISTRY_PORT=50050
WEB_PORT=8080
LOG_DIR="$PROJECT_DIR/logs"
RPC="http://localhost:8545"
STAKE_AMOUNT="200"
CLIENT_DEPOSIT="5000"

# ─── Anvil deterministic accounts (mnemonic: test test ... junk) ───
# Account 0: Operator
OPERATOR_ADDR="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
OPERATOR_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Accounts 1-12: Staking nodes (vision 0-1, mpc_a, text 1-9)
NODE_ADDRS=(
    "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"  # 1: vision shard 0
    "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"  # 2: vision shard 1
    "0x90F79bf6EB2c4f870365E785982E1f101E93b906"  # 3: mpc node A (text shard 0)
    "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"  # 4: text shard 1
    "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"  # 5: text shard 2
    "0x976EA74026E726554dB657fA54763abd0C3a0aa9"  # 6: text shard 3
    "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955"  # 7: text shard 4
    "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f"  # 8: text shard 5
    "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720"  # 9: text shard 6
    "0xBcd4042DE499D14e55001CcbB24a551F3b954096"  # 10: text shard 7
    "0x71bE63f3384f5fb98995898A86B02Fb2426c5788"  # 11: text shard 8
    "0xFABB0ac9d68B0B445fB7357272Ff202C5651694a"  # 12: text shard 9
)
NODE_KEYS=(
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"
    "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"
    "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"
    "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e"
    "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356"
    "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97"
    "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6"
    "0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897"
    "0x701b615bbdfb9de65240bc28bd21bbc0d996645a3dd57e7b12bc2bdf6f192c82"
    "0xa267530f49f8280200edf313ee7af6b827f2a8bce2897751d06a843f644967b1"
)
NODE_NAMES=(
    "Vision-0" "Vision-1" "MPC-A" "Text-1" "Text-2" "Text-3"
    "Text-4" "Text-5" "Text-6" "Text-7" "Text-8" "Text-9"
)

# Account 13: Client
CLIENT_ADDR="0x1CBd3b2770909D4e10f157cABC84C7264073C9Ec"
CLIENT_KEY="0x47c99abed3324a2707c28affff1267e45918ec8c3f20b8aa892e8b065d2942dd"

# Port assignments
VISION_0_PORT=50060
VISION_1_PORT=50061
MPC_A_PORT=50070
MPC_B_PORT=50071
TEXT_BASE_PORT=50072  # text shards 1-9 use ports 50072-50080

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
    echo "Run: python -m tools.cli split <model_path> -o $SHARDS_DIR --shards text_decoder=10 vision_encoder=2 --format pt"
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

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  UNFED AI — SmolVLM-256M — 12 Shards + On-Chain Staking      ${NC}"
echo -e "${CYAN}  (2 vision x 6 layers) + (10 text x 3 layers) = 14 nodes     ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Blockchain Setup
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 1] Blockchain Setup${NC}"
echo -e "─────────────────────────────────────────────"

kill $(cat /tmp/anvil.pid 2>/dev/null) 2>/dev/null || true
sleep 1

echo -e "  ${BLUE}Starting Anvil (20 accounts) + deploying contracts...${NC}"
UNFED_ANVIL_ACCOUNTS=20 bash scripts/start_local_chain.sh > "$LOG_DIR/chain.log" 2>&1
source contracts/deployed.env

TOKEN=$TOKEN_ADDRESS
ESCROW=$ESCROW_ADDRESS

echo -e "  ${GREEN}Chain:${NC}   $RPC"
echo -e "  ${GREEN}Token:${NC}   $TOKEN"
echo -e "  ${GREEN}Escrow:${NC}  $ESCROW"
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Fund & Stake All 12 Nodes
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 2] Fund & Stake 12 Nodes${NC}"
echo -e "─────────────────────────────────────────────"

fund_and_stake() {
    local name="$1" addr="$2" key="$3" amount="$4"
    local wei=$(cast --to-wei "$amount")

    cast send "$TOKEN" "transfer(address,uint256)" "$addr" "$wei" \
        --private-key "$OPERATOR_KEY" --rpc-url "$RPC" > /dev/null 2>&1

    cast send "$TOKEN" "approve(address,uint256)" "$ESCROW" "$wei" \
        --private-key "$key" --rpc-url "$RPC" > /dev/null 2>&1

    cast send "$ESCROW" "stake(uint256)" "$wei" \
        --private-key "$key" --rpc-url "$RPC" > /dev/null 2>&1

    local eligible=$(cast call "$ESCROW" "isEligible(address)(bool)" "$addr" --rpc-url "$RPC")
    echo -e "  ${GREEN}${name}${NC} (${addr:0:10}...): staked ${YELLOW}${amount} UNFED${NC}, eligible=$eligible"
}

for i in "${!NODE_ADDRS[@]}"; do
    fund_and_stake "${NODE_NAMES[$i]}" "${NODE_ADDRS[$i]}" "${NODE_KEYS[$i]}" "$STAKE_AMOUNT"
done

# Client deposit
echo -e "  ${BLUE}Client deposits $CLIENT_DEPOSIT UNFED to escrow...${NC}"
WEI_CLIENT=$(cast --to-wei $CLIENT_DEPOSIT)
cast send "$TOKEN" "transfer(address,uint256)" "$CLIENT_ADDR" "$WEI_CLIENT" \
    --private-key "$OPERATOR_KEY" --rpc-url "$RPC" > /dev/null 2>&1
cast send "$TOKEN" "approve(address,uint256)" "$ESCROW" "$WEI_CLIENT" \
    --private-key "$CLIENT_KEY" --rpc-url "$RPC" > /dev/null 2>&1
cast send "$ESCROW" "deposit(uint256)" "$WEI_CLIENT" \
    --private-key "$CLIENT_KEY" --rpc-url "$RPC" > /dev/null 2>&1
echo -e "  ${GREEN}Client${NC}   (${CLIENT_ADDR:0:10}...): deposited ${YELLOW}$CLIENT_DEPOSIT UNFED${NC}"
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
    "name": "SmolVLM-12Shard-Cluster",
    "description": "SmolVLM-256M with 12 shards (2 vision + 10 text) and on-chain staking",
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
# PHASE 4: Start Network (16 processes)
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 4] Start Network Components${NC}"
echo -e "─────────────────────────────────────────────"

ALL_PIDS=()
STEP=1
TOTAL_STEPS=16

start_step() {
    echo -e "  ${BLUE}[$STEP/$TOTAL_STEPS]${NC} $1"
    STEP=$((STEP + 1))
}

# 1. Registry
start_step "Registry (port $REGISTRY_PORT, escrow-gated)..."
python -m network.registry_server \
    --port $REGISTRY_PORT \
    --cluster-config "$CLUSTER_CONFIG" \
    > "$LOG_DIR/registry.log" 2>&1 &
REGISTRY_PID=$!
ALL_PIDS+=($REGISTRY_PID)
sleep 3
if ! kill -0 $REGISTRY_PID 2>/dev/null; then
    echo -e "  ${RED}Registry failed! Check $LOG_DIR/registry.log${NC}"
    exit 1
fi
echo -e "  ${GREEN}PID: $REGISTRY_PID${NC}"

# 2. Vision shard 0
start_step "Vision shard 0 (port $VISION_0_PORT, layers 0-6)..."
python -m node.server \
    --shard-index 0 \
    --port $VISION_0_PORT \
    --model-type smolvlm_vision \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    --eth-address "${NODE_ADDRS[0]}" \
    > "$LOG_DIR/vision_0.log" 2>&1 &
VISION_0_PID=$!
ALL_PIDS+=($VISION_0_PID)
sleep 4
if ! kill -0 $VISION_0_PID 2>/dev/null; then
    echo -e "  ${RED}Vision shard 0 failed! Check $LOG_DIR/vision_0.log${NC}"
    exit 1
fi
echo -e "  ${GREEN}PID: $VISION_0_PID${NC}"

# 3. Vision shard 1
start_step "Vision shard 1 (port $VISION_1_PORT, layers 6-12)..."
python -m node.server \
    --shard-index 1 \
    --port $VISION_1_PORT \
    --model-type smolvlm_vision \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    --eth-address "${NODE_ADDRS[1]}" \
    > "$LOG_DIR/vision_1.log" 2>&1 &
VISION_1_PID=$!
ALL_PIDS+=($VISION_1_PID)
sleep 4
if ! kill -0 $VISION_1_PID 2>/dev/null; then
    echo -e "  ${RED}Vision shard 1 failed! Check $LOG_DIR/vision_1.log${NC}"
    exit 1
fi
echo -e "  ${GREEN}PID: $VISION_1_PID${NC}"

# 4. MPC Node B (text shard 0, role B — start first, waits for A)
start_step "MPC Node B (port $MPC_B_PORT, text shard 0, role B)..."
python -m network.mpc_shard0 \
    --role B \
    --port $MPC_B_PORT \
    --peer "localhost:$MPC_A_PORT" \
    --shards-dir "$SHARDS_DIR" \
    > "$LOG_DIR/mpc_b.log" 2>&1 &
MPC_B_PID=$!
ALL_PIDS+=($MPC_B_PID)
sleep 2
echo -e "  ${GREEN}PID: $MPC_B_PID${NC}"

# 5. MPC Node A (text shard 0, role A — registers with registry)
start_step "MPC Node A (port $MPC_A_PORT, text shard 0, role A)..."
python -m network.mpc_shard0 \
    --role A \
    --port $MPC_A_PORT \
    --peer "localhost:$MPC_B_PORT" \
    --advertise "localhost:$MPC_A_PORT" \
    --registry "localhost:$REGISTRY_PORT" \
    --shards-dir "$SHARDS_DIR" \
    --eth-address "${NODE_ADDRS[2]}" \
    > "$LOG_DIR/mpc_a.log" 2>&1 &
MPC_A_PID=$!
ALL_PIDS+=($MPC_A_PID)
sleep 5
if ! kill -0 $MPC_A_PID 2>/dev/null; then
    echo -e "  ${RED}MPC Node A failed! Check $LOG_DIR/mpc_a.log${NC}"
    exit 1
fi
echo -e "  ${GREEN}PID: $MPC_A_PID${NC}"

# 6-14. Text shards 1-9
TEXT_PIDS=()
for i in $(seq 1 9); do
    PORT=$((TEXT_BASE_PORT + i - 1))
    NODE_IDX=$((i + 2))  # NODE_ADDRS index: 3=text1, 4=text2, ..., 11=text9
    start_step "Text shard $i (port $PORT, layers $((i*3))-$((i*3+3)))..."
    python -m node.server \
        --shard-index $i \
        --port $PORT \
        --model-type smolvlm \
        --shards-dir "$SHARDS_DIR" \
        --registry "localhost:$REGISTRY_PORT" \
        --eth-address "${NODE_ADDRS[$NODE_IDX]}" \
        > "$LOG_DIR/text_${i}.log" 2>&1 &
    PID=$!
    TEXT_PIDS+=($PID)
    ALL_PIDS+=($PID)
    sleep 3
    if ! kill -0 $PID 2>/dev/null; then
        echo -e "  ${RED}Text shard $i failed! Check $LOG_DIR/text_${i}.log${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}PID: $PID${NC}"
done

# 15. Web dashboard
start_step "Web dashboard (port $WEB_PORT)..."
python -m web.server \
    --port $WEB_PORT \
    --registry "localhost:$REGISTRY_PORT" \
    > "$LOG_DIR/web.log" 2>&1 &
WEB_PID=$!
ALL_PIDS+=($WEB_PID)
sleep 2
echo -e "  ${GREEN}PID: $WEB_PID${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Network is UP — 12 Shards, On-Chain Staking Enforced!       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}Blockchain:${NC}"
echo -e "    Anvil:            $RPC"
echo -e "    Token (UNFED):    $TOKEN"
echo -e "    Escrow:           $ESCROW"
echo ""
echo -e "  ${YELLOW}Vision Shards (2 shards x 6 layers):${NC}"
echo -e "    Shard 0:  localhost:$VISION_0_PORT  (${NODE_ADDRS[0]:0:10}...)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
echo -e "    Shard 1:  localhost:$VISION_1_PORT  (${NODE_ADDRS[1]:0:10}...)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
echo ""
echo -e "  ${YELLOW}MPC Text Shard 0 (3 layers, secret-shared):${NC}"
echo -e "    Node A:   localhost:$MPC_A_PORT  (${NODE_ADDRS[2]:0:10}...)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
echo -e "    Node B:   localhost:$MPC_B_PORT  (internal, no stake)"
echo ""
echo -e "  ${YELLOW}Text Shards 1-9 (3 layers each):${NC}"
for i in $(seq 1 9); do
    PORT=$((TEXT_BASE_PORT + i - 1))
    NODE_IDX=$((i + 2))
    echo -e "    Shard $i:  localhost:$PORT  (${NODE_ADDRS[$NODE_IDX]:0:10}...)  ${GREEN}$STAKE_AMOUNT UNFED${NC}"
done
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
echo -e "  Stop: ${RED}./start_smolvlm_12shards.sh --stop${NC}"
echo ""

# Save PIDs
cat > "$LOG_DIR/pids_12shards.txt" <<EOF
anvil=$(cat /tmp/anvil.pid)
registry=$REGISTRY_PID
vision_0=$VISION_0_PID
vision_1=$VISION_1_PID
mpc_b=$MPC_B_PID
mpc_a=$MPC_A_PID
$(for i in $(seq 1 9); do echo "text_${i}=${TEXT_PIDS[$((i-1))]}"; done)
web=$WEB_PID
EOF

echo "Waiting for processes (Ctrl+C to stop all)..."
echo ""

cleanup() {
    echo ""
    echo -e "${RED}Shutting down all ${#ALL_PIDS[@]} processes...${NC}"
    for pid in "${ALL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    kill $(cat /tmp/anvil.pid 2>/dev/null) 2>/dev/null
    wait 2>/dev/null
    echo -e "${GREEN}All processes stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

tail -f "$LOG_DIR/registry.log" "$LOG_DIR/mpc_a.log" "$LOG_DIR/web.log" &
TAIL_PID=$!
wait "${ALL_PIDS[@]}" 2>/dev/null
kill $TAIL_PID 2>/dev/null
