#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# UNFED AI — SmolVLM-256M Auto-Assignment Demo
#
# Demonstrates the capacity-aware auto-assignment system.
# Instead of manually specifying shard indices and roles, each node
# starts in --auto mode and asks the registry for an assignment.
#
# Boots:
#   0. Anvil local chain (20 accounts) + deploy contracts
#   1. Fund & stake all node Ethereum accounts
#   2. Registry server with escrow config + manifest  (port 50050)
#   3. 14 auto-assigned nodes                         (ports 50060-50073)
#   4. Web dashboard                                  (port 8080)
#
# The registry auto-distributes roles: 2 vision, 2 MPC, 9 text, 1 guard
#
# Usage:
#   ./start_smolvlm_auto.sh          # start everything
#   ./start_smolvlm_auto.sh --stop   # kill all processes
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
NUM_NODES=14
BASE_PORT=50060

# ─── Anvil deterministic accounts (mnemonic: test test ... junk) ───
OPERATOR_ADDR="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
OPERATOR_KEY="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

NODE_ADDRS=(
    "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
    "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
    "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
    "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"
    "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"
    "0x976EA74026E726554dB657fA54763abd0C3a0aa9"
    "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955"
    "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f"
    "0xa0Ee7A142d267C1f36714E4a8F75612F20a79720"
    "0xBcd4042DE499D14e55001CcbB24a551F3b954096"
    "0x71bE63f3384f5fb98995898A86B02Fb2426c5788"
    "0xFABB0ac9d68B0B445fB7357272Ff202C5651694a"
    "0x1CBd3b2770909D4e10f157cABC84C7264073C9Ec"
    "0xdF3e18d64BC6A983f673Ab319CCaE4f1a57C7097"
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
    "0x47c99abed3324a2707c28affff1267e45918ec8c3f20b8aa892e8b065d2942dd"
    "0xc526ee95bf44d8fc405a158bb884d9d1238d99f0612e9f33d006bb0789009aaa"
)

# Client uses account 18
CLIENT_ADDR="0xcd3B766CCDd6AE721141F452C550Ca635964ce71"
CLIENT_KEY="0x8166f546bab6da521a8369cab06c5d2b9e46670292d85c875ee9ec20e84ffb61"

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
echo -e "${CYAN}  UNFED AI — SmolVLM-256M — Auto-Assignment Demo              ${NC}"
echo -e "${CYAN}  $NUM_NODES nodes, registry assigns roles automatically      ${NC}"
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
# PHASE 2: Fund & Stake All Nodes
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 2] Fund & Stake $NUM_NODES Nodes${NC}"
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
    echo -e "  ${GREEN}Node-$name${NC} (${addr:0:10}...): staked ${YELLOW}${amount} UNFED${NC}, eligible=$eligible"
}

for i in $(seq 0 $((NUM_NODES - 1))); do
    fund_and_stake "$i" "${NODE_ADDRS[$i]}" "${NODE_KEYS[$i]}" "$STAKE_AMOUNT"
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
    "name": "SmolVLM-AutoAssign-Cluster",
    "description": "SmolVLM-256M with auto-assigned roles and capacity-aware placement",
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
# PHASE 4: Start Registry + Upload Manifest
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 4] Start Registry & Upload Manifest${NC}"
echo -e "─────────────────────────────────────────────"

ALL_PIDS=()

echo -e "  ${BLUE}[1] Registry (port $REGISTRY_PORT, escrow-gated)...${NC}"
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

# Upload manifest to registry so auto-assignment can discover shards
MANIFEST_PATH="$SHARDS_DIR/manifest.json"
if [[ -f "$MANIFEST_PATH" ]]; then
    MODEL_ID=$(python -c "import json; print(json.load(open('$MANIFEST_PATH'))['model_id'])")
    echo -e "  ${BLUE}[2] Uploading manifest for $MODEL_ID to registry...${NC}"
    python -c "
import grpc, json, sys
sys.path.insert(0, 'proto')
import registry_pb2, registry_pb2_grpc

channel = grpc.insecure_channel('localhost:$REGISTRY_PORT')
stub = registry_pb2_grpc.RegistryStub(channel)
manifest = open('$MANIFEST_PATH').read()
model_id = json.loads(manifest)['model_id']
resp = stub.PutManifest(registry_pb2.PutManifestRequest(
    model_id=model_id, manifest_json=manifest))
print(f'  Manifest uploaded: success={resp.success}')
channel.close()
"
else
    echo -e "  ${RED}No manifest at $MANIFEST_PATH${NC}"
    exit 1
fi
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Launch Auto-Assigned Nodes
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 5] Launch $NUM_NODES Auto-Assigned Nodes${NC}"
echo -e "─────────────────────────────────────────────"
echo -e "  ${YELLOW}Each node probes hardware, asks registry for assignment,${NC}"
echo -e "  ${YELLOW}then loads its assigned shard and starts serving.${NC}"
echo ""

NODE_PIDS=()
for i in $(seq 0 $((NUM_NODES - 1))); do
    PORT=$((BASE_PORT + i))
    echo -e "  ${BLUE}[$((i+3))/$((NUM_NODES+3))] Auto-node $i (port $PORT)...${NC}"
    python -m node.server \
        --auto \
        --port $PORT \
        --shards-dir "$SHARDS_DIR" \
        --registry "localhost:$REGISTRY_PORT" \
        --eth-address "${NODE_ADDRS[$i]}" \
        > "$LOG_DIR/auto_node_${i}.log" 2>&1 &
    PID=$!
    NODE_PIDS+=($PID)
    ALL_PIDS+=($PID)
    # Stagger node launches to let assignment spread
    sleep 3
    if ! kill -0 $PID 2>/dev/null; then
        echo -e "  ${RED}Node $i failed! Check $LOG_DIR/auto_node_${i}.log${NC}"
        echo -e "  ${YELLOW}Last 5 lines:${NC}"
        tail -5 "$LOG_DIR/auto_node_${i}.log" 2>/dev/null || true
    else
        # Show what role was assigned
        ASSIGNED=$(grep -o 'Assigned:.*' "$LOG_DIR/auto_node_${i}.log" 2>/dev/null | head -1)
        echo -e "  ${GREEN}PID: $PID${NC} — ${ASSIGNED:-starting...}"
    fi
done
echo ""

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Web Dashboard
# ═══════════════════════════════════════════════════════════════

echo -e "${MAGENTA}[Phase 6] Web Dashboard${NC}"
echo -e "─────────────────────────────────────────────"

echo -e "  ${BLUE}[$((NUM_NODES+3))/$((NUM_NODES+3))] Web dashboard (port $WEB_PORT)...${NC}"
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
echo -e "${GREEN}  Network is UP — Auto-Assignment Mode!                       ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}Blockchain:${NC}"
echo -e "    Anvil:            $RPC"
echo -e "    Token (UNFED):    $TOKEN"
echo -e "    Escrow:           $ESCROW"
echo ""
echo -e "  ${YELLOW}Auto-Assigned Nodes ($NUM_NODES total):${NC}"
for i in $(seq 0 $((NUM_NODES - 1))); do
    PORT=$((BASE_PORT + i))
    ASSIGNED=$(grep -o 'Assigned:.*' "$LOG_DIR/auto_node_${i}.log" 2>/dev/null | head -1)
    echo -e "    Node $i:  localhost:$PORT  (${NODE_ADDRS[$i]:0:10}...)  $ASSIGNED"
done
echo ""
echo -e "  ${YELLOW}Services:${NC}"
echo -e "    Registry:         localhost:$REGISTRY_PORT  (escrow-gated, auto-assignment)"
echo -e "    Web dashboard:    ${CYAN}http://localhost:$WEB_PORT${NC}"
echo ""
echo -e "  ${YELLOW}Client:${NC}"
echo -e "    Address:          $CLIENT_ADDR"
echo -e "    Escrow balance:   ${GREEN}$CLIENT_DEPOSIT UNFED${NC}"
echo ""
echo -e "  Logs: $LOG_DIR/"
echo -e "  Stop: ${RED}./start_smolvlm_auto.sh --stop${NC}"
echo ""

# Save PIDs
cat > "$LOG_DIR/pids_auto.txt" <<EOF
anvil=$(cat /tmp/anvil.pid 2>/dev/null)
registry=$REGISTRY_PID
$(for i in $(seq 0 $((NUM_NODES - 1))); do echo "auto_node_${i}=${NODE_PIDS[$i]}"; done)
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

tail -f "$LOG_DIR/registry.log" "$LOG_DIR/auto_node_0.log" "$LOG_DIR/web.log" &
TAIL_PID=$!
wait "${ALL_PIDS[@]}" 2>/dev/null
kill $TAIL_PID 2>/dev/null
