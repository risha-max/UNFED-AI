#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# UNFED AI — Start SmolVLM-256M-Instruct network
#
# Boots:
#   1. Registry server        (port 50050)
#   2. Vision shard 0         (port 50060) — SigLIP encoder + connector
#   3. Text shard 0           (port 50061) — LLM layers 0-14 + embed_tokens
#   4. Text shard 1           (port 50062) — LLM layers 15-29 + norm + lm_head
#   5. Web dashboard          (port 8080)
#
# Prerequisites:
#   - Model split into shards_smolvlm/ (run: python -m shard.smolvlm_splitter)
#   - Virtual environment activated
#
# Usage:
#   ./start_smolvlm.sh          # start everything
#   ./start_smolvlm.sh --stop   # kill all UNFED processes
# ═══════════════════════════════════════════════════════════════════════════

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_ID="/home/oasis/models/SmolVLM-256M-Instruct"
SHARDS_DIR="shards_smolvlm"
REGISTRY_PORT=50050
VISION_PORT=50060
TEXT_PORT_BASE=50061
WEB_PORT=8080
LOG_DIR="$PROJECT_DIR/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─── Stop mode ───
if [[ "$1" == "--stop" || "$1" == "stop" ]]; then
    echo -e "${RED}Stopping UNFED AI processes...${NC}"
    pkill -f "network.registry_server" 2>/dev/null && echo "  Stopped registry" || true
    pkill -f "node.server.*smolvlm" 2>/dev/null && echo "  Stopped SmolVLM nodes" || true
    pkill -f "web.server" 2>/dev/null && echo "  Stopped web server" || true
    echo -e "${GREEN}Done.${NC}"
    exit 0
fi

# ─── Checks ───
if [[ ! -d "$SHARDS_DIR" ]]; then
    echo -e "${RED}Error: $SHARDS_DIR not found.${NC}"
    echo "Run: python -m shard.smolvlm_splitter --model-path $MODEL_ID"
    exit 1
fi

if [[ ! -f "$SHARDS_DIR/manifest.json" ]]; then
    echo -e "${RED}Error: $SHARDS_DIR/manifest.json not found.${NC}"
    exit 1
fi

# Activate venv if not already
if [[ -z "$VIRTUAL_ENV" && -d ".venv" ]]; then
    source .venv/bin/activate
fi

mkdir -p "$LOG_DIR"

echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${CYAN}  UNFED AI — SmolVLM-256M Network Startup  ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""

# ─── 1. Registry ───
echo -e "${BLUE}[1/5]${NC} Starting registry on port $REGISTRY_PORT..."
python -m network.registry_server --port $REGISTRY_PORT \
    > "$LOG_DIR/registry.log" 2>&1 &
REGISTRY_PID=$!
echo -e "  ${GREEN}PID: $REGISTRY_PID${NC}"
sleep 2

# Check if registry started
if ! kill -0 $REGISTRY_PID 2>/dev/null; then
    echo -e "${RED}Registry failed to start. Check $LOG_DIR/registry.log${NC}"
    exit 1
fi

# ─── 2. Vision shard ───
echo -e "${BLUE}[2/5]${NC} Starting vision shard 0 on port $VISION_PORT..."
python -m node.server \
    --shard-index 0 \
    --port $VISION_PORT \
    --model-type smolvlm_vision \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    > "$LOG_DIR/vision_0.log" 2>&1 &
VISION_PID=$!
echo -e "  ${GREEN}PID: $VISION_PID${NC}"
sleep 3

# ─── 3. Text shard 0 ───
TEXT_PORT_0=$TEXT_PORT_BASE
echo -e "${BLUE}[3/5]${NC} Starting text shard 0 on port $TEXT_PORT_0..."
python -m node.server \
    --shard-index 0 \
    --port $TEXT_PORT_0 \
    --model-type smolvlm \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    > "$LOG_DIR/text_0.log" 2>&1 &
TEXT0_PID=$!
echo -e "  ${GREEN}PID: $TEXT0_PID${NC}"
sleep 3

# ─── 4. Text shard 1 ───
TEXT_PORT_1=$((TEXT_PORT_BASE + 1))
echo -e "${BLUE}[4/5]${NC} Starting text shard 1 on port $TEXT_PORT_1..."
python -m node.server \
    --shard-index 1 \
    --port $TEXT_PORT_1 \
    --model-type smolvlm \
    --shards-dir "$SHARDS_DIR" \
    --registry "localhost:$REGISTRY_PORT" \
    > "$LOG_DIR/text_1.log" 2>&1 &
TEXT1_PID=$!
echo -e "  ${GREEN}PID: $TEXT1_PID${NC}"
sleep 3

# ─── 5. Web dashboard ───
echo -e "${BLUE}[5/5]${NC} Starting web dashboard on port $WEB_PORT..."
python -m web.server \
    --port $WEB_PORT \
    --registry "localhost:$REGISTRY_PORT" \
    > "$LOG_DIR/web.log" 2>&1 &
WEB_PID=$!
echo -e "  ${GREEN}PID: $WEB_PID${NC}"
sleep 2

# ─── Summary ───
echo ""
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Network is UP!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "  Registry:       ${YELLOW}localhost:$REGISTRY_PORT${NC}  (PID $REGISTRY_PID)"
echo -e "  Vision shard 0: ${YELLOW}localhost:$VISION_PORT${NC}   (PID $VISION_PID)"
echo -e "  Text shard 0:   ${YELLOW}localhost:$TEXT_PORT_0${NC}   (PID $TEXT0_PID)"
echo -e "  Text shard 1:   ${YELLOW}localhost:$TEXT_PORT_1${NC}   (PID $TEXT1_PID)"
echo -e "  Web dashboard:  ${YELLOW}http://localhost:$WEB_PORT${NC}   (PID $WEB_PID)"
echo ""
echo -e "  Model: ${GREEN}$MODEL_ID${NC}"
echo -e "  Shards: ${GREEN}1 vision + 2 text${NC}"
echo ""
echo -e "  Logs: $LOG_DIR/"
echo -e "  Stop: ${RED}./start_smolvlm.sh --stop${NC}"
echo ""

# Save PIDs for cleanup
cat > "$LOG_DIR/pids.txt" <<EOF
registry=$REGISTRY_PID
vision_0=$VISION_PID
text_0=$TEXT0_PID
text_1=$TEXT1_PID
web=$WEB_PID
EOF

echo "Waiting for processes (Ctrl+C to stop all)..."
echo ""

# Wait and handle Ctrl+C
cleanup() {
    echo ""
    echo -e "${RED}Shutting down...${NC}"
    kill $REGISTRY_PID $VISION_PID $TEXT0_PID $TEXT1_PID $WEB_PID 2>/dev/null
    wait 2>/dev/null
    echo -e "${GREEN}All processes stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Tail logs to show activity
tail -f "$LOG_DIR/registry.log" "$LOG_DIR/vision_0.log" \
    "$LOG_DIR/text_0.log" "$LOG_DIR/text_1.log" "$LOG_DIR/web.log" &
TAIL_PID=$!

wait $REGISTRY_PID $VISION_PID $TEXT0_PID $TEXT1_PID $WEB_PID 2>/dev/null
kill $TAIL_PID 2>/dev/null
