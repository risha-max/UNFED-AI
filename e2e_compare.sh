#!/usr/bin/env bash
# Compare optimized vs baseline latency.
set -euo pipefail

export PYTHONUNBUFFERED=1
PYTHON="${UNFED_PYTHON:-.venv/bin/python}"
RUNS=5
WARMUP=1
MAX_TOKENS=30
PIDS=()

cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    PIDS=()
}
trap cleanup EXIT

start_cluster() {
    pkill -f "registry_server.*50050" 2>/dev/null || true
    pkill -f "node/server.py" 2>/dev/null || true
    sleep 1

    $PYTHON -m network.registry_server --port 50050 &
    PIDS+=($!)
    sleep 2

    for i in 0 1 2 3; do
        PORT=$((50051 + i))
        $PYTHON node/server.py --shard-index $i --port $PORT --shards-dir shards &
        PIDS+=($!)
    done

    echo "  Waiting for nodes to load and register..."
    sleep 15
}

stop_cluster() {
    cleanup
    sleep 2
}

echo "============================================================"
echo "  UNFED AI — Before/After Latency Comparison"
echo "  Model: Qwen/Qwen2.5-0.5B, 4 shards, CPU"
echo "============================================================"

# ─── BASELINE (original settings) ─────────────────────────────
echo ""
echo ">>> Phase 1/2: BASELINE (float32 wire, 4KB compress threshold)"
export UNFED_WIRE_DTYPE=""
export UNFED_COMPRESS_THRESHOLD=4096
export UNFED_PREFILL_MIN=999999

start_cluster
echo ""
$PYTHON benchmark.py --runs $RUNS --warmup $WARMUP --max-tokens $MAX_TOKENS
stop_cluster

# ─── OPTIMIZED (new settings) ─────────────────────────────────
echo ""
echo ">>> Phase 2/2: OPTIMIZED (float16 wire, 16KB compress threshold)"
export UNFED_WIRE_DTYPE="float16"
export UNFED_COMPRESS_THRESHOLD=16384
export UNFED_PREFILL_MIN=64

start_cluster
echo ""
$PYTHON benchmark.py --runs $RUNS --warmup $WARMUP --max-tokens $MAX_TOKENS
stop_cluster

echo ""
echo "============================================================"
echo "  Comparison complete."
echo "============================================================"
