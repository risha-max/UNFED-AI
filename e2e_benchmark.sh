#!/usr/bin/env bash
# E2E benchmark: starts registry + 4 nodes, runs benchmark, then cleans up.
set -euo pipefail

export PYTHONUNBUFFERED=1
PYTHON="${UNFED_PYTHON:-.venv/bin/python}"
PIDS=()

cleanup() {
    echo ""
    echo "[E2E] Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[E2E] Done."
}
trap cleanup EXIT

echo "============================================================"
echo "  UNFED AI — End-to-End Latency Benchmark"
echo "============================================================"
echo ""

# 0. Kill any leftover processes from previous runs
pkill -f "registry_server.*50050" 2>/dev/null || true
pkill -f "node/server.py" 2>/dev/null || true
sleep 1

# 1. Start registry
echo "[E2E] Starting registry on port 50050..."
$PYTHON -m network.registry_server --port 50050 &
PIDS+=($!)
sleep 2

# 2. Start 4 compute nodes (shards 0-3)
for i in 0 1 2 3; do
    PORT=$((50051 + i))
    echo "[E2E] Starting shard $i on port $PORT..."
    $PYTHON node/server.py --shard-index $i --port $PORT --shards-dir shards &
    PIDS+=($!)
done

echo "[E2E] Waiting for nodes to load weights and register..."
sleep 15

# 3. Run benchmark
echo ""
echo "============================================================"
echo "  Running Benchmark"
echo "============================================================"
echo ""

$PYTHON benchmark.py --runs 5 --warmup 1 --max-tokens 30

echo ""
echo "[E2E] Benchmark complete."
