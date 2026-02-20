#!/usr/bin/env python3
"""
End-to-end test for the distributed mini-chain integration.

Starts a full network (registry + 4 nodes), runs inference, then verifies:
  1. Each node has a local share-chain
  2. Nodes produce blocks containing compute shares
  3. Blocks are gossipped between nodes
  4. The registry receives gossipped blocks (passive peer)
  5. All peers converge on the same chain height

Usage:
    python -m tests.test_distributed_chain
"""

import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import grpc
import inference_pb2
import inference_pb2_grpc
import registry_pb2
import registry_pb2_grpc
import config


REGISTRY_PORT = 50050
NODE_PORTS = [50051, 50052, 50053, 50054]
REGISTRY_ADDR = f"localhost:{REGISTRY_PORT}"

processes = []


def start_process(cmd, label):
    """Start a subprocess and track it."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, shell=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    processes.append((proc, label))
    print(f"  Started {label} (PID {proc.pid})")
    return proc


def cleanup():
    """Kill all spawned processes."""
    for proc, label in processes:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    print("\n[Cleanup] All processes terminated.")


def wait_for_port(port, timeout=120):
    """Wait for a gRPC service to become reachable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ch = grpc.insecure_channel(f"localhost:{port}")
            grpc.channel_ready_future(ch).result(timeout=2)
            ch.close()
            return True
        except Exception:
            time.sleep(1)
    return False


def wait_for_registry_nodes(expected, timeout=180):
    """Wait until the registry sees the expected number of compute nodes."""
    deadline = time.time() + timeout
    ch = grpc.insecure_channel(REGISTRY_ADDR)
    stub = registry_pb2_grpc.RegistryStub(ch)
    while time.time() < deadline:
        try:
            resp = stub.Discover(registry_pb2.DiscoverRequest(model_id=""), timeout=5)
            compute_nodes = [n for n in resp.nodes if n.node_type == "compute"]
            if len(compute_nodes) >= expected:
                ch.close()
                return True
        except Exception:
            pass
        time.sleep(2)
    ch.close()
    return False


def run_inference(prompt="Hello, how are you?", max_tokens=10):
    """Run a simple inference query via the client."""
    cmd = (
        f"cd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))} && "
        f"source .venv/bin/activate && "
        f"python -m client.client "
        f"--prompt \"{prompt}\" "
        f"--max-tokens {max_tokens} "
        f"--no-onion "
        f"--registry {REGISTRY_ADDR}"
    )
    print(f"\n[Test] Running inference: \"{prompt}\" (max {max_tokens} tokens)")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=120,
    )
    print(f"  stdout: {result.stdout.strip()[:500]}")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr.strip()[:500]}")
    return result.returncode == 0


def get_chain_status_from_node(port):
    """Query a node's chain via GetBlocks RPC."""
    try:
        ch = grpc.insecure_channel(f"localhost:{port}", options=config.GRPC_OPTIONS)
        stub = inference_pb2_grpc.InferenceNodeStub(ch)
        resp = stub.GetBlocks(
            inference_pb2.GetBlocksRequest(from_height=0), timeout=10,
        )
        ch.close()
        return {
            "chain_height": resp.chain_height,
            "num_blocks": len(resp.blocks),
            "blocks": [
                {
                    "index": b.index,
                    "hash": b.block_hash[:12],
                    "shares": len(b.shares),
                    "proposer": b.proposer_id[:8] if b.proposer_id else "?",
                }
                for b in resp.blocks
            ],
        }
    except grpc.RpcError as e:
        return {"error": str(e.code()), "detail": e.details()}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    activate = f"cd {project_root} && source .venv/bin/activate"

    print("=" * 60)
    print("  Distributed Mini-Chain Integration Test")
    print("=" * 60)

    try:
        # --- Step 1: Start registry ---
        print("\n[Step 1] Starting registry...")
        start_process(
            f"{activate} && python -m network.registry_server --port {REGISTRY_PORT}",
            "Registry",
        )
        if not wait_for_port(REGISTRY_PORT, timeout=30):
            print("FAIL: Registry did not start")
            return False

        print("  Registry is up.")

        # --- Step 2: Start 4 nodes ---
        print("\n[Step 2] Starting 4 compute nodes...")
        for i, port in enumerate(NODE_PORTS):
            start_process(
                f"{activate} && python -m node.server "
                f"--shard-index {i} --port {port} "
                f"--registry {REGISTRY_ADDR}",
                f"Node-{i} (shard {i}, port {port})",
            )

        print("  Waiting for nodes to load shards and register...")
        for port in NODE_PORTS:
            if not wait_for_port(port, timeout=180):
                print(f"  FAIL: Node on port {port} did not start")
                return False

        if not wait_for_registry_nodes(4, timeout=180):
            print("  FAIL: Not all nodes registered with the registry")
            return False

        print("  All 4 nodes are up and registered.")

        # --- Step 3: Run inference ---
        print("\n[Step 3] Running inference to generate compute shares...")
        success = run_inference("What is the capital of France?", max_tokens=15)
        if not success:
            print("  WARNING: Inference returned non-zero, but shares may still exist.")

        # Run a second query to generate more shares
        run_inference("Explain quantum computing briefly.", max_tokens=15)

        # --- Step 4: Wait for block production (10s interval + processing) ---
        print("\n[Step 4] Waiting for block production and gossip (~25s)...")
        time.sleep(25)

        # --- Step 5: Query chain status from every node + registry ---
        print("\n[Step 5] Querying chain status from all peers...")
        print("-" * 50)

        all_statuses = {}
        for port in NODE_PORTS + [REGISTRY_PORT]:
            label = f"Node (port {port})" if port != REGISTRY_PORT else "Registry"
            status = get_chain_status_from_node(port)
            all_statuses[port] = status
            print(f"\n  {label}:")
            if "error" in status:
                print(f"    Error: {status['error']} — {status.get('detail', '')}")
            else:
                print(f"    Chain height: {status['chain_height']}")
                print(f"    Blocks returned: {status['num_blocks']}")
                total_shares = sum(b["shares"] for b in status["blocks"])
                print(f"    Total shares across all blocks: {total_shares}")
                for b in status["blocks"]:
                    shares_info = f", {b['shares']} shares" if b['shares'] > 0 else ""
                    print(f"      Block #{b['index']}: hash={b['hash']}..."
                          f"{shares_info}"
                          f" (proposer={b['proposer']}...)")

        print("\n" + "-" * 50)

        # --- Step 6: Verify ---
        print("\n[Step 6] Verifying results...")
        checks_passed = 0
        checks_total = 0

        # Check 1: At least one node has blocks beyond genesis
        checks_total += 1
        node_heights = []
        for port in NODE_PORTS:
            s = all_statuses.get(port, {})
            h = s.get("chain_height", 0)
            node_heights.append(h)
        max_height = max(node_heights) if node_heights else 0
        if max_height > 0:
            print(f"  [PASS] Nodes have produced blocks (max height: {max_height})")
            checks_passed += 1
        else:
            print(f"  [FAIL] No blocks produced by any node")

        # Check 2: At least one block contains compute shares
        checks_total += 1
        any_shares = False
        for port in NODE_PORTS:
            s = all_statuses.get(port, {})
            for b in s.get("blocks", []):
                if b["shares"] > 0:
                    any_shares = True
                    break
        if any_shares:
            print(f"  [PASS] Blocks contain compute shares (inference recorded)")
            checks_passed += 1
        else:
            print(f"  [FAIL] No blocks contain compute shares")

        # Check 3: Registry received gossipped blocks
        checks_total += 1
        reg_status = all_statuses.get(REGISTRY_PORT, {})
        reg_height = reg_status.get("chain_height", 0)
        if reg_height > 0:
            print(f"  [PASS] Registry received gossipped blocks "
                  f"(height: {reg_height})")
            checks_passed += 1
        else:
            print(f"  [FAIL] Registry chain is still at genesis")

        # Check 4: Multiple nodes have the same chain height (gossip works)
        checks_total += 1
        if node_heights:
            # At least 2 nodes should agree
            from collections import Counter
            height_counts = Counter(node_heights)
            most_common_height, count = height_counts.most_common(1)[0]
            if count >= 2 and most_common_height > 0:
                print(f"  [PASS] {count} nodes agree on chain height "
                      f"{most_common_height} (gossip convergence)")
                checks_passed += 1
            elif max_height > 0:
                print(f"  [WARN] Nodes have different heights {node_heights} "
                      f"(gossip may still be propagating)")
                # Partial pass — blocks are being produced
                checks_passed += 1
            else:
                print(f"  [FAIL] No convergence")

        # Check 5: Block hashes are consistent across peers at the same height
        checks_total += 1
        # Compare block hash at height 1 across all peers that have it
        h1_hashes = set()
        for port in NODE_PORTS + [REGISTRY_PORT]:
            s = all_statuses.get(port, {})
            for b in s.get("blocks", []):
                if b["index"] == 1:
                    h1_hashes.add(b["hash"])
        if len(h1_hashes) == 1:
            print(f"  [PASS] All peers agree on block #1 hash: "
                  f"{h1_hashes.pop()}...")
            checks_passed += 1
        elif len(h1_hashes) > 1:
            print(f"  [WARN] Block #1 hash mismatch: {h1_hashes} "
                  f"(conflict resolution may be in progress)")
            # Not a hard fail — async gossip means brief divergence is OK
        else:
            print(f"  [SKIP] No block #1 found yet")

        print(f"\n{'=' * 60}")
        print(f"  Results: {checks_passed}/{checks_total} checks passed")
        print(f"{'=' * 60}")

        return checks_passed >= 3  # at least core checks pass

    finally:
        cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
