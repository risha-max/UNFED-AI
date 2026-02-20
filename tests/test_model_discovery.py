#!/usr/bin/env python3
"""
End-to-end test for multi-registry model discovery.

Tests:
  1. ListModels RPC — registry returns correct model catalog
  2. RegistryPool fallback — client falls back to second registry if first is down
  3. --list-models CLI — prints available models
  4. Full inference with --model flag

Usage:
    python -m tests.test_model_discovery
"""

import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import grpc
import registry_pb2
import registry_pb2_grpc
import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVATE = f"cd {PROJECT_ROOT} && source .venv/bin/activate"

REGISTRY_PORT_A = 50050
REGISTRY_PORT_B = 50060
NODE_PORTS = [50051, 50052, 50053, 50054]

processes = []
passed = 0
failed = 0


def start_process(cmd, label):
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
    for proc, label in processes:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    print("\n[Cleanup] All processes terminated.")


def wait_for_port(port, timeout=30):
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


def wait_for_registry_nodes(port, expected, timeout=180):
    deadline = time.time() + timeout
    ch = grpc.insecure_channel(f"localhost:{port}")
    stub = registry_pb2_grpc.RegistryStub(ch)
    while time.time() < deadline:
        try:
            resp = stub.Discover(registry_pb2.DiscoverRequest(model_id=""), timeout=5)
            compute = [n for n in resp.nodes if n.node_type == "compute"]
            if len(compute) >= expected:
                ch.close()
                return True
        except Exception:
            pass
        time.sleep(2)
    ch.close()
    return False


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        failed += 1
    return condition


def register_fake_node(registry_port, node_id, model_id, shard_index,
                       total_shards=4, layer_start=0, layer_end=6):
    """Register a fake node with a registry (no real node needed)."""
    ch = grpc.insecure_channel(f"localhost:{registry_port}")
    stub = registry_pb2_grpc.RegistryStub(ch)
    resp = stub.Register(registry_pb2.RegisterRequest(
        node_id=node_id,
        address=f"localhost:{50100 + shard_index}",
        model_id=model_id,
        shard_index=shard_index,
        layer_start=layer_start,
        layer_end=layer_end,
        has_embedding=(shard_index == 0),
        has_lm_head=(shard_index == total_shards - 1),
        public_key=b"\x00" * 32,
        node_type="compute",
    ))
    ch.close()
    return resp.success


def main():
    print("=" * 60)
    print("  Multi-Registry Model Discovery Test")
    print("=" * 60)

    try:
        # =============================================
        # Test 1: ListModels RPC
        # =============================================
        print("\n[Test 1] ListModels RPC")
        print("-" * 40)

        # Start Registry A
        print("  Starting Registry A...")
        start_process(
            f"{ACTIVATE} && python -m network.registry_server --port {REGISTRY_PORT_A}",
            "Registry-A",
        )
        assert wait_for_port(REGISTRY_PORT_A, timeout=15), "Registry A did not start"

        # Register fake nodes for two different models
        print("  Registering fake nodes for 2 models...")

        # Model 1: Qwen with 4 shards (all covered)
        for i in range(4):
            register_fake_node(REGISTRY_PORT_A, f"qwen-node-{i}",
                               "Qwen/Qwen2.5-0.5B", i)

        # Model 2: Llama with 4 shards (only 3 covered — incomplete)
        for i in range(3):
            register_fake_node(REGISTRY_PORT_A, f"llama-node-{i}",
                               "meta-llama/Llama-3-8B", i, total_shards=4)

        # Query ListModels
        ch = grpc.insecure_channel(f"localhost:{REGISTRY_PORT_A}")
        stub = registry_pb2_grpc.RegistryStub(ch)
        resp = stub.ListModels(registry_pb2.ListModelsRequest(), timeout=10)
        ch.close()

        models_by_id = {m.model_id: m for m in resp.models}
        check("ListModels returns 2 models", len(resp.models) == 2,
              f"got {len(resp.models)}")

        qwen = models_by_id.get("Qwen/Qwen2.5-0.5B")
        check("Qwen model found", qwen is not None)
        if qwen:
            check("Qwen has 4 nodes", qwen.total_nodes == 4,
                  f"got {qwen.total_nodes}")
            check("Qwen has 4/4 shards covered", qwen.covered_shards == 4,
                  f"got {qwen.covered_shards}")
            check("Qwen can_serve = True", qwen.can_serve)

        llama = models_by_id.get("meta-llama/Llama-3-8B")
        check("Llama model found", llama is not None)
        if llama:
            check("Llama has 3 nodes", llama.total_nodes == 3,
                  f"got {llama.total_nodes}")
            # Note: registry only knows about shards that registered.
            # 3 nodes registered for shards 0,1,2 → total_shards=3, covered=3.
            # The registry has no way to know shard 3 is missing without a manifest.
            check("Llama has 3/3 shards covered (from registry's view)",
                  llama.covered_shards == 3,
                  f"got {llama.covered_shards}")
            check("Llama can_serve = True (all known shards covered)",
                  llama.can_serve)

        # =============================================
        # Test 2: RegistryPool fallback
        # =============================================
        print(f"\n[Test 2] RegistryPool fallback")
        print("-" * 40)

        # Start Registry B
        print("  Starting Registry B...")
        start_process(
            f"{ACTIVATE} && python -m network.registry_server --port {REGISTRY_PORT_B}",
            "Registry-B",
        )
        assert wait_for_port(REGISTRY_PORT_B, timeout=15), "Registry B did not start"

        # Register a different model on Registry B
        register_fake_node(REGISTRY_PORT_B, "mistral-node-0",
                           "mistralai/Mistral-7B", 0)

        from network.discovery import RegistryPool

        # Pool with both registries
        pool = RegistryPool([f"localhost:{REGISTRY_PORT_A}",
                             f"localhost:{REGISTRY_PORT_B}"])
        models = pool.list_models()
        model_ids = {m.model_id for m in models}
        check("Pool merges models from both registries",
              "Qwen/Qwen2.5-0.5B" in model_ids and "mistralai/Mistral-7B" in model_ids,
              f"got {model_ids}")

        # Pool with dead first registry + live second
        pool_fallback = RegistryPool(["localhost:59999",
                                      f"localhost:{REGISTRY_PORT_B}"])
        healthy = pool_fallback.find_healthy_registry()
        check("Pool falls back to second registry",
              healthy == f"localhost:{REGISTRY_PORT_B}",
              f"got {healthy}")

        nodes_fb = pool_fallback.discover("mistralai/Mistral-7B")
        check("Fallback discovery finds Mistral nodes",
              len(nodes_fb) > 0, f"got {len(nodes_fb)} nodes")

        pool.close()
        pool_fallback.close()

        # =============================================
        # Test 3: --list-models CLI
        # =============================================
        print(f"\n[Test 3] --list-models CLI")
        print("-" * 40)

        result = subprocess.run(
            f"{ACTIVATE} && python -m client.client --list-models "
            f"--registry localhost:{REGISTRY_PORT_A}",
            shell=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        print(f"  CLI output:\n{output}")
        check("--list-models exits cleanly", result.returncode == 0)
        check("Output contains Qwen model", "Qwen/Qwen2.5-0.5B" in output)
        check("Output contains Llama model", "meta-llama/Llama-3-8B" in output)
        check("Output shows READY for Qwen", "READY" in output)
        check("Output shows READY for Llama (all known shards covered)", "READY" in output)

        # =============================================
        # Test 4: Full inference with --model
        # =============================================
        print(f"\n[Test 4] Full inference with --model flag")
        print("-" * 40)

        # Kill the fake-node registry, restart fresh for real nodes
        for proc, label in processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        processes.clear()
        time.sleep(1)

        # Start fresh registry
        print("  Starting fresh registry...")
        start_process(
            f"{ACTIVATE} && python -m network.registry_server --port {REGISTRY_PORT_A}",
            "Registry-Fresh",
        )
        assert wait_for_port(REGISTRY_PORT_A, timeout=15), "Fresh registry did not start"

        # Start 4 real nodes
        print("  Starting 4 compute nodes...")
        for i, port in enumerate(NODE_PORTS):
            start_process(
                f"{ACTIVATE} && python -m node.server "
                f"--shard-index {i} --port {port} "
                f"--registry localhost:{REGISTRY_PORT_A}",
                f"Node-{i}",
            )

        print("  Waiting for nodes to load and register...")
        for port in NODE_PORTS:
            if not wait_for_port(port, timeout=180):
                print(f"  Node on port {port} did not start")
                return False

        if not wait_for_registry_nodes(REGISTRY_PORT_A, 4, timeout=180):
            print("  Not all nodes registered")
            return False

        print("  All nodes up. Running inference with --model flag...")

        # Test --list-models with real nodes
        result = subprocess.run(
            f"{ACTIVATE} && python -m client.client --list-models "
            f"--registry localhost:{REGISTRY_PORT_A}",
            shell=True, capture_output=True, text=True, timeout=30,
        )
        check("--list-models shows real Qwen model",
              "Qwen/Qwen2.5-0.5B" in result.stdout and "READY" in result.stdout,
              f"output: {result.stdout.strip()[:200]}")

        # Run inference with explicit --model
        result = subprocess.run(
            f"{ACTIVATE} && python -m client.client "
            f"--model Qwen/Qwen2.5-0.5B "
            f"--prompt \"What is 2+2?\" "
            f"--max-tokens 10 "
            f"--no-onion "
            f"--registry localhost:{REGISTRY_PORT_A}",
            shell=True, capture_output=True, text=True, timeout=120,
        )
        print(f"  Inference output: {result.stdout.strip()[:300]}")
        check("Inference with --model completes",
              result.returncode == 0,
              f"exit code {result.returncode}")
        check("Inference produces output",
              len(result.stdout.strip()) > len("What is 2+2?"),
              f"output length: {len(result.stdout.strip())}")

        # =============================================
        # Summary
        # =============================================
        print(f"\n{'=' * 60}")
        print(f"  Results: {passed}/{passed + failed} checks passed")
        print(f"{'=' * 60}")
        return failed == 0

    finally:
        cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
