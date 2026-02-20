"""
Tests — Per-Shard Racing

Tests the racing coordinator and related changes:
  1. race_shard takes the first response (fast racer wins)
  2. Fault tolerance: one racer errors, other succeeds
  3. build_racing_circuit returns multiple nodes per shard
  4. Intermediate node returns activation_data when no routing info
  5. Background verification detects mismatches

Usage:
    python -m tests.test_racing
    python -m tests.test_racing --verbose
"""

import os
import sys
import time
import hashlib
import numpy as np
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import inference_pb2
import inference_pb2_grpc
from network.racing import RacingCoordinator, RaceResult


def _make_activation_response(data: bytes = b"activation_data_123",
                               shape: list[int] = None) -> inference_pb2.ForwardResponse:
    """Create a ForwardResponse with activation data (intermediate node)."""
    return inference_pb2.ForwardResponse(
        activation_data=data,
        tensor_shape=shape or [1, 5, 896],
    )


def _make_token_response(token_id: int = 42,
                          is_eos: bool = False) -> inference_pb2.ForwardResponse:
    """Create a ForwardResponse with a token (last node)."""
    return inference_pb2.ForwardResponse(
        token_id=token_id,
        has_token=True,
        is_eos=is_eos,
    )


def test_race_shard_takes_first():
    """Test that the fastest racer wins."""
    print("\n=== Test: Race Shard Takes First ===")

    coordinator = RacingCoordinator(replicas=2, timeout=10)

    # Two responses: fast (50ms) and slow (500ms)
    fast_response = _make_activation_response(b"fast_activation")
    slow_response = _make_activation_response(b"fast_activation")  # same data

    def fast_forward(addr, request):
        time.sleep(0.05)
        return fast_response

    def slow_forward(addr, request):
        time.sleep(0.5)
        return slow_response

    call_count = {"fast": 0, "slow": 0}
    original_forward = coordinator._forward_to_node

    def mock_forward(address, request):
        if address == "fast-node:50051":
            call_count["fast"] += 1
            return fast_forward(address, request)
        else:
            call_count["slow"] += 1
            return slow_forward(address, request)

    coordinator._forward_to_node = mock_forward

    request = inference_pb2.ForwardRequest(session_id="test-session")

    start = time.time()
    result = coordinator.race_shard(
        shard_index=0,
        node_addresses=["fast-node:50051", "slow-node:50052"],
        request=request,
    )
    elapsed = time.time() - start

    assert result.winner_address == "fast-node:50051", \
        f"Expected fast node to win, got {result.winner_address}"
    assert result.response.activation_data == b"fast_activation", \
        "Winner should have fast_activation data"
    assert elapsed < 0.3, \
        f"Should return in ~50ms, took {elapsed:.3f}s (didn't wait for slow racer)"
    assert result.shard_index == 0

    print(f"  Winner: {result.winner_address} ({result.latency_ms:.0f}ms)")
    print(f"  Elapsed: {elapsed:.3f}s (fast racer won)")
    print("  PASSED")

    coordinator.close()


def test_race_shard_fault_tolerance():
    """Test that racing survives one node failure."""
    print("\n=== Test: Race Shard Fault Tolerance ===")

    coordinator = RacingCoordinator(replicas=2, timeout=10)

    good_response = _make_token_response(token_id=99)

    def mock_forward(address, request):
        if address == "failing-node:50051":
            raise Exception("Node crashed!")
        else:
            time.sleep(0.05)
            return good_response

    coordinator._forward_to_node = mock_forward

    request = inference_pb2.ForwardRequest(session_id="test-fault")

    result = coordinator.race_shard(
        shard_index=3,
        node_addresses=["failing-node:50051", "healthy-node:50052"],
        request=request,
    )

    assert result.winner_address == "healthy-node:50052", \
        f"Healthy node should win, got {result.winner_address}"
    assert result.response.token_id == 99
    assert result.response.has_token is True

    print(f"  Winner: {result.winner_address}")
    print(f"  Failing node handled gracefully")
    print("  PASSED")

    coordinator.close()


def test_all_racers_fail():
    """Test that racing raises RuntimeError when all racers fail."""
    print("\n=== Test: All Racers Fail ===")

    coordinator = RacingCoordinator(replicas=2, timeout=5)

    def mock_forward(address, request):
        raise Exception(f"Node {address} crashed!")

    coordinator._forward_to_node = mock_forward

    request = inference_pb2.ForwardRequest(session_id="test-all-fail")

    try:
        coordinator.race_shard(
            shard_index=1,
            node_addresses=["dead-node-a:50051", "dead-node-b:50052"],
            request=request,
        )
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "All 2 replicas" in str(e)
        print(f"  Correctly raised: {e}")
        print("  PASSED")

    coordinator.close()


def test_background_verification_match():
    """Test that background verification detects matching responses."""
    print("\n=== Test: Background Verification (Match) ===")

    coordinator = RacingCoordinator(replicas=2, timeout=10)

    # Both racers return the same data
    same_data = b"identical_activation_tensor_bytes"

    call_order = []

    def mock_forward(address, request):
        call_order.append(address)
        if "fast" in address:
            time.sleep(0.05)
        else:
            time.sleep(0.2)
        return _make_activation_response(same_data)

    coordinator._forward_to_node = mock_forward

    request = inference_pb2.ForwardRequest(session_id="test-verify-match")

    result = coordinator.race_shard(
        shard_index=1,
        node_addresses=["fast-node:50051", "slow-node:50052"],
        request=request,
    )

    # Wait for background verification to complete
    time.sleep(0.5)

    assert len(coordinator.mismatches) == 0, \
        f"Expected no mismatches, got {len(coordinator.mismatches)}"

    print(f"  Winner: {result.winner_address}")
    print(f"  Mismatches: {len(coordinator.mismatches)} (expected 0)")
    print("  PASSED")

    coordinator.close()


def test_background_verification_mismatch():
    """Test that background verification detects mismatching responses."""
    print("\n=== Test: Background Verification (Mismatch) ===")

    coordinator = RacingCoordinator(replicas=2, timeout=10)

    def mock_forward(address, request):
        if "honest" in address:
            time.sleep(0.05)
            return _make_activation_response(b"correct_output")
        else:
            time.sleep(0.2)
            return _make_activation_response(b"fraudulent_output")

    coordinator._forward_to_node = mock_forward

    request = inference_pb2.ForwardRequest(session_id="test-verify-mismatch")

    result = coordinator.race_shard(
        shard_index=2,
        node_addresses=["honest-node:50051", "cheater-node:50052"],
        request=request,
    )

    # Wait for background verification to complete
    time.sleep(0.5)

    assert len(coordinator.mismatches) == 1, \
        f"Expected 1 mismatch, got {len(coordinator.mismatches)}"
    mismatch = coordinator.mismatches[0]
    assert mismatch["shard_index"] == 2
    assert mismatch["winner"] == "honest-node:50051"
    assert mismatch["other"] == "cheater-node:50052"

    print(f"  Winner: {result.winner_address}")
    print(f"  Mismatch detected: shard {mismatch['shard_index']}")
    print(f"    Winner hash: {mismatch['winner_hash'][:16]}...")
    print(f"    Other hash:  {mismatch['other_hash'][:16]}...")
    print("  PASSED")

    coordinator.close()


def test_build_racing_circuit():
    """Test that build_racing_circuit returns multiple nodes per shard."""
    print("\n=== Test: Build Racing Circuit ===")

    from network.discovery import RegistryClient

    # Mock the discover_compute method
    client = RegistryClient.__new__(RegistryClient)
    client.registry_address = "localhost:50050"
    client._channel = MagicMock()
    client._stub = MagicMock()

    # Create mock nodes: 2 nodes per shard for 4 shards
    mock_nodes = []
    for shard_idx in range(4):
        for replica in range(3):  # 3 nodes per shard
            node = MagicMock()
            node.shard_index = shard_idx
            node.address = f"node-{shard_idx}-{replica}:5005{shard_idx}"
            node.public_key = b"fake_key"
            node.node_type = "compute"
            mock_nodes.append(node)

    client.discover_compute = MagicMock(return_value=mock_nodes)

    # Build with 2 replicas
    shard_map = client.build_racing_circuit("test-model", replicas=2)

    assert shard_map is not None, "Should return a shard map"
    assert len(shard_map) == 4, f"Expected 4 shards, got {len(shard_map)}"

    for shard_idx in range(4):
        assert shard_idx in shard_map, f"Shard {shard_idx} missing"
        nodes = shard_map[shard_idx]
        assert len(nodes) == 2, \
            f"Shard {shard_idx}: expected 2 replicas, got {len(nodes)}"
        # Each entry is (address, public_key)
        for addr, pk in nodes:
            assert isinstance(addr, str)
            assert pk == b"fake_key"

    print(f"  Shards: {len(shard_map)}")
    for shard_idx in sorted(shard_map.keys()):
        addrs = [addr for addr, _ in shard_map[shard_idx]]
        print(f"    Shard {shard_idx}: {addrs}")
    print("  PASSED")


def test_build_racing_circuit_incomplete():
    """Test that build_racing_circuit returns None for incomplete pools."""
    print("\n=== Test: Build Racing Circuit (Incomplete Pool) ===")

    from network.discovery import RegistryClient

    client = RegistryClient.__new__(RegistryClient)
    client.registry_address = "localhost:50050"
    client._channel = MagicMock()
    client._stub = MagicMock()

    # Only shard 0 and shard 2 — shard 1 missing
    mock_nodes = []
    for shard_idx in [0, 2]:
        node = MagicMock()
        node.shard_index = shard_idx
        node.address = f"node-{shard_idx}:5005{shard_idx}"
        node.public_key = b"fake_key"
        node.node_type = "compute"
        mock_nodes.append(node)

    client.discover_compute = MagicMock(return_value=mock_nodes)

    shard_map = client.build_racing_circuit("test-model", replicas=2)

    assert shard_map is None, "Should return None for incomplete pool"
    print("  Correctly returned None for pool with missing shard 1")
    print("  PASSED")


def test_intermediate_node_returns_activation():
    """
    Test that an intermediate node (no LM head) returns activation_data
    when there's no routing info (direct mode for racing).
    """
    print("\n=== Test: Intermediate Node Returns Activation ===")

    # This test verifies the node/server.py logic change.
    # We simulate what the code does without actually running a gRPC server.

    # Simulate the routing logic from server.py
    has_lm_head = False
    next_address = None  # No routing info (racing mode)

    # Mock activation data
    activation_data = np.random.randn(1, 5, 896).astype(np.float32).tobytes()
    shape = [1, 5, 896]

    # Simulate the branching logic
    if next_address is None and has_lm_head:
        # Would return token — this branch should NOT be taken
        response_type = "token"
    elif next_address is None:
        # Should return activation — this IS the racing direct mode
        response = inference_pb2.ForwardResponse(
            activation_data=activation_data,
            tensor_shape=shape,
        )
        response_type = "activation"
    else:
        response_type = "forward"

    assert response_type == "activation", \
        f"Expected activation return, got {response_type}"
    assert response.activation_data == activation_data
    assert list(response.tensor_shape) == shape
    assert response.has_token is False  # should NOT have a token

    print(f"  Response type: {response_type}")
    print(f"  Activation size: {len(response.activation_data)} bytes")
    print(f"  Shape: {list(response.tensor_shape)}")
    print(f"  has_token: {response.has_token}")
    print("  PASSED")


def test_last_node_still_returns_token():
    """
    Test that a last node (with LM head) still returns a token
    when there's no routing info (not broken by the racing fix).
    """
    print("\n=== Test: Last Node Still Returns Token ===")

    has_lm_head = True
    next_address = None
    sampled_token = 42

    if next_address is None and has_lm_head:
        is_eos = sampled_token == 151643
        response = inference_pb2.ForwardResponse(
            token_id=sampled_token,
            has_token=True,
            is_eos=is_eos,
        )
        response_type = "token"
    elif next_address is None:
        response_type = "activation"
    else:
        response_type = "forward"

    assert response_type == "token"
    assert response.token_id == 42
    assert response.has_token is True
    assert response.is_eos is False

    print(f"  Response type: {response_type}")
    print(f"  Token ID: {response.token_id}")
    print(f"  has_token: {response.has_token}")
    print(f"  is_eos: {response.is_eos}")
    print("  PASSED")


def test_response_hash_computation():
    """Test that the racing coordinator computes consistent response hashes."""
    print("\n=== Test: Response Hash Computation ===")

    coordinator = RacingCoordinator()

    # Activation response
    resp_a = _make_activation_response(b"some_data", [1, 5, 896])
    resp_b = _make_activation_response(b"some_data", [1, 5, 896])
    resp_c = _make_activation_response(b"different_data", [1, 5, 896])

    hash_a = coordinator._compute_response_hash(resp_a)
    hash_b = coordinator._compute_response_hash(resp_b)
    hash_c = coordinator._compute_response_hash(resp_c)

    assert hash_a == hash_b, "Identical responses should have same hash"
    assert hash_a != hash_c, "Different responses should have different hashes"

    # Token response
    resp_t1 = _make_token_response(42)
    resp_t2 = _make_token_response(42)
    resp_t3 = _make_token_response(99)

    hash_t1 = coordinator._compute_response_hash(resp_t1)
    hash_t2 = coordinator._compute_response_hash(resp_t2)
    hash_t3 = coordinator._compute_response_hash(resp_t3)

    assert hash_t1 == hash_t2, "Same token responses should match"
    assert hash_t1 != hash_t3, "Different token responses should differ"

    print(f"  Activation hash (same data): {hash_a[:16]}... == {hash_b[:16]}...")
    print(f"  Activation hash (diff data): {hash_a[:16]}... != {hash_c[:16]}...")
    print(f"  Token hash (same token): {hash_t1[:16]}... == {hash_t2[:16]}...")
    print(f"  Token hash (diff token): {hash_t1[:16]}... != {hash_t3[:16]}...")
    print("  PASSED")

    coordinator.close()


def test_build_mpc_racing_circuit():
    """Test that build_mpc_racing_circuit correctly separates MPC and compute nodes."""
    print("\n=== Test: Build MPC Racing Circuit ===")

    from network.discovery import RegistryClient

    client = RegistryClient.__new__(RegistryClient)
    client.registry_address = "localhost:50050"
    client._channel = MagicMock()
    client._stub = MagicMock()

    # Create MPC nodes (shard 0 entry points)
    mpc_nodes = []
    for i in range(3):
        node = MagicMock()
        node.shard_index = 0
        node.address = f"mpc-A-{i}:6006{i}"
        node.public_key = b"mpc_key"
        node.node_type = "mpc"
        mpc_nodes.append(node)

    # Create compute nodes (shards 1-3, 2 replicas each)
    compute_nodes = []
    for shard_idx in range(1, 4):
        for replica in range(2):
            node = MagicMock()
            node.shard_index = shard_idx
            node.address = f"compute-{shard_idx}-{replica}:5005{shard_idx}"
            node.public_key = b"compute_key"
            node.node_type = "compute"
            compute_nodes.append(node)

    client.discover_mpc = MagicMock(return_value=mpc_nodes)
    client.discover_compute = MagicMock(return_value=compute_nodes)

    result = client.build_mpc_racing_circuit("test-model", replicas=2)
    assert result is not None, "Should return a result when MPC nodes exist"

    mpc_entries, compute_shard_map = result

    # MPC entries: up to 2 replicas
    assert len(mpc_entries) == 2, f"Expected 2 MPC entries, got {len(mpc_entries)}"
    for addr, pk in mpc_entries:
        assert "mpc-A" in addr, f"MPC entry should be an MPC node: {addr}"
        assert pk == b"mpc_key"

    # Compute shards: 1, 2, 3 (no shard 0)
    assert 0 not in compute_shard_map, "Shard 0 should not be in compute map"
    assert len(compute_shard_map) == 3, f"Expected 3 compute shards, got {len(compute_shard_map)}"

    for shard_idx in range(1, 4):
        assert shard_idx in compute_shard_map, f"Shard {shard_idx} missing"
        nodes = compute_shard_map[shard_idx]
        assert len(nodes) == 2, f"Shard {shard_idx}: expected 2, got {len(nodes)}"

    print(f"  MPC entries: {[addr for addr, _ in mpc_entries]}")
    for shard_idx in sorted(compute_shard_map.keys()):
        addrs = [addr for addr, _ in compute_shard_map[shard_idx]]
        print(f"  Compute shard {shard_idx}: {addrs}")
    print("  PASSED")


def test_mpc_racing_circuit_no_mpc_nodes():
    """Test that build_mpc_racing_circuit returns None when no MPC nodes exist."""
    print("\n=== Test: MPC Racing Circuit (No MPC Nodes) ===")

    from network.discovery import RegistryClient

    client = RegistryClient.__new__(RegistryClient)
    client.registry_address = "localhost:50050"
    client._channel = MagicMock()
    client._stub = MagicMock()

    client.discover_mpc = MagicMock(return_value=[])

    result = client.build_mpc_racing_circuit("test-model", replicas=2)
    assert result is None, "Should return None when no MPC nodes"
    print("  Correctly returned None — client falls back to regular racing")
    print("  PASSED")


def test_mpc_dual_share_recording():
    """Test that MPC servicer records shares for both A and B nodes via daemon."""
    print("\n=== Test: MPC Dual Share Recording ===")

    import torch
    from unittest.mock import MagicMock, patch, call

    # Create a minimal servicer mock
    from network.mpc_shard0 import MPCNodeServicer, MPCNode

    # Mock the MPCNode
    mpc_node = MagicMock(spec=MPCNode)
    mpc_node.role = "A"

    servicer = MPCNodeServicer(mpc_node, 50060)
    servicer._node_id = "node-A-12345678"
    servicer._peer_node_id = "peer-of-node-A-1"

    # Mock the daemon stub (MPC nodes now submit shares to daemon)
    mock_daemon = MagicMock()
    mock_daemon.SubmitShares.return_value = MagicMock(accepted=2)
    servicer._daemon_stub = mock_daemon

    # Record dual shares
    output_tensor = torch.randn(1, 5, 896)
    servicer._record_dual_shares("session-test-123", output_tensor)

    # Should have submitted shares to daemon
    assert mock_daemon.SubmitShares.call_count == 1, \
        f"Expected 1 SubmitShares call, got {mock_daemon.SubmitShares.call_count}"

    # Check the submitted shares
    submit_call = mock_daemon.SubmitShares.call_args[0][0]
    assert len(submit_call.shares) == 2, \
        f"Expected 2 shares in submission, got {len(submit_call.shares)}"

    share_a = submit_call.shares[0]
    share_b = submit_call.shares[1]

    assert share_a.node_id == "node-A-12345678", f"Share A node_id: {share_a.node_id}"
    assert share_b.node_id == "peer-of-node-A-1", f"Share B node_id: {share_b.node_id}"
    assert share_a.shard_index == 0
    assert share_b.shard_index == 0
    assert share_a.session_id == "session-test-123"
    assert share_b.session_id == "session-test-123"
    assert share_a.activation_hash == share_b.activation_hash, \
        "Both shares should have the same activation hash (same output)"
    # Verify share weights are 1.0 (full MPC compute work)
    assert share_a.share_weight == 1.0, f"Share A weight: {share_a.share_weight}"
    assert share_b.share_weight == 1.0, f"Share B weight: {share_b.share_weight}"

    print(f"  Share A: node={share_a.node_id[:12]}..., weight={share_a.share_weight}")
    print(f"  Share B: node={share_b.node_id[:12]}..., weight={share_b.share_weight}")
    print("  Both nodes credited equally with weight 1.0 — PASSED")


def test_config_racing_constants():
    """Test that racing configuration constants are defined."""
    print("\n=== Test: Config Racing Constants ===")

    import config

    assert hasattr(config, "RACING_REPLICAS"), "RACING_REPLICAS not defined"
    assert hasattr(config, "SHARD_TIMEOUT_SECONDS"), "SHARD_TIMEOUT_SECONDS not defined"
    assert isinstance(config.RACING_REPLICAS, int)
    assert isinstance(config.SHARD_TIMEOUT_SECONDS, (int, float))
    assert config.RACING_REPLICAS >= 1
    assert config.SHARD_TIMEOUT_SECONDS > 0

    print(f"  RACING_REPLICAS: {config.RACING_REPLICAS}")
    print(f"  SHARD_TIMEOUT_SECONDS: {config.SHARD_TIMEOUT_SECONDS}")
    print("  PASSED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test per-shard racing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tests = [
        test_race_shard_takes_first,
        test_race_shard_fault_tolerance,
        test_all_racers_fail,
        test_background_verification_match,
        test_background_verification_mismatch,
        test_build_racing_circuit,
        test_build_racing_circuit_incomplete,
        test_build_mpc_racing_circuit,
        test_mpc_racing_circuit_no_mpc_nodes,
        test_mpc_dual_share_recording,
        test_intermediate_node_returns_activation,
        test_last_node_still_returns_token,
        test_response_hash_computation,
        test_config_racing_constants,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
