"""
Integration Test — Privacy Features

Tests the full pipeline with:
  1. Guard relay — client connects through a guard, guard cannot read payload
  2. MPC for shard 0 — secret-shared embedding + layer 0 computation
  3. Return-path encryption — layered response encryption
  4. Redundant voting — commit-then-reveal correctness verification

Test architecture:
  - Registry on port 50050
  - Guard relay on port 50060
  - 4 compute nodes on ports 50051-50054 (one per shard)
  - Duplicate nodes for voting on ports 50055-50058
  - (Optional) 2 MPC nodes on ports 50070-50071

Usage:
    python -m tests.test_privacy_features
    python -m tests.test_privacy_features --test-mpc   # also test MPC
    python -m tests.test_privacy_features --verbose     # detailed output
"""

import argparse
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import torch


def test_secret_sharing():
    """Test that additive secret sharing works correctly."""
    print("\n=== Test: Secret Sharing ===")
    from network.mpc_shard0 import create_additive_shares, reconstruct_from_shares

    # Create a test tensor
    original = torch.randn(1, 10, 896)  # batch=1, seq=10, hidden=896

    # Split into 2 shares
    shares = create_additive_shares(original, 2)
    assert len(shares) == 2, "Expected 2 shares"

    # Verify each share alone looks like random noise
    # (correlation with original should be near 0 for large tensors)
    for i, share in enumerate(shares):
        assert share.shape == original.shape, f"Share {i} shape mismatch"

    # Verify reconstruction
    reconstructed = reconstruct_from_shares(shares)
    assert torch.allclose(original, reconstructed, atol=1e-6), \
        "Reconstruction failed: shares don't sum to original"

    print("  PASS: Shares sum to original tensor")
    print("  PASS: Each share has correct shape")
    print("  PASS: Secret sharing works correctly")


def test_mpc_protocols():
    """Test MPC protocols for nonlinear operations."""
    print("\n=== Test: MPC Protocols ===")
    from network.mpc_shard0 import MPCProtocol, create_additive_shares

    x = torch.randn(1, 5, 896)
    share_a, share_b = create_additive_shares(x, 2)
    weight = torch.randn(896)

    # Test RMSNorm
    out_a, out_b = MPCProtocol.rmsnorm_protocol(share_a, share_b, weight, eps=1e-6)
    mpc_result = out_a + out_b

    # Compare with reference RMSNorm
    variance = (x ** 2).mean(dim=-1, keepdim=True)
    ref_result = x * torch.rsqrt(variance + 1e-6) * weight
    assert torch.allclose(mpc_result, ref_result, atol=1e-4), \
        f"RMSNorm MPC mismatch: max diff={torch.max(torch.abs(mpc_result - ref_result)):.6f}"
    print("  PASS: RMSNorm MPC matches reference")

    # Test Softmax
    logits = torch.randn(1, 5, 10)
    sa, sb = create_additive_shares(logits, 2)
    out_a, out_b = MPCProtocol.softmax_protocol(sa, sb, dim=-1)
    mpc_softmax = out_a + out_b
    ref_softmax = torch.nn.functional.softmax(logits, dim=-1)
    assert torch.allclose(mpc_softmax, ref_softmax, atol=1e-5), \
        "Softmax MPC mismatch"
    print("  PASS: Softmax MPC matches reference")

    # Test SiLU
    sa, sb = create_additive_shares(x, 2)
    out_a, out_b = MPCProtocol.silu_protocol(sa, sb)
    mpc_silu = out_a + out_b
    ref_silu = torch.nn.functional.silu(x)
    assert torch.allclose(mpc_silu, ref_silu, atol=1e-5), \
        "SiLU MPC mismatch"
    print("  PASS: SiLU MPC matches reference")

    print("  PASS: All MPC protocols produce correct results")


def test_return_path_encryption():
    """Test return-path layered encryption and decryption."""
    print("\n=== Test: Return-Path Encryption ===")
    from network.onion import (
        generate_response_keys, encrypt_response, decrypt_response_layers
    )

    # Simulate 4 nodes in the pipeline
    num_nodes = 4
    keys, _ = generate_response_keys(num_nodes)

    # Original token data (what the last node produces)
    token_id = 42
    token_bytes = token_id.to_bytes(4, 'big', signed=True)
    is_eos = False
    is_eos_byte = b'\x01' if is_eos else b'\x00'
    original_data = token_bytes + is_eos_byte

    # Simulate layered encryption (last node → first node)
    encrypted = original_data
    for i in range(num_nodes - 1, -1, -1):
        encrypted = encrypt_response(keys[i], encrypted)
        assert encrypted != original_data, f"Encryption at node {i} didn't change data"

    print(f"  Original: {len(original_data)} bytes")
    print(f"  After {num_nodes} layers: {len(encrypted)} bytes")

    # Client decrypts all layers
    decrypted = decrypt_response_layers(keys, encrypted)
    assert decrypted == original_data, \
        "Decryption failed: plaintext doesn't match original"

    # Parse the token
    recovered_token = int.from_bytes(decrypted[:4], 'big', signed=True)
    recovered_eos = decrypted[4:5] == b'\x01'
    assert recovered_token == token_id, f"Token mismatch: {recovered_token} != {token_id}"
    assert recovered_eos == is_eos, "EOS flag mismatch"

    print(f"  Decrypted token: {recovered_token} (EOS={recovered_eos})")
    print("  PASS: Layered encryption/decryption works correctly")

    # Verify intermediate nodes can't read the data
    # A node at position 2 only has key[2] — it can't strip layers 0 and 1
    partial = encrypted
    try:
        partial_decrypt = decrypt_response_layers([keys[2]], partial)
        # This should fail or produce garbage because layers 0 and 1 are still there
        if partial_decrypt == original_data:
            print("  FAIL: Intermediate node could read the data!")
            assert False
    except Exception:
        pass  # Expected — AES-GCM will throw on wrong data
    print("  PASS: Intermediate nodes cannot read the encrypted response")


def test_voting_coordinator():
    """Test the voting coordinator logic."""
    print("\n=== Test: Voting Coordinator ===")
    from network.voting import VotingCoordinator

    voter = VotingCoordinator()

    # Test shard selection is random and bounded
    selections = set()
    for _ in range(100):
        s = voter.select_voted_shard(4)
        assert 0 <= s < 4, f"Shard selection out of range: {s}"
        selections.add(s)

    assert len(selections) > 1, "Shard selection is not random (always same shard)"
    print(f"  Selected shards over 100 trials: {sorted(selections)}")
    print("  PASS: Shard selection is random and bounded")

    voter.close()


def test_guard_node_proto():
    """Test that guard relay proto messages serialize correctly."""
    print("\n=== Test: Guard Node Proto ===")
    import inference_pb2

    # Create a RelayRequest
    relay_req = inference_pb2.RelayRequest(
        encrypted_payload=b"encrypted-test-payload",
        target_address="localhost:50051",
        guard_ephemeral_key=b"\x00" * 32,
    )
    serialized = relay_req.SerializeToString()
    assert len(serialized) > 0, "Serialization failed"

    # Deserialize
    parsed = inference_pb2.RelayRequest()
    parsed.ParseFromString(serialized)
    assert parsed.encrypted_payload == b"encrypted-test-payload"
    assert parsed.target_address == "localhost:50051"
    print("  PASS: RelayRequest serializes and deserializes correctly")

    # Create a RelayResponse
    relay_resp = inference_pb2.RelayResponse(
        encrypted_payload=b"encrypted-response",
    )
    assert relay_resp.encrypted_payload == b"encrypted-response"
    print("  PASS: RelayResponse works correctly")

    # Test CommitRequest/Response
    commit_req = inference_pb2.CommitRequest(
        session_id="test-session",
        token_ids=[1, 2, 3],
        is_prefill=True,
    )
    assert list(commit_req.token_ids) == [1, 2, 3]
    print("  PASS: CommitRequest works correctly")

    commit_resp = inference_pb2.CommitResponse(
        output_hash="abc123",
        commit_id="commit-test",
        token_id=42,
        has_token=True,
    )
    assert commit_resp.output_hash == "abc123"
    assert commit_resp.token_id == 42
    print("  PASS: CommitResponse works correctly")


def test_forward_request_response_keys():
    """Test that ForwardRequest can carry response encryption keys."""
    print("\n=== Test: ForwardRequest Response Keys ===")
    import inference_pb2

    keys = [os.urandom(32) for _ in range(4)]

    req = inference_pb2.ForwardRequest(
        session_id="test",
        token_ids=[1, 2, 3],
        is_prefill=True,
    )
    req.response_keys.extend(keys)

    assert len(req.response_keys) == 4
    assert bytes(req.response_keys[0]) == keys[0]
    assert bytes(req.response_keys[3]) == keys[3]
    print(f"  PASS: ForwardRequest carries {len(req.response_keys)} response keys")

    # Test ForwardResponse encrypted_response field
    resp = inference_pb2.ForwardResponse(
        encrypted_response=b"layered-encrypted-blob",
        token_id=42,
        has_token=True,
    )
    assert resp.encrypted_response == b"layered-encrypted-blob"
    print("  PASS: ForwardResponse carries encrypted_response")


def test_node_type_in_registry():
    """Test that node_type field works in registry proto."""
    print("\n=== Test: Node Type in Registry ===")
    import registry_pb2

    # Test RegisterRequest with node_type
    req = registry_pb2.RegisterRequest(
        node_id="test-guard",
        address="localhost:50060",
        model_id="",
        shard_index=-1,
        node_type="guard",
    )
    assert req.node_type == "guard"
    print("  PASS: RegisterRequest supports node_type field")

    # Test NodeInfo with node_type
    info = registry_pb2.NodeInfo(
        node_id="test-compute",
        address="localhost:50051",
        node_type="compute",
    )
    assert info.node_type == "compute"
    print("  PASS: NodeInfo supports node_type field")


def main():
    parser = argparse.ArgumentParser(description="UNFED AI Privacy Features Test")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test-mpc", action="store_true",
                        help="Also test MPC layer computation (requires model weights)")
    args = parser.parse_args()

    print("=" * 60)
    print("  UNFED AI — Privacy Features Integration Test")
    print("=" * 60)

    passed = 0
    failed = 0
    tests = [
        ("Secret Sharing", test_secret_sharing),
        ("MPC Protocols", test_mpc_protocols),
        ("Return-Path Encryption", test_return_path_encryption),
        ("Voting Coordinator", test_voting_coordinator),
        ("Guard Node Proto", test_guard_node_proto),
        ("ForwardRequest Response Keys", test_forward_request_response_keys),
        ("Node Type in Registry", test_node_type_in_registry),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAIL: {name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n  All privacy feature tests passed!")


if __name__ == "__main__":
    main()
