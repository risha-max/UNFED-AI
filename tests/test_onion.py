"""
Unit tests for onion routing — layered encryption round-trip.

Verifies:
  - Encrypt N layers, peel N layers, recover original payload
  - Each node learns only its next hop
  - Return-path encryption works correctly
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.onion import (
    generate_keypair,
    public_key_to_bytes,
    public_key_from_bytes,
    private_key_to_bytes,
    private_key_from_bytes,
    derive_shared_key,
    encrypt,
    decrypt,
    build_onion,
    peel_onion,
    generate_response_keys,
    encrypt_response,
    decrypt_response_layers,
    OnionLayer,
)


class TestKeyManagement:
    """Tests for key generation and serialization."""

    def test_keypair_generation(self):
        sk, pk = generate_keypair()
        assert sk is not None
        assert pk is not None

    def test_public_key_roundtrip(self):
        sk, pk = generate_keypair()
        pk_bytes = public_key_to_bytes(pk)
        assert len(pk_bytes) == 32
        pk_restored = public_key_from_bytes(pk_bytes)
        assert public_key_to_bytes(pk_restored) == pk_bytes

    def test_private_key_roundtrip(self):
        sk, pk = generate_keypair()
        sk_bytes = private_key_to_bytes(sk)
        assert len(sk_bytes) == 32
        sk_restored = private_key_from_bytes(sk_bytes)
        # The restored key should produce the same public key
        pk_restored = sk_restored.public_key()
        assert public_key_to_bytes(pk_restored) == public_key_to_bytes(pk)


class TestBasicEncryption:
    """Tests for AES-256-GCM encrypt/decrypt."""

    def test_encrypt_decrypt(self):
        sk1, pk1 = generate_keypair()
        sk2, pk2 = generate_keypair()
        key = derive_shared_key(sk1, pk2)
        key2 = derive_shared_key(sk2, pk1)
        assert key == key2  # DH key exchange is symmetric

        plaintext = b"hello world"
        ct = encrypt(key, plaintext)
        assert ct != plaintext
        pt = decrypt(key, ct)
        assert pt == plaintext

    def test_different_keys_fail(self):
        _, pk = generate_keypair()
        sk, _ = generate_keypair()
        key1 = derive_shared_key(sk, pk)
        key2 = os.urandom(32)  # wrong key

        ct = encrypt(key1, b"secret")
        with pytest.raises(Exception):
            decrypt(key2, ct)

    def test_empty_plaintext(self):
        key = os.urandom(32)
        ct = encrypt(key, b"")
        pt = decrypt(key, ct)
        assert pt == b""

    def test_large_plaintext(self):
        key = os.urandom(32)
        data = os.urandom(10000)
        ct = encrypt(key, data)
        pt = decrypt(key, ct)
        assert pt == data


class TestOnionRouting:
    """Tests for build_onion / peel_onion."""

    def test_single_node_circuit(self):
        """Circuit with just one node (exit only)."""
        sk, pk = generate_keypair()
        onion, eph_key = build_onion(["node1:50051"], [pk])

        layer, next_eph = peel_onion(sk, eph_key, onion)
        assert layer.next_hop == ""  # exit node
        assert layer.payload == b""
        assert next_eph == b""

    def test_two_node_circuit(self):
        """Circuit with entry + exit."""
        sk1, pk1 = generate_keypair()
        sk2, pk2 = generate_keypair()

        addresses = ["node1:50051", "node2:50052"]
        onion, eph_key = build_onion(addresses, [pk1, pk2])

        # Node 1 (entry) peels its layer
        layer1, next_eph = peel_onion(sk1, eph_key, onion)
        assert layer1.next_hop == "node2:50052"
        assert layer1.payload != b""
        assert next_eph != b""

        # Node 2 (exit) peels its layer
        layer2, next_eph2 = peel_onion(sk2, next_eph, layer1.payload)
        assert layer2.next_hop == ""
        assert next_eph2 == b""

    def test_four_node_circuit(self):
        """Full 4-node circuit."""
        keys = [generate_keypair() for _ in range(4)]
        sks = [k[0] for k in keys]
        pks = [k[1] for k in keys]
        addresses = [f"node{i}:{50051 + i}" for i in range(4)]

        onion, eph_key = build_onion(addresses, pks)

        current_onion = onion
        current_eph = eph_key

        for i in range(4):
            layer, next_eph = peel_onion(sks[i], current_eph, current_onion)
            if i < 3:
                assert layer.next_hop == addresses[i + 1]
                assert layer.payload != b""
                current_onion = layer.payload
                current_eph = next_eph
            else:
                assert layer.next_hop == ""

    def test_wrong_key_fails(self):
        """Wrong node key can't peel the layer."""
        sk_right, pk_right = generate_keypair()
        sk_wrong, _ = generate_keypair()

        onion, eph_key = build_onion(["node:50051"], [pk_right])

        with pytest.raises(Exception):
            peel_onion(sk_wrong, eph_key, onion)


class TestReturnPathEncryption:
    """Tests for return-path layered encryption."""

    def test_roundtrip(self):
        """Encrypt through N nodes, decrypt all layers on client."""
        keys, _ = generate_response_keys(4)
        original = b"generated token response"

        # Nodes encrypt in reverse order (node 3 first, node 0 last)
        encrypted = original
        for key in reversed(keys):
            encrypted = encrypt_response(key, encrypted)

        # Client decrypts all layers
        decrypted = decrypt_response_layers(keys, encrypted)
        assert decrypted == original

    def test_single_node(self):
        keys, _ = generate_response_keys(1)
        data = b"single hop"
        encrypted = encrypt_response(keys[0], data)
        decrypted = decrypt_response_layers(keys, encrypted)
        assert decrypted == data

    def test_keys_are_same(self):
        client_keys, node_keys = generate_response_keys(3)
        assert client_keys == node_keys

    def test_key_length(self):
        keys, _ = generate_response_keys(5)
        assert len(keys) == 5
        for k in keys:
            assert len(k) == 32  # 256-bit AES keys
