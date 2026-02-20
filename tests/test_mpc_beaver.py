"""
Unit tests for Beaver triple generation and secure multiplication.
"""

import os
import sys
import threading

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.mpc_beaver import (
    BeaverTriple,
    BeaverTripleShares,
    BeaverTripleStore,
    TrustedDealer,
    OTBeaverTripleProvider,
    get_triple_provider,
    serialize_triple_shares,
    deserialize_triple_shares,
)


class TestBeaverTriple:
    """Tests for BeaverTriple generation."""

    def test_triple_correctness(self):
        """Verify a * b == c for a generated triple."""
        triple = BeaverTriple.generate((4, 4))
        a = triple.party_0.a + triple.party_1.a
        b = triple.party_0.b + triple.party_1.b
        c = triple.party_0.c + triple.party_1.c
        assert torch.allclose(a * b, c, atol=1e-5)

    def test_shares_reconstruct(self):
        """Shares for each component sum to the full value."""
        triple = BeaverTriple.generate((8,))
        for component in ('a', 'b', 'c'):
            p0 = getattr(triple.party_0, component)
            p1 = getattr(triple.party_1, component)
            full = p0 + p1
            assert full.shape == (8,)

    def test_matmul_triple_correctness(self):
        """Verify A @ B^T == C for matmul triples."""
        a_shape = (1, 2, 4, 8)
        b_shape = (1, 2, 4, 8)
        triple = BeaverTriple.generate_matmul(a_shape, b_shape, transpose_b=True)
        a = triple.party_0.a + triple.party_1.a
        b = triple.party_0.b + triple.party_1.b
        c = triple.party_0.c + triple.party_1.c
        expected = torch.matmul(a, b.transpose(-2, -1))
        assert torch.allclose(c, expected, atol=1e-4)

    def test_matmul_triple_no_transpose(self):
        """Verify A @ B == C for matmul triples without transpose."""
        a_shape = (1, 2, 4, 8)
        b_shape = (1, 2, 8, 3)
        triple = BeaverTriple.generate_matmul(a_shape, b_shape, transpose_b=False)
        a = triple.party_0.a + triple.party_1.a
        b = triple.party_0.b + triple.party_1.b
        c = triple.party_0.c + triple.party_1.c
        expected = torch.matmul(a, b)
        assert torch.allclose(c, expected, atol=1e-4)

    def test_different_dtypes(self):
        """Triple generation works with float32."""
        triple = BeaverTriple.generate((4,), dtype=torch.float32)
        assert triple.party_0.a.dtype == torch.float32


class TestBeaverTripleStore:
    """Tests for the triple pool."""

    def test_get_returns_triple(self):
        store = BeaverTripleStore(pool_size=4)
        triple = store.get((3, 3))
        assert triple.party_0.a.shape == (3, 3)

    def test_pregenerate(self):
        store = BeaverTripleStore(pool_size=4)
        store.pregenerate((5, 5), count=10)
        stats = store.stats()
        assert "(5, 5)" in stats
        assert stats["(5, 5)"] == 10

    def test_thread_safety(self):
        """Concurrent access doesn't crash."""
        store = BeaverTripleStore(pool_size=8)
        results = []

        def fetch():
            t = store.get((4, 4))
            results.append(t)

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10


class TestSerializeDeserialize:
    """Tests for triple serialization/deserialization."""

    def test_roundtrip_elementwise(self):
        triple = BeaverTriple.generate((3, 4))
        shares = triple.party_0
        a_b, b_b, c_b, a_s, b_s, c_s = serialize_triple_shares(shares)
        restored = deserialize_triple_shares(a_b, b_b, c_b, a_s, b_s, c_s)
        assert torch.allclose(shares.a, restored.a, atol=1e-6)
        assert torch.allclose(shares.b, restored.b, atol=1e-6)
        assert torch.allclose(shares.c, restored.c, atol=1e-6)

    def test_roundtrip_matmul(self):
        triple = BeaverTriple.generate_matmul((2, 3), (2, 3), transpose_b=True)
        shares = triple.party_1
        a_b, b_b, c_b, a_s, b_s, c_s = serialize_triple_shares(shares)
        restored = deserialize_triple_shares(a_b, b_b, c_b, a_s, b_s, c_s)
        assert torch.allclose(shares.a, restored.a, atol=1e-6)
        assert torch.allclose(shares.c, restored.c, atol=1e-6)

    def test_backward_compat_shape(self):
        """b_shape/c_shape default to a_shape for element-wise triples."""
        triple = BeaverTriple.generate((5,))
        shares = triple.party_0
        a_b, b_b, c_b, a_s, b_s, c_s = serialize_triple_shares(shares)
        restored = deserialize_triple_shares(a_b, b_b, c_b, a_s)
        assert restored.a.shape == (5,)


class TestSecureMultiply:
    """Tests for the Beaver multiply protocol."""

    def test_secure_multiply_correct(self):
        """secure_multiply produces correct product."""
        from network.mpc_protocols import secure_multiply, LocalPeerExchanger

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        triple = BeaverTriple.generate((4, 4))

        exchanger = LocalPeerExchanger()

        def party_0():
            return secure_multiply(
                x * 0.5, y * 0.5, triple.party_0,
                exchanger, "s1", "op1", is_party_0=True)

        def party_1():
            return secure_multiply(
                x * 0.5, y * 0.5, triple.party_1,
                exchanger, "s1", "op1", is_party_0=False)

        t0 = [None]
        t1 = [None]
        threads = [
            threading.Thread(target=lambda: t0.__setitem__(0, party_0())),
            threading.Thread(target=lambda: t1.__setitem__(0, party_1())),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = t0[0] + t1[0]
        expected = x * y
        assert torch.allclose(result, expected, atol=1e-4)


class TestOTBeaverTripleProvider:
    """Tests for the OT-based triple provider."""

    def test_ot_triple_correctness(self):
        """OT-generated triples satisfy a * b == c."""
        provider = OTBeaverTripleProvider()
        triples = provider.generate_for_shape((4, 4), count=3)
        for triple in triples:
            a = triple.party_0.a + triple.party_1.a
            b = triple.party_0.b + triple.party_1.b
            c = triple.party_0.c + triple.party_1.c
            assert torch.allclose(a * b, c, atol=1e-4)

    def test_get_returns_valid(self):
        provider = OTBeaverTripleProvider()
        triple = provider.get((3, 3))
        a = triple.party_0.a + triple.party_1.a
        b = triple.party_0.b + triple.party_1.b
        c = triple.party_0.c + triple.party_1.c
        assert torch.allclose(a * b, c, atol=1e-4)


class TestTripleProviderFactory:
    """Tests for the get_triple_provider factory."""

    def test_ot_mode(self):
        provider = get_triple_provider("ot")
        assert isinstance(provider, OTBeaverTripleProvider)

    def test_trusted_dealer_mode(self):
        provider = get_triple_provider("trusted_dealer")
        assert isinstance(provider, TrustedDealer)
