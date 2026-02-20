"""
Unit tests for MPC protocols: RMSNorm, softmax, SiLU approximation accuracy.

Compares MPC outputs against PyTorch native implementations and asserts
that the error is below acceptable thresholds.
"""

import os
import sys
import threading

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.mpc_beaver import BeaverTriple, BeaverTripleShares
from network.mpc_protocols import (
    secure_multiply,
    secure_square,
    secure_matmul,
    secure_rmsnorm,
    secure_softmax,
    secure_silu,
    secure_gate_up,
    LocalPeerExchanger,
    TripleAllocator,
    allocate_layer0_triples,
    layer0_op_ids,
)


def _run_two_party(fn_party_0, fn_party_1):
    """Helper: run two functions in parallel and return both results."""
    r0 = [None]
    r1 = [None]
    exc = [None]

    def run0():
        try:
            r0[0] = fn_party_0()
        except Exception as e:
            exc[0] = e

    def run1():
        try:
            r1[0] = fn_party_1()
        except Exception as e:
            exc[0] = e

    t0 = threading.Thread(target=run0)
    t1 = threading.Thread(target=run1)
    t0.start()
    t1.start()
    t0.join(timeout=30)
    t1.join(timeout=30)

    if exc[0]:
        raise exc[0]
    return r0[0], r1[0]


def _split_shares(x: torch.Tensor):
    """Split a tensor into additive shares."""
    x0 = torch.randn_like(x)
    x1 = x - x0
    return x0, x1


class TestSecureMultiply:
    """Element-wise multiplication."""

    def test_basic(self):
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        x0, x1 = _split_shares(x)
        y0, y1 = _split_shares(y)
        triple = BeaverTriple.generate((4, 8))
        exchanger = LocalPeerExchanger()

        z0, z1 = _run_two_party(
            lambda: secure_multiply(x0, y0, triple.party_0, exchanger, "s", "op", True),
            lambda: secure_multiply(x1, y1, triple.party_1, exchanger, "s", "op", False),
        )
        result = z0 + z1
        expected = x * y
        assert torch.allclose(result, expected, atol=1e-4), \
            f"Max error: {(result - expected).abs().max()}"

    def test_large_values(self):
        x = torch.randn(2, 16) * 10
        y = torch.randn(2, 16) * 10
        x0, x1 = _split_shares(x)
        y0, y1 = _split_shares(y)
        triple = BeaverTriple.generate((2, 16))
        exchanger = LocalPeerExchanger()

        z0, z1 = _run_two_party(
            lambda: secure_multiply(x0, y0, triple.party_0, exchanger, "s", "op", True),
            lambda: secure_multiply(x1, y1, triple.party_1, exchanger, "s", "op", False),
        )
        result = z0 + z1
        expected = x * y
        assert torch.allclose(result, expected, atol=1e-3)


class TestSecureSquare:
    """Square via Beaver triple."""

    def test_square(self):
        x = torch.randn(3, 5)
        x0, x1 = _split_shares(x)
        triple = BeaverTriple.generate((3, 5))
        exchanger = LocalPeerExchanger()

        z0, z1 = _run_two_party(
            lambda: secure_square(x0, triple.party_0, exchanger, "s", "sq", True),
            lambda: secure_square(x1, triple.party_1, exchanger, "s", "sq", False),
        )
        result = z0 + z1
        expected = x ** 2
        assert torch.allclose(result, expected, atol=1e-4)


class TestSecureMatmul:
    """Secure matrix multiplication."""

    def test_matmul_transpose(self):
        a = torch.randn(1, 2, 4, 8)
        b = torch.randn(1, 2, 4, 8)
        a0, a1 = _split_shares(a)
        b0, b1 = _split_shares(b)
        triple = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=True)
        exchanger = LocalPeerExchanger()

        z0, z1 = _run_two_party(
            lambda: secure_matmul(a0, b0, triple.party_0, exchanger, "s", "mm", True, True),
            lambda: secure_matmul(a1, b1, triple.party_1, exchanger, "s", "mm", False, True),
        )
        result = z0 + z1
        expected = torch.matmul(a, b.transpose(-2, -1))
        assert torch.allclose(result, expected, atol=1e-3), \
            f"Max error: {(result - expected).abs().max()}"

    def test_matmul_no_transpose(self):
        a = torch.randn(1, 2, 4, 4)
        b = torch.randn(1, 2, 4, 8)
        a0, a1 = _split_shares(a)
        b0, b1 = _split_shares(b)
        triple = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=False)
        exchanger = LocalPeerExchanger()

        z0, z1 = _run_two_party(
            lambda: secure_matmul(a0, b0, triple.party_0, exchanger, "s", "mm", True, False),
            lambda: secure_matmul(a1, b1, triple.party_1, exchanger, "s", "mm", False, False),
        )
        result = z0 + z1
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-3)


class TestSecureRMSNorm:
    """RMSNorm with variance revealing."""

    def test_accuracy(self, hidden_size, seq_len, batch):
        x = torch.randn(batch, seq_len, hidden_size)
        weight = torch.randn(hidden_size)
        eps = 1e-5

        # Cleartext reference
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(variance + eps) * weight

        # MPC
        x0, x1 = _split_shares(x)
        triples = allocate_layer0_triples(
            hidden_size, 4, hidden_size // 4, hidden_size * 4, seq_len, batch)
        exchanger = LocalPeerExchanger()

        alloc_0 = TripleAllocator({
            k: v.party_0 for k, v in triples.items()})
        alloc_1 = TripleAllocator({
            k: v.party_1 for k, v in triples.items()})

        z0, z1 = _run_two_party(
            lambda: secure_rmsnorm(x0, weight, eps, alloc_0, exchanger, "s", True, "rmsnorm_in"),
            lambda: secure_rmsnorm(x1, weight, eps, alloc_1, exchanger, "s", False, "rmsnorm_in"),
        )
        result = z0 + z1
        assert not torch.isnan(result).any(), "NaN in RMSNorm output"
        error = (result - expected).abs().mean().item()
        assert error < 0.1, f"Mean RMSNorm error too large: {error}"


class TestSecureSoftmax:
    """Softmax with score revealing."""

    def test_accuracy(self, num_heads, seq_len, batch):
        x = torch.randn(batch, num_heads, seq_len, seq_len)
        expected = torch.softmax(x, dim=-1)

        x0, x1 = _split_shares(x)
        triples = allocate_layer0_triples(
            num_heads * seq_len, num_heads, seq_len, 1, seq_len, batch)
        exchanger = LocalPeerExchanger()

        alloc_0 = TripleAllocator({
            k: v.party_0 for k, v in triples.items()})
        alloc_1 = TripleAllocator({
            k: v.party_1 for k, v in triples.items()})

        z0, z1 = _run_two_party(
            lambda: secure_softmax(x0, alloc_0, exchanger, "s", True),
            lambda: secure_softmax(x1, alloc_1, exchanger, "s", False),
        )
        result = z0 + z1
        assert not torch.isnan(result).any(), "NaN in softmax output"
        # Since we reveal and compute in cleartext, result should be exact
        assert torch.allclose(result, expected, atol=1e-5)


class TestSecureSiLU:
    """SiLU via polynomial approximation."""

    def test_reasonable_range(self, intermediate_size, seq_len, batch):
        x = torch.randn(batch, seq_len, intermediate_size)
        expected = torch.nn.functional.silu(x)

        x0, x1 = _split_shares(x)
        triples = allocate_layer0_triples(
            intermediate_size, 1, intermediate_size, intermediate_size,
            seq_len, batch)
        exchanger = LocalPeerExchanger()

        alloc_0 = TripleAllocator({
            k: v.party_0 for k, v in triples.items()})
        alloc_1 = TripleAllocator({
            k: v.party_1 for k, v in triples.items()})

        z0, z1 = _run_two_party(
            lambda: secure_silu(x0, alloc_0, exchanger, "s", True),
            lambda: secure_silu(x1, alloc_1, exchanger, "s", False),
        )
        result = z0 + z1
        assert not torch.isnan(result).any(), "NaN in SiLU output"
        error = (result - expected).abs().mean().item()
        # Polynomial approximation has ~1e-2 error
        assert error < 0.5, f"Mean SiLU error too large: {error}"


class TestLayer0OpIds:
    """Verify operation ID scheme."""

    def test_expected_ops(self):
        ops = layer0_op_ids()
        assert "rmsnorm_in_sq" in ops
        assert "attn_qk_matmul" in ops
        assert "attn_av_matmul" in ops
        assert "silu_x2" in ops
        assert "mlp_gate_up" in ops

    def test_no_duplicates(self):
        ops = layer0_op_ids()
        assert len(ops) == len(set(ops))


class TestAllocateLayer0Triples:
    """Verify triple allocation produces all expected ops."""

    def test_all_ops_allocated(self):
        triples = allocate_layer0_triples(
            hidden_size=64, num_heads=4, head_dim=16,
            intermediate_size=256, seq_len=8, batch=1)
        expected_ops = layer0_op_ids()
        for op in expected_ops:
            assert op in triples, f"Missing triple for {op}"
