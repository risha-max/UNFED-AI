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
    MpcDncConfig,
    configure_mpc_dnc,
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


class _CorruptMacExchanger(LocalPeerExchanger):
    """Test helper that flips one MAC exchange payload."""

    def __init__(self):
        super().__init__()
        self._corrupted = False
        self._lock_corrupt = threading.Lock()

    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor, my_delta: torch.Tensor):
        peer_eps, peer_del = super().exchange(session_id, op_id, my_epsilon, my_delta)
        if op_id.endswith("::mac"):
            with self._lock_corrupt:
                if not self._corrupted:
                    self._corrupted = True
                    return peer_eps + 1.0, peer_del
        return peer_eps, peer_del


@pytest.fixture(autouse=True)
def _reset_mpc_dnc_config():
    configure_mpc_dnc(MpcDncConfig(mode="off"))
    yield
    configure_mpc_dnc(MpcDncConfig(mode="off"))


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
            lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "op", True),
            lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "op", False),
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
            lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "op", True),
            lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "op", False),
        )
        result = z0 + z1
        expected = x * y
        assert torch.allclose(result, expected, atol=1e-3)

    def test_partitioned_matches_baseline(self):
        x = torch.randn(2, 64)
        y = torch.randn(2, 64)
        x0, x1 = _split_shares(x)
        y0, y1 = _split_shares(y)
        triple = BeaverTriple.generate((2, 64))
        exchanger = LocalPeerExchanger()

        configure_mpc_dnc(MpcDncConfig(mode="off"))
        b0, b1 = _run_two_party(
            lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "mul-base", True),
            lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "mul-base", False),
        )
        baseline = b0 + b1

        configure_mpc_dnc(MpcDncConfig(
            mode="manual",
            depth=2,
            split_dim=-1,
            parallel_workers=1,
        ))
        p0, p1 = _run_two_party(
            lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "mul-part", True),
            lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "mul-part", False),
        )
        partitioned = p0 + p1
        assert torch.allclose(baseline, partitioned, atol=1e-5), \
            f"Max gap: {(baseline - partitioned).abs().max()}"

    def test_auto_mode_matches_baseline(self):
        x = torch.randn(2, 512)
        y = torch.randn(2, 512)
        x0, x1 = _split_shares(x)
        y0, y1 = _split_shares(y)
        triple = BeaverTriple.generate((2, 512))
        exchanger = LocalPeerExchanger()

        configure_mpc_dnc(MpcDncConfig(mode="off"))
        b0, b1 = _run_two_party(
            lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "mul-auto-base", True),
            lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "mul-auto-base", False),
        )
        baseline = b0 + b1

        configure_mpc_dnc(MpcDncConfig(
            mode="auto",
            auto_min_elems=1,
            auto_chunk_min_elems=1,
            auto_max_depth=2,
        ))
        p0, p1 = _run_two_party(
            lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "mul-auto", True),
            lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "mul-auto", False),
        )
        auto_res = p0 + p1
        assert torch.allclose(baseline, auto_res, atol=1e-5), \
            f"Max gap: {(baseline - auto_res).abs().max()}"

    def test_mac_tamper_fails_closed(self):
        x = torch.randn(2, 16)
        y = torch.randn(2, 16)
        x0, x1 = _split_shares(x)
        y0, y1 = _split_shares(y)
        triple = BeaverTriple.generate((2, 16))
        exchanger = _CorruptMacExchanger()

        with pytest.raises(RuntimeError, match="MAC verification failed"):
            _run_two_party(
                lambda: secure_multiply(x0, y0, triple.party_0, None, exchanger, "s", "mul-mac", True),
                lambda: secure_multiply(x1, y1, triple.party_1, None, exchanger, "s", "mul-mac", False),
            )

    def test_triple_sacrifice_detects_corrupt_check_triple(self):
        x = torch.randn(2, 16)
        y = torch.randn(2, 16)
        x0, x1 = _split_shares(x)
        y0, y1 = _split_shares(y)
        triple = BeaverTriple.generate((2, 16))
        check = BeaverTriple.generate((2, 16))
        # Corrupt one party's check triple share => sacrifice must fail.
        check.party_1.c = check.party_1.c + 1.0
        exchanger = LocalPeerExchanger()

        with pytest.raises(RuntimeError, match="triple sacrifice failed"):
            _run_two_party(
                lambda: secure_multiply(
                    x0, y0, triple.party_0, check.party_0, exchanger, "s", "mul-sac", True
                ),
                lambda: secure_multiply(
                    x1, y1, triple.party_1, check.party_1, exchanger, "s", "mul-sac", False
                ),
            )


class TestSecureSquare:
    """Square via Beaver triple."""

    def test_square(self):
        x = torch.randn(3, 5)
        x0, x1 = _split_shares(x)
        triple = BeaverTriple.generate((3, 5))
        exchanger = LocalPeerExchanger()

        z0, z1 = _run_two_party(
            lambda: secure_square(x0, triple.party_0, None, exchanger, "s", "sq", True),
            lambda: secure_square(x1, triple.party_1, None, exchanger, "s", "sq", False),
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
            lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm", True, True),
            lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm", False, True),
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
            lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm", True, False),
            lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm", False, False),
        )
        result = z0 + z1
        expected = torch.matmul(a, b)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_matmul_partitioned_matches_baseline(self):
        a = torch.randn(1, 4, 8, 16)
        b = torch.randn(1, 4, 8, 16)
        a0, a1 = _split_shares(a)
        b0, b1 = _split_shares(b)
        triple = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=True)
        exchanger = LocalPeerExchanger()

        configure_mpc_dnc(MpcDncConfig(mode="off"))
        base0, base1 = _run_two_party(
            lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm-base", True, True),
            lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm-base", False, True),
        )
        base = base0 + base1

        configure_mpc_dnc(MpcDncConfig(
            mode="manual",
            depth=2,
            matmul_split_dim=-3,
            parallel_workers=1,
        ))
        part0, part1 = _run_two_party(
            lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm-part", True, True),
            lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm-part", False, True),
        )
        part = part0 + part1
        assert torch.allclose(base, part, atol=1e-5), \
            f"Max gap: {(base - part).abs().max()}"

    def test_matmul_auto_mode_matches_baseline(self):
        a = torch.randn(1, 8, 8, 16)
        b = torch.randn(1, 8, 8, 16)
        a0, a1 = _split_shares(a)
        b0, b1 = _split_shares(b)
        triple = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=True)
        exchanger = LocalPeerExchanger()

        configure_mpc_dnc(MpcDncConfig(mode="off"))
        base0, base1 = _run_two_party(
            lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm-auto-base", True, True),
            lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm-auto-base", False, True),
        )
        base = base0 + base1

        configure_mpc_dnc(MpcDncConfig(
            mode="auto",
            auto_min_elems=1,
            auto_chunk_min_elems=1,
            auto_max_depth=2,
        ))
        auto0, auto1 = _run_two_party(
            lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm-auto", True, True),
            lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm-auto", False, True),
        )
        auto_res = auto0 + auto1
        assert torch.allclose(base, auto_res, atol=1e-5), \
            f"Max gap: {(base - auto_res).abs().max()}"

    def test_matmul_mac_tamper_fails_closed(self):
        a = torch.randn(1, 2, 4, 8)
        b = torch.randn(1, 2, 4, 8)
        a0, a1 = _split_shares(a)
        b0, b1 = _split_shares(b)
        triple = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=True)
        exchanger = _CorruptMacExchanger()

        with pytest.raises(RuntimeError, match="MAC verification failed"):
            _run_two_party(
                lambda: secure_matmul(a0, b0, triple.party_0, None, exchanger, "s", "mm-mac", True, True),
                lambda: secure_matmul(a1, b1, triple.party_1, None, exchanger, "s", "mm-mac", False, True),
            )

    def test_matmul_triple_sacrifice_detects_corrupt_check_triple(self):
        a = torch.randn(1, 2, 4, 8)
        b = torch.randn(1, 2, 4, 8)
        a0, a1 = _split_shares(a)
        b0, b1 = _split_shares(b)
        triple = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=True)
        check = BeaverTriple.generate_matmul(a.shape, b.shape, transpose_b=True)
        check.party_0.c = check.party_0.c - 1.0
        exchanger = LocalPeerExchanger()

        with pytest.raises(RuntimeError, match="triple sacrifice failed"):
            _run_two_party(
                lambda: secure_matmul(
                    a0, b0, triple.party_0, check.party_0, exchanger, "s", "mm-sac", True, True
                ),
                lambda: secure_matmul(
                    a1, b1, triple.party_1, check.party_1, exchanger, "s", "mm-sac", False, True
                ),
            )


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
