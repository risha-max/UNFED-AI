"""
Beaver Triple infrastructure for 2-party MPC.

A Beaver triple (a, b, c) satisfies c = a * b.  Given additive shares of
a, b, c distributed to two parties, they can multiply *any* two secret-shared
values x and y without learning anything:

  1. Both parties open  epsilon = x - a  and  delta = y - b
     (random-looking — reveals nothing about x or y)
  2. Party i computes   z_i = c_i + epsilon * b_i + delta * a_i
     (Party 0 additionally adds epsilon * delta)
  3. z_0 + z_1 = x * y

This module provides:
  - BeaverTriple: a single (a, b, c) triple with shares for both parties
  - BeaverTripleStore: pre-generates pools of triples keyed by tensor shape
  - TrustedDealer: generates triples where one party creates both shares
    (suitable for demos; production would use OT-based generation)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BeaverTripleShares:
    """One party's share of a Beaver triple."""
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


@dataclass
class BeaverTriple:
    """A complete Beaver triple with shares for both parties."""
    party_0: BeaverTripleShares
    party_1: BeaverTripleShares

    @staticmethod
    def generate(shape: tuple[int, ...],
                 dtype: torch.dtype = torch.float32) -> "BeaverTriple":
        """Generate a single Beaver triple (a, b, c = a*b) and split into
        additive shares for two parties."""
        a = torch.randn(shape, dtype=dtype)
        b = torch.randn(shape, dtype=dtype)
        c = a * b

        # Split into additive shares: x = x_0 + x_1
        a_0 = torch.randn_like(a)
        a_1 = a - a_0
        b_0 = torch.randn_like(b)
        b_1 = b - b_0
        c_0 = torch.randn_like(c)
        c_1 = c - c_0

        return BeaverTriple(
            party_0=BeaverTripleShares(a=a_0, b=b_0, c=c_0),
            party_1=BeaverTripleShares(a=a_1, b=b_1, c=c_1),
        )

    @staticmethod
    def generate_matmul(
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
        transpose_b: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> "BeaverTriple":
        """Generate a Beaver triple for matrix multiplication.

        For transpose_b=True:  C = A @ B^T
        For transpose_b=False: C = A @ B

        a and b may have different shapes; c has the product shape.
        Shares are stored in BeaverTripleShares where a/b/c have their
        respective shapes (not necessarily identical).
        """
        a = torch.randn(a_shape, dtype=dtype)
        b = torch.randn(b_shape, dtype=dtype)
        if transpose_b:
            c = torch.matmul(a, b.transpose(-2, -1))
        else:
            c = torch.matmul(a, b)

        a_0 = torch.randn_like(a)
        a_1 = a - a_0
        b_0 = torch.randn_like(b)
        b_1 = b - b_0
        c_0 = torch.randn_like(c)
        c_1 = c - c_0

        return BeaverTriple(
            party_0=BeaverTripleShares(a=a_0, b=b_0, c=c_0),
            party_1=BeaverTripleShares(a=a_1, b=b_1, c=c_1),
        )


class BeaverTripleStore:
    """
    Pre-generates and dispenses Beaver triples for secure multiplications.

    Triples are pooled by shape key.  When a pool runs low, more are
    generated in the background.  Thread-safe.
    """

    def __init__(self, pool_size: int = 64):
        self._pool_size = pool_size
        self._pools: dict[tuple[int, ...], list[BeaverTriple]] = {}
        self._lock = threading.Lock()

    def _refill(self, shape: tuple[int, ...]):
        """Refill the pool for a given shape (caller holds lock)."""
        pool = self._pools.setdefault(shape, [])
        while len(pool) < self._pool_size:
            pool.append(BeaverTriple.generate(shape))

    def get(self, shape: tuple[int, ...]) -> BeaverTriple:
        """Get one Beaver triple for the given tensor shape."""
        with self._lock:
            pool = self._pools.get(shape)
            if not pool:
                self._refill(shape)
                pool = self._pools[shape]
            if len(pool) < 4:
                self._refill(shape)
            return pool.pop()

    def pregenerate(self, shape: tuple[int, ...], count: Optional[int] = None):
        """Pre-generate triples for a specific shape (called at startup)."""
        n = count or self._pool_size
        with self._lock:
            pool = self._pools.setdefault(shape, [])
            for _ in range(n):
                pool.append(BeaverTriple.generate(shape))

    def stats(self) -> dict[str, int]:
        """Return current pool sizes."""
        with self._lock:
            return {str(k): len(v) for k, v in self._pools.items()}


class TrustedDealer:
    """
    Trusted dealer that generates Beaver triples and distributes shares
    to both MPC parties.

    In this demo model, Node A acts as the dealer: it generates the full
    triple, keeps party_0 shares, and sends party_1 shares to Node B
    during the setup phase.

    In production, use OTBeaverTripleProvider which generates triples
    via Oblivious Transfer so neither party trusts the other.
    """

    def __init__(self, store: Optional[BeaverTripleStore] = None):
        self._store = store or BeaverTripleStore()

    def generate_for_shape(self, shape: tuple[int, ...]) -> BeaverTriple:
        """Generate one triple and return both parties' shares."""
        return self._store.get(shape)

    def generate_batch(self, shape: tuple[int, ...],
                       count: int) -> list[BeaverTriple]:
        """Generate a batch of triples for a given shape."""
        triples = []
        for _ in range(count):
            triples.append(BeaverTriple.generate(shape))
        return triples

    def pregenerate_for_layer(self, hidden_size: int,
                              num_heads: int,
                              head_dim: int,
                              intermediate_size: int,
                              seq_len: int = 128,
                              count: int = 64):
        """
        Pre-generate triples for all operations in one transformer layer.

        Shapes needed:
          - RMSNorm:    (1, seq_len, hidden_size) — for x^2, x*norm
          - Attention:   (1, num_heads, seq_len, seq_len) — for softmax terms
          - SiLU/MLP:   (1, seq_len, intermediate_size) — for gate activation
        """
        shapes = [
            (1, seq_len, hidden_size),          # RMSNorm, general hidden
            (1, num_heads, seq_len, seq_len),   # Attention scores
            (1, seq_len, intermediate_size),     # MLP intermediate
            (1, seq_len, 1),                     # Scalar per-position (norms)
        ]
        for shape in shapes:
            self._store.pregenerate(shape, count)
        return shapes


class OTBeaverTripleProvider:
    """
    Generate Beaver triples via OT extension (no trusted dealer).

    Uses the IKNP OT extension protocol so neither party needs to trust
    the other for triple generation.  Both parties run the OT protocol
    cooperatively and each receives their share of valid triples.

    Config selection:
        - ``mpc_triple_mode: "ot"`` in ClusterConfig to use this provider
        - ``mpc_triple_mode: "trusted_dealer"`` for the demo TrustedDealer

    Falls back to TrustedDealer-compatible BeaverTriple format so the rest
    of the MPC pipeline is unchanged.
    """

    def __init__(self):
        self._cache: dict[tuple[int, ...], list[BeaverTriple]] = {}
        self._lock = threading.Lock()

    def generate_for_shape(self, shape: tuple[int, ...],
                           count: int = 1) -> list[BeaverTriple]:
        """Generate triples for a given shape using OT extension.

        Returns list of BeaverTriple (both parties' shares) for local
        testing.  In a real 2-party setting, each party only receives
        their own shares via the OT protocol.
        """
        from network.mpc_ot_extension import generate_beaver_triples_ot

        p0_list, p1_list = generate_beaver_triples_ot(
            n_triples=count, shape=shape)

        triples = []
        for p0, p1 in zip(p0_list, p1_list):
            triples.append(BeaverTriple(
                party_0=BeaverTripleShares(a=p0['a'], b=p0['b'], c=p0['c']),
                party_1=BeaverTripleShares(a=p1['a'], b=p1['b'], c=p1['c']),
            ))
        return triples

    def get(self, shape: tuple[int, ...]) -> BeaverTriple:
        """Get one triple, generating on demand."""
        with self._lock:
            cache = self._cache.get(shape)
            if cache:
                return cache.pop()
        # Generate a small batch
        batch = self.generate_for_shape(shape, count=8)
        with self._lock:
            pool = self._cache.setdefault(shape, [])
            pool.extend(batch[1:])
        return batch[0]

    def pregenerate(self, shape: tuple[int, ...], count: int = 16):
        """Pre-generate a pool of OT-based triples."""
        batch = self.generate_for_shape(shape, count=count)
        with self._lock:
            pool = self._cache.setdefault(shape, [])
            pool.extend(batch)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {str(k): len(v) for k, v in self._cache.items()}


def get_triple_provider(mode: str = "ot"):
    """Factory for triple providers.

    Args:
        mode: "ot" for OT-based generation, "trusted_dealer" for the demo dealer.

    Returns:
        A provider with .get(shape) and .generate_for_shape(shape) methods.
    """
    if mode == "ot":
        return OTBeaverTripleProvider()
    else:
        return TrustedDealer()


def serialize_triple_shares(
    shares: BeaverTripleShares,
) -> tuple[bytes, bytes, bytes, list[int], list[int], list[int]]:
    """Serialize a party's triple shares to bytes for gRPC transfer.

    Returns (a_bytes, b_bytes, c_bytes, a_shape, b_shape, c_shape).
    For element-wise triples all shapes are identical; for matmul
    triples a/b/c may differ.
    """
    a_bytes = shares.a.contiguous().float().numpy().tobytes()
    b_bytes = shares.b.contiguous().float().numpy().tobytes()
    c_bytes = shares.c.contiguous().float().numpy().tobytes()
    return (a_bytes, b_bytes, c_bytes,
            list(shares.a.shape), list(shares.b.shape), list(shares.c.shape))


def deserialize_triple_shares(
    a_bytes: bytes,
    b_bytes: bytes,
    c_bytes: bytes,
    a_shape: list[int],
    b_shape: list[int] | None = None,
    c_shape: list[int] | None = None,
) -> BeaverTripleShares:
    """Deserialize triple shares received from the dealer.

    b_shape/c_shape default to a_shape for backward compat (element-wise).
    """
    import numpy as np
    if b_shape is None:
        b_shape = a_shape
    if c_shape is None:
        c_shape = a_shape
    a = torch.from_numpy(
        np.frombuffer(a_bytes, dtype=np.float32).copy().reshape(a_shape))
    b = torch.from_numpy(
        np.frombuffer(b_bytes, dtype=np.float32).copy().reshape(b_shape))
    c = torch.from_numpy(
        np.frombuffer(c_bytes, dtype=np.float32).copy().reshape(c_shape))
    return BeaverTripleShares(a=a, b=b, c=c)
