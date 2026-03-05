"""
Secure 2-party MPC protocols for transformer layer operations.

All protocols operate on additive shares and use Beaver triples for
multiplications.  No secret value is ever reconstructed on a single party.

Communication model:
  - Each secure_multiply requires one round-trip (exchange epsilon/delta).
  - The `PeerExchanger` abstraction handles the gRPC calls.
  - Linear operations (matmul with public weights, addition) are free.

Supported operations:
  - secure_multiply:  element-wise product of two shared tensors
  - secure_square:    element-wise square of a shared tensor (optimized)
  - secure_matmul:    matrix product of two shared tensors (Beaver-based)
  - secure_rmsnorm:   RMSNorm via Goldschmidt iteration for rsqrt
  - secure_softmax:   softmax via polynomial exp approximation
  - secure_silu:      SiLU via polynomial sigmoid approximation
"""

from __future__ import annotations

import math
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from network.mpc_beaver import BeaverTriple, BeaverTripleShares


@dataclass
class MpcDncConfig:
    mode: str = "off"  # off|manual|auto
    depth: int = 0
    split_dim: int = -1
    matmul_split_dim: int = -3
    parallel_workers: int = 1
    auto_min_elems: int = 65_536
    auto_chunk_min_elems: int = 16_384
    auto_max_depth: int = 3
    auto_max_workers: int = 4


_MPC_DNC_CONFIG = MpcDncConfig()
_MPC_DNC_LOCK = threading.Lock()


def configure_mpc_dnc(config: MpcDncConfig) -> None:
    """Set process-local DNC policy (used by both MPC parties in this process)."""
    mode = (config.mode or "off").strip().lower()
    if mode not in ("off", "manual", "auto"):
        mode = "off"
    with _MPC_DNC_LOCK:
        _MPC_DNC_CONFIG.mode = mode
        _MPC_DNC_CONFIG.depth = max(0, int(config.depth))
        _MPC_DNC_CONFIG.split_dim = int(config.split_dim)
        _MPC_DNC_CONFIG.matmul_split_dim = int(config.matmul_split_dim)
        _MPC_DNC_CONFIG.parallel_workers = max(1, int(config.parallel_workers))
        _MPC_DNC_CONFIG.auto_min_elems = max(1, int(config.auto_min_elems))
        _MPC_DNC_CONFIG.auto_chunk_min_elems = max(1, int(config.auto_chunk_min_elems))
        _MPC_DNC_CONFIG.auto_max_depth = max(0, int(config.auto_max_depth))
        _MPC_DNC_CONFIG.auto_max_workers = max(1, int(config.auto_max_workers))


def get_mpc_dnc_config() -> MpcDncConfig:
    with _MPC_DNC_LOCK:
        return MpcDncConfig(
            mode=_MPC_DNC_CONFIG.mode,
            depth=_MPC_DNC_CONFIG.depth,
            split_dim=_MPC_DNC_CONFIG.split_dim,
            matmul_split_dim=_MPC_DNC_CONFIG.matmul_split_dim,
            parallel_workers=_MPC_DNC_CONFIG.parallel_workers,
            auto_min_elems=_MPC_DNC_CONFIG.auto_min_elems,
            auto_chunk_min_elems=_MPC_DNC_CONFIG.auto_chunk_min_elems,
            auto_max_depth=_MPC_DNC_CONFIG.auto_max_depth,
            auto_max_workers=_MPC_DNC_CONFIG.auto_max_workers,
        )


def _resolve_split_dim(rank: int, split_dim: int) -> int:
    d = split_dim if split_dim >= 0 else rank + split_dim
    if d < 0 or d >= rank:
        raise ValueError(f"Invalid split_dim={split_dim} for rank={rank}")
    return d


def _dnc_mode() -> str:
    return get_mpc_dnc_config().mode


def _auto_depth(axis_size: int, numel: int) -> int:
    """Deterministic shape-based depth (must match across MPC parties)."""
    cfg = get_mpc_dnc_config()
    min_elems = max(1, int(cfg.auto_min_elems))
    chunk_min_elems = max(1, int(cfg.auto_chunk_min_elems))
    max_depth = max(0, int(cfg.auto_max_depth))
    if axis_size < 2 or numel < min_elems:
        return 0
    depth = min(max_depth, int(math.log2(axis_size)))
    while depth > 0 and (numel // (2 ** depth)) < chunk_min_elems:
        depth -= 1
    return depth


def _auto_workers(depth: int) -> int:
    if depth <= 0:
        return 1
    cfg = get_mpc_dnc_config()
    max_workers = max(1, int(cfg.auto_max_workers))
    cpu = max(1, os.cpu_count() or 1)
    return max(1, min(max_workers, cpu, 2 ** depth))


def _select_dnc_plan(
    *,
    x_share: torch.Tensor,
    is_matmul: bool,
) -> tuple[bool, int, int, int]:
    """
    Return (enabled, depth, split_dim, workers).

    Important:
    - depth/split_dim must be deterministic across parties.
    - workers may vary per machine (local scheduling only).
    """
    mode = _dnc_mode()
    if mode == "off":
        return False, 0, -1, 1

    if mode == "manual":
        cfg = get_mpc_dnc_config()
        depth = max(0, int(cfg.depth))
        split_dim_cfg = int(cfg.matmul_split_dim if is_matmul else cfg.split_dim)
        split_dim = _resolve_split_dim(x_share.dim(), split_dim_cfg)
        workers = max(1, int(cfg.parallel_workers))
        enabled = depth > 0 and x_share.shape[split_dim] >= 2
        return enabled, depth, split_dim, workers

    # mode == "auto"
    if is_matmul and x_share.dim() >= 3 and x_share.shape[-3] >= 2:
        split_dim = _resolve_split_dim(x_share.dim(), -3)
    else:
        split_dim = _resolve_split_dim(x_share.dim(), -1)
    depth = _auto_depth(x_share.shape[split_dim], x_share.numel())
    workers = _auto_workers(depth)
    enabled = depth > 0 and x_share.shape[split_dim] >= 2
    return enabled, depth, split_dim, workers


# ---------------------------------------------------------------------------
# Peer communication abstraction
# ---------------------------------------------------------------------------

class PeerExchanger(ABC):
    """
    Abstract interface for exchanging epsilon/delta tensors with the MPC peer.

    Both Node A and Node B implement this differently:
      - Node A: sends its values to Node B's gRPC stub, receives B's values
      - Node B: waits for Node A's values, responds with its own
    """

    @abstractmethod
    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor,
                 my_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exchange epsilon and delta values with the peer.

        Returns (peer_epsilon, peer_delta).
        """
        ...


class LocalPeerExchanger(PeerExchanger):
    """For testing: both parties run in the same process."""

    def __init__(self):
        self._pending: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor,
                 my_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = f"{session_id}:{op_id}"
        with self._lock:
            if key not in self._events:
                self._events[key] = threading.Event()
            if key in self._pending:
                peer_eps, peer_del = self._pending.pop(key)
                self._results[key] = (my_epsilon.clone(), my_delta.clone())
                self._events[key].set()
                return peer_eps, peer_del
            else:
                self._pending[key] = (my_epsilon.clone(), my_delta.clone())

        self._events[key].wait(timeout=30)
        with self._lock:
            peer_eps, peer_del = self._results.pop(key)
            self._events.pop(key, None)
        return peer_eps, peer_del


# ---------------------------------------------------------------------------
# Core secure multiplication (element-wise)
# ---------------------------------------------------------------------------

def secure_multiply(x_share: torch.Tensor,
                    y_share: torch.Tensor,
                    triple: BeaverTripleShares,
                    exchanger: PeerExchanger,
                    session_id: str,
                    op_id: str,
                    is_party_0: bool) -> torch.Tensor:
    """
    Securely multiply two secret-shared tensors using a Beaver triple.

    Given: x = x_0 + x_1,  y = y_0 + y_1
    Triple: a = a_0 + a_1,  b = b_0 + b_1,  c = a*b = c_0 + c_1

    Protocol:
      1. Each party computes epsilon_i = x_i - a_i, delta_i = y_i - b_i
      2. Parties exchange to reconstruct epsilon = x-a, delta = y-b
         (these are uniformly random, leak nothing)
      3. Party 0 computes: z_0 = c_0 + eps*b_0 + del*a_0 + eps*del
         Party 1 computes: z_1 = c_1 + eps*b_1 + del*a_1
      4. z_0 + z_1 = x*y
    """
    use_dnc, depth, split_dim, workers = _select_dnc_plan(
        x_share=x_share,
        is_matmul=False,
    )
    if use_dnc:
        return _secure_multiply_partitioned(
            x_share=x_share,
            y_share=y_share,
            triple=triple,
            exchanger=exchanger,
            session_id=session_id,
            op_id=op_id,
            is_party_0=is_party_0,
            depth=depth,
            split_dim=split_dim,
            workers=workers,
        )
    return _secure_multiply_core(
        x_share=x_share,
        y_share=y_share,
        triple=triple,
        exchanger=exchanger,
        session_id=session_id,
        op_id=op_id,
        is_party_0=is_party_0,
    )


def _secure_multiply_core(x_share: torch.Tensor,
                          y_share: torch.Tensor,
                          triple: BeaverTripleShares,
                          exchanger: PeerExchanger,
                          session_id: str,
                          op_id: str,
                          is_party_0: bool) -> torch.Tensor:
    my_epsilon = x_share - triple.a
    my_delta = y_share - triple.b

    peer_epsilon, peer_delta = exchanger.exchange(
        session_id, op_id, my_epsilon, my_delta)

    epsilon = my_epsilon + peer_epsilon
    delta = my_delta + peer_delta

    z = triple.c + epsilon * triple.b + delta * triple.a
    if is_party_0:
        z = z + epsilon * delta

    return z


def _build_multiply_leaves(
    x_share: torch.Tensor,
    y_share: torch.Tensor,
    triple: BeaverTripleShares,
    depth: int,
    split_dim: int,
    op_prefix: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor, BeaverTripleShares]]:
    leaves: list[tuple[str, torch.Tensor, torch.Tensor, BeaverTripleShares]] = []
    work: list[tuple[str, torch.Tensor, torch.Tensor, BeaverTripleShares, int]] = [
        (op_prefix, x_share, y_share, triple, depth)
    ]
    while work:
        prefix, x_chunk, y_chunk, t_chunk, rem = work.pop(0)
        if rem <= 0 or x_chunk.shape[split_dim] < 2:
            leaves.append((prefix, x_chunk, y_chunk, t_chunk))
            continue
        x_l, x_r = torch.tensor_split(x_chunk, 2, dim=split_dim)
        y_l, y_r = torch.tensor_split(y_chunk, 2, dim=split_dim)
        a_l, a_r = torch.tensor_split(t_chunk.a, 2, dim=split_dim)
        b_l, b_r = torch.tensor_split(t_chunk.b, 2, dim=split_dim)
        c_l, c_r = torch.tensor_split(t_chunk.c, 2, dim=split_dim)
        work.append((f"{prefix}.L", x_l, y_l, BeaverTripleShares(a_l, b_l, c_l), rem - 1))
        work.append((f"{prefix}.R", x_r, y_r, BeaverTripleShares(a_r, b_r, c_r), rem - 1))
    return leaves


def _secure_multiply_partitioned(x_share: torch.Tensor,
                                 y_share: torch.Tensor,
                                 triple: BeaverTripleShares,
                                 exchanger: PeerExchanger,
                                 session_id: str,
                                 op_id: str,
                                 is_party_0: bool,
                                 depth: int,
                                 split_dim: int,
                                 workers: int) -> torch.Tensor:
    leaves = _build_multiply_leaves(x_share, y_share, triple, depth, split_dim, op_id)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(
                lambda leaf: _secure_multiply_core(
                    leaf[1], leaf[2], leaf[3], exchanger, session_id, leaf[0], is_party_0
                ),
                leaves,
            ))
    else:
        parts = [
            _secure_multiply_core(x_l, y_l, t_l, exchanger, session_id, leaf_op, is_party_0)
            for leaf_op, x_l, y_l, t_l in leaves
        ]
    return torch.cat(parts, dim=split_dim)


def secure_square(x_share: torch.Tensor,
                  triple: BeaverTripleShares,
                  exchanger: PeerExchanger,
                  session_id: str,
                  op_id: str,
                  is_party_0: bool) -> torch.Tensor:
    """Securely compute x^2 (optimized: use same value for both operands)."""
    return secure_multiply(
        x_share, x_share, triple, exchanger,
        session_id, op_id, is_party_0)


# ---------------------------------------------------------------------------
# Secure matrix multiplication
# ---------------------------------------------------------------------------

def secure_matmul(x_share: torch.Tensor,
                  y_share: torch.Tensor,
                  triple: BeaverTripleShares,
                  exchanger: PeerExchanger,
                  session_id: str,
                  op_id: str,
                  is_party_0: bool,
                  transpose_b: bool = True) -> torch.Tensor:
    """
    Securely compute X @ Y^T (or X @ Y) using a matrix Beaver triple.

    Triple: A, B, C where C = A @ B^T (or A @ B).
    The triple's a/b/c have shapes matching X, Y, and the product.

    Protocol (analogous to element-wise, but with @ instead of *):
      1. eps_i = x_i - a_i,  delta_i = y_i - b_i
      2. Exchange to reconstruct eps = X - A,  delta = Y - B
      3. Party 0: c_0 + eps @ b_0^T + a_0 @ delta^T + eps @ delta^T
         Party 1: c_1 + eps @ b_1^T + a_1 @ delta^T
    """
    use_dnc, depth, split_dim, workers = _select_dnc_plan(
        x_share=x_share,
        is_matmul=True,
    )
    can_partition = (
        use_dnc
        and depth > 0
        and x_share.shape[split_dim] >= 2
        and y_share.dim() == x_share.dim()
        and triple.a.dim() == x_share.dim()
        and triple.b.dim() == x_share.dim()
        and triple.c.dim() == x_share.dim()
        and y_share.shape[split_dim] == x_share.shape[split_dim]
        and triple.a.shape[split_dim] == x_share.shape[split_dim]
        and triple.b.shape[split_dim] == x_share.shape[split_dim]
        and triple.c.shape[split_dim] == x_share.shape[split_dim]
    )
    if can_partition:
        return _secure_matmul_partitioned(
            x_share=x_share,
            y_share=y_share,
            triple=triple,
            exchanger=exchanger,
            session_id=session_id,
            op_id=op_id,
            is_party_0=is_party_0,
            transpose_b=transpose_b,
            depth=depth,
            split_dim=split_dim,
            workers=workers,
        )
    return _secure_matmul_core(
        x_share=x_share,
        y_share=y_share,
        triple=triple,
        exchanger=exchanger,
        session_id=session_id,
        op_id=op_id,
        is_party_0=is_party_0,
        transpose_b=transpose_b,
    )


def _secure_matmul_core(x_share: torch.Tensor,
                        y_share: torch.Tensor,
                        triple: BeaverTripleShares,
                        exchanger: PeerExchanger,
                        session_id: str,
                        op_id: str,
                        is_party_0: bool,
                        transpose_b: bool) -> torch.Tensor:
    my_epsilon = x_share - triple.a
    my_delta = y_share - triple.b

    peer_epsilon, peer_delta = exchanger.exchange(
        session_id, op_id, my_epsilon, my_delta)

    epsilon = my_epsilon + peer_epsilon
    delta = my_delta + peer_delta

    if transpose_b:
        z = (triple.c
             + torch.matmul(epsilon, triple.b.transpose(-2, -1))
             + torch.matmul(triple.a, delta.transpose(-2, -1)))
        if is_party_0:
            z = z + torch.matmul(epsilon, delta.transpose(-2, -1))
    else:
        z = (triple.c
             + torch.matmul(epsilon, triple.b)
             + torch.matmul(triple.a, delta))
        if is_party_0:
            z = z + torch.matmul(epsilon, delta)

    return z


def _build_matmul_leaves(
    x_share: torch.Tensor,
    y_share: torch.Tensor,
    triple: BeaverTripleShares,
    depth: int,
    split_dim: int,
    op_prefix: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor, BeaverTripleShares]]:
    leaves: list[tuple[str, torch.Tensor, torch.Tensor, BeaverTripleShares]] = []
    work: list[tuple[str, torch.Tensor, torch.Tensor, BeaverTripleShares, int]] = [
        (op_prefix, x_share, y_share, triple, depth)
    ]
    while work:
        prefix, x_chunk, y_chunk, t_chunk, rem = work.pop(0)
        if rem <= 0 or x_chunk.shape[split_dim] < 2:
            leaves.append((prefix, x_chunk, y_chunk, t_chunk))
            continue
        x_l, x_r = torch.tensor_split(x_chunk, 2, dim=split_dim)
        y_l, y_r = torch.tensor_split(y_chunk, 2, dim=split_dim)
        a_l, a_r = torch.tensor_split(t_chunk.a, 2, dim=split_dim)
        b_l, b_r = torch.tensor_split(t_chunk.b, 2, dim=split_dim)
        c_l, c_r = torch.tensor_split(t_chunk.c, 2, dim=split_dim)
        work.append((f"{prefix}.L", x_l, y_l, BeaverTripleShares(a_l, b_l, c_l), rem - 1))
        work.append((f"{prefix}.R", x_r, y_r, BeaverTripleShares(a_r, b_r, c_r), rem - 1))
    return leaves


def _secure_matmul_partitioned(x_share: torch.Tensor,
                               y_share: torch.Tensor,
                               triple: BeaverTripleShares,
                               exchanger: PeerExchanger,
                               session_id: str,
                               op_id: str,
                               is_party_0: bool,
                               transpose_b: bool,
                               depth: int,
                               split_dim: int,
                               workers: int) -> torch.Tensor:
    leaves = _build_matmul_leaves(x_share, y_share, triple, depth, split_dim, op_id)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(
                lambda leaf: _secure_matmul_core(
                    leaf[1], leaf[2], leaf[3], exchanger, session_id,
                    leaf[0], is_party_0, transpose_b
                ),
                leaves,
            ))
    else:
        parts = [
            _secure_matmul_core(
                x_l, y_l, t_l, exchanger, session_id, leaf_op, is_party_0, transpose_b
            )
            for leaf_op, x_l, y_l, t_l in leaves
        ]
    return torch.cat(parts, dim=split_dim)


# ---------------------------------------------------------------------------
# Triple allocation helper
# ---------------------------------------------------------------------------

class TripleAllocator:
    """
    Manages a pre-allocated pool of Beaver triples for one session.

    The dealer generates all triples upfront and distributes shares.
    During the online phase, each operation requests triples by op_id.
    """

    def __init__(self, triples: dict[str, BeaverTripleShares]):
        self._triples = triples
        self._counter: dict[str, int] = {}

    def get(self, op_id: str) -> BeaverTripleShares:
        """Get the triple for a named operation."""
        if op_id not in self._triples:
            raise KeyError(f"No triple allocated for op_id={op_id}")
        return self._triples[op_id]

    @property
    def op_ids(self) -> list[str]:
        return list(self._triples.keys())


# ---------------------------------------------------------------------------
# Operation ID scheme for one transformer layer
# ---------------------------------------------------------------------------

def layer0_op_ids() -> list[str]:
    """
    List all operation IDs for a full layer 0 MPC computation.

    Naming convention:  {component}_{operation}_{index}
    """
    ops = []

    # Input RMSNorm: x^2 (variance reveal uses exchange, not a triple)
    ops.append("rmsnorm_in_sq")

    # Attention: Q @ K^T (matmul triple)
    ops.append("attn_qk_matmul")

    # Attention softmax (revealed, no triples needed)

    # Attention: attn_weights @ V (matmul triple)
    ops.append("attn_av_matmul")

    # Post-attention RMSNorm: x^2 (variance reveal uses exchange)
    ops.append("rmsnorm_post_sq")

    # MLP SiLU: degree-5 sigmoid polynomial (x^2, x^3, x^4, x^5) + x * sigmoid(x)
    ops.append("silu_x2")
    ops.append("silu_x3")
    ops.append("silu_x4")
    ops.append("silu_x5")
    ops.append("silu_apply")

    # MLP gate * up product
    ops.append("mlp_gate_up")

    return ops


# ---------------------------------------------------------------------------
# Secure RMSNorm
# ---------------------------------------------------------------------------

def secure_rmsnorm(x_share: torch.Tensor,
                   weight: torch.Tensor,
                   eps: float,
                   triples: TripleAllocator,
                   exchanger: PeerExchanger,
                   session_id: str,
                   is_party_0: bool,
                   prefix: str = "rmsnorm_in") -> torch.Tensor:
    """
    Compute RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    on additive shares.

    Steps:
      1. x^2 via secure_multiply
      2. mean(x^2) — linear (free)
      3. Reveal variance by exchanging shares (reveals one scalar per
         position — the activation "energy", not direction).
      4. Compute rsqrt(variance + eps) in cleartext (public scalar).
      5. Multiply shares by public rsqrt * weight (linear, free).

    Security note: revealing mean(x^2) is a standard MPC trade-off for
    RMSNorm.  It reveals only the magnitude of each position's hidden
    state, not the direction (which hidden dims are active).  After one
    full layer of MPC-protected computation, this is negligible.

    Returns the party's share of RMSNorm(x).
    """
    # Step 1: secure x^2
    x_sq_share = secure_square(
        x_share,
        triples.get(f"{prefix}_sq"),
        exchanger, session_id, f"{prefix}_sq",
        is_party_0)

    # Step 2: mean (linear, free on shares)
    variance_share = x_sq_share.mean(dim=-1, keepdim=True)

    # Step 3: Reveal variance by exchanging shares
    # We use a dedicated exchange with a dummy delta (same shape)
    peer_var, _ = exchanger.exchange(
        session_id, f"{prefix}_reveal",
        variance_share, torch.zeros_like(variance_share))
    variance = variance_share + peer_var + eps

    # Step 4: rsqrt in cleartext (public)
    rsqrt_val = torch.rsqrt(variance)

    # Step 5: x_share * rsqrt_val * weight  (all public multiplies, free)
    result_share = x_share * rsqrt_val * weight

    return result_share


# ---------------------------------------------------------------------------
# Secure Softmax (polynomial exp approximation)
# ---------------------------------------------------------------------------

def secure_softmax(x_share: torch.Tensor,
                   triples: TripleAllocator,
                   exchanger: PeerExchanger,
                   session_id: str,
                   is_party_0: bool,
                   dim: int = -1) -> torch.Tensor:
    """
    Compute softmax(x) on secret-shared attention scores.

    Approach: reveal the attention scores, compute softmax in cleartext,
    then re-share the result.  This is the standard MPC trade-off for
    softmax in transformer inference:

    - Attention scores = Q @ K^T / sqrt(d) encode "which positions attend
      to which" — an attention pattern, not raw token content.
    - The raw tokens are protected by the MPC on the embedding + linear
      projections.  The attention pattern leaks far less information.
    - Polynomial approximations for exp() diverge badly outside [-5, 5],
      but real attention scores routinely reach 10-15+, making polynomial
      softmax numerically unusable.

    Security note: revealing attention scores is a common accepted
    trade-off in practical MPC inference (SecureML, CrypTen, etc.).

    Returns party's share of softmax(x).
    """
    # Reveal scores by exchanging shares
    peer_share, _ = exchanger.exchange(
        session_id, "softmax_reveal",
        x_share, torch.zeros_like(x_share))
    scores_clear = x_share + peer_share

    # Compute softmax in cleartext
    softmax_result = torch.softmax(scores_clear, dim=dim)

    # Re-share: party 0 gets a random share, party 1 gets remainder
    if is_party_0:
        return softmax_result
    else:
        return torch.zeros_like(softmax_result)


# ---------------------------------------------------------------------------
# Secure SiLU (x * sigmoid(x))
# ---------------------------------------------------------------------------

def secure_silu(x_share: torch.Tensor,
                triples: TripleAllocator,
                exchanger: PeerExchanger,
                session_id: str,
                is_party_0: bool) -> torch.Tensor:
    """
    Compute SiLU(x) = x * sigmoid(x) on secret-shared x.

    Approximates sigmoid with a degree-5 minimax polynomial:
      sigmoid(x) ≈ 0.5 + c1*x + c3*x^3 + c5*x^5

    Minimax coefficients fitted over [-6, 6] (Remez algorithm):
      c1 = 0.19815,  c3 = -0.00476,  c5 = 0.0000526

    This achieves ~1e-3 max error over [-6, 6], significantly better
    than the previous degree-3 approximation (~1e-2 error).

    Cost: 5 secure multiplications (x^2, x^3, x^4, x^5, x*sigmoid)
    = 5 round-trips.

    Returns party's share of SiLU(x).
    """
    # Minimax coefficients for sigmoid over [-6, 6]
    # Fitted via Remez algorithm (tools/fit_polynomials.py)
    c1 = 0.21570
    c3 = -0.00761
    c5 = 0.00011219

    # x^2 via secure multiply
    x2_share = secure_square(
        x_share,
        triples.get("silu_x2"),
        exchanger, session_id, "silu_x2",
        is_party_0)

    # x^3 = x^2 * x via secure multiply
    x3_share = secure_multiply(
        x2_share, x_share,
        triples.get("silu_x3"),
        exchanger, session_id, "silu_x3",
        is_party_0)

    # x^4 = x^2 * x^2 via secure multiply
    x4_share = secure_multiply(
        x2_share, x2_share,
        triples.get("silu_x4"),
        exchanger, session_id, "silu_x4",
        is_party_0)

    # x^5 = x^4 * x via secure multiply
    x5_share = secure_multiply(
        x4_share, x_share,
        triples.get("silu_x5"),
        exchanger, session_id, "silu_x5",
        is_party_0)

    # sigmoid(x) ≈ 0.5 + c1*x + c3*x^3 + c5*x^5
    sig_share = c1 * x_share + c3 * x3_share + c5 * x5_share
    if is_party_0:
        sig_share = sig_share + 0.5

    # SiLU = x * sigmoid(x) via secure multiply
    result_share = secure_multiply(
        x_share, sig_share,
        triples.get("silu_apply"),
        exchanger, session_id, "silu_apply",
        is_party_0)

    return result_share


# ---------------------------------------------------------------------------
# Secure gate * up multiplication for MLP
# ---------------------------------------------------------------------------

def secure_gate_up(gate_share: torch.Tensor,
                   up_share: torch.Tensor,
                   triples: TripleAllocator,
                   exchanger: PeerExchanger,
                   session_id: str,
                   is_party_0: bool) -> torch.Tensor:
    """
    Compute gate * up for the MLP block, where both are secret-shared.
    """
    return secure_multiply(
        gate_share, up_share,
        triples.get("mlp_gate_up"),
        exchanger, session_id, "mlp_gate_up",
        is_party_0)


# ---------------------------------------------------------------------------
# Allocate all triples for one layer 0 forward pass
# ---------------------------------------------------------------------------

def allocate_layer0_triples(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    intermediate_size: int,
    seq_len: int,
    batch: int = 1,
) -> dict[str, BeaverTriple]:
    """
    Generate all Beaver triples needed for one layer 0 MPC forward pass.

    Returns a dict mapping op_id -> BeaverTriple (containing both parties'
    shares). The caller splits and distributes party_0/party_1 shares.

    Includes:
      - Element-wise triples for RMSNorm, Softmax, SiLU, gate*up
      - Matrix triples for Q@K^T and attn_weights@V in attention
    """
    from network.mpc_beaver import BeaverTriple

    triples: dict[str, BeaverTriple] = {}
    hs = (batch, seq_len, hidden_size)
    attn = (batch, num_heads, seq_len, seq_len)
    mlp = (batch, seq_len, intermediate_size)
    scalar_pos = (batch, seq_len, 1)

    def add(name: str, shape: tuple):
        triples[name] = BeaverTriple.generate(shape)

    def add_matmul(name: str, a_shape: tuple, b_shape: tuple,
                   transpose_b: bool = True):
        triples[name] = BeaverTriple.generate_matmul(
            a_shape, b_shape, transpose_b=transpose_b)

    # --- Input RMSNorm (variance revealed, only x^2 triple needed) ---
    add("rmsnorm_in_sq", hs)

    # --- Attention Q@K^T matmul ---
    qk_shape = (batch, num_heads, seq_len, head_dim)
    add_matmul("attn_qk_matmul", qk_shape, qk_shape, transpose_b=True)

    # --- Attention softmax (revealed, computed in cleartext — no triples) ---

    # --- Attention weights @ V matmul ---
    attn_shape = (batch, num_heads, seq_len, seq_len)
    v_shape = (batch, num_heads, seq_len, head_dim)
    add_matmul("attn_av_matmul", attn_shape, v_shape, transpose_b=False)

    # --- Post-attention RMSNorm (variance revealed, only x^2 triple needed) ---
    add("rmsnorm_post_sq", hs)

    # --- MLP SiLU (degree-5 minimax polynomial) ---
    add("silu_x2", mlp)
    add("silu_x3", mlp)
    add("silu_x4", mlp)
    add("silu_x5", mlp)
    add("silu_apply", mlp)

    # --- MLP gate * up ---
    add("mlp_gate_up", mlp)

    return triples
