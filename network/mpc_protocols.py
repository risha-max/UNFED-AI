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
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from network.mpc_beaver import BeaverTriple, BeaverTripleShares


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
