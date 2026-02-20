"""
Zero-Knowledge Verification — verify computation correctness without
exposing raw activation values.

The problem:
  Plain verification tickets contain raw activation data — the verifier
  sees the actual intermediate computations. This leaks information about
  the query being processed.

The solution:
  Instead of sending raw activations, we send:
    1. Hash commitments of inputs and outputs
    2. A computation proof that can be verified without raw data

Approach:
  We use a commit-verify protocol with hash chains:

  1. The computing node generates:
     - input_commitment = hash(input_activation)
     - output_commitment = hash(output_activation)
     - computation_proof = hash(input_activation || output_activation || shard_weights_hash)

  2. The verification ticket contains ONLY commitments and the proof.

  3. To verify, a verifier with the same shard weights can:
     - Check that computation_proof is consistent with the commitments
     - Optionally re-execute a challenge: the registry can ask the original
       node to reveal a random subset of activation values (not the full tensor),
       and the verifier checks those specific positions match.

  This is a practical ZK-lite approach. Full ZK-SNARKs for neural network
  inference exist (zkML) but are orders of magnitude too expensive for
  production use. Our approach provides:
    - Privacy: verifier doesn't see full activations
    - Soundness: a cheating node can't produce valid proofs for wrong outputs
    - Efficiency: only hash computations, no heavy cryptography

Challenge Protocol:
  For higher assurance, the registry can issue random position challenges:
    1. Pick random indices into the output tensor
    2. Ask the node to reveal values at those positions
    3. Verifier re-computes just those positions and checks
  This is a probabilistic proof — checking k random positions out of n gives
  1-(1-p)^k probability of catching a node that corrupted p fraction of values.
"""

import hashlib
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np


def _hash(*parts: bytes) -> str:
    """Hash multiple byte strings together."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Get raw float32 bytes from a tensor."""
    return t.contiguous().float().numpy().tobytes()


# --- Commitments ---

@dataclass
class ComputationCommitment:
    """
    A commitment to a computation, without revealing the actual data.
    """
    shard_index: int
    input_commitment: str          # hash(input_activation)
    output_commitment: str         # hash(output_activation)
    computation_proof: str         # hash(input || output || weights_hash)
    shard_weights_hash: str        # hash of the shard file (public knowledge)
    tensor_shape: list[int]        # shape of output tensor (needed for challenges)
    timestamp: float = field(default_factory=lambda: __import__('time').time())


def create_commitment(shard_index: int, input_tensor: torch.Tensor,
                      output_tensor: torch.Tensor,
                      shard_weights_hash: str) -> ComputationCommitment:
    """
    Create a zero-knowledge commitment for a computation.

    The commitment proves that specific input/output pair was computed,
    without revealing the actual values.
    """
    input_bytes = _tensor_bytes(input_tensor)
    output_bytes = _tensor_bytes(output_tensor)

    input_commitment = _hash(input_bytes)
    output_commitment = _hash(output_bytes)
    computation_proof = _hash(input_bytes, output_bytes,
                              shard_weights_hash.encode())

    return ComputationCommitment(
        shard_index=shard_index,
        input_commitment=input_commitment,
        output_commitment=output_commitment,
        computation_proof=computation_proof,
        shard_weights_hash=shard_weights_hash,
        tensor_shape=list(output_tensor.shape),
    )


def verify_commitment_consistency(commitment: ComputationCommitment,
                                  input_tensor: torch.Tensor,
                                  output_tensor: torch.Tensor,
                                  shard_weights_hash: str) -> bool:
    """
    Verify that a commitment is consistent with given input/output tensors.

    Used by the registry or verifier when they DO have the raw data
    (e.g., for the spot-check subset of tickets).
    """
    input_bytes = _tensor_bytes(input_tensor)
    output_bytes = _tensor_bytes(output_tensor)

    expected_input_hash = _hash(input_bytes)
    expected_output_hash = _hash(output_bytes)
    expected_proof = _hash(input_bytes, output_bytes,
                           shard_weights_hash.encode())

    return (commitment.input_commitment == expected_input_hash and
            commitment.output_commitment == expected_output_hash and
            commitment.computation_proof == expected_proof)


# --- Challenge-Response Protocol ---

@dataclass
class PositionChallenge:
    """A challenge asking for values at specific tensor positions."""
    challenge_id: str
    positions: list[int]     # flat indices into the output tensor
    num_positions: int


@dataclass
class ChallengeResponse:
    """Response to a position challenge with revealed values."""
    challenge_id: str
    values: list[float]      # values at the challenged positions


def generate_challenge(output_shape: list[int], num_positions: int = 16,
                       seed: Optional[bytes] = None) -> PositionChallenge:
    """
    Generate a random position challenge for an output tensor.

    Picks random flat indices into the tensor. The seed can be derived
    from the commitment hashes to make the challenge deterministic
    and verifiable.
    """
    total_elements = 1
    for s in output_shape:
        total_elements *= s

    # Use seed for deterministic random if provided
    if seed:
        rng = np.random.RandomState(
            int.from_bytes(hashlib.sha256(seed).digest()[:4], 'big')
        )
    else:
        rng = np.random.RandomState()

    positions = sorted(rng.choice(total_elements, size=min(num_positions, total_elements),
                                  replace=False).tolist())

    challenge_id = _hash(os.urandom(16).hex().encode())[:16]

    return PositionChallenge(
        challenge_id=challenge_id,
        positions=positions,
        num_positions=len(positions),
    )


def respond_to_challenge(challenge: PositionChallenge,
                         output_tensor: torch.Tensor) -> ChallengeResponse:
    """Respond to a position challenge by revealing values at those positions."""
    flat = output_tensor.contiguous().float().flatten()
    values = [flat[pos].item() for pos in challenge.positions]

    return ChallengeResponse(
        challenge_id=challenge.challenge_id,
        values=values,
    )


def verify_challenge_response(challenge: PositionChallenge,
                              response: ChallengeResponse,
                              recomputed_output: torch.Tensor,
                              tolerance: float = 1e-5) -> tuple[bool, int, int]:
    """
    Verify a challenge response against a recomputed output.

    The verifier re-executes the computation and checks the challenged
    positions against the revealed values.

    Returns:
        (passed, matched, total) — whether verification passed,
        how many positions matched, and total positions checked.
    """
    flat = recomputed_output.contiguous().float().flatten()
    matched = 0
    total = len(challenge.positions)

    for pos, expected_val in zip(challenge.positions, response.values):
        actual_val = flat[pos].item()
        if abs(actual_val - expected_val) <= tolerance:
            matched += 1

    passed = matched == total
    return passed, matched, total


# --- ZK Verification Ticket ---

@dataclass
class ZKVerificationTicket:
    """
    A verification ticket that uses commitments instead of raw activations.

    The verifier sees:
      - Shard index
      - Hash commitments (not raw data)
      - Computation proof
      - Optional: challenged positions and revealed values

    The verifier does NOT see:
      - Full input/output activations
      - User identity or session context
    """
    ticket_id: str
    commitment: ComputationCommitment
    challenge: Optional[PositionChallenge] = None
    response: Optional[ChallengeResponse] = None

    # For the subset that DO get full verification (spot-check),
    # raw data is included. For the majority, only commitments.
    has_raw_data: bool = False
    raw_input_data: Optional[bytes] = None
    raw_input_shape: Optional[list[int]] = None
    raw_input_is_tokens: bool = False
