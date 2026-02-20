"""
Computational Randomness Routing — next node determined by hash(activation_output).

How it works:
  1. When a node finishes its layers, it computes hash(output_activation).
  2. It commits to this hash (sends it to the caller) BEFORE revealing the output.
  3. The hash determines which node handles the next shard:
       next_node_index = int(hash_hex[:16], 16) % num_candidates_for_next_shard
  4. The node then forwards the activation to the chosen next node.

This prevents:
  - Central authorities from controlling routing (it's determined by computation)
  - Adversaries from positioning nodes to intercept specific queries
  - Nodes from manipulating routing (the commit-reveal ensures they can't
    change the output after seeing which node would be selected)

Constraints:
  - Within a session, the circuit is built incrementally (node by node, hop by hop)
  - Each hop is random but the KV cache constraint means subsequent tokens in the
    same session MUST go to the same node (since it holds the session's KV cache)
  - So: first token → random routing, subsequent tokens → same circuit

Commit-Reveal Protocol:
  Phase 1 (Commit): Node hashes its output, sends commitment to caller
  Phase 2 (Reveal): Node sends the actual output
  Verification: Caller can verify hash(revealed_output) == commitment
"""

import hashlib
import torch


def compute_activation_hash(activation: torch.Tensor) -> str:
    """
    Compute a deterministic hash of an activation tensor.

    The hash is computed over the raw float32 bytes of the tensor.
    This hash is used for:
      1. Commit-reveal protocol (proving the node didn't tamper with output)
      2. Random routing (selecting the next node deterministically)
    """
    # Ensure consistent dtype and layout
    data = activation.contiguous().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def select_next_node(activation_hash: str, num_candidates: int) -> int:
    """
    Select the next node index based on the activation hash.

    Uses the first 16 hex chars (64 bits) of the hash as an integer,
    then takes modulo of the number of candidates.

    Args:
        activation_hash: SHA256 hex digest of the activation tensor
        num_candidates: Number of available nodes for the next shard

    Returns:
        Index (0-based) into the candidate list for the next shard
    """
    hash_int = int(activation_hash[:16], 16)
    return hash_int % num_candidates


def verify_commitment(activation: torch.Tensor, commitment: str) -> bool:
    """
    Verify that an activation tensor matches a previously committed hash.

    Called by the receiving node to verify the sender didn't tamper
    with the output after making the commitment.

    Args:
        activation: The revealed activation tensor
        commitment: The hash commitment made before reveal

    Returns:
        True if the activation matches the commitment
    """
    actual_hash = compute_activation_hash(activation)
    return actual_hash == commitment


class SessionCircuit:
    """
    Tracks the dynamically-built circuit for a session.

    After the first token establishes the circuit via computational randomness,
    subsequent tokens reuse the same circuit (because KV caches are on those nodes).
    """

    def __init__(self):
        # Maps shard_index -> chosen node address
        self._circuit: dict[int, str] = {}

    def has_route(self, shard_index: int) -> bool:
        """Check if a route has been established for a shard."""
        return shard_index in self._circuit

    def get_route(self, shard_index: int) -> str | None:
        """Get the established route for a shard."""
        return self._circuit.get(shard_index)

    def set_route(self, shard_index: int, address: str):
        """Record the chosen route for a shard."""
        self._circuit[shard_index] = address

    def get_full_circuit(self) -> list[tuple[int, str]]:
        """Get the full established circuit as [(shard_index, address), ...]."""
        return sorted(self._circuit.items())

    def is_complete(self, total_shards: int) -> bool:
        """Check if all shards have routes established."""
        return all(i in self._circuit for i in range(total_shards))
