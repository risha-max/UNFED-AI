"""
Verification System — spot-check verifier nodes that randomly re-execute
computations to ensure honest behavior.

How it works:
  1. A Verifier node runs in the background, periodically selecting random
     computations to re-verify.
  2. Verification tickets are context-free: the verifier receives:
       - Input activation tensor (or token IDs for shard 0)
       - Expected output activation tensor (or token)
       - Shard index (which shard to re-execute)
     But NOT: user identity, session ID, query ID, or which node computed it.
  3. The verifier loads the same shard weights and re-runs the forward pass.
  4. If the result doesn't match, a fraud proof is generated.

Sampling strategy:
  - ~5% of forward passes are selected for verification
  - Selection is random per-token (independent of the node being verified)
  - Verification is done asynchronously (doesn't block inference)

Fraud proofs:
  - A fraud proof contains: input, expected_output, actual_recomputed_output, shard_index
  - This is published to the network for anyone to verify
  - The offending node gets penalized (stake slashing in Phase 5)
"""

import hashlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# --- Verification Ticket ---

@dataclass
class VerificationTicket:
    """
    A context-free verification ticket.

    Contains only what's needed to re-execute and verify one computation.
    No user identity, session ID, or node identity.
    """
    ticket_id: str              # Unique ID for this ticket
    shard_index: int            # Which shard to verify
    input_data: bytes           # Serialized input (activation or token IDs)
    input_shape: list[int]      # Shape of input tensor (empty if token IDs)
    input_is_tokens: bool       # True if input_data is token IDs, not activations
    expected_output_data: bytes  # Serialized expected output
    expected_output_shape: list[int]  # Shape of expected output tensor
    expected_token: Optional[int]     # Expected sampled token (for last shard)
    timestamp: float = field(default_factory=time.time)


@dataclass
class FraudProof:
    """
    Evidence that a node returned incorrect output for a computation.
    """
    ticket_id: str
    shard_index: int
    input_hash: str              # SHA256 of input
    expected_output_hash: str    # SHA256 of what was claimed
    actual_output_hash: str      # SHA256 of what the verifier got
    expected_token: Optional[int]
    actual_token: Optional[int]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "shard_index": self.shard_index,
            "input_hash": self.input_hash,
            "expected_output_hash": self.expected_output_hash,
            "actual_output_hash": self.actual_output_hash,
            "expected_token": self.expected_token,
            "actual_token": self.actual_token,
            "timestamp": self.timestamp,
        }


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Ticket Collector (runs on each node, samples and emits tickets) ---

class TicketCollector:
    """
    Collects verification tickets by sampling forward passes on a node.

    Runs as a hook in the node's Forward handler. When a forward pass is
    selected for verification (~5% probability), it captures the input and
    output and creates a VerificationTicket.
    """

    def __init__(self, sampling_rate: float = 0.05):
        self.sampling_rate = sampling_rate
        self._pending_tickets: list[VerificationTicket] = []
        self._lock = threading.Lock()
        self._ticket_count = 0

    def should_sample(self) -> bool:
        """Decide whether to sample this forward pass."""
        return random.random() < self.sampling_rate

    def collect(self, shard_index: int, input_tensor: torch.Tensor,
                output_tensor: torch.Tensor, input_is_tokens: bool = False,
                sampled_token: Optional[int] = None,
                is_prefill: bool = True) -> Optional[VerificationTicket]:
        """
        Collect a verification ticket for a forward pass.

        Only collects tickets for prefill (first token) steps, where the KV cache
        is empty. This ensures the verifier can reproduce the computation exactly
        without needing the accumulated cache state.

        Called after a successful forward pass on a node.
        """
        if not is_prefill:
            return None  # Only verify prefill steps (stateless, reproducible)

        import numpy as np

        self._ticket_count += 1
        ticket_id = f"vt-{self._ticket_count}-{_hash_bytes(os.urandom(8))[:8]}"

        if input_is_tokens:
            input_data = input_tensor.numpy().tobytes()
            input_shape = []
        else:
            input_data = input_tensor.contiguous().float().numpy().tobytes()
            input_shape = list(input_tensor.shape)

        output_data = output_tensor.contiguous().float().numpy().tobytes()
        output_shape = list(output_tensor.shape)

        ticket = VerificationTicket(
            ticket_id=ticket_id,
            shard_index=shard_index,
            input_data=input_data,
            input_shape=input_shape,
            input_is_tokens=input_is_tokens,
            expected_output_data=output_data,
            expected_output_shape=output_shape,
            expected_token=sampled_token,
        )

        with self._lock:
            self._pending_tickets.append(ticket)

        return ticket

    def drain_tickets(self) -> list[VerificationTicket]:
        """Get and clear all pending tickets."""
        with self._lock:
            tickets = self._pending_tickets
            self._pending_tickets = []
        return tickets


# --- Verifier Node ---

class Verifier:
    """
    Verifier node that re-executes computations from verification tickets.

    Loads shard weights and re-runs the forward pass to verify correctness.
    """

    def __init__(self, shards_dir: str = None):
        self._runners: dict = {}  # shard_index -> runner (any type)
        self._fraud_proofs: list[FraudProof] = []
        self._verified_count = 0
        self._failed_count = 0
        self._lock = threading.Lock()
        self._shards_dir = shards_dir

    def _get_runner(self, shard_index: int):
        """Get or create a runner for verification."""
        if shard_index not in self._runners:
            manifest_path = config.MANIFEST_PATH
            if self._shards_dir:
                manifest_path = os.path.join(self._shards_dir, "manifest.json")

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Resolve shard info
            if "text_shards" in manifest and manifest["text_shards"]:
                shard_info = manifest["text_shards"][shard_index]
            else:
                shard_info = manifest["shards"][shard_index]

            arch_config = manifest.get("architecture", {}).get("text")
            if not arch_config:
                raise ValueError(
                    f"Manifest has no 'architecture.text' config — "
                    f"only v2+ manifests are supported. Re-split the model."
                )

            from node.runtime.generic_runner import GenericTextRunner
            shard_file = shard_info.get("file", f"text_shard_{shard_index}.pt")
            base_dir = self._shards_dir or os.path.dirname(manifest_path)
            shard_path = os.path.join(base_dir, shard_file)
            image_token_id = manifest.get("image_token_id")

            self._runners[shard_index] = GenericTextRunner(
                config=arch_config,
                shard_info=shard_info,
                shard_path=shard_path,
                image_token_id=image_token_id,
            )
            print(f"[Verifier] Loaded shard {shard_index} (generic runtime)")
        return self._runners[shard_index]

    @torch.no_grad()
    def verify_ticket(self, ticket: VerificationTicket) -> Optional[FraudProof]:
        """
        Verify a single ticket by re-executing the computation.

        Returns a FraudProof if the computation doesn't match, or None if it's valid.
        """
        import numpy as np

        runner = self._get_runner(ticket.shard_index)

        # Reconstruct input
        if ticket.input_is_tokens:
            token_ids = torch.from_numpy(
                np.frombuffer(ticket.input_data, dtype=np.int64).copy()
            ).unsqueeze(0)
            hidden, sampled_token = runner.forward(
                token_ids=token_ids,
                session_id=f"verify-{ticket.ticket_id}",
            )
        else:
            input_tensor = torch.from_numpy(
                np.frombuffer(ticket.input_data, dtype=np.float32)
                .reshape(ticket.input_shape).copy()
            )
            hidden, sampled_token = runner.forward(
                hidden_states=input_tensor,
                session_id=f"verify-{ticket.ticket_id}",
            )

        # Clean up verification session's KV cache
        runner.clear_session(f"verify-{ticket.ticket_id}")

        # Compare output
        actual_output_data = hidden.contiguous().float().numpy().tobytes()
        expected_hash = _hash_bytes(ticket.expected_output_data)
        actual_hash = _hash_bytes(actual_output_data)

        self._verified_count += 1

        if actual_hash != expected_hash:
            # Token mismatch for last shard
            fraud = FraudProof(
                ticket_id=ticket.ticket_id,
                shard_index=ticket.shard_index,
                input_hash=_hash_bytes(ticket.input_data),
                expected_output_hash=expected_hash,
                actual_output_hash=actual_hash,
                expected_token=ticket.expected_token,
                actual_token=sampled_token,
            )
            with self._lock:
                self._fraud_proofs.append(fraud)
            self._failed_count += 1
            print(f"[Verifier] FRAUD DETECTED on shard {ticket.shard_index}! "
                  f"Ticket: {ticket.ticket_id}")
            return fraud

        return None

    def verify_batch(self, tickets: list[VerificationTicket]) -> list[FraudProof]:
        """Verify a batch of tickets, return any fraud proofs found."""
        proofs = []
        for ticket in tickets:
            proof = self.verify_ticket(ticket)
            if proof:
                proofs.append(proof)
        return proofs

    def get_stats(self) -> dict:
        return {
            "verified": self._verified_count,
            "failed": self._failed_count,
            "fraud_proofs": len(self._fraud_proofs),
        }

    def get_fraud_proofs(self) -> list[FraudProof]:
        with self._lock:
            return list(self._fraud_proofs)
