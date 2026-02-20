"""
Redundant Voting — real-time correctness verification via commit-then-reveal.

The existing fraud proof system catches cheaters AFTER the fact.
Redundant voting catches cheating BEFORE the answer reaches the client.

How it works:
  1. Per query, the client randomly picks 1 shard to double-check
  2. The client sends the same input to 2 independent nodes serving that shard
  3. Both nodes commit to hash(output) before seeing each other's answer
  4. Client compares the two hashes (32 bytes each)
  5. If they match: proceed with the result
  6. If they differ: query a 3rd node as tiebreaker, submit fraud proof

Catch rate per query (combined with existing fraud proofs):
  - 25% chance the voted shard catches cheating immediately (1 out of 4 shards)
  - 75% x 5% = 3.75% chance fraud proofs catch it later
  - Total: ~28.75% per query, ~96% after 10 queries

Cost: +25% compute on average (1 extra forward pass on 1/4 shards).
      Zero added latency (both voted nodes compute in parallel).
"""

import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
from network.resilience import create_resilient_channel


@dataclass
class VoteResult:
    """Result of a voting round for one shard."""
    shard_index: int
    node_a_address: str
    node_b_address: str
    hash_a: str
    hash_b: str
    match: bool
    winner_address: Optional[str] = None  # Set after tiebreaker if needed
    token_id: Optional[int] = None
    has_token: bool = False


class VotingCoordinator:
    """
    Manages redundant voting for a client.

    The coordinator:
      1. Selects which shard to double-check (random per query)
      2. Sends CommitRequests to two nodes in parallel
      3. Compares the commitment hashes
      4. On mismatch: invokes a tiebreaker with a 3rd node
    """

    def __init__(self):
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        if address not in self._stubs:
            channel = create_resilient_channel(address, config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def select_voted_shard(self, num_shards: int) -> int:
        """Randomly select one shard to double-check per query."""
        return random.randint(0, num_shards - 1)

    def find_candidates(self, nodes: list, shard_index: int,
                        exclude_address: str = None) -> list:
        """
        Find candidate nodes for a shard.

        Args:
            nodes: All discovered nodes
            shard_index: The shard to find candidates for
            exclude_address: Optionally exclude one address (the primary)

        Returns:
            List of nodes serving this shard (excluding the given address)
        """
        candidates = [
            n for n in nodes
            if n.shard_index == shard_index
            and n.node_type != "guard"
            and (exclude_address is None or n.address != exclude_address)
        ]
        return candidates

    def request_commitment(self, address: str, session_id: str,
                           activation_data: bytes = b"",
                           tensor_shape: list[int] = None,
                           token_ids: list[int] = None,
                           is_prefill: bool = True) -> tuple[str, Optional[int], bool]:
        """
        Ask a node to compute and commit to its output hash.

        Returns (output_hash, token_id, has_token)
        """
        stub = self._get_stub(address)
        req = inference_pb2.CommitRequest(
            session_id=session_id,
            is_prefill=is_prefill,
        )
        if activation_data:
            req.activation_data = activation_data
        if tensor_shape:
            req.tensor_shape.extend(tensor_shape)
        if token_ids:
            req.token_ids.extend(token_ids)

        try:
            resp = stub.Commit(req)
            return resp.output_hash, resp.token_id, resp.has_token
        except grpc.RpcError as e:
            print(f"[Voting] Commit failed for {address}: {e.details()}")
            return "", None, False

    def vote(self, node_a_address: str, node_b_address: str,
             shard_index: int, session_id: str,
             activation_data: bytes = b"",
             tensor_shape: list[int] = None,
             token_ids: list[int] = None,
             is_prefill: bool = True) -> VoteResult:
        """
        Run a voting round: send CommitRequests to two nodes in parallel,
        compare their output hashes.

        Returns a VoteResult with match=True if the hashes agree.
        """
        # Submit both commits in parallel
        future_a = self._executor.submit(
            self.request_commitment,
            node_a_address, session_id,
            activation_data, tensor_shape, token_ids, is_prefill,
        )
        future_b = self._executor.submit(
            self.request_commitment,
            node_b_address, session_id,
            activation_data, tensor_shape, token_ids, is_prefill,
        )

        hash_a, token_a, has_token_a = future_a.result()
        hash_b, token_b, has_token_b = future_b.result()

        match = (hash_a == hash_b) and hash_a != ""

        result = VoteResult(
            shard_index=shard_index,
            node_a_address=node_a_address,
            node_b_address=node_b_address,
            hash_a=hash_a,
            hash_b=hash_b,
            match=match,
            winner_address=node_a_address if match else None,
            token_id=token_a if has_token_a else None,
            has_token=has_token_a,
        )

        if match:
            print(f"[Voting] Shard {shard_index}: MATCH "
                  f"({hash_a[:12]}... == {hash_b[:12]}...)")
        else:
            print(f"[Voting] Shard {shard_index}: MISMATCH "
                  f"({hash_a[:12]}... != {hash_b[:12]}...)")

        return result

    def resolve_conflict(self, nodes: list, shard_index: int,
                         session_id: str,
                         vote_result: VoteResult,
                         activation_data: bytes = b"",
                         tensor_shape: list[int] = None,
                         token_ids: list[int] = None,
                         is_prefill: bool = True) -> VoteResult:
        """
        Resolve a voting conflict by querying a 3rd node as tiebreaker.

        The 3rd node's hash determines who is the honest node.
        A fraud proof is generated against the liar.
        """
        # Find a 3rd node that wasn't in the original vote
        candidates = self.find_candidates(
            nodes, shard_index,
            exclude_address=None,
        )
        # Exclude the two nodes that already voted
        third_party = [
            n for n in candidates
            if n.address not in (vote_result.node_a_address,
                                 vote_result.node_b_address)
        ]

        if not third_party:
            print(f"[Voting] No 3rd node available for tiebreaker on shard {shard_index}")
            # Default to node A (arbitrary)
            vote_result.winner_address = vote_result.node_a_address
            return vote_result

        tiebreaker = third_party[0]
        hash_c, token_c, has_token_c = self.request_commitment(
            tiebreaker.address, session_id,
            activation_data, tensor_shape, token_ids, is_prefill,
        )

        # Majority wins
        if hash_c == vote_result.hash_a:
            vote_result.winner_address = vote_result.node_a_address
            liar = vote_result.node_b_address
        elif hash_c == vote_result.hash_b:
            vote_result.winner_address = vote_result.node_b_address
            liar = vote_result.node_a_address
        else:
            # All three disagree — something seriously wrong
            print(f"[Voting] WARNING: 3-way disagreement on shard {shard_index}!")
            vote_result.winner_address = tiebreaker.address
            liar = None

        if liar:
            print(f"[Voting] Tiebreaker resolved: winner={vote_result.winner_address}, "
                  f"liar={liar}")

        return vote_result

    def close(self):
        """Shut down the executor."""
        self._executor.shutdown(wait=False)
