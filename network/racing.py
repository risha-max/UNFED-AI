"""
Per-Shard Racing — fault-tolerant inference via redundant parallel computation.

Instead of a single pipeline (shard 0 → 1 → 2 → 3), the client orchestrates
each shard hop and races N replica nodes in parallel at each step.

How it works:
  1. Client discovers N nodes per shard from the registry
  2. For each shard hop, the client sends the input to all N replicas
  3. The first response wins (concurrent.futures.FIRST_COMPLETED)
  4. All replicas process every token to keep their KV caches in sync
  5. When the slower response arrives, its hash is compared for verification

Fault tolerance:
  - If a node disconnects, the other racer(s) still have valid KV caches
  - The client seamlessly uses whichever node responds
  - A single node failure only costs one racer, not the whole pipeline

Free verification:
  - When 2+ racers return, their output hashes are compared
  - A mismatch flags potential fraud (analogous to the voting system)

Cost: Nx compute per shard (N = number of replicas), zero added latency
      (the fastest racer determines speed).
"""

import hashlib
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, Future
from dataclasses import dataclass, field
from typing import Optional

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
from network.resilience import create_resilient_channel


@dataclass
class RaceResult:
    """Result of racing replicas for one shard hop."""
    shard_index: int
    winner_address: str
    response: inference_pb2.ForwardResponse
    latency_ms: float
    verification_match: Optional[bool] = None  # None if only 1 racer returned
    mismatch_address: Optional[str] = None      # address of the mismatching racer


class RacingCoordinator:
    """
    Orchestrates per-shard racing for fault-tolerant inference.

    The coordinator:
      1. Builds a racing circuit (N nodes per shard)
      2. Races replicas at each shard hop
      3. Background-verifies slower results when they arrive
    """

    def __init__(self, replicas: int = None, timeout: float = None):
        self.replicas = replicas or config.RACING_REPLICAS
        self.timeout = timeout or config.SHARD_TIMEOUT_SECONDS
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.replicas * 4)
        self._mismatches: list[dict] = []  # log of detected mismatches

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        """Get or create a gRPC stub for a compute node."""
        if address not in self._stubs:
            channel = create_resilient_channel(address, config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def _forward_to_node(self, address: str, request: inference_pb2.ForwardRequest,
                         ) -> inference_pb2.ForwardResponse:
        """Send a ForwardRequest to a node. Returns the ForwardResponse."""
        stub = self._get_stub(address)
        return stub.Forward(request)

    def race_shard(self, shard_index: int, node_addresses: list[str],
                   request: inference_pb2.ForwardRequest,
                   guard_address: str = None) -> RaceResult:
        """Race multiple replicas for a single shard hop.

        Sends the same ForwardRequest to all nodes in parallel.
        Returns the first successful response. When slower responses
        arrive, compares their output hash for verification.
        """
        import time
        start = time.time()

        future_to_addr: dict[Future, str] = {}
        for addr in node_addresses[:self.replicas]:
            future = self._executor.submit(
                self._forward_to_node, addr, request
            )
            future_to_addr[future] = addr

        # Wait for the first successful response
        winner_response = None
        winner_address = None
        pending = set(future_to_addr.keys())
        errors = []

        while pending:
            done, pending = wait(pending, timeout=self.timeout,
                                 return_when=FIRST_COMPLETED)

            if not done and not winner_response:
                # Timeout: no response within the deadline
                break

            for future in done:
                addr = future_to_addr[future]
                try:
                    response = future.result()

                    if winner_response is None:
                        # First successful response — this is the winner
                        winner_response = response
                        winner_address = addr
                        latency = (time.time() - start) * 1000
                    else:
                        # Slower response — verify against winner
                        self._verify_responses(
                            shard_index, winner_address, winner_response,
                            addr, response)

                except grpc.RpcError as e:
                    errors.append((addr, str(e)))
                except Exception as e:
                    errors.append((addr, str(e)))

            # If we have a winner, don't block on the remaining futures
            # (they'll be verified in background when they complete)
            if winner_response is not None and pending:
                # Schedule background verification for remaining futures
                self._schedule_background_verification(
                    shard_index, winner_address, winner_response,
                    pending, future_to_addr)
                break

        if winner_response is None:
            if errors:
                error_details = "; ".join(f"{a}: {e}" for a, e in errors)
                raise RuntimeError(
                    f"All {len(node_addresses)} replicas for shard {shard_index} "
                    f"failed: {error_details}")
            raise TimeoutError(
                f"All {len(node_addresses)} replicas for shard {shard_index} "
                f"timed out after {self.timeout}s")

        result = RaceResult(
            shard_index=shard_index,
            winner_address=winner_address,
            response=winner_response,
            latency_ms=latency,
        )

        return result

    def _compute_response_hash(self, response: inference_pb2.ForwardResponse) -> str:
        """Compute a hash of a ForwardResponse for verification."""
        h = hashlib.sha256()
        if response.activation_data:
            h.update(response.activation_data)
        if response.has_token:
            h.update(response.token_id.to_bytes(4, 'big', signed=True))
        return h.hexdigest()

    def _verify_responses(self, shard_index: int,
                          winner_addr: str, winner_resp: inference_pb2.ForwardResponse,
                          other_addr: str, other_resp: inference_pb2.ForwardResponse):
        """Compare two responses and log any mismatches."""
        winner_hash = self._compute_response_hash(winner_resp)
        other_hash = self._compute_response_hash(other_resp)

        if winner_hash == other_hash:
            print(f"[Racing] Shard {shard_index}: VERIFIED "
                  f"({winner_hash[:12]}... == {other_hash[:12]}...)")
        else:
            mismatch = {
                "shard_index": shard_index,
                "winner": winner_addr,
                "other": other_addr,
                "winner_hash": winner_hash,
                "other_hash": other_hash,
            }
            self._mismatches.append(mismatch)
            print(f"[Racing] Shard {shard_index}: MISMATCH "
                  f"winner={winner_addr} ({winner_hash[:12]}...) "
                  f"vs {other_addr} ({other_hash[:12]}...)")

    def _schedule_background_verification(
            self, shard_index: int,
            winner_addr: str, winner_resp: inference_pb2.ForwardResponse,
            pending_futures: set[Future],
            future_to_addr: dict[Future, str]):
        """
        Schedule background verification for pending futures.

        When a slower racer finishes, compare its result to the winner's.
        """
        def _verify_when_done(future: Future, addr: str):
            try:
                response = future.result(timeout=self.timeout)
                self._verify_responses(
                    shard_index, winner_addr, winner_resp,
                    addr, response)
            except Exception:
                pass  # Node failed — not a verification issue

        for future in pending_futures:
            addr = future_to_addr[future]
            # Use a separate thread so we don't block
            self._executor.submit(_verify_when_done, future, addr)

    @property
    def mismatches(self) -> list[dict]:
        """Return all detected mismatches."""
        return list(self._mismatches)

    def close(self):
        """Shut down the executor."""
        self._executor.shutdown(wait=False)
