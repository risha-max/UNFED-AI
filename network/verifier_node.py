"""
Verifier Node — standalone process that pulls verification tickets from
the registry, re-executes the computations, and reports fraud proofs.

The verifier:
  1. Polls the registry for pending verification tickets
  2. Loads the required shard weights (lazily, on first use)
  3. Re-runs the forward pass with the captured input
  4. Compares the output hash against the expected output
  5. Submits fraud proofs to the registry if a mismatch is found

Context-free:
  The verifier never sees user identity, session ID, or which node computed it.
  It only sees: shard_index + input activation + expected output activation.

Usage:
    python -m network.verifier_node
    python -m network.verifier_node --poll-interval 5 --registry localhost:50050
"""

import argparse
import sys
import os
import time

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import registry_pb2
import registry_pb2_grpc

from network.verification import Verifier, VerificationTicket


def _proto_to_ticket(proto) -> VerificationTicket:
    """Convert a VerificationTicketProto to a VerificationTicket."""
    return VerificationTicket(
        ticket_id=proto.ticket_id,
        shard_index=proto.shard_index,
        input_data=bytes(proto.input_data),
        input_shape=list(proto.input_shape),
        input_is_tokens=proto.input_is_tokens,
        expected_output_data=bytes(proto.expected_output_data),
        expected_output_shape=list(proto.expected_output_shape),
        expected_token=proto.expected_token if proto.has_expected_token else None,
        timestamp=proto.timestamp,
    )


def run_verifier(registry_address: str = None, poll_interval: float = 5.0,
                 max_tickets_per_poll: int = 10, shards_dir: str = None):
    """Run the verifier loop."""
    addr = registry_address or config.REGISTRY_ADDRESS
    channel = grpc.insecure_channel(addr)
    stub = registry_pb2_grpc.RegistryStub(channel)

    verifier = Verifier(shards_dir=shards_dir)

    print(f"[Verifier] Connected to registry at {addr}")
    print(f"[Verifier] Polling every {poll_interval}s for up to {max_tickets_per_poll} tickets")
    print(f"[Verifier] Waiting for verification tickets...")
    print()

    try:
        while True:
            try:
                # Pull pending tickets
                resp = stub.GetPendingTickets(registry_pb2.GetPendingTicketsRequest(
                    max_tickets=max_tickets_per_poll,
                ))

                tickets = [_proto_to_ticket(t) for t in resp.tickets]

                if tickets:
                    print(f"[Verifier] Got {len(tickets)} ticket(s) to verify")

                    for ticket in tickets:
                        print(f"  Verifying ticket {ticket.ticket_id} "
                              f"(shard {ticket.shard_index})...", end="")

                        proof = verifier.verify_ticket(ticket)

                        if proof:
                            print(f" FRAUD!")
                            # Submit fraud proof to registry
                            try:
                                stub.SubmitFraudProof(registry_pb2.FraudProofMessage(
                                    ticket_id=proof.ticket_id,
                                    shard_index=proof.shard_index,
                                    input_hash=proof.input_hash,
                                    expected_output_hash=proof.expected_output_hash,
                                    actual_output_hash=proof.actual_output_hash,
                                    expected_token=proof.expected_token or 0,
                                    actual_token=proof.actual_token or 0,
                                    timestamp=proof.timestamp,
                                ))
                                print(f"    Fraud proof submitted to registry")
                            except grpc.RpcError as e:
                                print(f"    Failed to submit fraud proof: {e}")
                        else:
                            print(f" OK")

                    stats = verifier.get_stats()
                    print(f"  [Stats] verified={stats['verified']}, "
                          f"failed={stats['failed']}")
                    print()

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"[Verifier] Registry unreachable, retrying...")
                else:
                    print(f"[Verifier] gRPC error: {e.details()}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n[Verifier] Shutting down...")
        stats = verifier.get_stats()
        print(f"[Verifier] Final stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Verifier Node")
    parser.add_argument("--registry", type=str, default=None,
                        help=f"Registry address (default: {config.REGISTRY_ADDRESS})")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between polling for tickets")
    parser.add_argument("--max-tickets", type=int, default=10,
                        help="Max tickets per poll")
    parser.add_argument("--shards-dir", type=str, default=None,
                        help="Shards directory (for loading model weights)")
    args = parser.parse_args()

    run_verifier(args.registry, args.poll_interval, args.max_tickets,
                 shards_dir=args.shards_dir)
