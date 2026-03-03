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
import json
import sys
import os
import time
import uuid

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


def _parse_policy(config_json: str) -> dict:
    defaults = {
        "poll_interval_seconds": 5.0,
        "max_tickets_per_poll": 10,
        "heartbeat_timeout_seconds": 30,
    }
    if not config_json:
        return defaults
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError:
        return defaults
    for key in defaults:
        if key in parsed:
            defaults[key] = parsed[key]
    return defaults


def run_verifier(registry_address: str = None, poll_interval: float = 5.0,
                 max_tickets_per_poll: int = 10, shards_dir: str = None,
                 verifier_id: str = "", advertise_address: str = ""):
    """Run the verifier loop."""
    addr = registry_address or config.REGISTRY_ADDRESS
    channel = grpc.insecure_channel(addr)
    stub = registry_pb2_grpc.RegistryStub(channel)
    verifier_id = verifier_id or f"verifier-{uuid.uuid4().hex[:12]}"

    verifier = Verifier(shards_dir=shards_dir)
    policy = {
        "poll_interval_seconds": poll_interval,
        "max_tickets_per_poll": max_tickets_per_poll,
        "heartbeat_timeout_seconds": 30,
    }

    # Bootstrap registration with registry-owned policy.
    while True:
        try:
            reg = stub.RegisterVerifier(
                registry_pb2.RegisterVerifierRequest(
                    verifier_id=verifier_id,
                    address=advertise_address,
                )
            )
            if reg.success:
                policy = _parse_policy(reg.config_json)
                break
            print(f"[Verifier] Register failed: {reg.message}")
        except grpc.RpcError:
            print("[Verifier] Registry unreachable during register, retrying...")
        time.sleep(2)

    print(f"[Verifier] Connected to registry at {addr}")
    print(f"[Verifier] ID: {verifier_id}")
    print(
        f"[Verifier] Polling every {policy['poll_interval_seconds']}s "
        f"for up to {policy['max_tickets_per_poll']} tickets"
    )
    print(f"[Verifier] Waiting for verification tickets...")
    print()
    next_heartbeat = 0.0
    next_config_refresh = 0.0

    try:
        while True:
            try:
                now = time.time()
                if now >= next_heartbeat:
                    hb = stub.VerifierHeartbeat(
                        registry_pb2.VerifierHeartbeatRequest(verifier_id=verifier_id)
                    )
                    if not hb.acknowledged:
                        reg = stub.RegisterVerifier(
                            registry_pb2.RegisterVerifierRequest(
                                verifier_id=verifier_id,
                                address=advertise_address,
                            )
                        )
                        if reg.success:
                            policy = _parse_policy(reg.config_json)
                    elif hb.config_json:
                        policy = _parse_policy(hb.config_json)
                    timeout = max(3.0, float(policy["heartbeat_timeout_seconds"]))
                    next_heartbeat = now + max(1.0, timeout / 3.0)

                if now >= next_config_refresh:
                    cfg = stub.GetVerifierConfig(
                        registry_pb2.GetVerifierConfigRequest(verifier_id=verifier_id)
                    )
                    if cfg.success:
                        policy = _parse_policy(cfg.config_json)
                    next_config_refresh = now + 15.0

                # Pull pending tickets
                resp = stub.GetPendingTickets(registry_pb2.GetPendingTicketsRequest(
                    max_tickets=int(policy["max_tickets_per_poll"]),
                    verifier_id=verifier_id,
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
                                    verifier_id=verifier_id,
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

            time.sleep(float(policy["poll_interval_seconds"]))

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
    parser.add_argument("--verifier-id", type=str, default="",
                        help="Stable verifier identity (default: auto-generated)")
    parser.add_argument("--advertise", type=str, default="",
                        help="Verifier advertised address for registry metadata")
    args = parser.parse_args()

    run_verifier(
        args.registry,
        args.poll_interval,
        args.max_tickets,
        shards_dir=args.shards_dir,
        verifier_id=args.verifier_id,
        advertise_address=args.advertise,
    )
