"""
Guard Relay Node — lightweight gRPC relay between client and compute pipeline.

The guard separates client identity from compute nodes:
  - The guard sees the client's IP address (from the TCP connection)
  - The guard receives an encrypted payload it CANNOT read
  - The guard forwards the encrypted payload to the target compute node
  - The guard returns the encrypted response to the client

This is analogous to Tor's entry guard:
  - Client picks a trusted guard and sticks with it long-term
  - The guard knows WHO is talking but not WHAT they're saying
  - The compute nodes know WHAT is being computed but not WHO asked

Usage:
    python -m network.guard_node --port 50060
"""

import argparse
import logging
import os
import sys
import time
import uuid
import threading

logger = logging.getLogger("unfed.guard")
from concurrent import futures

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
import registry_pb2
import registry_pb2_grpc

from network.onion import generate_keypair, public_key_to_bytes, private_key_to_bytes
from network.resilience import create_resilient_channel


class GuardRelayServicer(inference_pb2_grpc.GuardRelayServicer):
    """
    Guard relay — forwards encrypted requests to compute nodes.

    The guard:
      1. Receives an encrypted payload from the client
      2. Reads the target_address (where to forward)
      3. Forwards the encrypted payload as a ForwardRequest to the target
      4. Returns the response (also encrypted) to the client

    The guard CANNOT read the encrypted_payload — it's encrypted for Node 0's key.

    The guard earns a "relay share" on the mini-chain for every successful
    relay — IP-hiding is valuable privacy work and deserves compensation.
    """

    def __init__(self, node_id: str = ""):
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._channels: dict[str, grpc.Channel] = {}
        self._node_id = node_id
        self._daemon_stub = None

    def init_daemon_connection(self, registry_address: str):
        """Discover and connect to the chain daemon for share submission."""
        try:
            channel = grpc.insecure_channel(registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(
                registry_pb2.DiscoverRequest(model_id=""),
                timeout=10,
            )
            daemons = [n for n in resp.nodes if n.node_type == "daemon"]
            channel.close()
            if daemons:
                self._daemon_stub = inference_pb2_grpc.InferenceNodeStub(
                    grpc.insecure_channel(
                        daemons[0].address, options=config.GRPC_OPTIONS)
                )
                print(f"[Guard] Connected to chain daemon at {daemons[0].address}")
            else:
                print(f"[Guard] No daemon found — relay shares won't be recorded")
        except Exception:
            print(f"[Guard] Could not connect to daemon")

    def _record_relay_share(self, session_id: str, payload_hash: str):
        """Record a relay share on the mini-chain.

        Guards earn shares for IP-hiding — each successful relay
        proves the guard did privacy-preserving work for the network.
        The share uses shard_index=-1 to distinguish relay work from
        compute work.
        """
        if not self._daemon_stub or not self._node_id:
            return

        import hashlib
        from economics.share_chain import ComputeShare
        from economics.distributed_chain import share_to_proto

        guard_fee_ratio = getattr(config, 'GUARD_FEE_RATIO', 0.05)

        share = ComputeShare(
            node_id=self._node_id,
            shard_index=-1,  # -1 = relay/guard work (not compute)
            session_id=session_id,
            activation_hash=payload_hash,
            tokens_processed=1,
            share_weight=guard_fee_ratio,  # guards earn a fraction of compute rate
        )

        try:
            self._daemon_stub.SubmitShares(
                inference_pb2.SubmitSharesRequest(
                    shares=[share_to_proto(share)],
                    submitter_id=self._node_id,
                ),
                timeout=5,
            )
        except Exception:
            pass  # non-critical — don't break relay for share recording

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        if address not in self._stubs:
            channel = create_resilient_channel(address, config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def _get_channel(self, address: str) -> grpc.Channel:
        """Get (or create) a raw gRPC channel for opaque byte forwarding."""
        if address not in self._channels:
            self._channels[address] = create_resilient_channel(
                address, config.GRPC_OPTIONS)
        return self._channels[address]

    def Relay(self, request, context):
        """Relay an encrypted request to a compute node."""
        target = request.target_address
        if not target:
            context.set_details("No target_address specified")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return inference_pb2.RelayResponse()

        try:
            # Forward the payload as OPAQUE BYTES — the guard never
            # deserializes or inspects the request content.  We use
            # gRPC's low-level unary_unary call to send the raw
            # serialized ForwardRequest bytes directly to the target
            # node's Forward RPC, then return the raw response bytes.
            import hashlib
            payload_hash = hashlib.sha256(
                request.encrypted_payload[:256]
            ).hexdigest()[:16]

            channel = self._get_channel(target)
            raw_response_bytes = channel.unary_unary(
                '/unfed.InferenceNode/Forward',
                request_serializer=None,
                response_deserializer=None,
            )(request.encrypted_payload)

            # Record relay share — the guard earned credit for IP-hiding.
            # We use the payload hash as a session proxy since we
            # intentionally cannot read the session_id from the payload.
            self._record_relay_share(payload_hash, payload_hash)

            return inference_pb2.RelayResponse(
                encrypted_payload=raw_response_bytes,
            )

        except grpc.RpcError as e:
            context.set_details(f"Failed to relay to {target}: {e.details()}")
            context.set_code(e.code())
            return inference_pb2.RelayResponse()


def _serialize_response(response) -> bytes:
    """Serialize a ForwardResponse to bytes for relay."""
    return response.SerializeToString()


class GuardRegistration:
    """Manages the guard's lifecycle with the registry."""

    def __init__(self, address: str, port: int, registry_address: str = None):
        self.node_id = str(uuid.uuid4())
        self.address = address
        self.registry_address = registry_address or config.REGISTRY_ADDRESS

        # Generate keypair for the guard's encryption layer
        self.private_key, self.public_key = generate_keypair()
        self.public_key_bytes = public_key_to_bytes(self.public_key)

        self._channel = grpc.insecure_channel(self.registry_address)
        self._stub = registry_pb2_grpc.RegistryStub(self._channel)
        self._running = False

    def start(self):
        """Register as a guard node."""
        try:
            self._stub.Register(registry_pb2.RegisterRequest(
                node_id=self.node_id,
                address=self.address,
                model_id="",  # guards don't serve a model
                shard_index=-1,
                layer_start=0,
                layer_end=0,
                has_embedding=False,
                has_lm_head=False,
                public_key=self.public_key_bytes,
                node_type="guard",
            ))
            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            print(f"[Guard {self.node_id[:8]}...] Registered with registry")
        except grpc.RpcError as e:
            print(f"[Guard] Failed to register: {e.details()}")

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(config.HEARTBEAT_INTERVAL_SECONDS)
            if self._running:
                try:
                    self._stub.Heartbeat(registry_pb2.HeartbeatRequest(
                        node_id=self.node_id))
                except grpc.RpcError as e:
                    logger.debug("Guard heartbeat failed: %s", e)

    def stop(self):
        self._running = False
        try:
            self._stub.Unregister(registry_pb2.UnregisterRequest(
                node_id=self.node_id))
        except grpc.RpcError as e:
            logger.debug("Guard unregister failed: %s", e)
        print(f"[Guard {self.node_id[:8]}...] Unregistered")


def serve(port: int, host: str = "[::]", advertise: str = None,
          registry_address: str = None, node_config=None):
    """Start the guard relay server."""
    public_address = advertise or f"localhost:{port}"

    # Resolve configurable values
    grpc_max_workers = (node_config.grpc_max_workers
                        if node_config else 10)
    grpc_options = (node_config.grpc_options
                    if node_config else config.GRPC_OPTIONS)

    registration = GuardRegistration(
        public_address, port, registry_address)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
        options=grpc_options,
    )
    servicer = GuardRelayServicer(node_id=registration.node_id)
    inference_pb2_grpc.add_GuardRelayServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"[Guard] Relay listening on {host}:{port}")
    print(f"[Guard] Advertised as {public_address}")

    registration.start()

    # Connect to daemon for relay share recording
    reg_addr = registry_address or config.REGISTRY_ADDRESS
    servicer.init_daemon_connection(reg_addr)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Guard] Shutting down...")
        registration.stop()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Guard Relay Node")
    parser.add_argument("--port", type=int, default=50060,
                        help="Port for the guard relay")
    parser.add_argument("--host", type=str, default="[::]",
                        help="Bind address")
    parser.add_argument("--advertise", type=str, default=None,
                        help="Address to advertise to registry")
    parser.add_argument("--registry", type=str, default=None,
                        help=f"Registry address (default: {config.REGISTRY_ADDRESS})")
    args = parser.parse_args()
    serve(args.port, args.host, args.advertise, args.registry)
