#!/usr/bin/env python3
"""
Register a lightweight mock compute node in the registry and keep heartbeating.

This is useful for preflight/runtime checks in environments that do not yet have
real model shard files available locally.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import uuid

import grpc

sys.path.insert(0, "proto")
sys.path.insert(0, ".")
import registry_pb2  # noqa: E402
import registry_pb2_grpc  # noqa: E402
from network.share_auth import (  # noqa: E402
    generate_signing_keypair,
    heartbeat_auth_payload,
    registration_pop_payload,
    sign_bytes,
)


def _register(
    stub,
    *,
    node_id: str,
    address: str,
    model_id: str,
    shard_index: int,
    signing_private_key: bytes,
    signing_public_key: bytes,
) -> None:
    pop_payload = registration_pop_payload(
        node_id=node_id,
        address=address,
        model_id=model_id,
        shard_index=shard_index,
        node_type="compute",
    )
    pop_signature = sign_bytes(signing_private_key, pop_payload)
    req = registry_pb2.RegisterRequest(
        node_id=node_id,
        address=address,
        model_id=model_id,
        shard_index=shard_index,
        layer_start=0,
        layer_end=1,
        has_embedding=True,
        has_lm_head=True,
        node_type="compute",
        share_signing_public_key=signing_public_key,
        share_signing_pop=pop_signature,
    )
    resp = stub.Register(req, timeout=5)
    if not resp.success:
        raise RuntimeError(f"register failed: {resp.message}")


def main() -> int:
    p = argparse.ArgumentParser(description="Mock UNFED serving node registrar")
    p.add_argument("--registry", default="127.0.0.1:50050", help="registry host:port")
    p.add_argument("--model-id", default="mock/testnet-model", help="model id to advertise")
    p.add_argument("--address", default="mock-node:59999", help="advertised node address")
    p.add_argument("--node-id", default="", help="override node id")
    p.add_argument("--shard-index", type=int, default=0, help="mock shard index")
    p.add_argument("--heartbeat-seconds", type=float, default=5.0, help="heartbeat cadence")
    args = p.parse_args()

    node_id = args.node_id.strip() or f"mock-{uuid.uuid4()}"
    signing_private_key, signing_public_key = generate_signing_keypair()
    channel = grpc.insecure_channel(args.registry)
    stub = registry_pb2_grpc.RegistryStub(channel)

    stopped = False

    def _stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    _register(
        stub,
        node_id=node_id,
        address=args.address,
        model_id=args.model_id,
        shard_index=args.shard_index,
        signing_private_key=signing_private_key,
        signing_public_key=signing_public_key,
    )
    print(f"[mock-node] registered node_id={node_id} model={args.model_id}")

    while not stopped:
        try:
            now_ms = int(time.time() * 1000)
            nonce = str(uuid.uuid4())
            hb_payload = heartbeat_auth_payload(
                node_id=node_id,
                timestamp_ms=now_ms,
                nonce=nonce,
            )
            hb_sig = sign_bytes(signing_private_key, hb_payload)
            hb = registry_pb2.HeartbeatRequest(
                node_id=node_id,
                auth_timestamp_ms=now_ms,
                auth_nonce=nonce,
                auth_signature=hb_sig,
            )
            ack = stub.Heartbeat(hb, timeout=5)
            if not ack.acknowledged:
                _register(
                    stub,
                    node_id=node_id,
                    address=args.address,
                    model_id=args.model_id,
                    shard_index=args.shard_index,
                    signing_private_key=signing_private_key,
                    signing_public_key=signing_public_key,
                )
        except grpc.RpcError:
            time.sleep(args.heartbeat_seconds)
            continue
        time.sleep(args.heartbeat_seconds)

    channel.close()
    print("[mock-node] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
