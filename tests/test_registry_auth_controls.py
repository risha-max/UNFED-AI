import os
import sys
import time
import secrets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from network.registry_server import RegistryServicer
from network.share_auth import (
    generate_signing_keypair,
    registration_pop_payload,
    heartbeat_auth_payload,
    unregister_auth_payload,
    sign_bytes,
)


def _register_node(svc: RegistryServicer, node_id: str = "node-auth-1"):
    priv, pub = generate_signing_keypair()
    address = "localhost:50110"
    req = registry_pb2.RegisterRequest(
        node_id=node_id,
        address=address,
        model_id="model-auth",
        shard_index=0,
        layer_start=0,
        layer_end=1,
        has_embedding=True,
        has_lm_head=False,
        node_type="compute",
        share_signing_public_key=pub,
        share_signing_pop=sign_bytes(
            priv,
            registration_pop_payload(
                node_id=node_id,
                address=address,
                model_id="model-auth",
                shard_index=0,
                node_type="compute",
            ),
        ),
    )
    resp = svc.Register(req, None)
    assert resp.success is True
    return priv


def test_heartbeat_requires_valid_signature():
    svc = RegistryServicer(no_chain=True)
    priv = _register_node(svc)
    ts_ms = int(time.time() * 1000)
    nonce = secrets.token_hex(16)
    ok = svc.Heartbeat(
        registry_pb2.HeartbeatRequest(
            node_id="node-auth-1",
            auth_timestamp_ms=ts_ms,
            auth_nonce=nonce,
            auth_signature=sign_bytes(
                priv,
                heartbeat_auth_payload(
                    node_id="node-auth-1",
                    timestamp_ms=ts_ms,
                    nonce=nonce,
                ),
            ),
        ),
        None,
    )
    assert ok.acknowledged is True

    bad = svc.Heartbeat(
        registry_pb2.HeartbeatRequest(
            node_id="node-auth-1",
            auth_timestamp_ms=ts_ms,
            auth_nonce=nonce,
            auth_signature=b"bad",
        ),
        None,
    )
    assert bad.acknowledged is False


def test_unregister_requires_valid_signature():
    svc = RegistryServicer(no_chain=True)
    priv = _register_node(svc, node_id="node-auth-2")
    ts_ms = int(time.time() * 1000)
    nonce = secrets.token_hex(16)
    bad = svc.Unregister(
        registry_pb2.UnregisterRequest(
            node_id="node-auth-2",
            auth_timestamp_ms=ts_ms,
            auth_nonce=nonce,
            auth_signature=b"",
        ),
        None,
    )
    assert bad.success is False

    ts_ms2 = int(time.time() * 1000)
    nonce2 = secrets.token_hex(16)
    ok = svc.Unregister(
        registry_pb2.UnregisterRequest(
            node_id="node-auth-2",
            auth_timestamp_ms=ts_ms2,
            auth_nonce=nonce2,
            auth_signature=sign_bytes(
                priv,
                unregister_auth_payload(
                    node_id="node-auth-2",
                    timestamp_ms=ts_ms2,
                    nonce=nonce2,
                ),
            ),
        ),
        None,
    )
    assert ok.success is True
