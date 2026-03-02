import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import inference_pb2
from economics.share_chain import ShareChain
from network.daemon_node import DaemonServicer
from network.share_auth import (
    PAYLOAD_HASH_VERSION,
    SharePayload,
    canonical_share_payload_bytes,
    generate_signing_keypair,
    sign_bytes,
)


def _signed_share_proto(*, node_id: str, private_key: bytes, step_index: int = 0, timestamp_ms: int | None = None):
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)
    payload = SharePayload(
        node_id=node_id,
        shard_index=0,
        session_id="sess-1",
        session_nonce="nonce-1",
        step_index=step_index,
        activation_hash="abc123",
        tokens_processed=1,
        share_weight=1.0,
        timestamp_ms=timestamp_ms,
        payload_hash_version=PAYLOAD_HASH_VERSION,
    )
    sig = sign_bytes(private_key, canonical_share_payload_bytes(payload))
    return inference_pb2.ShareProto(
        node_id=payload.node_id,
        shard_index=payload.shard_index,
        session_id=payload.session_id,
        activation_hash=payload.activation_hash,
        tokens_processed=payload.tokens_processed,
        timestamp=time.time(),
        share_weight=payload.share_weight,
        session_nonce=payload.session_nonce,
        step_index=payload.step_index,
        timestamp_ms=payload.timestamp_ms,
        signature=sig,
        payload_hash_version=payload.payload_hash_version,
    )


def _servicer_with_signer(node_id: str, public_key: bytes) -> DaemonServicer:
    chain = ShareChain(block_interval=1.0, settlement_blocks=2, store=None)
    svc = DaemonServicer(chain=chain, node_id="daemon-test", registry_address="localhost:0")
    svc._signer_cache = {node_id: public_key}
    svc._signer_cache_ts = time.time()
    return svc


def test_daemon_accepts_valid_signed_share():
    priv, pub = generate_signing_keypair()
    node_id = "0x1234567890abcdef1234567890abcdef12345678"
    svc = _servicer_with_signer(node_id, pub)
    req = inference_pb2.SubmitSharesRequest(
        shares=[_signed_share_proto(node_id=node_id, private_key=priv, step_index=0)],
        submitter_id=node_id,
    )
    resp = svc.SubmitShares(req, None)
    assert resp.accepted == 1
    assert svc.chain.has_pending_shares() is True


def test_daemon_rejects_replay_share():
    priv, pub = generate_signing_keypair()
    node_id = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    svc = _servicer_with_signer(node_id, pub)
    share = _signed_share_proto(node_id=node_id, private_key=priv, step_index=0)
    req = inference_pb2.SubmitSharesRequest(shares=[share], submitter_id=node_id)
    first = svc.SubmitShares(req, None)
    second = svc.SubmitShares(req, None)
    assert first.accepted == 1
    assert second.accepted == 0


def test_daemon_rejects_stale_timestamp():
    priv, pub = generate_signing_keypair()
    node_id = "0x9999999999999999999999999999999999999999"
    svc = _servicer_with_signer(node_id, pub)
    stale_ms = int(time.time() * 1000) - (10 * 60 * 1000)
    req = inference_pb2.SubmitSharesRequest(
        shares=[_signed_share_proto(node_id=node_id, private_key=priv, step_index=0, timestamp_ms=stale_ms)],
        submitter_id=node_id,
    )
    resp = svc.SubmitShares(req, None)
    assert resp.accepted == 0


def test_daemon_enforces_session_cap(monkeypatch):
    monkeypatch.setenv("UNFED_MAX_SHARES_PER_SESSION_NODE", "1")
    priv, pub = generate_signing_keypair()
    node_id = "0x7777777777777777777777777777777777777777"
    svc = _servicer_with_signer(node_id, pub)
    first = inference_pb2.SubmitSharesRequest(
        shares=[_signed_share_proto(node_id=node_id, private_key=priv, step_index=0)],
        submitter_id=node_id,
    )
    second = inference_pb2.SubmitSharesRequest(
        shares=[_signed_share_proto(node_id=node_id, private_key=priv, step_index=1)],
        submitter_id=node_id,
    )
    r1 = svc.SubmitShares(first, None)
    r2 = svc.SubmitShares(second, None)
    assert r1.accepted == 1
    assert r2.accepted == 0
