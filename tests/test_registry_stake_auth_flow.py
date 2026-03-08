import os
import sys
import time
import secrets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from eth_account import Account
from eth_account.messages import encode_defunct

from network.registry_server import RegistryServicer
from network.share_auth import (
    generate_signing_keypair,
    registration_pop_payload,
    registration_stake_auth_payload,
    heartbeat_auth_payload,
    sign_bytes,
)


class _FakeOnchainEscrow:
    def __init__(self, *, min_stake: int = 100):
        self._eligible: dict[str, bool] = {}
        self._min_stake = int(min_stake)

    def set_eligible(self, node_id: str, is_eligible: bool) -> None:
        self._eligible[str(node_id).lower()] = bool(is_eligible)

    def is_eligible(self, node_address: str) -> bool:
        return bool(self._eligible.get(str(node_address).lower(), False))

    def min_stake(self) -> int:
        return self._min_stake


class _FakeContext:
    def __init__(self, peer: str = "ipv4:127.0.0.1:55555"):
        self._peer = peer

    def peer(self) -> str:
        return self._peer


def _build_register_request(
    *,
    node_priv: str,
    address: str = "localhost:50110",
    model_id: str = "model-auth",
    shard_index: int = 0,
    node_type: str = "compute",
    nonce: str | None = None,
    timestamp_ms: int | None = None,
    include_stake_sig: bool = True,
):
    acct = Account.from_key(node_priv)
    node_id = acct.address
    share_priv, share_pub = generate_signing_keypair()
    ts_ms = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    reg_nonce = nonce or secrets.token_hex(16)

    pop = sign_bytes(
        share_priv,
        registration_pop_payload(
            node_id=node_id,
            address=address,
            model_id=model_id,
            shard_index=shard_index,
            node_type=node_type,
        ),
    )
    payload = registration_stake_auth_payload(
        node_id=node_id,
        address=address,
        model_id=model_id,
        shard_index=shard_index,
        node_type=node_type,
        share_signing_public_key=share_pub,
        timestamp_ms=ts_ms,
        nonce=reg_nonce,
    )
    sig = b""
    if include_stake_sig:
        sig = bytes(
            Account.sign_message(
                encode_defunct(text=payload),
                private_key=node_priv,
            ).signature
        )
    req = registry_pb2.RegisterRequest(
        node_id=node_id,
        address=address,
        model_id=model_id,
        shard_index=shard_index,
        layer_start=0,
        layer_end=1,
        has_embedding=True,
        has_lm_head=False,
        node_type=node_type,
        share_signing_public_key=share_pub,
        share_signing_pop=pop,
        stake_auth_timestamp_ms=ts_ms,
        stake_auth_nonce=reg_nonce,
        stake_auth_signature=sig,
    )
    return req, share_priv


def test_onchain_rejects_unauthenticated_registration():
    svc = RegistryServicer(no_chain=True)
    fake = _FakeOnchainEscrow(min_stake=123)
    svc._onchain_escrow = fake

    node_priv = Account.create().key.hex()
    req, _ = _build_register_request(
        node_priv=node_priv,
        include_stake_sig=False,
    )
    fake.set_eligible(req.node_id, True)

    resp = svc.Register(req, None)
    assert resp.success is False
    assert "stake_auth_signature" in resp.message


def test_onchain_accepts_authenticated_registration_and_blocks_replay():
    svc = RegistryServicer(no_chain=True)
    fake = _FakeOnchainEscrow(min_stake=123)
    svc._onchain_escrow = fake

    node_priv = Account.create().key.hex()
    req, _ = _build_register_request(node_priv=node_priv)
    fake.set_eligible(req.node_id, True)

    ok = svc.Register(req, None)
    assert ok.success is True

    replay = svc.Register(req, None)
    assert replay.success is False
    assert "replayed" in replay.message or "nonce" in replay.message


def test_stake_then_unstake_excludes_and_evicts_node():
    svc = RegistryServicer(no_chain=True)
    fake = _FakeOnchainEscrow(min_stake=123)
    svc._onchain_escrow = fake
    svc._stake_revalidate_ttl_seconds = 0.0

    node_priv = Account.create().key.hex()
    req, share_priv = _build_register_request(node_priv=node_priv)
    fake.set_eligible(req.node_id, True)
    assert svc.Register(req, None).success is True

    listed = svc.Discover(registry_pb2.DiscoverRequest(model_id=""), None)
    assert any(n.node_id == req.node_id for n in listed.nodes)

    # Simulate unstake after admission.
    fake.set_eligible(req.node_id, False)
    filtered = svc.Discover(registry_pb2.DiscoverRequest(model_id=""), None)
    assert all(n.node_id != req.node_id for n in filtered.nodes)

    ts_ms = int(time.time() * 1000)
    nonce = secrets.token_hex(16)
    hb_sig = sign_bytes(
        share_priv,
        heartbeat_auth_payload(
            node_id=req.node_id,
            timestamp_ms=ts_ms,
            nonce=nonce,
        ),
    )
    hb = svc.Heartbeat(
        registry_pb2.HeartbeatRequest(
            node_id=req.node_id,
            auth_timestamp_ms=ts_ms,
            auth_nonce=nonce,
            auth_signature=hb_sig,
        ),
        None,
    )
    assert hb.acknowledged is False

    final = svc.Discover(registry_pb2.DiscoverRequest(model_id=""), None)
    assert all(n.node_id != req.node_id for n in final.nodes)


def test_daemon_spoof_without_auth_material_is_rejected():
    svc = RegistryServicer(no_chain=True)
    fake = _FakeOnchainEscrow(min_stake=123)
    svc._onchain_escrow = fake
    daemon_id = "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"
    fake.set_eligible(daemon_id, True)
    resp = svc.Register(
        registry_pb2.RegisterRequest(
            node_id=daemon_id,
            address="localhost:50070",
            model_id="",
            shard_index=-1,
            layer_start=-1,
            layer_end=-1,
            node_type="daemon",
            public_key=b"",
        ),
        None,
    )
    assert resp.success is False
    assert (
        "share_signing_public_key" in resp.message
        or "share_signing_pop" in resp.message
        or "stake_auth_signature" in resp.message
    )


def test_heartbeat_timestamp_skew_boundaries():
    svc = RegistryServicer(no_chain=True)
    svc._auth_max_skew_ms = 1000
    node_priv = Account.create().key.hex()
    req, share_priv = _build_register_request(node_priv=node_priv)
    assert svc.Register(req, None).success is True

    ts_ok = int(time.time() * 1000) - 200
    nonce_ok = secrets.token_hex(16)
    hb_ok = svc.Heartbeat(
        registry_pb2.HeartbeatRequest(
            node_id=req.node_id,
            auth_timestamp_ms=ts_ok,
            auth_nonce=nonce_ok,
            auth_signature=sign_bytes(
                share_priv,
                heartbeat_auth_payload(
                    node_id=req.node_id,
                    timestamp_ms=ts_ok,
                    nonce=nonce_ok,
                ),
            ),
        ),
        None,
    )
    assert hb_ok.acknowledged is True

    ts_old = int(time.time() * 1000) - 5000
    nonce_old = secrets.token_hex(16)
    hb_old = svc.Heartbeat(
        registry_pb2.HeartbeatRequest(
            node_id=req.node_id,
            auth_timestamp_ms=ts_old,
            auth_nonce=nonce_old,
            auth_signature=sign_bytes(
                share_priv,
                heartbeat_auth_payload(
                    node_id=req.node_id,
                    timestamp_ms=ts_old,
                    nonce=nonce_old,
                ),
            ),
        ),
        None,
    )
    assert hb_old.acknowledged is False

    ts_future = int(time.time() * 1000) + 5000
    nonce_future = secrets.token_hex(16)
    hb_future = svc.Heartbeat(
        registry_pb2.HeartbeatRequest(
            node_id=req.node_id,
            auth_timestamp_ms=ts_future,
            auth_nonce=nonce_future,
            auth_signature=sign_bytes(
                share_priv,
                heartbeat_auth_payload(
                    node_id=req.node_id,
                    timestamp_ms=ts_future,
                    nonce=nonce_future,
                ),
            ),
        ),
        None,
    )
    assert hb_future.acknowledged is False


def test_unstake_recheck_ttl_window_then_expires():
    svc = RegistryServicer(no_chain=True)
    fake = _FakeOnchainEscrow(min_stake=123)
    svc._onchain_escrow = fake
    svc._stake_revalidate_ttl_seconds = 3600.0

    node_priv = Account.create().key.hex()
    req, _ = _build_register_request(node_priv=node_priv)
    fake.set_eligible(req.node_id, True)
    assert svc.Register(req, None).success is True

    fake.set_eligible(req.node_id, False)
    immediate = svc.Discover(registry_pb2.DiscoverRequest(model_id=""), None)
    assert any(n.node_id == req.node_id for n in immediate.nodes)

    # Force cache expiry to emulate passing the revalidation window.
    prev = svc._stake_eligibility_cache[req.node_id][0]
    svc._stake_eligibility_cache[req.node_id] = (prev, time.time() - 4000.0)
    post_expiry = svc.Discover(registry_pb2.DiscoverRequest(model_id=""), None)
    assert all(n.node_id != req.node_id for n in post_expiry.nodes)


def test_register_rate_limit_rejects_burst():
    svc = RegistryServicer(no_chain=True)
    svc._auth_rate_limit_per_minute = 2
    node_priv = Account.create().key.hex()
    ctx = _FakeContext()
    req1, _ = _build_register_request(node_priv=node_priv, nonce="n1")
    req2, _ = _build_register_request(node_priv=node_priv, nonce="n2")
    req3, _ = _build_register_request(node_priv=node_priv, nonce="n3")
    assert svc.Register(req1, ctx).success is True
    assert svc.Register(req2, ctx).success is True
    third = svc.Register(req3, ctx)
    assert third.success is False
    assert "rate limit" in third.message


def test_nonce_store_is_capped_for_control_plane():
    svc = RegistryServicer(no_chain=True)
    svc._auth_nonce_max_entries = 2
    node_priv = Account.create().key.hex()
    req, share_priv = _build_register_request(node_priv=node_priv)
    assert svc.Register(req, None).success is True
    for i in range(3):
        ts_ms = int(time.time() * 1000)
        nonce = f"hb-{i}"
        hb = svc.Heartbeat(
            registry_pb2.HeartbeatRequest(
                node_id=req.node_id,
                auth_timestamp_ms=ts_ms,
                auth_nonce=nonce,
                auth_signature=sign_bytes(
                    share_priv,
                    heartbeat_auth_payload(
                        node_id=req.node_id,
                        timestamp_ms=ts_ms,
                        nonce=nonce,
                    ),
                ),
            ),
            None,
        )
        assert hb.acknowledged is True
    assert len(svc._used_control_nonces) <= 2
