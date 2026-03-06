"""
Share authentication helpers for signed ComputeShare ingestion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


PAYLOAD_HASH_VERSION = "v1"


@dataclass(frozen=True)
class SharePayload:
    node_id: str
    shard_index: int
    session_id: str
    session_nonce: str
    step_index: int
    activation_hash: str
    tokens_processed: int
    share_weight: float
    timestamp_ms: int
    payload_hash_version: str = PAYLOAD_HASH_VERSION
    prev_block_hash: str = ""
    prev_share_hash: str = ""
    idempotency_key: str = ""


def canonical_share_payload_bytes(payload: SharePayload) -> bytes:
    return (
        f"{payload.payload_hash_version}|{payload.node_id}|{payload.shard_index}|"
        f"{payload.session_id}|{payload.session_nonce}|{payload.step_index}|"
        f"{payload.activation_hash}|{payload.tokens_processed}|{payload.share_weight:.8f}|"
        f"{payload.timestamp_ms}|{payload.prev_block_hash}|{payload.prev_share_hash}|"
        f"{payload.idempotency_key}"
    ).encode("utf-8")


def registration_pop_payload(
    *,
    node_id: str,
    address: str,
    model_id: str,
    shard_index: int,
    node_type: str,
) -> bytes:
    return (
        f"unfed-share-key|{node_id}|{address}|{model_id}|"
        f"{shard_index}|{node_type or 'compute'}"
    ).encode("utf-8")


def generate_signing_keypair() -> tuple[bytes, bytes]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def load_private_key(private_key_bytes: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(private_key_bytes)


def public_key_from_private(private_key_bytes: bytes) -> bytes:
    return load_private_key(private_key_bytes).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def sign_bytes(private_key_bytes: bytes, payload: bytes) -> bytes:
    return load_private_key(private_key_bytes).sign(payload)


def verify_signature(public_key_bytes: bytes, payload: bytes, signature: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(signature, payload)
        return True
    except (InvalidSignature, ValueError):
        return False


def key_file_paths(node_id: str) -> tuple[str, str]:
    safe_node = (node_id or "unknown").replace("/", "_")
    base = os.path.expanduser("~/.unfed/share_keys")
    return (
        os.path.join(base, f"{safe_node}.ed25519.key"),
        os.path.join(base, f"{safe_node}.ed25519.pub"),
    )

