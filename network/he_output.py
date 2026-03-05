"""
HE-style output artifact protection.

This module provides a client-key encrypted token artifact path used to
protect output tokens in transit. It is intentionally lightweight and
implemented with X25519 key agreement + AES-GCM.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def generate_client_keypair() -> tuple[bytes, bytes]:
    """Return (private_bytes, public_bytes) for a client HE-output session."""
    private = X25519PrivateKey.generate()
    public = private.public_key()
    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def _derive_key(shared_secret: bytes, *, session_id: str, key_id: str) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=(session_id + ":" + key_id).encode("utf-8"),
        info=b"unfed-he-output-v1",
    )
    return hkdf.derive(shared_secret)


def _aad(*, session_id: str, step: int, key_id: str) -> bytes:
    return f"{session_id}:{step}:{key_id}".encode("utf-8")


def _nonce(*, session_id: str, step: int) -> bytes:
    h = hashlib.sha256(f"{session_id}:{step}".encode("utf-8")).digest()
    return h[:12]


def encrypt_token_artifact(
    *,
    client_public_key: bytes,
    sender_private_key: bytes | None,
    session_id: str,
    step: int,
    key_id: str,
    token_id: int,
    is_eos: bool,
) -> dict[str, bytes | str | int]:
    """
    Encrypt token payload for client-only decryption.

    Payload format: 4-byte signed token_id + 1-byte eos flag.
    """
    if sender_private_key:
        sender_private = X25519PrivateKey.from_private_bytes(sender_private_key)
    else:
        sender_private = X25519PrivateKey.generate()
    sender_public = sender_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    client_public = X25519PublicKey.from_public_bytes(client_public_key)
    shared = sender_private.exchange(client_public)
    key = _derive_key(shared, session_id=session_id, key_id=key_id)
    nonce = _nonce(session_id=session_id, step=step)
    aad = _aad(session_id=session_id, step=step, key_id=key_id)

    payload = token_id.to_bytes(4, "big", signed=True) + (b"\x01" if is_eos else b"\x00")
    ciphertext = AESGCM(key).encrypt(nonce, payload, aad)
    return {
        "ciphertext": ciphertext,
        "nonce": nonce,
        "sender_public_key": sender_public,
        "algo": "x25519-aesgcm-v1",
        "session_id": session_id,
        "step": int(step),
        "key_id": key_id,
    }


def decrypt_token_artifact(
    *,
    client_private_key: bytes,
    sender_public_key: bytes,
    session_id: str,
    step: int,
    key_id: str,
    nonce: bytes,
    ciphertext: bytes,
) -> tuple[int, bool]:
    """Decrypt and parse a token payload."""
    client_private = X25519PrivateKey.from_private_bytes(client_private_key)
    sender_public = X25519PublicKey.from_public_bytes(sender_public_key)
    shared = client_private.exchange(sender_public)
    key = _derive_key(shared, session_id=session_id, key_id=key_id)
    aad = _aad(session_id=session_id, step=step, key_id=key_id)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, aad)
    if len(plaintext) < 5:
        raise ValueError("Invalid HE output artifact payload length")
    token_id = int.from_bytes(plaintext[:4], "big", signed=True)
    is_eos = plaintext[4:5] == b"\x01"
    return token_id, is_eos

