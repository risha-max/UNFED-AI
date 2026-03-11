"""
Manifest signing and verification helpers for shard distribution.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import time
from typing import Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


SIGNATURE_FIELD = "manifest_signature"
SIGNATURE_ALGORITHM = "ed25519-sha256-json-v1"


def _canonical_manifest_bytes(manifest: dict) -> bytes:
    data = copy.deepcopy(manifest)
    data.pop(SIGNATURE_FIELD, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_private_key(private_key_bytes: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(private_key_bytes)


def _public_key_bytes_from_private(private_key_bytes: bytes) -> bytes:
    return _load_private_key(private_key_bytes).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _decode_key_material(raw: str) -> bytes:
    value = (raw or "").strip()
    if value.startswith("0x"):
        return bytes.fromhex(value[2:])
    try:
        return base64.b64decode(value)
    except Exception as e:
        raise ValueError(f"invalid key encoding: {e}") from e


def decode_public_key_bytes(raw: str) -> bytes:
    key = _decode_key_material(raw)
    if len(key) != 32:
        raise ValueError("ed25519 public key must be 32 bytes")
    return key


def load_trusted_keys_file(path: str) -> dict[str, bytes]:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("trusted keys file must be a JSON object")
    out: dict[str, bytes] = {}
    for key_id, key_raw in data.items():
        kid = str(key_id).strip()
        if not kid:
            continue
        out[kid] = decode_public_key_bytes(str(key_raw))
    return out


def resolve_private_key_bytes(*, private_key_hex: str = "", private_key_file: str = "") -> bytes:
    if private_key_hex:
        key = _decode_key_material(private_key_hex)
    elif private_key_file:
        with open(os.path.expanduser(private_key_file), "r", encoding="utf-8") as f:
            key = _decode_key_material(f.read())
    else:
        raise ValueError("private key is required")
    if len(key) != 32:
        raise ValueError("ed25519 private key must be 32 bytes")
    return key


def sign_manifest(
    manifest: dict,
    *,
    private_key_bytes: bytes,
    key_id: str = "",
    expires_in_seconds: int = 0,
) -> dict:
    payload = _canonical_manifest_bytes(manifest)
    signature = _load_private_key(private_key_bytes).sign(payload)
    public_key = _public_key_bytes_from_private(private_key_bytes)

    out = copy.deepcopy(manifest)
    signed_at = int(time.time())
    out[SIGNATURE_FIELD] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": str(key_id or "default"),
        "signed_at": signed_at,
        "public_key_b64": base64.b64encode(public_key).decode("ascii"),
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    if int(expires_in_seconds or 0) > 0:
        out[SIGNATURE_FIELD]["expires_at"] = signed_at + int(expires_in_seconds)
    return out


def verify_manifest_signature(
    manifest: dict,
    *,
    trusted_public_key_bytes: bytes | None = None,
    trusted_key_id: str = "",
    trusted_keys_by_id: dict[str, bytes] | None = None,
    active_key_ids: Iterable[str] | None = None,
    max_age_seconds: int = 0,
) -> tuple[bool, str]:
    meta = manifest.get(SIGNATURE_FIELD)
    if not isinstance(meta, dict):
        return False, "manifest_signature missing"

    algo = str(meta.get("algorithm", "")).strip().lower()
    if algo != SIGNATURE_ALGORITHM:
        return False, f"unsupported signature algorithm: {algo or 'missing'}"

    key_id = str(meta.get("key_id", "")).strip()
    if trusted_key_id and key_id != trusted_key_id:
        return False, f"manifest key_id mismatch: expected {trusted_key_id}, got {key_id or 'missing'}"
    if active_key_ids is not None:
        active = {str(x).strip() for x in active_key_ids if str(x).strip()}
        if active and key_id not in active:
            return False, f"manifest key_id '{key_id or 'missing'}' is not active"

    try:
        signer_pub = base64.b64decode(str(meta.get("public_key_b64", "")))
        signature = base64.b64decode(str(meta.get("signature_b64", "")))
    except Exception as e:
        return False, f"invalid manifest signature encoding: {e}"

    if len(signer_pub) != 32:
        return False, "manifest public key must be 32 bytes"

    if trusted_public_key_bytes is not None and signer_pub != trusted_public_key_bytes:
        return False, "manifest signer public key does not match trusted key"
    if trusted_keys_by_id:
        expected = trusted_keys_by_id.get(key_id)
        if expected is None:
            return False, f"manifest key_id '{key_id or 'missing'}' is not trusted"
        if signer_pub != expected:
            return False, "manifest signer public key does not match trusted key set entry"

    if max_age_seconds and max_age_seconds > 0:
        try:
            signed_at = int(meta.get("signed_at", 0))
        except Exception:
            signed_at = 0
        if signed_at <= 0:
            return False, "manifest signed_at missing/invalid"
        age = int(time.time()) - signed_at
        if age > int(max_age_seconds):
            return False, f"manifest signature expired: age={age}s max={int(max_age_seconds)}s"
    try:
        expires_at = int(meta.get("expires_at", 0) or 0)
    except Exception:
        expires_at = 0
    if expires_at > 0 and int(time.time()) > expires_at:
        return False, f"manifest signature expired at {expires_at}"

    payload = _canonical_manifest_bytes(manifest)
    try:
        Ed25519PublicKey.from_public_bytes(signer_pub).verify(signature, payload)
    except Exception:
        return False, "manifest signature verification failed"
    return True, "ok"
