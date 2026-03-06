"""
Forward-hop attestation helpers.

These helpers sign/verify a canonical payload for inter-node ForwardRequest
messages so the receiving node can verify sender authenticity and binding
before trusting activation bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from network.share_auth import sign_bytes, verify_signature


FORWARD_ATTESTATION_VERSION = "fwd-attest-v1"


@dataclass(frozen=True)
class ForwardAttestationPayload:
    version: str
    signer_node_id: str
    signer_node_type: str
    session_id: str
    he_step: int
    he_key_id: str
    activation_digest: str
    tensor_shape: str
    compressed: bool
    wire_dtype: str
    proof_format: str
    proof_hash: str


def canonical_forward_attestation_bytes(payload: ForwardAttestationPayload) -> bytes:
    return (
        f"{payload.version}|{payload.signer_node_id}|{payload.signer_node_type}|"
        f"{payload.session_id}|{payload.he_step}|{payload.he_key_id}|"
        f"{payload.activation_digest}|{payload.tensor_shape}|"
        f"{1 if payload.compressed else 0}|{payload.wire_dtype}|"
        f"{payload.proof_format}|{payload.proof_hash}"
    ).encode("utf-8")


def tensor_bytes_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def proof_bytes_hash(proof: bytes) -> str:
    return hashlib.sha256(proof).hexdigest()


def make_tensor_shape_signature(shape: list[int]) -> str:
    if not shape:
        return ""
    return ",".join(str(int(x)) for x in shape)


def sign_forward_attestation(
    private_key_bytes: bytes,
    payload: ForwardAttestationPayload,
) -> bytes:
    return sign_bytes(private_key_bytes, canonical_forward_attestation_bytes(payload))


def verify_forward_attestation(
    public_key_bytes: bytes,
    payload: ForwardAttestationPayload,
    signature: bytes,
) -> bool:
    return verify_signature(
        public_key_bytes,
        canonical_forward_attestation_bytes(payload),
        signature,
    )
