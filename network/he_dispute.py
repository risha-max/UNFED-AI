"""
Helpers for HE sidecar dispute reporting and deterministic audit sampling.
"""

from __future__ import annotations

import hashlib
import json

from network.share_auth import sign_bytes, verify_signature


ANOMALY_REASON_CODES = {
    "response_missing_payload",
    "response_decode_error",
    "response_format_not_allowed",
    "response_binding_mismatch",
    "response_schema_invalid",
    "response_sanity_invalid",
}

SAMPLED_AUDIT_REASON_CODE = "sampled_deep_audit"


def canonical_report_payload(payload: dict) -> bytes:
    """Stable byte representation used for report signatures."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return serialized.encode("utf-8")


def report_payload_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_report_payload(payload)).hexdigest()


def sign_report_payload(private_key: bytes, payload: dict) -> bytes:
    return sign_bytes(private_key, canonical_report_payload(payload))


def verify_report_signature(public_key: bytes, payload: dict, signature: bytes) -> bool:
    if not public_key or not signature:
        return False
    return verify_signature(public_key, canonical_report_payload(payload), signature)


def deterministic_sample(session_id: str, step: int, rate: float) -> bool:
    bounded = max(0.0, min(1.0, float(rate)))
    if bounded <= 0.0:
        return False
    if bounded >= 1.0:
        return True
    seed = f"{session_id}:{int(step)}".encode("utf-8")
    bucket = int(hashlib.sha256(seed).hexdigest()[:8], 16) % 10_000
    threshold = int(bounded * 10_000)
    return bucket < threshold


def is_anomaly_reason(reason_code: str) -> bool:
    return (reason_code or "").strip() in ANOMALY_REASON_CODES
