"""
Production-oriented sidecar contract for full-vocab HE projection/sampling.

This is a scaffold adapter:
- Node sends encrypted hidden-state payload + metadata to an external sidecar.
- Sidecar performs heavy HE full-vocab projection/sampling.
- Node returns sidecar artifact to client.
"""

from __future__ import annotations

import base64
import hashlib
import requests


class SidecarUnavailableError(RuntimeError):
    pass


def request_full_vocab_he_artifact(
    *,
    sidecar_url: str,
    timeout_ms: int,
    session_id: str,
    step: int,
    key_id: str,
    compute_format: str,
    compute_payload: bytes,
    top_k: int,
    temperature: float,
    top_p: float,
) -> dict:
    """
    Request HE artifact from an external sidecar.
    """
    if not sidecar_url:
        raise SidecarUnavailableError("HE sidecar URL is not configured.")

    endpoint = sidecar_url.rstrip("/") + "/v1/he/full-vocab"
    req_json = {
        "session_id": session_id,
        "step": int(step),
        "key_id": key_id,
        "compute_format": compute_format,
        "compute_payload_b64": base64.b64encode(compute_payload).decode("ascii"),
        "compute_payload_hash": hashlib.sha256(compute_payload).hexdigest(),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    try:
        resp = requests.post(endpoint, json=req_json, timeout=timeout_ms / 1000.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise SidecarUnavailableError(f"HE sidecar request failed: {exc}") from exc

    if not isinstance(data, dict):
        raise SidecarUnavailableError("HE sidecar response is not a JSON object.")
    if "he_compute_payload_b64" not in data:
        raise SidecarUnavailableError("HE sidecar response missing he_compute_payload_b64.")
    payload = base64.b64decode(str(data["he_compute_payload_b64"]))
    return {
        "he_compute_payload": payload,
        "he_compute_format": str(data.get("he_compute_format", "")),
        "he_top_k": int(data.get("he_top_k", top_k)),
        "he_error": str(data.get("he_error", "")),
        "session_id": str(data.get("session_id", session_id)),
        "step": int(data.get("step", step)),
        "key_id": str(data.get("key_id", key_id)),
        "sidecar_node_id": str(data.get("sidecar_node_id", "")),
        "sidecar_stake_identity": str(data.get("sidecar_stake_identity", "")),
    }
