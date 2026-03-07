"""
Helpers for N-1/N output-MPC payload contracts.

This module defines canonical payload envelopes and metadata hashes used for
penultimate->final output handoff and final->client sampled-token response.
"""

from __future__ import annotations

import hashlib
import json

import torch

HE_COMPUTE_MODE_MPC_N_MINUS_1_N = "mpc_nminus1_n"
MPC_OUTPUT_REQUEST_FORMAT_V1 = "mpc-output-request-v1"
MPC_OUTPUT_RESPONSE_FORMAT_V1 = "mpc-output-response-v1"


def _canonical_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def build_output_mpc_request_payload(
    *,
    hidden_last_token: torch.Tensor,
    session_id: str,
    step: int,
    key_id: str,
    op: str = "penultimate_share",
    payload_type: str = "hidden_last_token",
) -> tuple[bytes, str]:
    if hidden_last_token.dim() != 1:
        raise ValueError("Expected a 1D hidden vector for output MPC request payload.")
    payload = {
        "format": MPC_OUTPUT_REQUEST_FORMAT_V1,
        "session_id": str(session_id),
        "step": int(step),
        "key_id": str(key_id or ""),
        "op": str(op),
        "payload_type": str(payload_type),
        "hidden": [float(x) for x in hidden_last_token.detach().cpu().tolist()],
    }
    payload_hash = _payload_hash(payload)
    payload["payload_hash"] = payload_hash
    return _canonical_json_bytes(payload), payload_hash


def parse_output_mpc_request_payload(
    *,
    payload_bytes: bytes,
    expected_session_id: str,
    expected_step: int,
    expected_key_id: str,
    expected_payload_hash: str = "",
) -> torch.Tensor:
    payload = json.loads(payload_bytes.decode("utf-8"))
    if payload.get("format") != MPC_OUTPUT_REQUEST_FORMAT_V1:
        raise ValueError("Unsupported output MPC request format.")
    if payload.get("session_id") != expected_session_id:
        raise ValueError("Output MPC request session mismatch.")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError("Output MPC request step mismatch.")
    if str(payload.get("key_id", "")) != str(expected_key_id or ""):
        raise ValueError("Output MPC request key mismatch.")
    claimed_hash = str(payload.get("payload_hash", ""))
    materialized = dict(payload)
    materialized.pop("payload_hash", None)
    computed_hash = _payload_hash(materialized)
    if claimed_hash != computed_hash:
        raise ValueError("Output MPC request payload hash mismatch.")
    if expected_payload_hash and expected_payload_hash != computed_hash:
        raise ValueError("Output MPC request transport hash mismatch.")
    hidden = payload.get("hidden", [])
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("Output MPC request hidden payload missing.")
    return torch.tensor([float(v) for v in hidden], dtype=torch.float32)


def build_output_mpc_response_payload(
    *,
    token_id: int,
    is_eos: bool,
    session_id: str,
    step: int,
    key_id: str,
    op: str = "final_sample",
    payload_type: str = "token_sample",
) -> tuple[bytes, str]:
    payload = {
        "format": MPC_OUTPUT_RESPONSE_FORMAT_V1,
        "session_id": str(session_id),
        "step": int(step),
        "key_id": str(key_id or ""),
        "op": str(op),
        "payload_type": str(payload_type),
        "token_id": int(token_id),
        "is_eos": bool(is_eos),
    }
    payload_hash = _payload_hash(payload)
    payload["payload_hash"] = payload_hash
    return _canonical_json_bytes(payload), payload_hash


def parse_output_mpc_response_payload(
    *,
    payload_bytes: bytes,
    expected_session_id: str,
    expected_step: int,
    expected_key_id: str,
    expected_payload_hash: str = "",
) -> tuple[int, bool]:
    payload = json.loads(payload_bytes.decode("utf-8"))
    if payload.get("format") != MPC_OUTPUT_RESPONSE_FORMAT_V1:
        raise ValueError("Unsupported output MPC response format.")
    if payload.get("session_id") != expected_session_id:
        raise ValueError("Output MPC response session mismatch.")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError("Output MPC response step mismatch.")
    if str(payload.get("key_id", "")) != str(expected_key_id or ""):
        raise ValueError("Output MPC response key mismatch.")
    claimed_hash = str(payload.get("payload_hash", ""))
    materialized = dict(payload)
    materialized.pop("payload_hash", None)
    computed_hash = _payload_hash(materialized)
    if claimed_hash != computed_hash:
        raise ValueError("Output MPC response payload hash mismatch.")
    if expected_payload_hash and expected_payload_hash != computed_hash:
        raise ValueError("Output MPC response transport hash mismatch.")
    return int(payload.get("token_id", 0)), bool(payload.get("is_eos", False))
