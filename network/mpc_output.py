"""
Helpers for full output-stage 2PC artifact contracts.

This module defines canonical artifact envelopes and metadata hashes used for
penultimate->final output handoff and final->client sampled-token response.
"""

from __future__ import annotations

import hashlib
import json
import struct

import torch

HE_COMPUTE_MODE_FULL_OUTPUT_2PC = "full_output_2pc"
OUTPUT_2PC_REQUEST_FORMAT_V1 = "output-2pc-request-v1"
OUTPUT_2PC_RESPONSE_FORMAT_V1 = "output-2pc-response-v1"
OUTPUT_2PC_REQUEST_FORMAT_V2 = "output-2pc-request-bin-v2"
OUTPUT_2PC_RESPONSE_FORMAT_V2 = "output-2pc-response-bin-v2"
OUTPUT_2PC_HIDDEN_CODEC_FP32 = "fp32"
OUTPUT_2PC_HIDDEN_CODEC_FP16 = "fp16"
OUTPUT_2PC_HIDDEN_CODEC_INT8 = "int8"

_CODEC_FP32 = 0
_CODEC_FP16 = 1
_CODEC_INT8 = 2


def _canonical_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _artifact_hash(payload: dict) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _artifact_hash_v2(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _hash_prefix(
    *,
    wire_format: str,
    session_id: str,
    step: int,
    key_id: str,
    stage: str,
    artifact_type: str,
) -> list[bytes]:
    return [
        wire_format.encode("utf-8"),
        b"|",
        str(session_id).encode("utf-8"),
        b"|",
        str(int(step)).encode("utf-8"),
        b"|",
        str(key_id or "").encode("utf-8"),
        b"|",
        str(stage).encode("utf-8"),
        b"|",
        str(artifact_type).encode("utf-8"),
        b"|",
    ]


def build_output_2pc_request_artifact(
    *,
    hidden_last_token: torch.Tensor,
    session_id: str,
    step: int,
    key_id: str,
    stage: str = "penultimate_share",
    artifact_type: str = "hidden_last_token",
    wire_format: str = OUTPUT_2PC_REQUEST_FORMAT_V1,
    quantization_mode: str = OUTPUT_2PC_HIDDEN_CODEC_FP32,
) -> tuple[bytes, str]:
    if hidden_last_token.dim() != 1:
        raise ValueError("Expected a 1D hidden vector for output 2PC request artifact.")
    if wire_format == OUTPUT_2PC_REQUEST_FORMAT_V2:
        qmode = str(quantization_mode or OUTPUT_2PC_HIDDEN_CODEC_FP32).strip().lower()
        if qmode not in (
            OUTPUT_2PC_HIDDEN_CODEC_FP32,
            OUTPUT_2PC_HIDDEN_CODEC_FP16,
            OUTPUT_2PC_HIDDEN_CODEC_INT8,
        ):
            raise ValueError(f"Unsupported output 2PC hidden codec: {quantization_mode}")
        hidden_cpu = hidden_last_token.detach().to(device="cpu").contiguous()
        dim = int(hidden_cpu.numel())
        if qmode == OUTPUT_2PC_HIDDEN_CODEC_FP16:
            hidden_raw = hidden_cpu.to(dtype=torch.float16).numpy().tobytes()
            payload = struct.pack("<BI", _CODEC_FP16, dim) + hidden_raw
        elif qmode == OUTPUT_2PC_HIDDEN_CODEC_INT8:
            hidden_f32 = hidden_cpu.to(dtype=torch.float32)
            max_abs = float(hidden_f32.abs().max().item())
            scale = max(max_abs / 127.0, 1e-12)
            q = torch.clamp(torch.round(hidden_f32 / scale), -127, 127).to(torch.int8)
            payload = (
                struct.pack("<BI", _CODEC_INT8, dim)
                + struct.pack("<f", float(scale))
                + q.numpy().tobytes()
            )
        else:
            hidden_raw = hidden_cpu.to(dtype=torch.float32).numpy().tobytes()
            payload = struct.pack("<BI", _CODEC_FP32, dim) + hidden_raw
        artifact_hash = _artifact_hash_v2(
            _hash_prefix(
                wire_format=wire_format,
                session_id=session_id,
                step=step,
                key_id=key_id,
                stage=stage,
                artifact_type=artifact_type,
            )
            + [payload]
        )
        return payload, artifact_hash
    payload = {
        "format": OUTPUT_2PC_REQUEST_FORMAT_V1,
        "session_id": str(session_id),
        "step": int(step),
        "key_id": str(key_id or ""),
        "stage": str(stage),
        "artifact_type": str(artifact_type),
        "hidden": [float(x) for x in hidden_last_token.detach().cpu().tolist()],
    }
    artifact_hash = _artifact_hash(payload)
    payload["artifact_hash"] = artifact_hash
    return _canonical_json_bytes(payload), artifact_hash


def parse_output_2pc_request_artifact(
    *,
    artifact_bytes: bytes,
    expected_session_id: str,
    expected_step: int,
    expected_key_id: str,
    expected_artifact_hash: str = "",
    wire_format: str = "",
) -> torch.Tensor:
    fmt = str(wire_format or "").strip()
    if not fmt:
        fmt = OUTPUT_2PC_REQUEST_FORMAT_V1 if artifact_bytes[:1] == b"{" else OUTPUT_2PC_REQUEST_FORMAT_V2
    if fmt == OUTPUT_2PC_REQUEST_FORMAT_V2:
        if len(artifact_bytes) < 5:
            raise ValueError("Output 2PC request binary payload too short.")
        codec, dim = struct.unpack("<BI", artifact_bytes[:5])
        payload_body = artifact_bytes[5:]
        if codec == _CODEC_FP32:
            raw = payload_body
            if dim <= 0 or len(raw) != dim * 4:
                raise ValueError("Output 2PC request binary fp32 payload shape mismatch.")
            vec = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        elif codec == _CODEC_FP16:
            raw = payload_body
            if dim <= 0 or len(raw) != dim * 2:
                raise ValueError("Output 2PC request binary fp16 payload shape mismatch.")
            vec = torch.frombuffer(bytearray(raw), dtype=torch.float16).to(torch.float32).clone()
        elif codec == _CODEC_INT8:
            if len(payload_body) < 4:
                raise ValueError("Output 2PC request binary int8 payload too short.")
            (scale,) = struct.unpack("<f", payload_body[:4])
            raw = payload_body[4:]
            if dim <= 0 or len(raw) != dim:
                raise ValueError("Output 2PC request binary int8 payload shape mismatch.")
            q = torch.frombuffer(bytearray(raw), dtype=torch.int8).to(torch.float32)
            vec = (q * float(scale)).clone()
        else:
            # Backward-compat: first implementation packed <I dim> + fp32 bytes.
            legacy_dim = struct.unpack("<I", artifact_bytes[:4])[0]
            legacy_raw = artifact_bytes[4:]
            if legacy_dim <= 0 or len(legacy_raw) != legacy_dim * 4:
                raise ValueError("Output 2PC request binary payload codec mismatch.")
            vec = torch.frombuffer(bytearray(legacy_raw), dtype=torch.float32).clone()
        computed_hash = _artifact_hash_v2(
            _hash_prefix(
                wire_format=OUTPUT_2PC_REQUEST_FORMAT_V2,
                session_id=expected_session_id,
                step=int(expected_step),
                key_id=expected_key_id,
                stage="penultimate_share",
                artifact_type="hidden_last_token",
            )
            + [artifact_bytes]
        )
        if expected_artifact_hash and expected_artifact_hash != computed_hash:
            raise ValueError("Output 2PC request transport hash mismatch.")
        return vec
    payload = json.loads(artifact_bytes.decode("utf-8"))
    if payload.get("format") != OUTPUT_2PC_REQUEST_FORMAT_V1:
        raise ValueError("Unsupported output 2PC request format.")
    if payload.get("session_id") != expected_session_id:
        raise ValueError("Output 2PC request session mismatch.")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError("Output 2PC request step mismatch.")
    if str(payload.get("key_id", "")) != str(expected_key_id or ""):
        raise ValueError("Output 2PC request key mismatch.")
    claimed_hash = str(payload.get("artifact_hash", ""))
    materialized = dict(payload)
    materialized.pop("artifact_hash", None)
    computed_hash = _artifact_hash(materialized)
    if claimed_hash != computed_hash:
        raise ValueError("Output 2PC request artifact hash mismatch.")
    if expected_artifact_hash and expected_artifact_hash != computed_hash:
        raise ValueError("Output 2PC request transport hash mismatch.")
    hidden = payload.get("hidden", [])
    if not isinstance(hidden, list) or not hidden:
        raise ValueError("Output 2PC request hidden payload missing.")
    return torch.tensor([float(v) for v in hidden], dtype=torch.float32)


def build_output_2pc_response_artifact(
    *,
    token_id: int,
    is_eos: bool,
    session_id: str,
    step: int,
    key_id: str,
    stage: str = "final_sample",
    artifact_type: str = "token_sample",
    wire_format: str = OUTPUT_2PC_RESPONSE_FORMAT_V1,
) -> tuple[bytes, str]:
    if wire_format == OUTPUT_2PC_RESPONSE_FORMAT_V2:
        payload = struct.pack("<qB", int(token_id), 1 if bool(is_eos) else 0)
        artifact_hash = _artifact_hash_v2(
            _hash_prefix(
                wire_format=wire_format,
                session_id=session_id,
                step=step,
                key_id=key_id,
                stage=stage,
                artifact_type=artifact_type,
            )
            + [payload]
        )
        return payload, artifact_hash
    payload = {
        "format": OUTPUT_2PC_RESPONSE_FORMAT_V1,
        "session_id": str(session_id),
        "step": int(step),
        "key_id": str(key_id or ""),
        "stage": str(stage),
        "artifact_type": str(artifact_type),
        "token_id": int(token_id),
        "is_eos": bool(is_eos),
    }
    artifact_hash = _artifact_hash(payload)
    payload["artifact_hash"] = artifact_hash
    return _canonical_json_bytes(payload), artifact_hash


def parse_output_2pc_response_artifact(
    *,
    artifact_bytes: bytes,
    expected_session_id: str,
    expected_step: int,
    expected_key_id: str,
    expected_artifact_hash: str = "",
    wire_format: str = "",
) -> tuple[int, bool]:
    fmt = str(wire_format or "").strip()
    if not fmt:
        fmt = OUTPUT_2PC_RESPONSE_FORMAT_V1 if artifact_bytes[:1] == b"{" else OUTPUT_2PC_RESPONSE_FORMAT_V2
    if fmt == OUTPUT_2PC_RESPONSE_FORMAT_V2:
        if len(artifact_bytes) != 9:
            raise ValueError("Output 2PC response binary payload length mismatch.")
        token_id, eos = struct.unpack("<qB", artifact_bytes)
        computed_hash = _artifact_hash_v2(
            _hash_prefix(
                wire_format=OUTPUT_2PC_RESPONSE_FORMAT_V2,
                session_id=expected_session_id,
                step=int(expected_step),
                key_id=expected_key_id,
                stage="final_sample",
                artifact_type="token_sample",
            )
            + [artifact_bytes]
        )
        if expected_artifact_hash and expected_artifact_hash != computed_hash:
            raise ValueError("Output 2PC response transport hash mismatch.")
        return int(token_id), bool(eos)
    payload = json.loads(artifact_bytes.decode("utf-8"))
    if payload.get("format") != OUTPUT_2PC_RESPONSE_FORMAT_V1:
        raise ValueError("Unsupported output 2PC response format.")
    if payload.get("session_id") != expected_session_id:
        raise ValueError("Output 2PC response session mismatch.")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError("Output 2PC response step mismatch.")
    if str(payload.get("key_id", "")) != str(expected_key_id or ""):
        raise ValueError("Output 2PC response key mismatch.")
    claimed_hash = str(payload.get("artifact_hash", ""))
    materialized = dict(payload)
    materialized.pop("artifact_hash", None)
    computed_hash = _artifact_hash(materialized)
    if claimed_hash != computed_hash:
        raise ValueError("Output 2PC response artifact hash mismatch.")
    if expected_artifact_hash and expected_artifact_hash != computed_hash:
        raise ValueError("Output 2PC response transport hash mismatch.")
    return int(payload.get("token_id", 0)), bool(payload.get("is_eos", False))

