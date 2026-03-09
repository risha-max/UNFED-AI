import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.mpc_output import (
    OUTPUT_2PC_HIDDEN_CODEC_FP16,
    OUTPUT_2PC_HIDDEN_CODEC_INT8,
    OUTPUT_2PC_REQUEST_FORMAT_V2,
    OUTPUT_2PC_RESPONSE_FORMAT_V2,
    build_output_2pc_request_artifact,
    build_output_2pc_response_artifact,
    parse_output_2pc_request_artifact,
    parse_output_2pc_response_artifact,
)


def test_output_2pc_request_roundtrip():
    hidden = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    payload, payload_hash = build_output_2pc_request_artifact(
        hidden_last_token=hidden,
        session_id="sess-out-1",
        step=3,
        key_id="kid-out-1",
    )
    parsed = parse_output_2pc_request_artifact(
        artifact_bytes=payload,
        expected_session_id="sess-out-1",
        expected_step=3,
        expected_key_id="kid-out-1",
        expected_artifact_hash=payload_hash,
    )
    assert parsed.shape == hidden.shape
    assert torch.allclose(parsed, hidden, atol=1e-6)


def test_output_2pc_response_roundtrip():
    payload, payload_hash = build_output_2pc_response_artifact(
        token_id=42,
        is_eos=False,
        session_id="sess-out-2",
        step=7,
        key_id="kid-out-2",
    )
    token_id, is_eos = parse_output_2pc_response_artifact(
        artifact_bytes=payload,
        expected_session_id="sess-out-2",
        expected_step=7,
        expected_key_id="kid-out-2",
        expected_artifact_hash=payload_hash,
    )
    assert token_id == 42
    assert is_eos is False


def test_output_2pc_request_rejects_hash_mismatch():
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float32)
    payload, _ = build_output_2pc_request_artifact(
        hidden_last_token=hidden,
        session_id="sess-out-3",
        step=1,
        key_id="kid-out-3",
    )
    with pytest.raises(ValueError):
        parse_output_2pc_request_artifact(
            artifact_bytes=payload,
            expected_session_id="sess-out-3",
            expected_step=1,
            expected_key_id="kid-out-3",
            expected_artifact_hash="deadbeef",
        )


def test_output_2pc_request_roundtrip_binary_fp16():
    hidden = torch.tensor([0.125, -0.75, 1.5, 2.25], dtype=torch.float32)
    payload, payload_hash = build_output_2pc_request_artifact(
        hidden_last_token=hidden,
        session_id="sess-out-v2-fp16",
        step=5,
        key_id="kid-out-v2-fp16",
        wire_format=OUTPUT_2PC_REQUEST_FORMAT_V2,
        quantization_mode=OUTPUT_2PC_HIDDEN_CODEC_FP16,
    )
    parsed = parse_output_2pc_request_artifact(
        artifact_bytes=payload,
        expected_session_id="sess-out-v2-fp16",
        expected_step=5,
        expected_key_id="kid-out-v2-fp16",
        expected_artifact_hash=payload_hash,
        wire_format=OUTPUT_2PC_REQUEST_FORMAT_V2,
    )
    assert parsed.shape == hidden.shape
    assert torch.allclose(parsed, hidden, atol=1e-3)


def test_output_2pc_request_roundtrip_binary_int8_and_response_v2():
    hidden = torch.linspace(-1.0, 1.0, steps=32, dtype=torch.float32)
    payload, payload_hash = build_output_2pc_request_artifact(
        hidden_last_token=hidden,
        session_id="sess-out-v2-int8",
        step=6,
        key_id="kid-out-v2-int8",
        wire_format=OUTPUT_2PC_REQUEST_FORMAT_V2,
        quantization_mode=OUTPUT_2PC_HIDDEN_CODEC_INT8,
    )
    parsed = parse_output_2pc_request_artifact(
        artifact_bytes=payload,
        expected_session_id="sess-out-v2-int8",
        expected_step=6,
        expected_key_id="kid-out-v2-int8",
        expected_artifact_hash=payload_hash,
        wire_format=OUTPUT_2PC_REQUEST_FORMAT_V2,
    )
    assert parsed.shape == hidden.shape
    assert torch.allclose(parsed, hidden, atol=0.02)

    resp_payload, resp_hash = build_output_2pc_response_artifact(
        token_id=123,
        is_eos=True,
        session_id="sess-out-v2-int8",
        step=6,
        key_id="kid-out-v2-int8",
        wire_format=OUTPUT_2PC_RESPONSE_FORMAT_V2,
    )
    token_id, is_eos = parse_output_2pc_response_artifact(
        artifact_bytes=resp_payload,
        expected_session_id="sess-out-v2-int8",
        expected_step=6,
        expected_key_id="kid-out-v2-int8",
        expected_artifact_hash=resp_hash,
        wire_format=OUTPUT_2PC_RESPONSE_FORMAT_V2,
    )
    assert token_id == 123
    assert is_eos is True
