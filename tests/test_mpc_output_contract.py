import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.mpc_output import (
    build_output_mpc_request_payload,
    build_output_mpc_response_payload,
    parse_output_mpc_request_payload,
    parse_output_mpc_response_payload,
)


def test_output_mpc_request_roundtrip():
    hidden = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    payload, payload_hash = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-out-1",
        step=3,
        key_id="kid-out-1",
    )
    parsed = parse_output_mpc_request_payload(
        payload_bytes=payload,
        expected_session_id="sess-out-1",
        expected_step=3,
        expected_key_id="kid-out-1",
        expected_payload_hash=payload_hash,
    )
    assert parsed.shape == hidden.shape
    assert torch.allclose(parsed, hidden, atol=1e-6)


def test_output_mpc_response_roundtrip():
    payload, payload_hash = build_output_mpc_response_payload(
        token_id=42,
        is_eos=False,
        session_id="sess-out-2",
        step=7,
        key_id="kid-out-2",
    )
    token_id, is_eos = parse_output_mpc_response_payload(
        payload_bytes=payload,
        expected_session_id="sess-out-2",
        expected_step=7,
        expected_key_id="kid-out-2",
        expected_payload_hash=payload_hash,
    )
    assert token_id == 42
    assert is_eos is False


def test_output_mpc_request_rejects_hash_mismatch():
    hidden = torch.tensor([1.0, 2.0], dtype=torch.float32)
    payload, _ = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-out-3",
        step=1,
        key_id="kid-out-3",
    )
    with pytest.raises(ValueError):
        parse_output_mpc_request_payload(
            payload_bytes=payload,
            expected_session_id="sess-out-3",
            expected_step=1,
            expected_key_id="kid-out-3",
            expected_payload_hash="deadbeef",
        )
