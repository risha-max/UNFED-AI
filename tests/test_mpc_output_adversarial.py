import random

import pytest
import torch

from network.mpc_output import (
    build_output_mpc_request_payload,
    build_output_mpc_response_payload,
    parse_output_mpc_request_payload,
    parse_output_mpc_response_payload,
)


def _mutate_one_byte(payload: bytes) -> bytes:
    if not payload:
        return payload
    idx = random.randint(0, len(payload) - 1)
    mutated = bytearray(payload)
    mutated[idx] ^= 0x01
    return bytes(mutated)


def test_request_payload_replay_rejected_on_step_mismatch():
    hidden = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    payload, _ = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-adv-1",
        step=4,
        key_id="kid-adv-1",
    )
    with pytest.raises(ValueError, match="step mismatch"):
        parse_output_mpc_request_payload(
            payload_bytes=payload,
            expected_session_id="sess-adv-1",
            expected_step=5,
            expected_key_id="kid-adv-1",
        )


def test_request_payload_cross_session_substitution_rejected():
    hidden = torch.tensor([0.7, -0.2], dtype=torch.float32)
    payload, _ = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-adv-2",
        step=1,
        key_id="kid-adv-2",
    )
    with pytest.raises(ValueError, match="session mismatch"):
        parse_output_mpc_request_payload(
            payload_bytes=payload,
            expected_session_id="sess-other",
            expected_step=1,
            expected_key_id="kid-adv-2",
        )


def test_response_payload_cross_key_substitution_rejected():
    payload, _ = build_output_mpc_response_payload(
        token_id=8,
        is_eos=False,
        session_id="sess-adv-3",
        step=2,
        key_id="kid-adv-3",
    )
    with pytest.raises(ValueError, match="key mismatch"):
        parse_output_mpc_response_payload(
            payload_bytes=payload,
            expected_session_id="sess-adv-3",
            expected_step=2,
            expected_key_id="kid-wrong",
        )


def test_request_payload_transport_hash_substitution_rejected():
    hidden = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)
    payload, _ = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-adv-4",
        step=3,
        key_id="kid-adv-4",
    )
    with pytest.raises(ValueError, match="transport hash mismatch"):
        parse_output_mpc_request_payload(
            payload_bytes=payload,
            expected_session_id="sess-adv-4",
            expected_step=3,
            expected_key_id="kid-adv-4",
            expected_payload_hash="00" * 32,
        )


def test_response_payload_transport_hash_substitution_rejected():
    payload, _ = build_output_mpc_response_payload(
        token_id=13,
        is_eos=True,
        session_id="sess-adv-5",
        step=9,
        key_id="kid-adv-5",
    )
    with pytest.raises(ValueError, match="transport hash mismatch"):
        parse_output_mpc_response_payload(
            payload_bytes=payload,
            expected_session_id="sess-adv-5",
            expected_step=9,
            expected_key_id="kid-adv-5",
            expected_payload_hash="ff" * 32,
        )


def test_request_payload_fuzz_mutation_rejected():
    random.seed(1234)
    hidden = torch.tensor([0.9, 0.1, -0.5], dtype=torch.float32)
    payload, _ = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-adv-6",
        step=0,
        key_id="kid-adv-6",
    )
    failures = 0
    for _ in range(40):
        mutated = _mutate_one_byte(payload)
        try:
            parse_output_mpc_request_payload(
                payload_bytes=mutated,
                expected_session_id="sess-adv-6",
                expected_step=0,
                expected_key_id="kid-adv-6",
            )
            failures += 1
        except Exception:
            pass
    # All single-byte corruptions should fail integrity/binding checks.
    assert failures == 0


def test_response_payload_fuzz_mutation_rejected():
    random.seed(5678)
    payload, _ = build_output_mpc_response_payload(
        token_id=21,
        is_eos=False,
        session_id="sess-adv-7",
        step=11,
        key_id="kid-adv-7",
    )
    failures = 0
    for _ in range(40):
        mutated = _mutate_one_byte(payload)
        try:
            parse_output_mpc_response_payload(
                payload_bytes=mutated,
                expected_session_id="sess-adv-7",
                expected_step=11,
                expected_key_id="kid-adv-7",
            )
            failures += 1
        except Exception:
            pass
    assert failures == 0
