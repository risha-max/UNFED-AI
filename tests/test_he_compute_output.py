import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.he_compute import (
    build_encrypted_topk_artifact,
    decrypt_topk_artifact,
    generate_client_compute_keypair,
    sample_from_topk_scores,
)


def test_he_compute_topk_roundtrip():
    priv, pub = generate_client_compute_keypair()
    logits = torch.tensor([0.1, -1.5, 2.4, 0.3, 1.8], dtype=torch.float32)
    artifact = build_encrypted_topk_artifact(
        logits_last_token=logits,
        client_public_key=pub,
        top_k=3,
        session_id="sess-he-1",
        step=2,
        key_id="kid-he-1",
    )
    token_ids, scores = decrypt_topk_artifact(
        artifact_bytes=artifact,
        client_private_key=priv,
        expected_session_id="sess-he-1",
        expected_step=2,
        expected_key_id="kid-he-1",
    )
    assert len(token_ids) == 3
    assert token_ids[0] == 2
    assert scores[0] > scores[-1]


def test_he_compute_binding_mismatch_rejected():
    priv, pub = generate_client_compute_keypair()
    logits = torch.tensor([0.2, 0.5, 0.1], dtype=torch.float32)
    artifact = build_encrypted_topk_artifact(
        logits_last_token=logits,
        client_public_key=pub,
        top_k=2,
        session_id="sess-he-2",
        step=5,
        key_id="kid-he-2",
    )
    with pytest.raises(ValueError):
        decrypt_topk_artifact(
            artifact_bytes=artifact,
            client_private_key=priv,
            expected_session_id="sess-he-2",
            expected_step=6,
            expected_key_id="kid-he-2",
        )


def test_topk_sampling_returns_valid_candidate():
    token_ids = [10, 11, 12]
    scores = [2.0, 1.0, 0.5]
    g = torch.Generator().manual_seed(42)
    sampled = sample_from_topk_scores(
        token_ids=token_ids,
        scores=scores,
        temperature=0.8,
        top_p=0.9,
        generator=g,
    )
    assert sampled in token_ids
