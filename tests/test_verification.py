"""
Unit tests for ZK verification: ticket creation, fraud proof detection,
hash comparison, and challenge-response protocol.
"""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.zk_verification import (
    create_commitment,
    verify_commitment_consistency,
    generate_challenge,
    respond_to_challenge,
    verify_challenge_response,
    ComputationCommitment,
    ZKVerificationTicket,
)


class TestCommitment:
    """Tests for computation commitments."""

    def test_create_commitment(self):
        inp = torch.randn(1, 4, 16)
        out = torch.randn(1, 4, 16)
        commitment = create_commitment(
            shard_index=0, input_tensor=inp,
            output_tensor=out, shard_weights_hash="abc123")

        assert commitment.shard_index == 0
        assert len(commitment.input_commitment) == 64  # SHA-256 hex
        assert len(commitment.output_commitment) == 64
        assert len(commitment.computation_proof) == 64
        assert commitment.tensor_shape == [1, 4, 16]

    def test_commitment_deterministic(self):
        inp = torch.randn(2, 8)
        out = torch.randn(2, 8)
        c1 = create_commitment(0, inp, out, "w")
        c2 = create_commitment(0, inp, out, "w")
        assert c1.input_commitment == c2.input_commitment
        assert c1.output_commitment == c2.output_commitment
        assert c1.computation_proof == c2.computation_proof

    def test_different_inputs_different_commitments(self):
        inp1 = torch.randn(2, 8)
        inp2 = torch.randn(2, 8)
        out = torch.randn(2, 8)
        c1 = create_commitment(0, inp1, out, "w")
        c2 = create_commitment(0, inp2, out, "w")
        assert c1.input_commitment != c2.input_commitment

    def test_verify_consistency_passes(self):
        inp = torch.randn(1, 4, 16)
        out = torch.randn(1, 4, 16)
        commitment = create_commitment(0, inp, out, "weights_hash")
        assert verify_commitment_consistency(
            commitment, inp, out, "weights_hash")

    def test_verify_consistency_fails_wrong_output(self):
        inp = torch.randn(1, 4, 16)
        out = torch.randn(1, 4, 16)
        wrong_out = torch.randn(1, 4, 16)
        commitment = create_commitment(0, inp, out, "weights_hash")
        assert not verify_commitment_consistency(
            commitment, inp, wrong_out, "weights_hash")

    def test_verify_consistency_fails_wrong_weights(self):
        inp = torch.randn(1, 4, 16)
        out = torch.randn(1, 4, 16)
        commitment = create_commitment(0, inp, out, "correct")
        assert not verify_commitment_consistency(
            commitment, inp, out, "tampered")


class TestChallengeResponse:
    """Tests for the position challenge protocol."""

    def test_generate_challenge(self):
        challenge = generate_challenge(
            output_shape=[1, 4, 16], num_positions=8)
        assert challenge.num_positions == 8
        assert len(challenge.positions) == 8
        # All positions should be valid indices
        total_elements = 1 * 4 * 16
        for pos in challenge.positions:
            assert 0 <= pos < total_elements

    def test_deterministic_with_seed(self):
        seed = b"test_seed_12345"
        c1 = generate_challenge([2, 4], num_positions=3, seed=seed)
        c2 = generate_challenge([2, 4], num_positions=3, seed=seed)
        assert c1.positions == c2.positions

    def test_respond_to_challenge(self):
        output = torch.randn(1, 4, 16)
        challenge = generate_challenge(
            output_shape=list(output.shape), num_positions=5)
        response = respond_to_challenge(challenge, output)
        assert len(response.values) == 5
        assert response.challenge_id == challenge.challenge_id

    def test_verify_honest_response(self):
        output = torch.randn(1, 4, 16)
        challenge = generate_challenge(
            output_shape=list(output.shape), num_positions=8)
        response = respond_to_challenge(challenge, output)

        passed, matched, total = verify_challenge_response(
            challenge, response, output)
        assert passed
        assert matched == total

    def test_detect_cheating(self):
        real_output = torch.randn(1, 4, 16)
        fake_output = torch.randn(1, 4, 16)

        challenge = generate_challenge(
            output_shape=list(real_output.shape), num_positions=16)
        # Node responds with fake values
        response = respond_to_challenge(challenge, fake_output)

        # Verifier checks against the real output
        passed, matched, total = verify_challenge_response(
            challenge, response, real_output)
        # Extremely unlikely all 16 random positions match between two
        # independent random tensors
        assert not passed


class TestZKVerificationTicket:
    """Tests for the ZK ticket dataclass."""

    def test_ticket_creation(self):
        inp = torch.randn(2, 8)
        out = torch.randn(2, 8)
        commitment = create_commitment(0, inp, out, "w_hash")

        ticket = ZKVerificationTicket(
            ticket_id="ticket_001",
            commitment=commitment,
        )
        assert ticket.ticket_id == "ticket_001"
        assert ticket.commitment.shard_index == 0
        assert not ticket.has_raw_data

    def test_ticket_with_challenge(self):
        inp = torch.randn(2, 8)
        out = torch.randn(2, 8)
        commitment = create_commitment(0, inp, out, "w")
        challenge = generate_challenge([2, 8], 4)
        response = respond_to_challenge(challenge, out)

        ticket = ZKVerificationTicket(
            ticket_id="ticket_002",
            commitment=commitment,
            challenge=challenge,
            response=response,
        )
        assert ticket.challenge is not None
        assert ticket.response is not None
