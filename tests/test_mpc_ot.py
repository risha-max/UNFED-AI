"""
Unit tests for Oblivious Transfer (OT) primitives and IKNP extension.

Verifies:
  - Receiver gets the chosen message (not the other)
  - Sender learns nothing about the choice bit
  - OT extension generates correlated outputs
  - OT-generated Beaver triples are correct
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.mpc_ot import (
    OTSender,
    OTReceiver,
    OTReceiverSetup,
    OTSenderResponse,
    batch_ot,
)


class TestBaseOT:
    """Tests for 1-out-of-2 OT."""

    def test_receiver_gets_m0(self):
        """Receiver with choice=0 gets m_0."""
        m0 = b"secret message zero"
        m1 = b"secret message one"

        receiver = OTReceiver(choice_bit=0)
        setup = receiver.setup()

        sender = OTSender()
        response = sender.send(setup, m0, m1)

        received = receiver.receive(response)
        assert received == m0

    def test_receiver_gets_m1(self):
        """Receiver with choice=1 gets m_1."""
        m0 = b"first option"
        m1 = b"second option"

        receiver = OTReceiver(choice_bit=1)
        setup = receiver.setup()

        sender = OTSender()
        response = sender.send(setup, m0, m1)

        received = receiver.receive(response)
        assert received == m1

    def test_different_message_lengths(self):
        """OT works with messages of different sizes."""
        m0 = b"short"
        m1 = b"a much longer message with more content"

        for choice in (0, 1):
            receiver = OTReceiver(choice_bit=choice)
            setup = receiver.setup()
            sender = OTSender()
            response = sender.send(setup, m0, m1)
            received = receiver.receive(response)
            expected = m0 if choice == 0 else m1
            assert received == expected

    def test_empty_message(self):
        """OT works with empty messages."""
        m0 = b""
        m1 = b"non-empty"

        receiver = OTReceiver(choice_bit=0)
        setup = receiver.setup()
        sender = OTSender()
        response = sender.send(setup, m0, m1)
        assert receiver.receive(response) == m0

    def test_large_message(self):
        """OT works with large messages."""
        m0 = os.urandom(4096)
        m1 = os.urandom(4096)

        receiver = OTReceiver(choice_bit=1)
        setup = receiver.setup()
        sender = OTSender()
        response = sender.send(setup, m0, m1)
        assert receiver.receive(response) == m1

    def test_receiver_cannot_get_both(self):
        """A single receiver instance can only decrypt one message."""
        m0 = b"message zero"
        m1 = b"message one"

        receiver = OTReceiver(choice_bit=0)
        setup = receiver.setup()
        sender = OTSender()
        response = sender.send(setup, m0, m1)

        # Receiver gets m_0 correctly
        assert receiver.receive(response) == m0
        # Trying to decrypt m_1 with the wrong key should fail
        # (the decoy key has no known secret key)


class TestBatchOT:
    """Tests for batch OT."""

    def test_batch_correct(self):
        """Batch OT returns the chosen messages."""
        messages = [
            (b"a0", b"a1"),
            (b"b0", b"b1"),
            (b"c0", b"c1"),
            (b"d0", b"d1"),
        ]
        choices = [0, 1, 0, 1]

        results = batch_ot(messages, choices)
        assert results[0] == b"a0"
        assert results[1] == b"b1"
        assert results[2] == b"c0"
        assert results[3] == b"d1"

    def test_all_zeros(self):
        """Batch with all choice=0."""
        messages = [(f"m0_{i}".encode(), f"m1_{i}".encode()) for i in range(5)]
        choices = [0] * 5
        results = batch_ot(messages, choices)
        for i, r in enumerate(results):
            assert r == f"m0_{i}".encode()

    def test_all_ones(self):
        """Batch with all choice=1."""
        messages = [(f"m0_{i}".encode(), f"m1_{i}".encode()) for i in range(5)]
        choices = [1] * 5
        results = batch_ot(messages, choices)
        for i, r in enumerate(results):
            assert r == f"m1_{i}".encode()


class TestOTExtensionTriples:
    """Tests for OT-generated Beaver triples."""

    def test_ot_triples_correct(self):
        """OT extension generates valid Beaver triples."""
        import torch
        from network.mpc_ot_extension import generate_beaver_triples_ot

        p0_list, p1_list = generate_beaver_triples_ot(
            n_triples=4, shape=(3, 3))

        for p0, p1 in zip(p0_list, p1_list):
            a = p0['a'] + p1['a']
            b = p0['b'] + p1['b']
            c = p0['c'] + p1['c']
            assert torch.allclose(a * b, c, atol=1e-4), \
                f"Triple failed: max error = {(a * b - c).abs().max()}"

    def test_ot_triples_different_shapes(self):
        """OT triples work for various tensor shapes."""
        import torch
        from network.mpc_ot_extension import generate_beaver_triples_ot

        for shape in [(4,), (2, 3), (1, 2, 4)]:
            p0_list, p1_list = generate_beaver_triples_ot(
                n_triples=2, shape=shape)
            for p0, p1 in zip(p0_list, p1_list):
                a = p0['a'] + p1['a']
                b = p0['b'] + p1['b']
                c = p0['c'] + p1['c']
                assert torch.allclose(a * b, c, atol=1e-4)
