"""
Oblivious Transfer (OT) primitives for MPC Beaver triple generation.

Implements a Diffie-Hellman-based 1-out-of-2 OT using X25519 key exchange
so neither MPC party needs to trust a dealer for triple generation.

Protocol (Naor-Pinkas simplified OT via X25519):

  Receiver has choice bit b and wants to learn m_b without revealing b.

  1. Receiver picks a real keypair (sk, pk_b).
  2. Receiver generates a decoy pk_{1-b} = random 32 bytes (no known sk).
  3. Receiver sends (pk_0, pk_1) to Sender — Sender can't tell which is real.
  4. Sender encrypts m_0 under pk_0 and m_1 under pk_1.
  5. Receiver can only decrypt m_b (only has sk for pk_b).

Crypto: X25519 for DH key exchange + AES-256-GCM for encryption (same
primitives already used for onion routing).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _derive_key(private_key: X25519PrivateKey,
                peer_public_key: X25519PublicKey,
                context: bytes = b"unfed-ot") -> bytes:
    """Derive a 256-bit AES key from an X25519 key exchange."""
    shared_secret = private_key.exchange(peer_public_key)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context,
    ).derive(shared_secret)


def _encrypt(key: bytes, plaintext: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns nonce (12 bytes) + ciphertext."""
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, None)


def _decrypt(key: bytes, data: bytes) -> bytes:
    """AES-256-GCM decrypt. Expects nonce (12 bytes) + ciphertext."""
    return AESGCM(key).decrypt(data[:12], data[12:], None)


def _pub_bytes(key: X25519PublicKey) -> bytes:
    return key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def _pub_from_bytes(data: bytes) -> X25519PublicKey:
    return X25519PublicKey.from_public_bytes(data)


# ---------------------------------------------------------------------------
# Base OT data structures
# ---------------------------------------------------------------------------

@dataclass
class OTReceiverSetup:
    """Receiver's setup message (sent to Sender)."""
    pk_0: bytes  # 32-byte public key for slot 0
    pk_1: bytes  # 32-byte public key for slot 1


@dataclass
class OTSenderResponse:
    """Sender's encrypted messages."""
    ct_0: bytes  # Ciphertext for m_0
    ct_1: bytes  # Ciphertext for m_1


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------

class OTReceiver:
    """
    1-out-of-2 OT receiver.

    Usage:
        receiver = OTReceiver(choice_bit=1)
        setup = receiver.setup()                     # send to sender
        m_b = receiver.receive(sender_response)      # decrypt chosen message
    """

    def __init__(self, choice_bit: int):
        assert choice_bit in (0, 1)
        self._choice = choice_bit
        self._sk = X25519PrivateKey.generate()
        self._pk_real = self._sk.public_key()
        # Decoy key — random 32 bytes, nobody knows the private key
        self._pk_decoy = X25519PublicKey.from_public_bytes(os.urandom(32))

    def setup(self) -> OTReceiverSetup:
        """Generate the receiver's setup message."""
        if self._choice == 0:
            return OTReceiverSetup(
                pk_0=_pub_bytes(self._pk_real),
                pk_1=_pub_bytes(self._pk_decoy),
            )
        else:
            return OTReceiverSetup(
                pk_0=_pub_bytes(self._pk_decoy),
                pk_1=_pub_bytes(self._pk_real),
            )

    def receive(self, response: OTSenderResponse) -> bytes:
        """Decrypt the chosen message."""
        ct = response.ct_0 if self._choice == 0 else response.ct_1
        # Derive key from our real SK + the sender's ephemeral PK
        # The sender encrypted under our pk, so we can decrypt
        # Actually, sender uses an ephemeral key — the ct includes
        # sender's ephemeral pk (first 32 bytes) + actual ciphertext
        sender_eph_pk = _pub_from_bytes(ct[:32])
        key = _derive_key(self._sk, sender_eph_pk)
        return _decrypt(key, ct[32:])


# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

class OTSender:
    """
    1-out-of-2 OT sender.

    Usage:
        sender = OTSender()
        response = sender.send(receiver_setup, m_0, m_1)  # send to receiver
    """

    def send(self, setup: OTReceiverSetup,
             m_0: bytes, m_1: bytes) -> OTSenderResponse:
        """Encrypt both messages under the receiver's public keys."""
        ct_0 = self._encrypt_for(setup.pk_0, m_0)
        ct_1 = self._encrypt_for(setup.pk_1, m_1)
        return OTSenderResponse(ct_0=ct_0, ct_1=ct_1)

    @staticmethod
    def _encrypt_for(receiver_pk_bytes: bytes, msg: bytes) -> bytes:
        """Encrypt a message for a receiver public key.

        Returns: sender_ephemeral_pk (32 bytes) + nonce (12) + ciphertext.
        """
        eph_sk = X25519PrivateKey.generate()
        eph_pk = eph_sk.public_key()
        receiver_pk = _pub_from_bytes(receiver_pk_bytes)
        key = _derive_key(eph_sk, receiver_pk)
        ct = _encrypt(key, msg)
        return _pub_bytes(eph_pk) + ct


# ---------------------------------------------------------------------------
# Convenience: batch OT for multiple bits
# ---------------------------------------------------------------------------

def batch_ot(sender_messages: list[tuple[bytes, bytes]],
             choice_bits: list[int]) -> list[bytes]:
    """
    Run multiple independent 1-out-of-2 OTs locally (for testing/demo).

    Args:
        sender_messages: List of (m_0, m_1) pairs from the sender.
        choice_bits: List of choice bits from the receiver.

    Returns:
        List of received messages (one per OT instance).
    """
    assert len(sender_messages) == len(choice_bits)
    results = []
    sender = OTSender()
    for (m0, m1), bit in zip(sender_messages, choice_bits):
        receiver = OTReceiver(choice_bit=bit)
        setup = receiver.setup()
        response = sender.send(setup, m0, m1)
        results.append(receiver.receive(response))
    return results
