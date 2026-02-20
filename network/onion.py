"""
Onion Routing — Tor-style layered encryption for inference circuits.

How it works:
  1. Each node has a public/private key pair (X25519 for key exchange).
  2. The client fetches all node public keys from the registry.
  3. The client builds an onion-encrypted circuit:
     - Innermost layer: encrypted for the last node (contains "you are the exit")
     - Next layer: encrypted for the second-to-last node (contains next hop + inner onion)
     - ... and so on until the outermost layer is encrypted for the entry node
  4. Each node decrypts its layer, learns only its next hop, and forwards.
  5. No single node sees the full circuit.

Encryption scheme:
  - X25519 key exchange for ephemeral shared secrets
  - AES-256-GCM for authenticated encryption of each onion layer
  - Each layer contains: next_hop_address + encrypted_payload_for_next_node

This gives us:
  - Entry node: sees the user's IP but NOT the query content (just encrypted blobs)
  - Middle nodes: see neither the user's IP nor the full query
  - Exit node: sees the query content (activations) but NOT the user's IP
"""

import base64
import json
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# --- Key Management ---

def generate_keypair() -> tuple[X25519PrivateKey, X25519PublicKey]:
    """Generate a new X25519 key pair for a node."""
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def public_key_to_bytes(key: X25519PublicKey) -> bytes:
    """Serialize a public key to raw bytes (32 bytes)."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    return key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def public_key_from_bytes(data: bytes) -> X25519PublicKey:
    """Deserialize a public key from raw bytes."""
    return X25519PublicKey.from_public_bytes(data)


def private_key_to_bytes(key: X25519PrivateKey) -> bytes:
    """Serialize a private key to raw bytes (32 bytes)."""
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    return key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def private_key_from_bytes(data: bytes) -> X25519PrivateKey:
    """Deserialize a private key from raw bytes."""
    return X25519PrivateKey.from_private_bytes(data)


# --- Key Exchange & Encryption ---

def derive_shared_key(private_key: X25519PrivateKey, peer_public_key: X25519PublicKey) -> bytes:
    """
    Derive a 256-bit AES key from an X25519 key exchange.
    Uses HKDF for proper key derivation.
    """
    shared_secret = private_key.exchange(peer_public_key)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"unfed-onion-routing",
    ).derive(shared_secret)
    return derived_key


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns nonce + ciphertext (nonce is 12 bytes)."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt(key: bytes, data: bytes) -> bytes:
    """AES-256-GCM decrypt. Expects nonce (12 bytes) + ciphertext."""
    nonce = data[:12]
    ciphertext = data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# --- Onion Layer Format ---
# Each onion layer, when decrypted, contains:
#   {
#     "next_hop": "address:port" or "" (if this is the exit node),
#     "payload": "<base64 encrypted blob for the next node>"
#   }
# For the exit node, "payload" is empty.


@dataclass
class OnionLayer:
    """A single decrypted onion layer."""
    next_hop: str       # Address of the next node, or "" if exit
    payload: bytes      # Encrypted blob to forward to the next node


def build_onion(circuit_addresses: list[str],
                circuit_public_keys: list[X25519PublicKey]) -> tuple[bytes, list[bytes]]:
    """
    Build an onion-encrypted circuit.

    Args:
        circuit_addresses: Ordered list of node addresses [entry, ..., exit]
        circuit_public_keys: Corresponding public keys for each node

    Returns:
        (onion_blob, ephemeral_public_keys)
        - onion_blob: The outermost encrypted blob to send to the entry node
        - ephemeral_public_keys: One ephemeral public key per layer (sent alongside
          the onion so nodes can derive the shared secret)
    """
    assert len(circuit_addresses) == len(circuit_public_keys)
    n = len(circuit_addresses)

    # Build from inside out
    ephemeral_keys_private = []
    ephemeral_keys_public = []

    # Generate one ephemeral key pair per node
    for _ in range(n):
        eph_priv, eph_pub = generate_keypair()
        ephemeral_keys_private.append(eph_priv)
        ephemeral_keys_public.append(eph_pub)

    # Start with the innermost layer (exit node)
    inner_payload = b""  # exit node has no further payload

    # Build layers from exit to entry
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            # Exit node: no next hop
            layer = json.dumps({
                "next_hop": "",
                "payload": "",
            }).encode()
        else:
            # Intermediate node: contains next hop + encrypted inner payload
            layer = json.dumps({
                "next_hop": circuit_addresses[i + 1],
                "payload": base64.b64encode(inner_payload).decode(),
                "ephemeral_key": base64.b64encode(
                    public_key_to_bytes(ephemeral_keys_public[i + 1])
                ).decode(),
            }).encode()

        # Encrypt this layer for node i
        shared_key = derive_shared_key(ephemeral_keys_private[i], circuit_public_keys[i])
        inner_payload = encrypt(shared_key, layer)

    # The final inner_payload is the outermost onion blob
    outermost_ephemeral_key = public_key_to_bytes(ephemeral_keys_public[0])
    return inner_payload, outermost_ephemeral_key


# --- Return-Path Encryption ---
# Each node encrypts the response with a symmetric key shared with the client.
# The response travels back through the pipeline with layered encryption:
#   Node 3 → encrypt(key3, token)
#   Node 2 → encrypt(key2, encrypted_by_3)
#   Node 1 → encrypt(key1, encrypted_by_2_and_3)
#   Node 0 → encrypt(key0, encrypted_by_1_2_3)
# Client decrypts all layers: decrypt(key0, decrypt(key1, decrypt(key2, decrypt(key3, blob))))


def generate_response_keys(num_nodes: int) -> tuple[list[bytes], list[bytes]]:
    """
    Generate symmetric AES-256 keys for return-path encryption.

    Returns:
        (keys_for_client, keys_for_nodes)
        Both lists are the same keys — the client keeps all of them to decrypt,
        and each node receives only its own key via the request.
    """
    keys = [os.urandom(32) for _ in range(num_nodes)]
    return keys, keys


def encrypt_response(key: bytes, data: bytes) -> bytes:
    """
    Encrypt a response payload with AES-256-GCM.

    Called by each node on the return path. The node wraps the
    (possibly already encrypted) response in another encryption layer.
    """
    return encrypt(key, data)


def decrypt_response_layers(keys: list[bytes], encrypted_blob: bytes) -> bytes:
    """
    Decrypt all return-path encryption layers.

    Called by the client. Peels layers in reverse order (outermost first):
      keys[0] was applied last (by the first node), so decrypt it first.
      keys[-1] was applied first (by the last node), decrypt it last.

    Args:
        keys: Symmetric keys in circuit order [node0_key, node1_key, ..., nodeN_key]
        encrypted_blob: The fully-encrypted response blob

    Returns:
        The original plaintext response data
    """
    data = encrypted_blob
    for key in keys:
        data = decrypt(key, data)
    return data


def peel_onion(private_key: X25519PrivateKey, ephemeral_key_bytes: bytes,
               onion_blob: bytes) -> tuple[OnionLayer, bytes]:
    """
    Peel one layer off the onion (called by a node).

    Args:
        private_key: This node's private key
        ephemeral_key_bytes: The ephemeral public key for this layer (32 bytes)
        onion_blob: The encrypted blob to decrypt

    Returns:
        (OnionLayer, next_ephemeral_key_bytes)
        - OnionLayer with next_hop and payload for the next node
        - next_ephemeral_key_bytes: ephemeral public key for the next layer (empty if exit)
    """
    ephemeral_pub = public_key_from_bytes(ephemeral_key_bytes)
    shared_key = derive_shared_key(private_key, ephemeral_pub)
    plaintext = decrypt(shared_key, onion_blob)

    layer_data = json.loads(plaintext)
    next_hop = layer_data.get("next_hop", "")
    payload_b64 = layer_data.get("payload", "")
    next_ephemeral_b64 = layer_data.get("ephemeral_key", "")

    payload = base64.b64decode(payload_b64) if payload_b64 else b""
    next_ephemeral = base64.b64decode(next_ephemeral_b64) if next_ephemeral_b64 else b""

    return OnionLayer(next_hop=next_hop, payload=payload), next_ephemeral
