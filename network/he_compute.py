"""
HE compute helpers for output-layer decode artifacts.

This module provides a practical homomorphic path for output artifacts using
Paillier additive HE (via `phe`). The final node can return encrypted top-k
logit candidates while the client performs decryption and sampling locally.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from phe import paillier


HE_COMPUTE_MODE_OFF = "off"
HE_COMPUTE_MODE_DECODE_CLIENT_SAMPLE = "decode_client_sample"
HE_COMPUTE_MODE_MPC_N_MINUS_1_N = "mpc_nminus1_n"
HE_COMPUTE_FORMAT_PAILLIER_V1 = "paillier-topk-v1"
HE_COMPUTE_FORMAT_PAILLIER_HIDDEN_V1 = "paillier-hidden-v1"


def get_he_compute_mode() -> str:
    mode = os.environ.get("UNFED_HE_COMPUTE_MODE", HE_COMPUTE_MODE_OFF).strip()
    if mode in (
        HE_COMPUTE_MODE_OFF,
        HE_COMPUTE_MODE_DECODE_CLIENT_SAMPLE,
        HE_COMPUTE_MODE_MPC_N_MINUS_1_N,
    ):
        return mode
    return HE_COMPUTE_MODE_OFF


def generate_client_compute_keypair() -> tuple[bytes, bytes]:
    """Return (private_key_bytes, public_key_bytes) for HE-compute session."""
    public_key, private_key = paillier.generate_paillier_keypair()
    pub_payload = {
        "n": str(public_key.n),
    }
    priv_payload = {
        "p": str(private_key.p),
        "q": str(private_key.q),
        "public_n": str(public_key.n),
    }
    return (
        json.dumps(priv_payload, separators=(",", ":")).encode("utf-8"),
        json.dumps(pub_payload, separators=(",", ":")).encode("utf-8"),
    )


def _load_public_key(public_key_bytes: bytes) -> paillier.PaillierPublicKey:
    data = json.loads(public_key_bytes.decode("utf-8"))
    n_val = int(data["n"])
    return paillier.PaillierPublicKey(n=n_val)


def _load_private_key(private_key_bytes: bytes) -> paillier.PaillierPrivateKey:
    data = json.loads(private_key_bytes.decode("utf-8"))
    public_key = paillier.PaillierPublicKey(n=int(data["public_n"]))
    return paillier.PaillierPrivateKey(
        public_key=public_key,
        p=int(data["p"]),
        q=int(data["q"]),
    )


def build_encrypted_topk_artifact(
    *,
    logits_last_token: torch.Tensor,
    client_public_key: bytes,
    top_k: int,
    session_id: str,
    step: int,
    key_id: str,
) -> bytes:
    """
    Build encrypted top-k artifact from a final-token logits vector.
    """
    if logits_last_token.dim() != 1:
        raise ValueError("Expected 1D logits tensor for final token.")
    vocab = int(logits_last_token.shape[0])
    k = max(1, min(int(top_k), vocab))
    values, indices = torch.topk(logits_last_token, k=k, dim=-1)
    pub = _load_public_key(client_public_key)

    enc_scores = []
    for score in values.detach().cpu().tolist():
        enc = pub.encrypt(float(score))
        enc_scores.append(
            {
                "ciphertext": str(enc.ciphertext()),
                "exponent": int(enc.exponent),
            }
        )

    payload = {
        "format": HE_COMPUTE_FORMAT_PAILLIER_V1,
        "session_id": session_id,
        "step": int(step),
        "key_id": key_id,
        "token_ids": [int(x) for x in indices.detach().cpu().tolist()],
        "scores": enc_scores,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def build_encrypted_hidden_state_artifact(
    *,
    hidden_last_token: torch.Tensor,
    client_public_key: bytes,
    session_id: str,
    step: int,
    key_id: str,
) -> bytes:
    """
    Encrypt the last-token hidden vector for sidecar-only decode flow.
    """
    if hidden_last_token.dim() != 1:
        raise ValueError("Expected 1D hidden-state tensor for final token.")
    pub = _load_public_key(client_public_key)
    enc_hidden = []
    for value in hidden_last_token.detach().cpu().tolist():
        enc = pub.encrypt(float(value))
        enc_hidden.append(
            {
                "ciphertext": str(enc.ciphertext()),
                "exponent": int(enc.exponent),
            }
        )
    payload = {
        "format": HE_COMPUTE_FORMAT_PAILLIER_HIDDEN_V1,
        "session_id": session_id,
        "step": int(step),
        "key_id": key_id,
        "hidden_size": int(hidden_last_token.shape[0]),
        "hidden": enc_hidden,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def decrypt_topk_artifact(
    *,
    artifact_bytes: bytes,
    client_private_key: bytes,
    expected_session_id: str,
    expected_step: int,
    expected_key_id: str,
) -> tuple[list[int], list[float]]:
    """
    Decrypt top-k artifact and return (token_ids, scores).
    """
    payload = json.loads(artifact_bytes.decode("utf-8"))
    if payload.get("format") != HE_COMPUTE_FORMAT_PAILLIER_V1:
        raise ValueError("Unsupported HE compute payload format.")
    if payload.get("session_id") != expected_session_id:
        raise ValueError("HE compute session mismatch.")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError("HE compute step mismatch.")
    if str(payload.get("key_id", "")) != expected_key_id:
        raise ValueError("HE compute key mismatch.")

    priv = _load_private_key(client_private_key)
    token_ids = [int(x) for x in payload.get("token_ids", [])]
    enc_scores = payload.get("scores", [])
    if len(token_ids) != len(enc_scores):
        raise ValueError("Malformed HE compute artifact lengths.")

    scores: list[float] = []
    for enc in enc_scores:
        enc_num = paillier.EncryptedNumber(
            public_key=priv.public_key,
            ciphertext=int(enc["ciphertext"]),
            exponent=int(enc["exponent"]),
        )
        scores.append(float(priv.decrypt(enc_num)))
    return token_ids, scores


def sample_from_topk_scores(
    *,
    token_ids: list[int],
    scores: list[float],
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """
    Sample token ID from decrypted top-k scores using temperature/top-p.
    """
    if not token_ids or not scores or len(token_ids) != len(scores):
        raise ValueError("Empty or malformed top-k scores.")
    t = max(float(temperature), 1e-6)
    p = min(max(float(top_p), 1e-6), 1.0)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    probs = torch.softmax(score_tensor / t, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = cumsum <= p
    keep_mask[0] = True
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep_mask]] = probs[sorted_idx[keep_mask]]
    total = filtered.sum()
    if float(total.item()) <= 0.0:
        chosen = int(torch.argmax(probs).item())
    else:
        filtered = filtered / total
        chosen = int(torch.multinomial(filtered, num_samples=1, generator=generator).item())
    return int(token_ids[chosen])
