import pytest

from network.he_output import generate_client_keypair, encrypt_token_artifact, decrypt_token_artifact


def test_he_output_roundtrip():
    priv, pub = generate_client_keypair()
    artifact = encrypt_token_artifact(
        client_public_key=pub,
        sender_private_key=None,
        session_id="sess-1",
        step=3,
        key_id="kid-1",
        token_id=42,
        is_eos=False,
    )
    token_id, is_eos = decrypt_token_artifact(
        client_private_key=priv,
        sender_public_key=artifact["sender_public_key"],
        session_id="sess-1",
        step=3,
        key_id="kid-1",
        nonce=artifact["nonce"],
        ciphertext=artifact["ciphertext"],
    )
    assert token_id == 42
    assert is_eos is False


def test_he_output_replay_or_step_substitution_rejected():
    priv, pub = generate_client_keypair()
    artifact = encrypt_token_artifact(
        client_public_key=pub,
        sender_private_key=None,
        session_id="sess-2",
        step=1,
        key_id="kid-2",
        token_id=7,
        is_eos=True,
    )
    with pytest.raises(Exception):
        decrypt_token_artifact(
            client_private_key=priv,
            sender_public_key=artifact["sender_public_key"],
            session_id="sess-2",
            step=2,  # wrong step
            key_id="kid-2",
            nonce=artifact["nonce"],
            ciphertext=artifact["ciphertext"],
        )


def test_he_output_key_mismatch_rejected():
    priv, pub = generate_client_keypair()
    artifact = encrypt_token_artifact(
        client_public_key=pub,
        sender_private_key=None,
        session_id="sess-3",
        step=0,
        key_id="kid-3",
        token_id=999,
        is_eos=False,
    )
    with pytest.raises(Exception):
        decrypt_token_artifact(
            client_private_key=priv,
            sender_public_key=artifact["sender_public_key"],
            session_id="sess-3",
            step=0,
            key_id="kid-other",
            nonce=artifact["nonce"],
            ciphertext=artifact["ciphertext"],
        )
