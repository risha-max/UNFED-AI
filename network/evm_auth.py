"""
EVM signature verification helpers.
"""

from __future__ import annotations


def recover_evm_signer(message_text: str, signature: bytes) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct

    msg = encode_defunct(text=message_text)
    signer = Account.recover_message(msg, signature=bytes(signature))
    return str(signer)
