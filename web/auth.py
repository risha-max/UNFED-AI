"""
Client Wallet Authentication — maps API keys / wallet signatures to addresses.

Supports two authentication modes:

1. **API Key mode** (demo/testing): A simple mapping from API keys to
   Ethereum addresses.  The web server assigns a default client address
   for unauthenticated requests.

2. **Wallet Signature mode** (production): The client signs a challenge
   message with their Ethereum private key, and we recover the address
   from the signature.  This proves the client controls the wallet that
   has funds deposited in escrow.

Usage:
    auth = WalletAuth()

    # Demo mode — use a default address
    addr = auth.authenticate(api_key="demo")

    # Or verify a signed challenge
    addr = auth.verify_signature(challenge, signature)
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Default client address (for demo / unauthenticated requests)
# ---------------------------------------------------------------------------

# This address is the 4th Anvil default account — used by the demo scripts
# for client deposits.  In production, every client would have their own
# wallet.
DEFAULT_CLIENT_ADDRESS = "0x90F79bf6EB2c4f870365E785982E1f101E93b906"


@dataclass
class AuthSession:
    """An authenticated client session."""
    client_address: str
    api_key: str = ""
    authenticated_at: float = 0.0
    last_activity: float = 0.0


class WalletAuth:
    """
    Client authentication for escrow balance enforcement.

    Maps client identities to Ethereum addresses so we can check their
    escrow deposit before allowing inference.
    """

    def __init__(self, default_address: str = DEFAULT_CLIENT_ADDRESS):
        self._default_address = default_address
        # API key -> client address mapping
        self._api_keys: dict[str, str] = {
            "demo": default_address,
        }
        # Active sessions (session_id -> AuthSession)
        self._sessions: dict[str, AuthSession] = {}

    def register_api_key(self, api_key: str, address: str):
        """Register an API key -> address mapping."""
        self._api_keys[api_key] = address

    def authenticate(self, api_key: str = "",
                     wallet_address: str = "") -> str:
        """
        Authenticate a client and return their Ethereum address.

        Args:
            api_key: API key for key-based auth.
            wallet_address: Direct wallet address (for wallet-signed requests).

        Returns:
            The client's Ethereum address.
        """
        if wallet_address:
            return wallet_address

        if api_key and api_key in self._api_keys:
            return self._api_keys[api_key]

        # Default: return the demo address (allows unauthenticated access
        # in demo mode, but escrow balance still checked)
        return self._default_address

    def create_session(self, client_address: str,
                       api_key: str = "") -> str:
        """Create an authenticated session.  Returns session ID."""
        session_id = hashlib.sha256(
            f"{client_address}:{time.time()}:{os.urandom(16).hex()}"
            .encode()
        ).hexdigest()[:32]

        self._sessions[session_id] = AuthSession(
            client_address=client_address,
            api_key=api_key,
            authenticated_at=time.time(),
            last_activity=time.time(),
        )
        return session_id

    def get_session_address(self, session_id: str) -> Optional[str]:
        """Get the address for an active session."""
        session = self._sessions.get(session_id)
        if session:
            session.last_activity = time.time()
            return session.client_address
        return None

    @property
    def default_address(self) -> str:
        return self._default_address

    def generate_challenge(self) -> str:
        """Generate a challenge string for wallet signature auth."""
        nonce = os.urandom(16).hex()
        timestamp = int(time.time())
        return f"UNFED-AUTH:{timestamp}:{nonce}"

    def verify_signature(self, challenge: str,
                         signature: str) -> Optional[str]:
        """
        Verify a wallet signature and recover the signer's address.

        Uses eth_account for signature recovery.  Returns the address
        if valid, None otherwise.
        """
        try:
            from eth_account.messages import encode_defunct
            from eth_account import Account

            message = encode_defunct(text=challenge)
            address = Account.recover_message(message, signature=signature)
            return address
        except Exception:
            return None
