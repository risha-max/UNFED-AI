"""
On-Chain Escrow Client — Python web3 wrapper for UnfedEscrow.sol.

Bridges the registry server to the on-chain staking/escrow contract.
Used when a cluster's ClusterConfig has ``escrow_contract_address`` set.

Usage:
    escrow = OnChainEscrow(
        rpc_url="http://localhost:8545",
        contract_address="0x...",
        operator_private_key="0x...",
    )
    # Reads (free)
    balance = escrow.staked_balance("0xNodeAddress")
    eligible = escrow.is_eligible("0xNodeAddress")

    # Writes (cost gas, signed by operator)
    escrow.post_settlement(hash, nodes, amounts)
    escrow.finalize_settlement(hash)
    escrow.slash_node("0xNodeAddress")
"""

import json
import os
from typing import Optional

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

# ---------------------------------------------------------------------------
# Minimal ABI — only the functions the registry needs to call.
# The full ABI is in contracts/out/UnfedEscrow.sol/UnfedEscrow.json
# ---------------------------------------------------------------------------

_ESCROW_ABI = [
    # Views
    {
        "type": "function",
        "name": "stakedBalance",
        "inputs": [{"name": "node", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "isEligible",
        "inputs": [{"name": "node", "type": "address"}],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "clientBalances",
        "inputs": [{"name": "", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "escrowPool",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "earnings",
        "inputs": [{"name": "", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "minStake",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "stakes",
        "inputs": [{"name": "", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    # Operator writes
    {
        "type": "function",
        "name": "postSettlement",
        "inputs": [
            {"name": "hash", "type": "bytes32"},
            {"name": "nodes", "type": "address[]"},
            {"name": "amounts", "type": "uint256[]"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "finalizeSettlement",
        "inputs": [{"name": "hash", "type": "bytes32"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "challengeSettlement",
        "inputs": [{"name": "hash", "type": "bytes32"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "slashNode",
        "inputs": [{"name": "node", "type": "address"}],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "depositFor",
        "inputs": [
            {"name": "client", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [],
        "stateMutability": "nonpayable",
    },
]

_ERC20_ABI = [
    {
        "type": "function",
        "name": "balanceOf",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "symbol",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "decimals",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
    },
    {
        "type": "function",
        "name": "approve",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    },
    {
        "type": "function",
        "name": "transfer",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
    },
]


class OnChainEscrow:
    """Python wrapper for the UnfedEscrow smart contract.

    Provides read methods (free, no gas) and write methods (operator-signed
    transactions) for the registry to interact with on-chain staking and
    escrow.
    """

    def __init__(
        self,
        rpc_url: str,
        contract_address: str,
        operator_private_key: str,
        token_address: str = "",
    ):
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))
        # Support PoA chains (e.g. some L2s)
        self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if not self._w3.is_connected():
            raise ConnectionError(
                f"Cannot connect to chain at {rpc_url}")

        self._contract = self._w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=_ESCROW_ABI,
        )
        self._operator_key = operator_private_key
        self._operator_address = self._w3.eth.account.from_key(
            operator_private_key).address

        # Optional: cache token info
        self._token_address = token_address
        self._token_contract = None
        if token_address:
            self._token_contract = self._w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=_ERC20_ABI,
            )

        print(f"[OnChain] Connected to {rpc_url}")
        print(f"[OnChain] Escrow contract: {contract_address}")
        print(f"[OnChain] Operator: {self._operator_address}")

    # ------------------------------------------------------------------
    # Read methods (free, no gas)
    # ------------------------------------------------------------------

    def staked_balance(self, node_address: str) -> int:
        """Get a node's staked balance in wei."""
        addr = Web3.to_checksum_address(node_address)
        return self._contract.functions.stakedBalance(addr).call()

    def is_eligible(self, node_address: str) -> bool:
        """Check if a node meets the on-chain minimum stake."""
        addr = Web3.to_checksum_address(node_address)
        return self._contract.functions.isEligible(addr).call()

    def client_balance(self, client_address: str) -> int:
        """Get a client's escrow balance in wei."""
        addr = Web3.to_checksum_address(client_address)
        return self._contract.functions.clientBalances(addr).call()

    def escrow_pool(self) -> int:
        """Get the total escrow pool balance in wei."""
        return self._contract.functions.escrowPool().call()

    def node_earnings(self, node_address: str) -> int:
        """Get a node's unclaimed earnings in wei."""
        addr = Web3.to_checksum_address(node_address)
        return self._contract.functions.earnings(addr).call()

    def min_stake(self) -> int:
        """Get the on-chain minimum stake in wei."""
        return self._contract.functions.minStake().call()

    def get_client_balance(self, client_address: str) -> int:
        """Get a client's deposit balance in wei (convenience alias)."""
        return self.client_balance(client_address)

    def get_client_balance_formatted(self, client_address: str,
                                     decimals: int = 18) -> float:
        """Get a client's deposit balance as a float (in token units)."""
        wei = self.client_balance(client_address)
        return wei / (10 ** decimals)

    # ------------------------------------------------------------------
    # Write methods (operator-signed transactions)
    # ------------------------------------------------------------------

    def _send_tx(self, fn, description: str = "") -> str:
        """Build, sign, and send a transaction. Returns tx hash hex."""
        tx = fn.build_transaction({
            "from": self._operator_address,
            "nonce": self._w3.eth.get_transaction_count(
                self._operator_address),
            "gas": 500_000,
            "gasPrice": self._w3.eth.gas_price,
        })
        signed = self._w3.eth.account.sign_transaction(
            tx, self._operator_key)
        tx_hash = self._w3.eth.send_raw_transaction(
            signed.raw_transaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash)

        status = "OK" if receipt["status"] == 1 else "FAILED"
        print(f"[OnChain] {description}: {status} "
              f"(tx={tx_hash.hex()[:16]}... gas={receipt['gasUsed']})")
        return tx_hash.hex()

    def post_settlement(
        self,
        settlement_hash: str,
        node_addresses: list[str],
        amounts_wei: list[int],
    ) -> str:
        """Post a batch settlement on-chain.

        Args:
            settlement_hash: Hex string (will be converted to bytes32)
            node_addresses: List of node Ethereum addresses
            amounts_wei: List of payout amounts in wei (same order)

        Returns:
            Transaction hash hex string
        """
        hash_bytes = bytes.fromhex(
            settlement_hash.replace("0x", "").ljust(64, "0")[:64])
        nodes = [Web3.to_checksum_address(a) for a in node_addresses]
        fn = self._contract.functions.postSettlement(
            hash_bytes, nodes, amounts_wei)
        return self._send_tx(fn, f"postSettlement({settlement_hash[:16]}...)")

    def finalize_settlement(self, settlement_hash: str) -> str:
        """Finalize a settlement after challenge window."""
        hash_bytes = bytes.fromhex(
            settlement_hash.replace("0x", "").ljust(64, "0")[:64])
        fn = self._contract.functions.finalizeSettlement(hash_bytes)
        return self._send_tx(
            fn, f"finalizeSettlement({settlement_hash[:16]}...)")

    def challenge_settlement(self, settlement_hash: str) -> str:
        """Challenge a settlement (fraud detected)."""
        hash_bytes = bytes.fromhex(
            settlement_hash.replace("0x", "").ljust(64, "0")[:64])
        fn = self._contract.functions.challengeSettlement(hash_bytes)
        return self._send_tx(
            fn, f"challengeSettlement({settlement_hash[:16]}...)")

    def slash_node(self, node_address: str) -> str:
        """Slash a node's stake (operator-attested fraud)."""
        addr = Web3.to_checksum_address(node_address)
        fn = self._contract.functions.slashNode(addr)
        return self._send_tx(fn, f"slashNode({node_address[:10]}...)")

    # ------------------------------------------------------------------
    # Faucet (testnet only)
    # ------------------------------------------------------------------

    FAUCET_DRIP_AMOUNT = 100   # tokens per drip
    FAUCET_COOLDOWN = 3600     # seconds between drips per address

    def faucet_drip(self, client_address: str,
                    amount_tokens: float = 0) -> str:
        """Transfer test tokens from operator to a client's escrow balance.

        Executes two transactions: approve + depositFor.
        Returns the depositFor tx hash.
        """
        if not self._token_contract:
            raise ValueError("No token contract configured")

        amount = amount_tokens or self.FAUCET_DRIP_AMOUNT
        amount_wei = int(amount * 10**18)
        client = Web3.to_checksum_address(client_address)
        escrow_addr = self._contract.address

        approve_fn = self._token_contract.functions.approve(
            escrow_addr, amount_wei)
        self._send_tx(approve_fn, f"approve({amount} tokens)")

        deposit_fn = self._contract.functions.depositFor(
            client, amount_wei)
        return self._send_tx(
            deposit_fn, f"depositFor({client_address[:10]}..., {amount})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def token_symbol(self) -> str:
        """Get the staking token's symbol."""
        if self._token_contract:
            return self._token_contract.functions.symbol().call()
        return "?"

    def format_amount(self, wei: int, decimals: int = 18) -> str:
        """Format a wei amount to a human-readable string."""
        return f"{wei / (10 ** decimals):.4f}"

    @property
    def connected(self) -> bool:
        return self._w3.is_connected()

    @property
    def operator_address(self) -> str:
        return self._operator_address


def load_from_env(env_file: str) -> OnChainEscrow:
    """Create an OnChainEscrow from a deployed.env file.

    The env file is generated by scripts/start_local_chain.sh.
    """
    env = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip()

    return OnChainEscrow(
        rpc_url=env["CHAIN_RPC_URL"],
        contract_address=env["ESCROW_ADDRESS"],
        operator_private_key=env["OPERATOR_PRIVATE_KEY"],
        token_address=env.get("TOKEN_ADDRESS", ""),
    )
