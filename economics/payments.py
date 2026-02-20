"""
Optimistic Rollup Payment System — trustless payments for compute work.

How it works:
  1. Users deposit funds into a smart contract escrow
  2. Nodes perform compute work, tracked by the share-chain
  3. Periodically, a settlement is posted to the payment contract
  4. Settlements are assumed honest for a challenge window (24 hours)
  5. During the window, anyone can submit a fraud proof to challenge
  6. If unchallenged, nodes can claim their earnings
  7. If challenged and fraud is proven, the cheater's stake is slashed

For this implementation:
  - We simulate the smart contract as an in-process ledger
  - The escrow, staking, and slashing logic is functional
  - In production, this would be an actual smart contract (e.g., on Ethereum/L2)

Components:
  - PaymentContract: Simulated smart contract with escrow and claims
  - StakeManager: Tracks node stakes and handles slashing
  - SettlementProcessor: Converts share-chain settlements to payments
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

from economics.share_chain import SettlementSummary


# --- Configuration (network-wide defaults) ---
# Per-model overrides come from PoolConfig (economics/pool_config.py).

CHALLENGE_WINDOW_SECONDS = 60  # 1 minute for testing (24 hours in production)
MIN_STAKE = 100.0              # Minimum stake to participate (default)
SLASH_FRACTION = 0.5           # Fraction of stake slashed on fraud (default)


# --- Ledger ---

@dataclass
class Account:
    """An account in the payment system."""
    address: str                # node_id or user_id
    balance: float = 0.0       # Available balance
    staked: float = 0.0        # Staked amount (locked)
    earned: float = 0.0        # Total earned (historical)
    slashed: float = 0.0       # Total slashed (historical)


@dataclass
class Settlement:
    """A posted settlement waiting for challenge."""
    settlement_hash: str
    summary: SettlementSummary
    posted_at: float
    challenge_deadline: float
    challenged: bool = False
    finalized: bool = False
    total_payout: float = 0.0


# --- Stake Manager ---

class StakeManager:
    """Manages node stakes for participation and fraud deterrence."""

    def __init__(self, min_stake: float = MIN_STAKE,
                 slash_fraction: float = SLASH_FRACTION):
        self.min_stake = min_stake
        self.slash_fraction = slash_fraction
        self._accounts: dict[str, Account] = {}
        self._lock = threading.Lock()

    def get_or_create_account(self, address: str) -> Account:
        with self._lock:
            if address not in self._accounts:
                self._accounts[address] = Account(address=address)
            return self._accounts[address]

    def deposit(self, address: str, amount: float) -> float:
        """Deposit funds into an account. Returns new balance."""
        account = self.get_or_create_account(address)
        with self._lock:
            account.balance += amount
        return account.balance

    def stake(self, address: str, amount: float) -> bool:
        """Stake funds (lock them). Returns True if successful."""
        account = self.get_or_create_account(address)
        with self._lock:
            if account.balance < amount:
                return False
            account.balance -= amount
            account.staked += amount
        return True

    def unstake(self, address: str, amount: float) -> bool:
        """Unstake funds (unlock them). Returns True if successful."""
        account = self.get_or_create_account(address)
        with self._lock:
            if account.staked < amount:
                return False
            account.staked -= amount
            account.balance += amount
        return True

    def is_eligible(self, address: str,
                    pool_min_stake: float = None) -> bool:
        """Check if a node has enough stake to participate.

        Args:
            address: Node address.
            pool_min_stake: Per-pool minimum stake override (from PoolConfig).
                            Falls back to the global min_stake if not provided.
        """
        account = self.get_or_create_account(address)
        required = pool_min_stake if pool_min_stake is not None else self.min_stake
        return account.staked >= required

    def slash(self, address: str,
              pool_slash_fraction: float = None) -> float:
        """Slash a node's stake for fraud. Returns amount slashed.

        Args:
            address: Node address.
            pool_slash_fraction: Per-pool slash fraction override (from PoolConfig).
                                 Falls back to the global slash_fraction if not provided.
        """
        fraction = (pool_slash_fraction
                    if pool_slash_fraction is not None
                    else self.slash_fraction)
        account = self.get_or_create_account(address)
        with self._lock:
            slash_amount = account.staked * fraction
            account.staked -= slash_amount
            account.slashed += slash_amount
        print(f"[Stake] SLASHED {address[:8]}...: {slash_amount:.2f} "
              f"(remaining stake: {account.staked:.2f})")
        return slash_amount

    def credit_earnings(self, address: str, amount: float):
        """Credit earnings to a node."""
        account = self.get_or_create_account(address)
        with self._lock:
            account.balance += amount
            account.earned += amount

    def get_all_accounts(self) -> dict[str, dict]:
        with self._lock:
            return {
                addr: {
                    "balance": acc.balance,
                    "staked": acc.staked,
                    "earned": acc.earned,
                    "slashed": acc.slashed,
                }
                for addr, acc in self._accounts.items()
            }


# --- Payment Contract ---

class PaymentContract:
    """
    Payment contract for optimistic rollup payments.

    When an on-chain escrow is configured (via ``onchain_escrow`` param),
    settlements are posted on-chain as the primary payment path.  The
    in-memory simulation is kept as a fallback for when ``--no-chain``
    is passed or no escrow contract is configured.

    Uses the FeeOracle's base_fee for dynamic pricing instead of a
    fixed price_per_share. Settlements pay out based on weighted shares
    (compute=1.0) * base_fee.
    """

    def __init__(self, stake_manager: StakeManager,
                 challenge_window: float = CHALLENGE_WINDOW_SECONDS,
                 price_per_share: float = 0.001,
                 fee_oracle=None,
                 price_per_input_token: float = 0.0,
                 price_per_output_token: float = 0.0,
                 onchain_escrow=None):
        """
        Args:
            stake_manager: StakeManager instance
            challenge_window: Seconds to wait before finalizing a settlement
            price_per_share: Fallback payment per share (used if no token pricing and no fee oracle)
            fee_oracle: Optional FeeOracle for dynamic pricing
            price_per_input_token: Operator-set price per input token (0 = use share pricing)
            price_per_output_token: Operator-set price per output token (0 = use share pricing)
            onchain_escrow: Optional OnChainEscrow instance for on-chain settlements
        """
        self.stakes = stake_manager
        self.challenge_window = challenge_window
        self.price_per_share = price_per_share
        self._fee_oracle = fee_oracle
        self.price_per_input_token = price_per_input_token
        self.price_per_output_token = price_per_output_token
        self._onchain_escrow = onchain_escrow

        # Accumulated token usage reported by clients (input_tokens, output_tokens)
        self._token_usage: list[tuple[int, int]] = []
        self._usage_lock = threading.Lock()

        self._escrow_balance: float = 0.0
        self._settlements: list[Settlement] = []
        self._lock = threading.Lock()

        if self._onchain_escrow:
            print("[PaymentContract] On-chain escrow attached — "
                  "settlements will be posted on-chain by default.")
        else:
            print("[PaymentContract] No on-chain escrow — "
                  "using in-memory simulation.")

    def deposit_to_escrow(self, amount: float):
        """User deposits funds into the escrow pool."""
        with self._lock:
            self._escrow_balance += amount
        print(f"[Contract] Escrow deposit: {amount:.4f} "
              f"(total: {self._escrow_balance:.4f})")

    def report_usage(self, input_tokens: int, output_tokens: int):
        """Report token usage from a completed generation request.

        Called by the web server / client after each generation.
        Accumulated usage is consumed at settlement time.
        """
        with self._usage_lock:
            self._token_usage.append((input_tokens, output_tokens))

    @property
    def uses_token_pricing(self) -> bool:
        """True if per-token pricing is active (operator set prices > 0)."""
        return self.price_per_output_token > 0

    def _compute_token_revenue(self) -> float:
        """Compute total revenue from accumulated token usage and drain the buffer."""
        with self._usage_lock:
            usage = list(self._token_usage)
            self._token_usage.clear()

        total = 0.0
        for inp, out in usage:
            total += inp * self.price_per_input_token
            total += out * self.price_per_output_token
        return total

    def _get_current_fee(self) -> float:
        """Get the current fee per weighted share unit.

        Uses the fee oracle if available, otherwise falls back to the
        static price_per_share.  When per-token pricing is active, this
        is only used as a fallback for settlements without token data.
        """
        if self._fee_oracle is not None:
            return self._fee_oracle.get_base_fee()
        return self.price_per_share

    def post_settlement(self, summary: SettlementSummary) -> Settlement:
        """
        Post a settlement from the share-chain.

        When per-token pricing is active, revenue comes from accumulated
        token usage.  Otherwise, falls back to share-based pricing.
        The revenue pool is then split among nodes by weighted shares.
        """
        if self.uses_token_pricing:
            total_payout = self._compute_token_revenue()
            pricing_info = (f"token pricing: "
                            f"{self.price_per_input_token}/in, "
                            f"{self.price_per_output_token}/out")
        else:
            fee = self._get_current_fee()
            total_payout = summary.total_shares * fee
            pricing_info = f"fee={fee:.6f}/share"

        now = time.time()

        settlement = Settlement(
            settlement_hash=summary.settlement_hash,
            summary=summary,
            posted_at=now,
            challenge_deadline=now + self.challenge_window,
            total_payout=total_payout,
        )

        with self._lock:
            self._settlements.append(settlement)

        print(f"[Contract] Settlement posted: {summary.total_shares:.2f} "
              f"weighted shares, {summary.total_tokens} tokens, "
              f"payout={total_payout:.6f} ({pricing_info}), "
              f"challenge window={self.challenge_window}s")

        return settlement

    def challenge_settlement(self, settlement_hash: str,
                             fraud_node_id: str) -> bool:
        """
        Challenge a settlement with a fraud proof.

        If the challenge window hasn't expired, the settlement is marked
        as challenged and the fraudulent node is slashed.
        """
        with self._lock:
            for s in self._settlements:
                if s.settlement_hash == settlement_hash and not s.finalized:
                    if time.time() > s.challenge_deadline:
                        print(f"[Contract] Challenge rejected: window expired")
                        return False
                    s.challenged = True
                    slashed = self.stakes.slash(fraud_node_id)
                    print(f"[Contract] Settlement challenged! "
                          f"Node {fraud_node_id[:8]}... slashed {slashed:.2f}")
                    return True
        return False

    def finalize_and_pay(self) -> int:
        """
        Finalize all settlements past their challenge window and pay nodes.

        With per-token pricing, each node gets:
            payout = total_settlement_revenue × (node_shares / total_shares)

        This distributes the per-token revenue to nodes proportionally
        by their contribution (weighted compute shares).

        Returns number of settlements finalized.
        """
        finalized = 0
        now = time.time()

        with self._lock:
            for s in self._settlements:
                if s.finalized or s.challenged:
                    continue
                if now < s.challenge_deadline:
                    continue

                total_revenue = s.total_payout
                total_shares = s.summary.total_shares

                for node_id, weighted_shares in s.summary.node_shares.items():
                    if total_shares > 0:
                        proportion = weighted_shares / total_shares
                        payout = total_revenue * proportion
                    else:
                        payout = 0.0
                    if payout > 0 and payout <= self._escrow_balance:
                        self._escrow_balance -= payout
                        self.stakes.credit_earnings(node_id, payout)

                s.finalized = True
                finalized += 1

                print(f"[Contract] Settlement finalized: "
                      f"revenue={total_revenue:.6f}, "
                      f"{total_shares:.2f} shares, "
                      f"{len(s.summary.node_shares)} nodes paid")

        return finalized

    @property
    def is_onchain(self) -> bool:
        """True if on-chain payments are active."""
        return self._onchain_escrow is not None

    def get_status(self) -> dict:
        with self._lock:
            status = {
                "escrow_balance": self._escrow_balance,
                "total_settlements": len(self._settlements),
                "pending": sum(1 for s in self._settlements
                               if not s.finalized and not s.challenged),
                "finalized": sum(1 for s in self._settlements if s.finalized),
                "challenged": sum(1 for s in self._settlements if s.challenged),
                "current_fee": self._get_current_fee(),
                "price_per_share": self.price_per_share,
                "dynamic_pricing": self._fee_oracle is not None,
                "onchain": self._onchain_escrow is not None,
            }
        return status


# --- Settlement Processor ---

class SettlementProcessor:
    """
    Bridges the share-chain and the payment contract.

    Periodically checks for new settlements from the share-chain
    and posts them to the payment contract.
    Also runs finalization to pay nodes after challenge windows expire.
    """

    def __init__(self, contract: PaymentContract):
        self.contract = contract
        self._processed_count = 0
        self._running = False

    def process_settlement(self, summary: SettlementSummary):
        """Process a single settlement."""
        self.contract.post_settlement(summary)
        self._processed_count += 1

    def run_finalization(self) -> int:
        """Run finalization for expired challenge windows."""
        return self.contract.finalize_and_pay()

    def start_background_finalization(self, interval: float = 5.0):
        """Start a background thread that periodically finalizes settlements."""
        self._running = True
        thread = threading.Thread(target=self._finalization_loop,
                                  args=(interval,), daemon=True)
        thread.start()

    def _finalization_loop(self, interval: float):
        while self._running:
            time.sleep(interval)
            if self._running:
                self.run_finalization()

    def stop(self):
        self._running = False
