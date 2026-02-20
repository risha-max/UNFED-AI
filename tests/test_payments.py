"""
Unit tests for the payment system: settlement flow, stake/slash, escrow.

Tests the full cycle:
  post settlement -> challenge window -> finalize -> earnings
  stake -> insufficient stake -> slash
"""

import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from economics.payments import (
    StakeManager,
    PaymentContract,
    SettlementProcessor,
    Account,
)
from economics.share_chain import SettlementSummary


def _make_settlement(node_shares: dict[str, float],
                     total_tokens: int = 100) -> SettlementSummary:
    """Create a test settlement summary."""
    import hashlib
    total = sum(node_shares.values())
    now = time.time()
    s = SettlementSummary(
        period_start=now - 10,
        period_end=now,
        node_shares=node_shares,
        total_shares=total,
        total_tokens=total_tokens,
        block_range=(1, 5),
    )
    s.finalize()
    return s


class TestStakeManager:
    """Tests for stake management."""

    def test_deposit_and_stake(self, stake_manager):
        stake_manager.deposit("node_A", 200.0)
        assert stake_manager.stake("node_A", 150.0)
        account = stake_manager.get_or_create_account("node_A")
        assert account.staked == 150.0
        assert account.balance == 50.0

    def test_insufficient_balance_for_stake(self, stake_manager):
        stake_manager.deposit("node_B", 50.0)
        assert not stake_manager.stake("node_B", 100.0)

    def test_unstake(self, stake_manager):
        stake_manager.deposit("node_C", 200.0)
        stake_manager.stake("node_C", 200.0)
        assert stake_manager.unstake("node_C", 100.0)
        account = stake_manager.get_or_create_account("node_C")
        assert account.staked == 100.0
        assert account.balance == 100.0

    def test_is_eligible(self, stake_manager):
        stake_manager.deposit("node_D", 200.0)
        stake_manager.stake("node_D", 200.0)
        assert stake_manager.is_eligible("node_D")
        assert not stake_manager.is_eligible("node_E")

    def test_eligibility_with_pool_override(self, stake_manager):
        stake_manager.deposit("node_F", 50.0)
        stake_manager.stake("node_F", 50.0)
        # Pool requires only 30
        assert stake_manager.is_eligible("node_F", pool_min_stake=30.0)
        # Pool requires 100
        assert not stake_manager.is_eligible("node_F", pool_min_stake=100.0)

    def test_slash(self, stake_manager):
        stake_manager.deposit("cheater", 200.0)
        stake_manager.stake("cheater", 200.0)
        slashed = stake_manager.slash("cheater")
        assert slashed == 100.0  # 50% of 200
        account = stake_manager.get_or_create_account("cheater")
        assert account.staked == 100.0
        assert account.slashed == 100.0

    def test_slash_with_pool_fraction(self, stake_manager):
        stake_manager.deposit("node_G", 200.0)
        stake_manager.stake("node_G", 200.0)
        slashed = stake_manager.slash("node_G", pool_slash_fraction=0.25)
        assert slashed == 50.0  # 25% of 200

    def test_credit_earnings(self, stake_manager):
        stake_manager.credit_earnings("worker", 42.0)
        account = stake_manager.get_or_create_account("worker")
        assert account.earned == 42.0
        assert account.balance == 42.0

    def test_get_all_accounts(self, stake_manager):
        stake_manager.deposit("a", 100.0)
        stake_manager.deposit("b", 200.0)
        accounts = stake_manager.get_all_accounts()
        assert "a" in accounts
        assert "b" in accounts
        assert accounts["a"]["balance"] == 100.0


class TestPaymentContract:
    """Tests for the payment contract."""

    def test_deposit_to_escrow(self, payment_contract):
        payment_contract.deposit_to_escrow(1000.0)
        status = payment_contract.get_status()
        assert status["escrow_balance"] == 1000.0

    def test_post_settlement_token_pricing(self, payment_contract):
        payment_contract.report_usage(100, 50)
        summary = _make_settlement({"node_A": 5.0, "node_B": 3.0})
        settlement = payment_contract.post_settlement(summary)

        # With token pricing: 100 * 0.0001 + 50 * 0.001 = 0.01 + 0.05 = 0.06
        assert abs(settlement.total_payout - 0.06) < 1e-6

    def test_finalize_settlement(self, payment_contract):
        payment_contract.deposit_to_escrow(1000.0)
        payment_contract.report_usage(100, 50)
        summary = _make_settlement({"node_A": 5.0, "node_B": 3.0})
        payment_contract.post_settlement(summary)

        # Wait past challenge window
        time.sleep(0.2)
        finalized = payment_contract.finalize_and_pay()
        assert finalized == 1

    def test_challenge_settlement(self, payment_contract):
        payment_contract.stakes.deposit("cheater", 200.0)
        payment_contract.stakes.stake("cheater", 200.0)

        payment_contract.report_usage(100, 50)
        summary = _make_settlement({"cheater": 5.0, "honest": 3.0})
        s = payment_contract.post_settlement(summary)

        result = payment_contract.challenge_settlement(
            s.settlement_hash, "cheater")
        assert result is True

        # Cheater should be slashed
        account = payment_contract.stakes.get_or_create_account("cheater")
        assert account.slashed > 0

    def test_finalization_skips_challenged(self, payment_contract):
        payment_contract.deposit_to_escrow(1000.0)
        payment_contract.stakes.deposit("cheater", 200.0)
        payment_contract.stakes.stake("cheater", 200.0)

        payment_contract.report_usage(100, 50)
        summary = _make_settlement({"cheater": 5.0})
        s = payment_contract.post_settlement(summary)
        payment_contract.challenge_settlement(s.settlement_hash, "cheater")

        time.sleep(0.2)
        finalized = payment_contract.finalize_and_pay()
        assert finalized == 0  # challenged settlement should not finalize

    def test_expired_challenge_rejected(self, payment_contract):
        payment_contract.report_usage(100, 50)
        summary = _make_settlement({"node": 5.0})
        s = payment_contract.post_settlement(summary)

        # Wait past challenge window
        time.sleep(0.2)
        result = payment_contract.challenge_settlement(
            s.settlement_hash, "node")
        assert result is False  # too late

    def test_status_includes_onchain(self, payment_contract):
        status = payment_contract.get_status()
        assert "onchain" in status


class TestSettlementProcessor:
    """Tests for the settlement processor."""

    def test_process_and_finalize(self, payment_contract):
        payment_contract.deposit_to_escrow(1000.0)
        processor = SettlementProcessor(payment_contract)

        payment_contract.report_usage(100, 50)
        summary = _make_settlement({"node_X": 10.0})
        processor.process_settlement(summary)

        time.sleep(0.2)
        finalized = processor.run_finalization()
        assert finalized == 1
