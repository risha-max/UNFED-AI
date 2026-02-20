"""
Unit tests for the share-chain: block production, settlement summaries,
share counting, and chain integrity.
"""

import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from economics.share_chain import (
    ComputeShare,
    Block,
    ShareChain,
    SettlementSummary,
)


class TestComputeShare:
    """Tests for ComputeShare."""

    def test_to_dict_roundtrip(self):
        share = ComputeShare(
            node_id="node_0",
            shard_index=1,
            session_id="sess_1",
            activation_hash="abc123",
            tokens_processed=10,
            share_weight=1.0,
        )
        d = share.to_dict()
        restored = ComputeShare.from_dict(d)
        assert restored.node_id == share.node_id
        assert restored.shard_index == share.shard_index
        assert restored.activation_hash == share.activation_hash
        assert restored.share_weight == share.share_weight

    def test_hash_deterministic(self):
        share = ComputeShare(
            node_id="n", shard_index=0, session_id="s",
            activation_hash="h", tokens_processed=5,
            timestamp=12345.0,
        )
        h1 = share.hash()
        h2 = share.hash()
        assert h1 == h2

    def test_different_shares_different_hashes(self):
        s1 = ComputeShare("n1", 0, "s", "h1", 5)
        s2 = ComputeShare("n2", 0, "s", "h2", 5)
        assert s1.hash() != s2.hash()


class TestBlock:
    """Tests for Block."""

    def test_finalize_sets_hash(self, sample_shares):
        block = Block(
            index=0, previous_hash="0" * 64,
            shares=sample_shares,
        )
        assert block.block_hash == ""
        block.finalize()
        assert block.block_hash != ""
        assert len(block.block_hash) == 64  # SHA-256 hex

    def test_hash_deterministic(self, sample_shares):
        block = Block(
            index=0, previous_hash="0" * 64,
            shares=sample_shares, timestamp=12345.0,
        )
        block.finalize()
        h1 = block.block_hash

        block2 = Block(
            index=0, previous_hash="0" * 64,
            shares=sample_shares, timestamp=12345.0,
        )
        block2.finalize()
        assert block2.block_hash == h1

    def test_to_dict(self, sample_shares):
        block = Block(index=1, previous_hash="abc", shares=sample_shares)
        block.finalize()
        d = block.to_dict()
        assert d["index"] == 1
        assert d["previous_hash"] == "abc"
        assert len(d["shares"]) == 5


class TestShareChain:
    """Tests for the ShareChain."""

    def test_new_chain_has_genesis(self):
        chain = ShareChain(block_interval=10.0, settlement_blocks=3)
        # Genesis block is created at init, so height starts at 1
        assert chain.height >= 1
        info = chain.get_chain_info()
        assert info["height"] >= 1

    def test_add_shares_and_produce_block(self, sample_shares):
        chain = ShareChain(block_interval=10.0, settlement_blocks=3)
        initial_height = chain.height
        for s in sample_shares:
            chain.add_share(s)

        block = chain.produce_block()
        assert block is not None
        assert len(block.shares) == 5
        assert chain.height == initial_height + 1

    def test_no_duplicate_shares_in_blocks(self, sample_shares):
        chain = ShareChain(block_interval=10.0, settlement_blocks=3)
        for s in sample_shares:
            chain.add_share(s)

        block1 = chain.produce_block()
        assert block1 is not None

        # Second block should have no shares (all consumed)
        block2 = chain.produce_block()
        assert block2 is None

    def test_settlement_summaries(self):
        chain = ShareChain(block_interval=10.0, settlement_blocks=2)

        for i in range(3):
            chain.add_share(ComputeShare(
                node_id="node_A", shard_index=0,
                session_id=f"s{i}", activation_hash=f"h{i}",
                tokens_processed=10,
            ))
        chain.produce_block()

        for i in range(3):
            chain.add_share(ComputeShare(
                node_id="node_B", shard_index=1,
                session_id=f"s{10 + i}", activation_hash=f"h{10 + i}",
                tokens_processed=5,
            ))
        chain.produce_block()

        settlements = chain.get_settlements()
        assert len(settlements) >= 1

        s = settlements[0]
        assert isinstance(s, SettlementSummary)
        assert s.total_shares > 0
        assert s.total_tokens > 0

    def test_receive_external_block(self, sample_shares):
        chain = ShareChain(block_interval=10.0, settlement_blocks=3)
        initial_height = chain.height
        tip_height = chain.get_tip_height()
        # Get the hash of the tip block
        tip_block = chain.get_blocks_from(tip_height)[0]
        block = Block(
            index=tip_height + 1,
            previous_hash=tip_block.block_hash,
            shares=sample_shares,
        )
        block.finalize()

        accepted, reason = chain.receive_external_block(block)
        assert accepted, f"Block rejected: {reason}"
        assert chain.height == initial_height + 1

    def test_reject_wrong_height(self, sample_shares):
        chain = ShareChain(block_interval=10.0, settlement_blocks=3)
        block = Block(
            index=99, previous_hash="wrong",
            shares=sample_shares,
        )
        block.finalize()

        accepted, reason = chain.receive_external_block(block)
        assert not accepted

    def test_get_blocks_from(self, sample_shares):
        chain = ShareChain(block_interval=10.0, settlement_blocks=10)
        for i in range(5):
            chain.add_share(ComputeShare(
                node_id="n", shard_index=0,
                session_id=f"s{i}", activation_hash=f"h{i}",
                tokens_processed=1,
            ))
            chain.produce_block()

        blocks = chain.get_blocks_from(0)
        assert len(blocks) >= 5  # genesis + 5 blocks

    def test_node_totals(self):
        chain = ShareChain(block_interval=10.0, settlement_blocks=10)
        for i in range(4):
            chain.add_share(ComputeShare(
                node_id="nodeA", shard_index=0,
                session_id=f"s{i}", activation_hash=f"h{i}",
                tokens_processed=10,
            ))
        chain.add_share(ComputeShare(
            node_id="nodeB", shard_index=1,
            session_id="sx", activation_hash="hx",
            tokens_processed=5,
        ))
        chain.produce_block()

        totals = chain.get_node_totals()
        assert "nodeA" in totals
        assert "nodeB" in totals
        # get_node_totals returns dict[str, float] (weighted share sums)
        assert totals["nodeA"] == 4.0  # 4 shares × weight 1.0
        assert totals["nodeB"] == 1.0  # 1 share × weight 1.0
