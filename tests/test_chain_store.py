"""Unit tests for ChainStore SQLite persistence."""

import os
import tempfile
import time

import pytest

from economics.share_chain import Block, ComputeShare, SettlementSummary
from economics.chain_store import ChainStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_chain.db")
        s = ChainStore(db_path=db_path)
        yield s
        s.close()


def _make_share(node_id="node_a", shard_index=0, tokens=10):
    return ComputeShare(
        node_id=node_id,
        shard_index=shard_index,
        session_id="sess_1",
        activation_hash="",
        tokens_processed=tokens,
        timestamp=time.time(),
        share_weight=1.0,
    )


def _make_block(index=1, shares=None):
    if shares is None:
        shares = [_make_share()]
    block = Block(
        index=index,
        previous_hash="0" * 64,
        shares=shares,
        timestamp=time.time(),
    )
    block.finalize()
    return block


class TestChainStore:

    def test_save_and_load_block(self, store):
        block = _make_block(index=1)
        assert store.save_block(block) is True

        loaded = store.load_blocks(from_height=1, limit=1)
        assert len(loaded) == 1
        assert loaded[0].index == 1
        assert loaded[0].block_hash == block.block_hash

    def test_duplicate_block_returns_false(self, store):
        block = _make_block(index=1)
        assert store.save_block(block) is True
        assert store.save_block(block) is False

    def test_save_settlement_and_query(self, store):
        settlement = SettlementSummary(
            period_start=time.time() - 100,
            period_end=time.time(),
            block_range=(1, 10),
            node_shares={"node_a": 5.0, "node_b": 3.0},
            total_shares=8.0,
            total_tokens=80,
        )
        settlement.finalize()
        store.save_settlement(settlement)

        settlements = store.load_settlements()
        assert len(settlements) >= 1

    def test_prune_removes_old_blocks(self, store):
        for i in range(1, 21):
            store.save_block(_make_block(index=i))

        store.prune(keep_last_n=5)
        remaining = store.load_blocks(from_height=0, limit=100)
        for b in remaining:
            assert b.index >= 15, f"Block {b.index} should have been pruned"

    def test_vacuum_frequency(self, store):
        for i in range(1, 102):
            store.save_block(_make_block(index=i))

        initial_count = store._prune_count
        for j in range(50):
            store.prune(keep_last_n=50 - j % 10)

        assert store._prune_count > initial_count

    def test_get_node_totals_with_from_block(self, store):
        shares_b1 = [_make_share("node_a", tokens=10)]
        shares_b2 = [_make_share("node_a", tokens=20),
                      _make_share("node_b", tokens=5)]
        store.save_block(_make_block(index=1, shares=shares_b1))
        store.save_block(_make_block(index=2, shares=shares_b2))

        totals_all = store.get_node_totals(from_block=0)
        assert totals_all.get("node_a", 0) == 30
        assert totals_all.get("node_b", 0) == 5

        totals_from_2 = store.get_node_totals(from_block=2)
        assert totals_from_2.get("node_a", 0) == 20
        assert totals_from_2.get("node_b", 0) == 5
