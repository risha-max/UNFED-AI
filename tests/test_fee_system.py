"""
Tests for the Dynamic Fee System (EIP-1559-style).

Tests:
  1. FeeOracle basic adjustments (up and down)
  2. Weighted share creation (compute, MPC, guard)
  3. Weighted settlement tally
  4. PaymentContract dynamic pricing
  5. Proto round-trip with share_weight
  6. Config constants exist
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pytest
from economics.fee_oracle import FeeOracle
from economics.share_chain import ComputeShare, Block, SettlementSummary, ShareChain
from economics.payments import PaymentContract, StakeManager


# ---------------------------------------------------------------------------
# FeeOracle Tests
# ---------------------------------------------------------------------------

class TestFeeOracle:
    def test_initial_state(self):
        oracle = FeeOracle(base_fee=0.001)
        assert oracle.get_base_fee() == 0.001
        assert oracle.get_utilization() == 0.0

    def test_fee_increases_when_busy(self):
        oracle = FeeOracle(
            base_fee=0.001,
            target_utilization=0.7,
            target_capacity=10,
            adjustment_factor=0.125,
        )
        initial_fee = oracle.get_base_fee()

        # Simulate high-utilization blocks (10 shares each, full capacity)
        for i in range(5):
            block = Block(
                index=i + 1,
                previous_hash="0" * 64,
                shares=[
                    ComputeShare(
                        node_id=f"node-{j}",
                        shard_index=0,
                        session_id="sess",
                        activation_hash="abc",
                        tokens_processed=1,
                    )
                    for j in range(10)
                ],
            )
            block.finalize()
            oracle.update(block)

        assert oracle.get_base_fee() > initial_fee, \
            "Fee should increase when utilization > target"

    def test_fee_decreases_when_idle(self):
        oracle = FeeOracle(
            base_fee=0.01,
            target_utilization=0.7,
            target_capacity=10,
            adjustment_factor=0.125,
        )
        initial_fee = oracle.get_base_fee()

        # Simulate low-utilization blocks (1 share each)
        for i in range(5):
            block = Block(
                index=i + 1,
                previous_hash="0" * 64,
                shares=[
                    ComputeShare(
                        node_id="node-0",
                        shard_index=0,
                        session_id="sess",
                        activation_hash="abc",
                        tokens_processed=1,
                    )
                ],
            )
            block.finalize()
            oracle.update(block)

        assert oracle.get_base_fee() < initial_fee, \
            "Fee should decrease when utilization < target"

    def test_fee_clamped_to_bounds(self):
        oracle = FeeOracle(base_fee=0.001, min_fee=0.0001, max_fee=0.1)

        # Simulate many idle blocks to push fee down
        for i in range(100):
            block = Block(index=i + 1, previous_hash="0" * 64, shares=[])
            block.finalize()
            oracle.update(block)

        assert oracle.get_base_fee() >= 0.0001, "Fee should not go below min"

    def test_fee_estimate(self):
        oracle = FeeOracle(base_fee=0.002)
        est = oracle.get_fee_estimate(100)
        assert est["base_fee"] == 0.002
        assert est["estimated_cost"] == 0.2
        assert est["estimated_tokens"] == 100

    def test_fee_history(self):
        oracle = FeeOracle(base_fee=0.001)
        block = Block(index=1, previous_hash="0" * 64, shares=[])
        block.finalize()
        oracle.update(block)

        history = oracle.get_fee_history()
        assert len(history) == 1
        assert "base_fee" in history[0]
        assert "utilization" in history[0]


# ---------------------------------------------------------------------------
# Weighted Share Tests
# ---------------------------------------------------------------------------

class TestWeightedShares:
    def test_compute_share_default_weight(self):
        share = ComputeShare(
            node_id="node-1",
            shard_index=0,
            session_id="sess",
            activation_hash="abc",
            tokens_processed=1,
        )
        assert share.share_weight == 1.0

    def test_guard_share_weight(self):
        share = ComputeShare(
            node_id="guard-1",
            shard_index=-1,
            session_id="sess",
            activation_hash="abc",
            tokens_processed=1,
            share_weight=0.05,
        )
        assert share.share_weight == 0.05

    def test_share_to_dict_includes_weight(self):
        share = ComputeShare(
            node_id="node-1",
            shard_index=0,
            session_id="sess",
            activation_hash="abc",
            tokens_processed=1,
            share_weight=0.05,
        )
        d = share.to_dict()
        assert "share_weight" in d
        assert d["share_weight"] == 0.05

    def test_share_from_dict_includes_weight(self):
        d = {
            "node_id": "node-1",
            "shard_index": 0,
            "session_id": "sess",
            "activation_hash": "abc",
            "tokens_processed": 1,
            "share_weight": 0.05,
        }
        share = ComputeShare.from_dict(d)
        assert share.share_weight == 0.05

    def test_share_from_dict_default_weight(self):
        """Old shares without share_weight field should default to 1.0."""
        d = {
            "node_id": "node-1",
            "shard_index": 0,
            "session_id": "sess",
            "activation_hash": "abc",
            "tokens_processed": 1,
        }
        share = ComputeShare.from_dict(d)
        assert share.share_weight == 1.0


# ---------------------------------------------------------------------------
# Weighted Settlement Tests
# ---------------------------------------------------------------------------

class TestWeightedSettlement:
    def _make_chain_with_shares(self):
        """Create a chain with a mix of compute and guard shares."""
        chain = ShareChain(block_interval=10.0, settlement_blocks=2)

        # Compute shares (weight 1.0)
        for i in range(5):
            chain.add_share(ComputeShare(
                node_id="compute-node",
                shard_index=0,
                session_id=f"sess-{i}",
                activation_hash="abc",
                tokens_processed=1,
                share_weight=1.0,
            ))

        # Guard shares (weight 0.05)
        for i in range(3):
            chain.add_share(ComputeShare(
                node_id="guard-node",
                shard_index=-1,
                session_id=f"guard-sess-{i}",
                activation_hash="def",
                tokens_processed=1,
                share_weight=0.05,
            ))

        return chain

    def test_weighted_settlement_tally(self):
        chain = self._make_chain_with_shares()

        # Produce block 1
        b1 = chain.produce_block()
        assert b1 is not None

        # Check node totals use weighted sums
        totals = chain.get_node_totals()
        assert abs(totals["compute-node"] - 5.0) < 1e-9, \
            "Compute node should have 5.0 weighted shares"
        assert abs(totals["guard-node"] - 0.15) < 1e-9, \
            "Guard node should have 0.15 weighted shares (3 * 0.05)"

    def test_settlement_summary_float(self):
        chain = self._make_chain_with_shares()

        # Force produce 2 blocks to trigger settlement
        chain.produce_block()
        chain.add_share(ComputeShare(
            node_id="compute-node",
            shard_index=0,
            session_id="extra",
            activation_hash="xyz",
            tokens_processed=1,
            share_weight=1.0,
        ))
        chain.produce_block()

        settlements = chain.get_settlements()
        assert len(settlements) >= 1

        s = settlements[0]
        assert isinstance(s.total_shares, float)
        assert isinstance(s.node_shares.get("compute-node", 0), float)

    def test_settlement_from_dict_backward_compat(self):
        """Old settlements with int values should convert to float."""
        d = {
            "period_start": 0.0,
            "period_end": 1.0,
            "block_range": [1, 2],
            "node_shares": {"node-1": 5, "node-2": 3},
            "total_shares": 8,
        }
        s = SettlementSummary.from_dict(d)
        assert isinstance(s.total_shares, float)
        assert isinstance(s.node_shares["node-1"], float)
        assert s.total_shares == 8.0


# ---------------------------------------------------------------------------
# PaymentContract Dynamic Fee Tests
# ---------------------------------------------------------------------------

class TestPaymentContractDynamic:
    def test_dynamic_fee_post_settlement(self):
        oracle = FeeOracle(base_fee=0.002)
        stakes = StakeManager()
        contract = PaymentContract(stakes, fee_oracle=oracle)

        summary = SettlementSummary(
            period_start=0.0,
            period_end=1.0,
            block_range=(1, 2),
            node_shares={"compute-node": 10.0, "guard-node": 0.15},
            total_shares=10.15,
        )
        summary.finalize()

        settlement = contract.post_settlement(summary)
        # Total payout should be total_shares * base_fee
        expected = 10.15 * 0.002
        assert abs(settlement.total_payout - expected) < 1e-9

    def test_fallback_to_static_price(self):
        stakes = StakeManager()
        contract = PaymentContract(stakes, price_per_share=0.005)

        summary = SettlementSummary(
            period_start=0.0,
            period_end=1.0,
            block_range=(1, 2),
            node_shares={"node-1": 10.0},
            total_shares=10.0,
        )
        summary.finalize()

        settlement = contract.post_settlement(summary)
        expected = 10.0 * 0.005
        assert abs(settlement.total_payout - expected) < 1e-9


# ---------------------------------------------------------------------------
# Proto Round-trip Tests
# ---------------------------------------------------------------------------

class TestProtoRoundtrip:
    def test_share_weight_proto(self):
        from economics.distributed_chain import share_to_proto, proto_to_share

        original = ComputeShare(
            node_id="node-1",
            shard_index=0,
            session_id="sess",
            activation_hash="abc",
            tokens_processed=1,
            share_weight=0.05,
        )

        proto = share_to_proto(original)
        assert proto.share_weight == 0.05

        restored = proto_to_share(proto)
        assert restored.share_weight == 0.05
        assert restored.node_id == "node-1"

    def test_share_weight_default_proto(self):
        """Proto with share_weight=0 (unset) should default to 1.0."""
        from economics.distributed_chain import proto_to_share
        from proto import inference_pb2

        proto = inference_pb2.ShareProto(
            node_id="node-1",
            shard_index=0,
            session_id="sess",
            activation_hash="abc",
            tokens_processed=1,
            timestamp=time.time(),
            # share_weight not set = 0 in protobuf
        )
        share = proto_to_share(proto)
        assert share.share_weight == 1.0


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_fee_config_exists(self):
        import config
        assert hasattr(config, 'FEE_TARGET_UTILIZATION')
        assert hasattr(config, 'FEE_BASE_DEFAULT')
        assert hasattr(config, 'FEE_MIN')
        assert hasattr(config, 'FEE_MAX')
        assert hasattr(config, 'FEE_ADJUSTMENT_FACTOR')
        assert hasattr(config, 'FEE_WINDOW_BLOCKS')
        assert hasattr(config, 'FEE_TARGET_CAPACITY')
        assert hasattr(config, 'GUARD_FEE_RATIO')

    def test_guard_fee_ratio_value(self):
        import config
        assert config.GUARD_FEE_RATIO == 0.05

    def test_proto_fee_fields(self):
        """Verify proto stubs have new fields."""
        from proto import inference_pb2

        # ForwardRequest should have fee_per_token and tip
        req = inference_pb2.ForwardRequest()
        req.fee_per_token = 0.002
        req.tip = 0.001
        assert req.fee_per_token == 0.002
        assert req.tip == 0.001

        # FeeEstimateRequest/Response
        est_req = inference_pb2.FeeEstimateRequest(estimated_tokens=100)
        assert est_req.estimated_tokens == 100

        est_resp = inference_pb2.FeeEstimateResponse(
            base_fee=0.001,
            utilization=0.5,
            estimated_cost=0.1,
            suggested_tip=0.0,
        )
        assert est_resp.base_fee == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
