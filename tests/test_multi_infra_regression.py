import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from economics.payments import PaymentContract, SettlementSummary, StakeManager
from network.registry_server import NodeRecord, RegistryServicer


def _make_summary(node_shares: dict[str, float]) -> SettlementSummary:
    total_shares = float(sum(node_shares.values()))
    return SettlementSummary(
        period_start=0.0,
        period_end=1.0,
        block_range=(1, 6),
        node_shares=node_shares,
        total_shares=total_shares,
        total_tokens=int(total_shares),
        settlement_hash="s-hash",
    )


def test_registry_consume_infra_maps_filters_stale_and_applies_hybrid_weight():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.verifier_fraud_bonus_weight = 2.5
    svc._cluster_config.daemon_heartbeat_timeout_seconds = 2
    svc._cluster_config.verifier_heartbeat_timeout_seconds = 2

    with svc._lock:
        svc._nodes["d-healthy"] = NodeRecord(
            node_id="d-healthy",
            address="localhost:50070",
            model_id="",
            shard_index=-1,
            layer_start=0,
            layer_end=0,
            has_embedding=False,
            has_lm_head=False,
            node_type="daemon",
        )
        svc._nodes["d-stale"] = NodeRecord(
            node_id="d-stale",
            address="localhost:50071",
            model_id="",
            shard_index=-1,
            layer_start=0,
            layer_end=0,
            has_embedding=False,
            has_lm_head=False,
            node_type="daemon",
        )
        svc._nodes["d-stale"].last_heartbeat = time.time() - 100

    v1 = "0x0000000000000000000000000000000000000aa1"
    v2 = "0x0000000000000000000000000000000000000aa2"
    svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(verifier_id=v1, address="localhost:59001"),
        None,
    )
    svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(verifier_id=v2, address="localhost:59002"),
        None,
    )
    with svc._verifier_lock:
        svc._verifiers[v2].last_heartbeat = time.time() - 100

    with svc._infra_work_lock:
        svc._daemon_work_window["d-healthy"] = 10.0
        svc._daemon_work_window["d-stale"] = 99.0
        svc._verifier_ticket_window[v1] = 4.0
        svc._verifier_bonus_window[v1] = 2.0  # 4 + (2 * 2.5) = 9
        svc._verifier_ticket_window[v2] = 10.0

    daemon_map, verifier_map = svc._consume_infra_work_maps()
    assert daemon_map == {"d-healthy": 10.0}
    assert verifier_map == {v1: 9.0}


def test_payment_split_distributes_multi_infra_work_proportionally():
    contract = PaymentContract(StakeManager())
    contract.deposit_to_escrow(1000.0)
    contract.report_usage(100, 50)

    summary = _make_summary({"c1": 6.0, "c2": 4.0})
    settlement = contract.post_settlement(
        summary,
        daemon_fee_bps=200,  # 2% of 0.01 = 0.0002
        verifier_fee_bps=100,  # 1% of 0.01 = 0.0001
        daemon_work_map={"d1": 3.0, "d2": 1.0},  # 75/25 split
        verifier_work_map={"v1": 1.0, "v2": 2.0},  # 33/67 split
    )
    split = contract.settlement_payout_split(settlement.settlement_hash)

    # Compute pool remains 97% of payout.
    assert split["c1"] == pytest.approx(0.00582, rel=1e-9)
    assert split["c2"] == pytest.approx(0.00388, rel=1e-9)
    # Daemon cut: 0.0002 -> d1=0.00015, d2=0.00005
    assert split["d1"] == pytest.approx(0.00015, rel=1e-9)
    assert split["d2"] == pytest.approx(0.00005, rel=1e-9)
    # Verifier cut: 0.0001 -> v1=0.000033..., v2=0.000066...
    assert split["v1"] == pytest.approx(0.0001 / 3.0, rel=1e-9)
    assert split["v2"] == pytest.approx(0.0002 / 3.0, rel=1e-9)

