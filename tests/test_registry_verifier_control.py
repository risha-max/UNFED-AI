import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from network.registry_server import NodeRecord, RegistryServicer


def test_verifier_register_and_health():
    svc = RegistryServicer(no_chain=True)
    reg = svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(
            verifier_id="ver-1",
            address="localhost:59001",
        ),
        None,
    )
    assert reg.success is True
    health = svc.GetVerifierHealth(registry_pb2.GetVerifierHealthRequest(), None)
    assert health.healthy is True
    assert health.healthy_verifier_count >= 1
    assert health.required_verifier_count == 1


def test_verifier_stale_timeout_removal():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.verifier_heartbeat_timeout_seconds = 1
    svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(
            verifier_id="ver-stale",
            address="localhost:59002",
        ),
        None,
    )
    with svc._verifier_lock:
        svc._verifiers["ver-stale"].last_heartbeat = time.time() - 10
    svc._remove_stale_verifiers()
    health = svc.GetVerifierHealth(registry_pb2.GetVerifierHealthRequest(), None)
    assert health.healthy is False
    assert health.healthy_verifier_count == 0


def test_get_verifier_config_reflects_cluster_policy():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.verifier_poll_interval_seconds = 2.5
    svc._cluster_config.verifier_max_tickets_per_poll = 17
    cfg = svc.GetVerifierConfig(
        registry_pb2.GetVerifierConfigRequest(verifier_id="any"),
        None,
    )
    assert cfg.success is True
    assert '"poll_interval_seconds": 2.5' in cfg.config_json
    assert '"max_tickets_per_poll": 17' in cfg.config_json


def test_daemon_health_uses_required_count_and_timeout():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.daemon_required_count = 2
    svc._cluster_config.daemon_heartbeat_timeout_seconds = 2
    with svc._lock:
        svc._nodes["d1"] = NodeRecord(
            node_id="d1",
            address="localhost:50070",
            model_id="",
            shard_index=-1,
            layer_start=0,
            layer_end=0,
            has_embedding=False,
            has_lm_head=False,
            node_type="daemon",
        )
        svc._nodes["d2"] = NodeRecord(
            node_id="d2",
            address="localhost:50071",
            model_id="",
            shard_index=-1,
            layer_start=0,
            layer_end=0,
            has_embedding=False,
            has_lm_head=False,
            node_type="daemon",
        )
    assert svc._healthy_daemon_count() == 2
    assert svc._daemon_healthy() is True
    with svc._lock:
        svc._nodes["d2"].last_heartbeat = time.time() - 10
    assert svc._healthy_daemon_count() == 1
    assert svc._daemon_healthy() is False


def test_infra_telemetry_reports_window_and_payout_shares():
    svc = RegistryServicer(no_chain=True)
    verifier_eth = "0x00000000000000000000000000000000000000e1"
    svc._cluster_config.verifier_fraud_bonus_weight = 2.0
    svc._cluster_config.daemon_required_count = 1
    svc._cluster_config.verifier_required_count = 1
    with svc._lock:
        svc._nodes["0x00000000000000000000000000000000000000d1"] = NodeRecord(
            node_id="0x00000000000000000000000000000000000000d1",
            address="localhost:50070",
            model_id="",
            shard_index=-1,
            layer_start=0,
            layer_end=0,
            has_embedding=False,
            has_lm_head=False,
            node_type="daemon",
        )
    svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(
            verifier_id=verifier_eth,
            address="localhost:59001",
        ),
        None,
    )

    with svc._infra_work_lock:
        svc._daemon_work_window["0x00000000000000000000000000000000000000d1"] = 5.0
    svc.SubmitTickets(
        registry_pb2.SubmitTicketsRequest(
            node_id="compute-a",
            tickets=[registry_pb2.VerificationTicketProto(ticket_id="t-1", shard_index=0)],
        ),
        None,
    )
    svc.GetPendingTickets(
        registry_pb2.GetPendingTicketsRequest(max_tickets=10, verifier_id=verifier_eth),
        None,
    )
    svc.SubmitFraudProof(
        registry_pb2.FraudProofMessage(
            ticket_id="t-1",
            shard_index=0,
            input_hash="in",
            expected_output_hash="ok",
            actual_output_hash="bad",
            timestamp=time.time(),
            verifier_id=verifier_eth,
        ),
        None,
    )

    daemon_map, verifier_map = svc._consume_infra_work_maps()
    assert daemon_map["0x00000000000000000000000000000000000000d1"] == 5.0
    assert verifier_map[verifier_eth] == 3.0

    telemetry = svc.GetInfraTelemetry(registry_pb2.GetInfraTelemetryRequest(), None)
    assert telemetry.healthy_daemon_count == 1
    assert telemetry.required_daemon_count == 1
    assert telemetry.healthy_verifier_count == 1
    assert telemetry.required_verifier_count == 1
    assert telemetry.selected_daemon_recipient == "0x00000000000000000000000000000000000000d1"
    assert telemetry.selected_verifier_recipient == verifier_eth
    assert json.loads(telemetry.daemon_work_window_json) == {}
    assert json.loads(telemetry.verifier_work_window_json) == {}
    assert json.loads(telemetry.daemon_payout_share_json) == {
        "0x00000000000000000000000000000000000000d1": 1.0
    }
    assert json.loads(telemetry.verifier_payout_share_json) == {verifier_eth: 1.0}
