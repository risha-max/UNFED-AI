import os
import sys
import time
import json
import hashlib

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
            tickets=[registry_pb2.VerificationTicketProto(
                ticket_id="t-1",
                shard_index=0,
                input_data=b"in-data",
                input_shape=[1],
                input_is_tokens=False,
                expected_output_data=b"ok-data",
                expected_output_shape=[1],
            )],
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
            input_hash=hashlib.sha256(b"in-data").hexdigest(),
            expected_output_hash=hashlib.sha256(b"ok-data").hexdigest(),
            actual_output_hash=hashlib.sha256(b"bad-data").hexdigest(),
            timestamp=time.time(),
            verifier_id=verifier_eth,
            evidence_session_id="t-1",
            evidence_node_id="compute-a",
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


def test_claim_idempotency_and_rate_limit():
    svc = RegistryServicer(no_chain=True)
    verifier = "0x00000000000000000000000000000000000000e2"
    svc._cluster_config.verifier_claim_rate_limit_per_window = 1
    svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(verifier_id=verifier, address="localhost:59010"),
        None,
    )
    svc.SubmitTickets(
        registry_pb2.SubmitTicketsRequest(
            node_id="compute-a",
            tickets=[registry_pb2.VerificationTicketProto(
                ticket_id="t-limit-1",
                shard_index=0,
                input_data=b"aaa",
                input_shape=[1],
                expected_output_data=b"bbb",
                expected_output_shape=[1],
            )],
        ),
        None,
    )
    resp1 = svc.SubmitFraudProof(
        registry_pb2.FraudProofMessage(
            ticket_id="t-limit-1",
            shard_index=0,
            verifier_id=verifier,
            idempotency_key="idem-1",
            input_hash=hashlib.sha256(b"aaa").hexdigest(),
            expected_output_hash=hashlib.sha256(b"bbb").hexdigest(),
            actual_output_hash=hashlib.sha256(b"ccc").hexdigest(),
            evidence_session_id="t-limit-1",
            evidence_node_id="compute-a",
        ),
        None,
    )
    resp1_replay = svc.SubmitFraudProof(
        registry_pb2.FraudProofMessage(
            ticket_id="t-limit-1",
            shard_index=0,
            verifier_id=verifier,
            idempotency_key="idem-1",
            input_hash=hashlib.sha256(b"aaa").hexdigest(),
            expected_output_hash=hashlib.sha256(b"bbb").hexdigest(),
            actual_output_hash=hashlib.sha256(b"ccc").hexdigest(),
            evidence_session_id="t-limit-1",
            evidence_node_id="compute-a",
        ),
        None,
    )
    assert resp1.claim_status == "confirmed"
    assert resp1_replay.message.startswith("idempotent_replay:")

    svc.SubmitTickets(
        registry_pb2.SubmitTicketsRequest(
            node_id="compute-a",
            tickets=[registry_pb2.VerificationTicketProto(
                ticket_id="t-limit-2",
                shard_index=0,
                input_data=b"ddd",
                input_shape=[1],
                expected_output_data=b"eee",
                expected_output_shape=[1],
            )],
        ),
        None,
    )
    resp2 = svc.SubmitFraudProof(
        registry_pb2.FraudProofMessage(
            ticket_id="t-limit-2",
            shard_index=0,
            verifier_id=verifier,
            idempotency_key="idem-2",
            input_hash=hashlib.sha256(b"ddd").hexdigest(),
            expected_output_hash=hashlib.sha256(b"eee").hexdigest(),
            actual_output_hash=hashlib.sha256(b"fff").hexdigest(),
            evidence_session_id="t-limit-2",
            evidence_node_id="compute-a",
        ),
        None,
    )
    assert resp2.claim_status == "rejected"
    assert resp2.message == "rate_limited"


def test_false_claim_penalty_and_cooldown():
    svc = RegistryServicer(no_chain=True)
    verifier = "0x00000000000000000000000000000000000000e3"
    svc._cluster_config.verifier_false_claim_slash_bps = 1000
    svc._cluster_config.verifier_bonus_cooldown_windows = 2
    svc.RegisterVerifier(
        registry_pb2.RegisterVerifierRequest(verifier_id=verifier, address="localhost:59011"),
        None,
    )
    # Seed in-memory stake account so slash has effect.
    svc._payment_contract.stakes.deposit(verifier, 500.0)
    svc._payment_contract.stakes.stake(verifier, 200.0)

    svc.SubmitTickets(
        registry_pb2.SubmitTicketsRequest(
            node_id="compute-b",
            tickets=[registry_pb2.VerificationTicketProto(
                ticket_id="t-reject",
                shard_index=0,
                input_data=b"x",
                input_shape=[1],
                expected_output_data=b"y",
                expected_output_shape=[1],
            )],
        ),
        None,
    )
    rejected = svc.SubmitFraudProof(
        registry_pb2.FraudProofMessage(
            ticket_id="t-reject",
            shard_index=0,
            verifier_id=verifier,
            input_hash=hashlib.sha256(b"x").hexdigest(),
            # Invalid binding: wrong expected hash.
            expected_output_hash=hashlib.sha256(b"not-y").hexdigest(),
            actual_output_hash=hashlib.sha256(b"zzz").hexdigest(),
            evidence_session_id="t-reject",
            evidence_node_id="compute-b",
        ),
        None,
    )
    assert rejected.claim_status == "rejected"

    telemetry = svc.GetInfraTelemetry(registry_pb2.GetInfraTelemetryRequest(), None)
    events = json.loads(telemetry.verifier_penalty_events_json)
    assert any(e.get("kind") == "false_claim_penalty" for e in events)

    svc.SubmitTickets(
        registry_pb2.SubmitTicketsRequest(
            node_id="compute-b",
            tickets=[registry_pb2.VerificationTicketProto(
                ticket_id="t-confirmed-after-reject",
                shard_index=0,
                input_data=b"m",
                input_shape=[1],
                expected_output_data=b"n",
                expected_output_shape=[1],
            )],
        ),
        None,
    )
    confirmed = svc.SubmitFraudProof(
        registry_pb2.FraudProofMessage(
            ticket_id="t-confirmed-after-reject",
            shard_index=0,
            verifier_id=verifier,
            input_hash=hashlib.sha256(b"m").hexdigest(),
            expected_output_hash=hashlib.sha256(b"n").hexdigest(),
            actual_output_hash=hashlib.sha256(b"o").hexdigest(),
            evidence_session_id="t-confirmed-after-reject",
            evidence_node_id="compute-b",
        ),
        None,
    )
    assert confirmed.claim_status == "confirmed"
    claim_status = json.loads(
        svc.GetInfraTelemetry(registry_pb2.GetInfraTelemetryRequest(), None)
        .verifier_claim_status_json
    )
    latest = next(c for c in claim_status if c["claim_id"] == confirmed.claim_id)
    assert latest["bonus_awarded"] is False


def test_multi_verifier_claim_compatibility():
    svc = RegistryServicer(no_chain=True)
    v1 = "0x00000000000000000000000000000000000000e4"
    v2 = "0x00000000000000000000000000000000000000e5"
    svc.RegisterVerifier(registry_pb2.RegisterVerifierRequest(verifier_id=v1, address="a"), None)
    svc.RegisterVerifier(registry_pb2.RegisterVerifierRequest(verifier_id=v2, address="b"), None)
    svc.SubmitTickets(
        registry_pb2.SubmitTicketsRequest(
            node_id="compute-c",
            tickets=[registry_pb2.VerificationTicketProto(
                ticket_id="t-multi",
                shard_index=0,
                input_data=b"i",
                input_shape=[1],
                expected_output_data=b"j",
                expected_output_shape=[1],
            )],
        ),
        None,
    )
    for vid in (v1, v2):
        resp = svc.SubmitFraudProof(
            registry_pb2.FraudProofMessage(
                ticket_id="t-multi",
                shard_index=0,
                verifier_id=vid,
                idempotency_key=f"idem-{vid}",
                input_hash=hashlib.sha256(b"i").hexdigest(),
                expected_output_hash=hashlib.sha256(b"j").hexdigest(),
                actual_output_hash=hashlib.sha256(b"k").hexdigest(),
                evidence_session_id="t-multi",
                evidence_node_id="compute-c",
            ),
            None,
        )
        assert resp.claim_status == "confirmed"
    _, verifier_map = svc._consume_infra_work_maps()
    assert verifier_map[v1] > 0
    assert verifier_map[v2] > 0
