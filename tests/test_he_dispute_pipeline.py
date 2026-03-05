import os
import sys
import time
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from network.he_dispute import sign_report_payload
from network.registry_server import RegistryServicer
from network.share_auth import generate_signing_keypair


def _build_signed_report(
    *,
    reporter_node_id: str,
    sidecar_node_id: str,
    sidecar_stake_identity: str,
    reason_code: str,
    session_id: str = "sess-he-1",
    step: int = 2,
    key_id: str = "kid-he-1",
    request_payload_hash: str | None = None,
    response_payload_hash: str | None = None,
):
    priv, pub = generate_signing_keypair()
    req_hash = request_payload_hash or hashlib.sha256(b"req").hexdigest()
    resp_hash = response_payload_hash or hashlib.sha256(b"resp").hexdigest()
    report_id = f"rpt-{int(time.time() * 1000)}-{reason_code}"
    payload = {
        "report_id": report_id,
        "reporter_node_id": reporter_node_id,
        "reporter_node_type": "output",
        "sidecar_node_id": sidecar_node_id,
        "sidecar_stake_identity": sidecar_stake_identity,
        "session_id": session_id,
        "step": int(step),
        "key_id": key_id,
        "reason_code": reason_code,
        "he_compute_format": "paillier_v1",
        "request_payload_hash": req_hash,
        "response_payload_hash": resp_hash,
        "timestamp": float(time.time()),
    }
    sig = sign_report_payload(priv, payload)
    message = registry_pb2.HESuspicionReport(
        report_id=report_id,
        idempotency_key=f"{reporter_node_id}:{session_id}:{step}:{reason_code}:{resp_hash[:8]}",
        reporter_node_id=reporter_node_id,
        reporter_node_type="output",
        sidecar_node_id=sidecar_node_id,
        sidecar_stake_identity=sidecar_stake_identity,
        session_id=session_id,
        step=int(step),
        key_id=key_id,
        reason_code=reason_code,
        he_compute_format="paillier_v1",
        request_payload_hash=req_hash,
        response_payload_hash=resp_hash,
        evidence_json='{"test":true}',
        reporter_signature=sig,
        reporter_signing_public_key=pub,
        timestamp=payload["timestamp"],
    )
    return message


def test_he_dispute_invalid_verdict_slashes_sidecar_in_enforced_mode():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.he_dispute_rollout_stage = "enforced"
    svc._cluster_config.he_dispute_slash_fraction = 0.5
    sidecar = "0x0000000000000000000000000000000000000abc"

    svc._payment_contract.stakes.deposit(sidecar, 1000.0)
    assert svc._payment_contract.stakes.stake(sidecar, 200.0) is True

    submit = svc.SubmitHESuspicionReport(
        _build_signed_report(
            reporter_node_id="node-output-a",
            sidecar_node_id=sidecar,
            sidecar_stake_identity=sidecar,
            reason_code="response_schema_invalid",
        ),
        None,
    )
    assert submit.accepted is True
    tickets = svc.GetPendingHEDisputes(
        registry_pb2.GetPendingHEDisputesRequest(max_tickets=10, verifier_id="ver-he-1"),
        None,
    )
    assert len(tickets.tickets) == 1
    verdict = svc.SubmitHEVerifierVerdict(
        registry_pb2.HEVerifierVerdict(
            dispute_ticket_id=tickets.tickets[0].dispute_ticket_id,
            verifier_id="ver-he-1",
            verdict="invalid",
            reason="confirmed schema mismatch",
            sidecar_node_id=sidecar,
            sidecar_stake_identity=sidecar,
            timestamp=time.time(),
        ),
        None,
    )
    assert verdict.accepted is True
    assert verdict.status == "confirmed_invalid"
    accounts = svc._payment_contract.stakes.get_all_accounts()
    assert accounts[sidecar]["staked"] == 100.0
    assert accounts[sidecar]["slashed"] == 100.0


def test_he_dispute_valid_verdict_clears_without_slash():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.he_dispute_rollout_stage = "enforced"
    svc._cluster_config.he_dispute_slash_fraction = 0.5
    sidecar = "0x0000000000000000000000000000000000000abd"

    svc._payment_contract.stakes.deposit(sidecar, 1000.0)
    assert svc._payment_contract.stakes.stake(sidecar, 200.0) is True

    submit = svc.SubmitHESuspicionReport(
        _build_signed_report(
            reporter_node_id="node-output-b",
            sidecar_node_id=sidecar,
            sidecar_stake_identity=sidecar,
            reason_code="sampled_deep_audit",
            session_id="sess-he-valid",
            step=3,
        ),
        None,
    )
    assert submit.accepted is True
    tickets = svc.GetPendingHEDisputes(
        registry_pb2.GetPendingHEDisputesRequest(max_tickets=10, verifier_id="ver-he-2"),
        None,
    )
    verdict = svc.SubmitHEVerifierVerdict(
        registry_pb2.HEVerifierVerdict(
            dispute_ticket_id=tickets.tickets[0].dispute_ticket_id,
            verifier_id="ver-he-2",
            verdict="valid",
            reason="sampled audit clean",
            sidecar_node_id=sidecar,
            sidecar_stake_identity=sidecar,
            timestamp=time.time(),
        ),
        None,
    )
    assert verdict.accepted is True
    assert verdict.status == "confirmed_valid"
    accounts = svc._payment_contract.stakes.get_all_accounts()
    assert accounts[sidecar]["staked"] == 200.0
    assert accounts[sidecar]["slashed"] == 0.0


def test_he_dispute_missing_evidence_is_rejected():
    svc = RegistryServicer(no_chain=True)
    sidecar = "0x0000000000000000000000000000000000000abe"
    report = _build_signed_report(
        reporter_node_id="node-output-c",
        sidecar_node_id=sidecar,
        sidecar_stake_identity=sidecar,
        reason_code="response_schema_invalid",
    )
    report.request_payload_hash = ""
    submit = svc.SubmitHESuspicionReport(report, None)
    assert submit.accepted is False
    assert submit.status == "rejected"
    tickets = svc.GetPendingHEDisputes(
        registry_pb2.GetPendingHEDisputesRequest(max_tickets=10, verifier_id="ver-he-3"),
        None,
    )
    assert len(tickets.tickets) == 0


def test_he_dispute_report_spam_limits_are_enforced():
    svc = RegistryServicer(no_chain=True)
    svc._cluster_config.he_dispute_report_rate_limit_per_window = 1
    sidecar = "0x0000000000000000000000000000000000000abf"
    first = svc.SubmitHESuspicionReport(
        _build_signed_report(
            reporter_node_id="node-output-spam",
            sidecar_node_id=sidecar,
            sidecar_stake_identity=sidecar,
            reason_code="response_schema_invalid",
            session_id="sess-spam-1",
        ),
        None,
    )
    second = svc.SubmitHESuspicionReport(
        _build_signed_report(
            reporter_node_id="node-output-spam",
            sidecar_node_id=sidecar,
            sidecar_stake_identity=sidecar,
            reason_code="response_schema_invalid",
            session_id="sess-spam-2",
        ),
        None,
    )
    assert first.accepted is True
    assert second.accepted is False
    assert second.status == "rate_limited"


def test_he_dispute_rollout_stages_shadow_soft_enforced():
    sidecar = "0x0000000000000000000000000000000000000ac0"
    expected_stake = {
        "shadow": 200.0,
        "soft": 175.0,      # 25% of configured 0.5 slash => 12.5% total
        "enforced": 100.0,  # full configured 0.5 slash
    }
    for stage, expected in expected_stake.items():
        svc = RegistryServicer(no_chain=True)
        svc._cluster_config.he_dispute_rollout_stage = stage
        svc._cluster_config.he_dispute_slash_fraction = 0.5
        svc._payment_contract.stakes.deposit(sidecar, 1000.0)
        assert svc._payment_contract.stakes.stake(sidecar, 200.0) is True
        submit = svc.SubmitHESuspicionReport(
            _build_signed_report(
                reporter_node_id=f"node-output-{stage}",
                sidecar_node_id=sidecar,
                sidecar_stake_identity=sidecar,
                reason_code="response_schema_invalid",
                session_id=f"sess-{stage}",
            ),
            None,
        )
        assert submit.accepted is True
        tickets = svc.GetPendingHEDisputes(
            registry_pb2.GetPendingHEDisputesRequest(max_tickets=10, verifier_id=f"ver-{stage}"),
            None,
        )
        verdict = svc.SubmitHEVerifierVerdict(
            registry_pb2.HEVerifierVerdict(
                dispute_ticket_id=tickets.tickets[0].dispute_ticket_id,
                verifier_id=f"ver-{stage}",
                verdict="invalid",
                reason=f"{stage} stage invalid",
                sidecar_node_id=sidecar,
                sidecar_stake_identity=sidecar,
                timestamp=time.time(),
            ),
            None,
        )
        assert verdict.accepted is True
        accounts = svc._payment_contract.stakes.get_all_accounts()
        assert accounts[sidecar]["staked"] == expected
