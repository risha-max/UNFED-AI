import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from network.verifier_node import _adjudicate_he_dispute


def test_adjudicate_he_dispute_flags_inline_anomaly_invalid():
    ticket = registry_pb2.HEDisputeTicket(
        dispute_ticket_id="t1",
        session_id="sess",
        step=1,
        reason_code="response_schema_invalid",
        he_compute_format="paillier_v1",
        request_payload_hash="a",
        response_payload_hash="b",
    )
    verdict, reason, proof = _adjudicate_he_dispute(ticket)
    assert verdict == "invalid"
    assert "inline_integrity_violation" in reason
    assert proof["inline_check_failed"] == "response_schema_invalid"


def test_adjudicate_he_dispute_missing_evidence_insufficient():
    ticket = registry_pb2.HEDisputeTicket(
        dispute_ticket_id="t2",
        session_id="",
        step=0,
        reason_code="sampled_deep_audit",
        he_compute_format="",
        request_payload_hash="",
        response_payload_hash="",
    )
    verdict, _, _ = _adjudicate_he_dispute(ticket)
    assert verdict == "insufficient_evidence"
