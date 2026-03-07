import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import inference_pb2
from network.forward_attestation import (
    FORWARD_ATTESTATION_VERSION,
    ForwardAttestationPayload,
    make_tensor_shape_signature,
    proof_bytes_hash,
    sign_forward_attestation,
    tensor_bytes_digest,
)
from network.share_auth import generate_signing_keypair
from node.server import InferenceNodeServicer


def _build_attestation_servicer() -> InferenceNodeServicer:
    servicer = InferenceNodeServicer.__new__(InferenceNodeServicer)
    servicer._require_forward_attestation = True
    servicer._require_forward_proof = False
    servicer._forward_proof_max_bytes = 8192
    servicer._forward_proof_allowed_formats = {"none"}
    servicer._forward_signer_pubkeys = {}
    servicer._forward_signer_cache_ts = 0.0
    servicer._forward_signer_cache_ttl_seconds = 10.0
    servicer._refresh_forward_signer_cache = lambda force=False: None
    return servicer


def _make_signed_request(
    *,
    signer_node_id: str,
    signer_node_type: str,
    private_key: bytes,
    activation_data: bytes,
    tensor_shape: list[int],
) -> inference_pb2.ForwardRequest:
    req = inference_pb2.ForwardRequest(
        session_id="sess-forward-attest",
        activation_data=activation_data,
        tensor_shape=tensor_shape,
        compressed=False,
        wire_dtype="float32",
        route_prev_node_id=signer_node_id,
        route_prev_node_type=signer_node_type,
        prev_attestation_signer_node_id=signer_node_id,
        prev_attestation_version=FORWARD_ATTESTATION_VERSION,
        prev_attestation_proof_format="none",
        prev_attestation_proof=b"",
    )
    activation_digest = tensor_bytes_digest(activation_data)
    payload = ForwardAttestationPayload(
        version=FORWARD_ATTESTATION_VERSION,
        signer_node_id=signer_node_id,
        signer_node_type=signer_node_type,
        session_id=req.session_id,
        he_step=int(req.he_step),
        he_key_id=req.he_key_id,
        activation_digest=activation_digest,
        tensor_shape=make_tensor_shape_signature(tensor_shape),
        compressed=False,
        wire_dtype=req.wire_dtype,
        output_mpc_payload_hash=req.output_mpc_payload_hash,
        proof_format="none",
        proof_hash=proof_bytes_hash(b""),
    )
    signature = sign_forward_attestation(private_key, payload)
    req.prev_activation_digest = activation_digest
    req.prev_attestation_signature = signature
    return req


def test_verify_forward_attestation_accepts_valid_signature():
    priv, pub = generate_signing_keypair()
    servicer = _build_attestation_servicer()
    servicer._forward_signer_pubkeys = {"node-A": pub}
    req = _make_signed_request(
        signer_node_id="node-A",
        signer_node_type="compute",
        private_key=priv,
        activation_data=b"\x01\x02\x03\x04",
        tensor_shape=[1, 1, 1, 1],
    )

    ok, reason = servicer._verify_forward_attestation(req)
    assert ok is True
    assert reason == "ok"


def test_verify_forward_attestation_rejects_bad_signature():
    priv, pub = generate_signing_keypair()
    servicer = _build_attestation_servicer()
    servicer._forward_signer_pubkeys = {"node-A": pub}
    req = _make_signed_request(
        signer_node_id="node-A",
        signer_node_type="compute",
        private_key=priv,
        activation_data=b"\x05\x06\x07\x08",
        tensor_shape=[1, 1, 1, 1],
    )
    req.prev_attestation_signature = b"\x00" * 64

    ok, reason = servicer._verify_forward_attestation(req)
    assert ok is False
    assert reason == "invalid_attestation_signature"


def test_verify_forward_attestation_rejects_digest_mismatch():
    priv, pub = generate_signing_keypair()
    servicer = _build_attestation_servicer()
    servicer._forward_signer_pubkeys = {"node-A": pub}
    req = _make_signed_request(
        signer_node_id="node-A",
        signer_node_type="compute",
        private_key=priv,
        activation_data=b"\x09\x0a\x0b\x0c",
        tensor_shape=[1, 1, 1, 1],
    )
    req.prev_activation_digest = "deadbeef"

    ok, reason = servicer._verify_forward_attestation(req)
    assert ok is False
    assert reason == "activation_digest_mismatch"
