import os
import sys
import threading

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import inference_pb2
import node.server as node_server_mod
from network.he_compute import (
    decrypt_topk_artifact,
    generate_client_compute_keypair,
)
from network.mpc_output import (
    HE_COMPUTE_MODE_MPC_N_MINUS_1_N,
    build_output_mpc_request_payload,
    parse_output_mpc_response_payload,
)
from node.server import InferenceNodeServicer


class _DummyTicketCollector:
    def should_sample(self):
        return False

    def collect(self, **kwargs):
        return None


class _DummyRunner:
    def __init__(self):
        self.sample_token_flags = []
        self.lm_head = lambda h: torch.tensor([[[0.1, 2.3, 1.0, -0.4]]], dtype=torch.float32)

    def forward(
        self,
        hidden_states=None,
        token_ids=None,
        session_id="default",
        prefix_session_id="",
        prefix_length=0,
        image_embeddings=None,
        mrope_position_ids=None,
        sample_token=True,
    ):
        self.sample_token_flags.append(bool(sample_token))
        hidden = torch.zeros(1, 1, 4, dtype=torch.float32)
        sampled = 1 if sample_token else None
        return hidden, sampled


class _DummyContext:
    def __init__(self):
        self.details = ""
        self.code = None

    def set_details(self, details):
        self.details = details

    def set_code(self, code):
        self.code = code


def _build_light_servicer() -> InferenceNodeServicer:
    # Bypass heavy __init__ and set only attributes needed by Forward().
    servicer = InferenceNodeServicer.__new__(InferenceNodeServicer)
    servicer.shard_index = 999
    servicer.has_embedding = False
    servicer.has_lm_head = True
    servicer.runner = _DummyRunner()
    servicer._require_daemon = False
    servicer._daemon_stub = None
    servicer._inference_lock = threading.Lock()
    servicer._active_inferences = 0
    servicer._cheat_rate = 0.0
    servicer._ticket_collector = _DummyTicketCollector()
    servicer._seen_sessions = set()
    servicer.registration = None
    servicer._eos_token_id = 151643
    servicer._he_report_lock = threading.Lock()
    servicer._he_report_counts_by_window = {}
    servicer._wire_dtype_default = "float32"
    servicer._compress_activations = True
    servicer._compress_threshold = 16384
    servicer._he_compute_mode_default = "off"
    servicer._he_compute_top_k = 64
    servicer._he_compute_temperature = 1.0
    servicer._he_compute_top_p = 1.0
    servicer._he_full_vocab_sidecar_url = ""
    servicer._he_full_vocab_sidecar_timeout_ms = 2000
    servicer._he_full_vocab_sidecar_required = False
    servicer._he_sidecar_allowed_formats = {"paillier_v1"}
    servicer._he_sidecar_max_payload_bytes = 2 * 1024 * 1024
    servicer._he_dispute_sampling_rate = 0.05
    servicer._he_dispute_report_rate_limit_per_window = 64
    servicer._he_dispute_window_seconds = 60
    servicer._allowed_prev_node_types = set()
    servicer._require_forward_attestation = False
    servicer._require_forward_proof = False
    servicer._forward_proof_max_bytes = 8192
    servicer._forward_proof_allowed_formats = {"none"}
    servicer._forward_signer_pubkeys = {}
    servicer._forward_signer_cache_ts = 0.0
    servicer._forward_signer_cache_ttl_seconds = 10.0
    servicer._report_he_suspicion = lambda **kwargs: None
    return servicer


def test_he_compute_mode_disables_plaintext_sampling_and_token_fields():
    priv, pub = generate_client_compute_keypair()
    servicer = _build_light_servicer()
    context = _DummyContext()
    req = inference_pb2.ForwardRequest(
        session_id="sess-he-node-1",
        activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
        tensor_shape=[1, 1, 4],
        is_prefill=False,
        he_output_enabled=True,
        he_client_pubkey=pub,
        he_key_id="kid-he-node-1",
        he_step=7,
        he_compute_mode="decode_client_sample",
        he_disable_plaintext_sampling=True,
        he_top_k=3,
    )

    resp = servicer.Forward(req, context)

    # Critical check: output node did not run plaintext token sampling.
    assert servicer.runner.sample_token_flags == [False]

    # Critical check: no plaintext token returned.
    assert resp.has_token is False
    assert not resp.he_error
    assert resp.he_compute_payload

    # Artifact is decryptable only by client key and bound to session/step/key_id.
    token_ids, scores = decrypt_topk_artifact(
        artifact_bytes=bytes(resp.he_compute_payload),
        client_private_key=priv,
        expected_session_id="sess-he-node-1",
        expected_step=7,
        expected_key_id="kid-he-node-1",
    )
    assert len(token_ids) == 3
    assert len(scores) == 3


def test_mpc_nminus1_n_mode_returns_output_payload_without_plaintext_token():
    servicer = _build_light_servicer()
    context = _DummyContext()
    hidden = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    payload, payload_hash = build_output_mpc_request_payload(
        hidden_last_token=hidden,
        session_id="sess-mpc-output",
        step=1,
        key_id="kid-mpc-output",
    )
    req = inference_pb2.ForwardRequest(
        session_id="sess-mpc-output",
        activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
        tensor_shape=[1, 1, 4],
        he_output_enabled=True,
        he_key_id="kid-mpc-output",
        he_step=1,
        he_compute_mode=HE_COMPUTE_MODE_MPC_N_MINUS_1_N,
        he_top_k=1,
        he_disable_plaintext_sampling=True,
        he_compute_payload=payload,
        he_compute_format="mpc-output-request-v1",
        output_mpc_payload_hash=payload_hash,
    )
    resp = servicer.Forward(req, context)
    assert servicer.runner.sample_token_flags == [False]
    assert resp.has_token is False
    assert not resp.he_error
    assert resp.he_compute_payload
    token_id, is_eos = parse_output_mpc_response_payload(
        payload_bytes=bytes(resp.he_compute_payload),
        expected_session_id="sess-mpc-output",
        expected_step=1,
        expected_key_id="kid-mpc-output",
        expected_payload_hash=resp.output_mpc_payload_hash,
    )
    assert isinstance(token_id, int)
    assert isinstance(is_eos, bool)
