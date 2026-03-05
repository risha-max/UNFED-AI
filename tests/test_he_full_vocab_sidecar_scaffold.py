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
from network.he_full_vocab_sidecar import SidecarUnavailableError
from node.server import InferenceNodeServicer


class _DummyTicketCollector:
    def should_sample(self):
        return False

    def collect(self, **kwargs):
        return None


class _DummyRunner:
    def __init__(self):
        self.sample_token_flags = []
        self.lm_head = lambda h: torch.tensor([[[0.0, 1.5, 0.9, -0.2]]], dtype=torch.float32)

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
    def set_details(self, details):
        self.details = details

    def set_code(self, code):
        self.code = code


def _build_light_servicer() -> InferenceNodeServicer:
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
    servicer._report_he_suspicion = lambda **kwargs: None
    return servicer


def test_server_sample_mode_falls_back_to_client_sample_artifact():
    orig = node_server_mod.request_full_vocab_he_artifact

    def _fail_sidecar(**kwargs):
        raise SidecarUnavailableError("sidecar down")

    node_server_mod.request_full_vocab_he_artifact = _fail_sidecar
    try:
        priv, pub = generate_client_compute_keypair()
        servicer = _build_light_servicer()
        servicer._he_full_vocab_sidecar_required = False
        req = inference_pb2.ForwardRequest(
            session_id="sess-sidecar-fallback",
            activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
            tensor_shape=[1, 1, 4],
            he_output_enabled=True,
            he_client_pubkey=pub,
            he_key_id="kid-sidecar-fallback",
            he_step=3,
            he_compute_mode="server_sample",
            he_disable_plaintext_sampling=True,
            he_top_k=3,
        )
        resp = servicer.Forward(req, _DummyContext())
        assert resp.has_token is False
        assert resp.he_compute_payload
        token_ids, scores = decrypt_topk_artifact(
            artifact_bytes=bytes(resp.he_compute_payload),
            client_private_key=priv,
            expected_session_id="sess-sidecar-fallback",
            expected_step=3,
            expected_key_id="kid-sidecar-fallback",
        )
        assert len(token_ids) == 3
        assert len(scores) == 3
    finally:
        node_server_mod.request_full_vocab_he_artifact = orig


def test_server_sample_mode_strict_fails_when_sidecar_required():
    orig = node_server_mod.request_full_vocab_he_artifact

    def _fail_sidecar(**kwargs):
        raise SidecarUnavailableError("sidecar required and down")

    node_server_mod.request_full_vocab_he_artifact = _fail_sidecar
    try:
        _, pub = generate_client_compute_keypair()
        servicer = _build_light_servicer()
        servicer._he_full_vocab_sidecar_required = True
        req = inference_pb2.ForwardRequest(
            session_id="sess-sidecar-strict",
            activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
            tensor_shape=[1, 1, 4],
            he_output_enabled=True,
            he_client_pubkey=pub,
            he_key_id="kid-sidecar-strict",
            he_step=1,
            he_compute_mode="server_sample",
            he_disable_plaintext_sampling=True,
            he_top_k=2,
        )
        resp = servicer.Forward(req, _DummyContext())
        assert resp.has_token is False
        assert "sidecar required and down" in resp.he_error
    finally:
        node_server_mod.request_full_vocab_he_artifact = orig


def test_server_sample_mode_reports_inline_integrity_violation():
    orig = node_server_mod.request_full_vocab_he_artifact

    def _bad_sidecar(**kwargs):
        return {
            "he_compute_payload": b"abc",
            "he_compute_format": "unknown_format",
            "he_top_k": 2,
            "he_error": "",
            "session_id": kwargs["session_id"],
            "step": kwargs["step"],
            "key_id": kwargs["key_id"],
            "sidecar_node_id": "sidecar-1",
            "sidecar_stake_identity": "0x0000000000000000000000000000000000000def",
        }

    node_server_mod.request_full_vocab_he_artifact = _bad_sidecar
    reports = []
    try:
        _, pub = generate_client_compute_keypair()
        servicer = _build_light_servicer()
        servicer._he_sidecar_allowed_formats = {"paillier_v1"}
        servicer._report_he_suspicion = lambda **kwargs: reports.append(kwargs)
        req = inference_pb2.ForwardRequest(
            session_id="sess-sidecar-inline-check",
            activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
            tensor_shape=[1, 1, 4],
            he_output_enabled=True,
            he_client_pubkey=pub,
            he_key_id="kid-sidecar-inline-check",
            he_step=7,
            he_compute_mode="server_sample",
            he_disable_plaintext_sampling=True,
            he_top_k=2,
        )
        resp = servicer.Forward(req, _DummyContext())
        assert resp.has_token is False
        assert "integrity check failed" in resp.he_error
        assert len(reports) == 1
        assert reports[0]["reason_code"] == "response_format_not_allowed"
    finally:
        node_server_mod.request_full_vocab_he_artifact = orig
