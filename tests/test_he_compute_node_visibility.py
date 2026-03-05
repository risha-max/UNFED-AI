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


def test_server_sample_passthrough_skips_output_runner_forward():
    servicer = _build_light_servicer()
    context = _DummyContext()
    called = {"value": False}
    original = node_server_mod.request_full_vocab_he_artifact

    def _fake_sidecar(**kwargs):
        called["value"] = True
        return {
            "he_compute_payload": b'{"format":"paillier-topk-v1","session_id":"sess-he-sidecar-pass","step":1,"key_id":"kid-sidecar-pass","token_ids":[1],"scores":[{"ciphertext":"1","exponent":0}]}',
            "he_compute_format": "paillier_v1",
            "he_top_k": 1,
            "he_error": "",
            "session_id": "sess-he-sidecar-pass",
            "step": 1,
            "key_id": "kid-sidecar-pass",
            "sidecar_node_id": "sidecar-x",
            "sidecar_stake_identity": "0x0000000000000000000000000000000000000001",
        }

    node_server_mod.request_full_vocab_he_artifact = _fake_sidecar
    try:
        req = inference_pb2.ForwardRequest(
            session_id="sess-he-sidecar-pass",
            he_output_enabled=True,
            he_key_id="kid-sidecar-pass",
            he_step=1,
            he_compute_mode="server_sample",
            he_compute_payload=b"opaque-encrypted-hidden",
            he_compute_format="paillier-hidden-v1",
            he_top_k=1,
            he_disable_plaintext_sampling=True,
        )
        resp = servicer.Forward(req, context)
        assert called["value"] is True
        # Critical visibility check: output-node model forward was not run.
        assert servicer.runner.sample_token_flags == []
        assert resp.has_token is False
        assert resp.he_compute_payload
    finally:
        node_server_mod.request_full_vocab_he_artifact = original
