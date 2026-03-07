import json
import os
import sys
import threading
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

from network.mpc_output import (
    HE_COMPUTE_MODE_MPC_N_MINUS_1_N,
    build_output_mpc_request_payload,
    build_output_mpc_response_payload,
    parse_output_mpc_response_payload,
)

try:
    import grpc
    import inference_pb2
    from node.server import InferenceNodeServicer
    _RUNTIME_AVAILABLE = True
except Exception:
    grpc = None
    inference_pb2 = None
    InferenceNodeServicer = None
    _RUNTIME_AVAILABLE = False


if _RUNTIME_AVAILABLE:
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
        servicer = InferenceNodeServicer.__new__(InferenceNodeServicer)
        servicer.shard_index = 999
        servicer.num_shards = 1000
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
        servicer._record_success_share = lambda **kwargs: None
        return servicer


class TestMPCOutputLeakage(unittest.TestCase):
    def test_output_mode_constant_is_expected(self):
        self.assertEqual(HE_COMPUTE_MODE_MPC_N_MINUS_1_N, "mpc_nminus1_n")

    def test_response_payload_does_not_include_hidden_values(self):
        payload_bytes, payload_hash = build_output_mpc_response_payload(
            token_id=7,
            is_eos=False,
            session_id="sess-contract-1",
            step=0,
            key_id="kid-contract-1",
        )
        decoded = json.loads(payload_bytes.decode("utf-8"))
        self.assertNotIn("hidden", decoded)
        token_id, is_eos = parse_output_mpc_response_payload(
            payload_bytes=payload_bytes,
            expected_session_id="sess-contract-1",
            expected_step=0,
            expected_key_id="kid-contract-1",
            expected_payload_hash=payload_hash,
        )
        self.assertEqual(token_id, 7)
        self.assertFalse(is_eos)

    @unittest.skipUnless(_RUNTIME_AVAILABLE, "grpc runtime unavailable in this environment")
    def test_mpc_response_has_no_plaintext_token_fields(self):
        servicer = _build_light_servicer()
        context = _DummyContext()
        hidden = torch.tensor([111.0, 222.0, 333.0, 444.0], dtype=torch.float32)
        payload, payload_hash = build_output_mpc_request_payload(
            hidden_last_token=hidden,
            session_id="sess-leak-1",
            step=0,
            key_id="kid-leak-1",
        )
        req = inference_pb2.ForwardRequest(
            session_id="sess-leak-1",
            activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
            tensor_shape=[1, 1, 4],
            he_output_enabled=True,
            he_key_id="kid-leak-1",
            he_step=0,
            he_compute_mode=HE_COMPUTE_MODE_MPC_N_MINUS_1_N,
            he_top_k=2,
            he_compute_payload=payload,
            he_compute_format="mpc-output-request-v1",
            output_mpc_payload_hash=payload_hash,
        )
        resp = servicer.Forward(req, context)

        self.assertEqual(servicer.runner.sample_token_flags, [False])
        self.assertFalse(resp.has_token)
        self.assertEqual(resp.token_id, 0)
        self.assertEqual(bytes(resp.activation_data), b"")
        self.assertTrue(bytes(resp.he_compute_payload))
        self.assertEqual(resp.he_compute_format, "mpc-output-response-v1")
        self.assertEqual(resp.output_mpc_payload_type, "token_sample")

    @unittest.skipUnless(_RUNTIME_AVAILABLE, "grpc runtime unavailable in this environment")
    def test_mpc_response_payload_does_not_echo_hidden_values(self):
        servicer = _build_light_servicer()
        context = _DummyContext()
        hidden = torch.tensor([101.25, 202.5, 303.75, 404.0], dtype=torch.float32)
        payload, payload_hash = build_output_mpc_request_payload(
            hidden_last_token=hidden,
            session_id="sess-leak-2",
            step=1,
            key_id="kid-leak-2",
        )
        req = inference_pb2.ForwardRequest(
            session_id="sess-leak-2",
            activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
            tensor_shape=[1, 1, 4],
            he_output_enabled=True,
            he_key_id="kid-leak-2",
            he_step=1,
            he_compute_mode=HE_COMPUTE_MODE_MPC_N_MINUS_1_N,
            he_compute_payload=payload,
            he_compute_format="mpc-output-request-v1",
            output_mpc_payload_hash=payload_hash,
        )
        resp = servicer.Forward(req, context)

        response_payload = json.loads(bytes(resp.he_compute_payload).decode("utf-8"))
        self.assertNotIn("hidden", response_payload)
        encoded = bytes(resp.he_compute_payload).decode("utf-8")
        for sentinel in ("101.25", "202.5", "303.75", "404.0"):
            self.assertNotIn(sentinel, encoded)

    @unittest.skipUnless(_RUNTIME_AVAILABLE, "grpc runtime unavailable in this environment")
    def test_server_sample_mode_is_rejected(self):
        servicer = _build_light_servicer()
        context = _DummyContext()
        req = inference_pb2.ForwardRequest(
            session_id="sess-leak-3",
            activation_data=torch.zeros(1, 1, 4, dtype=torch.float32).numpy().tobytes(),
            tensor_shape=[1, 1, 4],
            he_output_enabled=True,
            he_compute_mode="server_sample",
        )
        resp = servicer.Forward(req, context)
        self.assertEqual(context.code, grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("retired", context.details.lower())
        self.assertEqual(bytes(resp.he_compute_payload), b"")


if __name__ == "__main__":
    unittest.main()
