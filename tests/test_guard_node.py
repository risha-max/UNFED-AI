"""
Unit tests for the guard node: opaque relay verification.

Ensures the guard node forwards payloads as opaque bytes without
deserialization or inspection.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestGuardOpacity:
    """
    Tests that the guard relay treats payloads as opaque bytes.

    These tests verify the guard's relay logic at the unit level,
    without starting actual gRPC servers.
    """

    def test_relay_preserves_bytes(self):
        """Encrypted payload bytes in == forwarded bytes out.

        The guard should never deserialize the encrypted_payload.
        We test this by verifying the guard code path uses raw byte
        forwarding (channel.unary_unary with request_serializer=None).
        """
        # Read the guard source to verify opaque byte forwarding
        guard_path = os.path.join(PROJECT_ROOT, "network", "guard_node.py")
        with open(guard_path) as f:
            source = f.read()

        # Verify the guard uses raw byte forwarding, not proto deserialization
        assert "request_serializer=None" in source, \
            "Guard should use raw byte forwarding (request_serializer=None)"
        assert "response_deserializer=None" in source, \
            "Guard should use raw byte response (response_deserializer=None)"

    def test_guard_does_not_import_forward_request(self):
        """Guard should not deserialize ForwardRequest from the payload.

        It receives the encrypted_payload and forwards it directly.
        The guard code may import proto types for the Relay RPC itself,
        but must not parse the encrypted content.
        """
        guard_path = os.path.join(PROJECT_ROOT, "network", "guard_node.py")
        with open(guard_path) as f:
            source = f.read()

        # The guard should NOT have ForwardRequest.FromString or ParseFromString
        # on the encrypted_payload
        assert "ParseFromString" not in source or \
               "encrypted_payload" not in source.split("ParseFromString")[0][-200:], \
            "Guard should not parse encrypted_payload"

    def test_guard_hash_is_prefix_only(self):
        """Guard only hashes a prefix of the payload for logging, not full content."""
        guard_path = os.path.join(PROJECT_ROOT, "network", "guard_node.py")
        with open(guard_path) as f:
            source = f.read()

        # The guard hashes only [:256] prefix for logging
        assert "[:256]" in source, \
            "Guard should only hash a prefix of the payload for logging"

    def test_relay_request_fields(self):
        """RelayRequest proto has the expected fields for opaque relay."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))
        import inference_pb2

        req = inference_pb2.RelayRequest()
        # These are the fields the guard uses
        assert hasattr(req, 'encrypted_payload')
        assert hasattr(req, 'target_address')
        assert hasattr(req, 'guard_ephemeral_key')

    def test_relay_response_fields(self):
        """RelayResponse proto has the expected fields."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))
        import inference_pb2

        resp = inference_pb2.RelayResponse()
        assert hasattr(resp, 'encrypted_payload')
