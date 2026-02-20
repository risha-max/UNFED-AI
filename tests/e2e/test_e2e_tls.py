"""E2E tests for optional TLS on gRPC."""

import os
import signal
import subprocess
import sys
import tempfile
import time

import grpc
import pytest

from tests.e2e.conftest import PROJECT_ROOT, PYTHON, _wait_for_port

TLS_REGISTRY_PORT = 51250


def _generate_self_signed_cert(tmpdir: str):
    """Generate a self-signed cert + key for testing."""
    cert_path = os.path.join(tmpdir, "server.crt")
    key_path = os.path.join(tmpdir, "server.key")

    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", key_path, "-out", cert_path,
        "-days", "1", "-nodes",
        "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ], check=True, capture_output=True)

    return cert_path, key_path


@pytest.mark.e2e
class TestTLS:

    def test_tls_server_starts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cert, key = _generate_self_signed_cert(tmpdir)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [PYTHON, "-m", "network.registry_server",
                 "--port", str(TLS_REGISTRY_PORT),
                 "--tls-cert", cert, "--tls-key", key],
                cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                assert _wait_for_port(TLS_REGISTRY_PORT, timeout=10), \
                    "TLS registry didn't start"
            finally:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def test_tls_client_connects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cert, key = _generate_self_signed_cert(tmpdir)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [PYTHON, "-m", "network.registry_server",
                 "--port", str(TLS_REGISTRY_PORT),
                 "--tls-cert", cert, "--tls-key", key],
                cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                assert _wait_for_port(TLS_REGISTRY_PORT, timeout=10)

                with open(cert, "rb") as f:
                    root_cert = f.read()
                creds = grpc.ssl_channel_credentials(root_certificates=root_cert)
                channel = grpc.secure_channel(
                    f"localhost:{TLS_REGISTRY_PORT}", creds)
                grpc.channel_ready_future(channel).result(timeout=5)
                channel.close()
            finally:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def test_insecure_client_rejected_by_tls_server(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cert, key = _generate_self_signed_cert(tmpdir)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [PYTHON, "-m", "network.registry_server",
                 "--port", str(TLS_REGISTRY_PORT),
                 "--tls-cert", cert, "--tls-key", key],
                cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                assert _wait_for_port(TLS_REGISTRY_PORT, timeout=10)

                channel = grpc.insecure_channel(
                    f"localhost:{TLS_REGISTRY_PORT}")
                with pytest.raises(grpc.FutureTimeoutError):
                    grpc.channel_ready_future(channel).result(timeout=3)
                channel.close()
            finally:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def test_insecure_still_works(self):
        port = TLS_REGISTRY_PORT + 1
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [PYTHON, "-m", "network.registry_server",
             "--port", str(port)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            assert _wait_for_port(port, timeout=10)
            channel = grpc.insecure_channel(f"localhost:{port}")
            grpc.channel_ready_future(channel).result(timeout=5)
            channel.close()
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
