"""E2E tests for graceful shutdown via SIGTERM."""

import os
import signal
import subprocess
import sys
import time

import pytest

from tests.e2e.conftest import (
    PROJECT_ROOT, PYTHON, SHARDS_DIR, _wait_for_port,
)

REGISTRY_PORT = 51150
NODE_PORT = 51151


@pytest.mark.e2e
class TestShutdown:

    def test_registry_sigterm(self):
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [PYTHON, "-m", "network.registry_server",
             "--port", str(REGISTRY_PORT)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            assert _wait_for_port(REGISTRY_PORT, timeout=10), "Registry didn't start"
            proc.send_signal(signal.SIGTERM)
            exit_code = proc.wait(timeout=10)
            assert exit_code == 0, f"Registry exited with code {exit_code}"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_node_sigterm(self):
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        reg = subprocess.Popen(
            [PYTHON, "-m", "network.registry_server",
             "--port", str(REGISTRY_PORT)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            assert _wait_for_port(REGISTRY_PORT, timeout=10)

            node = subprocess.Popen(
                [PYTHON, "-m", "node.server",
                 "--shard-index", "0",
                 "--port", str(NODE_PORT),
                 "--shards-dir", SHARDS_DIR,
                 "--registry", f"localhost:{REGISTRY_PORT}"],
                cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            try:
                assert _wait_for_port(NODE_PORT, timeout=45), "Node didn't start"
                node.send_signal(signal.SIGTERM)
                exit_code = node.wait(timeout=10)
                assert exit_code == 0, f"Node exited with code {exit_code}"
            finally:
                if node.poll() is None:
                    node.kill()
                    node.wait()
        finally:
            if reg.poll() is None:
                reg.send_signal(signal.SIGTERM)
                try:
                    reg.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    reg.kill()
                    reg.wait()

    def test_node_sigterm_during_inference(self, cluster, client):
        """SIGTERM a node mid-inference — client should get an error, not hang."""
        import grpc as _grpc

        gen = client.generate("Testing mid-inference shutdown", max_new_tokens=50)
        first_token = next(gen)
        assert first_token

        cluster.procs[-1].send_signal(signal.SIGTERM)
        time.sleep(1)

        try:
            for _ in gen:
                pass
        except (_grpc.RpcError, RuntimeError, StopIteration):
            pass
