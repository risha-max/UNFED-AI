"""E2E tests for environment-driven configuration."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import grpc
import pytest

from tests.e2e.conftest import (
    PROJECT_ROOT, PYTHON, SHARDS_DIR, _wait_for_port,
)

CONFIG_REGISTRY_PORT = 51350
CONFIG_NODE_PORT = 51351


@pytest.mark.e2e
class TestConfig:

    def test_env_registry_override(self):
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "UNFED_REGISTRY": f"localhost:{CONFIG_REGISTRY_PORT}",
        }
        reg = subprocess.Popen(
            [PYTHON, "-m", "network.registry_server",
             "--port", str(CONFIG_REGISTRY_PORT)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            assert _wait_for_port(CONFIG_REGISTRY_PORT, timeout=10)
            channel = grpc.insecure_channel(f"localhost:{CONFIG_REGISTRY_PORT}")
            grpc.channel_ready_future(channel).result(timeout=5)
            channel.close()
        finally:
            reg.send_signal(signal.SIGTERM)
            try:
                reg.wait(timeout=10)
            except subprocess.TimeoutExpired:
                reg.kill()
                reg.wait()

    def test_env_seeds_override(self):
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "UNFED_SEEDS": f"localhost:{CONFIG_REGISTRY_PORT},localhost:99999",
        }
        reg = subprocess.Popen(
            [PYTHON, "-m", "network.registry_server",
             "--port", str(CONFIG_REGISTRY_PORT)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            assert _wait_for_port(CONFIG_REGISTRY_PORT, timeout=10)

            result = subprocess.run(
                [PYTHON, "-c",
                 "import os; os.environ['UNFED_SEEDS'] = "
                 f"'localhost:{CONFIG_REGISTRY_PORT},localhost:99999'; "
                 "import config; print(','.join(config.SEED_REGISTRIES))"],
                cwd=PROJECT_ROOT, env=env,
                capture_output=True, text=True, timeout=10,
            )
            seeds = result.stdout.strip().split(",")
            assert f"localhost:{CONFIG_REGISTRY_PORT}" in seeds
        finally:
            reg.send_signal(signal.SIGTERM)
            try:
                reg.wait(timeout=10)
            except subprocess.TimeoutExpired:
                reg.kill()
                reg.wait()

    def test_node_config_json(self):
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        reg = subprocess.Popen(
            [PYTHON, "-m", "network.registry_server",
             "--port", str(CONFIG_REGISTRY_PORT)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            assert _wait_for_port(CONFIG_REGISTRY_PORT, timeout=10)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump({
                    "role": "compute",
                    "port": CONFIG_NODE_PORT,
                    "shard_index": 0,
                    "shards_dir": SHARDS_DIR,
                    "registry": f"localhost:{CONFIG_REGISTRY_PORT}",
                }, f)
                config_path = f.name

            try:
                node = subprocess.Popen(
                    [PYTHON, "-m", "node.run", "--config", config_path],
                    cwd=PROJECT_ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                try:
                    assert _wait_for_port(CONFIG_NODE_PORT, timeout=45), \
                        "Node via config didn't start"
                finally:
                    node.send_signal(signal.SIGTERM)
                    try:
                        node.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        node.kill()
                        node.wait()
            finally:
                os.unlink(config_path)
        finally:
            reg.send_signal(signal.SIGTERM)
            try:
                reg.wait(timeout=10)
            except subprocess.TimeoutExpired:
                reg.kill()
                reg.wait()

    def test_missing_operator_key_fails(self):
        script = os.path.join(PROJECT_ROOT, "scripts", "start_local_chain.sh")
        if not os.path.exists(script):
            pytest.skip("start_local_chain.sh not found")

        env = {k: v for k, v in os.environ.items()
               if k not in ("OPERATOR_PRIVATE_KEY", "OPERATOR_ADDRESS")}
        env["PATH"] = os.environ.get("PATH", "")

        result = subprocess.run(
            ["bash", script],
            env=env, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0, \
            "Script should fail without OPERATOR_PRIVATE_KEY"
