"""
Shared fixtures for e2e tests.

Starts a full cluster (registry + 4 compute nodes) as subprocesses,
waits for readiness, and tears down with SIGTERM.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

PYTHON = os.environ.get("UNFED_E2E_PYTHON", sys.executable)
SHARDS_DIR = os.environ.get("UNFED_E2E_SHARDS_DIR", os.path.join(PROJECT_ROOT, "shards"))
REGISTRY_PORT = int(os.environ.get("UNFED_E2E_REGISTRY_PORT", "51050"))
BASE_NODE_PORT = int(os.environ.get("UNFED_E2E_BASE_NODE_PORT", "51051"))
NUM_SHARDS = int(os.environ.get("UNFED_E2E_NUM_SHARDS", "4"))
STARTUP_TIMEOUT = int(os.environ.get("UNFED_E2E_STARTUP_TIMEOUT", "60"))


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(port: int, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False


class Cluster:
    """Manages a local UNFED cluster for testing."""

    def __init__(self):
        self.procs: list[subprocess.Popen] = []
        self.registry_addr = f"localhost:{REGISTRY_PORT}"
        self.node_addrs = [
            f"localhost:{BASE_NODE_PORT + i}" for i in range(NUM_SHARDS)
        ]

    def start(self):
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        self.procs.append(subprocess.Popen(
            [PYTHON, "-m", "network.registry_server", "--port", str(REGISTRY_PORT)],
            cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ))

        if not _wait_for_port(REGISTRY_PORT, timeout=15):
            raise RuntimeError("Registry did not start in time")

        for i in range(NUM_SHARDS):
            port = BASE_NODE_PORT + i
            self.procs.append(subprocess.Popen(
                [PYTHON, "-m", "node.server",
                 "--shard-index", str(i),
                 "--port", str(port),
                 "--shards-dir", SHARDS_DIR,
                 "--registry", self.registry_addr],
                cwd=PROJECT_ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            ))

        for i in range(NUM_SHARDS):
            port = BASE_NODE_PORT + i
            if not _wait_for_port(port, timeout=STARTUP_TIMEOUT):
                raise RuntimeError(f"Node shard {i} (port {port}) did not start")

        # Give nodes time to register with registry
        time.sleep(3)

    def stop(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.procs.clear()


@pytest.fixture(scope="module")
def cluster():
    """Start a full cluster, yield it, then tear down."""
    c = Cluster()
    c.start()
    yield c
    c.stop()


@pytest.fixture(scope="module")
def client(cluster):
    """Return an UnfedClient connected to the test cluster."""
    from client.client import UnfedClient
    c = UnfedClient(registry_address=cluster.registry_addr)
    yield c
    c.close()
