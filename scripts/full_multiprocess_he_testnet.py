#!/usr/bin/env python3
"""
Spin up a local multi-process HE testnet slice and validate dispute/slashing.

Topology:
- In-process registry (for deterministic stake seeding and direct inspection)
- In-process verifier loop
- Shard 2 node process (penultimate)
- Shard 3 node process (output)
- In-process mock HE sidecar HTTP server (cheating mode)

Validation:
1) Penultimate forwards to output in server_sample mode.
2) Output performs sidecar integrity checks and reports suspicion to registry.
3) Verifier adjudicates dispute ticket as invalid.
4) Registry slashes sidecar stake.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent import futures
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

import grpc
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import config
import inference_pb2
import inference_pb2_grpc
import registry_pb2
import registry_pb2_grpc
from economics.cluster_config import ClusterConfig
from network.he_compute import generate_client_compute_keypair
from network.registry_server import RegistryServicer
from network.verifier_node import run_verifier


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _CheatingSidecarHandler(BaseHTTPRequestHandler):
    sidecar_node_id = "devnet-he-sidecar"
    sidecar_stake_identity = "0x0000000000000000000000000000000000000abc"
    request_count = 0

    def do_POST(self):
        if self.path != "/v1/he/full-vocab":
            self.send_response(404)
            self.end_headers()
            return
        body_len = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(body_len)
        payload = json.loads(body.decode("utf-8")) if body else {}
        _CheatingSidecarHandler.request_count += 1

        # Intentionally malformed format to trigger output-node fail-closed + dispute.
        fake_payload = b'{"oops":"malformed"}'
        resp = {
            "session_id": payload.get("session_id", ""),
            "step": int(payload.get("step", 0)),
            "key_id": payload.get("key_id", ""),
            "he_compute_payload_b64": base64.b64encode(fake_payload).decode("ascii"),
            "he_compute_format": "unknown_format",
            "he_top_k": int(payload.get("top_k", 1)),
            "he_error": "",
            "sidecar_node_id": _CheatingSidecarHandler.sidecar_node_id,
            "sidecar_stake_identity": _CheatingSidecarHandler.sidecar_stake_identity,
            "compute_payload_hash_echo": payload.get("compute_payload_hash", ""),
        }
        wire = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(wire)))
        self.end_headers()
        self.wfile.write(wire)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep launcher output concise.
        return


def _start_registry(port: int) -> tuple[grpc.Server, RegistryServicer]:
    cluster_cfg = ClusterConfig(
        name="he-multiprocess-testnet",
        he_dispute_rollout_stage="enforced",
        he_dispute_slash_fraction=0.5,
        he_dispute_sampling_rate=0.05,
        he_dispute_report_rate_limit_per_window=64,
    )
    servicer = RegistryServicer(cluster_config=cluster_cfg, no_chain=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    registry_pb2_grpc.add_RegistryServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    return server, servicer


def _spawn_node(shard_index: int, port: int, registry_addr: str, sidecar_url: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT
    env["UNFED_REQUIRE_DAEMON"] = "0"
    env["UNFED_HE_COMPUTE_MODE"] = "server_sample"
    env["UNFED_HE_SIDECAR_URL"] = sidecar_url
    env["UNFED_HE_SIDECAR_REQUIRED"] = "1"
    env["UNFED_HE_SIDECAR_ALLOWED_FORMATS"] = "paillier_v1"
    env["UNFED_HE_TOP_K"] = "16"

    cmd = [
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        "-m",
        "node.server",
        "--shard-index",
        str(shard_index),
        "--port",
        str(port),
        "--registry",
        registry_addr,
        "--shards-dir",
        "shards",
    ]
    return subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_nodes(registry_addr: str, expected_ports: list[int], timeout_s: float = 90.0) -> None:
    channel = grpc.insecure_channel(registry_addr)
    stub = registry_pb2_grpc.RegistryStub(channel)
    deadline = time.time() + timeout_s
    expected_addrs = {f"localhost:{p}" for p in expected_ports}
    while time.time() < deadline:
        resp = stub.Discover(registry_pb2.DiscoverRequest(model_id=""))
        addrs = {n.address for n in resp.nodes}
        if expected_addrs.issubset(addrs):
            channel.close()
            return
        time.sleep(1.0)
    channel.close()
    raise RuntimeError("Timed out waiting for node registration")


def _read_proc_tail(proc: subprocess.Popen, limit: int = 80) -> str:
    if proc.stdout is None:
        return ""
    lines = proc.stdout.readlines()[-limit:]
    return "".join(lines)


def run() -> None:
    registry_port = _find_free_port()
    sidecar_port = _find_free_port()
    node2_port = _find_free_port()
    node3_port = _find_free_port()
    registry_addr = f"127.0.0.1:{registry_port}"
    sidecar_url = f"http://127.0.0.1:{sidecar_port}"

    print(f"[testnet] registry: {registry_addr}")
    print(f"[testnet] sidecar: {sidecar_url}")
    print(f"[testnet] node2: localhost:{node2_port} node3: localhost:{node3_port}")

    reg_server, reg_servicer = _start_registry(registry_port)
    # Seed sidecar stake for slashing visibility.
    target = _CheatingSidecarHandler.sidecar_stake_identity
    reg_servicer._payment_contract.stakes.deposit(target, 1000.0)
    reg_servicer._payment_contract.stakes.stake(target, 200.0)
    stake_before = reg_servicer._payment_contract.stakes.get_all_accounts()[target]["staked"]
    print(f"[testnet] sidecar stake before: {stake_before}")

    sidecar = _ThreadedHTTPServer(("127.0.0.1", sidecar_port), _CheatingSidecarHandler)
    sidecar_thread = threading.Thread(target=sidecar.serve_forever, daemon=True)
    sidecar_thread.start()

    verifier_thread = threading.Thread(
        target=run_verifier,
        kwargs={
            "registry_address": registry_addr,
            "poll_interval": 1.0,
            "max_tickets_per_poll": 16,
            "verifier_id": "mp-he-verifier-1",
        },
        daemon=True,
    )
    verifier_thread.start()

    node2 = _spawn_node(2, node2_port, registry_addr, sidecar_url)
    node3 = _spawn_node(3, node3_port, registry_addr, sidecar_url)

    try:
        _wait_for_nodes(registry_addr, [node2_port, node3_port], timeout_s=120.0)
        print("[testnet] nodes registered")

        # Send one HE server_sample request to shard 2.
        _, client_pub = generate_client_compute_keypair()
        hidden_size = int(json.load(open(os.path.join(PROJECT_ROOT, "shards", "manifest.json")))["architecture"]["text"]["hidden_size"])
        activation = torch.zeros(1, 1, hidden_size, dtype=torch.float32).numpy().tobytes()

        ch = grpc.insecure_channel(f"127.0.0.1:{node2_port}", options=config.GRPC_OPTIONS)
        stub = inference_pb2_grpc.InferenceNodeStub(ch)
        req = inference_pb2.ForwardRequest(
            session_id="mp-he-session-1",
            activation_data=activation,
            tensor_shape=[1, 1, hidden_size],
            remaining_circuit=[f"localhost:{node3_port}"],
            he_output_enabled=True,
            he_client_pubkey=client_pub,
            he_key_id="mp-he-kid-1",
            he_step=1,
            he_compute_mode="server_sample",
            he_top_k=8,
            he_disable_plaintext_sampling=True,
        )
        resp = stub.Forward(req, timeout=120.0)
        ch.close()
        if not resp.he_error:
            raise RuntimeError("Expected HE integrity failure from cheating sidecar, but got no he_error")
        print(f"[testnet] output returned fail-closed error: {resp.he_error}")

        # Wait for verifier adjudication.
        deadline = time.time() + 30.0
        ticket_id = ""
        while time.time() < deadline:
            with reg_servicer._he_dispute_lock:
                tickets = list(reg_servicer._he_disputes_by_ticket.values())
            if tickets:
                ticket = tickets[-1]
                ticket_id = ticket.dispute_ticket_id
                if ticket.status != "open":
                    break
            time.sleep(1.0)
        if not ticket_id:
            raise RuntimeError("No HE dispute ticket observed")

        with reg_servicer._he_dispute_lock:
            ticket = reg_servicer._he_disputes_by_ticket[ticket_id]
        if ticket.status != "invalid":
            raise RuntimeError(f"Expected invalid verdict, got {ticket.status}")
        print(f"[testnet] dispute ticket {ticket_id} -> {ticket.status}")

        stake_after = reg_servicer._payment_contract.stakes.get_all_accounts()[target]["staked"]
        if not (stake_after < stake_before):
            raise RuntimeError("Expected sidecar stake to be slashed")
        print(f"[testnet] sidecar stake after: {stake_after}")
        print(f"[testnet] sidecar requests seen: {_CheatingSidecarHandler.request_count}")
        print("[testnet] PASS")
    finally:
        for proc in (node2, node3):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        sidecar.shutdown()
        reg_server.stop(grace=1)

        # If node failed, expose output for debugging.
        for name, proc in (("node2", node2), ("node3", node3)):
            if proc.returncode not in (0, None, -15):
                print(f"[{name}] exit={proc.returncode}")
                print(_read_proc_tail(proc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full multi-process HE dispute testnet check")
    _ = parser.parse_args()
    run()


if __name__ == "__main__":
    main()
