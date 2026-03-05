#!/usr/bin/env python3
"""
Local devnet validator for MPC/HE privacy and HE-sidecar dispute slashing.

What it validates:
1) Registry + verifier dispute loop works over gRPC.
2) Simulated HE sidecar cheating report escalates and gets slashed (enforced mode).
3) Output node plaintext visibility guard remains intact via existing tests.
4) MPC divide-and-conquer experiment runs with larger tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time

from concurrent import futures

import grpc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
import registry_pb2_grpc

from economics.cluster_config import ClusterConfig
from network.he_dispute import sign_report_payload
from network.registry_server import RegistryServicer
from network.share_auth import generate_signing_keypair
from network.verifier_node import run_verifier


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _start_registry_with_servicer(port: int) -> tuple[grpc.Server, RegistryServicer]:
    cluster_cfg = ClusterConfig(
        name="local-he-dispute-devnet",
        he_dispute_rollout_stage="enforced",
        he_dispute_slash_fraction=0.5,
        he_dispute_sampling_rate=0.05,
        he_dispute_report_rate_limit_per_window=64,
    )
    servicer = RegistryServicer(cluster_config=cluster_cfg, no_chain=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    registry_pb2_grpc.add_RegistryServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    return server, servicer


def _build_signed_suspicion_report(
    *,
    reporter_node_id: str,
    sidecar_node_id: str,
    sidecar_stake_identity: str,
    reason_code: str,
    session_id: str,
    step: int,
    key_id: str,
) -> registry_pb2.HESuspicionReport:
    private_key, public_key = generate_signing_keypair()
    request_payload_hash = hashlib.sha256(b"devnet-request").hexdigest()
    response_payload_hash = hashlib.sha256(b"devnet-cheat-response").hexdigest()
    report_id = f"devnet-rpt-{int(time.time() * 1000)}"
    timestamp = time.time()
    payload = {
        "report_id": report_id,
        "reporter_node_id": reporter_node_id,
        "reporter_node_type": "output",
        "sidecar_node_id": sidecar_node_id,
        "sidecar_stake_identity": sidecar_stake_identity,
        "session_id": session_id,
        "step": int(step),
        "key_id": key_id,
        "reason_code": reason_code,
        "he_compute_format": "paillier_v1",
        "request_payload_hash": request_payload_hash,
        "response_payload_hash": response_payload_hash,
        "timestamp": float(timestamp),
    }
    signature = sign_report_payload(private_key, payload)
    return registry_pb2.HESuspicionReport(
        report_id=report_id,
        idempotency_key=(
            f"{reporter_node_id}:{session_id}:{step}:{reason_code}:{response_payload_hash[:8]}"
        ),
        reporter_node_id=reporter_node_id,
        reporter_node_type="output",
        sidecar_node_id=sidecar_node_id,
        sidecar_stake_identity=sidecar_stake_identity,
        session_id=session_id,
        step=int(step),
        key_id=key_id,
        reason_code=reason_code,
        he_compute_format="paillier_v1",
        request_payload_hash=request_payload_hash,
        response_payload_hash=response_payload_hash,
        evidence_json=json.dumps({"scenario": "simulated-cheating", "source": "devnet-script"}),
        reporter_signature=signature,
        reporter_signing_public_key=public_key,
        timestamp=float(timestamp),
    )


def _run_pytest_subset(repo_root: str, tests: list[str]) -> None:
    command = [
        os.path.join(repo_root, ".venv", "bin", "python"),
        "-m",
        "pytest",
        *tests,
        "-q",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root
    result = subprocess.run(command, cwd=repo_root, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"pytest subset failed: {' '.join(tests)}")


def _run_mpc_experiment(repo_root: str) -> None:
    command = [
        os.path.join(repo_root, ".venv", "bin", "python"),
        os.path.join(repo_root, "playground", "mpc_recursive_partition_experiment.py"),
        "--depth",
        "2",
        "--batch",
        "1",
        "--seq-len",
        "256",
        "--hidden-size",
        "1024",
        "--parallel-workers",
        "4",
        "--trials",
        "1",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root
    result = subprocess.run(command, cwd=repo_root, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError("MPC experiment command failed")


def run_validation() -> None:
    repo_root = PROJECT_ROOT
    registry_port = _find_free_port()
    registry_addr = f"127.0.0.1:{registry_port}"

    print(f"[devnet] starting registry at {registry_addr}")
    server, servicer = _start_registry_with_servicer(registry_port)

    sidecar_stake_identity = "0x0000000000000000000000000000000000000abc"
    servicer._payment_contract.stakes.deposit(sidecar_stake_identity, 1000.0)
    if not servicer._payment_contract.stakes.stake(sidecar_stake_identity, 200.0):
        raise RuntimeError("failed to seed sidecar stake")
    before = servicer._payment_contract.stakes.get_all_accounts()[sidecar_stake_identity]["staked"]
    print(f"[devnet] seeded sidecar stake: {before}")

    print("[devnet] starting verifier")
    verifier_thread = threading.Thread(
        target=run_verifier,
        kwargs={
            "registry_address": registry_addr,
            "poll_interval": 1.0,
            "max_tickets_per_poll": 8,
            "verifier_id": "devnet-verifier-1",
        },
        daemon=True,
    )
    verifier_thread.start()
    time.sleep(2.0)

    channel = grpc.insecure_channel(registry_addr)
    stub = registry_pb2_grpc.RegistryStub(channel)

    print("[devnet] submitting simulated cheating suspicion report")
    report = _build_signed_suspicion_report(
        reporter_node_id="devnet-output-node",
        sidecar_node_id="devnet-he-sidecar",
        sidecar_stake_identity=sidecar_stake_identity,
        reason_code="response_schema_invalid",
        session_id="devnet-session-1",
        step=1,
        key_id="devnet-kid-1",
    )
    response = stub.SubmitHESuspicionReport(report, timeout=3.0)
    if not response.accepted:
        raise RuntimeError(f"suspicion report rejected: {response.status} {response.message}")
    ticket_id = response.dispute_ticket_id
    print(f"[devnet] dispute ticket opened: {ticket_id}")

    timeout_at = time.time() + 15.0
    while time.time() < timeout_at:
        dispute = servicer._he_disputes_by_ticket.get(ticket_id)
        if dispute and dispute.status != "open":
            break
        time.sleep(0.5)
    dispute = servicer._he_disputes_by_ticket.get(ticket_id)
    if dispute is None or dispute.status != "invalid":
        raise RuntimeError("verifier did not finalize HE dispute as invalid")

    after = servicer._payment_contract.stakes.get_all_accounts()[sidecar_stake_identity]["staked"]
    print(f"[devnet] sidecar stake after verdict: {after}")
    if not (after < before):
        raise RuntimeError("expected sidecar stake to decrease after invalid verdict")
    print("[devnet] slashing check passed")

    print("[devnet] running HE visibility regression tests")
    _run_pytest_subset(
        repo_root,
        [
            "tests/test_he_compute_node_visibility.py",
            "tests/test_he_full_vocab_sidecar_scaffold.py",
        ],
    )

    print("[devnet] running MPC larger-tensor experiment")
    _run_mpc_experiment(repo_root)

    channel.close()
    server.stop(grace=1)
    print("[devnet] validation complete: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local HE/MPC devnet validation")
    _ = parser.parse_args()
    run_validation()


if __name__ == "__main__":
    main()
