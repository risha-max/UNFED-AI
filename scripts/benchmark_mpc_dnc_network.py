#!/usr/bin/env python3
"""
A/B benchmark for MPC divide-and-conquer in a local network stack.

Runs two scenarios:
  1) DNC disabled
  2) DNC enabled

Each scenario boots a local stack:
  - registry
  - MPC shard-0 role B
  - MPC shard-0 role A
  - compute shards 1,2,3

Then executes N prompts through the normal client path and reports latency.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import grpc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

from client.client import UnfedClient
import registry_pb2
import registry_pb2_grpc


PROMPTS = [
    "The capital of France is",
    "Explain transformers in one sentence:",
    "What is MPC in distributed inference?",
    "Summarize homomorphic encryption briefly.",
    "Why does divide and conquer help parallel systems?",
    "List two privacy risks in distributed AI.",
    "Define verifier slashing in one line.",
    "What is the benefit of fail-closed behavior?",
    "Give one practical use for secure matmul.",
    "How can sidecar cheating be detected?",
    "Why do we use registry-routed disputes?",
    "Compare MPC and HE in one paragraph.",
    "What is a Beaver triple?",
    "Why is top-k sampling useful?",
    "How would you benchmark latency?",
]


@dataclass
class ScenarioResult:
    name: str
    latencies_sec: list[float]
    failures: int

    @property
    def avg(self) -> float:
        return sum(self.latencies_sec) / max(1, len(self.latencies_sec))

    @property
    def p95(self) -> float:
        if not self.latencies_sec:
            return 0.0
        xs = sorted(self.latencies_sec)
        idx = max(0, min(len(xs) - 1, int(0.95 * (len(xs) - 1))))
        return xs[idx]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _wait_for_port(port: int, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _spawn(cmd: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _terminate_all(procs: Iterable[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except OSError:
                pass
    deadline = time.time() + 10
    for p in procs:
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                pass


def _wait_for_registry_nodes(registry_addr: str, expected: int, timeout_s: float = 180.0) -> bool:
    channel = grpc.insecure_channel(registry_addr)
    stub = registry_pb2_grpc.RegistryStub(channel)
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            try:
                resp = stub.Discover(registry_pb2.DiscoverRequest(model_id=""))
                if len(resp.nodes) >= expected:
                    return True
            except grpc.RpcError:
                pass
            time.sleep(1.0)
        return False
    finally:
        channel.close()


def _run_scenario(name: str, enable_dnc: bool, queries: int, max_new_tokens: int) -> ScenarioResult:
    reg_port = _free_port()
    mpc_a_port = _free_port()
    mpc_b_port = _free_port()
    n1_port = _free_port()
    n2_port = _free_port()
    n3_port = _free_port()
    registry_addr = f"127.0.0.1:{reg_port}"
    py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = PROJECT_ROOT
    base_env["UNFED_REQUIRE_DAEMON"] = "0"
    base_env["UNFED_REQUIRE_MPC"] = "1"
    base_env["UNFED_REQUIRE_VERIFIER"] = "0"
    procs: list[subprocess.Popen] = []
    try:
        procs.append(_spawn([py, "-m", "network.registry_server", "--port", str(reg_port)], base_env))
        if not _wait_for_port(reg_port, 40):
            raise RuntimeError("registry failed to start")

        procs.append(_spawn([
            py, "-m", "network.mpc_shard0",
            "--role", "B",
            "--port", str(mpc_b_port),
            "--peer", f"localhost:{mpc_a_port}",
            "--shards-dir", "shards",
            "--mpc-dnc-mode", "manual" if enable_dnc else "off",
            "--mpc-dnc-depth", "2",
            "--mpc-dnc-matmul-split-dim", "-3",
            "--mpc-dnc-split-dim", "-1",
            "--mpc-dnc-parallel-workers", "1",
        ], base_env))
        if not _wait_for_port(mpc_b_port, 120):
            raise RuntimeError("mpc role B failed to start")

        procs.append(_spawn([
            py, "-m", "network.mpc_shard0",
            "--role", "A",
            "--port", str(mpc_a_port),
            "--peer", f"localhost:{mpc_b_port}",
            "--registry", registry_addr,
            "--shards-dir", "shards",
            "--mpc-dnc-mode", "manual" if enable_dnc else "off",
            "--mpc-dnc-depth", "2",
            "--mpc-dnc-matmul-split-dim", "-3",
            "--mpc-dnc-split-dim", "-1",
            "--mpc-dnc-parallel-workers", "1",
        ], base_env))
        if not _wait_for_port(mpc_a_port, 120):
            raise RuntimeError("mpc role A failed to start")

        for idx, port in ((1, n1_port), (2, n2_port), (3, n3_port)):
            procs.append(_spawn([
                py, "-m", "node.server",
                "--shard-index", str(idx),
                "--port", str(port),
                "--registry", registry_addr,
                "--shards-dir", "shards",
            ], base_env))
            if not _wait_for_port(port, 120):
                raise RuntimeError(f"node shard {idx} failed to start")

        if not _wait_for_registry_nodes(registry_addr, expected=4, timeout_s=180):
            raise RuntimeError("registry did not see expected nodes")

        os.environ["UNFED_REQUIRE_VERIFIER"] = "0"
        os.environ["UNFED_REQUIRE_DAEMON"] = "0"
        client = UnfedClient(registry_address=registry_addr, model_id="./models/Qwen2.5-Coder-0.5B-Instruct")
        latencies: list[float] = []
        failures = 0
        first_error: str | None = None
        try:
            for i, prompt in enumerate(PROMPTS[:queries]):
                t0 = time.perf_counter()
                try:
                    _ = "".join(list(client.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        verbose=False,
                        use_onion=False,
                        use_random_routing=False,
                    )))
                    latencies.append(time.perf_counter() - t0)
                except Exception as e:
                    failures += 1
                    if first_error is None:
                        first_error = f"{type(e).__name__}: {e}"
                print(f"[{name}] query {i + 1}/{queries} done")
        finally:
            client.close()

        if first_error:
            print(f"[{name}] first error: {first_error}")

        return ScenarioResult(name=name, latencies_sec=latencies, failures=failures)
    finally:
        _terminate_all(procs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark network MPC DNC ON vs OFF")
    parser.add_argument("--queries", type=int, default=10, help="Number of prompts to run (<=15)")
    parser.add_argument("--max-new-tokens", type=int, default=6, help="Tokens to generate per prompt")
    args = parser.parse_args()

    queries = max(1, min(15, int(args.queries)))
    print(f"Running A/B benchmark with {queries} queries, max_new_tokens={args.max_new_tokens}")

    off = _run_scenario("dnc_off", enable_dnc=False, queries=queries, max_new_tokens=args.max_new_tokens)
    on = _run_scenario("dnc_on", enable_dnc=True, queries=queries, max_new_tokens=args.max_new_tokens)

    print("\n=== Results ===")
    print(f"dnc_off: success={len(off.latencies_sec)} fail={off.failures} avg={off.avg:.3f}s p95={off.p95:.3f}s")
    print(f"dnc_on : success={len(on.latencies_sec)} fail={on.failures} avg={on.avg:.3f}s p95={on.p95:.3f}s")
    if off.avg > 0 and on.avg > 0:
        speedup = off.avg / on.avg
        delta = ((on.avg - off.avg) / off.avg) * 100.0
        print(f"speedup (off/on): {speedup:.3f}x")
        print(f"avg latency delta: {delta:+.2f}%")


if __name__ == "__main__":
    main()
