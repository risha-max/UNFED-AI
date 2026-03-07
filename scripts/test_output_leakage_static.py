#!/usr/bin/env python3
"""
Static leakage audit for MPC N-1/N output mode.

This audit is dependency-free and validates key non-leakage invariants directly
from source text. It complements runtime tests in environments without torch/grpc.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _check(name: str, condition: bool, failures: list[str]) -> None:
    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")
        failures.append(name)


def main() -> int:
    failures: list[str] = []

    node_server = _read("node/server.py")
    fwd_attest = _read("network/forward_attestation.py")
    racing = _read("network/racing.py")
    proto = _read("proto/inference.proto")

    _check(
        "deprecated server_sample mode rejected",
        "he_compute_mode=server_sample is retired. Use mpc_nminus1_n." in node_server,
        failures,
    )
    _check(
        "mpc output branch does not emit plaintext has_token",
        "he_compute_mode == HE_COMPUTE_MODE_MPC_N_MINUS_1_N" in node_server
        and "has_token=False" in node_server,
        failures,
    )
    _check(
        "forward attestation binds output payload hash",
        "output_mpc_payload_hash" in fwd_attest and "output_mpc_payload_hash" in node_server,
        failures,
    )
    _check(
        "racing hash includes output payload metadata",
        "output_mpc_payload_hash" in racing
        and "output_mpc_op" in racing
        and "output_mpc_payload_type" in racing,
        failures,
    )
    _check(
        "protobuf defines output payload metadata fields",
        "output_mpc_op = 43;" in proto
        and "output_mpc_payload_type = 44;" in proto
        and "output_mpc_payload_hash = 45;" in proto,
        failures,
    )

    if failures:
        print(f"\nLeakage static audit FAILED ({len(failures)} checks).")
        return 1
    print("\nLeakage static audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
