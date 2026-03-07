#!/usr/bin/env python3
"""
UNFED public testnet preflight checks.

This script performs a lightweight Go/No-Go validation before starting
public-facing services.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
from dataclasses import dataclass


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    return default


def is_loopback_host(host: str) -> bool:
    value = (host or "").strip().lower()
    if value in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def is_local_advertise(address: str) -> bool:
    raw = (address or "").strip()
    if not raw:
        return True
    host = raw
    if host.startswith("[") and "]" in host:
        host = host[1:host.index("]")]
    elif ":" in host:
        host = host.rsplit(":", 1)[0]
    host = host.strip().lower()
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_loopback or ip.is_private or ip.is_link_local)
    except ValueError:
        return False


def parse_bool_arg(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


@dataclass
class CheckResult:
    ok: bool
    message: str


def check_web(args: argparse.Namespace) -> list[CheckResult]:
    host = (args.host or "127.0.0.1").strip()
    is_public_bind = not is_loopback_host(host)

    dev_auth_bypass = env_bool("UNFED_DEV_AUTH_BYPASS", False)
    allow_demo_auth = env_bool("UNFED_ALLOW_DEMO_AUTH", False)
    faucet_enabled = env_bool("UNFED_FAUCET_ENABLED", False)
    faucet_require_auth = env_bool("UNFED_FAUCET_REQUIRE_AUTH", True)
    faucet_db = os.path.expanduser(
        os.environ.get("UNFED_FAUCET_STATE_DB", "~/.unfed/faucet_state.db")
    )

    out: list[CheckResult] = []
    out.append(CheckResult(True, f"web host={host}"))
    out.append(CheckResult(not dev_auth_bypass, "UNFED_DEV_AUTH_BYPASS must be disabled"))
    out.append(CheckResult(not allow_demo_auth, "UNFED_ALLOW_DEMO_AUTH must be disabled"))
    if faucet_enabled:
        out.append(
            CheckResult(
                faucet_require_auth,
                "faucet enabled requires UNFED_FAUCET_REQUIRE_AUTH=1",
            )
        )
        faucet_dir = os.path.dirname(faucet_db) or "."
        out.append(
            CheckResult(
                os.path.isdir(faucet_dir),
                f"faucet state directory exists: {faucet_dir}",
            )
        )
    else:
        out.append(CheckResult(True, "faucet disabled (recommended default)"))

    if is_public_bind:
        out.append(
            CheckResult(
                not dev_auth_bypass,
                "public bind requires dev auth bypass disabled",
            )
        )
    return out


def check_node(args: argparse.Namespace) -> list[CheckResult]:
    advertise = (args.advertise or "").strip()
    tls_cert = (args.tls_cert or "").strip()
    tls_key = (args.tls_key or "").strip()
    require_tls_for_public = parse_bool_arg(args.require_tls_for_public, True)
    allow_insecure_public = env_bool("UNFED_ALLOW_INSECURE_PUBLIC", False)
    public_advertise = not is_local_advertise(advertise)

    out: list[CheckResult] = []
    out.append(CheckResult(True, f"advertise={advertise or '(default/local)'}"))
    out.append(
        CheckResult(
            bool(tls_cert) == bool(tls_key),
            "tls cert/key must be provided together",
        )
    )
    if tls_cert:
        out.append(CheckResult(os.path.isfile(tls_cert), f"tls cert exists: {tls_cert}"))
    if tls_key:
        out.append(CheckResult(os.path.isfile(tls_key), f"tls key exists: {tls_key}"))

    if public_advertise and require_tls_for_public and not allow_insecure_public:
        out.append(
            CheckResult(
                bool(tls_cert and tls_key),
                "public advertise requires TLS (or UNFED_ALLOW_INSECURE_PUBLIC=1 override)",
            )
        )
    else:
        out.append(CheckResult(True, "public TLS policy check passed"))
    return out


def print_report(service: str, checks: list[CheckResult]) -> int:
    failed = [c for c in checks if not c.ok]
    print(f"[Preflight] service={service}")
    for item in checks:
        tag = "PASS" if item.ok else "FAIL"
        print(f"[{tag}] {item.message}")
    if failed:
        print(f"[NO-GO] {len(failed)} check(s) failed.")
        return 1
    print("[GO] all checks passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UNFED public testnet preflight checks")
    sub = p.add_subparsers(dest="service", required=True)

    p_web = sub.add_parser("web", help="Validate web dashboard security settings")
    p_web.add_argument("--host", default="127.0.0.1", help="Web bind host")

    p_node = sub.add_parser("node", help="Validate compute node public/TLS settings")
    p_node.add_argument("--advertise", default="", help="Node advertised address host:port")
    p_node.add_argument("--tls-cert", default="", help="TLS cert PEM path")
    p_node.add_argument("--tls-key", default="", help="TLS key PEM path")
    p_node.add_argument(
        "--require-tls-for-public",
        default=None,
        choices=["true", "false", "1", "0"],
        help="Require TLS for non-local advertised addresses",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.service == "web":
        checks = check_web(args)
    elif args.service == "node":
        checks = check_node(args)
    else:
        print("Unknown service", file=sys.stderr)
        return 2
    return print_report(args.service, checks)


if __name__ == "__main__":
    raise SystemExit(main())
