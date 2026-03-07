"""
Unified Node Entry Point — start any UNFED node type from a single command.

Each role has its own config class (ComputeConfig, MPCConfig, etc.)
with only the fields that role needs. The 'role' field in the JSON or CLI
determines which config type is created.

Usage:
    # From config file (role is read from the JSON)
    python -m node.run --config compute_config.json

    # CLI overrides
    python -m node.run --config compute_config.json --port 50052

    # Pure CLI (no config file)
    python -m node.run --role compute --shard-index 0 --port 50051

    # Verifier
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node.node_config import (
    ComputeConfig, MPCConfig, DaemonConfig,
    load_config, print_config_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all config fields."""
    parser = argparse.ArgumentParser(
        description="UNFED AI Node — unified entry point for compute, verifier, and daemon nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m node.run --config node_config.json
  python -m node.run --role compute --shard-index 0 --port 50051
        """,
    )

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to node_config.json")

    # --- Core ---
    parser.add_argument("--role", type=str, default=None,
                        choices=["compute", "mpc", "daemon"],
                        help="Node role: compute, mpc, or daemon")
    parser.add_argument("--port", type=int, default=None,
                        help="gRPC listen port")
    parser.add_argument("--host", type=str, default=None,
                        help="Bind address (default: [::])")
    parser.add_argument("--advertise", type=str, default=None,
                        help="Address to advertise to registry")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry server address")
    parser.add_argument("--require-tls-for-public", type=str, default=None,
                        choices=["true", "false", "1", "0"],
                        help="Require TLS when advertising non-local addresses")

    # --- gRPC ---
    parser.add_argument("--grpc-max-workers", type=int, default=None,
                        help="gRPC thread pool size")
    parser.add_argument("--grpc-max-message-mb", type=int, default=None,
                        help="Max gRPC message size in MB")
    parser.add_argument("--heartbeat-interval", type=int, default=None,
                        dest="heartbeat_interval_seconds",
                        help="Heartbeat interval in seconds")

    # --- Compute-only ---
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Shard index to load (compute only)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="HuggingFace model identifier")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["qwen2", "qwen2_vl", "smolvlm"],
                        help="Model architecture type")
    parser.add_argument("--shards-dir", type=str, default=None,
                        help="Directory containing shard .pt files")
    parser.add_argument("--manifest-path", type=str, default=None,
                        help="Path to shard manifest JSON")
    parser.add_argument("--max-memory-gb", type=float, default=None,
                        help="Memory budget hint in GB")
    parser.add_argument("--session-timeout", type=int, default=None,
                        dest="session_timeout_seconds",
                        help="KV cache cleanup timeout in seconds")
    parser.add_argument("--eos-token-id", type=int, default=None,
                        help="EOS token ID (model-specific)")
    parser.add_argument("--verification-sampling-rate", type=float, default=None,
                        help="Fraction of forward passes to sample (0.0-1.0)")
    parser.add_argument("--ticket-submit-interval", type=int, default=None,
                        dest="ticket_submit_interval_seconds",
                        help="Ticket submission interval in seconds")

    # --- Network resilience ---
    parser.add_argument("--connect-timeout", type=float, default=None,
                        help="gRPC connection timeout in seconds")
    parser.add_argument("--forward-timeout", type=float, default=None,
                        help="Forward RPC timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Max gRPC retry attempts")
    parser.add_argument("--retry-backoff-base", type=float, default=None,
                        help="Retry backoff base in seconds")
    parser.add_argument("--retry-backoff-max", type=float, default=None,
                        help="Max retry backoff in seconds")
    parser.add_argument("--keepalive-time-ms", type=int, default=None,
                        help="gRPC keepalive ping interval in ms")
    parser.add_argument("--keepalive-timeout-ms", type=int, default=None,
                        help="gRPC keepalive timeout in ms")
    parser.add_argument("--shard-transfer-timeout", type=float, default=None,
                        help="Timeout for P2P shard download in seconds")
    parser.add_argument("--shard-chunk-size", type=int, default=None,
                        help="Shard transfer chunk size in bytes")
    parser.add_argument("--require-daemon", type=str, default=None,
                        choices=["true", "false", "1", "0"],
                        help="Require daemon availability for share submission")
    parser.add_argument("--allowed-prev-node-types", type=str, default=None,
                        help="Comma-separated allowed previous node types")
    parser.add_argument("--wire-dtype", type=str, default=None,
                        choices=["float32", "float16"],
                        help="Activation wire dtype")
    parser.add_argument("--compress-activations", type=str, default=None,
                        choices=["true", "false", "1", "0"],
                        help="Enable activation compression")
    parser.add_argument("--compress-threshold", type=int, default=None,
                        help="Compression threshold in bytes")
    parser.add_argument("--he-compute-mode", type=str, default=None,
                        choices=["off", "decode_client_sample", "server_sample"],
                        help="HE compute mode")
    parser.add_argument("--he-compute-top-k", type=int, default=None,
                        help="Default HE top-k")
    parser.add_argument("--he-compute-temperature", type=float, default=None,
                        help="Default HE temperature")
    parser.add_argument("--he-compute-top-p", type=float, default=None,
                        help="Default HE top-p")
    parser.add_argument("--he-sidecar-url", type=str, default=None,
                        dest="he_full_vocab_sidecar_url",
                        help="HE sidecar URL")
    parser.add_argument("--he-sidecar-timeout-ms", type=int, default=None,
                        dest="he_full_vocab_sidecar_timeout_ms",
                        help="HE sidecar timeout in ms")
    parser.add_argument("--he-sidecar-required", type=str, default=None,
                        choices=["true", "false", "1", "0"],
                        dest="he_full_vocab_sidecar_required",
                        help="Fail closed if sidecar unavailable")

    # --- MPC-only ---
    parser.add_argument("--mpc-role", type=str, default=None,
                        choices=["A", "B"],
                        help="MPC role: A (entry point) or B (peer)")
    parser.add_argument("--peer", type=str, default=None,
                        dest="peer_address",
                        help="Address of the other MPC node (required for MPC)")
    parser.add_argument("--mpc-dnc-mode", type=str, default=None,
                        choices=["off", "manual", "auto"],
                        help="MPC DNC mode (off/manual/auto)")
    parser.add_argument("--mpc-dnc-depth", type=int, default=None,
                        help="MPC DNC depth for manual mode")
    parser.add_argument("--mpc-dnc-split-dim", type=int, default=None,
                        help="MPC DNC split dim for element-wise ops")
    parser.add_argument("--mpc-dnc-matmul-split-dim", type=int, default=None,
                        help="MPC DNC split dim for matmul ops")
    parser.add_argument("--mpc-dnc-parallel-workers", type=int, default=None,
                        help="MPC DNC parallel workers for manual mode")
    parser.add_argument("--mpc-dnc-auto-min-elems", type=int, default=None,
                        help="MPC DNC auto: minimum tensor elements")
    parser.add_argument("--mpc-dnc-auto-chunk-min-elems", type=int, default=None,
                        help="MPC DNC auto: minimum chunk elements")
    parser.add_argument("--mpc-dnc-auto-max-depth", type=int, default=None,
                        help="MPC DNC auto: maximum depth")
    parser.add_argument("--mpc-dnc-auto-max-workers", type=int, default=None,
                        help="MPC DNC auto: max workers")

    # --- Daemon-only ---
    parser.add_argument("--db", type=str, default=None,
                        dest="db_path",
                        help="SQLite database path (daemon only)")
    parser.add_argument("--block-interval", type=float, default=None,
                        dest="block_interval_seconds",
                        help="Target block time in seconds (daemon only)")
    parser.add_argument("--fee-enabled", action="store_true", default=None,
                        help="Enable dynamic fee oracle (daemon only)")

    # --- Verifier-only ---
    parser.add_argument("--poll-interval", type=float, default=None,
                        dest="poll_interval_seconds",
                        help="Verifier poll interval in seconds")
    parser.add_argument("--max-tickets-per-poll", type=int, default=None,
                        help="Max tickets per verifier poll")
    parser.add_argument("--zk-challenge-positions", type=int, default=None,
                        help="Number of ZK challenge positions")
    parser.add_argument("--zk-tolerance", type=float, default=None,
                        help="ZK float comparison tolerance")

    return parser


def cli_to_overrides(args: argparse.Namespace) -> dict:
    """
    Convert parsed CLI args to a dict of overrides.

    Translates kebab-case CLI names to snake_case config keys.
    Only includes values that were explicitly provided (not None).
    """
    mapping = {
        # Base
        "role": "role",
        "port": "port",
        "host": "host",
        "advertise": "advertise",
        "registry": "registry",
        "require_tls_for_public": "require_tls_for_public",
        "grpc_max_workers": "grpc_max_workers",
        "grpc_max_message_mb": "grpc_max_message_mb",
        "heartbeat_interval_seconds": "heartbeat_interval_seconds",
        "connect_timeout": "connect_timeout",
        "forward_timeout": "forward_timeout",
        "max_retries": "max_retries",
        "retry_backoff_base": "retry_backoff_base",
        "retry_backoff_max": "retry_backoff_max",
        "keepalive_time_ms": "keepalive_time_ms",
        "keepalive_timeout_ms": "keepalive_timeout_ms",
        # Compute
        "shard_index": "shard_index",
        "model_name": "model_name",
        "model_type": "model_type",
        "shards_dir": "shards_dir",
        "manifest_path": "manifest_path",
        "max_memory_gb": "max_memory_gb",
        "session_timeout_seconds": "session_timeout_seconds",
        "eos_token_id": "eos_token_id",
        "verification_sampling_rate": "verification_sampling_rate",
        "ticket_submit_interval_seconds": "ticket_submit_interval_seconds",
        "shard_transfer_timeout": "shard_transfer_timeout",
        "shard_chunk_size": "shard_chunk_size",
        "require_daemon": "require_daemon",
        "allowed_prev_node_types": "allowed_prev_node_types",
        "wire_dtype": "wire_dtype",
        "compress_activations": "compress_activations",
        "compress_threshold": "compress_threshold",
        "he_compute_mode": "he_compute_mode",
        "he_compute_top_k": "he_compute_top_k",
        "he_compute_temperature": "he_compute_temperature",
        "he_compute_top_p": "he_compute_top_p",
        "he_full_vocab_sidecar_url": "he_full_vocab_sidecar_url",
        "he_full_vocab_sidecar_timeout_ms": "he_full_vocab_sidecar_timeout_ms",
        "he_full_vocab_sidecar_required": "he_full_vocab_sidecar_required",
        # MPC
        "mpc_role": "mpc_role",
        "peer_address": "peer_address",
        "mpc_dnc_mode": "mpc_dnc_mode",
        "mpc_dnc_depth": "mpc_dnc_depth",
        "mpc_dnc_split_dim": "mpc_dnc_split_dim",
        "mpc_dnc_matmul_split_dim": "mpc_dnc_matmul_split_dim",
        "mpc_dnc_parallel_workers": "mpc_dnc_parallel_workers",
        "mpc_dnc_auto_min_elems": "mpc_dnc_auto_min_elems",
        "mpc_dnc_auto_chunk_min_elems": "mpc_dnc_auto_chunk_min_elems",
        "mpc_dnc_auto_max_depth": "mpc_dnc_auto_max_depth",
        "mpc_dnc_auto_max_workers": "mpc_dnc_auto_max_workers",
        # Daemon
        "db_path": "db_path",
        "block_interval_seconds": "block_interval_seconds",
        "fee_enabled": "fee_enabled",
        # Verifier
        "poll_interval_seconds": "poll_interval_seconds",
        "max_tickets_per_poll": "max_tickets_per_poll",
        "zk_challenge_positions": "zk_challenge_positions",
        "zk_tolerance": "zk_tolerance",
    }

    overrides = {}
    args_dict = vars(args)
    for cli_key, config_key in mapping.items():
        val = args_dict.get(cli_key)
        if val is not None:
            if cli_key in (
                "require_daemon",
                "compress_activations",
                "he_full_vocab_sidecar_required",
                "require_tls_for_public",
            ):
                if isinstance(val, str):
                    val = val.strip().lower() in ("1", "true", "yes", "on")
            if cli_key == "allowed_prev_node_types" and isinstance(val, str):
                val = [x.strip() for x in val.split(",") if x.strip()]
            overrides[config_key] = val

    return overrides


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build overrides from CLI
    cli_overrides = cli_to_overrides(args)

    # Load config (file + CLI merge)
    try:
        cfg = load_config(args.config, cli_overrides)
    except ValueError as e:
        print(f"[Config Error] {e}")
        sys.exit(1)

    # Print startup summary
    print()
    print_config_summary(cfg, args.config)
    print()

    # Dispatch to the appropriate server based on the typed config
    # (check MPC before Compute since MPCConfig inherits from ComputeConfig)
    if isinstance(cfg, MPCConfig):
        from network.mpc_shard0 import serve as serve_mpc
        serve_mpc(
            role=cfg.mpc_role,
            port=cfg.port,
            peer_address=cfg.peer_address,
            host=cfg.host,
            advertise=cfg.advertise,
            registry_address=cfg.registry,
            require_daemon=cfg.require_daemon,
            allowed_prev_node_types=cfg.allowed_prev_node_types,
            mpc_dnc_mode=cfg.mpc_dnc_mode,
            mpc_dnc_depth=cfg.mpc_dnc_depth,
            mpc_dnc_split_dim=cfg.mpc_dnc_split_dim,
            mpc_dnc_matmul_split_dim=cfg.mpc_dnc_matmul_split_dim,
            mpc_dnc_parallel_workers=cfg.mpc_dnc_parallel_workers,
            mpc_dnc_auto_min_elems=cfg.mpc_dnc_auto_min_elems,
            mpc_dnc_auto_chunk_min_elems=cfg.mpc_dnc_auto_chunk_min_elems,
            mpc_dnc_auto_max_depth=cfg.mpc_dnc_auto_max_depth,
            mpc_dnc_auto_max_workers=cfg.mpc_dnc_auto_max_workers,
        )

    elif isinstance(cfg, ComputeConfig):
        from node.server import serve as serve_compute
        serve_compute(
            shard_index=cfg.shard_index,
            port=cfg.port,
            host=cfg.host,
            advertise_address=cfg.advertise,
            registry_address=cfg.registry,
            node_config=cfg,
            model_type=cfg.model_type,
            shards_dir=cfg.shards_dir,
            tls_cert=cfg.tls_cert or None,
            tls_key=cfg.tls_key or None,
        )

    elif isinstance(cfg, DaemonConfig):
        from network.daemon_node import serve as serve_daemon
        serve_daemon(
            port=cfg.port,
            host=cfg.host,
            advertise_address=cfg.advertise,
            registry_address=cfg.registry,
            db_path=cfg.db_path,
        )

    else:
        print(f"[Error] Unknown config type: {type(cfg).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()
