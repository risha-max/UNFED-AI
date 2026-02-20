"""
Node Configuration — per-role config classes with JSON + CLI override system.

Each node role has its own config class with only the fields it needs:

  BaseNodeConfig     — shared by all roles (port, host, registry, gRPC, network)
  ├── ComputeConfig  — shard, model, KV cache, P2P serving, verification
  ├── GuardConfig    — relay-specific settings
  └── VerifierConfig — polling, ZK verification, sampling

Priority: CLI > config file > defaults

Usage:
    from node.node_config import load_config, print_config_summary

    cfg = load_config("compute_config.json", cli_overrides={"port": 50052})
    print_config_summary(cfg, "compute_config.json")

    # Type-safe access:
    if isinstance(cfg, ComputeConfig):
        print(cfg.shard_index, cfg.kv_quantize)
"""

import json
import os
import sys
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as global_config

VALID_ROLES = ("compute", "guard", "verifier", "daemon", "mpc")

# Default ports per role
DEFAULT_PORTS = {
    "compute": 50051,
    "guard": 50060,
    "verifier": 50070,
    "daemon": 50080,
    "mpc": 50061,
}

# Default grpc_max_workers per role
DEFAULT_WORKERS = {
    "compute": 4,
    "guard": 10,
    "verifier": 2,
    "daemon": 10,
    "mpc": 4,
}


# ---------------------------------------------------------------------------
# Base config — shared by all node roles
# ---------------------------------------------------------------------------

@dataclass
class BaseNodeConfig:
    """Settings shared by every UNFED node role."""

    # --- Core ---
    role: str = "compute"
    port: int = 0                                # 0 = use DEFAULT_PORTS[role]
    host: str = "[::]"
    advertise: Optional[str] = None              # default: "localhost:<port>"
    registry: Optional[str] = None               # default: config.REGISTRY_ADDRESS

    # --- gRPC ---
    grpc_max_workers: int = 0                    # 0 = use DEFAULT_WORKERS[role]
    grpc_max_message_mb: int = 256
    heartbeat_interval_seconds: int = 10

    # --- Network resilience ---
    connect_timeout: float = 10.0
    forward_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    retry_backoff_max: float = 10.0
    keepalive_time_ms: int = 30000
    keepalive_timeout_ms: int = 10000

    # --- TLS ---
    tls_cert: str = ""
    tls_key: str = ""

    def resolve_defaults(self):
        """Fill in role-dependent defaults for fields left at sentinel values."""
        if self.port == 0:
            self.port = DEFAULT_PORTS.get(self.role, 50051)
        if self.grpc_max_workers == 0:
            self.grpc_max_workers = DEFAULT_WORKERS.get(self.role, 4)
        if self.advertise is None:
            self.advertise = f"localhost:{self.port}"
        if self.registry is None:
            self.registry = global_config.REGISTRY_ADDRESS

    @property
    def grpc_options(self) -> list:
        """Build gRPC channel options from config."""
        max_bytes = self.grpc_max_message_mb * 1024 * 1024
        return [
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ]


# ---------------------------------------------------------------------------
# Compute node config
# ---------------------------------------------------------------------------

@dataclass
class ComputeConfig(BaseNodeConfig):
    """Configuration for a compute node — runs model shards."""

    role: str = "compute"

    # --- Model / shard ---
    shard_index: Optional[int] = None
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_type: str = "qwen2"                    # "qwen2", "qwen2_vl", "smolvlm"
    shards_dir: str = "shards"
    manifest_path: str = "shards/manifest.json"
    max_memory_gb: Optional[float] = None
    session_timeout_seconds: int = 300
    eos_token_id: int = 151643

    # --- Verification ---
    verification_sampling_rate: float = 0.05     # fraction of forward passes to sample
    ticket_submit_interval_seconds: int = 10

    # --- Shard transfer ---
    shard_transfer_timeout: float = 600.0
    shard_chunk_size: int = 1048576              # 1 MB

    # --- KV cache management ---
    kv_quantize: str = "none"                    # "none" or "int8"
    kv_offload_enabled: bool = False
    kv_offload_after_seconds: float = 30.0
    max_kv_memory_gb: float = 0.0                # 0 = unlimited
    prefill_chunk_size: int = 0                  # 0 = no chunking

    # --- Compute device ---
    device: str = "cpu"                          # "cpu", "cuda", "cuda:0", or "auto"

    # --- P2P shard serving (bandwidth controls) ---
    serve_shards_enabled: bool = True
    max_upload_rate_mbps: float = 0.0            # 0 = unlimited
    max_concurrent_transfers: int = 0            # 0 = unlimited
    inference_priority: bool = True              # pause transfers during inference


# ---------------------------------------------------------------------------
# MPC node config (specialized compute for shard 0 privacy)
# ---------------------------------------------------------------------------

@dataclass
class MPCConfig(ComputeConfig):
    """Configuration for an MPC node — 2-party secret-shared shard 0.

    MPC nodes always run shard 0 with privacy-preserving secret sharing.
    Two MPC nodes (role A and B) work together: A is the entry point that
    clients talk to, B is the internal peer. Neither sees the raw tokens.
    """

    role: str = "mpc"

    # --- MPC-specific ---
    mpc_role: str = "A"                          # "A" (entry point) or "B" (peer)
    peer_address: str = ""                       # address of the other MPC node (required)

    # --- Overridden defaults ---
    shard_index: Optional[int] = 0               # MPC always runs shard 0


# ---------------------------------------------------------------------------
# Guard relay config
# ---------------------------------------------------------------------------

@dataclass
class GuardConfig(BaseNodeConfig):
    """Configuration for a guard relay node — IP anonymization layer."""

    role: str = "guard"

    # --- Relay settings ---
    max_concurrent_relays: int = 100             # max simultaneous relay sessions
    relay_timeout_seconds: float = 120.0         # timeout for a single relay hop
    log_connections: bool = False                 # log client connections (privacy tradeoff)


# ---------------------------------------------------------------------------
# Verifier node config
# ---------------------------------------------------------------------------

@dataclass
class VerifierConfig(BaseNodeConfig):
    """Configuration for a verifier node — spot-checks compute correctness."""

    role: str = "verifier"

    # --- Model (needed to re-execute and verify) ---
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model_type: str = "qwen2"
    shards_dir: str = "shards"
    manifest_path: str = "shards/manifest.json"

    # --- Verification ---
    poll_interval_seconds: float = 5.0
    max_tickets_per_poll: int = 10
    verification_sampling_rate: float = 0.05
    zk_challenge_positions: int = 16
    zk_tolerance: float = 1e-5


# ---------------------------------------------------------------------------
# Daemon node config
# ---------------------------------------------------------------------------

@dataclass
class DaemonConfig(BaseNodeConfig):
    """Configuration for a chain daemon — block production, gossip, fee oracle.

    Like Monero's monerod: no GPU, no model weights. Just chain management.
    """

    role: str = "daemon"

    # --- Chain storage ---
    db_path: str = "~/.unfed/chain.db"          # SQLite database path
    chain_prune_keep: int = 10000                # keep last N blocks (0 = no pruning)

    # --- Block production ---
    block_interval_seconds: float = 10.0         # target block time
    settlement_blocks: int = 6                   # blocks per settlement window
    collection_window_seconds: float = 4.0       # wait for shares before producing block

    # --- Gossip ---
    peer_refresh_interval_seconds: float = 30.0  # how often to refresh peer list
    gossip_timeout_seconds: float = 5.0          # timeout for gossip RPCs
    sync_timeout_seconds: float = 30.0           # timeout for initial sync

    # --- Fee oracle (EIP-1559-style) ---
    fee_enabled: bool = True                     # enable dynamic fee oracle
    fee_base: float = 0.001                      # starting base fee
    fee_min: float = 0.0001                      # floor
    fee_max: float = 0.1                         # ceiling
    fee_adjustment_factor: float = 0.125         # aggressiveness per block
    fee_target_utilization: float = 0.7          # target network load
    fee_window_blocks: int = 10                  # rolling window
    fee_target_capacity: int = 40                # expected shares/block at 100%


# ---------------------------------------------------------------------------
# Type alias for any config
# ---------------------------------------------------------------------------

NodeConfig = Union[ComputeConfig, MPCConfig, GuardConfig, VerifierConfig, DaemonConfig]

# Map role name -> config class
_ROLE_CONFIG_MAP = {
    "compute": ComputeConfig,
    "mpc": MPCConfig,
    "guard": GuardConfig,
    "verifier": VerifierConfig,
    "daemon": DaemonConfig,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str],
                cli_overrides: dict) -> NodeConfig:
    """
    Load a typed config from a JSON file + CLI overrides.

    Reads the 'role' field first (from CLI or file), then creates the
    correct config class with only the fields that role needs.

    Priority: CLI > config file > dataclass defaults.
    """
    merged = {}

    # 1. Load from JSON file
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_data = json.load(f)
        merged.update(file_data)

    # 2. Apply CLI overrides (skip None values — means "not provided")
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v

    # 3. Determine role (CLI > file > default "compute")
    role = merged.get("role", "compute")
    if role not in VALID_ROLES:
        raise ValueError(
            f"Invalid role '{role}'. Must be one of: {', '.join(VALID_ROLES)}")

    # 4. Get the right config class for this role
    config_cls = _ROLE_CONFIG_MAP[role]
    valid_keys = {fld.name for fld in fields(config_cls)}

    # Warn about unknown fields (but skip comments/metadata)
    if config_path:
        for k in merged:
            if k not in valid_keys and not k.startswith("_"):
                print(f"[Config] Warning: '{k}' is not a valid "
                      f"{config_cls.__name__} field (ignored)")

    # 5. Build the typed config
    filtered = {k: v for k, v in merged.items() if k in valid_keys}
    cfg = config_cls(**filtered)

    # 6. Validate
    _validate(cfg)

    # 7. Fill in role-dependent defaults
    cfg.resolve_defaults()

    return cfg


def _validate(cfg: NodeConfig):
    """Validate a config and raise clear errors."""
    if cfg.port != 0 and cfg.port < 0:
        raise ValueError(f"Port must be positive, got {cfg.port}")

    if cfg.grpc_max_workers != 0 and cfg.grpc_max_workers < 1:
        raise ValueError(
            f"grpc_max_workers must be >= 1, got {cfg.grpc_max_workers}")

    if isinstance(cfg, MPCConfig):
        # MPC-specific validation (before ComputeConfig check since MPC is a subclass)
        if cfg.mpc_role not in ("A", "B"):
            raise ValueError(
                f"mpc_role must be 'A' or 'B', got '{cfg.mpc_role}'")
        if not cfg.peer_address:
            raise ValueError(
                "MPCConfig requires 'peer_address' to be set "
                "(address of the other MPC node)")
        if cfg.shard_index != 0:
            raise ValueError(
                f"MPC nodes must run shard 0, got shard_index={cfg.shard_index}")

    elif isinstance(cfg, ComputeConfig):
        if cfg.shard_index is None:
            raise ValueError(
                "ComputeConfig requires 'shard_index' to be set "
                "(in config file or via --shard-index)")
        if not 0.0 <= cfg.verification_sampling_rate <= 1.0:
            raise ValueError(
                f"verification_sampling_rate must be in [0.0, 1.0], "
                f"got {cfg.verification_sampling_rate}")
        if cfg.kv_quantize not in ("none", "int8"):
            raise ValueError(
                f"kv_quantize must be 'none' or 'int8', "
                f"got '{cfg.kv_quantize}'")

    if isinstance(cfg, VerifierConfig):
        if not 0.0 <= cfg.verification_sampling_rate <= 1.0:
            raise ValueError(
                f"verification_sampling_rate must be in [0.0, 1.0], "
                f"got {cfg.verification_sampling_rate}")


# ---------------------------------------------------------------------------
# Shard info helper
# ---------------------------------------------------------------------------

def _get_shard_info(cfg: ComputeConfig) -> Optional[dict]:
    """Load shard metadata from the manifest, if available."""
    if cfg.shard_index is None:
        return None
    try:
        with open(cfg.manifest_path, "r") as f:
            manifest = json.load(f)
        shards = manifest.get("shards", [])
        # Also check text_shards for VL manifests
        if not shards:
            shards = manifest.get("text_shards", [])
        if cfg.shard_index < len(shards):
            return shards[cfg.shard_index]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_config_summary(cfg: NodeConfig, config_path: Optional[str]):
    """Print a clear startup summary of the resolved configuration."""
    src = config_path or "(none — CLI/defaults only)"
    sep = "=" * 60
    div = "-" * 60

    print(sep)
    print(f"  UNFED AI — {cfg.role.title()} Node")
    print(sep)
    print(f"  Port:            {cfg.port}")
    print(f"  Host:            {cfg.host}")
    print(f"  Advertise:       {cfg.advertise}")
    print(f"  Registry:        {cfg.registry}")

    # --- MPC-specific (check before Compute since MPC inherits from it) ---
    if isinstance(cfg, MPCConfig):
        print(div)
        print(f"  MPC role:        {cfg.mpc_role} ({'entry point' if cfg.mpc_role == 'A' else 'internal peer'})")
        print(f"  Peer address:    {cfg.peer_address}")
        print(f"  Shard:           0 (always — MPC protects embedding + layer 0)")
        print(f"  Model:           {cfg.model_name}")
        print(f"  Model type:      {cfg.model_type}")
        print(f"  Shards dir:      {cfg.shards_dir}")

    # --- Compute-specific ---
    elif isinstance(cfg, ComputeConfig):
        print(div)
        shard_info = _get_shard_info(cfg)
        if shard_info:
            layer_start = shard_info.get("layer_start", "?")
            layer_end = shard_info.get("layer_end", "?")
            size_mb = shard_info.get("size_bytes", 0) / (1024 * 1024)
            print(f"  Shard:           {cfg.shard_index} "
                  f"(layers {layer_start}-{int(layer_end) - 1}, {size_mb:.1f} MB)")
            if shard_info.get("has_embedding"):
                print(f"                   + embed_tokens")
            if shard_info.get("has_lm_head"):
                print(f"                   + norm + lm_head")
        else:
            print(f"  Shard:           {cfg.shard_index}")
        print(f"  Model:           {cfg.model_name}")
        print(f"  Model type:      {cfg.model_type}")
        print(f"  Shards dir:      {cfg.shards_dir}")
        if cfg.max_memory_gb is not None:
            print(f"  Memory limit:    {cfg.max_memory_gb} GB")
        print(div)
        print(f"  Session timeout: {cfg.session_timeout_seconds}s")
        print(f"  EOS token:       {cfg.eos_token_id}")
        print(div)
        quant_str = cfg.kv_quantize if cfg.kv_quantize != "none" else "disabled"
        offload_str = (f"after {cfg.kv_offload_after_seconds}s"
                       if cfg.kv_offload_enabled else "disabled")
        mem_str = (f"{cfg.max_kv_memory_gb} GB"
                   if cfg.max_kv_memory_gb > 0 else "unlimited")
        chunk_str = (f"{cfg.prefill_chunk_size} tokens"
                     if cfg.prefill_chunk_size > 0 else "disabled")
        print(f"  KV quantize:     {quant_str}")
        print(f"  KV offload:      {offload_str}")
        print(f"  KV memory cap:   {mem_str}")
        print(f"  Prefill chunks:  {chunk_str}")
        print(div)
        print(f"  Verification:    {cfg.verification_sampling_rate * 100:.1f}% sampling")
        print(f"  Ticket submit:   every {cfg.ticket_submit_interval_seconds}s")
        print(div)
        serve_str = "enabled" if cfg.serve_shards_enabled else "DISABLED"
        rate_str = (f"{cfg.max_upload_rate_mbps} Mbps"
                    if cfg.max_upload_rate_mbps > 0 else "unlimited")
        conc_str = (str(cfg.max_concurrent_transfers)
                    if cfg.max_concurrent_transfers > 0 else "unlimited")
        print(f"  P2P serving:     {serve_str}")
        print(f"  Upload rate:     {rate_str}")
        print(f"  Max transfers:   {conc_str}")
        print(f"  Infer priority:  {'yes' if cfg.inference_priority else 'no'}")

    # --- Guard-specific ---
    if isinstance(cfg, GuardConfig):
        print(div)
        print(f"  Max relays:      {cfg.max_concurrent_relays}")
        print(f"  Relay timeout:   {cfg.relay_timeout_seconds}s")
        print(f"  Log connections: {'yes' if cfg.log_connections else 'no'}")

    # --- Verifier-specific ---
    if isinstance(cfg, VerifierConfig):
        print(div)
        print(f"  Model:           {cfg.model_name}")
        print(f"  Model type:      {cfg.model_type}")
        print(f"  Shards dir:      {cfg.shards_dir}")
        print(div)
        print(f"  Poll interval:   {cfg.poll_interval_seconds}s")
        print(f"  Max tickets:     {cfg.max_tickets_per_poll} per poll")
        print(f"  Sampling rate:   {cfg.verification_sampling_rate * 100:.1f}%")
        print(f"  ZK positions:    {cfg.zk_challenge_positions}")
        print(f"  ZK tolerance:    {cfg.zk_tolerance}")

    # --- Daemon-specific ---
    if isinstance(cfg, DaemonConfig):
        print(div)
        db = os.path.expanduser(cfg.db_path)
        print(f"  Database:        {db}")
        print(f"  Chain pruning:   {'last ' + str(cfg.chain_prune_keep) + ' blocks' if cfg.chain_prune_keep > 0 else 'disabled'}")
        print(div)
        print(f"  Block interval:  {cfg.block_interval_seconds}s")
        print(f"  Settlement:      every {cfg.settlement_blocks} blocks")
        print(f"  Collection:      {cfg.collection_window_seconds}s")
        print(div)
        print(f"  Peer refresh:    every {cfg.peer_refresh_interval_seconds}s")
        print(f"  Gossip timeout:  {cfg.gossip_timeout_seconds}s")
        print(f"  Sync timeout:    {cfg.sync_timeout_seconds}s")
        print(div)
        if cfg.fee_enabled:
            print(f"  Fee oracle:      enabled")
            print(f"  Fee base:        {cfg.fee_base}")
            print(f"  Fee range:       [{cfg.fee_min}, {cfg.fee_max}]")
            print(f"  Fee target util: {cfg.fee_target_utilization * 100:.0f}%")
        else:
            print(f"  Fee oracle:      disabled")

    # --- gRPC + network (all roles) ---
    print(div)
    print(f"  gRPC workers:    {cfg.grpc_max_workers}")
    print(f"  gRPC max msg:    {cfg.grpc_max_message_mb} MB")
    print(f"  Heartbeat:       every {cfg.heartbeat_interval_seconds}s")
    print(div)
    print(f"  Connect timeout: {cfg.connect_timeout}s")
    print(f"  Forward timeout: {cfg.forward_timeout}s")
    print(f"  Max retries:     {cfg.max_retries}")
    print(f"  Retry backoff:   {cfg.retry_backoff_base}s base, "
          f"{cfg.retry_backoff_max}s max")
    print(f"  Keepalive:       {cfg.keepalive_time_ms}ms / "
          f"{cfg.keepalive_timeout_ms}ms")
    print(div)
    print(f"  Config file:     {src}")
    print(sep)
