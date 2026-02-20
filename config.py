"""
UNFED AI - Configuration

Model settings, registry address, and generation defaults.
Node addresses are no longer hardcoded — they're discovered dynamically via the registry.
"""

import json
import os

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
NUM_LAYERS = 24  # Qwen2.5-0.5B has 24 transformer layers

# Paths
SHARDS_DIR = "shards"
MODEL_CACHE_DIR = "model_cache"
MANIFEST_PATH = "shards/manifest.json"

# Registry (bootstrap / DNS seed)
REGISTRY_ADDRESS = os.environ.get("UNFED_REGISTRY", "localhost:50050")

# Seed registries — UNFED_SEEDS env (comma-separated), then seeds.json, then REGISTRY_ADDRESS
def _load_seed_registries() -> list[str]:
    env_seeds = os.environ.get("UNFED_SEEDS", "")
    if env_seeds:
        return [s.strip() for s in env_seeds.split(",") if s.strip()]
    seeds_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "network", "seeds.json"
    )
    try:
        with open(seeds_path, "r") as f:
            data = json.load(f)
        registries = data.get("registries", [])
        if registries:
            return registries
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return [REGISTRY_ADDRESS]

SEED_REGISTRIES = _load_seed_registries()

# Generation defaults
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9

# Session management
SESSION_TIMEOUT_SECONDS = 300  # clean up KV caches after 5 min of inactivity

# Node health
HEARTBEAT_INTERVAL_SECONDS = 10   # how often nodes send heartbeats
NODE_TIMEOUT_SECONDS = 30         # remove node from registry after no heartbeat

# gRPC settings
GRPC_MAX_MESSAGE_SIZE = 512 * 1024 * 1024  # 512 MB (vision tensors can be ~300MB)
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_SIZE),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_SIZE),
    # gzip compression: algorithm 2 = gzip (compress messages > 1KB)
    ("grpc.default_compression_algorithm", 2),
    ("grpc.grpc.min_sent_message_size_to_compress", 1024),
    # Keepalive: prevent TCP connections from being silently dropped
    ("grpc.keepalive_time_ms", 120000),           # ping every 120s (vision can take minutes)
    ("grpc.keepalive_timeout_ms", 30000),          # 30s timeout for pong
    ("grpc.keepalive_permit_without_calls", 1),   # keep alive even when idle
    ("grpc.http2.max_pings_without_data", 0),     # unlimited pings
    ("grpc.http2.min_ping_interval_without_data_ms", 60000),  # min 60s between pings
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 30000),  # server tolerates pings every 30s
]

# Per-shard racing (fault tolerance)
RACING_REPLICAS = 2            # number of nodes to race per shard
SHARD_TIMEOUT_SECONDS = 30     # timeout per shard hop

# Shard transfer
SHARD_CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunk size for gRPC transfer
P2P_PIECE_SIZE = 4 * 1024 * 1024  # 4 MB logical piece size for multi-peer download


# --- Daemon / Chain persistence ---
CHAIN_DB_PATH = os.path.expanduser("~/.unfed/chain.db")
DAEMON_PORT = 50070
CHAIN_PRUNE_KEEP = 10000     # keep last N blocks (0 = no pruning)

# --- Compression ---
GRPC_COMPRESSION = True       # gzip compression on gRPC channels
COMPRESS_ACTIVATIONS = True   # application-level gzip for activation tensors
COMPRESS_THRESHOLD = int(os.environ.get("UNFED_COMPRESS_THRESHOLD", "16384"))

# --- Wire format ---
WIRE_DTYPE = os.environ.get("UNFED_WIRE_DTYPE", "float16")

# --- Pipelined prefill ---
PREFILL_PIPELINE_MIN_TOKENS = int(os.environ.get("UNFED_PREFILL_MIN", "64"))
PREFILL_CHUNK_SIZE = 64            # tokens per chunk in pipelined prefill

# --- Fee / Economics (EIP-1559-style dynamic pricing) ---
# NOTE: These are *network-wide defaults*. Per-model economics (staking,
# reward schemes, fees) are configured via PoolConfig and published to the
# registry alongside the manifest. See economics/pool_config.py.
FEE_TARGET_UTILIZATION = 0.7     # 70% target network load
FEE_BASE_DEFAULT = 0.001         # starting base fee per token
FEE_MIN = 0.0001                 # floor (prevents race to zero)
FEE_MAX = 0.1                    # ceiling (prevents price shock)
FEE_ADJUSTMENT_FACTOR = 0.125    # how aggressively the fee adjusts per block
FEE_WINDOW_BLOCKS = 10           # rolling window for utilization calculation
FEE_TARGET_CAPACITY = 40         # expected shares/block at 100% utilization


def get_shard_path(shard_index: int) -> str:
    """Return the file path for a shard."""
    return f"{SHARDS_DIR}/shard_{shard_index}.pt"
