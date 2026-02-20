"""
Cluster Configuration — per-cluster identity and default economics.

A cluster is a single registry instance operated by one entity, analogous to
a mining pool in Monero.  The cluster operator:
  1. Creates a ClusterConfig (identity + default economics)
  2. Starts the registry with --cluster-config cluster.json
  3. Publishes model manifests (PutManifest)
  4. Optionally sets per-model economic overrides (PutPoolConfig)

Nodes are free to join any cluster.  Clients discover clusters via seed
lists and peer-exchange gossip, then choose based on price, latency, model
availability, reputation, etc.

The ClusterConfig provides *default* economic rules that apply to every model
hosted by this cluster.  Per-model overrides (via PoolConfig / PutPoolConfig)
take precedence where set.
"""

import json
import uuid
import time
from dataclasses import dataclass, fields, asdict


# ---------------------------------------------------------------------------
# Fields that can be overridden per-model.  These names must match between
# ClusterConfig (prefixed with ``default_``) and PoolConfig (unprefixed).
# ---------------------------------------------------------------------------
_OVERRIDABLE_FIELDS = (
    "min_stake",
    "slash_fraction",
    "reward_scheme",
    "base_rate",
    "price_per_input_token",
    "price_per_output_token",
    "fee_base",
    "fee_min",
    "fee_max",
    "fee_adjustment_factor",
    "fee_target_utilization",
    "fee_window_blocks",
    "fee_target_capacity",
    "pplns_window",
    "pps_rate",
    "max_payout_multiplier",
)


@dataclass
class ClusterConfig:
    """Cluster-wide identity and default economics."""

    # --- Identity ---
    cluster_id: str = ""           # UUID, auto-generated if empty
    name: str = ""                 # Human-readable, e.g. "FastAI Pool"
    description: str = ""          # "Low-latency GPU cluster in US-East"
    operator: str = ""             # Operator name or contact
    public_endpoint: str = ""      # Externally reachable "registry.fastai.io:50050"

    # --- Default pool economics (applied to all models unless overridden) ---
    default_min_stake: float = 100.0
    default_slash_fraction: float = 0.5
    default_reward_scheme: str = "proportional"
    default_base_rate: float = 0.001

    # --- Per-token pricing (set by cluster operator) ---
    # Clients are charged: (input_tokens × input_price) + (output_tokens × output_price)
    # Revenue is distributed to nodes proportionally by their weighted shares.
    default_price_per_input_token: float = 0.0001   # UNFED per input token
    default_price_per_output_token: float = 0.001    # UNFED per output token

    default_pplns_window: int = 1000
    default_pps_rate: float = 0.0
    default_max_payout_multiplier: float = 5.0

    default_fee_base: float = 0.001
    default_fee_min: float = 0.0001
    default_fee_max: float = 0.1
    default_fee_adjustment_factor: float = 0.125
    default_fee_target_utilization: float = 0.7
    default_fee_window_blocks: int = 10
    default_fee_target_capacity: int = 40

    # --- On-chain escrow (recommended — empty = in-memory simulation) ---
    # These fields are required for production deployment.  When both
    # chain_rpc_url and escrow_contract_address are set, on-chain payments
    # become the default path.  Use the registry's --no-chain flag to
    # force in-memory simulation for development.
    chain_rpc_url: str = ""              # e.g. "http://localhost:8545"
    escrow_contract_address: str = ""    # Deployed UnfedEscrow address
    staking_token_address: str = ""      # ERC-20 token address
    operator_private_key: str = ""       # Operator's signing key (hex)
    cooldown_seconds: int = 3600         # Unbonding delay
    challenge_window_seconds: int = 60   # Settlement dispute window

    # --- Metadata ---
    created_at: float = 0.0
    updated_at: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def ensure_id(self) -> None:
        """Generate a cluster_id if not already set."""
        if not self.cluster_id:
            self.cluster_id = str(uuid.uuid4())

    def validate(self) -> list[str]:
        """Basic validation.  Returns list of error strings (empty = OK)."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if self.default_min_stake < 0:
            errors.append(f"default_min_stake must be >= 0, got "
                          f"{self.default_min_stake}")
        if not 0.0 <= self.default_slash_fraction <= 1.0:
            errors.append(f"default_slash_fraction must be in [0, 1], got "
                          f"{self.default_slash_fraction}")
        if self.default_fee_min > self.default_fee_max:
            errors.append(f"default_fee_min ({self.default_fee_min}) > "
                          f"default_fee_max ({self.default_fee_max})")
        # Warn (not error) if on-chain fields are missing
        if not self.chain_rpc_url or not self.escrow_contract_address:
            errors.append(
                "WARNING: chain_rpc_url and escrow_contract_address are "
                "recommended for production. On-chain payments will be "
                "disabled; using in-memory simulation.")
        return errors

    @property
    def has_onchain_escrow(self) -> bool:
        """True if on-chain escrow fields are configured."""
        return bool(self.chain_rpc_url and self.escrow_contract_address)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ClusterConfig":
        d = json.loads(json_str)
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str) -> "ClusterConfig":
        with open(path) as f:
            return cls.from_json(f.read())

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


# ---------------------------------------------------------------------------
# Merge logic: cluster defaults + per-model overrides -> effective PoolConfig
# ---------------------------------------------------------------------------

def merge_with_defaults(cluster: ClusterConfig,
                        model_override: "PoolConfig | None",
                        model_id: str = "") -> "PoolConfig":
    """Build an effective PoolConfig by merging cluster defaults with
    optional per-model overrides.

    For each overridable field, the per-model value is used if the
    model_override is provided and its value differs from the PoolConfig
    default (indicating the operator explicitly set it).  Otherwise the
    cluster-wide default is used.

    Args:
        cluster: The cluster's default economics.
        model_override: Per-model overrides (may be None).
        model_id: Model ID to stamp on the result.

    Returns:
        A fully-populated PoolConfig.
    """
    from economics.pool_config import PoolConfig

    # Start from cluster defaults
    kwargs: dict = {"model_id": model_id or (
        model_override.model_id if model_override else "")}

    pool_defaults = PoolConfig()  # pristine defaults for comparison

    for field_name in _OVERRIDABLE_FIELDS:
        cluster_val = getattr(cluster, f"default_{field_name}")

        if model_override is not None:
            override_val = getattr(model_override, field_name)
            default_val = getattr(pool_defaults, field_name)
            # Use override if explicitly changed from PoolConfig defaults
            if override_val != default_val:
                kwargs[field_name] = override_val
                continue

        kwargs[field_name] = cluster_val

    # Non-overridable fields from model_override
    if model_override is not None:
        kwargs["shard_multipliers"] = model_override.shard_multipliers
        kwargs["description"] = (model_override.description
                                 or cluster.description)
        kwargs["created_at"] = model_override.created_at
        kwargs["updated_at"] = model_override.updated_at
    else:
        kwargs["created_at"] = cluster.created_at
        kwargs["updated_at"] = cluster.updated_at

    return PoolConfig(**kwargs)
