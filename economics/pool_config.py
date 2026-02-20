"""
Pool Configuration — per-model economic rules.

Each model pool on the network has its own PoolConfig that defines:
  - Staking requirements (how much a node must stake to join)
  - Reward scheme (PPLNS, PPS, or proportional — like mining pools)
  - Fee parameters (per-token pricing)
  - Shard-level payout overrides

The PoolConfig is published to the registry alongside the manifest.
The manifest describes *what* the model is (shards, hashes, layers).
The PoolConfig describes *the deal* for serving it (rewards, stakes, fees).

Lifecycle:
  1. Model pool creator sets economic rules in a PoolConfig
  2. PoolConfig is published to registry via PutPoolConfig RPC
  3. Nodes fetch it via GetPoolConfig to learn the deal before joining
  4. ModelPool uses the PoolConfig for payout calculations

Reward schemes (inspired by mining pool payout methods):
  - "proportional": shares in a settlement window are split proportionally
  - "pplns": Pay Per Last N Shares — only the last N shares count,
             rewarding consistent contributors over hop-in/hop-out miners
  - "pps": Pay Per Share — fixed rate per share regardless of block luck,
           the pool absorbs variance (needs a pool operator / treasury)
"""

import json
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Optional

VALID_REWARD_SCHEMES = ("proportional", "pplns", "pps")


@dataclass
class PoolConfig:
    """Per-model economic rules — published alongside the manifest."""

    # --- Identity ---
    model_id: str = ""                   # e.g. "Qwen/Qwen2.5-0.5B"

    # --- Staking ---
    min_stake: float = 100.0             # minimum stake to join this pool
    slash_fraction: float = 0.5          # fraction of stake burned on fraud (0.0-1.0)

    # --- Rewards ---
    reward_scheme: str = "proportional"  # "proportional", "pplns", or "pps"
    base_rate: float = 0.001             # base payment per compute share (legacy)

    # --- Per-token pricing (set by cluster/pool operator) ---
    # These are what clients see and pay.  Revenue from token pricing is
    # distributed to nodes proportionally by their weighted compute shares.
    price_per_input_token: float = 0.0001   # UNFED per input token
    price_per_output_token: float = 0.001    # UNFED per output token

    pplns_window: int = 1000             # N shares lookback (only for pplns)
    pps_rate: float = 0.0               # fixed rate per share (only for pps, 0 = use base_rate)
    max_payout_multiplier: float = 5.0   # cap on scarcity multiplier for underserved shards

    # --- Fees (EIP-1559-style dynamic pricing) ---
    fee_base: float = 0.001             # starting base fee per token
    fee_min: float = 0.0001             # floor (prevents race to zero)
    fee_max: float = 0.1                # ceiling (prevents price shock)
    fee_adjustment_factor: float = 0.125 # how aggressively the fee adjusts per block
    fee_target_utilization: float = 0.7  # target network load (70%)
    fee_window_blocks: int = 10          # rolling window for utilization calculation
    fee_target_capacity: int = 40        # expected shares/block at 100% utilization

    # --- Shard-level overrides (optional) ---
    # Manual payout boosts for specific shards, e.g. {0: 2.0, 3: 1.5}
    # On top of the automatic scarcity multiplier.
    shard_multipliers: dict[int, float] = field(default_factory=dict)

    # --- Metadata ---
    created_at: float = 0.0             # Unix timestamp (set on publish)
    updated_at: float = 0.0             # Unix timestamp (set on update)
    description: str = ""               # Human-readable pool description

    def validate(self) -> list[str]:
        """Validate the config. Returns a list of error messages (empty = valid)."""
        errors = []

        if not self.model_id:
            errors.append("model_id is required")

        if self.min_stake < 0:
            errors.append(f"min_stake must be >= 0, got {self.min_stake}")

        if not 0.0 <= self.slash_fraction <= 1.0:
            errors.append(f"slash_fraction must be in [0.0, 1.0], "
                          f"got {self.slash_fraction}")

        if self.reward_scheme not in VALID_REWARD_SCHEMES:
            errors.append(f"reward_scheme must be one of {VALID_REWARD_SCHEMES}, "
                          f"got '{self.reward_scheme}'")

        if self.base_rate < 0:
            errors.append(f"base_rate must be >= 0, got {self.base_rate}")

        if self.pplns_window < 1:
            errors.append(f"pplns_window must be >= 1, got {self.pplns_window}")

        if self.fee_min > self.fee_max:
            errors.append(f"fee_min ({self.fee_min}) > fee_max ({self.fee_max})")

        if self.fee_base < self.fee_min or self.fee_base > self.fee_max:
            errors.append(f"fee_base ({self.fee_base}) must be between "
                          f"fee_min ({self.fee_min}) and fee_max ({self.fee_max})")

        if not 0.0 < self.fee_target_utilization <= 1.0:
            errors.append(f"fee_target_utilization must be in (0.0, 1.0], "
                          f"got {self.fee_target_utilization}")

        if self.max_payout_multiplier < 1.0:
            errors.append(f"max_payout_multiplier must be >= 1.0, "
                          f"got {self.max_payout_multiplier}")

        for shard_idx, mult in self.shard_multipliers.items():
            if mult < 0:
                errors.append(f"shard_multipliers[{shard_idx}] must be >= 0, "
                              f"got {mult}")

        return errors

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        # Convert shard_multipliers keys to strings for JSON compatibility
        d["shard_multipliers"] = {
            str(k): v for k, v in d["shard_multipliers"].items()
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PoolConfig":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        # Convert shard_multipliers keys back to ints
        if "shard_multipliers" in d:
            d["shard_multipliers"] = {
                int(k): v for k, v in d["shard_multipliers"].items()
            }
        # Only pass known fields
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str) -> "PoolConfig":
        """Load from a JSON file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())

    def save(self, path: str):
        """Save to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


def default_pool_config(model_id: str) -> PoolConfig:
    """Create a PoolConfig with sensible defaults for a model."""
    return PoolConfig(
        model_id=model_id,
        created_at=time.time(),
        updated_at=time.time(),
    )


def print_pool_config(cfg: PoolConfig):
    """Print a human-readable summary of the pool config."""
    sep = "=" * 60
    div = "-" * 60

    print(sep)
    print(f"  Pool Config: {cfg.model_id}")
    print(sep)

    print(f"  Reward scheme:   {cfg.reward_scheme}")
    if cfg.reward_scheme == "pplns":
        print(f"  PPLNS window:    last {cfg.pplns_window} shares")
    elif cfg.reward_scheme == "pps":
        rate = cfg.pps_rate if cfg.pps_rate > 0 else cfg.base_rate
        print(f"  PPS rate:        {rate} per share")
    print(f"  Base rate:       {cfg.base_rate} per share")
    print(f"  Max multiplier:  {cfg.max_payout_multiplier}x")

    print(div)
    print(f"  Min stake:       {cfg.min_stake}")
    print(f"  Slash fraction:  {cfg.slash_fraction * 100:.0f}%")

    print(div)
    print(f"  Fee base:        {cfg.fee_base}")
    print(f"  Fee range:       [{cfg.fee_min}, {cfg.fee_max}]")
    print(f"  Fee target util: {cfg.fee_target_utilization * 100:.0f}%")
    print(f"  Fee adjustment:  {cfg.fee_adjustment_factor}")
    if cfg.shard_multipliers:
        print(div)
        print(f"  Shard overrides:")
        for shard_idx, mult in sorted(cfg.shard_multipliers.items()):
            print(f"    shard {shard_idx}: {mult}x")

    if cfg.description:
        print(div)
        print(f"  Description:     {cfg.description}")

    print(sep)
