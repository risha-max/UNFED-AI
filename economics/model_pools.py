"""
Model Pool System — manages model pools with health tracking and
self-balancing economic incentives.

A model pool is a collection of nodes collectively serving a specific model.
Each pool covers all shards needed for inference.

Pool features:
  - On-chain manifest: model ID, shard hashes, layer ranges
  - Health tracking: which shards are covered, redundancy level
  - Self-balancing incentives: underserved shards pay more to attract nodes
  - Multi-pool support: a node can participate in multiple model pools
  - Per-pool economics via PoolConfig: staking, reward scheme, fees

Incentive mechanism:
  - Base rate: per-pool payment per compute share (from PoolConfig)
  - Scarcity multiplier: shards with fewer nodes get higher payout
  - The multiplier is: max_nodes_per_shard / this_shard_node_count
  - Shard overrides: manual boosts from PoolConfig.shard_multipliers
  - This naturally attracts nodes to underserved shards
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from economics.pool_config import PoolConfig


@dataclass
class PoolManifest:
    """Describes a model pool — its shards and requirements."""
    model_id: str
    num_shards: int
    shard_hashes: dict[int, str]     # shard_index -> sha256 hash
    total_layers: int
    layers_per_shard: int
    created_at: float = field(default_factory=time.time)


@dataclass
class ShardStatus:
    """Real-time status of a shard in a pool."""
    shard_index: int
    node_count: int
    node_ids: list[str]
    is_covered: bool               # at least 1 node
    is_healthy: bool               # 3+ nodes (redundancy)
    payout_multiplier: float       # higher = more incentive to join


@dataclass
class PoolStatus:
    """Overall pool status."""
    model_id: str
    shards: list[ShardStatus]
    total_nodes: int
    can_serve: bool                # all shards covered
    health_score: float            # 0.0 - 1.0 (fraction of healthy shards)


class ModelPool:
    """
    Manages a single model pool — tracks nodes, health, and incentives.

    Economics are driven by the optional PoolConfig. If no PoolConfig is
    provided, the pool uses the base_rate passed at construction (backward
    compatible with existing code).
    """

    def __init__(self, manifest: PoolManifest, base_rate: float = 0.001,
                 pool_config: "Optional[PoolConfig]" = None):
        self.manifest = manifest
        self.base_rate = base_rate
        self.pool_config = pool_config

        # If a pool config is provided, use its economics
        if pool_config:
            self.base_rate = pool_config.base_rate

        # shard_index -> set of node_ids
        self._shard_nodes: dict[int, set[str]] = {
            i: set() for i in range(manifest.num_shards)
        }
        self._lock = threading.Lock()

    def add_node(self, shard_index: int, node_id: str):
        """Register a node as serving a shard."""
        with self._lock:
            if shard_index in self._shard_nodes:
                self._shard_nodes[shard_index].add(node_id)

    def remove_node(self, shard_index: int, node_id: str):
        """Remove a node from a shard."""
        with self._lock:
            if shard_index in self._shard_nodes:
                self._shard_nodes[shard_index].discard(node_id)

    def get_payout_multiplier(self, shard_index: int) -> float:
        """
        Calculate the payout multiplier for a shard.

        Underserved shards (fewer nodes) get a higher multiplier
        to attract more nodes.

        Formula: (max_count / this_count) * shard_override
        Capped at max_payout_multiplier from the PoolConfig (default 5.0).
        """
        # Max multiplier from pool config, or default 5.0
        cap = 5.0
        if self.pool_config:
            cap = self.pool_config.max_payout_multiplier

        with self._lock:
            counts = [len(nodes) for nodes in self._shard_nodes.values()]

        if not counts:
            return 1.0

        max_count = max(max(counts), 1)
        this_count = max(len(self._shard_nodes.get(shard_index, set())), 1)

        multiplier = max_count / this_count

        # Apply manual shard override from pool config (if set)
        if self.pool_config and shard_index in self.pool_config.shard_multipliers:
            multiplier *= self.pool_config.shard_multipliers[shard_index]

        return min(max(multiplier, 1.0), cap)

    def get_effective_rate(self, shard_index: int) -> float:
        """Get the effective payment rate for a shard (base * multiplier)."""
        return self.base_rate * self.get_payout_multiplier(shard_index)

    def get_status(self) -> PoolStatus:
        """Get the current pool status."""
        with self._lock:
            shards = []
            total_nodes = set()
            healthy_count = 0

            max_nodes = max(
                (len(nodes) for nodes in self._shard_nodes.values()), default=1
            )

            for i in range(self.manifest.num_shards):
                nodes = self._shard_nodes.get(i, set())
                count = len(nodes)
                is_covered = count >= 1
                is_healthy = count >= 3

                if is_healthy:
                    healthy_count += 1

                total_nodes.update(nodes)

                shards.append(ShardStatus(
                    shard_index=i,
                    node_count=count,
                    node_ids=list(nodes),
                    is_covered=is_covered,
                    is_healthy=is_healthy,
                    payout_multiplier=min(max(max_nodes / max(count, 1), 1.0), 5.0),
                ))

        can_serve = all(s.is_covered for s in shards)
        health_score = healthy_count / max(len(shards), 1)

        return PoolStatus(
            model_id=self.manifest.model_id,
            shards=shards,
            total_nodes=len(total_nodes),
            can_serve=can_serve,
            health_score=health_score,
        )


class PoolRegistry:
    """
    Manages all model pools in the network.

    In production, pool manifests would be registered on-chain.
    """

    def __init__(self):
        self._pools: dict[str, ModelPool] = {}  # model_id -> ModelPool
        self._lock = threading.Lock()

    def register_pool(self, manifest: PoolManifest,
                      base_rate: float = 0.001,
                      pool_config: "Optional[PoolConfig]" = None) -> ModelPool:
        """Register a new model pool, optionally with per-pool economics."""
        with self._lock:
            pool = ModelPool(manifest, base_rate, pool_config=pool_config)
            self._pools[manifest.model_id] = pool
        scheme = pool_config.reward_scheme if pool_config else "proportional"
        rate = pool_config.base_rate if pool_config else base_rate
        print(f"[PoolRegistry] Registered pool: {manifest.model_id} "
              f"({manifest.num_shards} shards, scheme={scheme}, rate={rate})")
        return pool

    def get_pool(self, model_id: str) -> ModelPool | None:
        return self._pools.get(model_id)

    def get_all_pools(self) -> dict[str, PoolStatus]:
        """Get status of all pools."""
        return {
            model_id: pool.get_status()
            for model_id, pool in self._pools.items()
        }

    def node_joined(self, model_id: str, shard_index: int, node_id: str):
        """Called when a node joins a pool."""
        pool = self.get_pool(model_id)
        if pool:
            pool.add_node(shard_index, node_id)

    def node_left(self, model_id: str, shard_index: int, node_id: str):
        """Called when a node leaves a pool."""
        pool = self.get_pool(model_id)
        if pool:
            pool.remove_node(shard_index, node_id)

    def get_best_opportunity(self) -> dict | None:
        """
        Find the shard with the highest payout multiplier across all pools.

        Useful for nodes deciding which shard to serve — they should pick
        the one with the highest incentive (most underserved).
        """
        best = None
        best_multiplier = 0.0

        for model_id, pool in self._pools.items():
            status = pool.get_status()
            for shard in status.shards:
                if shard.payout_multiplier > best_multiplier:
                    best_multiplier = shard.payout_multiplier
                    best = {
                        "model_id": model_id,
                        "shard_index": shard.shard_index,
                        "payout_multiplier": shard.payout_multiplier,
                        "current_nodes": shard.node_count,
                        "effective_rate": pool.get_effective_rate(shard.shard_index),
                    }

        return best
