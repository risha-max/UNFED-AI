"""
Fee Oracle — EIP-1559-style dynamic fee pricing for compute shares.

Tracks network utilization over a sliding window of recent blocks and
adjusts the base fee per token algorithmically:

  - When utilization > target (network is busy): fee goes UP
  - When utilization < target (network is idle): fee goes DOWN
  - Fee is clamped to [min_fee, max_fee]

This is analogous to Ethereum's EIP-1559 base fee mechanism, adapted
for a compute network instead of block space.

The oracle runs on the daemon node, since the daemon holds the chain
and produces all blocks.

Usage:
    oracle = FeeOracle()
    oracle.update(block)              # called after each new block
    fee = oracle.get_base_fee()       # current base fee per token
    est = oracle.get_fee_estimate(100) # estimate for 100 tokens
"""

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class FeeSnapshot:
    """A snapshot of fee state at a particular block."""
    block_index: int
    base_fee: float
    utilization: float
    shares_in_block: int
    timestamp: float


class FeeOracle:
    """EIP-1559-style dynamic fee oracle.

    Maintains a rolling window of recent blocks and adjusts the base fee
    based on observed utilization vs. the target.

    The adjustment formula per block:
        if utilization > target:
            base_fee *= (1 + adjustment_factor * (utilization - target) / target)
        else:
            base_fee *= (1 - adjustment_factor * (target - utilization) / target)

    This is a smoothed version of EIP-1559's discrete up/down adjustment.
    """

    def __init__(
        self,
        target_utilization: float = 0.7,
        base_fee: float = 0.001,
        min_fee: float = 0.0001,
        max_fee: float = 0.1,
        window_blocks: int = 10,
        adjustment_factor: float = 0.125,
        target_capacity: int = 40,
    ):
        """
        Args:
            target_utilization: Desired network utilization (0.0-1.0).
                When actual utilization exceeds this, fees rise.
            base_fee: Starting base fee per token.
            min_fee: Floor for the base fee (prevents race to zero).
            max_fee: Ceiling for the base fee (prevents price shock).
            window_blocks: Number of recent blocks to consider.
            adjustment_factor: How aggressively the fee adjusts (0.125 = 12.5%).
            target_capacity: Expected shares per block at 100% utilization.
                Default 40 = 4 shards x 10 tokens per block cycle.
        """
        self._target_utilization = target_utilization
        self._base_fee = base_fee
        self._min_fee = min_fee
        self._max_fee = max_fee
        self._window_blocks = window_blocks
        self._adjustment_factor = adjustment_factor
        self._target_capacity = target_capacity

        # Rolling window of recent block share counts
        self._recent_shares: deque[int] = deque(maxlen=window_blocks)
        self._history: deque[FeeSnapshot] = deque(maxlen=100)
        self._lock = threading.Lock()

        # Track last update time for fee staleness
        self._last_update = time.time()
        self._total_blocks_seen = 0

    def update(self, block) -> float:
        """Update the oracle with a new block. Returns the new base fee.

        Should be called by the daemon after each block is produced or received.
        """
        shares_count = len(block.shares) if hasattr(block, 'shares') else 0

        with self._lock:
            self._recent_shares.append(shares_count)
            self._total_blocks_seen += 1

            # Calculate current utilization
            utilization = self._compute_utilization()

            # Adjust base fee
            target = self._target_utilization
            if utilization > target:
                # Network is busy — increase fee
                excess = (utilization - target) / target
                self._base_fee *= (1.0 + self._adjustment_factor * excess)
            else:
                # Network is idle — decrease fee
                deficit = (target - utilization) / target
                self._base_fee *= (1.0 - self._adjustment_factor * deficit)

            # Clamp
            self._base_fee = max(self._min_fee,
                                 min(self._max_fee, self._base_fee))

            # Record snapshot
            snapshot = FeeSnapshot(
                block_index=block.index if hasattr(block, 'index') else 0,
                base_fee=self._base_fee,
                utilization=utilization,
                shares_in_block=shares_count,
                timestamp=time.time(),
            )
            self._history.append(snapshot)
            self._last_update = time.time()

            return self._base_fee

    def _compute_utilization(self) -> float:
        """Compute the current utilization from the rolling window.

        Returns a value in [0.0, ...] where 1.0 = 100% of target capacity.
        Can exceed 1.0 if the network is overloaded.
        """
        if not self._recent_shares:
            return 0.0

        avg_shares = sum(self._recent_shares) / len(self._recent_shares)
        if self._target_capacity <= 0:
            return 0.0

        return avg_shares / self._target_capacity

    def get_base_fee(self) -> float:
        """Get the current base fee per token."""
        with self._lock:
            return self._base_fee

    def get_utilization(self) -> float:
        """Get the current network utilization (0.0-1.0+)."""
        with self._lock:
            return self._compute_utilization()

    def get_fee_estimate(self, estimated_tokens: int,
                         tip: float = 0.0) -> dict:
        """Estimate the total cost for a generation session.

        Args:
            estimated_tokens: Expected number of tokens to generate.
            tip: Optional priority tip per token.

        Returns:
            Dict with base_fee, utilization, estimated_cost, suggested_tip.
        """
        with self._lock:
            base_fee = self._base_fee
            utilization = self._compute_utilization()

        estimated_cost = base_fee * estimated_tokens
        total_cost = estimated_cost + (tip * estimated_tokens)

        # Suggest a tip when the network is busy (>80% utilization)
        suggested_tip = 0.0
        if utilization > 0.8:
            # Suggest a tip proportional to how overloaded we are
            suggested_tip = base_fee * 0.1 * min(utilization - 0.8, 1.0) / 0.2

        return {
            "base_fee": base_fee,
            "utilization": utilization,
            "estimated_cost": estimated_cost,
            "total_cost": total_cost,
            "suggested_tip": suggested_tip,
            "estimated_tokens": estimated_tokens,
        }

    def get_fee_history(self, last_n: int = 20) -> list[dict]:
        """Get recent fee history (for dashboard display)."""
        with self._lock:
            snapshots = list(self._history)[-last_n:]
        return [
            {
                "block_index": s.block_index,
                "base_fee": s.base_fee,
                "utilization": s.utilization,
                "shares_in_block": s.shares_in_block,
                "timestamp": s.timestamp,
            }
            for s in snapshots
        ]

    def get_status(self) -> dict:
        """Get the oracle's full status (for API/dashboard)."""
        with self._lock:
            return {
                "base_fee": self._base_fee,
                "utilization": self._compute_utilization(),
                "target_utilization": self._target_utilization,
                "target_capacity": self._target_capacity,
                "min_fee": self._min_fee,
                "max_fee": self._max_fee,
                "adjustment_factor": self._adjustment_factor,
                "window_blocks": self._window_blocks,
                "blocks_seen": self._total_blocks_seen,
                "last_update": self._last_update,
            }
