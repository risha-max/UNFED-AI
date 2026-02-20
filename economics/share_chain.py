"""
Share-Chain — P2Pool-style side-chain tracking compute contributions.

A lightweight mini-blockchain that:
  1. Records every forward pass (compute share) from every node
  2. Tallies contributions per node per time period
  3. Produces periodic settlement summaries for the main chain

Block structure:
  - ~10 second block time (configurable)
  - Each block contains a list of compute shares from that period
  - Blocks form a chain (each references the previous block hash)
  - Any node can produce a block (no mining, just aggregation)

A "share" represents one unit of compute work:
  - node_id: which node did the work
  - shard_index: which shard was computed
  - session_id: which session (for deduplication)
  - activation_hash: hash of the output (proves the work was done)
  - timestamp: when the work was done

Settlement:
  - Every N blocks, a settlement summary is produced
  - The summary maps node_id -> total_shares for the period
  - This feeds into the payment system (optimistic rollup)
"""

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional


# --- Compute Share ---

@dataclass
class ComputeShare:
    """One unit of compute work by a node.

    share_weight determines how much this share counts at settlement:
      - 1.0 for compute and MPC nodes (full GPU work)
      - GUARD_FEE_RATIO (e.g., 0.05) for guard relay work
      - 0.0 for daemon nodes (unpaid infrastructure)
    """
    node_id: str
    shard_index: int
    session_id: str
    activation_hash: str
    tokens_processed: int
    timestamp: float = field(default_factory=time.time)
    share_weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ComputeShare':
        return cls(
            node_id=d["node_id"],
            shard_index=d["shard_index"],
            session_id=d["session_id"],
            activation_hash=d["activation_hash"],
            tokens_processed=d["tokens_processed"],
            timestamp=d.get("timestamp", 0.0),
            share_weight=d.get("share_weight", 1.0),
        )

    def hash(self) -> str:
        data = f"{self.node_id}:{self.shard_index}:{self.session_id}:" \
               f"{self.activation_hash}:{self.tokens_processed}:{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()


# --- Block ---

@dataclass
class Block:
    """A share-chain block containing compute shares."""
    index: int
    previous_hash: str
    shares: list[ComputeShare]
    timestamp: float = field(default_factory=time.time)
    block_hash: str = ""

    def compute_hash(self) -> str:
        """Compute the block hash from its contents."""
        data = f"{self.index}:{self.previous_hash}:{self.timestamp}:"
        for share in self.shares:
            data += share.hash()
        return hashlib.sha256(data.encode()).hexdigest()

    def finalize(self):
        """Compute and set the block hash."""
        self.block_hash = self.compute_hash()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "shares": [s.to_dict() for s in self.shares],
            "timestamp": self.timestamp,
            "block_hash": self.block_hash,
            "num_shares": len(self.shares),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Block':
        shares = [ComputeShare.from_dict(s) for s in d.get("shares", [])]
        return cls(
            index=d["index"],
            previous_hash=d["previous_hash"],
            shares=shares,
            timestamp=d.get("timestamp", 0.0),
            block_hash=d.get("block_hash", ""),
        )


# --- Settlement Summary ---

@dataclass
class SettlementSummary:
    """
    Periodic settlement: maps node_id -> weighted share sum for a time period.
    This feeds into the payment system.

    node_shares contains weighted sums (float), not raw counts.
    Compute shares contribute 1.0 each, guard shares contribute
    GUARD_FEE_RATIO (e.g., 0.05) each. This means the settlement
    automatically pays different roles proportionally.
    """
    period_start: float
    period_end: float
    block_range: tuple[int, int]  # (first_block_index, last_block_index)
    node_shares: dict[str, float]   # node_id -> weighted share sum
    total_shares: float
    total_tokens: int = 0           # total tokens processed in this period
    settlement_hash: str = ""

    def compute_hash(self) -> str:
        data = f"{self.period_start}:{self.period_end}:{self.block_range}:"
        for node_id in sorted(self.node_shares.keys()):
            data += f"{node_id}={self.node_shares[node_id]}:"
        return hashlib.sha256(data.encode()).hexdigest()

    def finalize(self):
        self.settlement_hash = self.compute_hash()

    def to_dict(self) -> dict:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "block_range": list(self.block_range),
            "node_shares": self.node_shares,
            "total_shares": self.total_shares,
            "total_tokens": self.total_tokens,
            "settlement_hash": self.settlement_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SettlementSummary':
        br = d.get("block_range", [0, 0])
        # Handle both int and float for backward compatibility
        raw_shares = d.get("node_shares", {})
        node_shares = {k: float(v) for k, v in raw_shares.items()}
        return cls(
            period_start=d["period_start"],
            period_end=d["period_end"],
            block_range=(br[0], br[1]),
            node_shares=node_shares,
            total_shares=float(d.get("total_shares", 0)),
            total_tokens=int(d.get("total_tokens", 0)),
            settlement_hash=d.get("settlement_hash", ""),
        )


# --- Share Chain ---

class ShareChain:
    """
    Share chain that tracks compute contributions.

    Supports optional SQLite persistence via ChainStore. When a store is
    provided, blocks are persisted on creation/acceptance and the chain
    is restored from disk on startup — surviving process restarts.

    Without a store, the chain is purely in-memory (legacy behavior).
    """

    def __init__(self, block_interval: float = 10.0,
                 settlement_blocks: int = 6,
                 store=None):
        """
        Args:
            block_interval: Seconds between blocks (default 10s)
            settlement_blocks: Number of blocks per settlement period
            store: Optional ChainStore for SQLite persistence
        """
        self.block_interval = block_interval
        self.settlement_blocks = settlement_blocks
        self._store = store  # ChainStore or None

        self._chain: list[Block] = []
        self._pending_shares: list[ComputeShare] = []
        self._settlements: list[SettlementSummary] = []
        self._lock = threading.Lock()
        self._running = False

        # Try to load chain from disk (if store is provided and has data)
        loaded = False
        if self._store is not None:
            loaded = self._load_from_store()

        if not loaded:
            # Create genesis block (fixed timestamp for deterministic hash —
            # all nodes must start from the same genesis for gossip to work)
            genesis = Block(
                index=0,
                previous_hash="0" * 64,
                shares=[],
                timestamp=0.0,
            )
            genesis.finalize()
            self._chain.append(genesis)
            # Persist genesis if we have a store
            if self._store is not None:
                self._store.save_block(genesis)

    def _load_from_store(self) -> bool:
        """Load the chain from SQLite. Returns True if blocks were loaded."""
        try:
            blocks = self._store.load_all_blocks()
            if blocks:
                self._chain = blocks
                # Also load settlements
                settlements = self._store.load_settlements()
                self._settlements = settlements
                print(f"[ShareChain] Loaded {len(blocks)} block(s) from disk "
                      f"(height={blocks[-1].index}, "
                      f"{len(settlements)} settlement(s))")
                return True
        except Exception as e:
            print(f"[ShareChain] Failed to load from store: {e}")
        return False

    @property
    def height(self) -> int:
        return len(self._chain)

    @property
    def latest_block(self) -> Block:
        return self._chain[-1]

    def add_share(self, share: ComputeShare):
        """Add a compute share to the pending pool."""
        with self._lock:
            self._pending_shares.append(share)

    def add_shares(self, shares: list[ComputeShare]):
        """Add multiple compute shares."""
        with self._lock:
            self._pending_shares.extend(shares)

    def has_pending_shares(self) -> bool:
        """Check if there are pending shares waiting to be committed."""
        with self._lock:
            return len(self._pending_shares) > 0

    def produce_block(self) -> Block | None:
        """Produce a new block from pending shares.

        Returns None if there are no pending shares — no empty blocks
        are ever created (matching real blockchain behavior).
        Persists to SQLite if a store is configured.
        """
        with self._lock:
            if not self._pending_shares:
                return None
            shares = self._pending_shares
            self._pending_shares = []

        prev = self._chain[-1]
        block = Block(
            index=prev.index + 1,
            previous_hash=prev.block_hash,
            shares=shares,
        )
        block.finalize()

        with self._lock:
            self._chain.append(block)

        # Persist to disk
        if self._store is not None:
            self._store.save_block(block)

        # Check if it's time for a settlement
        if block.index % self.settlement_blocks == 0 and block.index > 0:
            self._produce_settlement()

        return block

    def receive_external_block(self, block: Block) -> tuple[bool, str]:
        """Accept a block produced by another node (via gossip).

        Validates the block and handles conflicts at the same height:
        - If the block extends the chain (correct previous_hash, next index): accept.
        - If the block is at the same height as the tip: keep the one with the
          lowest block_hash (deterministic tie-break). Orphaned shares are re-queued.
        - If the block is old or invalid: reject.

        Returns (accepted: bool, reason: str).
        """
        with self._lock:
            tip = self._chain[-1]

            # Validate hash
            expected_hash = block.compute_hash()
            if block.block_hash != expected_hash:
                return False, "invalid_hash"

            # Block extends the chain normally
            if block.index == tip.index + 1 and block.previous_hash == tip.block_hash:
                self._chain.append(block)
                # Persist to disk
                if self._store is not None:
                    self._store.save_block(block)
                # Check for settlement
                if block.index % self.settlement_blocks == 0 and block.index > 0:
                    self._produce_settlement_locked()
                return True, "accepted"

            # Block is at the same height as the tip (conflict)
            if block.index == tip.index and block.previous_hash == tip.previous_hash:
                if block.block_hash < tip.block_hash:
                    # New block wins — orphan the current tip
                    orphaned = self._chain.pop()
                    self._chain.append(block)
                    # Persist replacement to disk
                    if self._store is not None:
                        self._store.save_block(block)
                    # Re-queue orphaned block's shares
                    self._pending_shares.extend(orphaned.shares)
                    return True, "replaced_tip"
                else:
                    return False, "higher_hash"

            # Block is behind our chain
            if block.index <= tip.index:
                return False, "stale"

            # Block is ahead (we're behind — need sync)
            if block.index > tip.index + 1:
                return False, "ahead"

            return False, "unknown"

    def _produce_settlement_locked(self):
        """Produce settlement while already holding the lock."""
        end_idx = len(self._chain) - 1
        start_idx = max(1, end_idx - self.settlement_blocks + 1)

        node_shares: dict[str, float] = {}
        total = 0.0
        earliest = float('inf')
        latest = 0.0

        for i in range(start_idx, end_idx + 1):
            block = self._chain[i]
            for share in block.shares:
                weight = getattr(share, 'share_weight', 1.0)
                node_shares[share.node_id] = node_shares.get(share.node_id, 0.0) + weight
                total += weight
                earliest = min(earliest, share.timestamp)
                latest = max(latest, share.timestamp)

        if earliest == float('inf'):
            earliest = time.time()
        if latest == 0.0:
            latest = time.time()

        settlement = SettlementSummary(
            period_start=earliest,
            period_end=latest,
            block_range=(start_idx, end_idx),
            node_shares=node_shares,
            total_shares=total,
        )
        settlement.finalize()
        self._settlements.append(settlement)
        # Persist settlement
        if self._store is not None:
            self._store.save_settlement(settlement)

    def requeue_shares(self, shares: list['ComputeShare']):
        """Put shares back into the pending pool (e.g., after a block conflict)."""
        with self._lock:
            self._pending_shares.extend(shares)

    def get_blocks_from(self, from_height: int) -> list[Block]:
        """Return all blocks from the given height onward (for sync)."""
        with self._lock:
            return [b for b in self._chain if b.index >= from_height]

    def get_tip_height(self) -> int:
        """Return the current chain tip height."""
        with self._lock:
            return self._chain[-1].index

    def _produce_settlement(self):
        """Produce a settlement summary for the last N blocks."""
        end_idx = len(self._chain) - 1
        start_idx = max(1, end_idx - self.settlement_blocks + 1)

        node_shares: dict[str, float] = {}
        total = 0.0
        total_tokens = 0
        earliest = float('inf')
        latest = 0.0

        for i in range(start_idx, end_idx + 1):
            block = self._chain[i]
            for share in block.shares:
                weight = getattr(share, 'share_weight', 1.0)
                node_shares[share.node_id] = node_shares.get(share.node_id, 0.0) + weight
                total += weight
                total_tokens += getattr(share, 'tokens_processed', 0)
                earliest = min(earliest, share.timestamp)
                latest = max(latest, share.timestamp)

        if earliest == float('inf'):
            earliest = time.time()
        if latest == 0.0:
            latest = time.time()

        settlement = SettlementSummary(
            period_start=earliest,
            period_end=latest,
            block_range=(start_idx, end_idx),
            node_shares=node_shares,
            total_shares=total,
            total_tokens=total_tokens,
        )
        settlement.finalize()

        with self._lock:
            self._settlements.append(settlement)

        # Persist settlement
        if self._store is not None:
            self._store.save_settlement(settlement)

        print(f"[ShareChain] Settlement #{len(self._settlements)}: "
              f"blocks {start_idx}-{end_idx}, {total:.2f} weighted shares, "
              f"{len(node_shares)} nodes")
        for nid, count in sorted(node_shares.items(), key=lambda x: -x[1]):
            print(f"  {nid[:8]}...: {count:.4f} weighted shares")

        return settlement

    def start_block_production(self):
        """Start automatic block production in a background thread."""
        self._running = True
        thread = threading.Thread(target=self._block_loop, daemon=True)
        thread.start()

    def _block_loop(self):
        """Produce blocks at regular intervals."""
        while self._running:
            time.sleep(self.block_interval)
            if self._running:
                block = self.produce_block()
                if block and block.shares:
                    print(f"[ShareChain] Block #{block.index}: "
                          f"{len(block.shares)} shares")

    def stop(self):
        self._running = False

    def get_node_totals(self) -> dict[str, float]:
        """Get total weighted shares per node across all blocks."""
        totals: dict[str, float] = {}
        for block in self._chain:
            for share in block.shares:
                weight = getattr(share, 'share_weight', 1.0)
                totals[share.node_id] = totals.get(share.node_id, 0.0) + weight
        return totals

    def get_chain_info(self) -> dict:
        return {
            "height": self.height,
            "latest_block_hash": self.latest_block.block_hash[:16],
            "total_shares": sum(len(b.shares) for b in self._chain),
            "settlements": len(self._settlements),
            "node_totals": self.get_node_totals(),
        }

    def get_settlements(self) -> list[SettlementSummary]:
        with self._lock:
            return list(self._settlements)
