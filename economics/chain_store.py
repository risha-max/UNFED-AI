"""
Chain Store — SQLite persistence layer for the mini-chain.

Stores blocks, shares, and settlements in a single SQLite database file,
similar to how Monero's monerod uses LMDB for blockchain storage.

The database lives at ~/.unfed/chain.db by default (configurable).
Blocks are stored with gzip-compressed JSON in a raw_data BLOB for compact
storage, plus indexed columns for efficient queries.

Usage:
    from economics.chain_store import ChainStore

    store = ChainStore("~/.unfed/chain.db")
    store.save_block(block)
    blocks = store.load_blocks(from_height=10, limit=50)
    totals = store.get_node_totals()
    store.close()
"""

import gzip
import json
import logging
import os
import sqlite3
import threading

logger = logging.getLogger("unfed.chain_store")
from typing import Optional

from economics.share_chain import Block, ComputeShare, SettlementSummary


class ChainStore:
    """SQLite-backed persistent storage for the share-chain.

    Thread-safe: uses a per-thread connection pattern via check_same_thread=False
    and an internal lock for write operations.
    """

    def __init__(self, db_path: str = "~/.unfed/chain.db"):
        self.db_path = os.path.expanduser(db_path)

        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._prune_count = 0
        self._create_tables()

    def _create_tables(self):
        """Create tables if they don't exist."""
        with self._lock:
            c = self._conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    block_index INTEGER PRIMARY KEY,
                    previous_hash TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    block_hash TEXT NOT NULL,
                    proposer_id TEXT DEFAULT '',
                    raw_data BLOB NOT NULL
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS shares (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_index INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    shard_index INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    activation_hash TEXT NOT NULL,
                    tokens_processed INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (block_index) REFERENCES blocks(block_index)
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS settlements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_start REAL NOT NULL,
                    period_end REAL NOT NULL,
                    block_start INTEGER NOT NULL,
                    block_end INTEGER NOT NULL,
                    total_shares INTEGER NOT NULL,
                    settlement_hash TEXT NOT NULL,
                    node_shares_json TEXT NOT NULL
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Composite: covers prune (block_index <), per-node-in-range, and GROUP BY node_id
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_shares_block_node
                ON shares(block_index, node_id)
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_shares_node
                ON shares(node_id)
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_shares_session
                ON shares(session_id)
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_settlements_block_range
                ON settlements(block_start, block_end)
            """)

            self._conn.commit()

    # --- Block operations ---

    def save_block(self, block: Block, proposer_id: str = ""):
        """Persist a block and its shares to SQLite.

        The full block is stored as gzip-compressed JSON in raw_data,
        and individual shares are also inserted into the shares table
        for indexed queries.
        """
        raw_data = gzip.compress(
            json.dumps(block.to_dict()).encode("utf-8")
        )

        with self._lock:
            c = self._conn.cursor()
            try:
                c.execute(
                    "INSERT OR REPLACE INTO blocks "
                    "(block_index, previous_hash, timestamp, block_hash, proposer_id, raw_data) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (block.index, block.previous_hash, block.timestamp,
                     block.block_hash, proposer_id, raw_data),
                )

                # Batch-insert shares for indexed queries
                if block.shares:
                    c.executemany(
                        "INSERT INTO shares "
                        "(block_index, node_id, shard_index, session_id, "
                        "activation_hash, tokens_processed, timestamp) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            (block.index, s.node_id, s.shard_index,
                             s.session_id, s.activation_hash,
                             s.tokens_processed, s.timestamp)
                            for s in block.shares
                        ],
                    )

                self._conn.commit()
            except sqlite3.IntegrityError:
                # Block already exists (e.g., re-save after conflict resolution)
                self._conn.rollback()

    def load_blocks(self, from_height: int = 0, limit: int = 100) -> list[Block]:
        """Load blocks from the database starting at from_height."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT raw_data FROM blocks WHERE block_index >= ? "
                "ORDER BY block_index ASC LIMIT ?",
                (from_height, limit),
            )
            rows = c.fetchall()

        blocks = []
        for (raw_data,) in rows:
            block_dict = json.loads(gzip.decompress(raw_data).decode("utf-8"))
            blocks.append(Block.from_dict(block_dict))
        return blocks

    def load_all_blocks(self) -> list[Block]:
        """Load the entire chain from the database."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT raw_data FROM blocks ORDER BY block_index ASC"
            )
            rows = c.fetchall()

        blocks = []
        for (raw_data,) in rows:
            block_dict = json.loads(gzip.decompress(raw_data).decode("utf-8"))
            blocks.append(Block.from_dict(block_dict))
        return blocks

    def get_tip(self) -> Optional[Block]:
        """Return the latest block, or None if the chain is empty."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT raw_data FROM blocks ORDER BY block_index DESC LIMIT 1"
            )
            row = c.fetchone()

        if row is None:
            return None
        block_dict = json.loads(gzip.decompress(row[0]).decode("utf-8"))
        return Block.from_dict(block_dict)

    def get_height(self) -> int:
        """Return the current chain height (tip block index), or -1 if empty."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT MAX(block_index) FROM blocks")
            row = c.fetchone()
        if row is None or row[0] is None:
            return -1
        return row[0]

    def has_block(self, block_index: int) -> bool:
        """Check if a block exists at the given height."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT 1 FROM blocks WHERE block_index = ?", (block_index,)
            )
            return c.fetchone() is not None

    # --- Settlement operations ---

    def save_settlement(self, settlement: SettlementSummary):
        """Persist a settlement summary."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "INSERT INTO settlements "
                "(period_start, period_end, block_start, block_end, "
                "total_shares, settlement_hash, node_shares_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (settlement.period_start, settlement.period_end,
                 settlement.block_range[0], settlement.block_range[1],
                 settlement.total_shares, settlement.settlement_hash,
                 json.dumps(settlement.node_shares)),
            )
            self._conn.commit()

    def load_settlements(self) -> list[SettlementSummary]:
        """Load all settlements from the database."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT period_start, period_end, block_start, block_end, "
                "total_shares, settlement_hash, node_shares_json "
                "FROM settlements ORDER BY id ASC"
            )
            rows = c.fetchall()

        settlements = []
        for row in rows:
            # Columns: period_start, period_end, block_start, block_end,
            #          total_shares, settlement_hash, node_shares_json
            settlements.append(SettlementSummary(
                period_start=row[0],
                period_end=row[1],
                block_range=(row[2], row[3]),
                node_shares=json.loads(row[6]),
                total_shares=row[4],
                settlement_hash=row[5],
            ))
        return settlements

    # --- Aggregate queries ---

    def get_node_totals(self, from_block: int = 0) -> dict[str, int]:
        """Get total shares per node, optionally scoped to blocks >= from_block."""
        with self._lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT node_id, SUM(tokens_processed) "
                "FROM shares WHERE block_index >= ? GROUP BY node_id",
                (from_block,),
            )
            return {row[0]: row[1] for row in c.fetchall()}

    def get_total_shares(self) -> int:
        """Get total number of shares across all blocks."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT COUNT(*) FROM shares")
            row = c.fetchone()
            return row[0] if row else 0

    def get_block_count(self) -> int:
        """Get total number of blocks."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT COUNT(*) FROM blocks")
            row = c.fetchone()
            return row[0] if row else 0

    # --- Meta operations ---

    def set_meta(self, key: str, value: str):
        """Set a metadata key-value pair."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        """Get a metadata value by key."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = c.fetchone()
        return row[0] if row else default

    # --- Pruning ---

    def prune(self, keep_last_n: int):
        """Delete blocks older than the last N, keeping only recent history.

        This is like Monero's blockchain pruning — reduces disk usage while
        keeping enough history for settlements and verification.
        """
        if keep_last_n <= 0:
            return

        height = self.get_height()
        if height < 0:
            return

        cutoff = height - keep_last_n
        if cutoff <= 0:
            return

        with self._lock:
            c = self._conn.cursor()
            c.execute("DELETE FROM shares WHERE block_index < ?", (cutoff,))
            c.execute("DELETE FROM blocks WHERE block_index < ?", (cutoff,))
            self._conn.commit()
            self._prune_count += 1
            # VACUUM rebuilds the entire file — only do it periodically
            if self._prune_count % 50 == 0:
                self._conn.execute("VACUUM")

        print(f"[ChainStore] Pruned blocks below height {cutoff} "
              f"(kept last {keep_last_n})")

    # --- Lifecycle ---

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.debug("ChainStore cleanup error: %s", e)
