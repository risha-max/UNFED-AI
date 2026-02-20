"""
Distributed Share-Chain — P2P mini-blockchain for compute contribution tracking.

Upgrades the centralized ShareChain (registry-only) to a fully distributed
chain where every node holds a copy, any node can propose blocks, and blocks
are gossipped directly between peers.

Key design:
  - Each node records its own compute shares locally
  - Every ~10s, the node proposes a block containing its pending shares
  - The block is gossipped to all known peers (direct P2P mesh)
  - Conflicts (two blocks at the same height) are resolved deterministically:
    the block with the lowest block_hash wins
  - New nodes catch up by syncing blocks from a peer on startup

Usage:
    from economics.distributed_chain import DistributedShareChain

    chain = DistributedShareChain(node_id="abc123", registry_address="localhost:50050")
    chain.start()                      # starts block production + gossip
    chain.add_local_share(share)       # record a compute share
    chain.receive_block(block_msg)     # handle an incoming gossipped block
    chain.sync_from_peer("localhost:50051")  # catch up on startup
"""

import os
import random
import sys
import time
import threading

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
import registry_pb2
import registry_pb2_grpc

from economics.chain_store import ChainStore
from economics.share_chain import ShareChain, ComputeShare, Block


# ---------------------------------------------------------------------------
# Proto conversion helpers
# ---------------------------------------------------------------------------

def share_to_proto(share: ComputeShare) -> inference_pb2.ShareProto:
    """Convert a ComputeShare to its protobuf representation."""
    return inference_pb2.ShareProto(
        node_id=share.node_id,
        shard_index=share.shard_index,
        session_id=share.session_id,
        activation_hash=share.activation_hash,
        tokens_processed=share.tokens_processed,
        timestamp=share.timestamp,
        share_weight=share.share_weight,
    )


def proto_to_share(proto: inference_pb2.ShareProto) -> ComputeShare:
    """Convert a ShareProto back to a ComputeShare."""
    return ComputeShare(
        node_id=proto.node_id,
        shard_index=proto.shard_index,
        session_id=proto.session_id,
        activation_hash=proto.activation_hash,
        tokens_processed=proto.tokens_processed,
        timestamp=proto.timestamp,
        share_weight=proto.share_weight if proto.share_weight > 0 else 1.0,
    )


def block_to_proto(block: Block, proposer_id: str = "") -> inference_pb2.BlockMessage:
    """Convert a Block to its protobuf representation."""
    return inference_pb2.BlockMessage(
        index=block.index,
        previous_hash=block.previous_hash,
        shares=[share_to_proto(s) for s in block.shares],
        timestamp=block.timestamp,
        block_hash=block.block_hash,
        proposer_id=proposer_id,
    )


def proto_to_block(proto: inference_pb2.BlockMessage) -> Block:
    """Convert a BlockMessage back to a Block."""
    shares = [proto_to_share(s) for s in proto.shares]
    block = Block(
        index=proto.index,
        previous_hash=proto.previous_hash,
        shares=shares,
        timestamp=proto.timestamp,
        block_hash=proto.block_hash,
    )
    return block


# ---------------------------------------------------------------------------
# DistributedShareChain
# ---------------------------------------------------------------------------

class DistributedShareChain:
    """Distributed P2P share-chain.

    Each node runs an instance. The node records its own compute shares
    locally, proposes blocks on a timer, and gossips them to peers.
    Incoming blocks from peers are validated and merged into the local chain.
    """

    def __init__(self, node_id: str, registry_address: str = None,
                 block_interval: float = 10.0, settlement_blocks: int = 6,
                 self_address: str = "",
                 db_path: str = "~/.unfed/chain.db"):
        """
        Args:
            node_id: This node's unique identifier.
            registry_address: Address of the registry for peer discovery.
            block_interval: Seconds between block proposals.
            settlement_blocks: Blocks per settlement period.
            self_address: This node's own advertised address (excluded from gossip).
            db_path: SQLite database path for chain persistence.
        """
        self.node_id = node_id
        self.registry_address = registry_address or config.REGISTRY_ADDRESS
        self.block_interval = block_interval
        self.self_address = self_address

        self._store = ChainStore(db_path)
        self.chain = ShareChain(
            block_interval=block_interval,
            settlement_blocks=settlement_blocks,
            store=self._store,
        )

        # Known peer addresses for gossip (updated periodically from registry)
        self._peers: list[str] = []
        self._peers_lock = threading.Lock()

        # gRPC stub cache for gossip
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}

        self._running = False

    # --- Local share recording ---

    def add_local_share(self, share: ComputeShare):
        """Record a compute share from this node's own forward passes."""
        self.chain.add_share(share)

    # --- Block proposal ---

    def propose_block(self) -> Block | None:
        """Produce a block from pending shares and gossip it to all peers.

        Returns the produced block, or None if there were no pending shares.
        No empty blocks are created.
        """
        block = self.chain.produce_block()
        if block is None:
            return None

        print(f"[Chain {self.node_id[:8]}] Proposed block #{block.index}: "
              f"{len(block.shares)} shares (hash={block.block_hash[:12]}...)")
        self._gossip_block(block)
        return block

    # --- Receiving gossipped blocks ---

    def receive_block(self, block_msg: inference_pb2.BlockMessage) -> tuple[bool, str]:
        """Handle an incoming gossipped block.

        Validates and merges it into the local chain. If accepted, re-gossips
        to peers (flood fill).

        Returns (accepted, reason).
        """
        block = proto_to_block(block_msg)
        accepted, reason = self.chain.receive_external_block(block)

        if accepted:
            print(f"[Chain {self.node_id[:8]}] Accepted block #{block.index} "
                  f"from {block_msg.proposer_id[:8]}... ({reason})")
            # Re-gossip to other peers (flood fill)
            self._gossip_block(block, exclude_proposer=block_msg.proposer_id)
        elif reason == "ahead":
            # We're behind — trigger a sync from the proposer
            # Find proposer address from peers list
            print(f"[Chain {self.node_id[:8]}] Block #{block.index} is ahead "
                  f"of our chain (height={self.chain.get_tip_height()}). "
                  f"Need sync.")

        return accepted, reason

    # --- Peer sync (catch-up on startup or when behind) ---

    def sync_from_peer(self, peer_address: str) -> int:
        """Sync blocks from a peer to catch up.

        Requests all blocks from our current height onward and applies them.
        Returns the number of blocks synced.
        """
        my_height = self.chain.get_tip_height()
        try:
            stub = self._get_stub(peer_address)
            resp = stub.GetBlocks(
                inference_pb2.GetBlocksRequest(from_height=my_height + 1),
                timeout=30,
            )

            synced = 0
            for block_msg in resp.blocks:
                block = proto_to_block(block_msg)
                accepted, reason = self.chain.receive_external_block(block)
                if accepted:
                    synced += 1

            if synced > 0:
                print(f"[Chain {self.node_id[:8]}] Synced {synced} block(s) "
                      f"from {peer_address} "
                      f"(height: {my_height} → {self.chain.get_tip_height()})")
            return synced

        except grpc.RpcError as e:
            print(f"[Chain {self.node_id[:8]}] Sync from {peer_address} "
                  f"failed: {e.code()}")
            return 0

    # --- Gossip ---

    def _gossip_block(self, block: Block, exclude_proposer: str = ""):
        """Push a block to all known peers."""
        msg = block_to_proto(block, proposer_id=self.node_id)

        with self._peers_lock:
            peers = list(self._peers)

        for peer in peers:
            # Don't gossip to ourselves or to the original proposer
            if peer == self.self_address or peer == exclude_proposer:
                continue
            try:
                stub = self._get_stub(peer)
                stub.GossipBlock(msg, timeout=5)
            except grpc.RpcError:
                pass  # peer unreachable, skip

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        """Get or create a gRPC stub for a peer."""
        if address not in self._stubs:
            channel = grpc.insecure_channel(address, options=config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    # --- Peer discovery ---

    def _refresh_peers(self):
        """Fetch the list of gossip peer addresses.

        Includes all compute nodes (from registry discovery) plus the
        registry itself (which is also a gossip peer).
        """
        try:
            channel = grpc.insecure_channel(self.registry_address)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(
                registry_pb2.DiscoverRequest(model_id=""),
                timeout=10,
            )
            addresses = [n.address for n in resp.nodes
                         if n.node_type == "compute"]
            # Include the registry as a gossip peer
            if self.registry_address and self.registry_address not in addresses:
                addresses.append(self.registry_address)
            with self._peers_lock:
                self._peers = addresses
            channel.close()
        except grpc.RpcError:
            pass  # registry unreachable, keep old peers

    # --- Background loops ---

    def start(self):
        """Start block production, gossip, and peer refresh loops."""
        self._running = True

        # Initial peer discovery
        self._refresh_peers()

        # Sync from a peer to catch up
        with self._peers_lock:
            peers = list(self._peers)
        for peer in peers:
            synced = self.sync_from_peer(peer)
            if synced > 0:
                break  # synced from one peer is enough

        # Block production thread
        t_blocks = threading.Thread(target=self._block_loop, daemon=True)
        t_blocks.start()

        # Peer refresh thread (every 30s)
        t_peers = threading.Thread(target=self._peer_refresh_loop, daemon=True)
        t_peers.start()

    def _block_loop(self):
        """Propose blocks only when there are pending shares.

        Uses a collection window: once shares appear, waits a few seconds
        to accumulate more shares (e.g., from multiple tokens in a single
        generation session) before proposing a block.  A random jitter
        (0-3s) staggers proposals so different nodes don't all propose at
        the exact same moment.

        Each node only records its OWN compute shares, so a block proposed
        by shard-0's node will only contain shard-0 shares.  This is by
        design — like how a Bitcoin miner's block contains their coinbase.
        The full picture emerges when the dashboard aggregates blocks from
        all nodes across the chain.

        No empty blocks are ever created — matching real blockchain behavior.
        """
        while self._running:
            time.sleep(1.0)
            if self._running and self.chain.has_pending_shares():
                # Collection window — let more shares accumulate.
                # During generation, a node processes many tokens in a
                # burst.  Waiting a few seconds captures more of the burst
                # in a single block rather than one-share-per-block.
                time.sleep(4.0)
                # Stagger across nodes
                time.sleep(random.uniform(0, 3.0))
                if self._running and self.chain.has_pending_shares():
                    self.propose_block()

    def _peer_refresh_loop(self):
        """Periodically refresh the peer list from the registry."""
        while self._running:
            time.sleep(30)
            if self._running:
                self._refresh_peers()

    def stop(self):
        """Stop all background loops and close the database."""
        self._running = False
        self.chain.stop()
        self._store.close()

    # --- Status ---

    def get_status(self) -> dict:
        """Return chain status for debugging/monitoring."""
        with self._peers_lock:
            peer_count = len(self._peers)
        info = self.chain.get_chain_info()
        info["node_id"] = self.node_id[:8]
        info["peer_count"] = peer_count
        return info
