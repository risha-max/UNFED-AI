"""
Daemon Node — Standalone chain daemon (like Monero's monerod).

A lightweight process that holds the full mini-chain in SQLite, produces
blocks from submitted shares, gossips blocks with other daemons, and
serves chain data to light clients (dashboard, CLI).

No GPU required. No model weights loaded. Just chain management.

Usage:
    python -m network.daemon_node --port 50070 --registry localhost:50050

Roles:
  - Accepts SubmitShares RPCs from compute nodes
  - Produces blocks from pending shares on a timer
  - Persists blocks to SQLite (~/.unfed/chain.db)
  - Gossips blocks to other daemon peers (P2P mesh)
  - Serves GetBlocks / SubscribeBlocks to light clients
  - Registers with the registry as node_type="daemon"
"""

import argparse
import logging
import os
import queue
import random
import signal
import sys
import threading
import time

logger = logging.getLogger("unfed.daemon")
from concurrent import futures

import grpc

# Fix imports for the project layout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import config
import inference_pb2
import inference_pb2_grpc
import registry_pb2
import registry_pb2_grpc

from economics.chain_store import ChainStore
from economics.share_chain import ShareChain, ComputeShare, Block
from economics.distributed_chain import (
    share_to_proto, proto_to_share, block_to_proto, proto_to_block,
)


# ---------------------------------------------------------------------------
# DaemonServicer — gRPC service
# ---------------------------------------------------------------------------

class DaemonServicer(inference_pb2_grpc.InferenceNodeServicer):
    """gRPC servicer for the chain daemon.

    Implements:
      - SubmitShares: compute nodes push shares here after each forward pass
      - SubscribeBlocks: light clients receive real-time block stream
      - GetBlocks: full sync (request blocks from a height)
      - GossipBlock: daemon-to-daemon block propagation
    """

    def __init__(self, chain: ShareChain, node_id: str, fee_oracle=None):
        self.chain = chain
        self.node_id = node_id
        self.fee_oracle = fee_oracle

        # Subscribers for SubscribeBlocks streaming RPC
        self._subscribers: list[queue.Queue] = []
        self._subscribers_lock = threading.Lock()

    # --- SubmitShares (compute nodes → daemon) ---

    def SubmitShares(self, request, context):
        """Accept compute shares from a compute/MPC node."""
        shares = [proto_to_share(s) for s in request.shares]
        self.chain.add_shares(shares)

        submitter = request.submitter_id[:8] if request.submitter_id else "unknown"
        print(f"[Daemon] Received {len(shares)} share(s) from {submitter}...")

        return inference_pb2.SubmitSharesResponse(
            accepted=len(shares),
            pending_pool_size=len(self.chain._pending_shares),
        )

    # --- SubscribeBlocks (light-wallet streaming) ---

    def SubscribeBlocks(self, request, context):
        """Stream blocks to a light client.

        First sends historical blocks from from_height, then keeps the
        stream open and pushes new blocks as they are produced/received.
        """
        from_height = request.from_height

        # Send historical blocks first
        historical = self.chain.get_blocks_from(from_height)
        for block in historical:
            if not context.is_active():
                return
            yield block_to_proto(block, proposer_id=self.node_id)

        # Subscribe for new blocks
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._subscribers_lock:
            self._subscribers.append(q)

        try:
            while context.is_active():
                try:
                    block_msg = q.get(timeout=5.0)
                    yield block_msg
                except queue.Empty:
                    continue  # keep the stream alive
        finally:
            with self._subscribers_lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    # --- GetBlocks (full sync) ---

    def GetBlocks(self, request, context):
        """Return blocks from a given height (for sync/catch-up)."""
        blocks = self.chain.get_blocks_from(request.from_height)
        block_msgs = [block_to_proto(b, proposer_id=self.node_id)
                      for b in blocks]
        return inference_pb2.GetBlocksResponse(
            blocks=block_msgs,
            chain_height=self.chain.get_tip_height(),
        )

    # --- GossipBlock (daemon-to-daemon P2P) ---

    def GossipBlock(self, request, context):
        """Receive a gossipped block from another daemon."""
        block = proto_to_block(request)
        accepted, reason = self.chain.receive_external_block(block)

        if accepted:
            print(f"[Daemon] Accepted block #{block.index} from "
                  f"{request.proposer_id[:8]}... ({reason})")
            # Update fee oracle with the gossipped block
            if self.fee_oracle is not None:
                self.fee_oracle.update(block)
            # Notify subscribers
            self._notify_subscribers(request)
            # Re-gossip to other peers (handled by the gossip manager)

        return inference_pb2.GossipResponse(
            accepted=accepted,
            reason=reason,
            current_height=self.chain.get_tip_height(),
        )

    # --- GetFeeEstimate (fee oracle) ---

    def GetFeeEstimate(self, request, context):
        """Return the current fee estimate from the oracle."""
        if self.fee_oracle is None:
            return inference_pb2.FeeEstimateResponse(
                base_fee=0.001,
                utilization=0.0,
                estimated_cost=0.001 * request.estimated_tokens,
                suggested_tip=0.0,
            )

        estimate = self.fee_oracle.get_fee_estimate(
            estimated_tokens=request.estimated_tokens,
        )
        return inference_pb2.FeeEstimateResponse(
            base_fee=estimate["base_fee"],
            utilization=estimate["utilization"],
            estimated_cost=estimate["estimated_cost"],
            suggested_tip=estimate["suggested_tip"],
        )

    # --- Forward / Commit stubs (daemon doesn't do inference) ---

    def Forward(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Daemon nodes do not serve inference")
        return inference_pb2.ForwardResponse()

    def Commit(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Daemon nodes do not serve inference")
        return inference_pb2.CommitResponse()

    # --- Internal ---

    def _notify_subscribers(self, block_msg):
        """Push a block message to all SubscribeBlocks streams."""
        with self._subscribers_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(block_msg)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def notify_new_block(self, block: Block):
        """Called by the block production loop when a new block is produced."""
        # Update fee oracle with the new block
        if self.fee_oracle is not None:
            new_fee = self.fee_oracle.update(block)
            if len(block.shares) > 0:
                print(f"[Daemon] Fee oracle updated: base_fee={new_fee:.6f}, "
                      f"utilization={self.fee_oracle.get_utilization():.2%}")
        msg = block_to_proto(block, proposer_id=self.node_id)
        self._notify_subscribers(msg)


# ---------------------------------------------------------------------------
# GossipManager — P2P gossip with other daemons
# ---------------------------------------------------------------------------

class GossipManager:
    """Manages P2P gossip between daemon peers."""

    def __init__(self, node_id: str, chain: ShareChain,
                 servicer: DaemonServicer,
                 registry_address: str,
                 self_address: str):
        self.node_id = node_id
        self.chain = chain
        self.servicer = servicer
        self.registry_address = registry_address
        self.self_address = self_address

        self._peers: list[str] = []
        self._peers_lock = threading.Lock()
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._running = False

    def start(self):
        """Start gossip and peer refresh loops."""
        self._running = True
        self._refresh_peers()

        # Sync from a peer on startup
        with self._peers_lock:
            peers = list(self._peers)
        for peer in peers:
            synced = self._sync_from_peer(peer)
            if synced > 0:
                break

        # Block production thread
        t_blocks = threading.Thread(target=self._block_loop, daemon=True)
        t_blocks.start()

        # Peer refresh thread
        t_peers = threading.Thread(target=self._peer_loop, daemon=True)
        t_peers.start()

    def stop(self):
        self._running = False

    def _block_loop(self):
        """Produce blocks when pending shares exist.

        The daemon aggregates shares from ALL compute nodes (they submit
        via SubmitShares), so a single block can contain shares from
        multiple nodes — unlike the old per-node block production.
        """
        while self._running:
            time.sleep(1.0)
            if self._running and self.chain.has_pending_shares():
                # Collection window: wait for more shares to arrive
                time.sleep(4.0)
                time.sleep(random.uniform(0, 2.0))
                if self._running and self.chain.has_pending_shares():
                    block = self.chain.produce_block()
                    if block is not None:
                        print(f"[Daemon] Produced block #{block.index}: "
                              f"{len(block.shares)} shares "
                              f"(hash={block.block_hash[:12]}...)")
                        # Notify subscribers
                        self.servicer.notify_new_block(block)
                        # Gossip to peers
                        self._gossip_block(block)

    def _peer_loop(self):
        """Periodically refresh the peer list."""
        while self._running:
            time.sleep(30)
            if self._running:
                self._refresh_peers()

    def _refresh_peers(self):
        """Discover other daemon nodes from the registry."""
        try:
            channel = grpc.insecure_channel(
                self.registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(
                registry_pb2.DiscoverRequest(model_id=""),
                timeout=10,
            )
            # Find all daemon nodes (excluding ourselves)
            addresses = [n.address for n in resp.nodes
                         if n.node_type == "daemon"
                         and n.address != self.self_address]
            with self._peers_lock:
                self._peers = addresses
            channel.close()
        except grpc.RpcError as e:
            logger.debug("Peer discovery from registry failed: %s", e)

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        if address not in self._stubs:
            channel = grpc.insecure_channel(address, options=config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def _gossip_block(self, block: Block, exclude: str = ""):
        """Push a block to all daemon peers."""
        msg = block_to_proto(block, proposer_id=self.node_id)
        with self._peers_lock:
            peers = list(self._peers)
        for peer in peers:
            if peer == self.self_address or peer == exclude:
                continue
            try:
                self._get_stub(peer).GossipBlock(msg, timeout=5)
            except grpc.RpcError as e:
                logger.debug("Gossip to %s failed: %s", peer, e)

    def _sync_from_peer(self, peer_address: str) -> int:
        """Sync blocks from a peer to catch up."""
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
                accepted, _ = self.chain.receive_external_block(block)
                if accepted:
                    synced += 1
            if synced > 0:
                print(f"[Daemon] Synced {synced} block(s) from {peer_address} "
                      f"(height: {my_height} → {self.chain.get_tip_height()})")
            return synced
        except grpc.RpcError:
            return 0


# ---------------------------------------------------------------------------
# Registration with the registry
# ---------------------------------------------------------------------------

class DaemonRegistration:
    """Register the daemon as node_type='daemon' with the registry."""

    def __init__(self, node_id: str, address: str,
                 registry_address: str):
        self.node_id = node_id
        self.address = address
        self.registry_address = registry_address
        self._running = False

    def start(self):
        """Register and start heartbeat loop."""
        try:
            channel = grpc.insecure_channel(
                self.registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            stub.Register(registry_pb2.RegisterRequest(
                node_id=self.node_id,
                address=self.address,
                model_id="",  # daemon is model-agnostic
                shard_index=-1,
                layer_start=-1,
                layer_end=-1,
                node_type="daemon",
                public_key=b"",  # daemon doesn't do onion routing
            ), timeout=10)
            channel.close()
            print(f"[Daemon] Registered as node_type='daemon' at {self.address}")
        except grpc.RpcError as e:
            print(f"[Daemon] WARNING: Failed to register: {e}")

        self._running = True
        t = threading.Thread(target=self._heartbeat_loop, daemon=True)
        t.start()

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(config.HEARTBEAT_INTERVAL_SECONDS)
            if not self._running:
                break
            try:
                channel = grpc.insecure_channel(
                    self.registry_address, options=config.GRPC_OPTIONS)
                stub = registry_pb2_grpc.RegistryStub(channel)
                resp = stub.Heartbeat(
                    registry_pb2.HeartbeatRequest(node_id=self.node_id),
                    timeout=10,
                )
                if not resp.acknowledged:
                    # Re-register
                    stub.Register(registry_pb2.RegisterRequest(
                        node_id=self.node_id,
                        address=self.address,
                        model_id="",
                        shard_index=-1,
                        layer_start=-1,
                        layer_end=-1,
                        node_type="daemon",
                        public_key=b"",
                    ), timeout=10)
                channel.close()
            except grpc.RpcError as e:
                logger.debug("Registry re-registration failed: %s", e)

    def stop(self):
        self._running = False
        try:
            channel = grpc.insecure_channel(
                self.registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            stub.Unregister(
                registry_pb2.UnregisterRequest(node_id=self.node_id),
                timeout=5,
            )
            channel.close()
        except grpc.RpcError as e:
            logger.debug("Unregister from registry failed: %s", e)


# ---------------------------------------------------------------------------
# serve() — main entry point
# ---------------------------------------------------------------------------

def serve(port: int = 50070, host: str = "[::]",
          advertise_address: str = None,
          registry_address: str = None,
          db_path: str = None):
    """Start the chain daemon.

    Args:
        port: gRPC port to listen on.
        host: Bind address.
        advertise_address: Address to register with registry.
        registry_address: Registry address for discovery.
        db_path: Path to SQLite database file.
    """
    import uuid

    registry_address = registry_address or config.REGISTRY_ADDRESS
    db_path = db_path or os.path.expanduser(
        getattr(config, 'CHAIN_DB_PATH', '~/.unfed/chain.db'))
    public_address = advertise_address or f"localhost:{port}"
    node_id = str(uuid.uuid4())

    print(f"[Daemon] Starting chain daemon")
    print(f"[Daemon]   Port: {port}")
    print(f"[Daemon]   Address: {public_address}")
    print(f"[Daemon]   Registry: {registry_address}")
    print(f"[Daemon]   Database: {db_path}")

    # Initialize SQLite store and chain
    store = ChainStore(db_path=db_path)
    chain = ShareChain(
        block_interval=10.0,
        settlement_blocks=6,
        store=store,
    )

    print(f"[Daemon] Chain loaded: height={chain.get_tip_height()}, "
          f"blocks={chain.height}")

    # Initialize fee oracle (EIP-1559-style dynamic pricing)
    from economics.fee_oracle import FeeOracle
    fee_oracle = FeeOracle(
        target_utilization=getattr(config, 'FEE_TARGET_UTILIZATION', 0.7),
        base_fee=getattr(config, 'FEE_BASE_DEFAULT', 0.001),
        min_fee=getattr(config, 'FEE_MIN', 0.0001),
        max_fee=getattr(config, 'FEE_MAX', 0.1),
        window_blocks=getattr(config, 'FEE_WINDOW_BLOCKS', 10),
        adjustment_factor=getattr(config, 'FEE_ADJUSTMENT_FACTOR', 0.125),
        target_capacity=getattr(config, 'FEE_TARGET_CAPACITY', 40),
    )
    # Warm up oracle with existing chain blocks
    for block in chain.get_blocks_from(0):
        if block.shares:  # skip genesis
            fee_oracle.update(block)
    print(f"[Daemon] Fee oracle initialized: base_fee={fee_oracle.get_base_fee():.6f}")

    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=config.GRPC_OPTIONS,
    )
    servicer = DaemonServicer(chain, node_id, fee_oracle=fee_oracle)
    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)

    bind_address = f"{host}:{port}"
    server.add_insecure_port(bind_address)
    server.start()
    print(f"[Daemon] gRPC server listening on {bind_address}")

    # Register with the registry
    registration = DaemonRegistration(
        node_id=node_id,
        address=public_address,
        registry_address=registry_address,
    )
    registration.start()

    # Start gossip + block production
    gossip = GossipManager(
        node_id=node_id,
        chain=chain,
        servicer=servicer,
        registry_address=registry_address,
        self_address=public_address,
    )
    gossip.start()

    print(f"[Daemon] Ready. Chain daemon is running.")
    print(f"[Daemon] Compute nodes should submit shares to {public_address}")

    # Graceful shutdown
    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        print(f"\n[Daemon] Shutting down...")
        gossip.stop()
        registration.stop()
        chain.stop()
        store.close()
        server.stop(grace=5)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    shutdown_event.wait()
    print("[Daemon] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UNFED AI Chain Daemon (like monerod)")
    parser.add_argument("--port", type=int, default=50070,
                        help="gRPC port (default: 50070)")
    parser.add_argument("--host", default="[::]",
                        help="Bind address (default: [::])")
    parser.add_argument("--advertise", default=None,
                        help="Public address to register (default: localhost:<port>)")
    parser.add_argument("--registry", default=None,
                        help="Registry address (default: from config)")
    parser.add_argument("--db", default=None,
                        help="SQLite database path (default: ~/.unfed/chain.db)")
    args = parser.parse_args()

    serve(
        port=args.port,
        host=args.host,
        advertise_address=args.advertise,
        registry_address=args.registry,
        db_path=args.db,
    )
