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
import builtins
import hashlib
import logging
import os
import queue
import random
import re
import signal
import sys
import threading
import time
from collections import defaultdict

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
from network.share_auth import SharePayload, canonical_share_payload_bytes, verify_signature

_SIMPLE_LOGS_ENABLED = True
_COLOR_ENABLED = (
    os.environ.get("UNFED_COLOR", "1") != "0"
    and "NO_COLOR" not in os.environ
    and sys.stdout.isatty()
)


def _colorize_tagged_line(line: str) -> str:
    if not _COLOR_ENABLED:
        return line
    m = re.match(r"^(\s*)(\[[^\]]+\])(.*)$", line)
    if not m:
        return line
    lead, tag, tail = m.groups()
    lower = tag.lower()
    if "daemon" in lower:
        code = "93"
    elif "registry" in lower:
        code = "96"
    elif "node" in lower:
        code = "92"
    elif "mpc" in lower:
        code = "94"
    else:
        code = "90"
    return f"{lead}\x1b[{code}m{tag}\x1b[0m{tail}"


def _colorize_text(text: str) -> str:
    if not _COLOR_ENABLED or not text:
        return text
    return "\n".join(_colorize_tagged_line(part) for part in text.split("\n"))


def print(*args, **kwargs):  # type: ignore[override]
    text = kwargs.get("sep", " ").join(str(a) for a in args)
    text = _colorize_text(text)
    out_kwargs = {}
    if "file" in kwargs:
        out_kwargs["file"] = kwargs["file"]
    if "flush" in kwargs:
        out_kwargs["flush"] = kwargs["flush"]
    if "end" in kwargs:
        out_kwargs["end"] = kwargs["end"]
    builtins.print(text, **out_kwargs)


def _simple_log(message: str):
    if _SIMPLE_LOGS_ENABLED:
        print(message)


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

    def __init__(self, chain: ShareChain, node_id: str, fee_oracle=None,
                 registry_address: str | None = None):
        self.chain = chain
        self.node_id = node_id
        self.fee_oracle = fee_oracle
        self.registry_address = registry_address or config.REGISTRY_ADDRESS
        self._strict_share_auth = (
            os.environ.get("UNFED_STRICT_SHARE_AUTH", "1").strip().lower()
            not in ("0", "false", "off", "no")
        )
        self._max_clock_skew_ms = int(os.environ.get("UNFED_SHARE_MAX_SKEW_MS", "120000"))
        self._max_shares_per_session_node = int(
            os.environ.get("UNFED_MAX_SHARES_PER_SESSION_NODE", "4096")
        )
        self._max_shares_per_node_window = int(
            os.environ.get("UNFED_MAX_SHARES_PER_NODE_WINDOW", "2048")
        )
        self._share_window_seconds = int(
            os.environ.get("UNFED_SHARE_WINDOW_SECONDS", "60")
        )
        self._tip_anchor_mode = os.environ.get(
            "UNFED_SHARE_TIP_ANCHOR_MODE", "strict"
        ).strip().lower()
        self._tip_max_lag_blocks = int(
            os.environ.get("UNFED_SHARE_TIP_MAX_LAG_BLOCKS", "2")
        )
        self._seen_keys: set[tuple[str, str, str, int]] = set()
        self._last_step: dict[tuple[str, str, int], int] = {}
        self._accepted_count: dict[tuple[str, str], int] = defaultdict(int)
        self._window_count: dict[tuple[str, int], int] = defaultdict(int)
        self._last_share_hash: dict[tuple[str, str], str] = {}
        self._reason_counters: dict[str, int] = defaultdict(int)
        self._signer_cache: dict[str, bytes] = {}
        self._signer_cache_ts = 0.0

        # Subscribers for SubscribeBlocks streaming RPC
        self._subscribers: list[queue.Queue] = []
        self._subscribers_lock = threading.Lock()

    # --- SubmitShares (compute nodes → daemon) ---

    def SubmitShares(self, request, context):
        """Accept compute shares from a compute/MPC node."""
        shares = [proto_to_share(s) for s in request.shares]
        accepted: list[ComputeShare] = []
        for share in shares:
            ok, reason = self._validate_share(share, request.submitter_id)
            if ok:
                share.validated = True
                accepted.append(share)
            else:
                self._reason_counters[reason] += 1
        if accepted:
            self.chain.add_shares(accepted)

        submitter = request.submitter_id[:8] if request.submitter_id else "unknown"
        _simple_log(
            f"[Daemon] Received {len(shares)} share(s) from {submitter}... "
            f"accepted={len(accepted)}"
        )

        return inference_pb2.SubmitSharesResponse(
            accepted=len(accepted),
            pending_pool_size=len(self.chain._pending_shares),
            chain_tip_hash=self.chain.latest_block.block_hash,
            chain_tip_height=self.chain.get_tip_height(),
        )

    def _validate_share(self, share: ComputeShare, submitter_id: str) -> tuple[bool, str]:
        if not self._strict_share_auth:
            return True, "ok_relaxed"
        if submitter_id and share.node_id != submitter_id:
            return False, "submitter_mismatch"
        if not share.signature:
            return False, "missing_signature"
        if not share.session_nonce:
            return False, "missing_session_nonce"
        if share.timestamp_ms <= 0:
            return False, "missing_timestamp_ms"
        now_ms = int(time.time() * 1000)
        if abs(now_ms - int(share.timestamp_ms)) > self._max_clock_skew_ms:
            return False, "stale_timestamp"
        if not self._verify_share_signature(share):
            return False, "invalid_signature"
        if share.idempotency_key:
            replay_key = ("idempotency", share.idempotency_key, share.node_id, int(share.step_index))
        else:
            replay_key = (share.session_id, share.session_nonce, share.node_id, int(share.step_index))
        if replay_key in self._seen_keys:
            return False, "replay"
        chain_tip_hash = self.chain.latest_block.block_hash
        if self._tip_anchor_mode not in ("off", "none", "disabled"):
            if not share.prev_block_hash:
                # First share in a chain may omit the anchor; later shares must provide it.
                if int(share.step_index) > 0:
                    return False, "missing_prev_block_hash"
            else:
                if self._tip_anchor_mode == "strict":
                    if share.prev_block_hash != chain_tip_hash:
                        return False, "tip_anchor_mismatch"
                else:
                    allowed = {chain_tip_hash}
                    lag = max(0, int(self._tip_max_lag_blocks))
                    for b in self.chain.get_blocks_from(max(0, self.chain.get_tip_height() - lag)):
                        allowed.add(b.block_hash)
                    if share.prev_block_hash not in allowed:
                        return False, "tip_anchor_mismatch"
        seq_key = (share.session_id, share.node_id)
        expected_prev_share_hash = self._last_share_hash.get(seq_key, "")
        if expected_prev_share_hash:
            if share.prev_share_hash != expected_prev_share_hash:
                return False, "prev_share_hash_mismatch"
        else:
            if share.prev_share_hash:
                return False, "unexpected_prev_share_hash"
        order_key = (share.session_id, share.node_id, int(share.shard_index))
        prev = self._last_step.get(order_key)
        if prev is not None and int(share.step_index) <= prev:
            return False, "non_monotonic_step"
        cap_key = (share.session_id, share.node_id)
        if self._accepted_count[cap_key] >= self._max_shares_per_session_node:
            return False, "cap_exceeded"
        current_window = int(time.time()) // max(1, self._share_window_seconds)
        window_key = (share.node_id, current_window)
        if self._window_count[window_key] >= self._max_shares_per_node_window:
            return False, "window_cap_exceeded"
        self._seen_keys.add(replay_key)
        self._last_step[order_key] = int(share.step_index)
        self._accepted_count[cap_key] += 1
        self._window_count[window_key] += 1
        self._last_share_hash[seq_key] = share.hash()
        return True, "ok"

    def _verify_share_signature(self, share: ComputeShare) -> bool:
        public_key = self._signer_cache.get(share.node_id)
        if public_key is None:
            self._refresh_signer_cache(force=True)
            public_key = self._signer_cache.get(share.node_id)
        if not public_key:
            return False
        payload = SharePayload(
            node_id=share.node_id,
            shard_index=int(share.shard_index),
            session_id=share.session_id,
            session_nonce=share.session_nonce,
            step_index=int(share.step_index),
            activation_hash=share.activation_hash,
            tokens_processed=int(share.tokens_processed),
            share_weight=float(share.share_weight),
            timestamp_ms=int(share.timestamp_ms),
            payload_hash_version=share.payload_hash_version or "v1",
            prev_block_hash=share.prev_block_hash,
            prev_share_hash=share.prev_share_hash,
            idempotency_key=share.idempotency_key,
        )
        return verify_signature(
            public_key,
            canonical_share_payload_bytes(payload),
            share.signature,
        )

    def _refresh_signer_cache(self, force: bool = False):
        now = time.time()
        if not force and (now - self._signer_cache_ts) < 15:
            return
        try:
            channel = grpc.insecure_channel(self.registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(registry_pb2.DiscoverRequest(model_id=""), timeout=5)
            new_cache: dict[str, bytes] = {}
            for node in resp.nodes:
                if node.share_signing_public_key:
                    new_cache[node.node_id] = bytes(node.share_signing_public_key)
            self._signer_cache = new_cache
            self._signer_cache_ts = now
            channel.close()
        except grpc.RpcError:
            pass

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

    # --- GetLoad (telemetry + fee oracle data) ---

    def GetLoad(self, request, context):
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
                _simple_log(
                    f"[Daemon] Fee oracle updated: base_fee={new_fee:.6f}, "
                    f"utilization={self.fee_oracle.get_utilization():.2%}"
                )
        msg = block_to_proto(block, proposer_id=self.node_id)
        self._notify_subscribers(msg)

    def submit_share_window_audit(self, block: Block):
        """Submit a deterministic share-root audit record to the registry."""
        try:
            h = hashlib.sha256()
            for s in block.shares:
                h.update(s.hash().encode("utf-8"))
            share_root = h.hexdigest()
            channel = grpc.insecure_channel(self.registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            stub.SubmitShareWindowAudit(
                registry_pb2.SubmitShareWindowAuditRequest(
                    daemon_node_id=self.node_id,
                    window_id=f"block-{block.index}",
                    start_block=int(block.index),
                    end_block=int(block.index),
                    share_count=len(block.shares),
                    share_root=share_root,
                    timestamp=time.time(),
                ),
                timeout=5,
            )
            channel.close()
        except grpc.RpcError:
            pass


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
                        _simple_log(
                            f"[Daemon] Produced block #{block.index}: "
                            f"{len(block.shares)} shares "
                            f"(hash={block.block_hash[:12]}...)"
                        )
                        # Notify subscribers
                        self.servicer.notify_new_block(block)
                        # Audit root to registry for independent settlement checks
                        self.servicer.submit_share_window_audit(block)
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
                _simple_log(
                    f"[Daemon] Synced {synced} block(s) from {peer_address} "
                    f"(height: {my_height} → {self.chain.get_tip_height()})"
                )
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
          db_path: str = None,
          node_id: str = ""):
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
    node_id = (node_id or "").strip() or str(uuid.uuid4())

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
    servicer = DaemonServicer(
        chain,
        node_id,
        fee_oracle=fee_oracle,
        registry_address=registry_address,
    )
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

    _simple_log("[Daemon] Ready. Chain daemon is running.")
    _simple_log(f"[Daemon] Compute nodes should submit shares to {public_address}")

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
    parser.add_argument("--eth-address", default="",
                        help="Stable daemon node identity (recommended: staked EVM address)")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable simple daemon status logs")
    args = parser.parse_args()
    if args.quiet:
        import builtins
        builtins.print = lambda *a, **k: None
        _SIMPLE_LOGS_ENABLED = False

    serve(
        port=args.port,
        host=args.host,
        advertise_address=args.advertise,
        registry_address=args.registry,
        db_path=args.db,
        node_id=args.eth_address,
    )
