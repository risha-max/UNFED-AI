"""
Registry Server — bootstrap node for peer discovery.

Nodes register on startup, send periodic heartbeats, and unregister on shutdown.
Clients query the registry to discover available nodes and build inference circuits.

In production, multiple independent registries would run (like Bitcoin DNS seeds).
For PoC/testing, we run a single registry on localhost.

Usage:
    python -m network.registry_server
    python -m network.registry_server --port 50050
"""

import argparse
import json
import logging
import signal
import sys
import os
import time
import threading

logger = logging.getLogger("unfed.registry")
from concurrent import futures

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import registry_pb2
import registry_pb2_grpc
import inference_pb2
import inference_pb2_grpc

from economics.share_chain import ShareChain, ComputeShare
from economics.payments import StakeManager, PaymentContract, SettlementProcessor
from economics.model_pools import PoolRegistry, PoolManifest
from economics.pool_config import PoolConfig
from economics.cluster_config import ClusterConfig, merge_with_defaults
from economics.distributed_chain import (
    DistributedShareChain, block_to_proto, proto_to_block,
)


class NodeRecord:
    """In-memory record of a registered node."""

    def __init__(self, node_id: str, address: str, model_id: str,
                 shard_index: int, layer_start: int, layer_end: int,
                 has_embedding: bool, has_lm_head: bool,
                 public_key: bytes = b"", node_type: str = "compute"):
        self.node_id = node_id
        self.address = address
        self.model_id = model_id
        self.shard_index = shard_index
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.public_key = public_key
        self.node_type = node_type
        self.last_heartbeat = time.time()
        self.registered_at = time.time()

    def to_proto(self) -> registry_pb2.NodeInfo:
        return registry_pb2.NodeInfo(
            node_id=self.node_id,
            address=self.address,
            model_id=self.model_id,
            shard_index=self.shard_index,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            has_embedding=self.has_embedding,
            has_lm_head=self.has_lm_head,
            last_heartbeat=self.last_heartbeat,
            public_key=self.public_key,
            node_type=self.node_type,
        )


class RegistryServicer(registry_pb2_grpc.RegistryServicer):
    """gRPC service implementation for the node registry."""

    def __init__(self, cluster_config: ClusterConfig | None = None,
                 seed_peers: list[str] | None = None,
                 no_chain: bool = False):
        self._nodes: dict[str, NodeRecord] = {}  # node_id -> NodeRecord
        self._lock = threading.Lock()

        # --- Cluster identity ---
        self._cluster_config = cluster_config or ClusterConfig(
            name="default")
        self._cluster_config.ensure_id()
        self._start_time = time.time()

        # --- Known peer registries (endpoint -> PeerInfo-like dict) ---
        self._known_peers: dict[str, dict] = {}
        self._peers_lock = threading.Lock()
        # Bootstrap from seed list
        for addr in (seed_peers or []):
            self._known_peers[addr] = {
                "endpoint": addr,
                "cluster_id": "",
                "name": "",
                "last_seen": 0.0,
            }

        # Verification ticket queue
        self._pending_tickets: list = []  # VerificationTicketProto messages
        self._tickets_lock = threading.Lock()

        # Fraud proofs
        self._fraud_proofs: list = []  # FraudProofMessage messages
        self._fraud_lock = threading.Lock()

        # Map ticket_id -> node_id (to identify the fraudulent node)
        self._ticket_owners: dict[str, str] = {}
        self._ticket_owners_lock = threading.Lock()

        # Model manifests (model_id -> manifest JSON string)
        self._manifests: dict[str, str] = {}
        self._manifests_lock = threading.Lock()

        # Per-model pool configs (model_id -> PoolConfig JSON string)
        self._pool_configs: dict[str, str] = {}
        self._pool_configs_lock = threading.Lock()

        # Economics: distributed share-chain, payments, model pools
        # The registry holds a chain copy as a passive gossip peer.
        # It does NOT produce blocks — nodes do that now.
        self._distributed_chain = DistributedShareChain(
            node_id="registry",
            registry_address=None,  # no self-discovery needed
            block_interval=10.0,
            settlement_blocks=6,
        )
        # Alias for settlement loop compatibility
        self._share_chain = self._distributed_chain.chain

        # --- On-chain escrow (default when configured, skip with --no-chain) ---
        self._onchain_escrow = None
        self._no_chain = no_chain
        cc = self._cluster_config
        if no_chain:
            print("[Registry] On-chain escrow DISABLED (--no-chain flag). "
                  "Using in-memory simulation.")
        elif cc.escrow_contract_address and cc.chain_rpc_url:
            try:
                from economics.onchain import OnChainEscrow
                self._onchain_escrow = OnChainEscrow(
                    rpc_url=cc.chain_rpc_url,
                    contract_address=cc.escrow_contract_address,
                    operator_private_key=cc.operator_private_key,
                    token_address=cc.staking_token_address,
                )
                print(f"[Registry] On-chain escrow ENABLED (default): "
                      f"{cc.escrow_contract_address}")
            except Exception as e:
                print(f"[Registry] WARNING: Failed to connect to on-chain "
                      f"escrow: {e}. Falling back to in-memory simulation.")
        else:
            print("[Registry] No on-chain escrow configured. "
                  "Using in-memory simulation. "
                  "Set chain_rpc_url + escrow_contract_address in cluster "
                  "config to enable on-chain payments.")

        self._stake_manager = StakeManager()
        # PaymentContract receives the on-chain escrow when available,
        # making on-chain the default settlement path.
        self._payment_contract = PaymentContract(
            self._stake_manager,
            challenge_window=cc.challenge_window_seconds
            if cc.challenge_window_seconds else 60.0,
            price_per_input_token=cc.default_price_per_input_token,
            price_per_output_token=cc.default_price_per_output_token,
            onchain_escrow=self._onchain_escrow,
        )
        self._settlement_processor = SettlementProcessor(self._payment_contract)
        self._pool_registry = PoolRegistry()

        # NOTE: No start_block_production() — registry is a passive peer.
        # It receives blocks via GossipBlock RPCs from compute nodes.
        # Start background payment finalization
        self._settlement_processor.start_background_finalization(interval=10.0)

        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        # Monitor share-chain settlements
        self._settlement_thread = threading.Thread(
            target=self._settlement_loop, daemon=True)
        self._settlement_thread.start()

        # Start peer gossip loop
        self._gossip_thread = threading.Thread(
            target=self._gossip_loop, daemon=True)
        self._gossip_thread.start()

        # --- Auto-assignment state ---
        self._mpc_waiting_queue: list[dict] = []  # nodes waiting for MPC partner
        self._assignment_history: dict[str, dict] = {}  # node_id -> assignment
        self._orphaned_shards: list[dict] = []  # shards that lost their last node

    def _cleanup_loop(self):
        """Periodically remove nodes that haven't sent a heartbeat."""
        while True:
            time.sleep(config.HEARTBEAT_INTERVAL_SECONDS)
            self._remove_stale_nodes()

    def _settlement_loop(self):
        """Monitor share-chain for new settlements and post them to payments.

        When on-chain escrow is enabled, also posts settlements on-chain
        and runs finalization for expired challenge windows.
        """
        last_count = 0
        # Track on-chain settlements pending finalization
        onchain_pending: list[tuple[str, float]] = []  # (hash, deadline)

        while True:
            time.sleep(5)
            settlements = self._share_chain.get_settlements()
            if len(settlements) > last_count:
                for s in settlements[last_count:]:
                    self._settlement_processor.process_settlement(s)

                    # Post on-chain if escrow is enabled
                    if self._onchain_escrow and s.node_shares:
                        try:
                            nodes = list(s.node_shares.keys())
                            # Convert weighted shares to token amounts
                            # (use 1e18 scale for ERC-20 wei)
                            amounts = [
                                int(shares * 1e18)
                                for shares in s.node_shares.values()
                            ]
                            self._onchain_escrow.post_settlement(
                                s.settlement_hash, nodes, amounts)
                            deadline = (time.time()
                                        + self._cluster_config
                                        .challenge_window_seconds)
                            onchain_pending.append(
                                (s.settlement_hash, deadline))
                        except Exception as e:
                            print(f"[Registry] On-chain settlement post "
                                  f"failed: {e}")

                last_count = len(settlements)

            # Finalize expired on-chain settlements
            if self._onchain_escrow and onchain_pending:
                now = time.time()
                still_pending = []
                for s_hash, deadline in onchain_pending:
                    if now >= deadline:
                        try:
                            self._onchain_escrow.finalize_settlement(s_hash)
                        except Exception as e:
                            print(f"[Registry] On-chain finalization "
                                  f"failed: {e}")
                    else:
                        still_pending.append((s_hash, deadline))
                onchain_pending = still_pending

    def _gossip_loop(self):
        """Periodically exchange peer lists with known registries."""
        while True:
            time.sleep(60)  # gossip every 60 seconds
            self._do_gossip_round()

    def _do_gossip_round(self):
        """Contact all known peers and exchange peer lists."""
        with self._peers_lock:
            peers_snapshot = list(self._known_peers.keys())

        my_endpoint = self._cluster_config.public_endpoint
        # Include ourselves so peers learn about us
        my_peers = [my_endpoint] if my_endpoint else []
        my_peers.extend(peers_snapshot)

        for peer_addr in peers_snapshot:
            try:
                channel = grpc.insecure_channel(peer_addr)
                stub = registry_pb2_grpc.RegistryStub(channel)
                resp = stub.ExchangePeers(
                    registry_pb2.ExchangePeersRequest(
                        known_peers=my_peers),
                    timeout=10,
                )
                # Merge their peers into ours
                with self._peers_lock:
                    for p in resp.peers:
                        ep = p.endpoint
                        if ep and ep != my_endpoint:
                            existing = self._known_peers.get(ep)
                            if (existing is None
                                    or p.last_seen > existing["last_seen"]):
                                self._known_peers[ep] = {
                                    "endpoint": ep,
                                    "cluster_id": p.cluster_id,
                                    "name": p.name,
                                    "last_seen": p.last_seen,
                                }
                channel.close()
            except Exception:
                pass  # peer unreachable, skip

    def _remove_stale_nodes(self):
        """Remove nodes that haven't heartbeated within the timeout.

        Tracks orphaned shards (shards that lost their last node) so that
        _compute_assignment prioritizes them for the next joining node.
        """
        now = time.time()
        with self._lock:
            stale = [
                nid for nid, record in self._nodes.items()
                if now - record.last_heartbeat > config.NODE_TIMEOUT_SECONDS
            ]
            for nid in stale:
                record = self._nodes.pop(nid)
                print(f"[Registry] Removed stale node {nid[:8]}... "
                      f"(shard {record.shard_index}, {record.address})")
                self._pool_registry.node_left(
                    record.model_id, record.shard_index, nid)

                # Check if this shard is now uncovered
                remaining = sum(
                    1 for n in self._nodes.values()
                    if n.model_id == record.model_id
                    and n.shard_index == record.shard_index
                )
                if remaining == 0:
                    self._orphaned_shards.append({
                        "model_id": record.model_id,
                        "shard_index": record.shard_index,
                        "node_type": record.node_type,
                        "timestamp": now,
                    })
                    print(f"[Registry] ORPHANED shard {record.shard_index} "
                          f"for {record.model_id} — queued for reassignment")

    # ------------------------------------------------------------------
    # Auto-assignment
    # ------------------------------------------------------------------

    def RequestAssignment(self, request, context):
        """Assign a role/shard to a node based on capacity and network needs."""
        node_id = request.node_id
        capacity = request.capacity
        available_shards = list(request.available_shards)
        address = request.address

        # Check if this node already has a pending assignment (MPC poll)
        if node_id in self._assignment_history:
            a = self._assignment_history[node_id]
            return registry_pb2.RequestAssignmentResponse(
                success=True,
                message="Assignment ready (from MPC pairing)",
                assignment=registry_pb2.Assignment(**a),
            )

        # MPC pairing attempt
        if request.willing_to_mpc:
            result = self._try_mpc_pairing(
                node_id, capacity, address, available_shards)
            if result is not None:
                return result

        # Compute optimal assignment
        assignment = self._compute_assignment(
            node_id, capacity, available_shards, address,
            request.preferred_model_id)

        if assignment is None:
            return registry_pb2.RequestAssignmentResponse(
                success=False,
                message="No assignment available — no models registered "
                        "or all shards well-covered",
            )

        self._assignment_history[node_id] = assignment
        print(f"[Registry] Assigned {node_id[:8]}... -> "
              f"role={assignment['role']} shard={assignment['shard_index']} "
              f"stack={assignment.get('stack', '')} "
              f"file={assignment.get('shard_file', '')}")

        return registry_pb2.RequestAssignmentResponse(
            success=True,
            message="Assigned",
            assignment=registry_pb2.Assignment(**assignment),
        )

    def _try_mpc_pairing(self, node_id, capacity, address, available_shards):
        """Try to pair this node with a waiting MPC partner.

        Returns a RequestAssignmentResponse if handled (paired or queued),
        or None to fall through to normal assignment.
        """
        with self._lock:
            if self._mpc_waiting_queue:
                partner = self._mpc_waiting_queue.pop(0)
                partner_id = partner["node_id"]
                partner_address = partner["address"]

                # Find the shard 0 info from the first registered model
                shard0_info = self._find_shard_info(0, "text_decoder")

                base = {
                    "model_id": shard0_info.get("model_id", ""),
                    "shard_index": 0,
                    "stack": "text_decoder",
                    "layer_start": shard0_info.get("layer_start", 0),
                    "layer_end": shard0_info.get("layer_end", 0),
                    "has_embedding": True,
                    "has_lm_head": False,
                    "estimated_ram_gb": shard0_info.get("estimated_ram_gb", 0),
                    "shard_file": shard0_info.get("file", "text_shard_0.pt"),
                }

                # Partner (first to arrive) = role B
                partner_assignment = {
                    **base,
                    "role": "mpc",
                    "mpc_role": "B",
                    "mpc_peer_address": address,
                    "download_peers": [],
                }
                self._assignment_history[partner_id] = partner_assignment

                # This node = role A (entry)
                this_assignment = {
                    **base,
                    "role": "mpc",
                    "mpc_role": "A",
                    "mpc_peer_address": partner_address,
                    "download_peers": [],
                }
                self._assignment_history[node_id] = this_assignment

                print(f"[Registry] MPC paired: {node_id[:8]}... (A) <-> "
                      f"{partner_id[:8]}... (B)")

                return registry_pb2.RequestAssignmentResponse(
                    success=True,
                    message="MPC paired as role A",
                    assignment=registry_pb2.Assignment(**this_assignment),
                )

            # No partner — queue and wait
            self._mpc_waiting_queue.append({
                "node_id": node_id,
                "capacity": capacity,
                "address": address,
                "available_shards": available_shards,
            })

        print(f"[Registry] MPC queued: {node_id[:8]}... waiting for partner")
        return registry_pb2.RequestAssignmentResponse(
            success=False,
            message="Queued for MPC pairing — waiting for a partner. "
                    "Poll again in a few seconds.",
        )

    def _compute_assignment(self, node_id, capacity, available_shards,
                            address, preferred_model_id=""):
        """Core assignment engine.

        Returns an assignment dict or None.
        Priority: orphaned shards > uncovered > under-replicated.
        GPU nodes prefer vision shards; CPU nodes prefer text.
        """
        has_gpu = capacity.has_gpu if capacity else False
        avail_ram = capacity.available_ram_gb if capacity else 0.0

        # Collect all shard info from registered manifests
        candidates = self._gather_shard_candidates(preferred_model_id)
        if not candidates:
            return None

        # Score each candidate
        scored = []
        for c in candidates:
            est_ram = c["estimated_ram_gb"]
            if avail_ram > 0 and est_ram > avail_ram:
                continue  # node doesn't have enough RAM

            # Check if node has the shard file or peers exist to download
            shard_file = c.get("file", "")
            has_file = shard_file in available_shards
            peers = self._find_download_peers(
                c["model_id"], c["shard_index"], c["stack"])
            if not has_file and not peers:
                continue  # can't get the shard

            score = 0.0

            # Orphaned shards get maximum priority
            if self._is_orphaned(c["model_id"], c["shard_index"]):
                score += 2000

            # Uncovered shards (no nodes currently serving)
            if c["node_count"] == 0:
                score += 1000

            # Under-replicated: fewer replicas = higher score
            max_replicas = max((x["node_count"] for x in candidates), default=1)
            score += (max_replicas - c["node_count"]) * 10

            # GPU affinity: vision shards prefer GPU nodes
            if c["stack"] == "vision_encoder" and has_gpu:
                score += 50
            if c["stack"] == "text_decoder" and not has_gpu:
                score += 20

            # Tie-break: larger shards are harder to place
            score += est_ram

            scored.append((score, c, has_file, peers))

        if not scored:
            return None

        # Pick highest-scoring shard
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best, has_file, peers = scored[0]

        # Clear from orphaned list if present
        self._orphaned_shards = [
            o for o in self._orphaned_shards
            if not (o["model_id"] == best["model_id"]
                    and o["shard_index"] == best["shard_index"])
        ]

        role = "vision" if best["stack"] == "vision_encoder" else "compute"
        return {
            "role": role,
            "model_id": best["model_id"],
            "shard_index": best["shard_index"],
            "stack": best["stack"],
            "layer_start": best["layer_start"],
            "layer_end": best["layer_end"],
            "has_embedding": best.get("has_embedding", False),
            "has_lm_head": best.get("has_lm_head", False),
            "estimated_ram_gb": best["estimated_ram_gb"],
            "shard_file": best.get("file", ""),
            "mpc_role": "",
            "mpc_peer_address": "",
            "download_peers": peers if not has_file else [],
        }

    def _gather_shard_candidates(self, preferred_model_id=""):
        """Build a list of all shards from registered manifests with node counts."""
        candidates = []

        with self._manifests_lock:
            manifest_items = list(self._manifests.items())

        for model_id, manifest_json in manifest_items:
            if preferred_model_id and model_id != preferred_model_id:
                continue
            try:
                manifest = json.loads(manifest_json)
            except (json.JSONDecodeError, TypeError):
                continue

            all_shards = manifest.get("shards", [])
            if not all_shards:
                all_shards = (manifest.get("text_shards", [])
                              + manifest.get("vision_shards", []))

            for s in all_shards:
                shard_idx = s.get("shard_index", 0)
                stack = s.get("stack", "text_decoder")
                node_type = "vision" if "vision" in stack else "compute"

                # Count active nodes serving this shard
                with self._lock:
                    node_count = sum(
                        1 for n in self._nodes.values()
                        if n.model_id == model_id
                        and n.shard_index == shard_idx
                        and n.node_type == node_type
                    )

                size_bytes = s.get("size_bytes", 0)
                candidates.append({
                    "model_id": model_id,
                    "shard_index": shard_idx,
                    "stack": stack,
                    "layer_start": s.get("layer_start", 0),
                    "layer_end": s.get("layer_end", 0),
                    "has_embedding": s.get("has_embedding", False),
                    "has_lm_head": s.get("has_lm_head", s.get("has_head", False)),
                    "file": s.get("file", ""),
                    "size_bytes": size_bytes,
                    "estimated_ram_gb": (size_bytes / 1e9) * 2.0,
                    "node_count": node_count,
                })

        return candidates

    def _find_download_peers(self, model_id, shard_index, stack):
        """Find addresses of nodes currently serving a given shard."""
        node_type = "vision" if "vision" in stack else "compute"
        with self._lock:
            return [
                n.address for n in self._nodes.values()
                if n.model_id == model_id
                and n.shard_index == shard_index
                and n.node_type == node_type
            ]

    def _find_shard_info(self, shard_index, stack):
        """Look up shard metadata from registered manifests."""
        with self._manifests_lock:
            for model_id, manifest_json in self._manifests.items():
                try:
                    manifest = json.loads(manifest_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                for s in manifest.get("shards", []):
                    if (s.get("shard_index") == shard_index
                            and s.get("stack", "text_decoder") == stack):
                        s["model_id"] = model_id
                        size_bytes = s.get("size_bytes", 0)
                        s["estimated_ram_gb"] = (size_bytes / 1e9) * 2.0
                        return s
        return {}

    def _is_orphaned(self, model_id, shard_index):
        return any(
            o["model_id"] == model_id and o["shard_index"] == shard_index
            for o in self._orphaned_shards
        )

    def _verify_capacity_async(self, node_id, address, expected_ram_gb,
                               has_gpu):
        """Post-registration capacity challenge (runs in background thread).

        Connects to the node's VerifyCapacity RPC, asks it to allocate
        ~80% of shard RAM on the appropriate device, and checks timing.
        On failure the node is unregistered.
        """
        challenge_bytes = int(expected_ram_gb * 0.8 * 1e9)
        if challenge_bytes <= 0:
            return  # no meaningful size to verify
        device = "cuda" if has_gpu else "cpu"

        try:
            channel = grpc.insecure_channel(address)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
            resp = stub.VerifyCapacity(
                registry_pb2.VerifyCapacityRequest(
                    node_id=node_id,
                    challenge_bytes=challenge_bytes,
                    expected_device=device,
                ),
                timeout=30,
            )
            channel.close()

            max_time = 10_000 if device == "cpu" else 5_000  # ms
            if not resp.passed or resp.allocation_time_ms > max_time:
                print(f"[Registry] CAPACITY VERIFICATION FAILED for "
                      f"{node_id[:8]}... (passed={resp.passed}, "
                      f"time={resp.allocation_time_ms:.0f}ms) — unregistering")
                with self._lock:
                    record = self._nodes.pop(node_id, None)
                if record:
                    self._pool_registry.node_left(
                        record.model_id, record.shard_index, node_id)
            else:
                print(f"[Registry] Capacity verified: {node_id[:8]}... "
                      f"({resp.allocation_time_ms:.0f}ms on {device})")
        except Exception as e:
            print(f"[Registry] Capacity verification error for "
                  f"{node_id[:8]}...: {e}")

    def Register(self, request, context):
        """Register a new node.

        If on-chain escrow is enabled, checks that the node has
        sufficient on-chain stake before admitting it.  The node_id
        is treated as an Ethereum address for stake lookups.
        """
        # --- On-chain stake gate ---
        if self._onchain_escrow:
            try:
                if not self._onchain_escrow.is_eligible(request.node_id):
                    min_stake = self._onchain_escrow.min_stake()
                    msg = (f"Insufficient on-chain stake. "
                           f"Minimum: {min_stake} wei")
                    print(f"[Registry] Rejected {request.node_id[:8]}...: "
                          f"{msg}")
                    return registry_pb2.RegisterResponse(
                        success=False, message=msg)
            except Exception as e:
                msg = (f"On-chain stake check failed: {e}. "
                       f"Registration denied (fail-closed).")
                print(f"[Registry] REJECTED {request.node_id[:8]}...: "
                      f"{msg}")
                return registry_pb2.RegisterResponse(
                    success=False, message=msg)

        with self._lock:
            record = NodeRecord(
                node_id=request.node_id,
                address=request.address,
                model_id=request.model_id,
                shard_index=request.shard_index,
                layer_start=request.layer_start,
                layer_end=request.layer_end,
                has_embedding=request.has_embedding,
                has_lm_head=request.has_lm_head,
                public_key=request.public_key,
                node_type=request.node_type or "compute",
            )
            self._nodes[request.node_id] = record

        node_type = request.node_type or "compute"
        print(f"[Registry] Node registered: {request.node_id[:8]}... "
              f"type={node_type} model={request.model_id} shard={request.shard_index} "
              f"layers={request.layer_start}-{request.layer_end - 1} "
              f"at {request.address}")

        # Track in model pool
        self._pool_registry.node_joined(
            request.model_id, request.shard_index, request.node_id)

        # Fire async capacity verification if the node has an assignment
        if request.node_id in self._assignment_history:
            a = self._assignment_history[request.node_id]
            threading.Thread(
                target=self._verify_capacity_async,
                args=(request.node_id, request.address,
                      a.get("estimated_ram_gb", 0), a.get("mpc_role") == ""),
                daemon=True,
            ).start()

        return registry_pb2.RegisterResponse(success=True, message="Registered")

    def Heartbeat(self, request, context):
        """Update heartbeat timestamp for a node."""
        with self._lock:
            if request.node_id in self._nodes:
                self._nodes[request.node_id].last_heartbeat = time.time()
                return registry_pb2.HeartbeatResponse(acknowledged=True)

        return registry_pb2.HeartbeatResponse(acknowledged=False)

    def Unregister(self, request, context):
        """Remove a node from the registry."""
        with self._lock:
            record = self._nodes.pop(request.node_id, None)

        if record:
            print(f"[Registry] Node unregistered: {request.node_id[:8]}... "
                  f"(shard {record.shard_index}, {record.address})")
            return registry_pb2.UnregisterResponse(success=True)

        return registry_pb2.UnregisterResponse(success=False)

    def Discover(self, request, context):
        """Return all known nodes, optionally filtered by model."""
        with self._lock:
            nodes = list(self._nodes.values())

        if request.model_id:
            nodes = [n for n in nodes if n.model_id == request.model_id]

        return registry_pb2.DiscoverResponse(
            nodes=[n.to_proto() for n in nodes]
        )

    def ListModels(self, request, context):
        """Return all models known to this registry with health summaries."""
        with self._lock:
            nodes = list(self._nodes.values())

        # Group compute nodes by model_id
        model_nodes: dict[str, list[NodeRecord]] = {}
        for n in nodes:
            if n.node_type == "compute":
                model_nodes.setdefault(n.model_id, []).append(n)

        models = []
        for model_id, mnodes in model_nodes.items():
            # Group by shard to compute coverage
            shard_set: set[int] = set()
            for n in mnodes:
                shard_set.add(n.shard_index)

            max_shard = max(n.shard_index for n in mnodes)
            total_shards = max_shard + 1
            covered = len(shard_set)

            models.append(registry_pb2.ModelInfo(
                model_id=model_id,
                total_nodes=len(mnodes),
                total_shards=total_shards,
                covered_shards=covered,
                can_serve=(covered == total_shards),
            ))

        return registry_pb2.ListModelsResponse(models=models)

    def GetPoolHealth(self, request, context):
        """Return health status of a model pool."""
        with self._lock:
            nodes = [n for n in self._nodes.values() if n.model_id == request.model_id]

        # Group nodes by shard
        shard_map: dict[int, list[NodeRecord]] = {}
        for n in nodes:
            shard_map.setdefault(n.shard_index, []).append(n)

        # Determine total shards from the nodes we know about
        if not nodes:
            return registry_pb2.PoolHealthResponse(
                model_id=request.model_id,
                total_shards=0,
                overall_status="incomplete",
                can_serve=False,
            )

        max_shard = max(n.shard_index for n in nodes)
        total_shards = max_shard + 1

        shard_healths = []
        all_covered = True
        any_fragile = False

        for i in range(total_shards):
            shard_nodes = shard_map.get(i, [])
            count = len(shard_nodes)

            if count == 0:
                status = "missing"
                all_covered = False
            elif count < 3:
                status = "degraded"
                any_fragile = True
            else:
                status = "healthy"

            # Get layer range from first node in shard (all should be the same)
            layer_start = shard_nodes[0].layer_start if shard_nodes else 0
            layer_end = shard_nodes[0].layer_end if shard_nodes else 0

            shard_healths.append(registry_pb2.ShardHealth(
                shard_index=i,
                layer_start=layer_start,
                layer_end=layer_end,
                node_count=count,
                node_ids=[n.node_id for n in shard_nodes],
                status=status,
            ))

        if not all_covered:
            overall = "incomplete"
        elif any_fragile:
            overall = "degraded"
        else:
            overall = "healthy"

        return registry_pb2.PoolHealthResponse(
            model_id=request.model_id,
            total_shards=total_shards,
            shards=shard_healths,
            overall_status=overall,
            can_serve=all_covered,
        )

    # --- Verification RPCs ---

    def SubmitTickets(self, request, context):
        """Accept verification tickets from a node and record compute shares."""
        with self._tickets_lock:
            for ticket in request.tickets:
                self._pending_tickets.append(ticket)
            count = len(request.tickets)

        # Track which node submitted which ticket (for fraud proof attribution)
        with self._ticket_owners_lock:
            for ticket in request.tickets:
                self._ticket_owners[ticket.ticket_id] = request.node_id

        # Record compute shares in the share-chain
        # Each ticket represents verified compute work
        for ticket in request.tickets:
            share = ComputeShare(
                node_id=request.node_id,
                shard_index=ticket.shard_index,
                session_id=ticket.ticket_id,  # use ticket_id as proxy
                activation_hash=str(hash(ticket.expected_output_data))[:16],
                tokens_processed=1,
            )
            self._share_chain.add_share(share)

        if count > 0:
            print(f"[Registry] Received {count} verification ticket(s) "
                  f"from node {request.node_id[:8]}...")

        return registry_pb2.SubmitTicketsResponse(accepted=count)

    def GetPendingTickets(self, request, context):
        """Return and clear pending verification tickets for verifiers."""
        with self._tickets_lock:
            limit = request.max_tickets if request.max_tickets > 0 else len(self._pending_tickets)
            tickets = self._pending_tickets[:limit]
            self._pending_tickets = self._pending_tickets[limit:]

        return registry_pb2.GetPendingTicketsResponse(tickets=tickets)

    def SubmitFraudProof(self, request, context):
        """Accept a fraud proof from a verifier and trigger slashing."""
        with self._fraud_lock:
            self._fraud_proofs.append(request)

        print(f"[Registry] FRAUD PROOF received for shard {request.shard_index} "
              f"(ticket: {request.ticket_id})")

        # Find the fraudulent node from ticket ownership
        with self._ticket_owners_lock:
            fraud_node_id = self._ticket_owners.get(request.ticket_id)

        if fraud_node_id:
            # Try to challenge any open settlement that contains this node's work
            settlements = self._share_chain.get_settlements()
            challenged = False
            for s in settlements:
                if fraud_node_id in s.node_shares:
                    result = self._payment_contract.challenge_settlement(
                        settlement_hash=s.settlement_hash,
                        fraud_node_id=fraud_node_id,
                    )
                    if result:
                        print(f"[Registry] Settlement challenged and node "
                              f"{fraud_node_id[:8]}... slashed!")
                        challenged = True
                        break

            # On-chain slash (if escrow is enabled)
            if self._onchain_escrow:
                try:
                    self._onchain_escrow.slash_node(fraud_node_id)
                    print(f"[Registry] On-chain slash executed for "
                          f"{fraud_node_id[:8]}...")
                except Exception as e:
                    print(f"[Registry] WARNING: On-chain slash failed "
                          f"for {fraud_node_id[:8]}...: {e}")

            if not challenged and not self._onchain_escrow:
                print(f"[Registry] Fraud proof recorded but no open settlement "
                      f"to challenge for node {fraud_node_id[:8]}...")
        else:
            print(f"[Registry] Fraud proof recorded but could not identify "
                  f"the submitting node for ticket {request.ticket_id}")

        return registry_pb2.FraudProofResponse(accepted=True)

    def GetFraudProofs(self, request, context):
        """Return all known fraud proofs."""
        with self._fraud_lock:
            proofs = list(self._fraud_proofs)
        return registry_pb2.GetFraudProofsResponse(proofs=proofs)

    # --- Manifest Distribution RPCs ---

    def GetManifest(self, request, context):
        """Return the manifest for a model, if available."""
        with self._manifests_lock:
            manifest_json = self._manifests.get(request.model_id)

        if manifest_json:
            return registry_pb2.GetManifestResponse(
                found=True, manifest_json=manifest_json)

        return registry_pb2.GetManifestResponse(found=False)

    def PutManifest(self, request, context):
        """Store or update a model manifest."""
        if not request.model_id or not request.manifest_json:
            return registry_pb2.PutManifestResponse(
                success=False, message="model_id and manifest_json required")

        # Basic validation: must be valid JSON
        try:
            import json
            json.loads(request.manifest_json)
        except json.JSONDecodeError as e:
            return registry_pb2.PutManifestResponse(
                success=False, message=f"Invalid JSON: {e}")

        with self._manifests_lock:
            self._manifests[request.model_id] = request.manifest_json

        print(f"[Registry] Manifest stored for model: {request.model_id}")
        return registry_pb2.PutManifestResponse(
            success=True, message="Manifest stored")

    # --- Pool Config RPCs (per-model economics) ---

    def GetPoolConfig(self, request, context):
        """Return the effective pool config for a model.

        Merges cluster-wide defaults with any per-model overrides.
        If no per-model config exists, returns cluster defaults with
        the requested model_id stamped in.
        """
        with self._pool_configs_lock:
            override_json = self._pool_configs.get(request.model_id)

        model_override = None
        if override_json:
            try:
                model_override = PoolConfig.from_json(override_json)
            except Exception:
                pass  # fall through to defaults

        effective = merge_with_defaults(
            self._cluster_config, model_override,
            model_id=request.model_id,
        )
        return registry_pb2.GetPoolConfigResponse(
            found=True, config_json=effective.to_json())

    def PutPoolConfig(self, request, context):
        """Store or update a model's pool config."""
        if not request.model_id or not request.config_json:
            return registry_pb2.PutPoolConfigResponse(
                success=False, message="model_id and config_json required")

        # Validate: must be valid JSON and a valid PoolConfig
        try:
            pool_cfg = PoolConfig.from_json(request.config_json)
        except (json.JSONDecodeError, TypeError) as e:
            return registry_pb2.PutPoolConfigResponse(
                success=False, message=f"Invalid pool config: {e}")

        errors = pool_cfg.validate()
        if errors:
            return registry_pb2.PutPoolConfigResponse(
                success=False,
                message=f"Validation errors: {'; '.join(errors)}")

        with self._pool_configs_lock:
            self._pool_configs[request.model_id] = request.config_json

        # Update the model pool's economic parameters if the pool exists
        pool = self._pool_registry.get_pool(request.model_id)
        if pool:
            pool.base_rate = pool_cfg.base_rate

        print(f"[Registry] Pool config stored for model: {request.model_id} "
              f"(scheme={pool_cfg.reward_scheme}, "
              f"stake={pool_cfg.min_stake}, "
              f"rate={pool_cfg.base_rate})")
        return registry_pb2.PutPoolConfigResponse(
            success=True, message="Pool config stored")

    # --- Cluster Identity & Gossip RPCs ---

    def GetClusterInfo(self, request, context):
        """Return this cluster's identity, default economics, and live stats."""
        cc = self._cluster_config

        with self._lock:
            total_nodes = len(self._nodes)
            model_ids = set(n.model_id for n in self._nodes.values())
            total_models = len(model_ids)

        uptime = time.time() - self._start_time

        return registry_pb2.ClusterInfoResponse(
            cluster_id=cc.cluster_id,
            name=cc.name,
            description=cc.description,
            operator=cc.operator,
            public_endpoint=cc.public_endpoint,
            total_nodes=total_nodes,
            total_models=total_models,
            uptime_seconds=uptime,
            default_config_json=cc.to_json(),
        )

    def SetClusterConfig(self, request, context):
        """Update the cluster config at runtime (operator use)."""
        if not request.config_json:
            return registry_pb2.SetClusterConfigResponse(
                success=False, message="config_json required")

        try:
            new_config = ClusterConfig.from_json(request.config_json)
        except (json.JSONDecodeError, TypeError) as e:
            return registry_pb2.SetClusterConfigResponse(
                success=False, message=f"Invalid config: {e}")

        errors = new_config.validate()
        if errors:
            return registry_pb2.SetClusterConfigResponse(
                success=False,
                message=f"Validation errors: {'; '.join(errors)}")

        # Preserve cluster_id from the original config
        new_config.cluster_id = self._cluster_config.cluster_id
        new_config.updated_at = time.time()
        self._cluster_config = new_config

        print(f"[Registry] Cluster config updated: name={new_config.name}")
        return registry_pb2.SetClusterConfigResponse(
            success=True, message="Cluster config updated")

    def ExchangePeers(self, request, context):
        """Gossip: merge caller's peer list with ours, return ours."""
        my_endpoint = self._cluster_config.public_endpoint
        now = time.time()

        # Merge caller's peers into our known set
        with self._peers_lock:
            for ep in request.known_peers:
                if ep and ep != my_endpoint and ep not in self._known_peers:
                    self._known_peers[ep] = {
                        "endpoint": ep,
                        "cluster_id": "",
                        "name": "",
                        "last_seen": now,
                    }

            # Build response with all our known peers (including ourselves)
            result_peers = []
            if my_endpoint:
                result_peers.append(registry_pb2.PeerInfo(
                    endpoint=my_endpoint,
                    cluster_id=self._cluster_config.cluster_id,
                    name=self._cluster_config.name,
                    last_seen=now,
                ))
            for info in self._known_peers.values():
                result_peers.append(registry_pb2.PeerInfo(
                    endpoint=info["endpoint"],
                    cluster_id=info.get("cluster_id", ""),
                    name=info.get("name", ""),
                    last_seen=info.get("last_seen", 0.0),
                ))

        return registry_pb2.ExchangePeersResponse(peers=result_peers)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def GetPricing(self, request, context):
        """Return per-token pricing for this cluster (or per-model override)."""
        cc = self._cluster_config
        input_price = cc.default_price_per_input_token
        output_price = cc.default_price_per_output_token
        model_id = request.model_id or ""

        # Check for per-model override
        if model_id:
            pool_cfg = self._pool_registry.get_config(model_id)
            if pool_cfg:
                input_price = pool_cfg.price_per_input_token
                output_price = pool_cfg.price_per_output_token

        # Token symbol
        symbol = "UNFED"
        if self._onchain_escrow:
            try:
                symbol = self._onchain_escrow.token_symbol()
            except Exception as e:
                logger.debug("Failed to fetch token symbol from chain: %s", e)

        return registry_pb2.GetPricingResponse(
            price_per_input_token=input_price,
            price_per_output_token=output_price,
            currency=symbol,
            model_id=model_id,
        )

    def ReportUsage(self, request, context):
        """Accept token usage report from a client/web-server.

        The PaymentContract accumulates these for the next settlement.
        """
        cost = (request.input_tokens * self._payment_contract.price_per_input_token
                + request.output_tokens * self._payment_contract.price_per_output_token)

        self._payment_contract.report_usage(
            request.input_tokens, request.output_tokens)

        return registry_pb2.ReportUsageResponse(
            cost=cost,
            accepted=True,
        )


class RegistryGossipServicer(inference_pb2_grpc.InferenceNodeServicer):
    """Minimal InferenceNode servicer for the registry to participate in gossip.

    The registry doesn't run inference — it only implements GossipBlock and
    GetBlocks so it can receive and serve share-chain blocks like any other peer.
    """

    def __init__(self, distributed_chain: DistributedShareChain):
        self._chain = distributed_chain

    def GossipBlock(self, request, context):
        """Accept a gossipped block from a compute node."""
        accepted, reason = self._chain.receive_block(request)
        return inference_pb2.GossipResponse(
            accepted=accepted,
            reason=reason,
            current_height=self._chain.chain.get_tip_height(),
        )

    def GetBlocks(self, request, context):
        """Return blocks from a given height for peer sync."""
        blocks = self._chain.chain.get_blocks_from(request.from_height)
        block_msgs = [
            block_to_proto(b, proposer_id="registry")
            for b in blocks
        ]
        return inference_pb2.GetBlocksResponse(
            blocks=block_msgs,
            chain_height=self._chain.chain.get_tip_height(),
        )

    # --- All other InferenceNode RPCs return UNIMPLEMENTED ---

    def Forward(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Registry does not serve inference")
        return inference_pb2.ForwardResponse()

    def GetShard(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Registry does not serve shards")
        return

    def Commit(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Registry does not serve inference")
        return inference_pb2.CommitResponse()


def _load_seed_peers(seeds_path: str | None = None) -> list[str]:
    """Load seed peers from seeds.json (excluding ourselves)."""
    if seeds_path is None:
        seeds_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "seeds.json")
    try:
        with open(seeds_path) as f:
            data = json.load(f)
        return data.get("registries", [])
    except FileNotFoundError:
        return []


def serve(port: int, cluster_config_path: str | None = None,
          no_chain: bool = False, tls_cert: str | None = None,
          tls_key: str | None = None):
    """Start the registry server.

    Args:
        port: Port to listen on.
        cluster_config_path: Path to cluster config JSON file.
        no_chain: If True, skip on-chain escrow and use in-memory simulation.
    """
    # Load cluster config
    cluster_cfg = None
    if cluster_config_path:
        cluster_cfg = ClusterConfig.from_file(cluster_config_path)
        print(f"[Registry] Loaded cluster config from {cluster_config_path}")
    else:
        cluster_cfg = ClusterConfig(name="default")
    cluster_cfg.ensure_id()
    # Auto-set public_endpoint if not configured
    if not cluster_cfg.public_endpoint:
        cluster_cfg.public_endpoint = f"localhost:{port}"

    # Load seed peers for gossip
    seed_peers = _load_seed_peers()
    # Remove ourselves from seeds
    my_endpoint = cluster_cfg.public_endpoint
    seed_peers = [p for p in seed_peers if p != my_endpoint]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    registry_servicer = RegistryServicer(
        cluster_config=cluster_cfg, seed_peers=seed_peers,
        no_chain=no_chain)
    registry_pb2_grpc.add_RegistryServicer_to_server(registry_servicer, server)

    # Also register the gossip servicer so the registry participates in
    # the P2P share-chain as a passive peer.
    gossip_servicer = RegistryGossipServicer(registry_servicer._distributed_chain)
    inference_pb2_grpc.add_InferenceNodeServicer_to_server(gossip_servicer, server)

    from network.tls import configure_server_port
    configure_server_port(server, "[::]", port, tls_cert, tls_key)
    server.start()

    print(f"[Registry] Running on port {port}")
    print(f"[Registry] Cluster: {cluster_cfg.name} "
          f"(id={cluster_cfg.cluster_id[:8]}...)")
    print(f"[Registry] Endpoint: {cluster_cfg.public_endpoint}")
    print(f"[Registry] Seed peers: {len(seed_peers)}")
    print(f"[Registry] Share-chain: passive gossip peer (nodes produce blocks)")
    print(f"[Registry] Nodes timeout after {config.NODE_TIMEOUT_SECONDS}s "
          f"without heartbeat")
    print(f"[Registry] Waiting for nodes to register...")

    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        print("\n[Registry] Shutting down...")
        registry_servicer._distributed_chain.stop()
        server.stop(grace=5)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    shutdown_event.wait()
    print("[Registry] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Registry Server")
    parser.add_argument("--port", type=int, default=50050,
                        help="Port to listen on")
    parser.add_argument("--cluster-config", type=str, default=None,
                        help="Path to cluster config JSON file")
    parser.add_argument("--no-chain", action="store_true",
                        help="Disable on-chain escrow; use in-memory simulation")
    parser.add_argument("--tls-cert", type=str, default=None,
                        help="Path to TLS certificate PEM file")
    parser.add_argument("--tls-key", type=str, default=None,
                        help="Path to TLS private key PEM file")
    args = parser.parse_args()
    serve(args.port, cluster_config_path=args.cluster_config,
          no_chain=args.no_chain, tls_cert=args.tls_cert,
          tls_key=args.tls_key)
