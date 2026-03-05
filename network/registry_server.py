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
import builtins
import hashlib
import json
import logging
import re
import signal
import sys
import os
import time
import threading
import uuid
from collections import defaultdict

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

from network.admission import resolve_mpc_required_flag
from network.he_dispute import (
    SAMPLED_AUDIT_REASON_CODE,
    is_anomaly_reason,
    verify_report_signature,
)
from network.share_auth import registration_pop_payload, verify_signature
from economics.share_chain import ShareChain, ComputeShare
from economics.payments import StakeManager, PaymentContract, SettlementProcessor
from economics.model_pools import PoolRegistry, PoolManifest
from economics.pool_config import PoolConfig
from economics.cluster_config import ClusterConfig, merge_with_defaults
from economics.distributed_chain import (
    DistributedShareChain, block_to_proto, proto_to_block,
)

MAX_REPORTED_INPUT_TOKENS = 200_000
MAX_REPORTED_OUTPUT_TOKENS = 50_000

_ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
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
    if "registry" in lower:
        code = "96"
    elif "daemon" in lower:
        code = "93"
    elif "vision" in lower:
        code = "95"
    elif "mpc" in lower:
        code = "94"
    elif "node" in lower:
        code = "92"
    elif "cheat" in lower:
        code = "91"
    elif "auto" in lower:
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


def _is_eth_address(value: str) -> bool:
    return isinstance(value, str) and bool(_ETH_ADDRESS_RE.fullmatch(value))


class NodeRecord:
    """In-memory record of a registered node."""

    def __init__(self, node_id: str, address: str, model_id: str,
                 shard_index: int, layer_start: int, layer_end: int,
                 has_embedding: bool, has_lm_head: bool,
                 public_key: bytes = b"", node_type: str = "compute",
                 share_signing_public_key: bytes = b"",
                 capability_json: str = "",
                 stake_identity: str = ""):
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
        self.share_signing_public_key = share_signing_public_key
        self.capability_json = capability_json
        self.stake_identity = stake_identity
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
            share_signing_public_key=self.share_signing_public_key,
            capability_json=self.capability_json,
            stake_identity=self.stake_identity,
        )


class VerifierRecord:
    """In-memory record of a registered verifier node."""

    def __init__(self, verifier_id: str, address: str, payout_address: str = ""):
        self.verifier_id = verifier_id
        self.address = address
        self.payout_address = payout_address
        self.last_heartbeat = time.time()
        self.registered_at = time.time()


class VerifierClaimRecord:
    """Registry-side lifecycle state for a verifier fraud claim."""

    def __init__(self, claim_id: str, idempotency_key: str, verifier_id: str,
                 ticket_id: str, shard_index: int):
        self.claim_id = claim_id
        self.idempotency_key = idempotency_key
        self.verifier_id = verifier_id
        self.ticket_id = ticket_id
        self.shard_index = shard_index
        self.status = "pending"  # pending | confirmed | rejected
        self.reason = ""
        self.fraud_node_id = ""
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.bonus_awarded = False
        self.slash_amount = 0.0
        self.cooldown_until_window = -1

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "idempotency_key": self.idempotency_key,
            "verifier_id": self.verifier_id,
            "ticket_id": self.ticket_id,
            "shard_index": self.shard_index,
            "status": self.status,
            "reason": self.reason,
            "fraud_node_id": self.fraud_node_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "bonus_awarded": self.bonus_awarded,
            "slash_amount": self.slash_amount,
            "cooldown_until_window": self.cooldown_until_window,
        }


class HEDisputeRecord:
    """Registry-side lifecycle state for HE sidecar disputes."""

    def __init__(self, dispute_ticket_id: str, report_id: str, idempotency_key: str):
        self.dispute_ticket_id = dispute_ticket_id
        self.report_id = report_id
        self.idempotency_key = idempotency_key
        self.status = "open"  # open|invalid|valid|insufficient_evidence|rejected
        self.reason = ""
        self.sidecar_node_id = ""
        self.sidecar_stake_identity = ""
        self.reporter_node_id = ""
        self.reporter_node_type = ""
        self.session_id = ""
        self.step = 0
        self.key_id = ""
        self.reason_code = ""
        self.request_payload_hash = ""
        self.response_payload_hash = ""
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.verifier_id = ""
        self.verdict = ""
        self.slash_amount = 0.0

    def to_dict(self) -> dict:
        return {
            "dispute_ticket_id": self.dispute_ticket_id,
            "report_id": self.report_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "reason": self.reason,
            "sidecar_node_id": self.sidecar_node_id,
            "sidecar_stake_identity": self.sidecar_stake_identity,
            "reporter_node_id": self.reporter_node_id,
            "reporter_node_type": self.reporter_node_type,
            "session_id": self.session_id,
            "step": self.step,
            "key_id": self.key_id,
            "reason_code": self.reason_code,
            "request_payload_hash": self.request_payload_hash,
            "response_payload_hash": self.response_payload_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verifier_id": self.verifier_id,
            "verdict": self.verdict,
            "slash_amount": self.slash_amount,
        }


class RegistryServicer(registry_pb2_grpc.RegistryServicer):
    """gRPC service implementation for the node registry."""

    def __init__(self, cluster_config: ClusterConfig | None = None,
                 seed_peers: list[str] | None = None,
                 no_chain: bool = False):
        self._nodes: dict[str, NodeRecord] = {}  # node_id -> NodeRecord
        self._lock = threading.Lock()
        self._verifiers: dict[str, VerifierRecord] = {}
        self._verifier_lock = threading.Lock()
        self._verifier_health_last_change = time.time()

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
        self._ticket_index: dict[str, registry_pb2.VerificationTicketProto] = {}
        self._ticket_owners_lock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._claims_by_id: dict[str, VerifierClaimRecord] = {}
        self._claim_id_by_idempotency: dict[str, str] = {}
        self._claim_rate_window: dict[tuple[str, int], int] = defaultdict(int)
        self._verifier_bonus_cooldown_until_window: dict[str, int] = {}
        self._verifier_penalty_events: list[dict] = []
        self._he_dispute_lock = threading.Lock()
        self._he_pending_tickets: list[registry_pb2.HEDisputeTicket] = []
        self._he_disputes_by_ticket: dict[str, HEDisputeRecord] = {}
        self._he_dispute_by_idempotency: dict[str, str] = {}
        self._he_report_rate_window: dict[tuple[str, int], int] = defaultdict(int)
        self._infra_work_lock = threading.Lock()
        self._daemon_work_window: dict[str, float] = defaultdict(float)
        self._verifier_ticket_window: dict[str, float] = defaultdict(float)
        self._verifier_bonus_window: dict[str, float] = defaultdict(float)
        self._daemon_height_cursor: dict[str, int] = {}
        self._last_daemon_payout_share: dict[str, float] = {}
        self._last_verifier_payout_share: dict[str, float] = {}

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
        # Finalization is driven from _settlement_loop so we can enforce
        # fail-closed verifier health gates.

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
        gate_last_state: tuple[bool, bool] | None = None

        while True:
            time.sleep(5)
            self._refresh_daemon_work_window()
            settlements = self._share_chain.get_settlements()
            verifier_ok = self._verifier_healthy()
            daemon_ok = self._daemon_healthy()
            infra_ok = verifier_ok and daemon_ok
            if gate_last_state is None or gate_last_state != (verifier_ok, daemon_ok):
                v_state = "healthy" if verifier_ok else "unhealthy"
                d_state = "healthy" if daemon_ok else "unhealthy"
                print(f"[Registry] Verifier gate={v_state}, daemon gate={d_state}")
                gate_last_state = (verifier_ok, daemon_ok)

            if len(settlements) > last_count:
                if not infra_ok:
                    # Fail closed: do not advance settlement cursor until infra recovers.
                    continue
                for s in settlements[last_count:]:
                    if s.total_shares <= 0:
                        print("[Registry] Skipping settlement with zero validated shares")
                        continue
                    daemon_recipient = self._select_daemon_recipient()
                    verifier_recipient = self._select_verifier_recipient()
                    daemon_work_map, verifier_work_map = self._consume_infra_work_maps()
                    self._settlement_processor.process_settlement(
                        s,
                        daemon_recipient=daemon_recipient,
                        verifier_recipient=verifier_recipient,
                        daemon_fee_bps=int(self._cluster_config.daemon_fee_bps),
                        verifier_fee_bps=int(self._cluster_config.verifier_fee_bps),
                        daemon_work_map=daemon_work_map,
                        verifier_work_map=verifier_work_map,
                    )

                    # Post on-chain if escrow is enabled
                    if self._onchain_escrow and s.node_shares:
                        try:
                            payout_map = self._payment_contract.settlement_payout_split(
                                s.settlement_hash
                            )
                            nodes = list(payout_map.keys())
                            invalid_nodes = [n for n in nodes if not _is_eth_address(n)]
                            if invalid_nodes:
                                print(
                                    "[Registry] Skipping on-chain settlement post: "
                                    f"non-EVM node IDs in settlement ({len(invalid_nodes)} invalid). "
                                    "This usually means replayed legacy share-chain data."
                                )
                                continue
                            # Convert token-denominated payouts to wei.
                            amounts = [
                                int(amount * 1e18)
                                for amount in payout_map.values()
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

            # Fail-closed: freeze finalization when verifier or daemon is unhealthy.
            if infra_ok:
                self._settlement_processor.run_finalization()

            # Finalize expired on-chain settlements
            if self._onchain_escrow and onchain_pending:
                now = time.time()
                still_pending = []
                for s_hash, deadline in onchain_pending:
                    if now >= deadline and verifier_ok and daemon_ok:
                        try:
                            self._onchain_escrow.finalize_settlement(s_hash)
                        except Exception as e:
                            print(f"[Registry] On-chain finalization "
                                  f"failed: {e}")
                    elif now >= deadline and (not verifier_ok or not daemon_ok):
                        still_pending.append((s_hash, deadline))
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

        self._remove_stale_verifiers()

    def _remove_stale_verifiers(self):
        timeout = max(1, int(self._cluster_config.verifier_heartbeat_timeout_seconds))
        now = time.time()
        with self._verifier_lock:
            stale = [
                vid for vid, rec in self._verifiers.items()
                if now - rec.last_heartbeat > timeout
            ]
            for vid in stale:
                rec = self._verifiers.pop(vid)
                print(f"[Registry] Removed stale verifier {vid[:8]}... ({rec.address})")
                self._verifier_health_last_change = now

    def _healthy_verifier_count(self) -> int:
        timeout = max(1, int(self._cluster_config.verifier_heartbeat_timeout_seconds))
        now = time.time()
        with self._verifier_lock:
            return sum(
                1 for rec in self._verifiers.values()
                if (now - rec.last_heartbeat) <= timeout
            )

    def _verifier_healthy(self) -> bool:
        return self._healthy_verifier_count() >= int(self._cluster_config.verifier_required_count)

    def _healthy_daemon_count(self) -> int:
        timeout = max(1, int(self._cluster_config.daemon_heartbeat_timeout_seconds))
        now = time.time()
        with self._lock:
            return sum(
                1 for rec in self._nodes.values()
                if rec.node_type == "daemon" and (now - rec.last_heartbeat) <= timeout
            )

    def _daemon_healthy(self) -> bool:
        return self._healthy_daemon_count() >= int(self._cluster_config.daemon_required_count)

    def _select_daemon_recipient(self) -> str:
        """Pick deterministic daemon payout recipient from healthy daemon set."""
        timeout = max(1, int(self._cluster_config.daemon_heartbeat_timeout_seconds))
        now = time.time()
        with self._lock:
            healthy = [
                rec.node_id for rec in self._nodes.values()
                if rec.node_type == "daemon" and (now - rec.last_heartbeat) <= timeout
            ]
        healthy.sort()
        return healthy[0] if healthy else ""

    def _select_verifier_recipient(self) -> str:
        """Pick deterministic verifier payout recipient from healthy verifier set."""
        timeout = max(1, int(self._cluster_config.verifier_heartbeat_timeout_seconds))
        now = time.time()
        with self._verifier_lock:
            healthy = [
                rec.payout_address for rec in self._verifiers.values()
                if rec.payout_address and (now - rec.last_heartbeat) <= timeout
            ]
        healthy.sort()
        return healthy[0] if healthy else ""

    def _refresh_daemon_work_window(self):
        """Accumulate daemon work units from newly produced daemon blocks."""
        timeout = max(1, int(self._cluster_config.daemon_heartbeat_timeout_seconds))
        now = time.time()
        with self._lock:
            daemons = [
                (rec.node_id, rec.address)
                for rec in self._nodes.values()
                if rec.node_type == "daemon" and (now - rec.last_heartbeat) <= timeout
            ]

        for daemon_id, daemon_addr in daemons:
            if not daemon_addr:
                continue
            from_height = self._daemon_height_cursor.get(daemon_id, 0) + 1
            try:
                channel = grpc.insecure_channel(daemon_addr, options=config.GRPC_OPTIONS)
                stub = inference_pb2_grpc.InferenceNodeStub(channel)
                resp = stub.GetBlocks(
                    inference_pb2.GetBlocksRequest(from_height=from_height),
                    timeout=3,
                )
                channel.close()
            except Exception:
                continue

            added_work = 0.0
            max_height = self._daemon_height_cursor.get(daemon_id, 0)
            for block in resp.blocks:
                added_work += float(len(block.shares))
                max_height = max(max_height, int(block.index))
            if added_work > 0:
                with self._infra_work_lock:
                    self._daemon_work_window[daemon_id] += added_work
            self._daemon_height_cursor[daemon_id] = max_height

    def _consume_infra_work_maps(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return and clear daemon/verifier work counters for current settlement."""
        now = time.time()
        daemon_timeout = max(1, int(self._cluster_config.daemon_heartbeat_timeout_seconds))
        verifier_timeout = max(1, int(self._cluster_config.verifier_heartbeat_timeout_seconds))

        with self._lock:
            healthy_daemon_ids = {
                rec.node_id
                for rec in self._nodes.values()
                if rec.node_type == "daemon" and (now - rec.last_heartbeat) <= daemon_timeout
            }
        with self._verifier_lock:
            healthy_verifier_payout = {
                vid: rec.payout_address
                for vid, rec in self._verifiers.items()
                if rec.payout_address and (now - rec.last_heartbeat) <= verifier_timeout
            }

        with self._infra_work_lock:
            daemon_raw = dict(self._daemon_work_window)
            verifier_ticket_raw = dict(self._verifier_ticket_window)
            verifier_bonus_raw = dict(self._verifier_bonus_window)
            self._daemon_work_window.clear()
            self._verifier_ticket_window.clear()
            self._verifier_bonus_window.clear()

        daemon_map: dict[str, float] = {}
        for daemon_id, units in daemon_raw.items():
            if daemon_id in healthy_daemon_ids and float(units) > 0:
                daemon_map[daemon_id] = float(units)

        bonus_weight = float(getattr(self._cluster_config, "verifier_fraud_bonus_weight", 3.0))
        verifier_map: dict[str, float] = {}
        for verifier_id, payout_addr in healthy_verifier_payout.items():
            ticket_units = float(verifier_ticket_raw.get(verifier_id, 0.0))
            bonus_units = float(verifier_bonus_raw.get(verifier_id, 0.0))
            total_units = ticket_units + (bonus_units * bonus_weight)
            if total_units > 0:
                verifier_map[payout_addr] = verifier_map.get(payout_addr, 0.0) + total_units

        with self._infra_work_lock:
            self._last_daemon_payout_share = self._normalize_map(daemon_map)
            self._last_verifier_payout_share = self._normalize_map(verifier_map)
        return daemon_map, verifier_map

    @staticmethod
    def _normalize_map(weight_map: dict[str, float]) -> dict[str, float]:
        """Convert weights to [0..1] payout proportions."""
        total = sum(v for v in weight_map.values() if float(v) > 0)
        if total <= 0:
            return {}
        return {
            key: float(value) / total
            for key, value in weight_map.items()
            if float(value) > 0
        }

    def _effective_verifier_config_json(self) -> str:
        policy = {
            "poll_interval_seconds": float(self._cluster_config.verifier_poll_interval_seconds),
            "max_tickets_per_poll": int(self._cluster_config.verifier_max_tickets_per_poll),
            "sampling_baseline": float(self._cluster_config.verifier_sampling_baseline),
            "sampling_suspicious": float(self._cluster_config.verifier_sampling_suspicious),
            "sampling_max": float(self._cluster_config.verifier_sampling_max),
            "adaptive_enabled": bool(self._cluster_config.verifier_adaptive_enabled),
            "heartbeat_timeout_seconds": int(self._cluster_config.verifier_heartbeat_timeout_seconds),
            "required_count": int(self._cluster_config.verifier_required_count),
            "bonus_requires_confirmation": bool(
                getattr(self._cluster_config, "verifier_bonus_requires_confirmation", True)
            ),
            "false_claim_slash_bps": int(
                getattr(self._cluster_config, "verifier_false_claim_slash_bps", 500)
            ),
            "claim_rate_limit_per_window": int(
                getattr(self._cluster_config, "verifier_claim_rate_limit_per_window", 64)
            ),
            "bonus_cooldown_windows": int(
                getattr(self._cluster_config, "verifier_bonus_cooldown_windows", 1)
            ),
            "he_dispute_sampling_rate": float(
                getattr(self._cluster_config, "he_dispute_sampling_rate", config.HE_DISPUTE_SAMPLING_RATE)
            ),
            "he_dispute_rollout_stage": str(
                getattr(self._cluster_config, "he_dispute_rollout_stage", config.HE_DISPUTE_ROLLOUT_STAGE)
            ),
        }
        return json.dumps(policy)

    def _current_settlement_window(self) -> int:
        blocks = max(1, int(self._share_chain.settlement_blocks))
        height = max(0, int(self._share_chain.get_tip_height()))
        return height // blocks

    def _current_he_dispute_window(self) -> int:
        window_seconds = int(
            max(1, getattr(self._cluster_config, "he_dispute_window_seconds", config.HE_DISPUTE_WINDOW_SECONDS))
        )
        return int(time.time()) // window_seconds

    def _he_rollout_stage(self) -> str:
        stage = str(
            getattr(self._cluster_config, "he_dispute_rollout_stage", config.HE_DISPUTE_ROLLOUT_STAGE)
        ).strip().lower()
        if stage not in {"shadow", "soft", "enforced"}:
            return "shadow"
        return stage

    def _validate_he_suspicion_report(self, request) -> tuple[bool, str]:
        if not request.reporter_node_id:
            return False, "missing_reporter_node_id"
        if not request.session_id:
            return False, "missing_session_id"
        if int(request.step) < 0:
            return False, "invalid_step"
        if not request.reason_code:
            return False, "missing_reason_code"
        if not request.request_payload_hash:
            return False, "missing_request_payload_hash"
        if not request.response_payload_hash:
            return False, "missing_response_payload_hash"
        payload = {
            "report_id": request.report_id,
            "reporter_node_id": request.reporter_node_id,
            "reporter_node_type": request.reporter_node_type,
            "sidecar_node_id": request.sidecar_node_id,
            "sidecar_stake_identity": request.sidecar_stake_identity,
            "session_id": request.session_id,
            "step": int(request.step),
            "key_id": request.key_id,
            "reason_code": request.reason_code,
            "he_compute_format": request.he_compute_format,
            "request_payload_hash": request.request_payload_hash,
            "response_payload_hash": request.response_payload_hash,
            "timestamp": float(request.timestamp),
        }
        if not verify_report_signature(
            bytes(request.reporter_signing_public_key),
            payload,
            bytes(request.reporter_signature),
        ):
            return False, "invalid_report_signature"
        return True, "ok"

    def _resolve_sidecar_stake_identity(self, sidecar_node_id: str, provided_identity: str) -> str:
        if provided_identity and _is_eth_address(provided_identity):
            return provided_identity
        with self._lock:
            rec = self._nodes.get(sidecar_node_id)
            if rec and rec.stake_identity and _is_eth_address(rec.stake_identity):
                return rec.stake_identity
            if rec and _is_eth_address(rec.node_id):
                return rec.node_id
        return provided_identity or sidecar_node_id

    def _claim_status_snapshot(self, limit: int = 200) -> list[dict]:
        with self._claim_lock:
            claims = sorted(
                self._claims_by_id.values(),
                key=lambda c: c.updated_at,
                reverse=True,
            )
            return [c.to_dict() for c in claims[:max(1, limit)]]

    def _penalty_events_snapshot(self, limit: int = 200) -> list[dict]:
        with self._claim_lock:
            return list(self._verifier_penalty_events[-max(1, limit):])

    def _record_penalty_event(self, event: dict):
        with self._claim_lock:
            self._verifier_penalty_events.append(event)
            if len(self._verifier_penalty_events) > 1000:
                self._verifier_penalty_events = self._verifier_penalty_events[-1000:]

    def _claim_in_bonus_cooldown(self, verifier_id: str, current_window: int) -> bool:
        with self._claim_lock:
            until = int(self._verifier_bonus_cooldown_until_window.get(verifier_id, -1))
        return current_window <= until

    def _set_claim_cooldown(self, verifier_id: str, current_window: int):
        cooldown_windows = int(
            max(0, getattr(self._cluster_config, "verifier_bonus_cooldown_windows", 1))
        )
        until = current_window + cooldown_windows - 1
        with self._claim_lock:
            self._verifier_bonus_cooldown_until_window[verifier_id] = max(
                until,
                int(self._verifier_bonus_cooldown_until_window.get(verifier_id, -1)),
            )

    def _validate_claim_evidence(self, request, verifier_id: str) -> tuple[bool, str, str]:
        """Validate ticket-bound evidence for a fraud claim."""
        with self._ticket_owners_lock:
            owner = self._ticket_owners.get(request.ticket_id, "")
            ticket = self._ticket_index.get(request.ticket_id)
        if not ticket:
            return False, "unknown_ticket", owner
        if request.evidence_session_id and request.evidence_session_id != request.ticket_id:
            return False, "session_binding_mismatch", owner
        if request.evidence_node_id and owner and request.evidence_node_id != owner:
            return False, "node_binding_mismatch", owner
        expected_input_hash = hashlib.sha256(bytes(ticket.input_data)).hexdigest()
        expected_output_hash = hashlib.sha256(bytes(ticket.expected_output_data)).hexdigest()
        if request.input_hash and request.input_hash != expected_input_hash:
            return False, "input_hash_mismatch", owner
        if request.expected_output_hash and request.expected_output_hash != expected_output_hash:
            return False, "expected_output_hash_mismatch", owner
        if request.expected_token and ticket.has_expected_token:
            if int(request.expected_token) != int(ticket.expected_token):
                return False, "expected_token_mismatch", owner
        # True fraud requires mismatch between claimed expected and verifier output.
        if request.actual_output_hash and request.actual_output_hash == expected_output_hash:
            return False, "no_output_mismatch", owner
        if not request.actual_output_hash:
            return False, "missing_actual_output_hash", owner
        return True, "ok", owner

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

        # --- Share signing key proof-of-possession gate ---
        if request.node_type in ("compute", "mpc"):
            if len(request.share_signing_public_key) != 32:
                return registry_pb2.RegisterResponse(
                    success=False,
                    message="share_signing_public_key must be 32 bytes",
                )
            if not request.share_signing_pop:
                return registry_pb2.RegisterResponse(
                    success=False,
                    message="share_signing_pop is required",
                )
            pop_payload = registration_pop_payload(
                node_id=request.node_id,
                address=request.address,
                model_id=request.model_id,
                shard_index=request.shard_index,
                node_type=request.node_type or "compute",
            )
            if not verify_signature(
                bytes(request.share_signing_public_key),
                pop_payload,
                bytes(request.share_signing_pop),
            ):
                return registry_pb2.RegisterResponse(
                    success=False,
                    message="invalid share_signing_pop signature",
                )
        if request.node_type == "he_sidecar":
            if not (request.capability_json or "").strip():
                return registry_pb2.RegisterResponse(
                    success=False,
                    message="he_sidecar requires capability_json",
                )
            stake_identity = (request.stake_identity or "").strip()
            if not stake_identity:
                stake_identity = (request.node_id or "").strip()
            if not _is_eth_address(stake_identity):
                return registry_pb2.RegisterResponse(
                    success=False,
                    message="he_sidecar requires EVM stake_identity (or node_id as EVM address)",
                )
            if self._onchain_escrow:
                try:
                    if not self._onchain_escrow.is_eligible(stake_identity):
                        min_stake = self._onchain_escrow.min_stake()
                        return registry_pb2.RegisterResponse(
                            success=False,
                            message=f"he_sidecar stake too low. Minimum: {min_stake} wei",
                        )
                except Exception as e:
                    return registry_pb2.RegisterResponse(
                        success=False,
                        message=f"he_sidecar stake check failed: {e}",
                    )

        with self._lock:
            effective_stake_identity = (request.stake_identity or "").strip()
            if not effective_stake_identity and request.node_type == "he_sidecar":
                effective_stake_identity = (request.node_id or "").strip()
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
                share_signing_public_key=bytes(request.share_signing_public_key),
                capability_json=(request.capability_json or "").strip(),
                stake_identity=effective_stake_identity,
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

    def RegisterVerifier(self, request, context):
        """Register a verifier process and return effective verifier config."""
        verifier_id = (request.verifier_id or "").strip()
        if not verifier_id:
            return registry_pb2.RegisterVerifierResponse(
                success=False, message="verifier_id is required"
            )
        payout_address = ""
        if _is_eth_address(verifier_id):
            payout_address = verifier_id
        elif _is_eth_address(request.address or ""):
            payout_address = request.address

        # When on-chain escrow is active and verifier payouts are enabled,
        # verifier recipient must be a staked EVM address.
        if self._onchain_escrow and int(self._cluster_config.verifier_fee_bps) > 0:
            if not payout_address:
                return registry_pb2.RegisterVerifierResponse(
                    success=False,
                    message="verifier must provide EVM payout identity when verifier_fee_bps > 0",
                )
            if not self._onchain_escrow.is_eligible(payout_address):
                min_stake = self._onchain_escrow.min_stake()
                return registry_pb2.RegisterVerifierResponse(
                    success=False,
                    message=f"verifier stake too low. Minimum: {min_stake} wei",
                )
        with self._verifier_lock:
            self._verifiers[verifier_id] = VerifierRecord(
                verifier_id=verifier_id,
                address=request.address or "",
                payout_address=payout_address,
            )
            self._verifier_health_last_change = time.time()
        print(f"[Registry] Verifier registered: {verifier_id[:8]}... at {request.address}")
        return registry_pb2.RegisterVerifierResponse(
            success=True,
            message="registered",
            config_json=self._effective_verifier_config_json(),
        )

    def VerifierHeartbeat(self, request, context):
        """Update verifier heartbeat and return latest effective config."""
        verifier_id = (request.verifier_id or "").strip()
        if not verifier_id:
            return registry_pb2.VerifierHeartbeatResponse(
                acknowledged=False,
                config_json=self._effective_verifier_config_json(),
            )
        with self._verifier_lock:
            rec = self._verifiers.get(verifier_id)
            if rec is None:
                return registry_pb2.VerifierHeartbeatResponse(
                    acknowledged=False,
                    config_json=self._effective_verifier_config_json(),
                )
            rec.last_heartbeat = time.time()
        return registry_pb2.VerifierHeartbeatResponse(
            acknowledged=True,
            config_json=self._effective_verifier_config_json(),
        )

    def GetVerifierConfig(self, request, context):
        """Return current effective verifier config."""
        return registry_pb2.GetVerifierConfigResponse(
            success=True,
            message="ok",
            config_json=self._effective_verifier_config_json(),
        )

    def GetVerifierHealth(self, request, context):
        """Return verifier liveness health for fail-closed gates."""
        healthy_count = self._healthy_verifier_count()
        required_count = int(self._cluster_config.verifier_required_count)
        return registry_pb2.GetVerifierHealthResponse(
            healthy_verifier_count=healthy_count,
            required_verifier_count=required_count,
            healthy=healthy_count >= required_count,
        )

    def GetInfraTelemetry(self, request, context):
        """Return routing/work/payout telemetry for daemon and verifier infra."""
        healthy_daemon_count = self._healthy_daemon_count()
        healthy_verifier_count = self._healthy_verifier_count()
        required_daemon_count = int(self._cluster_config.daemon_required_count)
        required_verifier_count = int(self._cluster_config.verifier_required_count)
        with self._infra_work_lock:
            daemon_work_window = dict(self._daemon_work_window)
            verifier_work_window = {
                verifier_id: (
                    float(self._verifier_ticket_window.get(verifier_id, 0.0))
                    + float(self._verifier_bonus_window.get(verifier_id, 0.0))
                    * float(getattr(self._cluster_config, "verifier_fraud_bonus_weight", 3.0))
                )
                for verifier_id in set(self._verifier_ticket_window.keys())
                | set(self._verifier_bonus_window.keys())
            }
            daemon_payout_share = dict(self._last_daemon_payout_share)
            verifier_payout_share = dict(self._last_verifier_payout_share)
        claim_status = self._claim_status_snapshot(limit=250)
        penalty_events = self._penalty_events_snapshot(limit=250)
        return registry_pb2.GetInfraTelemetryResponse(
            healthy_daemon_count=healthy_daemon_count,
            required_daemon_count=required_daemon_count,
            healthy_verifier_count=healthy_verifier_count,
            required_verifier_count=required_verifier_count,
            selected_daemon_recipient=self._select_daemon_recipient(),
            selected_verifier_recipient=self._select_verifier_recipient(),
            daemon_work_window_json=json.dumps(daemon_work_window),
            verifier_work_window_json=json.dumps(verifier_work_window),
            daemon_payout_share_json=json.dumps(daemon_payout_share),
            verifier_payout_share_json=json.dumps(verifier_payout_share),
            verifier_claim_status_json=json.dumps(claim_status),
            verifier_penalty_events_json=json.dumps(penalty_events),
        )

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

        mpc_required = resolve_mpc_required_flag()

        # Group text-serving nodes (compute + mpc) by model_id
        model_nodes: dict[str, list[NodeRecord]] = {}
        for n in nodes:
            if n.node_type in ("compute", "mpc"):
                model_nodes.setdefault(n.model_id, []).append(n)

        models = []
        for model_id, mnodes in model_nodes.items():
            # Group by shard to compute text coverage
            shard_set: set[int] = set()
            for n in mnodes:
                shard_set.add(n.shard_index)

            max_shard = max(n.shard_index for n in mnodes)
            total_shards = max_shard + 1
            covered = len(shard_set)
            mpc_available = any(
                n.node_type == "mpc" and n.shard_index == 0 for n in mnodes
            )
            can_serve = (covered == total_shards) and (
                (not mpc_required) or mpc_available
            )

            models.append(registry_pb2.ModelInfo(
                model_id=model_id,
                total_nodes=len(mnodes),
                total_shards=total_shards,
                covered_shards=covered,
                can_serve=can_serve,
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
                self._ticket_index[ticket.ticket_id] = ticket

        # Record compute shares in the share-chain
        # Each ticket represents verified compute work
        for ticket in request.tickets:
            share = ComputeShare(
                node_id=request.node_id,
                shard_index=ticket.shard_index,
                session_id=ticket.ticket_id,  # use ticket_id as proxy
                activation_hash=str(hash(ticket.expected_output_data))[:16],
                tokens_processed=1,
                validated=False,
            )
            self._share_chain.add_share(share)

        if count > 0:
            print(f"[Registry] Received {count} verification ticket(s) "
                  f"from node {request.node_id[:8]}...")

        return registry_pb2.SubmitTicketsResponse(accepted=count)

    def GetPendingTickets(self, request, context):
        """Return and clear pending verification tickets for verifiers."""
        verifier_id = (getattr(request, "verifier_id", "") or "").strip()
        with self._tickets_lock:
            limit = request.max_tickets if request.max_tickets > 0 else len(self._pending_tickets)
            tickets = self._pending_tickets[:limit]
            self._pending_tickets = self._pending_tickets[limit:]
        if verifier_id and tickets:
            with self._infra_work_lock:
                self._verifier_ticket_window[verifier_id] += float(len(tickets))

        return registry_pb2.GetPendingTicketsResponse(tickets=tickets)

    def SubmitFraudProof(self, request, context):
        """Accept a fraud proof, validate evidence, then confirm/reject claim."""
        verifier_id = (getattr(request, "verifier_id", "") or "").strip()
        claim_id = (getattr(request, "claim_id", "") or "").strip() or str(uuid.uuid4())
        idem = (getattr(request, "idempotency_key", "") or "").strip() or (
            f"{verifier_id}:{request.ticket_id}:{request.shard_index}"
        )
        now = time.time()
        current_window = self._current_settlement_window()

        with self._claim_lock:
            existing_id = self._claim_id_by_idempotency.get(idem)
            if existing_id and existing_id in self._claims_by_id:
                existing = self._claims_by_id[existing_id]
                return registry_pb2.FraudProofResponse(
                    accepted=(existing.status == "confirmed"),
                    claim_id=existing.claim_id,
                    claim_status=existing.status,
                    message=f"idempotent_replay:{existing.reason}",
                )
            self._claim_id_by_idempotency[idem] = claim_id
            claim = VerifierClaimRecord(
                claim_id=claim_id,
                idempotency_key=idem,
                verifier_id=verifier_id,
                ticket_id=request.ticket_id,
                shard_index=request.shard_index,
            )
            self._claims_by_id[claim_id] = claim
            self._claim_rate_window[(verifier_id, current_window)] += 1
            claim_count = self._claim_rate_window[(verifier_id, current_window)]

        limit = int(max(1, getattr(
            self._cluster_config, "verifier_claim_rate_limit_per_window", 64)))
        if claim_count > limit:
            claim.status = "rejected"
            claim.reason = "rate_limited"
            claim.updated_at = now
            self._record_penalty_event({
                "ts": now,
                "kind": "claim_rate_limited",
                "verifier_id": verifier_id,
                "claim_id": claim_id,
                "window": current_window,
                "limit": limit,
            })
            return registry_pb2.FraudProofResponse(
                accepted=False,
                claim_id=claim_id,
                claim_status=claim.status,
                message=claim.reason,
            )

        is_valid, reason, fraud_node_id = self._validate_claim_evidence(request, verifier_id)
        claim.fraud_node_id = fraud_node_id
        claim.reason = reason

        with self._fraud_lock:
            self._fraud_proofs.append(request)

        if is_valid:
            claim.status = "confirmed"
            challenged = False
            settlements = self._share_chain.get_settlements()
            if fraud_node_id:
                for s in settlements:
                    if fraud_node_id in s.node_shares:
                        result = self._payment_contract.challenge_settlement(
                            settlement_hash=s.settlement_hash,
                            fraud_node_id=fraud_node_id,
                        )
                        if result:
                            challenged = True
                            break
                if self._onchain_escrow:
                    try:
                        self._onchain_escrow.slash_node(fraud_node_id)
                    except Exception as e:
                        print(f"[Registry] WARNING: On-chain slash failed "
                              f"for {fraud_node_id[:8]}...: {e}")
            claim.reason = "confirmed" if challenged or fraud_node_id else "confirmed_no_owner"
            bonus_requires_confirmation = bool(
                getattr(self._cluster_config, "verifier_bonus_requires_confirmation", True)
            )
            in_cooldown = self._claim_in_bonus_cooldown(verifier_id, current_window)
            if verifier_id and (not in_cooldown) and bonus_requires_confirmation:
                with self._infra_work_lock:
                    self._verifier_bonus_window[verifier_id] += 1.0
                claim.bonus_awarded = True
            elif verifier_id and in_cooldown:
                claim.reason = "confirmed_bonus_blocked_by_cooldown"
        else:
            claim.status = "rejected"
            slash_bps = int(max(0, getattr(
                self._cluster_config, "verifier_false_claim_slash_bps", 500)))
            slash_fraction = float(slash_bps) / 10000.0
            slash_amount = 0.0
            if slash_fraction > 0 and verifier_id:
                payout_addr = ""
                with self._verifier_lock:
                    rec = self._verifiers.get(verifier_id)
                    if rec:
                        payout_addr = rec.payout_address or rec.address or rec.verifier_id
                target = payout_addr or verifier_id
                if self._onchain_escrow and _is_eth_address(target):
                    try:
                        self._onchain_escrow.slash_node(target)
                    except Exception:
                        pass
                slash_amount = self._payment_contract.stakes.slash(
                    target, pool_slash_fraction=slash_fraction
                )
            claim.slash_amount = float(slash_amount)
            self._set_claim_cooldown(verifier_id, current_window)
            claim.cooldown_until_window = self._verifier_bonus_cooldown_until_window.get(
                verifier_id, -1
            )
            self._record_penalty_event({
                "ts": now,
                "kind": "false_claim_penalty",
                "verifier_id": verifier_id,
                "claim_id": claim_id,
                "reason": reason,
                "slash_amount": float(slash_amount),
                "cooldown_until_window": claim.cooldown_until_window,
            })

        claim.updated_at = time.time()
        print(f"[Registry] Claim {claim.claim_id[:8]}... {claim.status}: {claim.reason}")
        return registry_pb2.FraudProofResponse(
            accepted=(claim.status == "confirmed"),
            claim_id=claim.claim_id,
            claim_status=claim.status,
            message=claim.reason,
        )

    def SubmitHESuspicionReport(self, request, context):
        """Accept signed HE suspicion reports and open verifier challenge tickets."""
        report_id = (getattr(request, "report_id", "") or "").strip() or str(uuid.uuid4())
        idem = (getattr(request, "idempotency_key", "") or "").strip() or (
            f"{request.reporter_node_id}:{request.session_id}:{request.step}:{request.reason_code}"
        )
        window = self._current_he_dispute_window()
        reporter = (request.reporter_node_id or "").strip()

        with self._he_dispute_lock:
            existing_ticket_id = self._he_dispute_by_idempotency.get(idem)
            if existing_ticket_id and existing_ticket_id in self._he_disputes_by_ticket:
                existing = self._he_disputes_by_ticket[existing_ticket_id]
                return registry_pb2.HESuspicionReportResponse(
                    accepted=True,
                    report_id=existing.report_id,
                    dispute_ticket_id=existing.dispute_ticket_id,
                    status="duplicate",
                    message="idempotent_replay",
                )
            self._he_report_rate_window[(reporter, window)] += 1
            report_count = self._he_report_rate_window[(reporter, window)]

        limit = int(
            max(1, getattr(
                self._cluster_config,
                "he_dispute_report_rate_limit_per_window",
                config.HE_DISPUTE_REPORT_RATE_LIMIT_PER_WINDOW,
            ))
        )
        if report_count > limit:
            self._record_penalty_event({
                "ts": time.time(),
                "kind": "he_report_rate_limited",
                "reporter_node_id": reporter,
                "report_id": report_id,
                "window": window,
                "limit": limit,
            })
            return registry_pb2.HESuspicionReportResponse(
                accepted=False,
                report_id=report_id,
                status="rate_limited",
                message="report_rate_limited",
            )

        valid, reason = self._validate_he_suspicion_report(request)
        if not valid:
            return registry_pb2.HESuspicionReportResponse(
                accepted=False,
                report_id=report_id,
                status="rejected",
                message=reason,
            )

        dispute_ticket_id = f"he-dsp-{uuid.uuid4().hex[:16]}"
        rec = HEDisputeRecord(
            dispute_ticket_id=dispute_ticket_id,
            report_id=report_id,
            idempotency_key=idem,
        )
        rec.sidecar_node_id = (request.sidecar_node_id or "").strip()
        rec.sidecar_stake_identity = self._resolve_sidecar_stake_identity(
            rec.sidecar_node_id,
            (request.sidecar_stake_identity or "").strip(),
        )
        rec.reporter_node_id = reporter
        rec.reporter_node_type = (request.reporter_node_type or "").strip()
        rec.session_id = request.session_id
        rec.step = int(request.step)
        rec.key_id = request.key_id
        rec.reason_code = request.reason_code
        rec.request_payload_hash = request.request_payload_hash
        rec.response_payload_hash = request.response_payload_hash

        ticket = registry_pb2.HEDisputeTicket(
            dispute_ticket_id=dispute_ticket_id,
            report_id=report_id,
            sidecar_node_id=rec.sidecar_node_id,
            sidecar_stake_identity=rec.sidecar_stake_identity,
            reporter_node_id=rec.reporter_node_id,
            reporter_node_type=rec.reporter_node_type,
            session_id=rec.session_id,
            step=rec.step,
            key_id=rec.key_id,
            reason_code=rec.reason_code,
            he_compute_format=request.he_compute_format,
            request_payload_hash=rec.request_payload_hash,
            response_payload_hash=rec.response_payload_hash,
            evidence_json=request.evidence_json,
            report_timestamp=float(request.timestamp or time.time()),
        )
        with self._he_dispute_lock:
            self._he_pending_tickets.append(ticket)
            self._he_disputes_by_ticket[dispute_ticket_id] = rec
            self._he_dispute_by_idempotency[idem] = dispute_ticket_id
        self._record_penalty_event({
            "ts": time.time(),
            "kind": "he_dispute_opened",
            "dispute_ticket_id": dispute_ticket_id,
            "report_id": report_id,
            "reason_code": rec.reason_code,
            "sampled_audit": rec.reason_code == SAMPLED_AUDIT_REASON_CODE,
            "anomaly_escalation": is_anomaly_reason(rec.reason_code),
        })
        return registry_pb2.HESuspicionReportResponse(
            accepted=True,
            report_id=report_id,
            dispute_ticket_id=dispute_ticket_id,
            status="accepted",
            message="challenge_opened",
        )

    def GetPendingHEDisputes(self, request, context):
        """Return and clear pending HE dispute tickets for verifier adjudication."""
        verifier_id = (getattr(request, "verifier_id", "") or "").strip()
        with self._he_dispute_lock:
            limit = request.max_tickets if request.max_tickets > 0 else len(self._he_pending_tickets)
            tickets = self._he_pending_tickets[:limit]
            self._he_pending_tickets = self._he_pending_tickets[limit:]
        if verifier_id and tickets:
            with self._infra_work_lock:
                self._verifier_ticket_window[verifier_id] += float(len(tickets))
        return registry_pb2.GetPendingHEDisputesResponse(tickets=tickets)

    def SubmitHEVerifierVerdict(self, request, context):
        """Apply verifier adjudication verdict for HE sidecar disputes."""
        ticket_id = (request.dispute_ticket_id or "").strip()
        if not ticket_id:
            return registry_pb2.HEVerifierVerdictResponse(
                accepted=False,
                dispute_ticket_id=ticket_id,
                status="rejected",
                message="missing_dispute_ticket_id",
            )
        verdict = (request.verdict or "").strip().lower()
        if verdict not in {"valid", "invalid", "insufficient_evidence"}:
            return registry_pb2.HEVerifierVerdictResponse(
                accepted=False,
                dispute_ticket_id=ticket_id,
                status="rejected",
                message="invalid_verdict",
            )
        with self._he_dispute_lock:
            rec = self._he_disputes_by_ticket.get(ticket_id)
            if rec is None:
                return registry_pb2.HEVerifierVerdictResponse(
                    accepted=False,
                    dispute_ticket_id=ticket_id,
                    status="rejected",
                    message="unknown_dispute_ticket",
                )
            if rec.status != "open":
                return registry_pb2.HEVerifierVerdictResponse(
                    accepted=True,
                    dispute_ticket_id=ticket_id,
                    status="duplicate",
                    message=f"already_{rec.status}",
                )

        rec.verifier_id = (request.verifier_id or "").strip()
        rec.verdict = verdict
        rec.updated_at = time.time()
        rec.reason = (request.reason or "").strip()
        slash_amount = 0.0
        status = "confirmed_valid"

        if verdict == "invalid":
            stage = self._he_rollout_stage()
            configured_fraction = float(
                max(0.0, min(1.0, getattr(self._cluster_config, "he_dispute_slash_fraction", 0.5)))
            )
            apply_fraction = 0.0
            if stage == "soft":
                apply_fraction = configured_fraction * 0.25
            elif stage == "enforced":
                apply_fraction = configured_fraction
            target = self._resolve_sidecar_stake_identity(
                request.sidecar_node_id or rec.sidecar_node_id,
                request.sidecar_stake_identity or rec.sidecar_stake_identity,
            )
            if apply_fraction > 0.0:
                if self._onchain_escrow and _is_eth_address(target):
                    try:
                        self._onchain_escrow.slash_node(target)
                    except Exception as e:
                        print(f"[Registry] WARNING: HE sidecar on-chain slash failed for {target}: {e}")
                slash_amount = float(
                    self._payment_contract.stakes.slash(target, pool_slash_fraction=apply_fraction)
                )
            rec.slash_amount = slash_amount
            rec.status = "invalid"
            status = "confirmed_invalid"
            self._record_penalty_event({
                "ts": rec.updated_at,
                "kind": "he_dispute_invalid",
                "stage": stage,
                "dispute_ticket_id": ticket_id,
                "sidecar_stake_identity": target,
                "slash_amount": slash_amount,
                "slash_fraction_applied": apply_fraction,
                "reason_code": rec.reason_code,
            })
        elif verdict == "insufficient_evidence":
            rec.status = "insufficient_evidence"
            status = "insufficient_evidence"
            self._record_penalty_event({
                "ts": rec.updated_at,
                "kind": "he_dispute_insufficient_evidence",
                "dispute_ticket_id": ticket_id,
                "reason_code": rec.reason_code,
            })
        else:
            rec.status = "valid"
            status = "confirmed_valid"
            self._record_penalty_event({
                "ts": rec.updated_at,
                "kind": "he_dispute_valid",
                "dispute_ticket_id": ticket_id,
                "reason_code": rec.reason_code,
            })

        if rec.verifier_id:
            with self._infra_work_lock:
                if verdict == "invalid":
                    self._verifier_bonus_window[rec.verifier_id] += 1.0
                else:
                    self._verifier_ticket_window[rec.verifier_id] += 0.25

        return registry_pb2.HEVerifierVerdictResponse(
            accepted=True,
            dispute_ticket_id=ticket_id,
            status=status,
            message=rec.reason or rec.status,
        )

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
        if request.input_tokens < 0 or request.output_tokens < 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Token counts must be non-negative")
            return registry_pb2.ReportUsageResponse(cost=0.0, accepted=False)

        if request.input_tokens > MAX_REPORTED_INPUT_TOKENS:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"input_tokens exceeds limit ({MAX_REPORTED_INPUT_TOKENS})")
            return registry_pb2.ReportUsageResponse(cost=0.0, accepted=False)

        if request.output_tokens > MAX_REPORTED_OUTPUT_TOKENS:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"output_tokens exceeds limit ({MAX_REPORTED_OUTPUT_TOKENS})")
            return registry_pb2.ReportUsageResponse(cost=0.0, accepted=False)

        cost = (request.input_tokens * self._payment_contract.price_per_input_token
                + request.output_tokens * self._payment_contract.price_per_output_token)

        self._payment_contract.report_usage(
            request.input_tokens, request.output_tokens)
        _simple_log(
            "[Registry] usage accepted "
            f"model={request.model_id or 'default'} "
            f"in={request.input_tokens} out={request.output_tokens} "
            f"cost={cost:.6f}"
        )

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
    print(
        "[Registry] Admission policy: "
        f"UNFED_REQUIRE_MPC={'1' if resolve_mpc_required_flag() else '0'}"
    )
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
    parser.add_argument("--quiet", action="store_true",
                        help="Disable simple usage/event logs")
    args = parser.parse_args()
    if args.quiet:
        import builtins
        builtins.print = lambda *a, **k: None
        _SIMPLE_LOGS_ENABLED = False
    serve(args.port, cluster_config_path=args.cluster_config,
          no_chain=args.no_chain, tls_cert=args.tls_cert,
          tls_key=args.tls_key)
