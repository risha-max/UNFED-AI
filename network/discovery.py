"""
Discovery Client — queries the registry to find available nodes and build circuits.

Used by:
  - Inference client: discover nodes, build a circuit for a query
  - Node server: register on startup, heartbeat, unregister on shutdown
"""

import json
import random
import sys
import os
import time
import threading
import uuid
import secrets

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import registry_pb2
import registry_pb2_grpc
import inference_pb2
import inference_pb2_grpc
from network.share_auth import (
    generate_signing_keypair,
    key_file_paths,
    public_key_from_private,
    registration_pop_payload,
    registration_stake_auth_payload,
    heartbeat_auth_payload,
    unregister_auth_payload,
    sign_bytes,
)
from network.secret_loader import load_secret_from_env


def _parse_mpc_capability_json(capability_json: str) -> dict:
    try:
        parsed = json.loads(capability_json or "{}")
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _node_has_mpc_capability(node, capability: str) -> bool:
    data = _parse_mpc_capability_json(getattr(node, "capability_json", ""))
    caps = data.get("mpc_capabilities")
    if isinstance(caps, list) and caps:
        return capability in {str(x).strip().lower() for x in caps}
    # Backward compatibility: legacy MPC entries are treated as both.
    return True


def _node_mpc_role(node) -> str:
    data = _parse_mpc_capability_json(getattr(node, "capability_json", ""))
    role = str(data.get("mpc_role", "")).strip().upper()
    return role if role in ("A", "B") else "A"


class RegistryClient:
    """Client for the registry service."""

    def __init__(self, registry_address: str = None):
        self.registry_address = registry_address or config.REGISTRY_ADDRESS
        self._channel = grpc.insecure_channel(self.registry_address)
        self._stub = registry_pb2_grpc.RegistryStub(self._channel)
        self._util_cache: dict[str, tuple[float, float]] = {}

    def register(self, node_id: str, address: str, model_id: str,
                 shard_index: int, layer_start: int, layer_end: int,
                 has_embedding: bool = False, has_lm_head: bool = False,
                 public_key: bytes = b"",
                 node_type: str = "compute",
                 capability_json: str = "",
                 stake_identity: str = "",
                 share_signing_public_key: bytes = b"",
                 share_signing_pop: bytes = b"",
                 stake_auth_timestamp_ms: int = 0,
                 stake_auth_nonce: str = "",
                 stake_auth_signature: bytes = b"") -> bool:
        """Register a node with the registry."""
        try:
            response = self._stub.Register(registry_pb2.RegisterRequest(
                node_id=node_id,
                address=address,
                model_id=model_id,
                shard_index=shard_index,
                layer_start=layer_start,
                layer_end=layer_end,
                has_embedding=has_embedding,
                has_lm_head=has_lm_head,
                public_key=public_key,
                node_type=node_type,
                capability_json=capability_json,
                stake_identity=stake_identity,
                share_signing_public_key=share_signing_public_key,
                share_signing_pop=share_signing_pop,
                stake_auth_timestamp_ms=stake_auth_timestamp_ms,
                stake_auth_nonce=stake_auth_nonce,
                stake_auth_signature=stake_auth_signature,
            ))
            return response.success
        except grpc.RpcError as e:
            print(f"[Discovery] Failed to register with registry: {e.details()}")
            return False

    def heartbeat(self, node_id: str, auth_timestamp_ms: int = 0,
                  auth_nonce: str = "", auth_signature: bytes = b"") -> bool:
        """Send a heartbeat to the registry."""
        try:
            response = self._stub.Heartbeat(registry_pb2.HeartbeatRequest(
                node_id=node_id,
                auth_timestamp_ms=auth_timestamp_ms,
                auth_nonce=auth_nonce,
                auth_signature=auth_signature,
            ))
            return response.acknowledged
        except grpc.RpcError:
            return False

    def unregister(self, node_id: str, auth_timestamp_ms: int = 0,
                   auth_nonce: str = "", auth_signature: bytes = b"") -> bool:
        """Unregister a node from the registry."""
        try:
            response = self._stub.Unregister(registry_pb2.UnregisterRequest(
                node_id=node_id,
                auth_timestamp_ms=auth_timestamp_ms,
                auth_nonce=auth_nonce,
                auth_signature=auth_signature,
            ))
            return response.success
        except grpc.RpcError:
            return False

    def discover(self, model_id: str = "") -> list:
        """Discover available nodes, optionally filtered by model."""
        try:
            response = self._stub.Discover(registry_pb2.DiscoverRequest(
                model_id=model_id,
            ))
            return list(response.nodes)
        except grpc.RpcError as e:
            print(f"[Discovery] Failed to query registry: {e.details()}")
            return []

    def get_pool_health(self, model_id: str):
        """Get pool health for a model."""
        try:
            return self._stub.GetPoolHealth(registry_pb2.PoolHealthRequest(
                model_id=model_id,
            ))
        except grpc.RpcError as e:
            print(f"[Discovery] Failed to get pool health: {e.details()}")
            return None

    def get_infra_telemetry(self):
        try:
            return self._stub.GetInfraTelemetry(registry_pb2.GetInfraTelemetryRequest())
        except grpc.RpcError as e:
            print(f"[Discovery] Failed to get infra telemetry: {e.details()}")
            return None

    def report_race_winner(
        self,
        *,
        model_id: str,
        session_id: str,
        shard_index: int,
        step_index: int,
        winner_node_id: str,
        winner_address: str,
        winner_response_hash: str,
        candidate_addresses: list[str] | None = None,
        timestamp_ms: int = 0,
        nonce: str = "",
    ):
        try:
            req = registry_pb2.ReportRaceWinnerRequest(
                model_id=str(model_id or ""),
                session_id=str(session_id or ""),
                shard_index=int(shard_index),
                step_index=int(step_index),
                winner_node_id=str(winner_node_id or ""),
                winner_address=str(winner_address or ""),
                winner_response_hash=str(winner_response_hash or ""),
                timestamp_ms=int(timestamp_ms or 0),
                nonce=str(nonce or ""),
            )
            if candidate_addresses:
                req.candidate_addresses.extend([str(a) for a in candidate_addresses if a])
            return self._stub.ReportRaceWinner(req, timeout=5)
        except grpc.RpcError as e:
            print(f"[Discovery] Failed to report race winner: {e.details()}")
            return None

    def report_race_winners(self, reports: list[dict]):
        if not reports:
            return None
        try:
            req = registry_pb2.ReportRaceWinnersRequest()
            for r in reports:
                item = registry_pb2.ReportRaceWinnerRequest(
                    model_id=str(r.get("model_id", "") or ""),
                    session_id=str(r.get("session_id", "") or ""),
                    shard_index=int(r.get("shard_index", 0) or 0),
                    step_index=int(r.get("step_index", 0) or 0),
                    winner_node_id=str(r.get("winner_node_id", "") or ""),
                    winner_address=str(r.get("winner_address", "") or ""),
                    winner_response_hash=str(r.get("winner_response_hash", "") or ""),
                    timestamp_ms=int(r.get("timestamp_ms", 0) or 0),
                    nonce=str(r.get("nonce", "") or ""),
                )
                cands = list(r.get("candidate_addresses", []) or [])
                if cands:
                    item.candidate_addresses.extend([str(a) for a in cands if a])
                req.winners.append(item)
            return self._stub.ReportRaceWinners(req, timeout=5)
        except grpc.RpcError as e:
            # Backward compatibility with registries that only support single-report RPC.
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                for r in reports:
                    self.report_race_winner(
                        model_id=str(r.get("model_id", "") or ""),
                        session_id=str(r.get("session_id", "") or ""),
                        shard_index=int(r.get("shard_index", 0) or 0),
                        step_index=int(r.get("step_index", 0) or 0),
                        winner_node_id=str(r.get("winner_node_id", "") or ""),
                        winner_address=str(r.get("winner_address", "") or ""),
                        winner_response_hash=str(r.get("winner_response_hash", "") or ""),
                        candidate_addresses=list(r.get("candidate_addresses", []) or []),
                        timestamp_ms=int(r.get("timestamp_ms", 0) or 0),
                        nonce=str(r.get("nonce", "") or ""),
                    )
                return None
            print(f"[Discovery] Failed to report race winners: {e.details()}")
            return None

    def discover_compute(self, model_id: str = "") -> list:
        """Discover only compute nodes (excludes vision and MPC nodes)."""
        all_nodes = self.discover(model_id)
        return [n for n in all_nodes
                if n.node_type not in ("vision", "mpc", "daemon", "he_sidecar", "verifier")]

    def discover_mpc(self, model_id: str = "", capability: str = "input") -> list:
        """Discover MPC entry nodes (role A) for shard 0."""
        all_nodes = self.discover(model_id)
        return [
            n for n in all_nodes
            if n.node_type == "mpc"
            and _node_mpc_role(n) == "A"
            and _node_has_mpc_capability(n, capability)
        ]

    def discover_vision(self, model_id: str = "") -> list:
        """Discover vision nodes for a multimodal model."""
        all_nodes = self.discover(model_id)
        return [n for n in all_nodes if n.node_type == "vision"]

    def build_vision_circuit(self, model_id: str
                             ) -> tuple[list[str], list[bytes]] | None:
        """
        Build a vision inference circuit by picking one node per vision shard.

        Returns (addresses, public_keys) ordered by shard index,
        or None if vision nodes aren't available.
        """
        nodes = self.discover_vision(model_id)
        if not nodes:
            print("[Discovery] No vision nodes found for model")
            return None

        # Group by shard index
        shard_map: dict[int, list] = {}
        for node in nodes:
            shard_map.setdefault(node.shard_index, []).append(node)

        # Check coverage
        max_shard = max(shard_map.keys())
        for i in range(max_shard + 1):
            if i not in shard_map:
                print(f"[Discovery] Missing vision shard {i} — cannot build circuit")
                return None

        # Pick one node per shard using least-utilized-first strategy.
        addresses = []
        public_keys = []
        for i in range(max_shard + 1):
            chosen = self._pick_best_node(shard_map[i])
            addresses.append(chosen.address)
            public_keys.append(bytes(chosen.public_key))

        return addresses, public_keys

    def build_circuit(self, model_id: str) -> tuple[list[str], list[bytes]] | None:
        """
        Build an inference circuit by picking one node per shard.

        Returns (addresses, public_keys) ordered by shard index,
        or None if the pool doesn't have full coverage.

        MPC nodes are included as candidates for their shard (typically
        shard 0) so that circuits work when shard 0 is MPC-only.
        """
        nodes = self.discover_compute(model_id)
        # Fundamental routing invariant: when input MPC entry nodes exist,
        # shard 0 must be routed through MPC (role A) instead of plain compute.
        mpc_nodes = self.discover_mpc(model_id, capability="input")
        all_nodes = list(nodes) + list(mpc_nodes)
        if not all_nodes:
            print("[Discovery] No compute nodes found for model")
            return None

        # Group by shard index
        shard_map: dict[int, list] = {}
        for node in all_nodes:
            shard_map.setdefault(node.shard_index, []).append(node)

        if mpc_nodes:
            shard_map[0] = list(mpc_nodes)

        # Check full coverage
        max_shard = max(shard_map.keys())
        for i in range(max_shard + 1):
            if i not in shard_map:
                print(f"[Discovery] Missing shard {i} — cannot build circuit")
                return None

        # Pick one node per shard using least-utilized-first strategy.
        addresses = []
        public_keys = []
        for i in range(max_shard + 1):
            chosen = self._pick_best_node(shard_map[i])
            addresses.append(chosen.address)
            public_keys.append(bytes(chosen.public_key))

        return addresses, public_keys

    def _pick_best_node(self, candidates: list):
        """Pick a node by lowest utilization with exploration."""
        if not candidates:
            raise ValueError("no candidates")
        if len(candidates) == 1:
            return candidates[0]

        scored = []
        for node in candidates:
            util = self._probe_utilization(node.address)
            scored.append((util, node))
        scored.sort(key=lambda x: x[0])

        # Prefer lower-utilization nodes, but keep small exploration for spread.
        # Important: with only 2 candidates, pure top-2 random would ignore util.
        top_k = min(3, len(scored))
        top = scored[:top_k]
        best_util = top[0][0]
        weights = []
        for util, _ in top:
            delta = max(0.0, util - best_util)
            # Lower util => larger weight. Keep a minimum for exploration.
            weights.append(0.08 + (1.0 / (1.0 + (delta * 12.0))))
        nodes = [node for _, node in top]
        return random.choices(nodes, weights=weights, k=1)[0]

    def _probe_utilization(self, address: str) -> float:
        """Read node utilization from fee endpoint with short cache."""
        now = time.time()
        cached = self._util_cache.get(address)
        if cached and (now - cached[0]) <= 2.0:
            return cached[1]
        try:
            channel = grpc.insecure_channel(address, options=config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
            fee = stub.GetLoad(
                inference_pb2.FeeEstimateRequest(estimated_tokens=1),
                timeout=1.0,
            )
            channel.close()
            util = float(getattr(fee, "utilization", 1.0))
        except Exception:
            util = 1.0
        self._util_cache[address] = (now, util)
        return util

    def build_racing_circuit(self, model_id: str,
                             replicas: int = 2) -> dict[int, list[tuple[str, bytes]]] | None:
        """
        Build a racing circuit: multiple nodes per shard for parallel racing.

        Returns {shard_index: [(address, public_key), ...]} with up to
        `replicas` nodes per shard, ordered by shard index.

        Returns None if the pool doesn't have full shard coverage.
        """
        nodes = self.discover_compute(model_id)
        if not nodes:
            print("[Discovery] No compute nodes found for model")
            return None

        # Group by shard index
        shard_map: dict[int, list[tuple[str, bytes]]] = {}
        for node in nodes:
            shard_map.setdefault(node.shard_index, []).append(
                (node.address, bytes(node.public_key)))

        # Check full coverage (every shard from 0 to max must exist)
        max_shard = max(shard_map.keys())
        for i in range(max_shard + 1):
            if i not in shard_map:
                print(f"[Discovery] Missing shard {i} — cannot build racing circuit")
                return None

        # Shuffle and trim to `replicas` per shard
        for shard_idx in shard_map:
            random.shuffle(shard_map[shard_idx])
            shard_map[shard_idx] = shard_map[shard_idx][:replicas]

        return shard_map

    def build_mpc_racing_circuit(self, model_id: str, replicas: int = 2
                                 ) -> tuple[list[tuple[str, bytes]], dict] | None:
        """
        Build a racing circuit that uses MPC for shard 0.

        Returns (mpc_entries, compute_racing_circuit) where:
          - mpc_entries: [(address, public_key), ...] — MPC entry nodes to race
          - compute_racing_circuit: {shard_index: [(addr, pk), ...]} — regular
            compute nodes for shards 1+ (same as build_racing_circuit but
            excluding shard 0)

        If no MPC nodes are available, returns None (caller should fall back
        to regular racing).
        """
        mpc_nodes = self.discover_mpc(model_id, capability="input")
        if not mpc_nodes:
            return None

        # MPC entries to race (each is a role-A entry point with its own peer)
        mpc_entries = [(n.address, bytes(n.public_key)) for n in mpc_nodes]
        random.shuffle(mpc_entries)
        mpc_entries = mpc_entries[:replicas]

        # Regular compute nodes for shards 1+ (no MPC needed)
        compute_nodes = self.discover_compute(model_id)
        shard_map: dict[int, list[tuple[str, bytes]]] = {}
        for node in compute_nodes:
            if node.shard_index == 0:
                continue  # Skip shard 0 — handled by MPC
            shard_map.setdefault(node.shard_index, []).append(
                (node.address, bytes(node.public_key)))

        if not shard_map:
            print("[Discovery] No compute nodes for shards 1+")
            return None

        # Verify coverage: shards 1 to max must exist
        max_shard = max(shard_map.keys())
        for i in range(1, max_shard + 1):
            if i not in shard_map:
                print(f"[Discovery] Missing shard {i} for MPC racing circuit")
                return None

        # Shuffle and trim
        for shard_idx in shard_map:
            random.shuffle(shard_map[shard_idx])
            shard_map[shard_idx] = shard_map[shard_idx][:replicas]

        print(f"[Discovery] MPC racing circuit: {len(mpc_entries)} MPC entries, "
              f"{len(shard_map)} compute shards")
        return mpc_entries, shard_map

    def close(self):
        """Close the gRPC channel."""
        self._channel.close()


# ---------------------------------------------------------------------------
# Peer cache — persists discovered registries across restarts
# ---------------------------------------------------------------------------

_PEER_CACHE_DIR = os.path.expanduser("~/.unfed")
_PEER_CACHE_FILE = os.path.join(_PEER_CACHE_DIR, "peer_cache.json")


def _load_peer_cache() -> list[str]:
    """Load cached peer endpoints from disk."""
    try:
        with open(_PEER_CACHE_FILE) as f:
            data = json.load(f)
        return data.get("peers", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_peer_cache(peers: list[str]) -> None:
    """Persist peer endpoints to disk."""
    os.makedirs(_PEER_CACHE_DIR, exist_ok=True)
    with open(_PEER_CACHE_FILE, "w") as f:
        json.dump({"peers": sorted(set(peers)),
                    "updated_at": time.time()}, f, indent=2)


class RegistryPool:
    """Multi-registry client with cluster-aware discovery.

    Wraps multiple registry addresses (loaded from the seed list + peer
    cache) and provides the same discovery interface as RegistryClient.
    Supports gossip-based peer exchange to discover new clusters.

    Usage:
        pool = RegistryPool()                        # uses SEED_REGISTRIES + cache
        pool = RegistryPool(["host1:50050", "host2:50050"])
        clusters = pool.discover_clusters()          # learn about all clusters
        models = pool.list_models()
        circuit = pool.build_circuit("meta-llama/Llama-3-70B")
    """

    def __init__(self, registry_addresses: list[str] = None):
        seeds = registry_addresses or list(config.SEED_REGISTRIES)
        cached = _load_peer_cache()
        # Merge: seeds first (higher trust), then cached, deduplicated
        seen: set[str] = set()
        self._addresses: list[str] = []
        for addr in seeds + cached:
            if addr not in seen:
                self._addresses.append(addr)
                seen.add(addr)
        self._clients: dict[str, RegistryClient] = {}
        # Cache of cluster info per endpoint
        self._cluster_cache: dict[str, dict] = {}

    def _get_client(self, address: str) -> RegistryClient:
        """Get or create a RegistryClient for an address."""
        if address not in self._clients:
            self._clients[address] = RegistryClient(address)
        return self._clients[address]

    def _try_each(self, operation, description: str = "operation"):
        """Try an operation on each registry until one succeeds.

        Args:
            operation: callable(RegistryClient) -> result
            description: human-readable name for error messages

        Returns the first successful result, or None if all fail.
        """
        for addr in self._addresses:
            try:
                client = self._get_client(addr)
                result = operation(client)
                if result is not None:
                    return result
            except grpc.RpcError:
                continue
        print(f"[RegistryPool] All registries failed for {description}")
        return None

    def list_models(self) -> list:
        """Query registries for available models. Returns merged, deduplicated list."""
        all_models: dict[str, object] = {}  # model_id -> ModelInfo

        for addr in self._addresses:
            try:
                client = self._get_client(addr)
                resp = client._stub.ListModels(
                    registry_pb2.ListModelsRequest(), timeout=10,
                )
                for m in resp.models:
                    existing = all_models.get(m.model_id)
                    if existing is None or m.total_nodes > existing.total_nodes:
                        all_models[m.model_id] = m
            except grpc.RpcError:
                continue

        return list(all_models.values())

    def discover(self, model_id: str = "") -> list:
        """Discover nodes, trying registries in order until one responds."""
        result = self._try_each(
            lambda c: c.discover(model_id) or None,
            f"discover({model_id})",
        )
        return result if result else []

    def discover_compute(self, model_id: str = "") -> list:
        """Discover only compute nodes (excludes vision and MPC)."""
        nodes = self.discover(model_id)
        return [n for n in nodes
                if n.node_type not in ("vision", "mpc", "daemon", "he_sidecar", "verifier")]

    def discover_mpc(self, model_id: str = "", capability: str = "input") -> list:
        """Discover MPC entry nodes."""
        nodes = self.discover(model_id)
        return [
            n for n in nodes
            if n.node_type == "mpc"
            and _node_mpc_role(n) == "A"
            and _node_has_mpc_capability(n, capability)
        ]

    def discover_vision(self, model_id: str = "") -> list:
        """Discover vision nodes."""
        nodes = self.discover(model_id)
        return [n for n in nodes if n.node_type == "vision"]

    def build_circuit(self, model_id: str) -> tuple[list[str], list[bytes]] | None:
        """Build a circuit, trying registries in order."""
        return self._try_each(
            lambda c: c.build_circuit(model_id),
            f"build_circuit({model_id})",
        )

    def build_vision_circuit(self, model_id: str
                             ) -> tuple[list[str], list[bytes]] | None:
        """Build a vision circuit, trying registries in order."""
        return self._try_each(
            lambda c: c.build_vision_circuit(model_id),
            f"build_vision_circuit({model_id})",
        )

    def build_racing_circuit(self, model_id: str,
                             replicas: int = 2) -> dict[int, list[tuple[str, bytes]]] | None:
        """Build a racing circuit, trying registries in order."""
        return self._try_each(
            lambda c: c.build_racing_circuit(model_id, replicas),
            f"build_racing_circuit({model_id})",
        )

    def build_mpc_racing_circuit(self, model_id: str,
                                 replicas: int = 2) -> tuple | None:
        """Build MPC racing circuit, trying registries in order."""
        return self._try_each(
            lambda c: c.build_mpc_racing_circuit(model_id, replicas),
            f"build_mpc_racing_circuit({model_id})",
        )

    def get_pool_health(self, model_id: str):
        """Get pool health, trying registries in order."""
        return self._try_each(
            lambda c: c.get_pool_health(model_id),
            f"get_pool_health({model_id})",
        )

    def get_infra_telemetry(self):
        return self._try_each(
            lambda c: c.get_infra_telemetry(),
            "get_infra_telemetry",
        )

    def report_race_winner(
        self,
        *,
        model_id: str,
        session_id: str,
        shard_index: int,
        step_index: int,
        winner_node_id: str,
        winner_address: str,
        winner_response_hash: str,
        candidate_addresses: list[str] | None = None,
        timestamp_ms: int = 0,
        nonce: str = "",
    ):
        return self._try_each(
            lambda c: c.report_race_winner(
                model_id=model_id,
                session_id=session_id,
                shard_index=shard_index,
                step_index=step_index,
                winner_node_id=winner_node_id,
                winner_address=winner_address,
                winner_response_hash=winner_response_hash,
                candidate_addresses=candidate_addresses or [],
                timestamp_ms=timestamp_ms,
                nonce=nonce,
            ),
            "report_race_winner",
        )

    def report_race_winners(self, reports: list[dict]):
        return self._try_each(
            lambda c: c.report_race_winners(reports),
            "report_race_winners",
        )

    def find_healthy_registry(self) -> str | None:
        """Ping registries and return the first responsive one."""
        for addr in self._addresses:
            try:
                ch = grpc.insecure_channel(addr)
                grpc.channel_ready_future(ch).result(timeout=3)
                ch.close()
                return addr
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Cluster discovery and gossip
    # ------------------------------------------------------------------

    def discover_clusters(self) -> list[dict]:
        """Query all known registries for their cluster info.

        Returns a list of dicts with keys: cluster_id, name, description,
        operator, endpoint, total_nodes, total_models, uptime_seconds,
        default_config_json.
        """
        results: list[dict] = []
        for addr in list(self._addresses):
            try:
                client = self._get_client(addr)
                resp = client._stub.GetClusterInfo(
                    registry_pb2.GetClusterInfoRequest(), timeout=10)
                info = {
                    "cluster_id": resp.cluster_id,
                    "name": resp.name,
                    "description": resp.description,
                    "operator": resp.operator,
                    "endpoint": resp.public_endpoint or addr,
                    "total_nodes": resp.total_nodes,
                    "total_models": resp.total_models,
                    "uptime_seconds": resp.uptime_seconds,
                    "default_config_json": resp.default_config_json,
                }
                results.append(info)
                self._cluster_cache[addr] = info
            except grpc.RpcError:
                continue
        return results

    def learn_peers(self) -> int:
        """Exchange peer lists with known registries to discover new ones.

        Returns the number of newly discovered peers.
        """
        initial_count = len(self._addresses)
        known_set = set(self._addresses)

        for addr in list(self._addresses):
            try:
                client = self._get_client(addr)
                resp = client._stub.ExchangePeers(
                    registry_pb2.ExchangePeersRequest(
                        known_peers=list(known_set)),
                    timeout=10,
                )
                for peer in resp.peers:
                    ep = peer.endpoint
                    if ep and ep not in known_set:
                        self._addresses.append(ep)
                        known_set.add(ep)
                        # Update cluster cache with gossip info
                        if peer.cluster_id:
                            self._cluster_cache[ep] = {
                                "cluster_id": peer.cluster_id,
                                "name": peer.name,
                                "endpoint": ep,
                            }
            except grpc.RpcError:
                continue

        new_count = len(self._addresses) - initial_count
        if new_count > 0:
            _save_peer_cache(self._addresses)
            print(f"[RegistryPool] Discovered {new_count} new peer(s), "
                  f"total: {len(self._addresses)}")
        return new_count

    def select_cluster(self, criteria: str = "most_nodes",
                       model_id: str = "") -> str | None:
        """Pick the best cluster endpoint based on criteria.

        Args:
            criteria: "most_nodes", "cheapest", or "first_healthy"
            model_id: If set, only consider clusters hosting this model

        Returns the best cluster's endpoint, or None.
        """
        clusters = self.discover_clusters()
        if not clusters:
            return None

        # Filter by model availability if requested
        if model_id:
            filtered = []
            for c in clusters:
                addr = c["endpoint"]
                try:
                    client = self._get_client(addr)
                    nodes = client.discover(model_id)
                    if nodes:
                        c["_has_model"] = True
                        filtered.append(c)
                except grpc.RpcError:
                    continue
            clusters = filtered

        if not clusters:
            return None

        if criteria == "cheapest":
            def get_fee(c):
                try:
                    cfg = json.loads(c.get("default_config_json", "{}"))
                    return cfg.get("default_fee_base", float("inf"))
                except (json.JSONDecodeError, TypeError):
                    return float("inf")
            clusters.sort(key=get_fee)
        elif criteria == "most_nodes":
            clusters.sort(key=lambda c: c.get("total_nodes", 0),
                          reverse=True)
        # "first_healthy" — already in order, first one wins

        return clusters[0]["endpoint"]

    @property
    def active_registry(self) -> str | None:
        """Return the first registry address (primary)."""
        return self._addresses[0] if self._addresses else None

    @property
    def addresses(self) -> list[str]:
        """Return all known registry addresses."""
        return list(self._addresses)

    def close(self):
        """Close all gRPC channels."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()


class NodeRegistration:
    """
    Manages a node's lifecycle with the registry:
    registration, periodic heartbeats, and graceful unregistration.
    Also manages this node's X25519 key pair for onion routing.
    """

    def __init__(self, address: str, model_id: str, shard_index: int,
                 layer_start: int, layer_end: int,
                 has_embedding: bool = False, has_lm_head: bool = False,
                 registry_address: str = None, node_type: str = "compute",
                 capability_json: str = "", stake_identity: str = "",
                 node_id: str = None, stake_evm_private_key: str | None = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.address = address
        self.model_id = model_id
        self.shard_index = shard_index
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.node_type = node_type
        self.capability_json = capability_json or ""
        self.stake_identity = stake_identity or ""
        self.stake_evm_private_key = (
            stake_evm_private_key
            if stake_evm_private_key is not None
            else load_secret_from_env(
                "UNFED_STAKE_EVM_PRIVATE_KEY",
                file_env_var="UNFED_STAKE_EVM_PRIVATE_KEY_FILE",
            )
        )

        # Generate X25519 key pair for onion routing
        from network.onion import generate_keypair, public_key_to_bytes
        self.private_key, self.public_key = generate_keypair()
        self.public_key_bytes = public_key_to_bytes(self.public_key)
        self.share_signing_private_key = self._load_or_create_share_signing_key()
        self.share_signing_public_key = public_key_from_private(
            self.share_signing_private_key
        )

        self._client = RegistryClient(registry_address)
        self._heartbeat_thread = None
        self._running = False

    def start(self) -> bool:
        """Register with the registry and start heartbeating."""
        pop_payload = registration_pop_payload(
            node_id=self.node_id,
            address=self.address,
            model_id=self.model_id,
            shard_index=self.shard_index,
            node_type=self.node_type,
        )
        share_signing_pop = sign_bytes(self.share_signing_private_key, pop_payload)
        ts_ms = int(time.time() * 1000)
        reg_nonce = secrets.token_hex(16)
        stake_auth_signature = self._sign_stake_registration(ts_ms, reg_nonce)
        success = self._client.register(
            node_id=self.node_id,
            address=self.address,
            model_id=self.model_id,
            shard_index=self.shard_index,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            has_embedding=self.has_embedding,
            has_lm_head=self.has_lm_head,
            public_key=self.public_key_bytes,
            node_type=self.node_type,
            capability_json=self.capability_json,
            stake_identity=self.stake_identity,
            share_signing_public_key=self.share_signing_public_key,
            share_signing_pop=share_signing_pop,
            stake_auth_timestamp_ms=ts_ms,
            stake_auth_nonce=reg_nonce,
            stake_auth_signature=stake_auth_signature,
        )

        if success:
            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()
            print(f"[Node {self.node_id[:8]}...] Registered with registry at {self._client.registry_address}")
        else:
            print(f"[Node {self.node_id[:8]}...] WARNING: Failed to register with registry")

        return success

    def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            time.sleep(config.HEARTBEAT_INTERVAL_SECONDS)
            if self._running:
                hb_ts_ms = int(time.time() * 1000)
                hb_nonce = secrets.token_hex(16)
                ack = self._client.heartbeat(
                    self.node_id,
                    auth_timestamp_ms=hb_ts_ms,
                    auth_nonce=hb_nonce,
                    auth_signature=sign_bytes(
                        self.share_signing_private_key,
                        heartbeat_auth_payload(
                            node_id=self.node_id,
                            timestamp_ms=hb_ts_ms,
                            nonce=hb_nonce,
                        ),
                    ),
                )
                if not ack:
                    # Registry may have restarted — re-register
                    re_ts_ms = int(time.time() * 1000)
                    re_nonce = secrets.token_hex(16)
                    self._client.register(
                        node_id=self.node_id,
                        address=self.address,
                        model_id=self.model_id,
                        shard_index=self.shard_index,
                        layer_start=self.layer_start,
                        layer_end=self.layer_end,
                        has_embedding=self.has_embedding,
                        has_lm_head=self.has_lm_head,
                        public_key=self.public_key_bytes,
                        node_type=self.node_type,
                        capability_json=self.capability_json,
                        stake_identity=self.stake_identity,
                        share_signing_public_key=self.share_signing_public_key,
                        share_signing_pop=sign_bytes(
                            self.share_signing_private_key,
                            registration_pop_payload(
                                node_id=self.node_id,
                                address=self.address,
                                model_id=self.model_id,
                                shard_index=self.shard_index,
                                node_type=self.node_type,
                            ),
                        ),
                        stake_auth_timestamp_ms=re_ts_ms,
                        stake_auth_nonce=re_nonce,
                        stake_auth_signature=self._sign_stake_registration(
                            re_ts_ms, re_nonce
                        ),
                    )

    def stop(self):
        """Unregister from the registry and stop heartbeating."""
        self._running = False
        ts_ms = int(time.time() * 1000)
        nonce = secrets.token_hex(16)
        self._client.unregister(
            self.node_id,
            auth_timestamp_ms=ts_ms,
            auth_nonce=nonce,
            auth_signature=sign_bytes(
                self.share_signing_private_key,
                unregister_auth_payload(
                    node_id=self.node_id,
                    timestamp_ms=ts_ms,
                    nonce=nonce,
                ),
            ),
        )
        print(f"[Node {self.node_id[:8]}...] Unregistered from registry")

    @property
    def short_id(self) -> str:
        return self.node_id[:8]

    def _load_or_create_share_signing_key(self) -> bytes:
        priv_path, pub_path = key_file_paths(self.node_id)
        os.makedirs(os.path.dirname(priv_path), exist_ok=True)
        if os.path.exists(priv_path):
            with open(priv_path, "rb") as f:
                private_bytes = f.read()
            if len(private_bytes) == 32:
                return private_bytes
        private_bytes, public_bytes = generate_signing_keypair()
        with open(priv_path, "wb") as f:
            f.write(private_bytes)
        with open(pub_path, "wb") as f:
            f.write(public_bytes)
        try:
            os.chmod(priv_path, 0o600)
        except OSError:
            pass
        return private_bytes

    def _sign_stake_registration(self, timestamp_ms: int, nonce: str) -> bytes:
        if not self.stake_evm_private_key:
            return b""
        try:
            from eth_account import Account
            from eth_account.messages import encode_defunct
        except Exception:
            return b""
        payload = registration_stake_auth_payload(
            node_id=self.node_id,
            address=self.address,
            model_id=self.model_id,
            shard_index=self.shard_index,
            node_type=self.node_type,
            share_signing_public_key=self.share_signing_public_key,
            timestamp_ms=timestamp_ms,
            nonce=nonce,
        )
        msg = encode_defunct(text=payload)
        signed = Account.sign_message(msg, private_key=self.stake_evm_private_key)
        return bytes(signed.signature)
