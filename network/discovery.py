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

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import registry_pb2
import registry_pb2_grpc


class RegistryClient:
    """Client for the registry service."""

    def __init__(self, registry_address: str = None):
        self.registry_address = registry_address or config.REGISTRY_ADDRESS
        self._channel = grpc.insecure_channel(self.registry_address)
        self._stub = registry_pb2_grpc.RegistryStub(self._channel)

    def register(self, node_id: str, address: str, model_id: str,
                 shard_index: int, layer_start: int, layer_end: int,
                 has_embedding: bool = False, has_lm_head: bool = False,
                 public_key: bytes = b"",
                 node_type: str = "compute") -> bool:
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
            ))
            return response.success
        except grpc.RpcError as e:
            print(f"[Discovery] Failed to register with registry: {e.details()}")
            return False

    def heartbeat(self, node_id: str) -> bool:
        """Send a heartbeat to the registry."""
        try:
            response = self._stub.Heartbeat(registry_pb2.HeartbeatRequest(
                node_id=node_id,
            ))
            return response.acknowledged
        except grpc.RpcError:
            return False

    def unregister(self, node_id: str) -> bool:
        """Unregister a node from the registry."""
        try:
            response = self._stub.Unregister(registry_pb2.UnregisterRequest(
                node_id=node_id,
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

    def discover_compute(self, model_id: str = "") -> list:
        """Discover only compute nodes (excludes vision and MPC nodes)."""
        all_nodes = self.discover(model_id)
        return [n for n in all_nodes
                if n.node_type not in ("vision", "mpc")]

    def discover_mpc(self, model_id: str = "") -> list:
        """Discover MPC entry nodes (role A) for shard 0."""
        all_nodes = self.discover(model_id)
        return [n for n in all_nodes if n.node_type == "mpc"]

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

        # Pick one random node per shard
        addresses = []
        public_keys = []
        for i in range(max_shard + 1):
            chosen = random.choice(shard_map[i])
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
        # Also include MPC nodes — they serve as shard 0 entry points
        mpc_nodes = self.discover_mpc(model_id)
        all_nodes = nodes + mpc_nodes
        if not all_nodes:
            print("[Discovery] No compute nodes found for model")
            return None

        # Group by shard index
        shard_map: dict[int, list] = {}
        for node in all_nodes:
            shard_map.setdefault(node.shard_index, []).append(node)

        # Check full coverage
        max_shard = max(shard_map.keys())
        for i in range(max_shard + 1):
            if i not in shard_map:
                print(f"[Discovery] Missing shard {i} — cannot build circuit")
                return None

        # Pick one random node per shard (this is where circuit randomization happens)
        addresses = []
        public_keys = []
        for i in range(max_shard + 1):
            chosen = random.choice(shard_map[i])
            addresses.append(chosen.address)
            public_keys.append(bytes(chosen.public_key))

        return addresses, public_keys

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
        mpc_nodes = self.discover_mpc(model_id)
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
                if n.node_type not in ("vision", "mpc")]

    def discover_mpc(self, model_id: str = "") -> list:
        """Discover MPC entry nodes."""
        nodes = self.discover(model_id)
        return [n for n in nodes if n.node_type == "mpc"]

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
                 node_id: str = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.address = address
        self.model_id = model_id
        self.shard_index = shard_index
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.has_embedding = has_embedding
        self.has_lm_head = has_lm_head
        self.node_type = node_type

        # Generate X25519 key pair for onion routing
        from network.onion import generate_keypair, public_key_to_bytes
        self.private_key, self.public_key = generate_keypair()
        self.public_key_bytes = public_key_to_bytes(self.public_key)

        self._client = RegistryClient(registry_address)
        self._heartbeat_thread = None
        self._running = False

    def start(self) -> bool:
        """Register with the registry and start heartbeating."""
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
                ack = self._client.heartbeat(self.node_id)
                if not ack:
                    # Registry may have restarted — re-register
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
                    )

    def stop(self):
        """Unregister from the registry and stop heartbeating."""
        self._running = False
        self._client.unregister(self.node_id)
        print(f"[Node {self.node_id[:8]}...] Unregistered from registry")

    @property
    def short_id(self) -> str:
        return self.node_id[:8]
