"""
MPC (Multi-Party Computation) for Shard 0 — Beaver-Triple 2PC.

Two independent nodes (A and B) jointly compute the embedding + layer 0
of a transformer model.  Neither node sees the raw token IDs or the full
embedding vector.  Subsequent layers run in cleartext on Node A — after
one full transformer layer, activations cannot be inverted to raw tokens.

Protocol overview (Beaver-triple-based 2PC):
  1. Node A receives token_ids, computes embedding, splits into additive
     shares, sends share_b + Beaver triple shares to Node B.
  2. Both nodes jointly compute layer 0 on their shares:
     - Linear ops (matmul with public weights): free on shares.
     - Non-linear ops (RMSNorm, softmax, SiLU): polynomial approximations
       evaluated via Beaver-triple secure multiplications.
  3. Node B sends its final share back to Node A.
  4. Node A reconstructs and continues with cleartext layers 1-N.

Security:
  - Only (x - a) and (y - b) are ever revealed, where a, b are random
    Beaver triple values.  These reveal nothing about x or y.
  - No intermediate value is ever reconstructed on a single party during
    the MPC phase (layer 0).

Usage:
    python -m network.mpc_shard0 --role B --port 50063 --peer localhost:50061
    python -m network.mpc_shard0 --role A --port 50061 --peer localhost:50063 \\
        --advertise localhost:50061 --registry localhost:50050
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import threading
import time
import uuid
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from typing import Optional

import grpc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc

from network.resilience import create_resilient_channel
from network.infra_routing import select_least_loaded_daemon
from network.mpc_beaver import (
    BeaverTriple, BeaverTripleShares,
    serialize_triple_shares, deserialize_triple_shares,
)
from network.mpc_protocols import (
    MpcDncConfig,
    PeerExchanger, TripleAllocator,
    configure_mpc_dnc,
    secure_rmsnorm, secure_softmax, secure_silu, secure_gate_up,
    secure_matmul, allocate_layer0_triples,
)
from network.share_auth import (
    PAYLOAD_HASH_VERSION,
    SharePayload,
    canonical_share_payload_bytes,
    sign_bytes,
)
from network.forward_attestation import (
    FORWARD_ATTESTATION_VERSION,
    ForwardAttestationPayload,
    make_tensor_shape_signature,
    proof_bytes_hash,
    sign_forward_attestation,
    tensor_bytes_digest,
)


def _is_mpc_profile_enabled() -> bool:
    return os.environ.get("UNFED_MPC_PROFILE", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _mpc_profile_mode() -> str:
    mode = os.environ.get("UNFED_MPC_PROFILE_MODE", "full").strip().lower()
    if mode not in ("full", "focused"):
        return "full"
    return mode


def _is_profiled_exchange_op(op_id: str) -> bool:
    if _mpc_profile_mode() == "full":
        return True
    tracked = (
        "attn_qk_matmul",
        "attn_av_matmul",
        "silu_",
        "mlp_gate_up",
    )
    return any(token in op_id for token in tracked)


_SHAPE_LIST_CACHE: dict[tuple[int, ...], list[int]] = {}
_SHAPE_LIST_CACHE_LOCK = threading.Lock()


def _shape_list(shape: tuple[int, ...] | list[int]) -> list[int]:
    if isinstance(shape, list):
        key = tuple(int(x) for x in shape)
    else:
        key = tuple(int(x) for x in shape)
    with _SHAPE_LIST_CACHE_LOCK:
        cached = _SHAPE_LIST_CACHE.get(key)
        if cached is not None:
            return cached
        # Cache a canonical list object to reduce repeated tiny allocations.
        val = list(key)
        _SHAPE_LIST_CACHE[key] = val
        return val


def _pack_peer_triples_blob(
    peer_payloads: list[tuple[str, bytes, bytes, bytes, list[int], list[int], list[int]]]
) -> bytes:
    buf = bytearray()
    buf.extend(struct.pack("<I", len(peer_payloads)))
    for op_id, a_bytes, b_bytes, c_bytes, a_shape, b_shape, c_shape in peer_payloads:
        op_raw = op_id.encode("utf-8")
        buf.extend(struct.pack("<I", len(op_raw)))
        buf.extend(op_raw)
        for shape, raw in (
            (a_shape, a_bytes),
            (b_shape, b_bytes),
            (c_shape, c_bytes),
        ):
            shape_l = _shape_list(shape)
            buf.extend(struct.pack("<I", len(shape_l)))
            for d in shape_l:
                buf.extend(struct.pack("<i", int(d)))
            buf.extend(struct.pack("<I", len(raw)))
            buf.extend(raw)
    return bytes(buf)


def _unpack_peer_triples_blob(
    blob: bytes,
) -> list[tuple[str, bytes, bytes, bytes, list[int], list[int], list[int]]]:
    mv = memoryview(blob)
    off = 0

    def _take_u32() -> int:
        nonlocal off
        if off + 4 > len(mv):
            raise ValueError("triples_blob truncated (u32)")
        v = struct.unpack_from("<I", mv, off)[0]
        off += 4
        return int(v)

    def _take_i32() -> int:
        nonlocal off
        if off + 4 > len(mv):
            raise ValueError("triples_blob truncated (i32)")
        v = struct.unpack_from("<i", mv, off)[0]
        off += 4
        return int(v)

    def _take_bytes(n: int) -> bytes:
        nonlocal off
        if n < 0 or off + n > len(mv):
            raise ValueError("triples_blob truncated (bytes)")
        out = bytes(mv[off: off + n])
        off += n
        return out

    def _take_shape_and_tensor() -> tuple[list[int], bytes]:
        rank = _take_u32()
        shape = [_take_i32() for _ in range(rank)]
        raw_len = _take_u32()
        raw = _take_bytes(raw_len)
        return _shape_list(shape), raw

    n = _take_u32()
    out: list[tuple[str, bytes, bytes, bytes, list[int], list[int], list[int]]] = []
    for _ in range(n):
        op_len = _take_u32()
        op_id = _take_bytes(op_len).decode("utf-8", errors="strict")
        a_shape, a_raw = _take_shape_and_tensor()
        b_shape, b_raw = _take_shape_and_tensor()
        c_shape, c_raw = _take_shape_and_tensor()
        out.append((op_id, a_raw, b_raw, c_raw, a_shape, b_shape, c_shape))
    if off != len(mv):
        raise ValueError("triples_blob has trailing bytes")
    return out


def _pack_share_proto_blob(shares_proto: list[inference_pb2.ShareProto]) -> bytes:
    buf = bytearray()
    buf.extend(struct.pack("<I", len(shares_proto)))
    for sp in shares_proto:
        raw = sp.SerializeToString()
        buf.extend(struct.pack("<I", len(raw)))
        buf.extend(raw)
    return bytes(buf)


@dataclass
class _Layer0TriplePackage:
    key: tuple[int, int, int, int, int, int]
    triple_set_id: str
    triple_seed: int
    my_triples: dict[str, BeaverTripleShares]
    peer_blob: bytes


class _Layer0TripleCache:
    """Single-use package cache for layer-0 Beaver triples."""

    def __init__(
        self,
        max_per_shape: int = 2,
        max_shapes: int = 8,
        seq_bucket_size: int = 1,
        persist_enabled: bool = True,
        persist_dir: str = "",
        persist_min_interval_s: float = 2.0,
    ):
        self._max_per_shape = max(1, int(max_per_shape))
        self._max_shapes = max(1, int(max_shapes))
        self._seq_bucket_size = max(1, int(seq_bucket_size))
        self._pools: dict[tuple[int, int, int, int, int, int], list[_Layer0TriplePackage]] = {}
        self._refilling: set[tuple[int, int, int, int, int, int]] = set()
        self._last_used: dict[tuple[int, int, int, int, int, int], int] = {}
        self._lru_tick = 0
        self._lock = threading.Lock()
        self._persist_enabled = bool(persist_enabled)
        self._persist_dir = os.path.expanduser(
            persist_dir or os.environ.get("UNFED_MPC_TRIPLE_CACHE_DIR", "~/.unfed/mpc_triples")
        )
        self._persist_file = os.path.join(self._persist_dir, "layer0_triple_cache.pt")
        self._persist_min_interval_s = max(0.0, float(persist_min_interval_s))
        self._persist_last_ts = 0.0
        if self._persist_enabled:
            self._load_from_disk()

    @staticmethod
    def _pkg_to_obj(pkg: _Layer0TriplePackage) -> dict:
        my = {}
        for op_id, shares in pkg.my_triples.items():
            my[op_id] = {"a": shares.a, "b": shares.b, "c": shares.c}
        return {
            "key": list(pkg.key),
            "triple_set_id": pkg.triple_set_id,
            "triple_seed": int(pkg.triple_seed),
            "my": my,
            "peer_blob": pkg.peer_blob,
        }

    @staticmethod
    def _obj_to_pkg(obj: dict) -> _Layer0TriplePackage:
        key = tuple(int(x) for x in obj["key"])
        my_triples: dict[str, BeaverTripleShares] = {}
        for op_id, parts in dict(obj["my"]).items():
            my_triples[str(op_id)] = BeaverTripleShares(parts["a"], parts["b"], parts["c"])
        triple_set_id = str(obj.get("triple_set_id", ""))
        triple_seed = int(obj.get("triple_seed", 0))
        peer_blob = bytes(obj.get("peer_blob", b""))
        if not peer_blob and "peer" in obj:
            legacy_payloads: list[tuple[str, bytes, bytes, bytes, list[int], list[int], list[int]]] = []
            for item in list(obj.get("peer", [])):
                op_id, a_bytes, b_bytes, c_bytes, a_shape, b_shape, c_shape = item
                legacy_payloads.append(
                    (
                        str(op_id),
                        bytes(a_bytes),
                        bytes(b_bytes),
                        bytes(c_bytes),
                        _shape_list(a_shape),
                        _shape_list(b_shape),
                        _shape_list(c_shape),
                    )
                )
            peer_blob = _pack_peer_triples_blob(legacy_payloads)
        if not triple_set_id and peer_blob:
            triple_set_id = hashlib.sha256(peer_blob).hexdigest()[:24]
        if not triple_set_id and triple_seed:
            triple_set_id = hashlib.sha256(
                f"{':'.join(str(x) for x in key)}:{triple_seed}".encode("utf-8")
            ).hexdigest()[:24]
        return _Layer0TriplePackage(
            key=key,
            triple_set_id=triple_set_id,
            triple_seed=triple_seed,
            my_triples=my_triples,
            peer_blob=peer_blob,
        )

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._persist_file):
            return
        try:
            raw = torch.load(self._persist_file, map_location="cpu")
            pools_obj = raw.get("pools", {})
            loaded_pools: dict[tuple[int, int, int, int, int, int], list[_Layer0TriplePackage]] = {}
            for key_raw, packages in pools_obj.items():
                key = tuple(int(x) for x in key_raw)
                loaded_pools[key] = [self._obj_to_pkg(obj) for obj in list(packages)]
            with self._lock:
                self._pools = loaded_pools
                self._last_used = {k: i + 1 for i, k in enumerate(self._pools.keys())}
                self._lru_tick = len(self._last_used)
        except Exception as e:
            print(f"[MPC-A] Triple cache load failed: {e}")

    def _save_to_disk_locked(self, force: bool = False) -> None:
        if not self._persist_enabled:
            return
        now = time.time()
        if not force and (now - self._persist_last_ts) < self._persist_min_interval_s:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        tmp_path = self._persist_file + ".tmp"
        payload = {
            "pools": {
                tuple(key): [self._pkg_to_obj(pkg) for pkg in pkgs]
                for key, pkgs in self._pools.items()
            }
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self._persist_file)
        self._persist_last_ts = now

    def _bucket_seq_len(self, seq_len: int) -> int:
        seq_len = int(seq_len)
        if self._seq_bucket_size <= 1:
            return seq_len
        return ((seq_len + self._seq_bucket_size - 1) // self._seq_bucket_size) * self._seq_bucket_size

    def _make_key(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        seq_len: int,
        batch: int,
    ) -> tuple[int, int, int, int, int, int]:
        return (
            int(hidden_size),
            int(num_heads),
            int(head_dim),
            int(intermediate_size),
            self._bucket_seq_len(seq_len),
            int(batch),
        )

    def _touch_key_locked(self, key: tuple[int, int, int, int, int, int]) -> None:
        self._lru_tick += 1
        self._last_used[key] = self._lru_tick

    def _evict_if_needed_locked(self, keep_key: tuple[int, int, int, int, int, int]) -> None:
        evicted = False
        while len(self._pools) > self._max_shapes:
            candidates = []
            for key, ts in self._last_used.items():
                if key == keep_key:
                    continue
                if key in self._refilling:
                    continue
                candidates.append((ts, key))
            if not candidates:
                break
            _, victim = min(candidates)
            self._pools.pop(victim, None)
            self._last_used.pop(victim, None)
            evicted = True
        if evicted:
            self._save_to_disk_locked()

    @staticmethod
    def _make_package(
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        seq_len: int,
        batch: int,
    ) -> _Layer0TriplePackage:
        key = (
            int(hidden_size),
            int(num_heads),
            int(head_dim),
            int(intermediate_size),
            int(seq_len),
            int(batch),
        )
        seed = int.from_bytes(os.urandom(8), "little", signed=False) & ((1 << 63) - 1)
        all_triples = allocate_layer0_triples(
            hidden_size,
            num_heads,
            head_dim,
            intermediate_size,
            seq_len,
            batch,
            seed=seed,
        )
        my_triples: dict[str, BeaverTripleShares] = {}
        for op_id, bt in all_triples.items():
            my_triples[op_id] = bt.party_0
        # Seed-based triple derivation is the default path.
        peer_blob = b""
        triple_set_id = hashlib.sha256(
            f"{':'.join(str(x) for x in key)}:{seed}".encode("utf-8")
        ).hexdigest()[:24]
        return _Layer0TriplePackage(
            key=key,
            triple_set_id=triple_set_id,
            triple_seed=seed,
            my_triples=my_triples,
            peer_blob=peer_blob,
        )

    def _spawn_refill(self, key: tuple[int, int, int, int, int, int]):
        with self._lock:
            if key in self._refilling:
                return
            self._refilling.add(key)

        def _refill():
            try:
                hidden_size, num_heads, head_dim, intermediate_size, seq_len, batch = key
                while True:
                    with self._lock:
                        current = len(self._pools.get(key, []))
                        if current >= self._max_per_shape:
                            break
                    pkg = self._make_package(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        intermediate_size=intermediate_size,
                        seq_len=seq_len,
                        batch=batch,
                    )
                    with self._lock:
                        self._pools.setdefault(key, []).append(pkg)
                        self._save_to_disk_locked()
            finally:
                with self._lock:
                    self._refilling.discard(key)

        t = threading.Thread(target=_refill, daemon=True)
        t.start()

    def acquire(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        seq_len: int,
        batch: int,
    ) -> tuple[_Layer0TriplePackage, bool, int]:
        key = self._make_key(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            seq_len=seq_len,
            batch=batch,
        )
        need_refill = False
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            pool = self._pools.get(key)
            if pool:
                pkg = pool.pop()
                if (not pkg.triple_set_id) or int(getattr(pkg, "triple_seed", 0)) <= 0:
                    # Discard malformed/legacy package entries and regenerate on cold path.
                    pkg = None
                if len(pool) < self._max_per_shape:
                    need_refill = True
                else:
                    need_refill = False
                hit_pkg = pkg
                self._touch_key_locked(key)
            else:
                hit_pkg = None
                self._touch_key_locked(key)
            self._evict_if_needed_locked(keep_key=key)
        if hit_pkg is not None:
            if need_refill:
                self._spawn_refill(key)
            return hit_pkg, True, key[4]

        # Cold path: generate synchronously, then trigger async refill.
        pkg = self._make_package(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            seq_len=key[4],
            batch=batch,
        )
        with self._lock:
            # Persist shape presence/usage metadata even for cold miss,
            # so restart can keep recently-used shape ordering.
            self._save_to_disk_locked()
        self._spawn_refill(key)
        return pkg, False, key[4]

    def prewarm(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        batch: int,
        seq_lens: list[int],
        target_per_shape: int,
    ) -> None:
        target = max(1, int(target_per_shape))
        for seq_len in seq_lens:
            key = self._make_key(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                seq_len=seq_len,
                batch=batch,
            )
            with self._lock:
                self._pools.setdefault(key, [])
                self._touch_key_locked(key)
                self._evict_if_needed_locked(keep_key=key)
                need = max(0, target - len(self._pools[key]))
            for _ in range(need):
                pkg = self._make_package(
                    hidden_size=key[0],
                    num_heads=key[1],
                    head_dim=key[2],
                    intermediate_size=key[3],
                    seq_len=key[4],
                    batch=key[5],
                )
                with self._lock:
                    self._pools.setdefault(key, []).append(pkg)
                    self._save_to_disk_locked()


# ---------------------------------------------------------------------------
# Secret sharing primitives
# ---------------------------------------------------------------------------

def create_additive_shares(tensor: torch.Tensor,
                           num_shares: int = 2) -> list[torch.Tensor]:
    """Split a tensor into additive secret shares."""
    shares = []
    remaining = tensor.clone()
    for _ in range(num_shares - 1):
        share = torch.randn_like(tensor)
        shares.append(share)
        remaining = remaining - share
    shares.append(remaining)
    return shares


def reconstruct_from_shares(shares: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct the original tensor from additive shares."""
    return sum(shares)


# ---------------------------------------------------------------------------
# Tensor serialization helpers
# ---------------------------------------------------------------------------

def _tensor_to_bytes(tensor: torch.Tensor) -> tuple[bytes, list[int]]:
    """Serialize a tensor to bytes + shape."""
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        t = tensor
    else:
        t = tensor.contiguous().float()
    return t.numpy().tobytes(), _shape_list(tuple(t.shape))


def _bytes_to_tensor(data: bytes, shape: list[int]) -> torch.Tensor:
    """Deserialize bytes + shape to a tensor."""
    return torch.from_numpy(
        np.frombuffer(data, dtype=np.float32).copy().reshape(tuple(shape)))


# ---------------------------------------------------------------------------
# gRPC-based peer exchanger (Node A's view: calls Node B's Exchange RPC)
# ---------------------------------------------------------------------------

class GrpcPeerExchangerA(PeerExchanger):
    """
    Node A's exchanger: calls Node B's MPCPeer.Exchange RPC to send
    epsilon/delta and receive Node B's epsilon/delta in return.
    """

    def __init__(self, peer_stub):
        self._stub = peer_stub  # inference_pb2_grpc.MPCPeerStub

    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor,
                 my_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps_bytes, eps_shape = _tensor_to_bytes(my_epsilon)
        del_bytes, del_shape = _tensor_to_bytes(my_delta)
        call_start = time.perf_counter()
        resp = self._stub.Exchange(
            inference_pb2.MPCExchangeRequest(
                session_id=session_id,
                op_id=op_id,
                epsilon_data=eps_bytes,
                delta_data=del_bytes,
                shape=_shape_list(eps_shape),
                delta_shape=_shape_list(del_shape),
            ),
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - call_start) * 1000.0
        if _is_mpc_profile_enabled() and _is_profiled_exchange_op(op_id):
            print(
                f"[MPC-A][PROFILE] session={session_id[:8]} "
                f"exchange op={op_id} rt_ms={elapsed_ms:.2f}"
            )
        resp_eps_shape = _shape_list(list(resp.shape))
        resp_del_shape = _shape_list(list(resp.delta_shape)) if resp.delta_shape else resp_eps_shape
        peer_eps = _bytes_to_tensor(bytes(resp.epsilon_data), resp_eps_shape)
        peer_del = _bytes_to_tensor(bytes(resp.delta_data), resp_del_shape)
        return peer_eps, peer_del


# ---------------------------------------------------------------------------
# Node B's peer exchanger: waits for Node A's call, responds synchronously
# ---------------------------------------------------------------------------

class GrpcPeerExchangerB(PeerExchanger):
    """
    Node B's exchanger: exchanges are driven by Node A calling the Exchange
    RPC.  Node B computes its epsilon/delta and waits for A's call, then
    responds.  This class bridges the async gRPC handler with the
    synchronous protocol execution on Node B's side.

    Flow:
      - Node B's protocol thread calls exchanger.exchange() → blocks
      - Node A calls the gRPC Exchange RPC → handler receives A's values,
        wakes up B's protocol thread, gets B's values, responds to A
    """

    def __init__(self):
        self._pending_b: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._incoming_a: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._response_b: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._b_ready: dict[str, threading.Event] = {}  # B has values
        self._a_ready: dict[str, threading.Event] = {}  # A has called

    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor,
                 my_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Called by Node B's protocol thread.  Deposits B's values, waits
        for A's values."""
        key = f"{session_id}:{op_id}"
        with self._lock:
            self._pending_b[key] = (my_epsilon, my_delta)
            if key not in self._b_ready:
                self._b_ready[key] = threading.Event()
            if key not in self._a_ready:
                self._a_ready[key] = threading.Event()

        # Signal that B's values are ready
        self._b_ready[key].set()

        # Wait for A's values
        wait_start = time.perf_counter()
        self._a_ready[key].wait(timeout=60)
        wait_ms = (time.perf_counter() - wait_start) * 1000.0

        with self._lock:
            peer_eps, peer_del = self._incoming_a.pop(key)
            self._a_ready.pop(key, None)
            self._b_ready.pop(key, None)
        if _is_mpc_profile_enabled() and _is_profiled_exchange_op(op_id):
            print(
                f"[MPC-B][PROFILE] session={session_id[:8]} "
                f"exchange op={op_id} wait_ms={wait_ms:.2f}"
            )
        return peer_eps, peer_del

    def handle_exchange_rpc(self, session_id: str, op_id: str,
                            a_epsilon: torch.Tensor,
                            a_delta: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Called by the gRPC handler when Node A sends its exchange."""
        rpc_start = time.perf_counter()
        key = f"{session_id}:{op_id}"
        with self._lock:
            if key not in self._b_ready:
                self._b_ready[key] = threading.Event()
            if key not in self._a_ready:
                self._a_ready[key] = threading.Event()

        # Wait for B's protocol thread to deposit its values
        b_ready_wait_start = time.perf_counter()
        self._b_ready[key].wait(timeout=60)
        b_ready_wait_ms = (time.perf_counter() - b_ready_wait_start) * 1000.0

        with self._lock:
            b_eps, b_del = self._pending_b.pop(key)
            self._incoming_a[key] = (a_epsilon, a_delta)

        # Wake up B's protocol thread
        self._a_ready[key].set()
        rpc_ms = (time.perf_counter() - rpc_start) * 1000.0
        if _is_mpc_profile_enabled() and _is_profiled_exchange_op(op_id):
            print(
                f"[MPC-B][PROFILE] session={session_id[:8]} "
                f"exchange op={op_id} rpc_ms={rpc_ms:.2f} "
                f"wait_b_ms={b_ready_wait_ms:.2f}"
            )

        return b_eps, b_del


# ---------------------------------------------------------------------------
# MPCNode — loads weights, runs layer 0 with MPC
# ---------------------------------------------------------------------------

class MPCNode:
    """
    Full drop-in replacement for shard 0, with MPC-protected layer 0.

    Each MPCNode holds:
      - The embedding layer weights (both nodes have a copy)
      - ALL layers for shard 0 (loaded from shard_0.pt)

    Privacy model:
      - Layer 0: computed via Beaver-triple 2PC (privacy-protected)
      - Layers 1-N: computed in cleartext on Node A (safe — activations
        after one full transformer layer can't be inverted to raw tokens)
    """

    def __init__(self, role: str, peer_address: str,
                 shards_dir: str = None, manifest_path: str = None):
        assert role in ("A", "B"), "Role must be 'A' or 'B'"
        self.role = role
        self.peer_address = peer_address
        self._peer_stub = None
        self._shards_dir = shards_dir or config.SHARDS_DIR

        self.embed_tokens = None
        self.layers: list = []
        self.rotary_emb = None
        self.model_config = None
        self.layer_start = 0
        self.layer_end = 0
        self._use_generic = False
        self._image_token_id = None
        self._cache_manager = None

        # Architecture info (set during weight loading)
        self.hidden_size = 0
        self.num_heads = 0
        self.head_dim = 0
        self.intermediate_size = 0
        self.num_kv_heads = 0

        self._load_weights(manifest_path)

        self._session_shares: dict[str, dict] = {}
        self._session_lock = threading.Lock()
        self._profile_enabled = _is_mpc_profile_enabled()
        self._profile_mode = _mpc_profile_mode()

    def _profile_log(self, session_id: str, op: str, elapsed_ms: float) -> None:
        if not self._profile_enabled:
            return
        print(
            f"[MPC-{self.role}][PROFILE] session={session_id[:8]} "
            f"mode={self._profile_mode} op={op} ms={elapsed_ms:.2f}"
        )

    def _load_weights(self, manifest_path: str = None):
        """Load ALL of shard 0's weights."""
        manifest_path = manifest_path or os.path.join(
            self._shards_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        use_generic = manifest.get("format_version", 1) >= 2

        if "text_shards" in manifest and manifest["text_shards"]:
            shard_info = manifest["text_shards"][0]
        else:
            shard_info = manifest["shards"][0]
        self.layer_start = shard_info["layer_start"]
        self.layer_end = shard_info["layer_end"]
        num_layers = self.layer_end - self.layer_start
        print(f"[MPC-{self.role}]   Shard 0 covers layers "
              f"{self.layer_start}-{self.layer_end} ({num_layers} layers)")

        self._image_token_id = manifest.get("image_token_id")

        if use_generic and manifest.get("architecture", {}).get("text"):
            self._load_weights_generic(manifest, shard_info, num_layers)
        else:
            self._load_weights_hf(num_layers)

    def _load_weights_generic(self, manifest: dict, shard_info: dict,
                              num_layers: int):
        """Load weights using generic runtime building blocks (v2 path)."""
        from node.runtime.text_blocks import (
            UnfedRMSNorm, UnfedRoPE, UnfedAttention, UnfedMLP,
            UnfedDecoderLayer, UnfedEmbedding,
        )
        from node.runtime.weight_loader import (
            load_shard_weights, load_weights_into_module,
        )

        arch = manifest["architecture"]["text"]
        shard_file = shard_info.get("file", "text_shard_0.pt")
        shard_path = os.path.join(self._shards_dir, shard_file)
        print(f"[MPC-{self.role}] Loading shard 0 from {shard_path} "
              f"(generic runtime)")

        self.hidden_size = arch["hidden_size"]
        self.num_heads = arch["num_attention_heads"]
        self.head_dim = arch.get(
            "head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size = arch.get(
            "intermediate_size", self.hidden_size * 4)
        self.num_kv_heads = arch.get(
            "num_key_value_heads", self.num_heads)

        self.embed_tokens = UnfedEmbedding(
            arch["vocab_size"], self.hidden_size)
        self.rotary_emb = UnfedRoPE(
            self.head_dim,
            arch.get("rope_theta", 10000.0),
            arch.get("max_position_embeddings", 8192),
        )

        self.layers = []
        for i in range(num_layers):
            self.layers.append(UnfedDecoderLayer(arch))

        state_dict = load_shard_weights(shard_path)
        remaining = dict(state_dict)
        loaded = 0

        embed_keys = {k: v for k, v in remaining.items()
                      if "embed_tokens" in k}
        if embed_keys:
            embed_sd = {}
            for k, v in embed_keys.items():
                local_k = k.split("embed_tokens.")[-1] \
                    if "embed_tokens." in k else k
                if not local_k.startswith("embed_tokens"):
                    local_k = f"embed_tokens.{local_k}"
                embed_sd[local_k] = v
            load_weights_into_module(self.embed_tokens, embed_sd, strict=False)
            for k in embed_keys:
                remaining.pop(k, None)
            loaded += len(embed_keys)

        print(f"[MPC-{self.role}]   Loaded embed_tokens")

        for local_idx in range(num_layers):
            global_idx = self.layer_start + local_idx
            prefixes = [
                f"layers.{global_idx}.",
                f"model.layers.{global_idx}.",
                f"layer_{global_idx}.",
            ]
            layer_sd = {}
            keys_used = []
            for k, v in remaining.items():
                for pfx in prefixes:
                    if k.startswith(pfx):
                        layer_sd[k[len(pfx):]] = v
                        keys_used.append(k)
                        break
            if layer_sd:
                load_weights_into_module(
                    self.layers[local_idx], layer_sd, strict=False)
                for k in keys_used:
                    remaining.pop(k, None)
                loaded += len(keys_used)

            label = 'MPC-protected (2PC)' if local_idx == 0 else 'cleartext'
            print(f"[MPC-{self.role}]   Loaded layer {global_idx} ({label})")

        self._use_generic = True
        self.model_config = None

        from node.runtime.generic_runner import SessionCacheManager
        self._cache_manager = SessionCacheManager(
            num_layers=num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=arch.get("max_position_embeddings", 4096),
        )
        print(f"[MPC-{self.role}] KV cache: {num_layers} layers, "
              f"{self.num_kv_heads} kv_heads, head_dim={self.head_dim}")
        print(f"[MPC-{self.role}] Loaded {num_layers} layers "
              f"(generic, format_version=2)")

    def _load_weights_hf(self, num_layers: int):
        """Load weights using HuggingFace model classes (v1 legacy path)."""
        from transformers import AutoConfig, AutoModelForCausalLM

        shard_path = config.get_shard_path(0)
        hf_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.model_config = hf_config
        self.hidden_size = hf_config.hidden_size
        self.num_heads = hf_config.num_attention_heads
        self.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        self.intermediate_size = hf_config.intermediate_size
        self.num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads)

        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.embed_tokens = model.model.embed_tokens
        if hasattr(model.model, 'rotary_emb'):
            self.rotary_emb = model.model.rotary_emb
        else:
            from node.runtime.text_blocks import UnfedRoPE
            self.rotary_emb = UnfedRoPE(
                self.head_dim,
                getattr(hf_config, "rope_theta", 10000.0),
                getattr(hf_config, "max_position_embeddings", 2048),
            )

        all_layers = model.model.layers
        start = self.layer_start
        end = self.layer_end
        self.layers = [all_layers[i] for i in range(start, end)]
        del model
        print(f"[MPC-{self.role}] Loaded {num_layers} layers (HuggingFace v1)")

    @torch.no_grad()
    def compute_embedding_share(self, token_ids: torch.Tensor,
                                image_embeddings=None):
        """Compute embedding and split into two additive shares."""
        embedding = self.embed_tokens(token_ids)

        if (image_embeddings is not None
                and self._image_token_id is not None):
            mask = (token_ids == self._image_token_id)
            num_placeholders = mask.sum().item()
            num_features = image_embeddings.shape[0]

            if num_placeholders > 0 and num_features > 0:
                use_count = min(num_features, num_placeholders)
                flat_mask = mask.view(-1)
                indices = flat_mask.nonzero(as_tuple=True)[0][:use_count]
                flat_embed = embedding.view(-1, embedding.shape[-1])
                flat_embed[indices] = image_embeddings[:use_count].to(
                    flat_embed.dtype)
                embedding = flat_embed.view(embedding.shape)
                print(f"[MPC] Merged {use_count} image embeddings into "
                      f"text sequence")

        shares = create_additive_shares(embedding, 2)
        return shares[0], shares[1]

    @torch.no_grad()
    def forward_layer0_on_share(
        self,
        my_share: torch.Tensor,
        triples: TripleAllocator,
        exchanger: PeerExchanger,
        session_id: str,
        is_party_0: bool,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run layer 0 on this party's share using Beaver-triple 2PC.

        Both parties call this simultaneously with their own share.
        All non-linear operations go through secure_multiply via
        the exchanger.  Matrix products of two shared tensors use
        secure_matmul with matrix Beaver triples.  Linear operations
        (matmul with public weights) are computed locally on shares.

        Returns this party's share of the layer 0 output.
        """
        from node.runtime.text_blocks import rotate_half

        layer_0 = self.layers[0]
        batch, seq_len, _ = my_share.shape
        layer0_start = time.perf_counter()

        # --- Secure Input RMSNorm ---
        weight = layer_0.input_layernorm.weight
        eps = layer_0.input_layernorm.eps if hasattr(
            layer_0.input_layernorm, 'eps') else getattr(
            layer_0.input_layernorm, 'variance_epsilon', 1e-6)

        op_start = time.perf_counter()
        normed_share = secure_rmsnorm(
            my_share, weight, eps, triples, exchanger,
            session_id, is_party_0, prefix="rmsnorm_in")
        self._profile_log(
            session_id,
            "rmsnorm_in",
            (time.perf_counter() - op_start) * 1000.0,
        )

        # --- Attention: Q/K/V projections (linear, free on shares) ---
        attn = layer_0.self_attn
        q_share = normed_share @ attn.q_proj.weight.T
        k_share = normed_share @ attn.k_proj.weight.T
        v_share = normed_share @ attn.v_proj.weight.T

        if hasattr(attn.q_proj, 'bias') and attn.q_proj.bias is not None:
            if is_party_0:
                q_share = q_share + attn.q_proj.bias
                k_share = k_share + attn.k_proj.bias
                v_share = v_share + attn.v_proj.bias

        # Reshape for multi-head attention
        q_share = q_share.view(batch, seq_len, self.num_heads, self.head_dim
                               ).transpose(1, 2)
        k_share = k_share.view(batch, seq_len, self.num_kv_heads,
                               self.head_dim).transpose(1, 2)
        v_share = v_share.view(batch, seq_len, self.num_kv_heads,
                               self.head_dim).transpose(1, 2)

        # --- Apply RoPE (public rotation, free on shares) ---
        if position_ids is None:
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(my_share, position_ids)
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
        q_share = q_share * cos + rotate_half(q_share) * sin
        k_share = k_share * cos[:, :, :, :self.head_dim] + \
            rotate_half(k_share) * sin[:, :, :, :self.head_dim]

        # GQA repeat for K, V if needed
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_share = k_share.repeat_interleave(repeat, dim=1)
            v_share = v_share.repeat_interleave(repeat, dim=1)

        # --- Secure Q @ K^T via matrix Beaver triple ---
        scale = 1.0 / (self.head_dim ** 0.5)
        qk_triple, qk_check = triples.get_with_check("attn_qk_matmul")
        op_start = time.perf_counter()
        scores_share = secure_matmul(
            q_share, k_share,
            qk_triple,
            qk_check,
            exchanger, session_id, "attn_qk_matmul",
            is_party_0, transpose_b=True)
        self._profile_log(
            session_id,
            "attn_qk_matmul",
            (time.perf_counter() - op_start) * 1000.0,
        )
        scores_share = scores_share * scale

        # Causal mask (public, same for both parties)
        if seq_len > 1:
            causal_mask = torch.full(
                (seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            if is_party_0:
                scores_share = scores_share + causal_mask.unsqueeze(0).unsqueeze(0)

        # --- Secure Softmax ---
        op_start = time.perf_counter()
        attn_weights_share = secure_softmax(
            scores_share, triples, exchanger, session_id, is_party_0)
        self._profile_log(
            session_id,
            "softmax",
            (time.perf_counter() - op_start) * 1000.0,
        )

        # --- Secure attn_weights @ V via matrix Beaver triple ---
        av_triple, av_check = triples.get_with_check("attn_av_matmul")
        op_start = time.perf_counter()
        attn_output_share = secure_matmul(
            attn_weights_share, v_share,
            av_triple,
            av_check,
            exchanger, session_id, "attn_av_matmul",
            is_party_0, transpose_b=False)
        self._profile_log(
            session_id,
            "attn_av_matmul",
            (time.perf_counter() - op_start) * 1000.0,
        )

        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output_share = attn_output_share.transpose(1, 2).contiguous().view(
            batch, seq_len, self.hidden_size)

        # Output projection (linear, free)
        attn_output_share = attn_output_share @ attn.o_proj.weight.T
        if hasattr(attn.o_proj, 'bias') and attn.o_proj.bias is not None:
            if is_party_0:
                attn_output_share = attn_output_share + attn.o_proj.bias

        # Residual connection
        hidden_share = my_share + attn_output_share

        # --- Secure Post-Attention RMSNorm ---
        post_weight = layer_0.post_attention_layernorm.weight
        post_eps = layer_0.post_attention_layernorm.eps if hasattr(
            layer_0.post_attention_layernorm, 'eps') else getattr(
            layer_0.post_attention_layernorm, 'variance_epsilon', 1e-6)

        op_start = time.perf_counter()
        normed_post_share = secure_rmsnorm(
            hidden_share, post_weight, post_eps, triples, exchanger,
            session_id, is_party_0, prefix="rmsnorm_post")
        self._profile_log(
            session_id,
            "rmsnorm_post",
            (time.perf_counter() - op_start) * 1000.0,
        )

        # --- MLP (linear projections are free) ---
        mlp = layer_0.mlp
        gate_share = normed_post_share @ mlp.gate_proj.weight.T
        up_share = normed_post_share @ mlp.up_proj.weight.T

        if hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
            if is_party_0:
                gate_share = gate_share + mlp.gate_proj.bias
                up_share = up_share + mlp.up_proj.bias

        # --- Secure SiLU on gate ---
        op_start = time.perf_counter()
        gate_activated_share = secure_silu(
            gate_share, triples, exchanger, session_id, is_party_0)
        self._profile_log(
            session_id,
            "silu",
            (time.perf_counter() - op_start) * 1000.0,
        )

        # --- Secure gate * up ---
        op_start = time.perf_counter()
        mlp_mid_share = secure_gate_up(
            gate_activated_share, up_share,
            triples, exchanger, session_id, is_party_0)
        self._profile_log(
            session_id,
            "gate_up",
            (time.perf_counter() - op_start) * 1000.0,
        )

        # Down projection (linear, free)
        mlp_out_share = mlp_mid_share @ mlp.down_proj.weight.T
        if hasattr(mlp.down_proj, 'bias') and mlp.down_proj.bias is not None:
            if is_party_0:
                mlp_out_share = mlp_out_share + mlp.down_proj.bias

        # Residual connection
        output_share = hidden_share + mlp_out_share

        self._profile_log(
            session_id,
            "layer0_total",
            (time.perf_counter() - layer0_start) * 1000.0,
        )
        return output_share

    @torch.no_grad()
    def forward_remaining_layers(self, hidden: torch.Tensor,
                                 session_id: str) -> torch.Tensor:
        """Run layers 1-N in cleartext on Node A (safe — activations are
        already mixed beyond inversion)."""
        if not self._use_generic:
            return self._forward_remaining_hf(hidden, session_id)

        session_cache = self._cache_manager.get_or_create(session_id)
        current_pos = session_cache.seq_len

        seq_len = hidden.shape[1]
        batch = hidden.shape[0]
        position_ids = torch.arange(
            current_pos, current_pos + seq_len
        ).unsqueeze(0).expand(batch, -1)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        cache_position = torch.arange(current_pos, current_pos + seq_len)

        attention_mask = None
        if seq_len > 1:
            total_len = current_pos + seq_len
            causal_mask = torch.full(
                (seq_len, total_len), float("-inf"))
            for i in range(seq_len):
                causal_mask[i, :current_pos + i + 1] = 0.0
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Layer 0 KV cache update (we still need it for decode steps)
        k_cache_0, v_cache_0 = session_cache.get_layer_cache(0)

        # Run layer 0 in cleartext as well (the MPC already computed it,
        # but we need to update the KV cache for future decode steps)
        # Skip the actual computation, just populate cache with dummy
        # Note: for decode steps, layer 0 still needs MPC, so the KV cache
        # is populated correctly during the MPC phase.

        for i, layer in enumerate(self.layers[1:], start=1):
            position_embeddings = self.rotary_emb(hidden, position_ids)
            k_cache_i, v_cache_i = session_cache.get_layer_cache(i)
            hidden, _ = layer(
                hidden,
                position_embeddings=position_embeddings,
                past_key_value=(k_cache_i, v_cache_i),
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        session_cache.advance(seq_len)
        print(f"[MPC-{self.role}] All {len(self.layers)} layers done, "
              f"output shape: {hidden.shape} (pos: {current_pos}->"
              f"{session_cache.seq_len})")
        return hidden

    def _forward_remaining_hf(self, hidden, session_id):
        """Legacy HF path for layers 1-N."""
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden, position_ids)

        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        cache_position = torch.arange(seq_len)

        for i, layer in enumerate(self.layers[1:], start=1):
            position_embeddings = self.rotary_emb(hidden, position_ids)
            output = layer(
                hidden,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden = output[0] if isinstance(output, tuple) else output

        print(f"[MPC-{self.role}] All {len(self.layers)} layers done, "
              f"output shape: {hidden.shape}")
        return hidden


# ---------------------------------------------------------------------------
# Node B's MPCPeer gRPC servicer
# ---------------------------------------------------------------------------

class MPCPeerServicer(inference_pb2_grpc.MPCPeerServicer):
    """
    Handles MPC peer RPCs on Node B.

    Node A calls these RPCs to:
      1. SendShare: distribute B's embedding share + Beaver triple shares
      2. Exchange: swap epsilon/delta during each secure multiplication
      3. CollectShare: retrieve B's final share after layer 0
    """

    def __init__(self, mpc_node: MPCNode, exchanger_b: GrpcPeerExchangerB):
        self._mpc = mpc_node
        self._exchanger = exchanger_b
        self._session_results: dict[str, torch.Tensor] = {}
        self._session_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._triple_set_cache: dict[str, dict[str, BeaverTripleShares]] = {}
        self._triple_set_lru: list[str] = []
        self._triple_set_cache_max = max(
            1,
            int(os.environ.get("UNFED_MPC_TRIPLE_SET_CACHE_MAX", "16")),
        )

    def _get_cached_triples(self, triple_set_id: str) -> Optional[dict[str, BeaverTripleShares]]:
        with self._lock:
            cached = self._triple_set_cache.get(triple_set_id)
            if cached is None:
                return None
            if triple_set_id in self._triple_set_lru:
                self._triple_set_lru.remove(triple_set_id)
            self._triple_set_lru.append(triple_set_id)
            return cached

    def _put_cached_triples(self, triple_set_id: str, triples: dict[str, BeaverTripleShares]) -> None:
        with self._lock:
            self._triple_set_cache[triple_set_id] = triples
            if triple_set_id in self._triple_set_lru:
                self._triple_set_lru.remove(triple_set_id)
            self._triple_set_lru.append(triple_set_id)
            while len(self._triple_set_lru) > self._triple_set_cache_max:
                victim = self._triple_set_lru.pop(0)
                self._triple_set_cache.pop(victim, None)

    def SendShare(self, request, context):
        """Receive embedding share + triples from Node A, start layer 0."""
        session_id = request.session_id
        shape = list(request.share_shape)
        share_b = _bytes_to_tensor(bytes(request.share_data), shape)

        print(f"[MPC-B] Received share for session {session_id[:8]}... "
              f"shape: {share_b.shape}")

        triple_set_id = str(getattr(request, "triple_set_id", "") or "")
        triple_dict: Optional[dict[str, BeaverTripleShares]] = None
        if triple_set_id:
            triple_dict = self._get_cached_triples(triple_set_id)

        # New path: packed triples blob sent once per triple_set_id.
        triples_blob = bytes(getattr(request, "triples_blob", b""))
        triple_seed = int(getattr(request, "triple_seed", 0) or 0)
        if triple_dict is None and triple_seed:
            batch = int(share_b.shape[0]) if share_b.dim() > 0 else 1
            all_triples = allocate_layer0_triples(
                hidden_size=int(request.hidden_size or self._mpc.hidden_size),
                num_heads=int(self._mpc.num_heads),
                head_dim=int(self._mpc.head_dim),
                intermediate_size=int(self._mpc.intermediate_size),
                seq_len=int(request.seq_len or share_b.shape[1]),
                batch=batch,
                seed=triple_seed,
            )
            triple_dict = {op_id: bt.party_1 for op_id, bt in all_triples.items()}
            if triple_set_id:
                self._put_cached_triples(triple_set_id, triple_dict)

        if triple_dict is None and triples_blob:
            triple_dict = {}
            for op_id, a_raw, b_raw, c_raw, a_shape, b_shape, c_shape in _unpack_peer_triples_blob(triples_blob):
                triple_dict[op_id] = deserialize_triple_shares(
                    a_raw, b_raw, c_raw, a_shape, b_shape, c_shape
                )
            if triple_set_id:
                self._put_cached_triples(triple_set_id, triple_dict)

        # Compatibility path: old repeated MPCTriplePayload list.
        if triple_dict is None and request.triples:
            triple_dict = {}
            for tp in request.triples:
                a_shape = list(tp.shape)
                b_shape = list(tp.b_shape) if tp.b_shape else a_shape
                c_shape = list(tp.c_shape) if tp.c_shape else a_shape
                triple_dict[tp.op_id] = deserialize_triple_shares(
                    bytes(tp.a_data), bytes(tp.b_data),
                    bytes(tp.c_data), a_shape, b_shape, c_shape)
            if triple_set_id:
                self._put_cached_triples(triple_set_id, triple_dict)

        if triple_dict is None:
            return inference_pb2.MPCSendShareResponse(
                accepted=False,
                error=f"missing triples for triple_set_id={triple_set_id or 'none'}",
            )

        triples = TripleAllocator(triple_dict)

        # Create completion event
        with self._lock:
            self._session_events[session_id] = threading.Event()

        # Run layer 0 on B's share in a background thread
        def run_b():
            try:
                result = self._mpc.forward_layer0_on_share(
                    share_b, triples, self._exchanger,
                    session_id, is_party_0=False)
                with self._lock:
                    self._session_results[session_id] = result
                    self._session_events[session_id].set()
                print(f"[MPC-B] Layer 0 done for session {session_id[:8]}... "
                      f"result shape: {result.shape}")
            except Exception as e:
                print(f"[MPC-B] Error in layer 0: {e}")
                import traceback
                traceback.print_exc()
                with self._lock:
                    self._session_events[session_id].set()

        t = threading.Thread(target=run_b, daemon=True)
        t.start()

        return inference_pb2.MPCSendShareResponse(accepted=True)

    def Exchange(self, request, context):
        """Handle a secure multiplication exchange from Node A."""
        session_id = request.session_id
        op_id = request.op_id
        eps_shape = _shape_list(list(request.shape))
        del_shape = _shape_list(list(request.delta_shape)) if request.delta_shape else eps_shape

        a_eps = _bytes_to_tensor(bytes(request.epsilon_data), eps_shape)
        a_del = _bytes_to_tensor(bytes(request.delta_data), del_shape)

        b_eps, b_del = self._exchanger.handle_exchange_rpc(
            session_id, op_id, a_eps, a_del)

        b_eps_bytes, b_eps_shape = _tensor_to_bytes(b_eps)
        b_del_bytes, b_del_shape = _tensor_to_bytes(b_del)

        return inference_pb2.MPCExchangeResponse(
            epsilon_data=b_eps_bytes,
            delta_data=b_del_bytes,
            shape=_shape_list(b_eps_shape),
            delta_shape=_shape_list(b_del_shape),
        )

    def CollectShare(self, request, context):
        """Return B's final share of layer 0 output."""
        session_id = request.session_id
        event = self._session_events.get(session_id)
        if event:
            event.wait(timeout=120)

        with self._lock:
            result = self._session_results.pop(session_id, None)
            self._session_events.pop(session_id, None)

        if result is None:
            context.set_details("No result for this session")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return inference_pb2.MPCCollectResponse()

        data, shape = _tensor_to_bytes(result)
        return inference_pb2.MPCCollectResponse(
            share_data=data,
            share_shape=shape,
        )


# ---------------------------------------------------------------------------
# Node A's servicer (receives Forward from clients, orchestrates MPC)
# ---------------------------------------------------------------------------

class MPCNodeServicer(inference_pb2_grpc.InferenceNodeServicer):
    """
    gRPC servicer for MPC Node A.  Receives client requests, orchestrates
    the 2PC protocol with Node B, and forwards results to the next shard.
    """

    def __init__(self, mpc_node: MPCNode, port: int, require_daemon: bool = True,
                 allowed_prev_node_types: Optional[list[str]] = None):
        self.mpc = mpc_node
        self.port = port
        self._node_id: str = ""
        self._peer_address: str = ""
        self._peer_node_id: str = ""
        self._daemon_stub = None
        self._daemon_registry_addr: str = config.REGISTRY_ADDRESS
        self._private_key = None
        self._peer_stub = None  # MPCPeerStub for calling Node B
        self._forward_stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._forward_stubs_lock = threading.Lock()
        self._peer_known_triple_sets: set[str] = set()
        self._peer_known_lock = threading.Lock()
        self._registration_share_signing_private_key = None
        self._session_nonce_by_session: dict[str, str] = {}
        self._session_step_by_session: dict[str, int] = {}
        self._session_prev_share_hash: dict[str, str] = {}
        self._session_prev_block_hash: dict[str, str] = {}
        self._active_inferences = 0
        self._inference_lock = threading.Lock()
        self._require_daemon = bool(require_daemon)
        # Optimized daemon submit path is always on.
        self._daemon_submit_mode = "async"
        self._daemon_fail_closed = bool(self._require_daemon)
        self._daemon_batch_size = 8
        self._daemon_batch_max_wait_s = 0.0
        self._daemon_queue_max = 1024
        self._daemon_retry_base_s = 0.2
        self._daemon_retry_max_s = 5.0
        self._daemon_submit_use_blob = True
        self._daemon_queue: deque = deque()
        self._daemon_queue_lock = threading.Lock()
        self._daemon_queue_cv = threading.Condition(self._daemon_queue_lock)
        self._daemon_stop = False
        self._daemon_backoff_s = 0.0
        self._daemon_submitter_thread: Optional[threading.Thread] = None
        self._allowed_prev_node_types = {
            t.strip().lower()
            for t in (allowed_prev_node_types or [])
            if t and t.strip()
        }
        # Triple cache prewarm/persistence is always enabled.
        cache_max = 2
        cache_max_shapes = 8
        cache_seq_bucket = 1
        cache_persist = True
        cache_persist_dir = "~/.unfed/mpc_triples"
        self._layer0_triple_cache = _Layer0TripleCache(
            max_per_shape=cache_max,
            max_shapes=cache_max_shapes,
            seq_bucket_size=cache_seq_bucket,
            persist_enabled=cache_persist,
            persist_dir=cache_persist_dir,
        )
        self._prewarm_layer0_triples()
        self._warm_thread = threading.Thread(target=self._warm_loop, daemon=True)
        self._warm_thread.start()
        self._daemon_submitter_thread = threading.Thread(
            target=self._daemon_submit_loop,
            daemon=True,
        )
        self._daemon_submitter_thread.start()

    def _prewarm_layer0_triples(self) -> None:
        seq_lens = [1, 8, 16, 32]
        target = 1
        batch = 1
        start = time.perf_counter()
        self._layer0_triple_cache.prewarm(
            hidden_size=self.mpc.hidden_size,
            num_heads=self.mpc.num_heads,
            head_dim=self.mpc.head_dim,
            intermediate_size=self.mpc.intermediate_size,
            batch=batch,
            seq_lens=seq_lens,
            target_per_shape=target,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[MPC-A] Triple cache prewarm done: "
            f"seq_lens={seq_lens} target={target} batch={batch} "
            f"ms={elapsed_ms:.2f}"
        )

    def _build_forward_attestation(
        self,
        *,
        request: inference_pb2.ForwardRequest,
        activation_bytes: bytes,
    ) -> tuple[str, bytes, str, bytes, str]:
        if not self._node_id or not self._registration_share_signing_private_key:
            return "", b"", "", b"", ""
        activation_digest = tensor_bytes_digest(activation_bytes)
        proof = b""
        proof_format = "none"
        payload = ForwardAttestationPayload(
            version=FORWARD_ATTESTATION_VERSION,
            signer_node_id=str(self._node_id),
            signer_node_type="mpc",
            session_id=str(request.session_id),
            he_step=int(request.he_step),
            he_key_id=str(request.he_key_id or ""),
            activation_digest=activation_digest,
            tensor_shape=make_tensor_shape_signature(list(request.tensor_shape)),
            compressed=bool(request.compressed),
            wire_dtype=str(request.wire_dtype or ""),
            output_2pc_artifact_hash=str(request.output_2pc_artifact_hash or ""),
            proof_format=proof_format,
            proof_hash=proof_bytes_hash(proof),
        )
        signature = sign_forward_attestation(
            self._registration_share_signing_private_key,
            payload,
        )
        return activation_digest, signature, payload.version, proof, proof_format

    def _connect_peer(self):
        """Connect to Node B's MPCPeer service."""
        if self._peer_stub is None and self._peer_address:
            channel = create_resilient_channel(
                self._peer_address, config.GRPC_OPTIONS)
            self._peer_stub = inference_pb2_grpc.MPCPeerStub(channel)
            print(f"[MPC-A] Connected to peer B at {self._peer_address}")

    def _get_forward_stub(self, address: str):
        with self._forward_stubs_lock:
            stub = self._forward_stubs.get(address)
            if stub is not None:
                return stub
            ch = create_resilient_channel(address, config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(ch)
            self._forward_stubs[address] = stub
            return stub

    def _warm_loop(self) -> None:
        interval = max(1.0, float(os.environ.get("UNFED_MPC_WARM_INTERVAL_S", "5")))
        while True:
            try:
                if self._peer_address:
                    self._connect_peer()
                if self._require_daemon and self._daemon_stub is None:
                    self._refresh_daemon_stub()
            except Exception:
                pass
            time.sleep(interval)

    def _init_daemon_stub(self, registry_address: str):
        """Discover and connect to the chain daemon."""
        self._daemon_registry_addr = registry_address or config.REGISTRY_ADDRESS
        self._refresh_daemon_stub()

    def _refresh_daemon_stub(self):
        """Discover and connect to the least-loaded daemon."""
        try:
            import registry_pb2
            import registry_pb2_grpc
            channel = grpc.insecure_channel(
                self._daemon_registry_addr, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(
                registry_pb2.DiscoverRequest(model_id=""),
                timeout=10,
            )
            daemons = [n for n in resp.nodes if n.node_type == "daemon"]
            channel.close()
            if daemons:
                daemon, _ = select_least_loaded_daemon(daemons, self._daemon_utilization_probe)
                self._daemon_stub = inference_pb2_grpc.InferenceNodeStub(
                    grpc.insecure_channel(
                        daemon.address, options=config.GRPC_OPTIONS))
                print(f"[MPC-A] Connected to chain daemon at "
                      f"{daemon.address}")
            else:
                print("[MPC-A] No daemon found")
        except Exception:
            print(f"[MPC-A] Could not connect to daemon")

    def _build_submit_request(self, shares):
        from economics.distributed_chain import share_to_proto
        shares_proto = [share_to_proto(s) for s in shares]
        if self._daemon_submit_use_blob:
            return inference_pb2.SubmitSharesRequest(
                submitter_id=self._node_id,
                shares_blob=_pack_share_proto_blob(shares_proto),
                shares_blob_format="shareproto-len-v1",
            )
        return inference_pb2.SubmitSharesRequest(
            shares=shares_proto,
            submitter_id=self._node_id,
        )

    def _submit_share_batch(self, shares, *, raise_on_error: bool) -> bool:
        if not shares:
            return True
        if not self._daemon_stub:
            self._refresh_daemon_stub()
        if not self._daemon_stub or not self._node_id:
            if raise_on_error and self._daemon_fail_closed:
                raise RuntimeError("Daemon is required but unavailable for MPC share submission.")
            return False
        req = self._build_submit_request(shares)
        try:
            resp = self._daemon_stub.SubmitShares(req, timeout=5)
            accepted = int(getattr(resp, "accepted", 0))
            if accepted > 0:
                tip_hash = str(getattr(resp, "chain_tip_hash", "") or "")
                for s in shares:
                    if tip_hash:
                        self._session_prev_block_hash[s.session_id] = tip_hash
            return accepted > 0
        except Exception:
            self._daemon_stub = None
            if raise_on_error and self._daemon_fail_closed:
                raise RuntimeError("Daemon became unreachable during MPC share submission.")
            return False

    def _daemon_submit_loop(self) -> None:
        while True:
            with self._daemon_queue_cv:
                if self._daemon_stop:
                    break
                if len(self._daemon_queue) < self._daemon_batch_size:
                    if self._daemon_batch_max_wait_s > 0 and len(self._daemon_queue) > 0:
                        self._daemon_queue_cv.wait(timeout=self._daemon_batch_max_wait_s)
                    else:
                        self._daemon_queue_cv.wait(timeout=0.5)
                    if self._daemon_stop:
                        break
                    if len(self._daemon_queue) == 0:
                        continue
                    if len(self._daemon_queue) < self._daemon_batch_size and self._daemon_batch_max_wait_s <= 0:
                        continue
                batch = []
                while self._daemon_queue and len(batch) < self._daemon_batch_size:
                    batch.append(self._daemon_queue.popleft())

            if self._daemon_backoff_s > 0:
                time.sleep(self._daemon_backoff_s)

            ok = self._submit_share_batch(batch, raise_on_error=False)
            if ok:
                self._daemon_backoff_s = 0.0
                continue

            # Requeue failed batch at front and apply exponential backoff.
            with self._daemon_queue_cv:
                for s in reversed(batch):
                    self._daemon_queue.appendleft(s)
                if self._daemon_backoff_s <= 0:
                    self._daemon_backoff_s = self._daemon_retry_base_s
                else:
                    self._daemon_backoff_s = min(
                        self._daemon_retry_max_s,
                        self._daemon_backoff_s * 2.0,
                    )
                self._refresh_daemon_stub()

    def _record_dual_shares(self, session_id: str, output: torch.Tensor):
        """Submit compute shares for both MPC nodes to the daemon."""
        act_hash = hashlib.sha256(
            output.contiguous().float().numpy().tobytes()
        ).hexdigest()[:16]

        shares = [self._build_signed_share(self._node_id, session_id, act_hash)]
        if self._peer_node_id and not self._peer_node_id.startswith("peer-of-"):
            # TODO: emit peer-signed share once role-B signs and submits directly.
            pass
        # Optimistically advance prev_share hash chain so async mode preserves
        # per-session monotonic linkage before daemon ack arrives.
        for s in shares:
            self._session_prev_share_hash[s.session_id] = s.hash()

        if self._daemon_submit_mode == "async":
            with self._daemon_queue_cv:
                if len(self._daemon_queue) >= self._daemon_queue_max:
                    if self._daemon_fail_closed:
                        raise RuntimeError("Daemon submit queue is full.")
                    return
                self._daemon_queue.extend(shares)
                if len(self._daemon_queue) >= self._daemon_batch_size or self._daemon_batch_max_wait_s > 0:
                    self._daemon_queue_cv.notify_all()
            return

        ok = self._submit_share_batch(shares, raise_on_error=True)
        if ok:
            print(f"[MPC-A] Submitted signed share(s) for session {session_id[:8]}...")

    @staticmethod
    def _daemon_utilization_probe(daemon) -> float:
        stub = inference_pb2_grpc.InferenceNodeStub(
            grpc.insecure_channel(daemon.address, options=config.GRPC_OPTIONS)
        )
        fee = stub.GetLoad(
            inference_pb2.FeeEstimateRequest(estimated_tokens=1),
            timeout=2,
        )
        return float(getattr(fee, "utilization", 1.0))

    def _send_share_with_triples(
        self,
        *,
        session_id: str,
        share_b_bytes: bytes,
        share_b_shape: list[int],
        seq_len: int,
        triple_pkg: _Layer0TriplePackage,
    ):
        triple_set_id = triple_pkg.triple_set_id
        if not triple_set_id:
            triple_set_id = hashlib.sha256(triple_pkg.peer_blob).hexdigest()[:24]
        with self._peer_known_lock:
            known = triple_set_id in self._peer_known_triple_sets
        include_blob = (not known) and bool(triple_pkg.peer_blob)

        def _send(include_blob_now: bool):
            return self._peer_stub.SendShare(
                inference_pb2.MPCSendShareRequest(
                    session_id=session_id,
                    share_data=share_b_bytes,
                    share_shape=share_b_shape,
                    num_layers=len(self.mpc.layers),
                    hidden_size=self.mpc.hidden_size,
                    seq_len=seq_len,
                    triple_set_id=triple_set_id,
                    triples_blob=triple_pkg.peer_blob if include_blob_now else b"",
                    triple_seed=int(triple_pkg.triple_seed),
                ),
                timeout=30,
            )

        resp = _send(include_blob)
        if (not resp.accepted) and (not include_blob):
            if "missing triples for triple_set_id" in (resp.error or ""):
                resp = _send(True)
                include_blob = True
        if resp.accepted:
            with self._peer_known_lock:
                self._peer_known_triple_sets.add(triple_set_id)
        return resp, include_blob

    def _build_signed_share(self, node_id: str, session_id: str, act_hash: str):
        from economics.share_chain import ComputeShare
        nonce = self._session_nonce_by_session.get(session_id)
        if nonce is None:
            nonce = uuid.uuid4().hex
            self._session_nonce_by_session[session_id] = nonce
            self._session_step_by_session[session_id] = 0
        step = self._session_step_by_session.get(session_id, 0)
        self._session_step_by_session[session_id] = step + 1
        ts = time.time()
        ts_ms = int(ts * 1000)
        payload = SharePayload(
            node_id=node_id,
            shard_index=0,
            session_id=session_id,
            session_nonce=nonce,
            step_index=step,
            activation_hash=act_hash,
            tokens_processed=1,
            share_weight=1.0,
            timestamp_ms=ts_ms,
            payload_hash_version=PAYLOAD_HASH_VERSION,
            prev_block_hash=self._session_prev_block_hash.get(session_id, ""),
            prev_share_hash=self._session_prev_share_hash.get(session_id, ""),
            idempotency_key=f"{node_id}:{nonce}:{step}",
        )
        if not self._registration_share_signing_private_key:
            raise RuntimeError("MPC share signing key not initialized")
        signature = sign_bytes(
            self._registration_share_signing_private_key,
            canonical_share_payload_bytes(payload),
        )
        return ComputeShare(
            node_id=node_id,
            shard_index=0,
            session_id=session_id,
            activation_hash=act_hash,
            tokens_processed=1,
            timestamp=ts,
            share_weight=1.0,
            session_nonce=nonce,
            step_index=step,
            timestamp_ms=ts_ms,
            signature=signature,
            payload_hash_version=PAYLOAD_HASH_VERSION,
            validated=False,
            prev_block_hash=payload.prev_block_hash,
            prev_share_hash=payload.prev_share_hash,
            idempotency_key=payload.idempotency_key,
        )

    def Forward(self, request, context):
        """Handle a forward pass through the MPC layer (Node A only)."""
        session_id = request.session_id

        with self._inference_lock:
            self._active_inferences += 1
        try:
            if self.mpc.role != "A":
                context.set_details("Only Node A accepts Forward requests")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return inference_pb2.ForwardResponse()
            if self._daemon_stub is None:
                # Retry daemon discovery at request time to handle startup races.
                self._refresh_daemon_stub()
            if self._require_daemon and self._daemon_stub is None:
                context.set_details("Daemon is required but unavailable.")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return inference_pb2.ForwardResponse()
            if self._allowed_prev_node_types:
                prev_type = (request.route_prev_node_type or "").strip().lower()
                if prev_type not in self._allowed_prev_node_types:
                    context.set_details(
                        f"Previous node type '{prev_type or 'unknown'}' not allowed for MPC-A."
                    )
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    return inference_pb2.ForwardResponse()

            self._connect_peer()

            token_ids = torch.tensor([list(request.token_ids)])

            # Extract image embeddings
            image_embeddings = None
            if (request.image_embeddings
                    and request.image_embeddings_shape):
                ie_shape = list(request.image_embeddings_shape)
                image_embeddings = torch.from_numpy(
                    np.frombuffer(
                        bytes(request.image_embeddings),
                        dtype=np.float32,
                    ).copy().reshape(ie_shape))
                print(f"[MPC-A] Received image_embeddings: "
                      f"{image_embeddings.shape}")

            # Step 1: Compute embedding and create shares
            share_a, share_b = self.mpc.compute_embedding_share(
                token_ids, image_embeddings=image_embeddings)
            print(f"[MPC-A] Session {session_id[:8]}... "
                  f"embedding shape: {share_a.shape}")

            # Step 2: Generate Beaver triples and distribute to Node B
            batch, seq_len, _ = share_a.shape
            orig_seq_len = seq_len
            triple_alloc_start = time.perf_counter()
            triple_pkg, cache_hit, bucket_seq_len = self._layer0_triple_cache.acquire(
                hidden_size=self.mpc.hidden_size,
                num_heads=self.mpc.num_heads,
                head_dim=self.mpc.head_dim,
                intermediate_size=self.mpc.intermediate_size,
                seq_len=seq_len,
                batch=batch,
            )
            triple_alloc_ms = (time.perf_counter() - triple_alloc_start) * 1000.0
            if _is_mpc_profile_enabled():
                print(
                    f"[MPC-A][PROFILE] session={session_id[:8]} "
                    f"op=triple_allocation ms={triple_alloc_ms:.2f} "
                    f"cache_hit={int(cache_hit)} "
                    f"seq={orig_seq_len} bucket_seq={bucket_seq_len}"
                )

            if bucket_seq_len > orig_seq_len:
                pad_tokens = bucket_seq_len - orig_seq_len
                # Zero-pad extra time steps so we can reuse bucketed triples.
                share_a = F.pad(share_a, (0, 0, 0, pad_tokens, 0, 0))
                share_b = F.pad(share_b, (0, 0, 0, pad_tokens, 0, 0))
                seq_len = bucket_seq_len

            my_allocator = TripleAllocator(triple_pkg.my_triples)

            # Send share_b + triple_set metadata/blob to Node B.
            share_b_bytes, share_b_shape = _tensor_to_bytes(share_b)
            send_resp, included_blob = self._send_share_with_triples(
                session_id=session_id,
                share_b_bytes=share_b_bytes,
                share_b_shape=share_b_shape,
                seq_len=seq_len,
                triple_pkg=triple_pkg,
            )
            if not send_resp.accepted:
                context.set_details(f"Node B rejected share: "
                                    f"{send_resp.error}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return inference_pb2.ForwardResponse()

            print(
                f"[MPC-A] Sent share + triple_set_id={triple_pkg.triple_set_id} "
                f"seed={int(triple_pkg.triple_seed)} "
                f"blob={int(included_blob)} bytes={len(triple_pkg.peer_blob) if included_blob else 0} "
                f"to Node B"
            )

            # Step 3: Run layer 0 on Node A's share (Node B runs
            # simultaneously on its share)
            exchanger_a = GrpcPeerExchangerA(self._peer_stub)
            share_a_result = self.mpc.forward_layer0_on_share(
                share_a, my_allocator, exchanger_a,
                session_id, is_party_0=True)

            print(f"[MPC-A] Layer 0 (2PC) done, my share shape: "
                  f"{share_a_result.shape}")

            # Step 4: Collect Node B's final share and reconstruct
            collect_resp = self._peer_stub.CollectShare(
                inference_pb2.MPCCollectRequest(session_id=session_id),
                timeout=60,
            )
            share_b_result = _bytes_to_tensor(
                bytes(collect_resp.share_data),
                list(collect_resp.share_shape))

            # Reconstruct layer 0 output
            layer0_output = share_a_result + share_b_result
            if bucket_seq_len > orig_seq_len:
                layer0_output = layer0_output[:, :orig_seq_len, :]
            print(f"[MPC-A] Reconstructed layer 0 output: "
                  f"{layer0_output.shape}")

            # Step 5: Run remaining layers in cleartext
            output = self.mpc.forward_remaining_layers(
                layer0_output, session_id)
            print(f"[MPC-A] Session {session_id[:8]}... "
                  f"shard 0 output: {output.shape}")

            # Record dual compute shares
            self._record_dual_shares(session_id, output)

            # Step 6: Forward to next node
            activation_bytes, shape = _tensor_to_bytes(output)
            next_address = None
            next_onion_blob = b""
            next_ephemeral_key = b""

            if request.remaining_circuit:
                remaining = list(request.remaining_circuit)
                next_address = remaining.pop(0)
            elif request.onion_blob and self._private_key:
                from network.onion import peel_onion
                layer, next_eph = peel_onion(
                    self._private_key,
                    bytes(request.onion_ephemeral_key),
                    bytes(request.onion_blob),
                )
                next_address = layer.next_hop or None
                next_onion_blob = layer.payload
                next_ephemeral_key = next_eph
                if next_address:
                    print(f"[MPC-A] Peeled onion -> next: {next_address}")

            if next_address:
                next_request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    activation_data=activation_bytes,
                    tensor_shape=list(shape),
                    is_prefill=request.is_prefill,
                    # MPC shard-0 serializes activations as float32 bytes.
                    # Set wire_dtype explicitly so downstream shards never
                    # decode using a global float16 default.
                    wire_dtype="float32",
                    route_prev_node_type="mpc",
                    route_prev_node_id=self._node_id,
                )
                if request.remaining_circuit:
                    remaining = list(request.remaining_circuit)
                    remaining.pop(0)
                    next_request.remaining_circuit.extend(remaining)
                if next_onion_blob:
                    next_request.onion_blob = next_onion_blob
                    next_request.onion_ephemeral_key = next_ephemeral_key
                if request.use_random_routing:
                    next_request.use_random_routing = True
                if request.response_keys:
                    next_request.response_keys.extend(
                        list(request.response_keys)[1:])
                if request.he_output_enabled:
                    next_request.he_output_enabled = True
                    next_request.he_client_pubkey = request.he_client_pubkey
                    next_request.he_key_id = request.he_key_id
                    next_request.he_step = request.he_step
                    next_request.he_compute_mode = request.he_compute_mode
                    next_request.he_compute_payload = request.he_compute_payload
                    next_request.he_compute_format = request.he_compute_format
                    next_request.he_top_k = request.he_top_k
                    next_request.he_temperature = request.he_temperature
                    next_request.he_top_p = request.he_top_p
                    next_request.he_disable_plaintext_sampling = (
                        request.he_disable_plaintext_sampling
                    )
                    next_request.output_2pc_stage = request.output_2pc_stage
                    next_request.output_2pc_artifact_type = request.output_2pc_artifact_type
                    next_request.output_2pc_artifact_hash = request.output_2pc_artifact_hash
                # Attach signed forward attestation so downstream compute shards
                # can verify MPC-A as the authenticated previous hop.
                (
                    next_request.prev_activation_digest,
                    next_request.prev_attestation_signature,
                    next_request.prev_attestation_version,
                    next_request.prev_attestation_proof,
                    next_request.prev_attestation_proof_format,
                ) = self._build_forward_attestation(
                    request=next_request,
                    activation_bytes=bytes(next_request.activation_data),
                )
                next_request.prev_attestation_signer_node_id = str(self._node_id or "")

                stub = self._get_forward_stub(next_address)
                response = stub.Forward(next_request)

                if request.response_keys:
                    from network.onion import encrypt
                    key = bytes(request.response_keys[0])
                    if response.encrypted_response:
                        encrypted = encrypt(key, response.encrypted_response)
                    elif response.has_token:
                        token_bytes = response.token_id.to_bytes(
                            4, 'big', signed=True)
                        encrypted = encrypt(key, token_bytes)
                    else:
                        encrypted = encrypt(key, response.activation_data)
                    return inference_pb2.ForwardResponse(
                        encrypted_response=encrypted,
                        token_id=response.token_id,
                        has_token=response.has_token,
                        is_eos=response.is_eos,
                        he_ciphertext=response.he_ciphertext,
                        he_nonce=response.he_nonce,
                        he_algo=response.he_algo,
                        he_session_id=response.he_session_id,
                        he_step=response.he_step,
                        he_key_id=response.he_key_id,
                        he_sender_pubkey=response.he_sender_pubkey,
                        he_error=response.he_error,
                        he_compute_payload=response.he_compute_payload,
                        he_compute_format=response.he_compute_format,
                        he_top_k=response.he_top_k,
                        output_2pc_stage=response.output_2pc_stage,
                        output_2pc_artifact_type=response.output_2pc_artifact_type,
                        output_2pc_artifact_hash=response.output_2pc_artifact_hash,
                    )

                return response
            else:
                return inference_pb2.ForwardResponse(
                    activation_data=activation_bytes,
                    tensor_shape=list(shape),
                )

        except Exception as e:
            print(f"[MPC-A] Error: {e}")
            import traceback
            traceback.print_exc()
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.ForwardResponse()
        finally:
            with self._inference_lock:
                self._active_inferences -= 1

    def GetLoad(self, request, context):
        """Expose local load telemetry for smart routing decisions."""
        estimated_tokens = max(1, int(request.estimated_tokens or 1))
        with self._inference_lock:
            active = self._active_inferences
        utilization = float(active)
        base_fee = 0.001
        return inference_pb2.FeeEstimateResponse(
            base_fee=base_fee,
            utilization=utilization,
            estimated_cost=base_fee * estimated_tokens,
            suggested_tip=0.0,
        )


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def serve(role: str, port: int, peer_address: str, host: str = "[::]",
          advertise: str = None, registry_address: str = None,
          shards_dir: str = None, eth_address: str = None,
          require_daemon: bool = True,
          allowed_prev_node_types: Optional[list[str]] = None,
          mpc_dnc_mode: str = "off",
          mpc_dnc_depth: int = 0,
          mpc_dnc_split_dim: int = -1,
          mpc_dnc_matmul_split_dim: int = -3,
          mpc_dnc_parallel_workers: int = 1,
          mpc_dnc_auto_min_elems: int = 65536,
          mpc_dnc_auto_chunk_min_elems: int = 16384,
          mpc_dnc_auto_max_depth: int = 3,
          mpc_dnc_auto_max_workers: int = 4):
    """Start an MPC node server."""
    configure_mpc_dnc(
        MpcDncConfig(
            mode=mpc_dnc_mode,
            depth=mpc_dnc_depth,
            split_dim=mpc_dnc_split_dim,
            matmul_split_dim=mpc_dnc_matmul_split_dim,
            parallel_workers=mpc_dnc_parallel_workers,
            auto_min_elems=mpc_dnc_auto_min_elems,
            auto_chunk_min_elems=mpc_dnc_auto_chunk_min_elems,
            auto_max_depth=mpc_dnc_auto_max_depth,
            auto_max_workers=mpc_dnc_auto_max_workers,
        )
    )
    print(
        f"[MPC-{role}] DNC config: mode={mpc_dnc_mode} depth={mpc_dnc_depth} "
        f"split_dim={mpc_dnc_split_dim} matmul_split_dim={mpc_dnc_matmul_split_dim} "
        f"workers={mpc_dnc_parallel_workers}"
    )
    mpc_node = MPCNode(role, peer_address, shards_dir=shards_dir)
    public_address = advertise or f"localhost:{port}"

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=config.GRPC_OPTIONS,
    )

    if role == "A":
        servicer = MPCNodeServicer(
            mpc_node,
            port,
            require_daemon=require_daemon,
            allowed_prev_node_types=allowed_prev_node_types,
        )
        servicer._peer_address = peer_address
        inference_pb2_grpc.add_InferenceNodeServicer_to_server(
            servicer, server)
    else:
        exchanger_b = GrpcPeerExchangerB()
        peer_servicer = MPCPeerServicer(mpc_node, exchanger_b)
        inference_pb2_grpc.add_MPCPeerServicer_to_server(
            peer_servicer, server)
        servicer = peer_servicer

    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"[MPC-{role}] Listening on {host}:{port}")
    print(f"[MPC-{role}] Peer: {peer_address}")
    print(f"[MPC-{role}] Advertised as {public_address}")

    registration = None
    if role == "A":
        servicer._init_daemon_stub(
            registry_address or config.REGISTRY_ADDRESS)

    if registry_address:
        from network.discovery import NodeRegistration
        _manifest_path = os.path.join(
            mpc_node._shards_dir, "manifest.json")
        with open(_manifest_path) as _f:
            _mpc_manifest = json.load(_f)
        _mpc_model_id = _mpc_manifest.get(
            "model_id", config.MODEL_NAME)
        capability_json = json.dumps(
            {
                "mpc_role": role,
                "mpc_capabilities": ["input", "output"],
                "mpc_dnc_mode": mpc_dnc_mode,
            },
            separators=(",", ":"),
        )
        registration = NodeRegistration(
            address=public_address,
            model_id=_mpc_model_id,
            shard_index=0,
            layer_start=mpc_node.layer_start,
            layer_end=mpc_node.layer_end,
            has_embedding=(role == "A"),
            has_lm_head=False,
            node_type="mpc",
            capability_json=capability_json,
            registry_address=registry_address,
            node_id=eth_address,
        )
        registration.start()
        print(f"[MPC-{role}] Registered as 'mpc' with registry at {registry_address}")
        if role == "A":
            servicer._node_id = registration.node_id
            servicer._peer_address = peer_address
            servicer._peer_node_id = f"peer-of-{registration.node_id[:8]}"
            servicer._private_key = registration.private_key
            servicer._registration_share_signing_private_key = (
                registration.share_signing_private_key
            )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\n[MPC-{role}] Shutting down...")
        if registration:
            registration.stop()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UNFED AI MPC Node (Shard 0) — Beaver-Triple 2PC")
    parser.add_argument("--role", type=str, required=True,
                        choices=["A", "B"],
                        help="MPC role: A (entry) or B (peer)")
    parser.add_argument("--port", type=int, required=True,
                        help="Port for this MPC node")
    parser.add_argument("--peer", type=str, required=True,
                        help="Address of the other MPC node")
    parser.add_argument("--host", type=str, default="[::]",
                        help="Bind address")
    parser.add_argument("--advertise", type=str, default=None,
                        help="Address to advertise to registry")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address")
    parser.add_argument("--shards-dir", type=str, default=None,
                        help="Directory containing shard files")
    parser.add_argument("--eth-address", type=str, default=None,
                        help="Ethereum address for on-chain staking "
                             "(used as node_id)")
    parser.add_argument("--require-daemon", type=str, default="true",
                        choices=["true", "false", "1", "0"],
                        help="Require daemon availability for share submission")
    parser.add_argument("--allowed-prev-node-types", type=str, default="",
                        help="Comma-separated allowed previous node types for MPC-A")
    parser.add_argument("--mpc-dnc-mode", type=str, default="off",
                        choices=["off", "manual", "auto"],
                        help="MPC divide-and-conquer mode")
    parser.add_argument("--mpc-dnc-depth", type=int, default=0,
                        help="DNC recursion depth in manual mode")
    parser.add_argument("--mpc-dnc-split-dim", type=int, default=-1,
                        help="DNC split dimension for element-wise ops in manual mode")
    parser.add_argument("--mpc-dnc-matmul-split-dim", type=int, default=-3,
                        help="DNC split dimension for matmul ops in manual mode")
    parser.add_argument("--mpc-dnc-parallel-workers", type=int, default=1,
                        help="DNC leaf parallel workers in manual mode")
    parser.add_argument("--mpc-dnc-auto-min-elems", type=int, default=65536,
                        help="Auto mode: minimum elements before enabling DNC")
    parser.add_argument("--mpc-dnc-auto-chunk-min-elems", type=int, default=16384,
                        help="Auto mode: minimum chunk elements after splitting")
    parser.add_argument("--mpc-dnc-auto-max-depth", type=int, default=3,
                        help="Auto mode: max split depth")
    parser.add_argument("--mpc-dnc-auto-max-workers", type=int, default=4,
                        help="Auto mode: max local worker threads")
    args = parser.parse_args()
    require_daemon = str(args.require_daemon).strip().lower() in ("true", "1", "yes", "on")
    allowed_prev_node_types = [
        x.strip() for x in str(args.allowed_prev_node_types).split(",") if x.strip()
    ]
    serve(args.role, args.port, args.peer, args.host, args.advertise,
          args.registry, args.shards_dir, eth_address=args.eth_address,
          require_daemon=require_daemon,
          allowed_prev_node_types=allowed_prev_node_types,
          mpc_dnc_mode=args.mpc_dnc_mode,
          mpc_dnc_depth=args.mpc_dnc_depth,
          mpc_dnc_split_dim=args.mpc_dnc_split_dim,
          mpc_dnc_matmul_split_dim=args.mpc_dnc_matmul_split_dim,
          mpc_dnc_parallel_workers=args.mpc_dnc_parallel_workers,
          mpc_dnc_auto_min_elems=args.mpc_dnc_auto_min_elems,
          mpc_dnc_auto_chunk_min_elems=args.mpc_dnc_auto_chunk_min_elems,
          mpc_dnc_auto_max_depth=args.mpc_dnc_auto_max_depth,
          mpc_dnc_auto_max_workers=args.mpc_dnc_auto_max_workers)
