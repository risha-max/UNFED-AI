"""
Shard Downloader — downloads shards from peer nodes (P2P weight distribution).

Supports two modes:
  1. Multi-peer parallel download (when manifest has chunk_hashes):
     - Discovers all peers serving the desired shard
     - Splits the shard into 4 MB chunks (pieces)
     - Downloads chunks from multiple peers in parallel
     - Verifies each chunk's hash independently on arrival
     - Resumes from where it left off on failure
     - Reassigns failed/slow chunks to other peers
  2. Legacy single-peer download (fallback for old manifests without chunk_hashes)

Can also fetch the manifest from the registry if no local copy exists.

Usage:
    python -m shard.downloader --shard-index 2
    python -m shard.downloader --shard-index 0 --manifest shards/manifest.json
    python -m shard.downloader --shard-index 1 --from-registry
"""

import argparse
import base64
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
from network.discovery import RegistryClient
from shard.manifest_signing import (
    decode_public_key_bytes,
    load_trusted_keys_file,
    verify_manifest_signature,
)


# ---------------------------------------------------------------------------
# Progress file for resume support
# ---------------------------------------------------------------------------

def _progress_path(output_path: str) -> str:
    """Return the path to the .progress JSON file for a shard download."""
    return output_path + ".progress"


def _load_progress(output_path: str) -> set[int]:
    """Load the set of verified chunk indices from the progress file."""
    path = _progress_path(output_path)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.get("verified_chunks", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def _save_progress(output_path: str, shard_index: int, verified: set[int], chunk_size: int):
    """Save the set of verified chunk indices to the progress file."""
    path = _progress_path(output_path)
    with open(path, "w") as f:
        json.dump({
            "shard_index": shard_index,
            "verified_chunks": sorted(verified),
            "chunk_size": chunk_size,
        }, f)


def _delete_progress(output_path: str):
    """Remove the progress file after a successful download."""
    path = _progress_path(output_path)
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Persistent peer score cache
# ---------------------------------------------------------------------------

def _peer_score_path() -> str:
    return os.path.expanduser("~/.unfed/peer_scores.json")


def _load_peer_scores() -> dict:
    path = _peer_score_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_peer_scores(scores: dict) -> None:
    path = _peer_score_path()
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


# ---------------------------------------------------------------------------
# MultiPeerDownloader
# ---------------------------------------------------------------------------

class MultiPeerDownloader:
    """Download a shard from multiple peers in parallel, with per-chunk verification.

    Inspired by BitTorrent piece downloading:
    - Splits the shard into fixed-size chunks (pieces)
    - Assigns chunks round-robin across available peers
    - Downloads chunks in parallel using a thread pool
    - Verifies each chunk's SHA256 hash independently
    - Reassigns failed chunks to other peers
    - Tracks progress for resume on interruption
    """

    def __init__(
        self,
        shard_index: int,
        shard_info: dict,
        model_id: str,
        peers: list,
        max_concurrent: int = 4,
        *,
        per_chunk_timeout_s: float = 12.0,
        backoff_base_s: float = 0.75,
        quarantine_seconds: float = 20.0,
        quarantine_error_threshold: int = 3,
        grpc_compression_mode: str = "auto",
    ):
        """
        Args:
            shard_index: Which shard to download.
            shard_info: Manifest entry for this shard (must have chunk_hashes).
            model_id: Model identifier for the GetShard RPC.
            peers: List of peer addresses (strings).
            max_concurrent: Maximum number of parallel chunk downloads.
        """
        self.shard_index = shard_index
        self.model_id = model_id
        self.peers = peers
        self.max_concurrent = max(1, int(max_concurrent))
        self._max_concurrent_cap = max(1, int(max_concurrent))
        self.per_chunk_timeout_s = float(per_chunk_timeout_s)
        self.backoff_base_s = float(backoff_base_s)
        self.quarantine_seconds = float(quarantine_seconds)
        self.quarantine_error_threshold = int(quarantine_error_threshold)
        self.grpc_compression_mode = str(grpc_compression_mode or "auto").strip().lower()

        self.chunk_size: int = shard_info["chunk_size"]
        self.chunk_hashes: list[str] = shard_info["chunk_hashes"]
        self.num_chunks: int = len(self.chunk_hashes)
        self.expected_size: int = shard_info["size_bytes"]
        self.expected_hash: str = shard_info["sha256"]

        # Per-peer stats (thread-safe via lock)
        self._peer_bytes: dict[str, int] = {p: 0 for p in peers}
        self._peer_time: dict[str, float] = {p: 0.0 for p in peers}  # seconds spent
        self._peer_errors: dict[str, int] = {p: 0 for p in peers}
        self._peer_successes: dict[str, int] = {p: 0 for p in peers}
        self._peer_quarantine_until: dict[str, float] = {p: 0.0 for p in peers}
        self._chunk_candidate_peers: dict[int, set[str]] = {
            i: set(peers) for i in range(self.num_chunks)
        }
        self._peer_channels: dict[str, grpc.Channel] = {}
        self._peer_stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._write_queue: queue.Queue = queue.Queue(
            maxsize=max(16, self.max_concurrent * 4)
        )
        self._persistent_scores = _load_peer_scores()
        self._lock = threading.Lock()

    def download(self, output_path: str) -> bool:
        """Download all chunks in parallel, verify each, assemble into file.

        Returns True on success, False on failure.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Determine which chunks are already verified (resume support)
        verified = _load_progress(output_path)
        pending = [i for i in range(self.num_chunks) if i not in verified]

        if not pending:
            print(f"  All {self.num_chunks} chunks already verified. Finalizing.")
            return self._finalize(output_path, verified)

        print(f"  {len(pending)} chunks to download "
              f"({len(verified)} already verified, {self.num_chunks} total)")

        # Pre-allocate the output file if it doesn't exist
        if not os.path.exists(output_path):
            with open(output_path, "wb") as f:
                f.truncate(self.expected_size)

        start_time = time.time()
        max_retries = 3
        self._seed_candidates_from_bitfields()
        writer = self._start_writer(output_path, verified, start_time)
        try:
            for attempt in range(max_retries):
                if not pending:
                    break

                # Assign chunks with rarest-first scheduling (fewest viable peers first).
                assignments: list[tuple[int, str]] = []
                active_peers = self._active_peers_now()
                if not active_peers:
                    print("  ERROR: All peers exhausted (too many errors)")
                    return False

                ranked_peers = self._rank_peers(active_peers)
                ordered_pending = sorted(
                    pending,
                    key=lambda c: len(self._chunk_candidate_peers.get(c, set(active_peers)) & set(active_peers)),
                )
                for i, chunk_idx in enumerate(ordered_pending):
                    candidates = list(self._chunk_candidate_peers.get(chunk_idx, set(active_peers)) & set(active_peers))
                    peer_pool = candidates if candidates else ranked_peers
                    if not peer_pool:
                        continue
                    peer = peer_pool[i % len(peer_pool)]
                    assignments.append((chunk_idx, peer))

                # Download in parallel
                failed_chunks = []
                with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                    futures: dict[Future, tuple[int, str]] = {}
                    for chunk_idx, peer in assignments:
                        fut = executor.submit(self._download_chunk, peer, chunk_idx)
                        futures[fut] = (chunk_idx, peer)

                    for fut in as_completed(futures):
                        chunk_idx, peer = futures[fut]
                        try:
                            data = fut.result()
                            if data is not None and self._verify_chunk(chunk_idx, data):
                                # bounded in-flight write queue
                                self._write_queue.put((chunk_idx, data, peer))
                            else:
                                failed_chunks.append(chunk_idx)
                                self._record_peer_failure(peer, chunk_idx)
                        except Exception:
                            failed_chunks.append(chunk_idx)
                            self._record_peer_failure(peer, chunk_idx)

                # Ensure queued writes are flushed before retry planning.
                self._write_queue.join()
                pending = failed_chunks
                self._adapt_concurrency(total=len(assignments), failed=len(failed_chunks))
                if pending:
                    print(f"\n  Retry {attempt + 1}/{max_retries}: "
                          f"{len(pending)} chunk(s) failed, reassigning...")
                    backoff_s = self.backoff_base_s * (2 ** attempt)
                    print(f"  Backoff: sleeping {backoff_s:.2f}s")
                    time.sleep(backoff_s)

            if pending:
                print(f"\n  ERROR: {len(pending)} chunk(s) could not be downloaded "
                      f"after {max_retries} retries")
                return False

            ok = self._finalize(output_path, verified)
            self._persist_peer_scores()
            return ok
        finally:
            self._stop_writer(writer)
            self._close_peer_clients()

    def _download_chunk(self, peer: str, chunk_idx: int) -> bytes | None:
        """Download a single chunk from a peer via GetShard with range params.

        Also records the elapsed time for bandwidth-aware peer ranking.
        """
        offset = chunk_idx * self.chunk_size
        # Last chunk may be shorter
        remaining = self.expected_size - offset
        length = min(self.chunk_size, remaining)

        t0 = time.time()
        try:
            stub = self._get_peer_stub(peer)

            request = inference_pb2.GetShardRequest(
                model_id=self.model_id,
                shard_index=self.shard_index,
                offset=offset,
                length=length,
            )

            data = b""
            compression = self._select_grpc_compression()
            for chunk in stub.GetShard(
                request,
                timeout=self.per_chunk_timeout_s,
                compression=compression,
            ):
                data += chunk.data

            elapsed = time.time() - t0
            with self._lock:
                self._peer_time[peer] = self._peer_time.get(peer, 0.0) + elapsed
            return data if data else None

        except grpc.RpcError:
            return None
        except Exception:
            return None

    def _get_peer_stub(self, peer: str) -> inference_pb2_grpc.InferenceNodeStub:
        with self._lock:
            stub = self._peer_stubs.get(peer)
            if stub is not None:
                return stub
            channel = grpc.insecure_channel(peer, options=config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
            self._peer_channels[peer] = channel
            self._peer_stubs[peer] = stub
            return stub

    def _close_peer_clients(self) -> None:
        with self._lock:
            channels = list(self._peer_channels.values())
            self._peer_channels = {}
            self._peer_stubs = {}
        for ch in channels:
            try:
                ch.close()
            except Exception:
                pass

    def _start_writer(self, output_path: str, verified: set[int], start_time: float) -> threading.Thread:
        t = threading.Thread(
            target=self._writer_loop,
            args=(output_path, verified, start_time),
            daemon=True,
        )
        t.start()
        return t

    def _stop_writer(self, writer: threading.Thread) -> None:
        try:
            self._write_queue.put((None, None, None))
        except Exception:
            pass
        if writer.is_alive():
            writer.join(timeout=5)

    def _writer_loop(self, output_path: str, verified: set[int], start_time: float) -> None:
        with open(output_path, "r+b") as f:
            while True:
                item = self._write_queue.get()
                try:
                    chunk_idx, data, peer = item
                    if chunk_idx is None:
                        break
                    offset = int(chunk_idx) * self.chunk_size
                    f.seek(offset)
                    f.write(data)
                    with self._lock:
                        verified.add(int(chunk_idx))
                        _save_progress(output_path, self.shard_index, verified, self.chunk_size)
                        self._peer_bytes[str(peer)] = self._peer_bytes.get(str(peer), 0) + len(data)
                        self._peer_successes[str(peer)] = self._peer_successes.get(str(peer), 0) + 1
                    self._print_progress(verified, start_time)
                finally:
                    self._write_queue.task_done()

    def _seed_candidates_from_bitfields(self) -> None:
        """Optionally refine per-chunk peer candidates using peer bitfields."""
        for peer in self.peers:
            availability = self._fetch_peer_chunk_availability(peer)
            if availability is None:
                continue
            for idx, available in enumerate(availability):
                if not available:
                    self._chunk_candidate_peers.setdefault(idx, set(self.peers)).discard(peer)

    def _fetch_peer_chunk_availability(self, peer: str) -> list[bool] | None:
        """Fetch chunk availability bitfield from peer.

        Returns:
            - list[bool] of length num_chunks on success
            - None if peer does not support bitfield endpoint
        """
        try:
            stub = self._get_peer_stub(peer)
            req = inference_pb2.GetShardBitfieldRequest(
                model_id=self.model_id,
                shard_index=self.shard_index,
                chunk_size=self.chunk_size,
            )
            resp = stub.GetShardBitfield(req, timeout=max(3.0, self.per_chunk_timeout_s / 2.0))
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                return None
            return None
        except Exception:
            return None

        if not bool(getattr(resp, "supported", False)):
            return None
        bitfield = bytes(getattr(resp, "bitfield", b"") or b"")
        if not bitfield:
            return None
        chunk_count = int(getattr(resp, "chunk_count", 0) or 0)
        count = min(self.num_chunks, chunk_count if chunk_count > 0 else self.num_chunks)
        out = [True] * self.num_chunks
        for i in range(count):
            byte_i = i // 8
            bit_i = i % 8
            present = byte_i < len(bitfield) and bool(bitfield[byte_i] & (1 << bit_i))
            out[i] = present
        return out

    def _verify_chunk(self, chunk_idx: int, data: bytes) -> bool:
        """Verify a chunk's SHA256 hash against the manifest."""
        actual = hashlib.sha256(data).hexdigest()
        expected = self.chunk_hashes[chunk_idx]
        if actual != expected:
            print(f"\n  Chunk {chunk_idx} hash mismatch: "
                  f"expected {expected[:12]}..., got {actual[:12]}...")
            return False
        return True

    def _rank_peers(self, active_peers: list[str]) -> list[str]:
        """Rank peers by observed download speed (bytes/sec), fastest first.

        On the first pass (no stats yet), returns the original order.
        On retries, peers that downloaded faster get more chunk assignments
        because they appear earlier (and more often) in the round-robin.
        """
        with self._lock:
            current_speeds = {}
            for p in active_peers:
                b = self._peer_bytes.get(p, 0)
                t = self._peer_time.get(p, 0.0)
                current_speeds[p] = b / t if t > 0 else 0.0

        def _score(peer: str) -> float:
            cur_speed = current_speeds.get(peer, 0.0)
            hist = self._persistent_scores.get(peer, {})
            hist_speed = float(hist.get("bytes", 0)) / max(float(hist.get("seconds", 0.0)), 1e-9)
            successes = float(hist.get("successes", 0))
            failures = float(hist.get("failures", 0))
            reliability = (successes + 1.0) / (successes + failures + 2.0)
            return (0.65 * cur_speed) + (0.35 * hist_speed * reliability)

        ranked = sorted(active_peers, key=_score, reverse=True)
        return ranked if ranked else list(active_peers)

    def _finalize(self, output_path: str, verified: set[int]) -> bool:
        """Verify the full-file hash and clean up."""
        if len(verified) < self.num_chunks:
            print(f"  ERROR: Only {len(verified)}/{self.num_chunks} chunks verified")
            return False

        # Final full-file hash check
        actual_hash = _compute_file_hash(output_path)
        if actual_hash != self.expected_hash:
            print(f"  FULL-FILE HASH MISMATCH!")
            print(f"    Expected: {self.expected_hash[:16]}...")
            print(f"    Got:      {actual_hash[:16]}...")
            return False

        _delete_progress(output_path)
        print(f"\n  Full-file hash verified: {actual_hash[:16]}...")

        # Print per-peer stats with speed
        total_bytes = sum(self._peer_bytes.values())
        if total_bytes > 0:
            print(f"  Per-peer contribution:")
            for peer in self.peers:
                pb = self._peer_bytes.get(peer, 0)
                pt = self._peer_time.get(peer, 0.0)
                pct = pb / total_bytes * 100 if total_bytes else 0
                speed = pb / (1024 * 1024) / pt if pt > 0 else 0
                print(f"    {peer}: {pb / 1024 / 1024:.1f} MB "
                      f"({pct:.0f}%) @ {speed:.1f} MB/s")

        return True

    def _print_progress(self, verified: set[int], start_time: float):
        """Print download progress."""
        done = len(verified)
        total = self.num_chunks
        pct = done / total * 100
        elapsed = time.time() - start_time
        total_bytes = sum(self._peer_bytes.values())
        speed = total_bytes / (1024 * 1024) / elapsed if elapsed > 0 else 0
        mb_done = total_bytes / (1024 * 1024)
        mb_total = self.expected_size / (1024 * 1024)
        print(f"\r  [{done}/{total}] {mb_done:.1f}/{mb_total:.1f} MB "
              f"({pct:.0f}%) - {speed:.1f} MB/s",
              end="", flush=True)

    def _active_peers_now(self) -> list[str]:
        now = time.time()
        return [
            p for p in self.peers
            if self._peer_errors.get(p, 0) < 8
            and now >= float(self._peer_quarantine_until.get(p, 0.0))
        ]

    def _record_peer_failure(self, peer: str, chunk_idx: int) -> None:
        with self._lock:
            self._peer_errors[peer] = self._peer_errors.get(peer, 0) + 1
            self._chunk_candidate_peers.setdefault(chunk_idx, set(self.peers)).discard(peer)
            if self._peer_errors[peer] >= self.quarantine_error_threshold:
                self._peer_quarantine_until[peer] = time.time() + self.quarantine_seconds

    def _adapt_concurrency(self, *, total: int, failed: int) -> None:
        if total <= 0:
            return
        fail_rate = failed / total
        if fail_rate >= 0.30:
            self.max_concurrent = max(1, self.max_concurrent - 1)
        elif fail_rate <= 0.05:
            self.max_concurrent = min(self._max_concurrent_cap, self.max_concurrent + 1)

    def _select_grpc_compression(self):
        mode = self.grpc_compression_mode
        if mode in ("off", "none"):
            return None
        if mode in ("on", "gzip"):
            return grpc.Compression.Gzip
        # auto mode: enable compression on slower links.
        total_bytes = sum(self._peer_bytes.values())
        total_secs = sum(self._peer_time.values())
        if total_secs <= 0:
            return None
        avg_mbps = (total_bytes / (1024 * 1024)) / total_secs
        return grpc.Compression.Gzip if avg_mbps < 8.0 else None

    def _persist_peer_scores(self) -> None:
        scores = dict(self._persistent_scores)
        now = int(time.time())
        for peer in self.peers:
            current = scores.get(peer, {})
            scores[peer] = {
                "bytes": int(current.get("bytes", 0)) + int(self._peer_bytes.get(peer, 0)),
                "seconds": float(current.get("seconds", 0.0)) + float(self._peer_time.get(peer, 0.0)),
                "successes": int(current.get("successes", 0)) + int(self._peer_successes.get(peer, 0)),
                "failures": int(current.get("failures", 0)) + int(self._peer_errors.get(peer, 0)),
                "updated_at": now,
            }
        _save_peer_scores(scores)
        self._persistent_scores = scores


# ---------------------------------------------------------------------------
# Manifest fetching from registry
# ---------------------------------------------------------------------------

def _resolve_trusted_manifest_key_bytes(
    *,
    trusted_key_hex: str = "",
    trusted_key_file: str = "",
) -> bytes | None:
    raw = ""
    if trusted_key_hex:
        raw = trusted_key_hex.strip()
    elif trusted_key_file:
        with open(os.path.expanduser(trusted_key_file), "r", encoding="utf-8") as f:
            raw = f.read().strip()
    if not raw:
        return None
    if raw.startswith("0x"):
        key = bytes.fromhex(raw[2:])
    else:
        key = base64.b64decode(raw)
    if len(key) != 32:
        raise ValueError("trusted manifest public key must be 32 bytes")
    return key


def _resolve_trusted_manifest_key_set(
    *,
    trusted_keys_file: str = "",
    trusted_keys_json: str = "",
) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if trusted_keys_file:
        out.update(load_trusted_keys_file(trusted_keys_file))
    if trusted_keys_json:
        data = json.loads(trusted_keys_json)
        if not isinstance(data, dict):
            raise ValueError("trusted_manifest_keys_json must be a JSON object")
        for key_id, raw in data.items():
            kid = str(key_id).strip()
            if not kid:
                continue
            out[kid] = decode_public_key_bytes(str(raw))
    return out


def _validate_manifest_signature(
    manifest: dict,
    *,
    require_signed: bool,
    trusted_key_bytes: bytes | None,
    trusted_key_id: str,
    trusted_keys_by_id: dict[str, bytes] | None,
    active_key_ids: list[str] | None,
    max_age_seconds: int,
) -> bool:
    has_sig = isinstance(manifest.get("manifest_signature"), dict)
    if not has_sig:
        if require_signed:
            print("ERROR: manifest signature required but missing")
            return False
        print("WARNING: manifest has no signature metadata (manifest_signature missing).")
        return True
    ok, msg = verify_manifest_signature(
        manifest,
        trusted_public_key_bytes=trusted_key_bytes,
        trusted_key_id=trusted_key_id,
        trusted_keys_by_id=trusted_keys_by_id,
        active_key_ids=active_key_ids,
        max_age_seconds=max_age_seconds,
    )
    if not ok:
        print(f"ERROR: manifest signature check failed: {msg}")
        return False
    print("Manifest signature verified.")
    return True

def fetch_manifest_from_registry(model_id: str,
                                 registry_address: str = None) -> dict | None:
    """Fetch a model manifest from the registry via GetManifest RPC.

    Returns the parsed manifest dict, or None if not found.
    """
    import registry_pb2
    import registry_pb2_grpc

    addr = registry_address or config.REGISTRY_ADDRESS
    try:
        channel = grpc.insecure_channel(addr)
        stub = registry_pb2_grpc.RegistryStub(channel)
        resp = stub.GetManifest(registry_pb2.GetManifestRequest(model_id=model_id))
        channel.close()
        if resp.found:
            return json.loads(resp.manifest_json)
        return None
    except grpc.RpcError:
        return None


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------

def download_shard(
    shard_index: int,
    manifest_path: str = None,
    registry_address: str = None,
    from_registry: bool = False,
    *,
    require_signed_manifest: bool = False,
    trusted_manifest_key_hex: str = "",
    trusted_manifest_key_file: str = "",
    trusted_manifest_key_id: str = "",
    trusted_manifest_keys_file: str = "",
    trusted_manifest_keys_json: str = "",
    trusted_manifest_active_key_ids: str = "",
    manifest_max_age_seconds: int = 0,
    max_concurrent: int | None = None,
    per_chunk_timeout_s: float = 12.0,
    backoff_base_s: float = 0.75,
    quarantine_seconds: float = 20.0,
    quarantine_error_threshold: int = 3,
    grpc_compression_mode: str = "auto",
    model_id_for_registry_fetch: str = "",
):
    """Download a shard from peer nodes.

    If `from_registry` is True and no local manifest exists, fetches
    the manifest from the registry first.
    """
    manifest_path = manifest_path or config.MANIFEST_PATH

    # Try loading local manifest
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    elif from_registry:
        print(f"No local manifest found. Fetching from registry...")
        requested_model_id = str(model_id_for_registry_fetch or config.MODEL_NAME)
        manifest = fetch_manifest_from_registry(
            requested_model_id, registry_address)
        if manifest:
            # Save locally for future use
            os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest saved to {manifest_path}")
        else:
            print(
                "ERROR: Could not fetch manifest from registry "
                f"for model_id='{requested_model_id}'."
            )
            return False

    if not manifest:
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("You need a manifest to verify shard integrity.")
        print("Options: run the splitter, get from another node, "
              "or use --from-registry.")
        return False

    trusted_key = _resolve_trusted_manifest_key_bytes(
        trusted_key_hex=trusted_manifest_key_hex,
        trusted_key_file=trusted_manifest_key_file,
    )
    trusted_key_set = _resolve_trusted_manifest_key_set(
        trusted_keys_file=trusted_manifest_keys_file,
        trusted_keys_json=trusted_manifest_keys_json,
    )
    active_key_ids = [
        x.strip()
        for x in str(trusted_manifest_active_key_ids or "").split(",")
        if x.strip()
    ]
    if not _validate_manifest_signature(
        manifest,
        require_signed=require_signed_manifest,
        trusted_key_bytes=trusted_key,
        trusted_key_id=trusted_manifest_key_id,
        trusted_keys_by_id=trusted_key_set or None,
        active_key_ids=active_key_ids or None,
        max_age_seconds=int(manifest_max_age_seconds or 0),
    ):
        return False

    if shard_index >= len(manifest["shards"]):
        print(f"ERROR: Shard index {shard_index} out of range "
              f"(max: {len(manifest['shards']) - 1})")
        return False

    shard_info = manifest["shards"][shard_index]
    expected_hash = shard_info["sha256"]
    output_path = config.get_shard_path(shard_index)

    # Check if shard already exists and is valid
    if os.path.exists(output_path):
        existing_hash = _compute_file_hash(output_path)
        if existing_hash == expected_hash:
            print(f"Shard {shard_index} already exists and hash matches. "
                  f"Skipping download.")
            return True
        else:
            print(f"Shard {shard_index} exists but hash mismatch — re-downloading.")

    # Discover peers that serve this shard
    discovery = RegistryClient(registry_address)
    nodes = discovery.discover(manifest["model_id"])
    peers = [n for n in nodes if n.shard_index == shard_index]

    if not peers:
        print(f"ERROR: No peers found serving shard {shard_index}")
        print("Ensure at least one node with this shard is running and registered.")
        discovery.close()
        return False

    peer_addresses = [p.address for p in peers]
    print(f"Found {len(peers)} peer(s) with shard {shard_index}")

    # Choose download strategy based on manifest capabilities
    has_chunks = ("chunk_hashes" in shard_info and
                  len(shard_info.get("chunk_hashes", [])) > 0)

    if has_chunks and len(peer_addresses) >= 1:
        # Multi-peer parallel download with chunk verification
        print(f"Using multi-peer download ({len(peer_addresses)} peer(s), "
              f"{len(shard_info['chunk_hashes'])} chunks)")
        downloader = MultiPeerDownloader(
            shard_index=shard_index,
            shard_info=shard_info,
            model_id=manifest["model_id"],
            peers=peer_addresses,
            max_concurrent=(max_concurrent or min(4, len(peer_addresses) * 2)),
            per_chunk_timeout_s=per_chunk_timeout_s,
            backoff_base_s=backoff_base_s,
            quarantine_seconds=quarantine_seconds,
            quarantine_error_threshold=quarantine_error_threshold,
            grpc_compression_mode=grpc_compression_mode,
        )
        success = downloader.download(output_path)
    else:
        # Legacy single-peer download
        print(f"Using legacy single-peer download (no chunk_hashes in manifest)")
        success = False
        for peer in peer_addresses:
            print(f"Downloading shard {shard_index} from {peer}...")
            success = _download_from_peer_legacy(
                peer, shard_index, manifest["model_id"],
                output_path, expected_hash, shard_info["size_bytes"],
            )
            if success:
                break
            print(f"  Failed from {peer}, trying next peer...")

    discovery.close()

    if not success:
        print("ERROR: Could not download shard from any peer.")
    return success


def _download_from_peer_legacy(address: str, shard_index: int, model_id: str,
                               output_path: str, expected_hash: str,
                               expected_size: int) -> bool:
    """Legacy: download a full shard from a single peer (no chunk verification)."""
    try:
        channel = grpc.insecure_channel(address, options=config.GRPC_OPTIONS)
        stub = inference_pb2_grpc.InferenceNodeStub(channel)

        request = inference_pb2.GetShardRequest(
            model_id=model_id,
            shard_index=shard_index,
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        start_time = time.time()
        received = 0

        with open(output_path, "wb") as f:
            for chunk in stub.GetShard(request):
                f.write(chunk.data)
                received += len(chunk.data)

                if chunk.total_size > 0:
                    pct = received / chunk.total_size * 100
                    mb = received / (1024 * 1024)
                    total_mb = chunk.total_size / (1024 * 1024)
                    elapsed = time.time() - start_time
                    speed = mb / elapsed if elapsed > 0 else 0
                    print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%) "
                          f"- {speed:.1f} MB/s", end="", flush=True)

        print()
        elapsed = time.time() - start_time
        speed = received / (1024 * 1024) / elapsed if elapsed > 0 else 0
        print(f"  Downloaded {received / (1024 * 1024):.1f} MB "
              f"in {elapsed:.1f}s ({speed:.1f} MB/s)")

        local_hash = _compute_file_hash(output_path)
        if local_hash != expected_hash:
            print(f"  HASH MISMATCH!")
            print(f"    Expected: {expected_hash[:16]}...")
            print(f"    Got:      {local_hash[:16]}...")
            os.remove(output_path)
            return False

        print(f"  Hash verified: {local_hash[:16]}...")
        channel.close()
        return True

    except grpc.RpcError as e:
        print(f"\n  gRPC error: {e.details()}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Shard Downloader")
    parser.add_argument("--shard-index", type=int, required=True,
                        help="Index of the shard to download")
    parser.add_argument("--manifest", type=str, default=None,
                        help=f"Path to manifest.json (default: {config.MANIFEST_PATH})")
    parser.add_argument("--registry", type=str, default=None,
                        help=f"Registry address (default: {config.REGISTRY_ADDRESS})")
    parser.add_argument("--from-registry", action="store_true",
                        help="Fetch manifest from registry if not found locally")
    parser.add_argument("--model-id", type=str, default="",
                        help="Model ID to use when fetching manifest from registry")
    parser.add_argument("--require-signed-manifest", action="store_true",
                        help="Fail if manifest signature metadata is missing or invalid")
    parser.add_argument("--trusted-manifest-key-hex", type=str, default="",
                        help="Trusted Ed25519 public key for manifest verification (hex/base64)")
    parser.add_argument("--trusted-manifest-key-file", type=str, default="",
                        help="Path to trusted Ed25519 public key file (hex/base64)")
    parser.add_argument("--trusted-manifest-key-id", type=str, default="",
                        help="Expected manifest signer key_id")
    parser.add_argument("--trusted-manifest-keys-file", type=str, default="",
                        help="JSON file of trusted manifest public keys by key_id")
    parser.add_argument("--trusted-manifest-keys-json", type=str, default="",
                        help="Inline JSON object of trusted key_id->public_key")
    parser.add_argument("--trusted-manifest-active-key-ids", type=str, default="",
                        help="Comma-separated active key IDs allowed for rotation")
    parser.add_argument("--manifest-max-age-seconds", type=int, default=0,
                        help="Reject signatures older than this many seconds (0=disabled)")
    parser.add_argument("--max-concurrent", type=int, default=0,
                        help="Override max parallel chunk downloads")
    parser.add_argument("--per-chunk-timeout", type=float, default=12.0,
                        help="Per-chunk gRPC timeout in seconds")
    parser.add_argument("--backoff-base", type=float, default=0.75,
                        help="Base exponential backoff seconds between retries")
    parser.add_argument("--quarantine-seconds", type=float, default=20.0,
                        help="How long to quarantine flaky peers")
    parser.add_argument("--quarantine-threshold", type=int, default=3,
                        help="Peer error threshold before temporary quarantine")
    parser.add_argument("--grpc-compression", type=str, default="auto",
                        choices=["auto", "on", "off", "gzip", "none"],
                        help="Shard transfer compression mode")
    args = parser.parse_args()

    success = download_shard(
        args.shard_index,
        args.manifest,
        args.registry,
        args.from_registry,
        require_signed_manifest=bool(args.require_signed_manifest),
        trusted_manifest_key_hex=args.trusted_manifest_key_hex,
        trusted_manifest_key_file=args.trusted_manifest_key_file,
        trusted_manifest_key_id=args.trusted_manifest_key_id,
        trusted_manifest_keys_file=args.trusted_manifest_keys_file,
        trusted_manifest_keys_json=args.trusted_manifest_keys_json,
        trusted_manifest_active_key_ids=args.trusted_manifest_active_key_ids,
        manifest_max_age_seconds=args.manifest_max_age_seconds,
        max_concurrent=(args.max_concurrent or None),
        per_chunk_timeout_s=args.per_chunk_timeout,
        backoff_base_s=args.backoff_base,
        quarantine_seconds=args.quarantine_seconds,
        quarantine_error_threshold=args.quarantine_threshold,
        grpc_compression_mode=args.grpc_compression,
        model_id_for_registry_fetch=args.model_id,
    )
    sys.exit(0 if success else 1)
