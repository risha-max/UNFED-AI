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
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

import grpc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
from network.discovery import RegistryClient


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

    def __init__(self, shard_index: int, shard_info: dict, model_id: str,
                 peers: list, max_concurrent: int = 4):
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
        self.max_concurrent = max_concurrent

        self.chunk_size: int = shard_info["chunk_size"]
        self.chunk_hashes: list[str] = shard_info["chunk_hashes"]
        self.num_chunks: int = len(self.chunk_hashes)
        self.expected_size: int = shard_info["size_bytes"]
        self.expected_hash: str = shard_info["sha256"]

        # Per-peer stats (thread-safe via lock)
        self._peer_bytes: dict[str, int] = {p: 0 for p in peers}
        self._peer_time: dict[str, float] = {p: 0.0 for p in peers}  # seconds spent
        self._peer_errors: dict[str, int] = {p: 0 for p in peers}
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

        for attempt in range(max_retries):
            if not pending:
                break

            # Assign chunks to peers (speed-weighted on retries, round-robin initially)
            assignments: list[tuple[int, str]] = []
            active_peers = [p for p in self.peers if self._peer_errors.get(p, 0) < 5]
            if not active_peers:
                print("  ERROR: All peers exhausted (too many errors)")
                return False

            ranked_peers = self._rank_peers(active_peers)
            for i, chunk_idx in enumerate(pending):
                peer = ranked_peers[i % len(ranked_peers)]
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
                            # Write chunk to the correct offset in the file
                            offset = chunk_idx * self.chunk_size
                            with self._lock:
                                with open(output_path, "r+b") as f:
                                    f.seek(offset)
                                    f.write(data)
                                verified.add(chunk_idx)
                                _save_progress(output_path, self.shard_index,
                                               verified, self.chunk_size)
                                self._peer_bytes[peer] = (
                                    self._peer_bytes.get(peer, 0) + len(data))
                            self._print_progress(verified, start_time)
                        else:
                            failed_chunks.append(chunk_idx)
                            with self._lock:
                                self._peer_errors[peer] = (
                                    self._peer_errors.get(peer, 0) + 1)
                    except Exception as e:
                        failed_chunks.append(chunk_idx)
                        with self._lock:
                            self._peer_errors[peer] = (
                                self._peer_errors.get(peer, 0) + 1)

            pending = failed_chunks
            if pending:
                print(f"\n  Retry {attempt + 1}/{max_retries}: "
                      f"{len(pending)} chunk(s) failed, reassigning...")

        if pending:
            print(f"\n  ERROR: {len(pending)} chunk(s) could not be downloaded "
                  f"after {max_retries} retries")
            return False

        return self._finalize(output_path, verified)

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
            channel = grpc.insecure_channel(peer, options=config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)

            request = inference_pb2.GetShardRequest(
                model_id=self.model_id,
                shard_index=self.shard_index,
                offset=offset,
                length=length,
            )

            data = b""
            for chunk in stub.GetShard(request):
                data += chunk.data

            channel.close()
            elapsed = time.time() - t0
            with self._lock:
                self._peer_time[peer] = self._peer_time.get(peer, 0.0) + elapsed
            return data if data else None

        except grpc.RpcError:
            return None
        except Exception:
            return None

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
            speeds = {}
            for p in active_peers:
                b = self._peer_bytes.get(p, 0)
                t = self._peer_time.get(p, 0.0)
                speeds[p] = b / t if t > 0 else 0.0

        # If no peer has stats yet, return original order
        if all(s == 0.0 for s in speeds.values()):
            return list(active_peers)

        # Sort fastest first
        ranked = sorted(active_peers, key=lambda p: speeds[p], reverse=True)
        return ranked

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


# ---------------------------------------------------------------------------
# Manifest fetching from registry
# ---------------------------------------------------------------------------

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

def download_shard(shard_index: int, manifest_path: str = None,
                   registry_address: str = None, from_registry: bool = False):
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
        manifest = fetch_manifest_from_registry(
            config.MODEL_NAME, registry_address)
        if manifest:
            # Save locally for future use
            os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Manifest saved to {manifest_path}")
        else:
            print("ERROR: Could not fetch manifest from registry.")
            return False

    if not manifest:
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("You need a manifest to verify shard integrity.")
        print("Options: run the splitter, get from another node, "
              "or use --from-registry.")
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
            max_concurrent=min(4, len(peer_addresses) * 2),
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
    args = parser.parse_args()

    success = download_shard(args.shard_index, args.manifest, args.registry,
                             args.from_registry)
    sys.exit(0 if success else 1)
