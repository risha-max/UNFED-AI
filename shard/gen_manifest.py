"""
Generate a manifest.json from existing shard files.
Useful if shards were created before the manifest feature was added,
or to regenerate with chunk hashes for multi-peer P2P downloads.

Usage:
    python -m shard.gen_manifest
    python -m shard.gen_manifest --num-shards 4 --layers-per-shard 6
    python -m shard.gen_manifest --registry localhost:50050
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_file_hash(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_chunk_hashes(path: str, chunk_size: int) -> list[str]:
    """Compute per-chunk SHA256 hashes for multi-peer P2P verification."""
    hashes = []
    with open(path, "rb") as f:
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            hashes.append(hashlib.sha256(piece).hexdigest())
    return hashes


def gen_manifest(num_shards: int, layers_per_shard: int) -> dict | None:
    num_layers = num_shards * layers_per_shard

    manifest = {
        "model_id": config.MODEL_NAME,
        "num_layers": num_layers,
        "num_shards": num_shards,
        "layers_per_shard": layers_per_shard,
        "shards": [],
    }

    for i in range(num_shards):
        start = i * layers_per_shard
        end = start + layers_per_shard
        path = config.get_shard_path(i)

        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            return None

        size = os.path.getsize(path)
        h = compute_file_hash(path)
        chunks = compute_chunk_hashes(path, config.P2P_PIECE_SIZE)

        manifest["shards"].append({
            "shard_index": i,
            "layer_start": start,
            "layer_end": end,
            "has_embedding": i == 0,
            "has_lm_head": i == num_shards - 1,
            "file": os.path.basename(path),
            "size_bytes": size,
            "sha256": h,
            "chunk_size": config.P2P_PIECE_SIZE,
            "chunk_hashes": chunks,
        })

        print(f"Shard {i}: layers {start}-{end - 1}, {size / 1024 / 1024:.1f} MB, "
              f"hash={h[:16]}... ({len(chunks)} chunks)")

    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {config.MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate manifest.json with chunk hashes for P2P downloads")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--layers-per-shard", type=int, default=6)
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address to publish manifest to (e.g. localhost:50050)")
    args = parser.parse_args()
    manifest = gen_manifest(args.num_shards, args.layers_per_shard)
    if args.registry and manifest:
        from shard.splitter import publish_manifest_to_registry
        publish_manifest_to_registry(manifest, args.registry)
