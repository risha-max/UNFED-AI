"""
Model Splitter — splits a Hugging Face transformer model into N layer shards.

Each shard contains:
  - Shard 0: embedding layer + layers 0..layers_per_shard-1
  - Shard 1..N-2: layers only (middle shards)
  - Shard N-1: last layers + final LayerNorm + LM head

Also generates a manifest.json describing all shards (hashes, sizes, layer ranges).

Usage:
    python -m shard.splitter
    python -m shard.splitter --num-shards 8
"""

import argparse
import hashlib
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_chunk_hashes(path: str, chunk_size: int) -> list[str]:
    """Compute per-chunk SHA256 hashes for a file (BitTorrent-style piece hashes).

    Each chunk is `chunk_size` bytes (last chunk may be shorter).
    Returns a list of hex-encoded SHA256 hashes, one per chunk.
    """
    hashes = []
    with open(path, "rb") as f:
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            hashes.append(hashlib.sha256(piece).hexdigest())
    return hashes


def split_model(num_shards: int = 4):
    print(f"Loading model: {config.MODEL_NAME}")

    # Load model config first to verify layer count
    model_config = AutoConfig.from_pretrained(config.MODEL_NAME)
    num_layers = model_config.num_hidden_layers
    print(f"Model has {num_layers} hidden layers")

    if num_layers % num_shards != 0:
        print(f"ERROR: {num_layers} layers not evenly divisible by {num_shards} shards")
        print(f"Try --num-shards with a divisor of {num_layers}")
        return

    layers_per_shard = num_layers // num_shards
    print(f"Will split into {num_shards} shards of {layers_per_shard} layers each")

    # Load the full model
    print("Loading full model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        dtype=torch.float32,  # Use float32 for CPU compatibility
    )
    model.eval()

    # Create shards directory
    os.makedirs(config.SHARDS_DIR, exist_ok=True)

    manifest = {
        "model_id": config.MODEL_NAME,
        "num_layers": num_layers,
        "num_shards": num_shards,
        "layers_per_shard": layers_per_shard,
        "shards": [],
    }

    for shard_idx in range(num_shards):
        start_layer = shard_idx * layers_per_shard
        end_layer = start_layer + layers_per_shard
        shard_path = config.get_shard_path(shard_idx)
        is_first = shard_idx == 0
        is_last = shard_idx == num_shards - 1

        print(f"\nCreating shard {shard_idx}: layers {start_layer}-{end_layer - 1}")

        shard = {}

        # First shard gets the embedding layer
        if is_first:
            shard["embed_tokens"] = model.model.embed_tokens.state_dict()
            print(f"  + embed_tokens")

        # All shards get their assigned layers
        for layer_idx in range(start_layer, end_layer):
            shard[f"layer_{layer_idx}"] = model.model.layers[layer_idx].state_dict()
        print(f"  + layers {start_layer}-{end_layer - 1}")

        # Last shard gets the final norm and LM head
        if is_last:
            shard["norm"] = model.model.norm.state_dict()
            shard["lm_head"] = model.lm_head.state_dict()
            print(f"  + norm + lm_head")

        # Save shard
        torch.save(shard, shard_path)
        shard_size = os.path.getsize(shard_path)
        shard_hash = compute_file_hash(shard_path)
        chunk_hashes = compute_chunk_hashes(shard_path, config.P2P_PIECE_SIZE)
        print(f"  Saved to {shard_path} ({shard_size / (1024 * 1024):.1f} MB)")
        print(f"  SHA256: {shard_hash[:16]}... ({len(chunk_hashes)} chunks)")

        manifest["shards"].append({
            "shard_index": shard_idx,
            "layer_start": start_layer,
            "layer_end": end_layer,
            "has_embedding": is_first,
            "has_lm_head": is_last,
            "file": os.path.basename(shard_path),
            "size_bytes": shard_size,
            "sha256": shard_hash,
            "chunk_size": config.P2P_PIECE_SIZE,
            "chunk_hashes": chunk_hashes,
        })

    # Save model config so nodes can reconstruct layer architecture
    model_config.save_pretrained(config.SHARDS_DIR)
    print(f"\nModel config saved to {config.SHARDS_DIR}/config.json")

    # Save manifest
    manifest_path = config.MANIFEST_PATH
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    print("\nDone! All shards created successfully.")
    print(f"\nTo start nodes, run:")
    for i in range(num_shards):
        port = 50051 + i
        print(f"  python -m node.server --shard-index {i} --port {port}")

    return manifest


def publish_manifest_to_registry(manifest: dict, registry_address: str):
    """Publish the manifest to the registry via PutManifest RPC."""
    import grpc
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
    import registry_pb2
    import registry_pb2_grpc

    channel = grpc.insecure_channel(registry_address)
    stub = registry_pb2_grpc.RegistryStub(channel)
    manifest_json = json.dumps(manifest)

    try:
        resp = stub.PutManifest(registry_pb2.PutManifestRequest(
            model_id=manifest["model_id"],
            manifest_json=manifest_json,
        ))
        if resp.success:
            print(f"Manifest published to registry at {registry_address}")
        else:
            print(f"Registry rejected manifest: {resp.message}")
    except grpc.RpcError as e:
        print(f"Failed to publish manifest to registry: {e.details()}")
    finally:
        channel.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Model Splitter")
    parser.add_argument("--num-shards", type=int, default=4,
                        help="Number of shards to split the model into")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address to publish manifest to (e.g. localhost:50050)")
    args = parser.parse_args()
    manifest = split_model(args.num_shards)
    if args.registry and manifest:
        publish_manifest_to_registry(manifest, args.registry)
