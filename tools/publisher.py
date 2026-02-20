"""
UNFED Shard Publisher — upload manifest to an UNFED registry.

The registry stores the manifest (metadata) so nodes can discover the model.
Shard files are NOT uploaded to the registry — they're served peer-to-peer
by nodes that already have them (see shard/downloader.py).

Flow:
  1. Pool creator runs `unfed split` → produces shards + manifest locally
  2. Pool creator runs `unfed publish` → pushes manifest to registry via gRPC
  3. Pool creator starts a seed node → serves shard files for P2P download
  4. Other nodes join → download shards via P2P, register with registry

Usage:
    python -m tools publish ./shards --registry localhost:50050
    python -m tools publish ./shards --registry localhost:50050 --pool-name my-model
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def publish_to_registry(
    manifest: dict,
    registry_address: str,
    pool_name: str | None = None,
) -> bool:
    """Publish a manifest to the UNFED registry via gRPC PutManifest.

    Args:
        manifest: The manifest dict to publish.
        registry_address: Registry gRPC address (e.g. "localhost:50050").
        pool_name: Optional pool name override (default: model_id from manifest).

    Returns:
        True if published successfully, False otherwise.
    """
    import grpc

    # Import generated protobuf modules
    proto_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"
    )
    sys.path.insert(0, proto_dir)
    import registry_pb2
    import registry_pb2_grpc

    model_id = pool_name or manifest.get("model_id", "unknown")
    manifest_json = json.dumps(manifest)

    print(f"  Connecting to registry at {registry_address}...")
    channel = grpc.insecure_channel(registry_address)

    try:
        # Quick connectivity check
        stub = registry_pb2_grpc.RegistryStub(channel)

        print(f"  Uploading manifest for '{model_id}'...")
        resp = stub.PutManifest(
            registry_pb2.PutManifestRequest(
                model_id=model_id,
                manifest_json=manifest_json,
            ),
            timeout=30,
        )

        if resp.success:
            print(f"  Manifest published successfully: {resp.message}")
            return True
        else:
            print(f"  Registry rejected manifest: {resp.message}")
            return False

    except grpc.RpcError as e:
        status = e.code().name if hasattr(e, 'code') else 'UNKNOWN'
        details = e.details() if hasattr(e, 'details') else str(e)
        print(f"  Failed to publish manifest: [{status}] {details}")
        return False
    finally:
        channel.close()


def run_publish(args):
    """CLI handler for 'unfed publish'."""
    shards_dir = args.shards_dir
    registry_address = args.registry
    pool_name = args.pool_name

    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest.json not found in {shards_dir}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    model_id = pool_name or manifest.get("model_id", "unknown")

    print("=" * 60)
    print("  UNFED Shard Publisher")
    print("=" * 60)
    print()
    print(f"  Shards dir: {shards_dir}")
    print(f"  Registry:   {registry_address}")
    print(f"  Model ID:   {model_id}")
    print(f"  Model type: {manifest.get('model_type', 'unknown')}")
    print()

    # Verify all shard files exist
    # Support v3 unified shards array and legacy v2 separate arrays
    if manifest.get("shards"):
        all_shards = manifest["shards"]
    else:
        all_shards = manifest.get("text_shards", []) + manifest.get("vision_shards", [])
    total_size = 0
    print("  Shard files:")
    for shard in all_shards:
        fname = shard.get("file", "")
        path = os.path.join(shards_dir, fname)
        if not os.path.isfile(path):
            print(f"    [MISSING] {fname}")
            print(f"\n  ERROR: Shard file not found: {path}")
            print(f"  Run 'unfed split' first to generate shard files.")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        total_size += os.path.getsize(path)
        shard_type = "vision" if "vision" in fname else "text"
        print(f"    [OK] {fname:40s} {size_mb:>8.1f} MB  ({shard_type})")

    print(f"\n  Total: {len(all_shards)} shards, "
          f"{total_size / (1024 * 1024):.1f} MB")
    print()

    # Publish manifest to registry
    print("  Publishing manifest to registry...")
    success = publish_to_registry(manifest, registry_address, pool_name)

    print()
    if success:
        print("=" * 60)
        print("  Published successfully!")
        print()
        print("  Next steps:")
        print(f"    1. Start a seed node to serve shards for P2P download:")
        print(f"       python -m node.run --role compute --shard-index 0 \\")
        print(f"           --shards-dir {shards_dir}")
        print(f"    2. Other nodes can download shards via P2P:")
        print(f"       python -m shard.downloader --model-id {model_id}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  Publication failed. Check that the registry is running at:")
        print(f"    {registry_address}")
        print("=" * 60)
        sys.exit(1)
