"""
Vision Model Splitter — splits Qwen2-VL-2B-Instruct into vision + text shards.

Shard layout (6 shards, 7 nodes with MPC):
  Vision shard 0: patch_embed (Conv3D) + ViT blocks 0-15   (MPC node owns this)
  Vision shard 1: ViT blocks 16-31 + PatchMerger
  Text shard 0:   embed_tokens + LLM layers 0-6   (has_embedding=True)
  Text shard 1:   LLM layers 7-13
  Text shard 2:   LLM layers 14-20
  Text shard 3:   LLM layers 21-27 + norm + lm_head  (has_lm_head=True)

Generates shards_vl/ directory with shard files and manifest.json.

Usage:
    python -m shard.vision_splitter
    python -m shard.vision_splitter --model-path /path/to/Qwen2-VL-2B-Instruct
"""

import argparse
import hashlib
import json
import os
import sys
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Default model name (can be overridden with --model-path for local checkpoints)
VL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
VL_SHARDS_DIR = "shards_vl"

# Vision shard layout
NUM_VIT_BLOCKS = 32
VIT_BLOCKS_PER_SHARD = 16  # 2 vision shards, 16 blocks each
NUM_VISION_SHARDS = 2

# Text shard layout
NUM_LLM_LAYERS = 28
LLM_LAYERS_PER_SHARD = 7  # 4 text shards, 7 layers each
NUM_TEXT_SHARDS = 4


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
    """Compute per-chunk SHA256 hashes for piece-level verification."""
    hashes = []
    with open(path, "rb") as f:
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            hashes.append(hashlib.sha256(piece).hexdigest())
    return hashes


def split_vision_model(model_path: str = VL_MODEL_NAME):
    """Split Qwen2-VL-2B-Instruct into vision + text shards."""

    print(f"Loading model: {model_path}")
    model_config = AutoConfig.from_pretrained(model_path)

    # Verify architecture
    vit_blocks = model_config.vision_config.depth
    llm_layers = model_config.text_config.num_hidden_layers
    print(f"  Vision: {vit_blocks} ViT blocks, hidden={model_config.vision_config.embed_dim}")
    print(f"  Text:   {llm_layers} LLM layers, hidden={model_config.text_config.hidden_size}")

    assert vit_blocks == NUM_VIT_BLOCKS, f"Expected {NUM_VIT_BLOCKS} ViT blocks, got {vit_blocks}"
    assert llm_layers == NUM_LLM_LAYERS, f"Expected {NUM_LLM_LAYERS} LLM layers, got {llm_layers}"

    # Load the full model
    print("Loading full model (this may take a moment)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model.eval()

    visual = model.model.visual
    language = model.model.language_model

    # Create output directory
    os.makedirs(VL_SHARDS_DIR, exist_ok=True)

    manifest = {
        "model_id": model_path,
        "model_type": "qwen2_vl",
        "vision": {
            "num_blocks": NUM_VIT_BLOCKS,
            "blocks_per_shard": VIT_BLOCKS_PER_SHARD,
            "num_shards": NUM_VISION_SHARDS,
            "hidden_size": model_config.vision_config.embed_dim,
            "merger_output_dim": model_config.text_config.hidden_size,
        },
        "text": {
            "num_layers": NUM_LLM_LAYERS,
            "layers_per_shard": LLM_LAYERS_PER_SHARD,
            "num_shards": NUM_TEXT_SHARDS,
            "hidden_size": model_config.text_config.hidden_size,
            "vocab_size": model_config.text_config.vocab_size,
        },
        "vision_shards": [],
        "text_shards": [],
    }

    # ─── Vision shards ───
    for shard_idx in range(NUM_VISION_SHARDS):
        start_block = shard_idx * VIT_BLOCKS_PER_SHARD
        end_block = start_block + VIT_BLOCKS_PER_SHARD
        is_first = shard_idx == 0
        is_last = shard_idx == NUM_VISION_SHARDS - 1
        shard_path = os.path.join(VL_SHARDS_DIR, f"vision_shard_{shard_idx}.pt")

        print(f"\nCreating vision shard {shard_idx}: ViT blocks {start_block}-{end_block - 1}")

        shard = {}

        # First vision shard: patch_embed (Conv3D) + rotary_pos_emb
        if is_first:
            shard["patch_embed"] = visual.patch_embed.state_dict()
            shard["rotary_pos_emb"] = visual.rotary_pos_emb.state_dict()
            print(f"  + patch_embed (Conv3D)")
            print(f"  + rotary_pos_emb")

        # ViT blocks
        for block_idx in range(start_block, end_block):
            shard[f"block_{block_idx}"] = visual.blocks[block_idx].state_dict()
        print(f"  + ViT blocks {start_block}-{end_block - 1}")

        # Last vision shard: PatchMerger
        if is_last:
            shard["merger"] = visual.merger.state_dict()
            print(f"  + PatchMerger")

        torch.save(shard, shard_path)
        shard_size = os.path.getsize(shard_path)
        shard_hash = compute_file_hash(shard_path)
        chunk_hashes = compute_chunk_hashes(shard_path, config.P2P_PIECE_SIZE)
        print(f"  Saved to {shard_path} ({shard_size / (1024 * 1024):.1f} MB)")

        manifest["vision_shards"].append({
            "shard_index": shard_idx,
            "block_start": start_block,
            "block_end": end_block,
            "has_patch_embed": is_first,
            "has_merger": is_last,
            "file": os.path.basename(shard_path),
            "size_bytes": shard_size,
            "sha256": shard_hash,
            "chunk_size": config.P2P_PIECE_SIZE,
            "chunk_hashes": chunk_hashes,
        })

    # ─── Text shards ───
    for shard_idx in range(NUM_TEXT_SHARDS):
        start_layer = shard_idx * LLM_LAYERS_PER_SHARD
        end_layer = start_layer + LLM_LAYERS_PER_SHARD
        is_first = shard_idx == 0
        is_last = shard_idx == NUM_TEXT_SHARDS - 1
        shard_path = os.path.join(VL_SHARDS_DIR, f"text_shard_{shard_idx}.pt")

        print(f"\nCreating text shard {shard_idx}: LLM layers {start_layer}-{end_layer - 1}")

        shard = {}

        # First text shard: embed_tokens
        if is_first:
            shard["embed_tokens"] = language.embed_tokens.state_dict()
            print(f"  + embed_tokens")

        # LLM layers
        for layer_idx in range(start_layer, end_layer):
            shard[f"layer_{layer_idx}"] = language.layers[layer_idx].state_dict()
        print(f"  + LLM layers {start_layer}-{end_layer - 1}")

        # Last text shard: norm + lm_head
        if is_last:
            shard["norm"] = language.norm.state_dict()
            shard["lm_head"] = model.lm_head.state_dict()
            print(f"  + norm + lm_head")

        torch.save(shard, shard_path)
        shard_size = os.path.getsize(shard_path)
        shard_hash = compute_file_hash(shard_path)
        chunk_hashes = compute_chunk_hashes(shard_path, config.P2P_PIECE_SIZE)
        print(f"  Saved to {shard_path} ({shard_size / (1024 * 1024):.1f} MB)")

        manifest["text_shards"].append({
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

    # Save model config for reconstruction
    model_config.save_pretrained(VL_SHARDS_DIR)
    print(f"\nModel config saved to {VL_SHARDS_DIR}/config.json")

    # Save manifest
    manifest_path = os.path.join(VL_SHARDS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    print("\nDone! All shards created successfully.")
    print(f"\nVision shards: {NUM_VISION_SHARDS} (ViT blocks)")
    print(f"Text shards:   {NUM_TEXT_SHARDS} (LLM layers)")
    print(f"Total:         {NUM_VISION_SHARDS + NUM_TEXT_SHARDS} shards")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Vision Model Splitter (Qwen2-VL)")
    parser.add_argument("--model-path", type=str, default=VL_MODEL_NAME,
                        help="Model name or local path (default: Qwen/Qwen2-VL-2B-Instruct)")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address to publish manifest to (e.g. localhost:50050)")
    args = parser.parse_args()

    manifest = split_vision_model(args.model_path)

    if args.registry and manifest:
        # Reuse the existing publish function from the text splitter
        from shard.splitter import publish_manifest_to_registry
        publish_manifest_to_registry(manifest, args.registry)
