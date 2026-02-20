"""
SmolVLM Splitter — splits HuggingFaceTB/SmolVLM-256M-Instruct into vision + text shards.

SmolVLM architecture:
  model (SmolVLMModel)
  ├── vision_model (SmolVLMVisionTransformer)
  │   ├── embeddings (SmolVLMVisionEmbeddings — patch embed + position embed)
  │   ├── encoder.layers[0..N-1] (SmolVLMEncoderLayer — SigLIP ViT blocks)
  │   └── post_layernorm (LayerNorm)
  ├── connector (SmolVLMConnector — pixel shuffle + MLP projection)
  └── text_model (LlamaModel / SmolLM)
      ├── embed_tokens
      ├── layers[0..M-1] (LlamaDecoderLayer)
      └── norm
  lm_head (Linear)

Shard layout (auto-computed from model config):
  Vision shard 0: embeddings + first half of ViT encoder layers
  Vision shard 1: second half of ViT encoder layers + post_layernorm + connector
  Text shard 0:   embed_tokens + first quarter of LLM layers
  ...
  Text shard N:   last quarter of LLM layers + norm + lm_head

Generates shards_smolvlm/ directory with shard files and manifest.json.

Usage:
    python -m shard.smolvlm_splitter
    python -m shard.smolvlm_splitter --model-path /path/to/SmolVLM-256M-Instruct
    python -m shard.smolvlm_splitter --num-text-shards 2
    python -m shard.smolvlm_splitter --registry localhost:50050
"""

import argparse
import hashlib
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Default model
SMOLVLM_MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
SMOLVLM_SHARDS_DIR = "shards_smolvlm"


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
    """Compute per-chunk SHA256 hashes for piece-level P2P verification."""
    hashes = []
    with open(path, "rb") as f:
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            hashes.append(hashlib.sha256(piece).hexdigest())
    return hashes


def _find_divisor(num_layers: int, target: int) -> int:
    """Find the largest divisor of num_layers that is <= target.
    Falls back to 1 if nothing divides evenly (single shard per type)."""
    for d in range(target, 0, -1):
        if num_layers % d == 0:
            return d
    return 1


def split_smolvlm(model_path: str = SMOLVLM_MODEL_NAME,
                   num_vision_shards: int = 0,
                   num_text_shards: int = 0):
    """Split SmolVLM into vision + text shards with manifest.

    If num_vision_shards or num_text_shards are 0, they are auto-computed
    to give roughly equal shard sizes (targeting 2 vision, 4 text as defaults).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"Loading model config: {model_path}")
    model_config = AutoConfig.from_pretrained(model_path)

    # Read architecture dimensions
    vision_config = model_config.vision_config
    text_config = model_config.text_config

    num_vit_layers = vision_config.num_hidden_layers
    num_llm_layers = text_config.num_hidden_layers
    vision_hidden = vision_config.hidden_size
    text_hidden = text_config.hidden_size
    vocab_size = text_config.vocab_size
    scale_factor = model_config.scale_factor
    image_token_id = model_config.image_token_id

    print(f"  Vision: {num_vit_layers} SigLIP encoder layers, hidden={vision_hidden}")
    print(f"  Text:   {num_llm_layers} LLM layers, hidden={text_hidden}, vocab={vocab_size}")
    print(f"  Connector: scale_factor={scale_factor}")
    print(f"  image_token_id={image_token_id}")

    # Auto-compute shard counts if not specified
    if num_vision_shards <= 0:
        num_vision_shards = _find_divisor(num_vit_layers, 2)
    if num_text_shards <= 0:
        num_text_shards = _find_divisor(num_llm_layers, 4)

    if num_vit_layers % num_vision_shards != 0:
        print(f"ERROR: {num_vit_layers} ViT layers not evenly divisible by "
              f"{num_vision_shards} vision shards")
        return None
    if num_llm_layers % num_text_shards != 0:
        print(f"ERROR: {num_llm_layers} LLM layers not evenly divisible by "
              f"{num_text_shards} text shards")
        return None

    vit_layers_per_shard = num_vit_layers // num_vision_shards
    llm_layers_per_shard = num_llm_layers // num_text_shards

    print(f"\nShard plan:")
    print(f"  Vision: {num_vision_shards} shards x {vit_layers_per_shard} ViT layers")
    print(f"  Text:   {num_text_shards} shards x {llm_layers_per_shard} LLM layers")

    # Load the full model
    print("\nLoading full model (this may take a moment)...")
    from transformers import SmolVLMForConditionalGeneration
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float32,
    )
    model.eval()

    # SmolVLMForConditionalGeneration structure:
    #   model.model.vision_model  (SmolVLMVisionTransformer)
    #   model.model.connector     (SmolVLMConnector)
    #   model.model.text_model    (AutoModel — usually LlamaModel)
    #   model.lm_head             (nn.Linear)
    vision_model = model.model.vision_model
    connector = model.model.connector
    text_model = model.model.text_model
    lm_head = model.lm_head

    # Create output directory
    os.makedirs(SMOLVLM_SHARDS_DIR, exist_ok=True)

    manifest = {
        "model_id": model_path,
        "model_type": "smolvlm",
        "image_token_id": image_token_id,
        "scale_factor": scale_factor,
        "vision": {
            "num_layers": num_vit_layers,
            "layers_per_shard": vit_layers_per_shard,
            "num_shards": num_vision_shards,
            "hidden_size": vision_hidden,
        },
        "text": {
            "num_layers": num_llm_layers,
            "layers_per_shard": llm_layers_per_shard,
            "num_shards": num_text_shards,
            "hidden_size": text_hidden,
            "vocab_size": vocab_size,
        },
        "vision_shards": [],
        "text_shards": [],
    }

    # ─── Vision shards ───
    # SmolVLM vision: embeddings + encoder.layers[0..N] + post_layernorm
    # Connector (pixel shuffle + MLP) goes with the last vision shard.
    for shard_idx in range(num_vision_shards):
        start_layer = shard_idx * vit_layers_per_shard
        end_layer = start_layer + vit_layers_per_shard
        is_first = shard_idx == 0
        is_last = shard_idx == num_vision_shards - 1
        shard_path = os.path.join(SMOLVLM_SHARDS_DIR,
                                  f"vision_shard_{shard_idx}.pt")

        print(f"\nCreating vision shard {shard_idx}: "
              f"ViT layers {start_layer}-{end_layer - 1}")

        shard = {}

        # First vision shard: patch/position embeddings
        if is_first:
            shard["embeddings"] = vision_model.embeddings.state_dict()
            print(f"  + embeddings (patch + position)")

        # Encoder layers
        for layer_idx in range(start_layer, end_layer):
            shard[f"encoder_layer_{layer_idx}"] = \
                vision_model.encoder.layers[layer_idx].state_dict()
        print(f"  + encoder layers {start_layer}-{end_layer - 1}")

        # Last vision shard: post_layernorm + connector
        if is_last:
            shard["post_layernorm"] = vision_model.post_layernorm.state_dict()
            shard["connector"] = connector.state_dict()
            print(f"  + post_layernorm")
            print(f"  + connector (pixel shuffle + MLP)")

        torch.save(shard, shard_path)
        shard_size = os.path.getsize(shard_path)
        shard_hash = compute_file_hash(shard_path)
        chunk_hashes = compute_chunk_hashes(shard_path, config.P2P_PIECE_SIZE)
        print(f"  Saved to {shard_path} ({shard_size / (1024 * 1024):.1f} MB)")
        print(f"  SHA256: {shard_hash[:16]}... ({len(chunk_hashes)} chunks)")

        manifest["vision_shards"].append({
            "shard_index": shard_idx,
            "layer_start": start_layer,
            "layer_end": end_layer,
            "has_embeddings": is_first,
            "has_connector": is_last,
            "file": os.path.basename(shard_path),
            "size_bytes": shard_size,
            "sha256": shard_hash,
            "chunk_size": config.P2P_PIECE_SIZE,
            "chunk_hashes": chunk_hashes,
        })

    # ─── Text shards ───
    # SmolVLM text: embed_tokens + layers[0..M] + norm  (+ lm_head)
    for shard_idx in range(num_text_shards):
        start_layer = shard_idx * llm_layers_per_shard
        end_layer = start_layer + llm_layers_per_shard
        is_first = shard_idx == 0
        is_last = shard_idx == num_text_shards - 1
        shard_path = os.path.join(SMOLVLM_SHARDS_DIR,
                                  f"text_shard_{shard_idx}.pt")

        print(f"\nCreating text shard {shard_idx}: "
              f"LLM layers {start_layer}-{end_layer - 1}")

        shard = {}

        # First text shard: embed_tokens
        if is_first:
            shard["embed_tokens"] = text_model.embed_tokens.state_dict()
            print(f"  + embed_tokens")

        # LLM layers
        for layer_idx in range(start_layer, end_layer):
            shard[f"layer_{layer_idx}"] = \
                text_model.layers[layer_idx].state_dict()
        print(f"  + LLM layers {start_layer}-{end_layer - 1}")

        # Last text shard: norm + lm_head
        if is_last:
            shard["norm"] = text_model.norm.state_dict()
            shard["lm_head"] = lm_head.state_dict()
            print(f"  + norm + lm_head")

        torch.save(shard, shard_path)
        shard_size = os.path.getsize(shard_path)
        shard_hash = compute_file_hash(shard_path)
        chunk_hashes = compute_chunk_hashes(shard_path, config.P2P_PIECE_SIZE)
        print(f"  Saved to {shard_path} ({shard_size / (1024 * 1024):.1f} MB)")
        print(f"  SHA256: {shard_hash[:16]}... ({len(chunk_hashes)} chunks)")

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
    model_config.save_pretrained(SMOLVLM_SHARDS_DIR)
    print(f"\nModel config saved to {SMOLVLM_SHARDS_DIR}/config.json")

    # Save manifest
    manifest_path = os.path.join(SMOLVLM_SHARDS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    # Clean up model to free memory
    del model, vision_model, connector, text_model, lm_head
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nDone! All shards created successfully.")
    print(f"\nVision shards: {num_vision_shards} "
          f"({vit_layers_per_shard} ViT layers each)")
    print(f"Text shards:   {num_text_shards} "
          f"({llm_layers_per_shard} LLM layers each)")
    print(f"Total:         {num_vision_shards + num_text_shards} shards")
    print(f"\nTo publish the manifest to the registry:")
    print(f"  python -m shard.smolvlm_splitter --model-path {model_path} "
          f"--registry localhost:50050")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UNFED AI SmolVLM Splitter")
    parser.add_argument("--model-path", type=str, default=SMOLVLM_MODEL_NAME,
                        help=f"Model name or local path "
                             f"(default: {SMOLVLM_MODEL_NAME})")
    parser.add_argument("--num-vision-shards", type=int, default=0,
                        help="Number of vision shards (0 = auto)")
    parser.add_argument("--num-text-shards", type=int, default=0,
                        help="Number of text shards (0 = auto)")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address to publish manifest to "
                             "(e.g. localhost:50050)")
    args = parser.parse_args()

    manifest = split_smolvlm(
        args.model_path,
        num_vision_shards=args.num_vision_shards,
        num_text_shards=args.num_text_shards,
    )

    if args.registry and manifest:
        from shard.splitter import publish_manifest_to_registry
        publish_manifest_to_registry(manifest, args.registry)
