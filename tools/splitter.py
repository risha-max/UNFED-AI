"""
UNFED Universal Model Splitter — splits any supported model into shards.

Auto-detects architecture via the inspector, then:
  1. Loads the full model (via transformers — only for splitting, not at runtime)
  2. Walks the module tree to find components as generic "stacks" (vision, text, audio, etc.)
  3. Splits each stack into shards with configurable per-stack shard counts
  4. Generates declarative manifest (v3) with unified shard list + legacy compat arrays
  5. Saves weights as safetensors (secure) or .pt (legacy)
  6. Computes per-shard + per-chunk SHA256 hashes
  7. Optionally generates verification vectors for cross-node consensus

Usage:
    python -m tools split /path/to/model -o ./shards --text-shards 2
    python -m tools split /path/to/model -o ./shards --shards text_decoder=30 vision_encoder=12
    python -m tools split HuggingFaceTB/SmolVLM-256M-Instruct --text-shards 2
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add project root for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.inspector import (
    detect_format,
    load_hf_config,
    extract_architecture_from_config,
)


# Default piece size for P2P chunk hashing
P2P_PIECE_SIZE = 4 * 1024 * 1024  # 4 MB

# Supported model types for auto-loading
_KNOWN_VL_MODEL_TYPES = {"idefics3", "smolvlm", "qwen2_vl"}


# ---------------------------------------------------------------------------
# ModelStack — universal representation of a group of transformer layers
# ---------------------------------------------------------------------------

@dataclass
class ModelStack:
    """A named group of transformer layers with optional embedding, head, and norm.

    This is the universal unit of splitting — the splitter iterates over a list
    of ModelStacks regardless of modality (vision, text, audio, etc.).
    """
    name: str                         # e.g. "vision_encoder", "text_decoder"
    stack_type: str                   # "encoder" or "decoder"
    layers: list                      # list of nn.Module (transformer layers)
    embedding: Optional[nn.Module] = None    # optional input embedding
    head: Optional[nn.Module] = None         # optional output head (lm_head, connector)
    norm: Optional[nn.Module] = None         # optional final norm
    layer_key_prefix: str = "layers"         # weight key prefix for layers
    embed_key: str = "embed_tokens"          # weight key prefix for embedding
    head_key: str = "lm_head"               # weight key prefix for head
    norm_key: str = "norm"                   # weight key prefix for norm
    # Legacy mapping for backward-compatible manifest arrays
    legacy_shard_array: str = ""             # e.g. "text_shards", "vision_shards"
    legacy_file_prefix: str = ""             # e.g. "text_shard", "vision_shard"
    # Extra metadata flags for legacy manifest entries
    legacy_embed_flag: str = "has_embedding"  # name of the boolean field
    legacy_head_flag: str = "has_lm_head"     # name of the boolean field
    # Architecture config key (for stacks{} in manifest)
    arch_config_key: str = ""                 # e.g. "text", "vision"
    # Extra metadata to carry through to the manifest
    extra_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

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


def compute_chunk_hashes(path: str, chunk_size: int = P2P_PIECE_SIZE) -> list[str]:
    """Compute per-chunk SHA256 hashes for piece-level P2P verification."""
    hashes = []
    with open(path, "rb") as f:
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            hashes.append(hashlib.sha256(piece).hexdigest())
    return hashes


# ---------------------------------------------------------------------------
# Shard saving
# ---------------------------------------------------------------------------

def save_shard(state_dict: dict[str, torch.Tensor], path: str,
               fmt: str = "safetensors"):
    """Save a state dict as a shard file.

    Args:
        state_dict: Weight tensors to save.
        path: Output file path.
        fmt: "safetensors" or "pt".
    """
    if fmt == "safetensors":
        try:
            from safetensors.torch import save_file
            # safetensors requires all tensors to be contiguous
            clean = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(clean, path)
            return
        except ImportError:
            print("  Warning: safetensors not installed, falling back to .pt")
            # Fall through to .pt
            path = path.replace(".safetensors", ".pt")

    torch.save(state_dict, path)


# ---------------------------------------------------------------------------
# Model loading — generic via transformers
# ---------------------------------------------------------------------------

def load_model(model_path: str, model_type: str, hf_config: Optional[dict] = None):
    """Load a full model for splitting.

    Uses transformers to load, but the loaded model is only used to extract
    weights — not for inference.  The generic runtime doesn't need transformers.

    Returns:
        (model, stacks, extras) where:
            stacks: list[ModelStack] — one per model component group
            extras: dict — model-level metadata (image_token_id, etc.)
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    # Check both the caller-provided hf_config and the auto-loaded config
    # for architecture detection.
    architectures = hf_config.get("architectures", []) if hf_config else []
    if not architectures:
        architectures = getattr(config, "architectures", None) or []
    arch_name = architectures[0] if architectures else ""
    # Also detect model_type from auto-config when the caller couldn't
    if model_type in ("unknown", ""):
        model_type = getattr(config, "model_type", model_type) or model_type

    print(f"  Loading model: {model_path}")
    print(f"  Architecture: {arch_name or model_type}")

    # Vision-language models need special loading classes
    if model_type in ("idefics3", "smolvlm") or "Idefics3" in arch_name:
        from transformers import Idefics3ForConditionalGeneration
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.float32,
        )
        model.eval()
        stacks, extras = _extract_stacks_idefics3(model, config)
        return model, stacks, extras

    elif model_type == "qwen2_vl" or "Qwen2VL" in arch_name:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.float32,
        )
        model.eval()
        stacks, extras = _extract_stacks_qwen2vl(model, config)
        return model, stacks, extras

    else:
        # Standard causal LM (Llama, Qwen2, Mistral, Gemma, etc.)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32,
        )
        model.eval()
        stacks, extras = _extract_stacks_causal_lm(model, config)
        return model, stacks, extras


# ---------------------------------------------------------------------------
# Component extraction — walk the model tree to find the building blocks
# ---------------------------------------------------------------------------

def _extract_stacks_idefics3(model, config) -> tuple[list[ModelStack], dict]:
    """Extract stacks from Idefics3/SmolVLM architecture.

    Returns (stacks, extras) where extras holds model-level metadata.
    """
    # The connector is the bridge between vision output and text input.
    # We attach it as the "head" of the vision encoder stack — the last
    # vision shard will contain post_layernorm + connector weights.
    connector = model.model.connector

    vision_stack = ModelStack(
        name="vision_encoder",
        stack_type="encoder",
        layers=list(model.model.vision_model.encoder.layers),
        embedding=model.model.vision_model.embeddings,
        head=connector,
        norm=model.model.vision_model.post_layernorm,
        layer_key_prefix="encoder.layers",
        embed_key="embeddings",
        head_key="connector",
        norm_key="post_layernorm",
        legacy_shard_array="vision_shards",
        legacy_file_prefix="vision_shard",
        legacy_embed_flag="has_embeddings",
        legacy_head_flag="has_connector",
        arch_config_key="vision",
    )

    text_stack = ModelStack(
        name="text_decoder",
        stack_type="decoder",
        layers=list(model.model.text_model.layers),
        embedding=model.model.text_model.embed_tokens,
        head=model.lm_head,
        norm=model.model.text_model.norm,
        layer_key_prefix="layers",
        embed_key="embed_tokens",
        head_key="lm_head",
        norm_key="norm",
        legacy_shard_array="text_shards",
        legacy_file_prefix="text_shard",
        legacy_embed_flag="has_embedding",
        legacy_head_flag="has_lm_head",
        arch_config_key="text",
    )

    extras = {
        "model_type": "smolvlm",
        "image_token_id": getattr(config, "image_token_id", None),
        "scale_factor": getattr(config, "scale_factor", None),
    }

    return [vision_stack, text_stack], extras


def _extract_stacks_qwen2vl(model, config) -> tuple[list[ModelStack], dict]:
    """Extract stacks from Qwen2-VL architecture.

    Returns (stacks, extras) where extras holds model-level metadata.
    """
    visual = model.visual
    merger = visual.merger if hasattr(visual, "merger") else None

    vision_stack = ModelStack(
        name="vision_encoder",
        stack_type="encoder",
        layers=list(visual.blocks),
        embedding=visual.patch_embed,
        head=merger,
        norm=getattr(visual, "post_layernorm", None),
        layer_key_prefix="encoder.layers",
        embed_key="embeddings",
        head_key="connector",
        norm_key="post_layernorm",
        legacy_shard_array="vision_shards",
        legacy_file_prefix="vision_shard",
        legacy_embed_flag="has_embeddings",
        legacy_head_flag="has_connector",
        arch_config_key="vision",
    )

    text_stack = ModelStack(
        name="text_decoder",
        stack_type="decoder",
        layers=list(model.model.layers),
        embedding=model.model.embed_tokens,
        head=model.lm_head,
        norm=model.model.norm,
        layer_key_prefix="layers",
        embed_key="embed_tokens",
        head_key="lm_head",
        norm_key="norm",
        legacy_shard_array="text_shards",
        legacy_file_prefix="text_shard",
        legacy_embed_flag="has_embedding",
        legacy_head_flag="has_lm_head",
        arch_config_key="text",
    )

    extras = {
        "model_type": "qwen2_vl",
        "image_token_id": getattr(config, "image_token_id", None),
        "spatial_merge_size": getattr(
            config.vision_config, "spatial_merge_size", 2
        ) if hasattr(config, "vision_config") else 2,
    }

    return [vision_stack, text_stack], extras


def _extract_stacks_causal_lm(model, config) -> tuple[list[ModelStack], dict]:
    """Extract stacks from standard causal LM (Llama, Qwen2, Mistral, etc.).

    Returns (stacks, extras) where extras holds model-level metadata.
    """
    inner = model.model if hasattr(model, "model") else model

    # Find layers — try common attribute names
    layers = None
    for attr in ("layers", "h", "blocks"):
        if hasattr(inner, attr):
            layers = list(getattr(inner, attr))
            break
    if layers is None:
        raise ValueError("Could not find decoder layers in model. "
                         "Tried: model.model.layers, model.model.h, model.model.blocks")

    # Find embedding
    embed = None
    for attr in ("embed_tokens", "wte", "embed_in"):
        if hasattr(inner, attr):
            embed = getattr(inner, attr)
            break

    # Find final norm
    norm = None
    for attr in ("norm", "ln_f", "final_layernorm"):
        if hasattr(inner, attr):
            norm = getattr(inner, attr)
            break

    # Find LM head
    lm_head = None
    for attr in ("lm_head",):
        if hasattr(model, attr):
            lm_head = getattr(model, attr)
            break

    text_stack = ModelStack(
        name="text_decoder",
        stack_type="decoder",
        layers=layers,
        embedding=embed,
        head=lm_head,
        norm=norm,
        layer_key_prefix="layers",
        embed_key="embed_tokens",
        head_key="lm_head",
        norm_key="norm",
        legacy_shard_array="text_shards",
        legacy_file_prefix="text_shard",
        legacy_embed_flag="has_embedding",
        legacy_head_flag="has_lm_head",
        arch_config_key="text",
    )

    extras = {
        "model_type": config.model_type if hasattr(config, "model_type") else "unknown",
    }

    return [text_stack], extras


# ---------------------------------------------------------------------------
# Find divisor helper
# ---------------------------------------------------------------------------

def _find_divisor(num_layers: int, target: int) -> int:
    """Find the largest divisor of num_layers that is <= target.
    Falls back to 1 if nothing divides evenly."""
    for d in range(target, 0, -1):
        if num_layers % d == 0:
            return d
    return 1


# ---------------------------------------------------------------------------
# Verification vectors
# ---------------------------------------------------------------------------

def generate_verification_vectors(
    stacks: list[ModelStack],
    num_vectors: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Generate deterministic test vectors for shard verification.

    Creates random inputs, runs them through the embedding of the first
    decoder stack, and records outputs. After splitting, each shard can
    be verified by running the same inputs through and checking outputs match.

    Returns list of {input, expected_output, metadata} dicts.
    """
    vectors = []
    torch.manual_seed(seed)

    # Find the first decoder stack with an embedding
    embed = None
    for stack in stacks:
        if stack.stack_type == "decoder" and stack.embedding is not None:
            embed = stack.embedding
            break

    if embed is None:
        return vectors

    vocab_size = embed.weight.shape[0]

    for i in range(num_vectors):
        input_ids = torch.randint(0, vocab_size, (1, 8 + i * 4))

        with torch.no_grad():
            hidden = embed(input_ids)

        vectors.append({
            "vector_id": i,
            "seed": seed + i,
            "input_ids": input_ids[0].tolist(),
            "embed_output_hash": hashlib.sha256(
                hidden.float().numpy().tobytes()
            ).hexdigest()[:16],
            "embed_output_norm": float(hidden.norm().item()),
        })

    return vectors


# ---------------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------------

def _resolve_shard_counts(
    stacks: list[ModelStack],
    shard_overrides: dict[str, int],
    legacy_text_shards: int = 0,
    legacy_vision_shards: int = 0,
) -> dict[str, int]:
    """Resolve per-stack shard counts from overrides and legacy CLI args.

    Priority: explicit --shards override > legacy --text-shards/--vision-shards > auto.

    Returns dict mapping stack.name -> num_shards.
    """
    result = {}
    for stack in stacks:
        num_layers = len(stack.layers)
        requested = 0

        # 1. Check explicit per-stack override (e.g. --shards text_decoder=30)
        if stack.name in shard_overrides:
            requested = shard_overrides[stack.name]

        # 2. Fall back to legacy flags
        elif legacy_text_shards > 0 and stack.stack_type == "decoder":
            requested = legacy_text_shards
        elif legacy_vision_shards > 0 and stack.stack_type == "encoder":
            requested = legacy_vision_shards

        # 3. Auto: default to 2 for decoders, 1 for encoders
        if requested <= 0:
            default_target = 4 if stack.stack_type == "decoder" else 2
            requested = _find_divisor(num_layers, default_target)

        # Ensure it divides evenly
        if num_layers % requested != 0:
            requested = _find_divisor(num_layers, requested)

        result[stack.name] = requested

    return result


def split_model(
    model_path: str,
    output_dir: str = "./shards",
    num_text_shards: int = 2,
    num_vision_shards: int = 1,
    output_format: str = "safetensors",
    generate_verify: bool = True,
    shard_overrides: Optional[dict[str, int]] = None,
) -> dict:
    """Split any supported model into shards with a declarative manifest.

    Args:
        model_path: HuggingFace model name or local path.
        output_dir: Where to save shards and manifest.
        num_text_shards: Legacy — number of text decoder shards (0 = auto).
        num_vision_shards: Legacy — number of vision encoder shards (0 = auto).
        output_format: "safetensors" or "pt".
        generate_verify: Whether to generate verification vectors.
        shard_overrides: Per-stack shard counts, e.g. {"text_decoder": 30}.
            Takes precedence over num_text_shards / num_vision_shards.

    Returns:
        The manifest dict.
    """
    start_time = time.time()
    ext = ".safetensors" if output_format == "safetensors" else ".pt"

    print("=" * 60)
    print("  UNFED Universal Model Splitter")
    print("=" * 60)
    print()

    # --- Step 1: Inspect ---
    print("[1/5] Inspecting model...")
    hf_config = None
    fmt = detect_format(model_path)
    if fmt == "hf_dir":
        hf_config = load_hf_config(model_path)
    elif fmt == "unknown":
        try:
            from transformers import AutoConfig as _AC
            _cfg = _AC.from_pretrained(model_path)
            hf_config = _cfg.to_dict() if hasattr(_cfg, "to_dict") else {}
        except Exception:
            hf_config = None

    arch = extract_architecture_from_config(hf_config) if hf_config else {}
    model_type = hf_config.get("model_type", "unknown") if hf_config else "unknown"
    print(f"  Model type: {model_type}")
    print(f"  Architecture: {arch.get('text', {}).get('num_layers', '?')} text layers, "
          f"{arch.get('vision', {}).get('num_layers', 0)} vision layers")

    # --- Step 2: Load model ---
    print()
    print("[2/5] Loading full model (this may take a moment)...")
    model, stacks, extras = load_model(model_path, model_type, hf_config)

    # Resolve per-stack shard counts
    shard_counts = _resolve_shard_counts(
        stacks,
        shard_overrides=shard_overrides or {},
        legacy_text_shards=num_text_shards,
        legacy_vision_shards=num_vision_shards,
    )

    # Print shard plan
    print()
    print("  Shard plan:")
    for stack in stacks:
        n_shards = shard_counts[stack.name]
        n_layers = len(stack.layers)
        lps = n_layers // n_shards
        print(f"    {stack.name}: {n_shards} shards x {lps} layers "
              f"(total {n_layers})")

    # --- Step 3: Create output directory and start building manifest ---
    os.makedirs(output_dir, exist_ok=True)

    resolved_model_type = extras.get("model_type", model_type)

    manifest = {
        "model_id": model_path,
        "model_type": resolved_model_type,
        "format_version": 3,
        "architecture": arch,
        # v3: unified shard list + stacks metadata
        "stacks": {},
        "shards": [],
        # Legacy v2 compat arrays (populated alongside unified list)
        "vision_shards": [],
        "text_shards": [],
    }

    # Add model-level extras (image_token_id, scale_factor, etc.)
    for key, value in extras.items():
        if key != "model_type" and value is not None:
            manifest[key] = value

    # Build stacks metadata and legacy summary counts
    for stack in stacks:
        n_shards = shard_counts[stack.name]
        n_layers = len(stack.layers)
        lps = n_layers // n_shards
        manifest["stacks"][stack.name] = {
            "stack_type": stack.stack_type,
            "num_layers": n_layers,
            "layers_per_shard": lps,
            "num_shards": n_shards,
        }
        # Also inject the architecture config for this stack if available
        if stack.arch_config_key and stack.arch_config_key in arch:
            manifest["stacks"][stack.name]["architecture"] = arch[stack.arch_config_key]

    # Legacy summary counts (for v2 readers)
    _vision_stacks = [s for s in stacks if s.stack_type == "encoder"]
    _text_stacks = [s for s in stacks if s.stack_type == "decoder"]
    manifest["vision"] = {
        "num_layers": sum(len(s.layers) for s in _vision_stacks),
        "layers_per_shard": (len(_vision_stacks[0].layers) // shard_counts[_vision_stacks[0].name]
                             if _vision_stacks else 0),
        "num_shards": (shard_counts[_vision_stacks[0].name]
                       if _vision_stacks else 0),
    }
    manifest["text"] = {
        "num_layers": sum(len(s.layers) for s in _text_stacks),
        "layers_per_shard": (len(_text_stacks[0].layers) // shard_counts[_text_stacks[0].name]
                             if _text_stacks else 0),
        "num_shards": (shard_counts[_text_stacks[0].name]
                       if _text_stacks else 0),
    }

    # --- Step 4: Split and save shards (generic loop over stacks) ---
    print()
    print("[3/5] Splitting into shards...")

    for stack in stacks:
        n_shards = shard_counts[stack.name]
        n_layers = len(stack.layers)
        lps = n_layers // n_shards

        for shard_idx in range(n_shards):
            start_layer = shard_idx * lps
            end_layer = start_layer + lps
            is_first = shard_idx == 0
            is_last = shard_idx == n_shards - 1

            file_prefix = stack.legacy_file_prefix or stack.name
            shard_file = f"{file_prefix}_{shard_idx}{ext}"
            shard_path = os.path.join(output_dir, shard_file)

            print(f"\n  {stack.name} shard {shard_idx}: "
                  f"layers [{start_layer}, {end_layer})")

            shard = {}

            # First shard: embedding
            if is_first and stack.embedding is not None:
                emb_sd = stack.embedding.state_dict()
                for k, v in emb_sd.items():
                    shard[f"{stack.embed_key}.{k}"] = v
                print(f"    + {stack.embed_key}")

            # Layers
            for layer_idx in range(start_layer, end_layer):
                layer_sd = stack.layers[layer_idx].state_dict()
                for k, v in layer_sd.items():
                    shard[f"{stack.layer_key_prefix}.{layer_idx}.{k}"] = v
            print(f"    + {stack.layer_key_prefix} [{start_layer}, {end_layer})")

            # Last shard: norm + head
            if is_last:
                if stack.norm is not None:
                    norm_sd = stack.norm.state_dict()
                    for k, v in norm_sd.items():
                        shard[f"{stack.norm_key}.{k}"] = v
                    print(f"    + {stack.norm_key}")

                if stack.head is not None:
                    head_sd = stack.head.state_dict()
                    for k, v in head_sd.items():
                        shard[f"{stack.head_key}.{k}"] = v
                    print(f"    + {stack.head_key}")

            save_shard(shard, shard_path, output_format)
            shard_size = os.path.getsize(shard_path)
            shard_hash = compute_file_hash(shard_path)
            chunk_hashes = compute_chunk_hashes(shard_path)
            print(f"    Saved: {shard_file} ({shard_size / (1024 * 1024):.1f} MB, "
                  f"{len(chunk_hashes)} chunks)")

            # v3 unified shard entry
            shard_entry = {
                "shard_index": shard_idx,
                "stack": stack.name,
                "layer_start": start_layer,
                "layer_end": end_layer,
                "has_embedding": is_first and stack.embedding is not None,
                "has_head": is_last and stack.head is not None,
                "has_norm": is_last and stack.norm is not None,
                "file": shard_file,
                "size_bytes": shard_size,
                "sha256": shard_hash,
                "chunk_size": P2P_PIECE_SIZE,
                "chunk_hashes": chunk_hashes,
            }
            manifest["shards"].append(shard_entry)

            # Legacy v2 compat entry
            if stack.legacy_shard_array:
                legacy_entry = {
                    "shard_index": shard_idx,
                    "layer_start": start_layer,
                    "layer_end": end_layer,
                    stack.legacy_embed_flag: is_first and stack.embedding is not None,
                    stack.legacy_head_flag: is_last and stack.head is not None,
                    "file": shard_file,
                    "size_bytes": shard_size,
                    "sha256": shard_hash,
                    "chunk_size": P2P_PIECE_SIZE,
                    "chunk_hashes": chunk_hashes,
                }
                manifest[stack.legacy_shard_array].append(legacy_entry)

    # --- Step 5: Verification vectors ---
    if generate_verify:
        print()
        print("[4/5] Generating verification vectors...")
        vectors = generate_verification_vectors(stacks)
        if vectors:
            manifest["verification_vectors"] = vectors
            print(f"  Generated {len(vectors)} verification vectors")
        else:
            print("  Skipped (could not generate)")
    else:
        print()
        print("[4/5] Verification vectors: skipped")

    # --- Save manifest ---
    print()
    print("[5/5] Saving manifest...")
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    # Also copy config.json if from HuggingFace
    if hf_config:
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(hf_config, f, indent=2)
        print(f"  Saved: {config_path} (original HF config)")

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    # Summary
    total_shards = len(manifest["shards"])
    total_size = sum(s["size_bytes"] for s in manifest["shards"])

    print()
    print("=" * 60)
    print(f"  Split complete in {elapsed:.1f}s")
    print(f"  Output:  {output_dir}/")
    print(f"  Shards:  {total_shards} ({total_size / (1024 * 1024):.1f} MB total)")
    for stack in stacks:
        n_shards = shard_counts[stack.name]
        lps = len(stack.layers) // n_shards
        print(f"    {stack.name}: {n_shards} shards ({lps} layers each)")
    print(f"  Format:  {output_format}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 60)

    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_shard_overrides(shard_args: Optional[list[str]]) -> dict[str, int]:
    """Parse --shards arguments like 'text_decoder=30 vision_encoder=12'.

    Returns dict mapping stack_name -> num_shards.
    """
    if not shard_args:
        return {}
    result = {}
    for arg in shard_args:
        if "=" not in arg:
            raise ValueError(
                f"Invalid --shards format: '{arg}'. "
                f"Expected format: stack_name=count (e.g. text_decoder=30)"
            )
        name, count_str = arg.split("=", 1)
        result[name.strip()] = int(count_str.strip())
    return result


def run_split(args):
    """CLI handler for 'unfed split'."""
    shard_overrides = _parse_shard_overrides(
        getattr(args, "shards", None)
    )
    split_model(
        model_path=args.model_path,
        output_dir=args.output,
        num_text_shards=args.text_shards,
        num_vision_shards=args.vision_shards,
        output_format=args.format,
        generate_verify=args.verify,
        shard_overrides=shard_overrides,
    )
