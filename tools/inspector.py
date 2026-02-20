"""
UNFED Model Inspector — analyse any model for UNFED compatibility.

Auto-detects:
  - Format (HuggingFace directory, safetensors, PyTorch .pt, GGUF)
  - Architecture (Llama, Qwen2, Mistral, SmolVLM, SigLIP, etc.)
  - Component structure (embedding, decoder layers, vision encoder, connector)
  - Key hyperparameters (hidden_size, num_heads, num_layers, vocab_size, etc.)

Infers architecture from weight tensor shapes even without config.json.

Usage:
    python -m tools inspect /path/to/model
    python -m tools inspect model.safetensors --verbose
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """Detect the model format from a path.

    Returns one of: "hf_dir", "safetensors", "pt", "gguf", "unknown".
    """
    p = Path(path)

    if p.is_dir():
        # Check for HuggingFace directory markers
        has_config = (p / "config.json").exists()
        has_index = (p / "model.safetensors.index.json").exists()
        has_st = any(p.glob("*.safetensors"))
        has_pt = any(p.glob("*.bin")) or any(p.glob("*.pt"))
        if has_config and (has_st or has_pt or has_index):
            return "hf_dir"
        if has_config:
            # Config-only HF directory (weights not downloaded yet)
            return "hf_dir"
        if has_st:
            return "safetensors"
        if has_pt:
            return "pt"
        return "unknown"

    ext = p.suffix.lower()
    if ext == ".safetensors":
        return "safetensors"
    elif ext in (".pt", ".pth", ".bin"):
        return "pt"
    elif ext == ".gguf":
        return "gguf"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Weight file loading (metadata only for inspection)
# ---------------------------------------------------------------------------

def load_weight_keys(path: str) -> dict[str, tuple[torch.Size, str]]:
    """Load weight metadata (key → (shape, dtype)) without loading full tensors.

    For safetensors, this is zero-copy and very fast.
    For .pt files, we need to fully load (slower).
    """
    ext = Path(path).suffix.lower()

    if ext == ".safetensors":
        try:
            from safetensors import safe_open
            result = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # Get metadata (shape/dtype) without full load
                    metadata = f.metadata()
                    tensor = f.get_slice(key)
                    shape = tensor.get_shape()
                    dtype = str(tensor.get_dtype())
                    result[key] = (torch.Size(shape), dtype)
            return result
        except (ImportError, AttributeError):
            # Fallback: get_slice not available in older versions
            try:
                from safetensors import safe_open
                result = {}
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        result[key] = (tensor.shape, str(tensor.dtype))
                return result
            except Exception:
                pass
        return {}

    elif ext in (".pt", ".pth", ".bin"):
        sd = torch.load(path, map_location="meta", weights_only=True)
        return {k: (v.shape, str(v.dtype)) for k, v in sd.items()}

    return {}


def gather_all_weights(model_path: str, fmt: str) -> dict[str, tuple[torch.Size, str]]:
    """Gather all weight keys and shapes from a model.

    For HuggingFace directories, merges all shard files.
    """
    p = Path(model_path)
    all_keys = {}

    if fmt == "hf_dir":
        # Load from all weight files
        for pattern in ("*.safetensors", "*.bin", "*.pt"):
            for f in sorted(p.glob(pattern)):
                try:
                    keys = load_weight_keys(str(f))
                    all_keys.update(keys)
                except Exception as e:
                    print(f"  Warning: could not load {f.name}: {e}")
    elif fmt in ("safetensors", "pt"):
        all_keys = load_weight_keys(model_path)
    else:
        print(f"  Cannot inspect format: {fmt}")

    return all_keys


# ---------------------------------------------------------------------------
# Architecture inference from weight shapes
# ---------------------------------------------------------------------------

# Known model architectures and their weight key patterns
_ARCH_PATTERNS = {
    "llama": {
        "embed": "model.embed_tokens.weight",
        "layer": "model.layers.{idx}.self_attn.q_proj.weight",
        "norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    },
    "qwen2": {
        "embed": "model.embed_tokens.weight",
        "layer": "model.layers.{idx}.self_attn.q_proj.weight",
        "norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    },
    "mistral": {
        "embed": "model.embed_tokens.weight",
        "layer": "model.layers.{idx}.self_attn.q_proj.weight",
        "norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    },
    "gemma": {
        "embed": "model.embed_tokens.weight",
        "layer": "model.layers.{idx}.self_attn.q_proj.weight",
        "norm": "model.norm.weight",
        # Gemma ties embeddings to lm_head
    },
    "smolvlm": {
        "embed": "model.text_model.embed_tokens.weight",
        "layer": "model.text_model.layers.{idx}.self_attn.q_proj.weight",
        "norm": "model.text_model.norm.weight",
        "lm_head": "lm_head.weight",
        "vision_embed": "model.vision_model.embeddings.patch_embedding.weight",
        "vision_layer": "model.vision_model.encoder.layers.{idx}.self_attn.q_proj.weight",
        "connector": "model.connector.modality_projection.0.weight",
    },
    "qwen2_vl": {
        "embed": "model.embed_tokens.weight",
        "layer": "model.layers.{idx}.self_attn.q_proj.weight",
        "norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "vision_embed": "visual.patch_embed.proj.weight",
        "vision_layer": "visual.blocks.{idx}.attn.qkv.weight",
        "connector": "visual.merger.mlp.0.weight",
    },
}


def infer_architecture_from_keys(
    keys: dict[str, tuple[torch.Size, str]],
) -> dict:
    """Infer model architecture from weight key names and shapes.

    Returns a dict with:
        model_type, text_config, vision_config, connector_config,
        num_text_layers, num_vision_layers, total_params
    """
    key_names = set(keys.keys())
    result = {
        "model_type": "unknown",
        "text": {},
        "vision": {},
        "connector": {},
        "num_text_layers": 0,
        "num_vision_layers": 0,
        "total_params": sum(
            s.numel() if hasattr(s, 'numel') else _prod(s)
            for s, _ in keys.values()
        ),
        "components": [],
    }

    # Match against known patterns
    best_match = None
    best_score = 0

    for arch_name, patterns in _ARCH_PATTERNS.items():
        score = 0
        for role, pattern in patterns.items():
            if "{idx}" in pattern:
                # Check if any layer index matches
                test = pattern.replace("{idx}", "0")
                if test in key_names:
                    score += 1
            else:
                if pattern in key_names:
                    score += 1
        if score > best_score:
            best_score = score
            best_match = arch_name

    if best_match:
        result["model_type"] = best_match

    # Count layers
    result["num_text_layers"] = _count_layers(key_names, "text")
    result["num_vision_layers"] = _count_layers(key_names, "vision")

    # Infer text config from shapes
    result["text"] = _infer_text_config(keys)
    result["vision"] = _infer_vision_config(keys)
    result["connector"] = _infer_connector_config(keys)

    # Component list
    components = []
    if result["text"].get("vocab_size"):
        components.append("embedding")
    if result["num_text_layers"] > 0:
        components.append(f"text_decoder ({result['num_text_layers']} layers)")
    if any("lm_head" in k for k in key_names):
        components.append("lm_head")
    if result["num_vision_layers"] > 0:
        components.append(f"vision_encoder ({result['num_vision_layers']} layers)")
    if result["connector"]:
        components.append("connector")
    result["components"] = components

    return result


def _count_layers(keys: set[str], component: str) -> int:
    """Count the number of transformer layers for a component."""
    max_idx = -1

    if component == "text":
        # Match: model.layers.{N}. or model.text_model.layers.{N}.
        for k in keys:
            for prefix in ("model.layers.", "model.text_model.layers.", "layers."):
                if k.startswith(prefix):
                    rest = k[len(prefix):]
                    parts = rest.split(".", 1)
                    if parts[0].isdigit():
                        max_idx = max(max_idx, int(parts[0]))

    elif component == "vision":
        for k in keys:
            for prefix in (
                "model.vision_model.encoder.layers.",
                "vision_model.encoder.layers.",
                "visual.blocks.",
                "vision.encoder.layers.",
            ):
                if k.startswith(prefix):
                    rest = k[len(prefix):]
                    parts = rest.split(".", 1)
                    if parts[0].isdigit():
                        max_idx = max(max_idx, int(parts[0]))

    return max_idx + 1 if max_idx >= 0 else 0


def _infer_text_config(keys: dict[str, tuple[torch.Size, str]]) -> dict:
    """Infer text decoder config from weight shapes."""
    config = {}

    # Find embedding shape → (vocab_size, hidden_size)
    for k, (shape, _) in keys.items():
        if "embed_tokens" in k and "weight" in k and len(shape) == 2:
            config["vocab_size"] = shape[0]
            config["hidden_size"] = shape[1]
            break

    # Find Q/K/V projection shapes → num_heads, head_dim, num_kv_heads
    hidden_size = config.get("hidden_size", 0)
    for k, (shape, _) in keys.items():
        if ("self_attn.q_proj.weight" in k or "attention.wq.weight" in k) and len(shape) == 2:
            q_out = shape[0]
            if hidden_size:
                # head_dim typically is hidden_size // num_heads
                # Try common head dims: 64, 128
                for hd in (64, 128, 96, 80, 48, 32):
                    if q_out % hd == 0:
                        config["num_attention_heads"] = q_out // hd
                        config["head_dim"] = hd
                        break
            break

    for k, (shape, _) in keys.items():
        if ("self_attn.k_proj.weight" in k or "attention.wk.weight" in k) and len(shape) == 2:
            k_out = shape[0]
            hd = config.get("head_dim", 0)
            if hd and k_out % hd == 0:
                config["num_key_value_heads"] = k_out // hd
            break

    # Find MLP intermediate size
    for k, (shape, _) in keys.items():
        if ("mlp.gate_proj.weight" in k or "mlp.up_proj.weight" in k) and len(shape) == 2:
            config["intermediate_size"] = shape[0]
            break

    # Find lm_head → confirm vocab_size
    for k, (shape, _) in keys.items():
        if "lm_head.weight" in k and len(shape) == 2:
            config["vocab_size"] = shape[0]
            break

    return config


def _infer_vision_config(keys: dict[str, tuple[torch.Size, str]]) -> dict:
    """Infer vision encoder config from weight shapes."""
    config = {}

    # Patch embedding: Conv2d weight is [out_channels, in_channels, kH, kW]
    for k, (shape, _) in keys.items():
        if ("patch_embedding.weight" in k or "patch_embed.proj.weight" in k) and len(shape) == 4:
            config["hidden_size"] = shape[0]
            config["num_channels"] = shape[1]
            config["patch_size"] = shape[2]  # Assuming square patches
            break

    # Vision attention Q projection → num_heads
    vision_hidden = config.get("hidden_size", 0)
    for k, (shape, _) in keys.items():
        if any(p in k for p in ("vision_model.encoder.layers.0.self_attn.q_proj",
                                 "visual.blocks.0.attn.qkv")):
            if "qkv" in k:
                # Fused QKV: [3 * hidden, hidden]
                pass
            else:
                q_out = shape[0]
                if vision_hidden:
                    config["num_attention_heads"] = q_out // (vision_hidden // (q_out // 64 if q_out % 64 == 0 else 1))
            break

    # Position embedding → image size
    for k, (shape, _) in keys.items():
        if "position_embedding.weight" in k and len(shape) == 2:
            num_positions = shape[0]
            patch_size = config.get("patch_size", 16)
            patches_per_side = int(num_positions ** 0.5)
            config["image_size"] = patches_per_side * patch_size
            break

    # MLP intermediate size
    for k, (shape, _) in keys.items():
        if any(p in k for p in (
            "vision_model.encoder.layers.0.mlp.fc1.weight",
            "visual.blocks.0.mlp.fc1.weight",
        )):
            config["intermediate_size"] = shape[0]
            break

    return config


def _infer_connector_config(keys: dict[str, tuple[torch.Size, str]]) -> dict:
    """Infer connector config from weight shapes."""
    config = {}

    # Pixel shuffle connector (SmolVLM): modality_projection.0.weight [text_hidden, vision_hidden*s^2]
    for k, (shape, _) in keys.items():
        if "modality_projection.0.weight" in k or "connector.modality_projection.0.weight" in k:
            text_hidden = shape[0]
            shuffle_dim = shape[1]
            config["type"] = "pixel_shuffle"
            config["text_hidden_size"] = text_hidden
            # shuffle_dim = vision_hidden * scale_factor^2
            # Try common scale factors
            for sf in (4, 2, 3):
                if shuffle_dim % (sf * sf) == 0:
                    config["vision_hidden_size"] = shuffle_dim // (sf * sf)
                    config["scale_factor"] = sf
                    break
            break

    # Patch merger (Qwen2-VL): merger.mlp.0.weight
    for k, (shape, _) in keys.items():
        if "merger.mlp.0.weight" in k or "merger.mlp.1.weight" in k:
            if "type" not in config:
                config["type"] = "patch_merger"
            break

    return config


def _prod(shape) -> int:
    """Product of shape dimensions."""
    result = 1
    for s in shape:
        result *= s
    return result


# ---------------------------------------------------------------------------
# Config.json loading (for HuggingFace models)
# ---------------------------------------------------------------------------

def load_hf_config(model_path: str) -> Optional[dict]:
    """Load config.json from a HuggingFace model directory."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def extract_architecture_from_config(hf_config: dict) -> dict:
    """Extract UNFED architecture config from HuggingFace config.json.

    Handles nested configs (e.g., SmolVLM has text_config + vision_config).
    """
    result = {"text": {}, "vision": {}, "connector": {}}

    # Direct text config (Llama, Qwen2, Mistral)
    if "hidden_size" in hf_config:
        result["text"] = _extract_text_from_dict(hf_config)

    # Nested text config (SmolVLM/Idefics3)
    if "text_config" in hf_config:
        text_cfg = hf_config["text_config"]
        result["text"] = _extract_text_from_dict(text_cfg)

    # Vision config
    if "vision_config" in hf_config:
        vis_cfg = hf_config["vision_config"]
        # Handle different naming conventions across models:
        # SmolVLM/SigLIP:  hidden_size, num_attention_heads, intermediate_size, num_hidden_layers
        # Qwen2-VL:        embed_dim, num_heads, depth, in_chans, mlp_ratio
        hidden_size = vis_cfg.get("hidden_size") or vis_cfg.get("embed_dim")
        num_heads = vis_cfg.get("num_attention_heads") or vis_cfg.get("num_heads")
        intermediate_size = vis_cfg.get("intermediate_size")
        if intermediate_size is None and hidden_size and vis_cfg.get("mlp_ratio"):
            intermediate_size = hidden_size * vis_cfg["mlp_ratio"]
        num_layers = vis_cfg.get("num_hidden_layers") or vis_cfg.get("depth")
        patch_size = vis_cfg.get("patch_size") or vis_cfg.get("spatial_patch_size")
        num_channels = vis_cfg.get("num_channels") or vis_cfg.get("in_chans", 3)

        result["vision"] = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "intermediate_size": intermediate_size,
            "patch_size": patch_size,
            "image_size": vis_cfg.get("image_size"),
            "num_channels": num_channels,
            "num_layers": num_layers,
            "hidden_act": vis_cfg.get("hidden_act"),
            "layer_norm_eps": vis_cfg.get("layer_norm_eps", 1e-6),
        }
        # Qwen2-VL specific fields
        if vis_cfg.get("spatial_merge_size"):
            result["vision"]["spatial_merge_size"] = vis_cfg["spatial_merge_size"]
        if vis_cfg.get("temporal_patch_size"):
            result["vision"]["temporal_patch_size"] = vis_cfg["temporal_patch_size"]
        result["vision"] = {k: v for k, v in result["vision"].items() if v is not None}

    # Connector
    if "scale_factor" in hf_config:
        result["connector"]["scale_factor"] = hf_config["scale_factor"]
        result["connector"]["type"] = "pixel_shuffle"
        # Populate hidden sizes from text/vision configs for the connector
        if result["text"].get("hidden_size"):
            result["connector"]["text_hidden_size"] = result["text"]["hidden_size"]
        if result["vision"].get("hidden_size"):
            result["connector"]["vision_hidden_size"] = result["vision"]["hidden_size"]

    # Qwen2-VL uses spatial_merge_size in vision_config as its connector
    if "vision_config" in hf_config:
        vis_cfg = hf_config["vision_config"]
        if vis_cfg.get("spatial_merge_size") and not result["connector"]:
            result["connector"]["type"] = "patch_merger"
            result["connector"]["spatial_merge_size"] = vis_cfg["spatial_merge_size"]
            if result["vision"].get("hidden_size"):
                result["connector"]["vision_hidden_size"] = result["vision"]["hidden_size"]
            if result["text"].get("hidden_size"):
                result["connector"]["text_hidden_size"] = result["text"]["hidden_size"]

    return result


def _extract_text_from_dict(cfg: dict) -> dict:
    """Extract text decoder params from a config dict."""
    fields = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "hidden_act": "hidden_act",
        "rms_norm_eps": "rms_norm_eps",
        "rope_theta": "rope_theta",
        "vocab_size": "vocab_size",
        "max_position_embeddings": "max_position_embeddings",
        "num_layers": "num_hidden_layers",
    }
    result = {}
    for target, source in fields.items():
        if source in cfg:
            result[target] = cfg[source]
    return result


# ---------------------------------------------------------------------------
# Main inspection logic
# ---------------------------------------------------------------------------

def inspect_model(model_path: str, fmt: str = "auto", verbose: bool = False) -> dict:
    """Inspect a model and return a full analysis report.

    Args:
        model_path: Path to model directory or weight file.
        fmt: Format hint ("auto", "hf", "safetensors", "pt", "gguf").
        verbose: Include per-tensor details.

    Returns:
        Dict with format, architecture, configs, components, recommendations.
    """
    report = {
        "path": model_path,
        "format": "unknown",
        "architecture": {},
        "config_source": "none",
        "components": [],
        "recommendations": [],
        "weight_files": [],
    }

    # Detect format
    if fmt == "auto":
        fmt = detect_format(model_path)
    report["format"] = fmt

    p = Path(model_path)

    # List weight files
    if p.is_dir():
        for pattern in ("*.safetensors", "*.bin", "*.pt"):
            for f in sorted(p.glob(pattern)):
                size_mb = f.stat().st_size / (1024 * 1024)
                report["weight_files"].append({
                    "name": f.name,
                    "size_mb": round(size_mb, 1),
                })

    # Try loading config.json
    hf_config = None
    if fmt == "hf_dir":
        hf_config = load_hf_config(model_path)
        if hf_config:
            report["config_source"] = "config.json"
            report["architecture"] = extract_architecture_from_config(hf_config)
            report["model_type"] = hf_config.get("model_type", "unknown")
            report["architectures"] = hf_config.get("architectures", [])

    # Fill layer counts from config.json if available
    if hf_config:
        arch = report.get("architecture", {})
        if arch.get("text", {}).get("num_layers"):
            report["num_text_layers"] = arch["text"]["num_layers"]
        if arch.get("vision", {}).get("num_layers"):
            report["num_vision_layers"] = arch["vision"]["num_layers"]
        # Build components from config
        components = []
        if arch.get("text", {}).get("vocab_size"):
            components.append("embedding")
        if report.get("num_text_layers", 0) > 0:
            components.append(f"text_decoder ({report['num_text_layers']} layers)")
        if not hf_config.get("tie_word_embeddings", False):
            components.append("lm_head")
        elif hf_config.get("tie_word_embeddings"):
            components.append("lm_head (tied to embedding)")
        if report.get("num_vision_layers", 0) > 0:
            components.append(f"vision_encoder ({report['num_vision_layers']} layers)")
        if arch.get("connector"):
            components.append(f"connector ({arch['connector'].get('type', 'unknown')})")
        if components:
            report["components"] = components

    # Load weight keys for shape analysis
    all_keys = gather_all_weights(model_path, fmt)

    if all_keys:
        # Infer architecture from weights
        inferred = infer_architecture_from_keys(all_keys)

        # Merge: prefer config.json values, fill gaps from inference
        if not report["architecture"].get("text"):
            report["architecture"]["text"] = inferred["text"]
        else:
            # Fill any gaps
            for k, v in inferred["text"].items():
                if k not in report["architecture"]["text"]:
                    report["architecture"]["text"][k] = v

        if not report["architecture"].get("vision"):
            report["architecture"]["vision"] = inferred["vision"]

        if not report["architecture"].get("connector"):
            report["architecture"]["connector"] = inferred["connector"]

        report["num_text_layers"] = inferred["num_text_layers"]
        report["num_vision_layers"] = inferred["num_vision_layers"]
        report["total_params"] = inferred["total_params"]
        report["components"] = inferred["components"]
        if "model_type" not in report:
            report["model_type"] = inferred["model_type"]

    # Weight format analysis
    has_safetensors = any(f["name"].endswith(".safetensors")
                          for f in report["weight_files"])
    has_pt = any(f["name"].endswith((".bin", ".pt"))
                 for f in report["weight_files"])

    if all_keys:

        if verbose:
            report["weight_details"] = {
                k: {"shape": list(s), "dtype": d}
                for k, (s, d) in sorted(all_keys.items())
            }

    # Generate recommendations
    recs = []
    if not hf_config:
        recs.append("No config.json found — architecture inferred from weight shapes. "
                     "Verify the inferred values before splitting.")
    if report.get("num_vision_layers", 0) > 0:
        recs.append("Model has vision encoder — will need separate vision shards.")
    if has_pt and not has_safetensors:
        recs.append("Weights are in PyTorch pickle format. "
                     "Convert to safetensors for security: unfed convert <file> -o <out>.safetensors")
    total_params = report.get("total_params", 0)
    if total_params > 0:
        size_gb = total_params * 4 / (1024 ** 3)  # Assuming float32
        recs.append(f"Total parameters: {total_params:,} (~{size_gb:.1f} GB in float32)")
    report["recommendations"] = recs

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_inspect(args):
    """CLI handler for 'unfed inspect'."""
    report = inspect_model(args.model_path, args.format, args.verbose)
    _print_report(report, args.verbose)


def _print_report(report: dict, verbose: bool = False):
    """Pretty-print an inspection report."""
    print()
    print("=" * 60)
    print("  UNFED Model Inspector")
    print("=" * 60)
    print()
    print(f"  Path:       {report['path']}")
    print(f"  Format:     {report['format']}")
    print(f"  Model type: {report.get('model_type', 'unknown')}")
    if report.get("architectures"):
        print(f"  HF arch:    {', '.join(report['architectures'])}")
    print(f"  Config:     {report['config_source']}")
    print()

    if report["weight_files"]:
        print("  Weight files:")
        for f in report["weight_files"]:
            print(f"    {f['name']:40s}  {f['size_mb']:>8.1f} MB")
        print()

    if report["components"]:
        print("  Components:")
        for c in report["components"]:
            print(f"    - {c}")
        print()

    arch = report.get("architecture", {})
    if arch.get("text"):
        print("  Text decoder config:")
        for k, v in sorted(arch["text"].items()):
            print(f"    {k:30s}  {v}")
        print()

    if arch.get("vision"):
        print("  Vision encoder config:")
        for k, v in sorted(arch["vision"].items()):
            print(f"    {k:30s}  {v}")
        print()

    if arch.get("connector"):
        print("  Connector config:")
        for k, v in sorted(arch["connector"].items()):
            print(f"    {k:30s}  {v}")
        print()

    if report.get("recommendations"):
        print("  Recommendations:")
        for r in report["recommendations"]:
            print(f"    * {r}")
        print()

    if verbose and report.get("weight_details"):
        print("  Weight tensor details:")
        for k, info in report["weight_details"].items():
            shape_str = "x".join(str(s) for s in info["shape"])
            print(f"    {k:60s}  {shape_str:>20s}  {info['dtype']}")
        print()

    print("=" * 60)
