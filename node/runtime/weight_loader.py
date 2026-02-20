"""
UNFED Weight Loader — loads tensors from safetensors or PyTorch files
with key remapping support.

The manifest contains a ``weight_map`` that translates between the runtime's
expected key names and the actual keys stored in the weight file.  This
handles the fact that different models name their weights differently::

    Llama:   "model.layers.0.self_attn.q_proj.weight"
    Qwen2:   "model.layers.0.self_attn.q_proj.weight"
    Mistral: "model.layers.0.self_attn.q_proj.weight"
    MPT:     "transformer.blocks.0.attn.Wqkv.weight"   # fused QKV

The weight_map normalises everything into the UNFED standard key naming.

Formats supported:
  - safetensors (.safetensors) — preferred, no code execution risk
  - PyTorch (.pt / .bin)       — loaded with weights_only=True for safety
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight file loading
# ---------------------------------------------------------------------------

def load_weight_file(path: str) -> dict[str, torch.Tensor]:
    """Load a weight file and return a flat dict of {key: tensor}.

    Automatically detects format from file extension.

    Args:
        path: Path to .safetensors, .pt, or .bin file.

    Returns:
        Dict mapping weight key names to tensors.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        return _load_safetensors(path)
    elif ext in (".pt", ".pth", ".bin"):
        return _load_pytorch(path)
    else:
        raise ValueError(
            f"Unknown weight file format: '{ext}' for {path}. "
            f"Supported: .safetensors, .pt, .pth, .bin"
        )


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load weights from safetensors format (safe — no code execution)."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError(
            "safetensors package not installed. "
            "Install with: pip install safetensors"
        )
    return load_file(path, device="cpu")


def _load_pytorch(path: str) -> dict[str, torch.Tensor]:
    """Load weights from PyTorch format (uses weights_only=True for safety)."""
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

def remap_keys(
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Remap weight keys using a weight_map.

    The weight_map maps from *target* key (what the runtime expects)
    to *source* key (what's in the file)::

        weight_map = {
            "embed_tokens.weight": "model.embed_tokens.weight",
            "layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
        }

    Supports pattern templates with ``{idx}`` for layer indices::

        weight_map = {
            "layers.{idx}.self_attn.q_proj.weight": "model.layers.{idx}.self_attn.q_proj.weight"
        }

    Args:
        state_dict: Raw state dict from the weight file.
        weight_map: Mapping from target keys → source keys.

    Returns:
        New state dict with remapped keys.
    """
    if not weight_map:
        return state_dict

    # Expand templates: find all {idx} patterns and resolve them
    expanded_map: dict[str, str] = {}
    for target_pattern, source_pattern in weight_map.items():
        if "{idx}" in target_pattern:
            # Find all matching layer indices from the source dict
            for source_key in state_dict:
                idx = _extract_idx(source_key, source_pattern)
                if idx is not None:
                    concrete_target = target_pattern.replace("{idx}", str(idx))
                    concrete_source = source_pattern.replace("{idx}", str(idx))
                    expanded_map[concrete_target] = concrete_source
        else:
            expanded_map[target_pattern] = source_pattern

    # Apply remapping
    remapped = {}
    reverse_map = {v: k for k, v in expanded_map.items()}

    for source_key, tensor in state_dict.items():
        if source_key in reverse_map:
            remapped[reverse_map[source_key]] = tensor
        else:
            # Keep unmapped keys as-is
            remapped[source_key] = tensor

    return remapped


def _extract_idx(key: str, pattern: str) -> Optional[int]:
    """Try to extract the {idx} value from a key matching a pattern.

    Example:
        key = "model.layers.5.self_attn.q_proj.weight"
        pattern = "model.layers.{idx}.self_attn.q_proj.weight"
        → returns 5
    """
    parts_key = key.split(".")
    parts_pat = pattern.split(".")

    if len(parts_key) != len(parts_pat):
        return None

    idx = None
    for pk, pp in zip(parts_key, parts_pat):
        if pp == "{idx}":
            if pk.isdigit():
                idx = int(pk)
            else:
                return None
        elif pk != pp:
            return None
    return idx


# ---------------------------------------------------------------------------
# Load into module
# ---------------------------------------------------------------------------

def load_weights_into_module(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
    strict: bool = False,
) -> list[str]:
    """Load weights from a state dict into a PyTorch module.

    Handles prefix stripping and reports any missing/unexpected keys.

    Args:
        module: Target PyTorch module.
        state_dict: Weight tensors.
        prefix: Optional prefix to strip from keys before loading.
        strict: If True, raise on missing/unexpected keys.

    Returns:
        List of keys that were loaded.
    """
    if prefix:
        # Strip prefix from keys
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            # Also try without the prefix (might already be stripped)
            elif k in dict(module.named_parameters()) or k in dict(module.named_buffers()):
                stripped[k] = v
        state_dict = stripped

    result = module.load_state_dict(state_dict, strict=strict)

    loaded_keys = [
        k for k in state_dict
        if k not in (result.unexpected_keys if hasattr(result, 'unexpected_keys') else [])
    ]
    return loaded_keys


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_shard_weights(
    shard_path: str,
    weight_map: Optional[dict[str, str]] = None,
) -> dict[str, torch.Tensor]:
    """Load and remap weights from a shard file.

    Convenience function combining load + remap.

    Args:
        shard_path: Path to the shard weight file.
        weight_map: Optional key remapping dict.

    Returns:
        Remapped state dict ready for loading into runtime modules.
    """
    state_dict = load_weight_file(shard_path)

    if weight_map:
        state_dict = remap_keys(state_dict, weight_map)

    return state_dict
