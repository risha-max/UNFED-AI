"""
UNFED Generic Runners — platform-agnostic text and vision model execution.

GenericTextRunner handles text decoder (autoregressive) execution.
GenericVisionRunner handles vision encoder (single-pass) execution.

Both assemble their architecture from a declarative config dictionary
and load weights from safetensors/pt files — zero HuggingFace imports.

Interface contracts (must match existing runners for server compatibility):

  GenericTextRunner.forward(
      hidden_states, token_ids, session_id, prefix_session_id,
      prefix_length, image_embeddings, mrope_position_ids
  ) → (output_tensor, sampled_token_id)

  GenericVisionRunner.forward(
      pixel_values, hidden_states, patch_attention_mask
  ) → {"hidden_states": tensor} or {"image_features": tensor}
"""

from __future__ import annotations

import math
import os
import time
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from node.runtime.text_blocks import (
    UnfedRMSNorm, UnfedRoPE, UnfedAttention, UnfedMLP,
    UnfedDecoderLayer, UnfedEmbedding, UnfedLMHead,
)
from node.runtime.vision_blocks import (
    UnfedPatchEmbedding, UnfedVisionEncoderLayer,
    UnfedPixelShuffleConnector, UnfedPatchMerger,
)
from node.runtime.weight_loader import load_shard_weights, load_weights_into_module


# ---------------------------------------------------------------------------
# Session KV Cache — lightweight, no transformers dependency
# ---------------------------------------------------------------------------

class SessionCache:
    """Per-session KV cache for autoregressive decoding.

    Stores pre-allocated KV tensors for each decoder layer this shard owns.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seq_len = 0  # current filled length
        self.last_used = time.time()

        # Pre-allocate: [batch=1, num_kv_heads, max_seq_len, head_dim]
        self.k_caches = [
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(1, num_kv_heads, max_seq_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

    def touch(self):
        self.last_used = time.time()

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (k_cache, v_cache) for a layer, sliced to current seq_len."""
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def advance(self, num_new_tokens: int):
        """Advance the sequence position counter."""
        self.seq_len += num_new_tokens
        self.touch()

    def clone_prefix(self, prefix_length: int) -> "SessionCache":
        """Create a new SessionCache with the first prefix_length tokens copied."""
        new_cache = SessionCache(
            num_layers=self.num_layers,
            num_kv_heads=self.k_caches[0].shape[1],
            head_dim=self.k_caches[0].shape[3],
            max_seq_len=self.max_seq_len,
            device=self.k_caches[0].device,
            dtype=self.k_caches[0].dtype,
        )
        for i in range(self.num_layers):
            new_cache.k_caches[i][:, :, :prefix_length] = \
                self.k_caches[i][:, :, :prefix_length].clone()
            new_cache.v_caches[i][:, :, :prefix_length] = \
                self.v_caches[i][:, :, :prefix_length].clone()
        new_cache.seq_len = prefix_length
        return new_cache


class SessionCacheManager:
    """Manages per-session KV caches for the generic text runner."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        self._sessions: dict[str, SessionCache] = {}
        self._lock = threading.Lock()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

    def get_or_create(self, session_id: str) -> SessionCache:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionCache(
                    num_layers=self.num_layers,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    max_seq_len=self.max_seq_len,
                    device=self.device,
                    dtype=self.dtype,
                )
            cache = self._sessions[session_id]
            cache.touch()
            return cache

    def clone_prefix(
        self, source_id: str, target_id: str, prefix_length: int
    ) -> bool:
        with self._lock:
            source = self._sessions.get(source_id)
            if source is None or source.seq_len < prefix_length:
                return False
            self._sessions[target_id] = source.clone_prefix(prefix_length)
            return True

    def clear_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_stale(self, timeout_seconds: float) -> int:
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, c in self._sessions.items()
                if now - c.last_used > timeout_seconds
            ]
            for sid in stale:
                del self._sessions[sid]
        return len(stale)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "total_tokens": sum(c.seq_len for c in self._sessions.values()),
            }


# ---------------------------------------------------------------------------
# GenericTextRunner
# ---------------------------------------------------------------------------

class GenericTextRunner(nn.Module):
    """Generic text decoder runner — model-agnostic.

    Assembles decoder layers, embedding, and LM head from config + weights.
    Manages KV cache per session for autoregressive generation.

    Args:
        config: Architecture config dict with keys:
            hidden_size, num_attention_heads, num_key_value_heads,
            intermediate_size, hidden_act, rms_norm_eps, rope_theta,
            vocab_size, max_position_embeddings
        shard_info: Shard metadata dict with keys:
            layer_start, layer_end, has_embedding, has_lm_head
        shard_path: Path to the weight file for this shard.
        weight_map: Optional key remapping dict.
        image_token_id: Token ID for image placeholder (VL models).
        device: Compute device.
    """

    def __init__(
        self,
        config: dict,
        shard_info: dict,
        shard_path: str,
        weight_map: Optional[dict[str, str]] = None,
        image_token_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.image_token_id = image_token_id

        # Shard metadata
        self.layer_start = shard_info["layer_start"]
        self.layer_end = shard_info["layer_end"]
        self.has_embedding = shard_info.get("has_embedding", False)
        self.has_lm_head = shard_info.get("has_lm_head", False) or shard_info.get("has_head", False)
        self.num_layers = self.layer_end - self.layer_start
        self.is_first = self.has_embedding
        self.is_last = self.has_lm_head

        # Architecture params
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        eps = config.get("rms_norm_eps", 1e-5)
        rope_theta = config.get("rope_theta", 10000.0)
        max_pos = config.get("max_position_embeddings", 8192)
        vocab_size = config["vocab_size"]

        # Build components
        self.embed = None
        if self.has_embedding:
            self.embed = UnfedEmbedding(vocab_size, hidden_size)

        self.rope = UnfedRoPE(head_dim, rope_theta, max_pos)

        self.layers = nn.ModuleList([
            UnfedDecoderLayer(config) for _ in range(self.num_layers)
        ])

        self.final_norm = None
        self.lm_head = None
        if self.has_lm_head:
            self.final_norm = UnfedRMSNorm(hidden_size, eps)
            self.lm_head = UnfedLMHead(hidden_size, vocab_size)

        # Load weights
        self._load_weights(shard_path, weight_map)
        self.to(device)
        self.eval()

        # KV cache manager
        self.cache_manager = SessionCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_pos,
            device=device,
            dtype=torch.float32,
        )

        print(f"[GenericTextRunner] Loaded shard: layers [{self.layer_start}, {self.layer_end}), "
              f"embed={self.has_embedding}, lm_head={self.has_lm_head}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def _load_weights(self, shard_path: str, weight_map: Optional[dict] = None):
        """Load weights from shard file into the runtime modules."""
        state_dict = load_shard_weights(shard_path, weight_map)

        # Map weights to components
        loaded_count = 0
        remaining = dict(state_dict)

        # Load embedding weights
        if self.embed is not None:
            embed_keys = {k: v for k, v in remaining.items() if "embed_tokens" in k}
            if embed_keys:
                # Remap: "embed_tokens.weight" → module expects it directly
                embed_sd = {}
                for k, v in embed_keys.items():
                    # Normalize key to what UnfedEmbedding expects
                    new_k = k.split("embed_tokens.")[-1] if "embed_tokens." in k else k
                    embed_sd[f"embed_tokens.{new_k}" if not new_k.startswith("embed_tokens") else new_k] = v
                load_weights_into_module(self.embed, embed_sd, strict=False)
                for k in embed_keys:
                    remaining.pop(k, None)
                loaded_count += len(embed_keys)

        # Load layer weights
        for local_idx in range(self.num_layers):
            global_idx = self.layer_start + local_idx
            layer_prefix = f"layers.{global_idx}."
            # Also try model.layers.{idx}. prefix
            alt_prefix = f"model.layers.{global_idx}."

            layer_sd = {}
            keys_used = []
            for k, v in remaining.items():
                local_key = None
                if k.startswith(layer_prefix):
                    local_key = k[len(layer_prefix):]
                elif k.startswith(alt_prefix):
                    local_key = k[len(alt_prefix):]
                if local_key is not None:
                    layer_sd[local_key] = v
                    keys_used.append(k)

            if layer_sd:
                load_weights_into_module(self.layers[local_idx], layer_sd, strict=False)
                for k in keys_used:
                    remaining.pop(k, None)
                loaded_count += len(keys_used)

        # Load final norm + LM head
        if self.final_norm is not None:
            norm_keys = {k: v for k, v in remaining.items() if "norm.weight" in k and "layernorm" not in k}
            if norm_keys:
                for k, v in norm_keys.items():
                    self.final_norm.weight = nn.Parameter(v)
                    remaining.pop(k, None)
                    loaded_count += 1
                    break

        if self.lm_head is not None:
            head_keys = {k: v for k, v in remaining.items() if "lm_head" in k}
            if head_keys:
                head_sd = {}
                for k, v in head_keys.items():
                    local_k = k.split("lm_head.")[-1] if "lm_head." in k else k
                    head_sd[f"lm_head.{local_k}" if not local_k.startswith("lm_head") else local_k] = v
                load_weights_into_module(self.lm_head, head_sd, strict=False)
                for k in head_keys:
                    remaining.pop(k, None)
                loaded_count += len(head_keys)

        if remaining:
            print(f"[GenericTextRunner] Warning: {len(remaining)} unmapped weight keys: "
                  f"{list(remaining.keys())[:5]}...")

        print(f"[GenericTextRunner] Loaded {loaded_count} weight tensors from {shard_path}")

    def merge_image_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Replace image placeholder tokens with actual image embeddings.

        Args:
            text_embeds: [batch, seq_len, hidden] — token embeddings
            image_embeds: [num_image_tokens, hidden] — vision features
            token_ids: [batch, seq_len] — token IDs (to find placeholders)

        Returns:
            [batch, seq_len, hidden] — merged embeddings
        """
        if self.image_token_id is None:
            return text_embeds

        mask = (token_ids == self.image_token_id)
        num_placeholders = mask.sum().item()
        num_features = image_embeds.shape[0]

        if num_placeholders == 0:
            return text_embeds

        if num_placeholders != num_features:
            print(f"[GenericTextRunner] Warning: {num_placeholders} <image> tokens "
                  f"but {num_features} image features — truncating/padding")
            if num_features > num_placeholders:
                image_embeds = image_embeds[:num_placeholders]
            else:
                # Pad with zeros
                pad = torch.zeros(
                    num_placeholders - num_features,
                    image_embeds.shape[1],
                    device=image_embeds.device,
                    dtype=image_embeds.dtype,
                )
                image_embeds = torch.cat([image_embeds, pad], dim=0)

        # Replace in-place using masked_scatter
        result = text_embeds.clone()
        result[mask] = image_embeds.to(result.dtype)
        return result

    @torch.no_grad()
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        session_id: str = "default",
        prefix_session_id: str = "",
        prefix_length: int = 0,
        image_embeddings: Optional[torch.Tensor] = None,
        mrope_position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Run text decoder forward pass.

        For the first shard (has_embedding=True):
            - Takes token_ids, runs embedding + layers
            - Optionally merges image_embeddings
        For intermediate shards:
            - Takes hidden_states, runs layers
        For the last shard (has_lm_head=True):
            - Runs final norm + LM head, samples token

        Args:
            hidden_states: [batch, seq_len, hidden] — from previous shard
            token_ids: [batch, seq_len] — token IDs (first shard only)
            session_id: Session identifier for KV cache
            prefix_session_id: Source session for prefix cloning
            prefix_length: Number of tokens to clone from prefix session
            image_embeddings: [num_tokens, hidden] — vision features (VL)
            mrope_position_ids: Not used (Qwen2-VL specific, kept for compat)

        Returns:
            (output_tensor, sampled_token_id)
            - output_tensor: hidden states or logits
            - sampled_token_id: generated token (last shard only), else None
        """
        # Handle prefix cloning
        if prefix_session_id and prefix_length > 0:
            self.cache_manager.clone_prefix(
                prefix_session_id, session_id, prefix_length
            )

        # Get session cache
        session_cache = self.cache_manager.get_or_create(session_id)
        current_pos = session_cache.seq_len

        # First shard: embedding
        if self.is_first and token_ids is not None:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
            hidden_states = self.embed(token_ids.to(self.device))

            # Merge image embeddings if provided
            if image_embeddings is not None:
                hidden_states = self.merge_image_embeddings(
                    hidden_states, image_embeddings.to(self.device), token_ids
                )

        if hidden_states is None:
            raise ValueError("GenericTextRunner: need either token_ids (first shard) "
                             "or hidden_states (subsequent shard)")

        hidden_states = hidden_states.to(self.device)
        batch, seq_len, _ = hidden_states.shape

        # Position IDs for RoPE
        position_ids = torch.arange(
            current_pos, current_pos + seq_len,
            device=self.device
        ).unsqueeze(0).expand(batch, -1)

        # Compute RoPE embeddings
        cos, sin = self.rope(hidden_states, position_ids)

        # Cache positions for scatter update
        cache_position = torch.arange(
            current_pos, current_pos + seq_len,
            device=self.device
        )

        # Causal attention mask (only needed for prefill, not single-token decode)
        attention_mask = None
        if seq_len > 1:
            total_len = current_pos + seq_len
            causal_mask = torch.zeros(seq_len, total_len, device=self.device)
            new_block = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device), diagonal=1
            )
            causal_mask[:, current_pos:] = new_block.masked_fill(new_block == 1, float("-inf"))
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Run through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache, v_cache = session_cache.get_layer_cache(layer_idx)
            hidden_states, (new_k, new_v) = layer(
                hidden_states,
                position_embeddings=(cos, sin),
                past_key_value=(k_cache, v_cache),
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        # Advance cache position
        session_cache.advance(seq_len)

        # Last shard: final norm + LM head + sampling
        sampled_token = None
        if self.is_last:
            hidden_states = self.final_norm(hidden_states)
            logits = self.lm_head(hidden_states)
            # Greedy sample from last position
            sampled_token = int(logits[0, -1].argmax(dim=-1).item())

        return hidden_states, sampled_token

    def clear_session(self, session_id: str):
        """Remove a session's KV cache."""
        self.cache_manager.clear_session(session_id)

    def cleanup_stale_sessions(self, timeout: float = 300.0) -> int:
        """Remove inactive sessions."""
        return self.cache_manager.cleanup_stale(timeout)

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return self.cache_manager.get_stats()


# ---------------------------------------------------------------------------
# GenericVisionRunner
# ---------------------------------------------------------------------------

class GenericVisionRunner(nn.Module):
    """Generic vision encoder runner — model-agnostic.

    Assembles vision encoder layers, patch embedding, and connector
    from config + weights.  Single-pass (no KV cache).

    Args:
        vision_config: Vision architecture config dict with keys:
            hidden_size, num_attention_heads, intermediate_size,
            hidden_act, layer_norm_eps, patch_size, image_size, num_channels
        connector_config: Optional connector config dict with keys:
            type ("pixel_shuffle" or "patch_merger"),
            scale_factor, vision_hidden_size, text_hidden_size
        shard_info: Shard metadata dict with keys:
            layer_start, layer_end, has_embeddings, has_connector
        shard_path: Path to the weight file.
        weight_map: Optional key remapping dict.
        device: Compute device.
    """

    def __init__(
        self,
        vision_config: dict,
        connector_config: Optional[dict],
        shard_info: dict,
        shard_path: str,
        weight_map: Optional[dict[str, str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.vision_config = vision_config
        self.device = device

        # Shard metadata
        self.layer_start = shard_info["layer_start"]
        self.layer_end = shard_info["layer_end"]
        self.has_embeddings = shard_info.get("has_embeddings", False)
        self.has_connector = shard_info.get("has_connector", False)
        self.num_layers = self.layer_end - self.layer_start

        # Architecture params
        hidden_size = vision_config["hidden_size"]
        eps = vision_config.get("layer_norm_eps", 1e-6)
        self.patch_size = vision_config.get("patch_size", 16)

        # Build components
        self.embeddings = None
        if self.has_embeddings:
            self.embeddings = UnfedPatchEmbedding(vision_config)

        self.encoder_layers = nn.ModuleList([
            UnfedVisionEncoderLayer(vision_config)
            for _ in range(self.num_layers)
        ])

        self.post_layernorm = None
        self.connector = None
        if self.has_connector:
            self.post_layernorm = nn.LayerNorm(hidden_size, eps=eps)
            if connector_config:
                conn_type = connector_config.get("type", "pixel_shuffle")
                if conn_type == "pixel_shuffle":
                    self.connector = UnfedPixelShuffleConnector(connector_config)
                elif conn_type == "patch_merger":
                    self.connector = UnfedPatchMerger(connector_config)
                else:
                    raise ValueError(f"Unknown connector type: {conn_type}")

        # Load weights
        self._load_weights(shard_path, weight_map)
        self.to(device)
        self.eval()

        print(f"[GenericVisionRunner] Loaded shard: layers [{self.layer_start}, {self.layer_end}), "
              f"embeddings={self.has_embeddings}, connector={self.has_connector}, "
              f"params={sum(p.numel() for p in self.parameters()):,}")

    def _load_weights(self, shard_path: str, weight_map: Optional[dict] = None):
        """Load weights from shard file into the runtime modules."""
        state_dict = load_shard_weights(shard_path, weight_map)

        loaded_count = 0
        remaining = dict(state_dict)

        # Load patch embedding weights
        if self.embeddings is not None:
            embed_keys = {k: v for k, v in remaining.items()
                          if "embeddings." in k or "patch_embedding" in k or "position_embedding" in k}
            if embed_keys:
                embed_sd = {}
                for k, v in embed_keys.items():
                    # Normalize to UnfedPatchEmbedding's expected names
                    local_k = k
                    # Strip common prefixes
                    for prefix in ("vision_model.embeddings.", "embeddings.", "vision.embeddings."):
                        if local_k.startswith(prefix):
                            local_k = local_k[len(prefix):]
                            break
                    embed_sd[local_k] = v
                load_weights_into_module(self.embeddings, embed_sd, strict=False)
                for k in embed_keys:
                    remaining.pop(k, None)
                loaded_count += len(embed_keys)

        # Load encoder layer weights
        for local_idx in range(self.num_layers):
            global_idx = self.layer_start + local_idx
            # Try various naming conventions
            prefixes = [
                f"encoder.layers.{global_idx}.",
                f"vision_model.encoder.layers.{global_idx}.",
                f"layers.{global_idx}.",
            ]

            layer_sd = {}
            keys_used = []
            for k, v in remaining.items():
                for prefix in prefixes:
                    if k.startswith(prefix):
                        local_key = k[len(prefix):]
                        layer_sd[local_key] = v
                        keys_used.append(k)
                        break

            if layer_sd:
                load_weights_into_module(
                    self.encoder_layers[local_idx], layer_sd, strict=False
                )
                for k in keys_used:
                    remaining.pop(k, None)
                loaded_count += len(keys_used)

        # Load post-layernorm
        if self.post_layernorm is not None:
            norm_keys = {k: v for k, v in remaining.items()
                         if "post_layernorm" in k or "layernorm_post" in k
                         or "post_layer_norm" in k}
            if norm_keys:
                norm_sd = {}
                for k, v in norm_keys.items():
                    if "weight" in k:
                        norm_sd["weight"] = v
                    elif "bias" in k:
                        norm_sd["bias"] = v
                load_weights_into_module(self.post_layernorm, norm_sd, strict=False)
                for k in norm_keys:
                    remaining.pop(k, None)
                loaded_count += len(norm_keys)

        # Load connector weights
        if self.connector is not None:
            conn_keys = {k: v for k, v in remaining.items()
                         if "connector" in k or "multi_modal_projector" in k
                         or "modality_projection" in k}
            if conn_keys:
                conn_sd = {}
                for k, v in conn_keys.items():
                    local_k = k
                    for prefix in ("connector.", "multi_modal_projector.",
                                   "modality_projection."):
                        if local_k.startswith(prefix):
                            local_k = local_k[len(prefix):]
                            break
                    # Map to our connector's naming
                    if isinstance(self.connector, UnfedPixelShuffleConnector):
                        conn_sd[f"modality_projection.{local_k}"
                                if not local_k.startswith("modality_projection") else local_k] = v
                    else:
                        conn_sd[f"mlp.{local_k}"
                                if not local_k.startswith("mlp") else local_k] = v
                load_weights_into_module(self.connector, conn_sd, strict=False)
                for k in conn_keys:
                    remaining.pop(k, None)
                loaded_count += len(conn_keys)

        if remaining:
            print(f"[GenericVisionRunner] Warning: {len(remaining)} unmapped keys: "
                  f"{list(remaining.keys())[:5]}...")

        print(f"[GenericVisionRunner] Loaded {loaded_count} weight tensors from {shard_path}")

    @torch.no_grad()
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        patch_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run vision encoder forward pass.

        For the first shard (has_embeddings=True):
            - Takes pixel_values, runs patch embedding + encoder layers
        For intermediate shards:
            - Takes hidden_states, runs encoder layers
        For the last shard (has_connector=True):
            - Runs post-layernorm + connector, returns image_features

        Args:
            pixel_values: [num_tiles, channels, height, width]
            hidden_states: [num_tiles, num_patches, hidden] — from previous shard
            patch_attention_mask: [num_tiles, patches_h, patches_w] — bool mask

        Returns:
            Dict with either:
                {"hidden_states": tensor} — intermediate shard output
                {"image_features": tensor} — final features after connector
        """
        # Build attention mask for padded patches
        attention_mask = None

        # First shard: patch embedding
        if self.has_embeddings:
            if pixel_values is None:
                raise ValueError("First vision shard requires pixel_values")

            pixel_values = pixel_values.to(self.device)
            batch_size = pixel_values.shape[0]

            # Default: all patches are real
            if patch_attention_mask is None:
                patches_h = pixel_values.shape[2] // self.patch_size
                patches_w = pixel_values.shape[3] // self.patch_size
                patch_attention_mask = torch.ones(
                    batch_size, patches_h, patches_w,
                    dtype=torch.bool, device=self.device,
                )
            else:
                patch_attention_mask = patch_attention_mask.to(self.device)

            hidden_states = self.embeddings(pixel_values, patch_attention_mask)

            # Build attention mask from patch_attention_mask
            # [B, ph, pw] → [B, num_patches]
            flat_mask = patch_attention_mask.view(batch_size, -1)  # [B, N]
            # Convert to SDPA format: [B, 1, N, N] additive mask
            # Real patches = 0.0 (attend), padded = -inf (block)
            # Use torch.where to avoid 0.0 * -inf = NaN
            mask_1d = torch.where(
                flat_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, N]
                torch.tensor(0.0, device=self.device),
                torch.tensor(float("-inf"), device=self.device),
            )
            # Expand to [B, 1, N, N] for full bidirectional
            attention_mask = mask_1d.expand(-1, -1, flat_mask.shape[1], -1)

        if hidden_states is None:
            raise ValueError("Vision shard requires pixel_values or hidden_states")

        hidden_states = hidden_states.to(self.device)

        # Run through encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # Last shard: post-layernorm + connector
        if self.has_connector:
            if self.post_layernorm is not None:
                hidden_states = self.post_layernorm(hidden_states)
            if self.connector is not None:
                image_features = self.connector(hidden_states)
                return {"image_features": image_features}

        return {"hidden_states": hidden_states}
