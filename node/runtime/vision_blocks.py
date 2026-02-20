"""
UNFED Vision Encoder Building Blocks — pure PyTorch, no external model libraries.

Vision encoders (SigLIP, CLIP, ViT) are built from similar primitives as text
decoders but with key differences:
  - Bidirectional attention (no causal mask)
  - Patch embedding (Conv2D) instead of token embedding
  - Learned position embeddings (not RoPE) for most vision encoders
  - Single-pass (no KV cache — not autoregressive)
  - Connector/projector to bridge vision → text hidden dimensions

Components:
  UnfedPatchEmbedding     — Conv2D patch projection + learned position embedding
  UnfedVisionAttention    — Bidirectional multi-head attention (no RoPE, no causal)
  UnfedVisionMLP          — Standard MLP (GELU, not gated)
  UnfedVisionEncoderLayer — norm → attn → residual → norm → mlp → residual
  UnfedPixelShuffleConnector — Pixel shuffle + MLP projection (vision → text dim)

Usage:
    vision_config = {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu_pytorch_tanh",
        "layer_norm_eps": 1e-6,
        "patch_size": 16,
        "image_size": 512,
        "num_channels": 3,
    }
    layer = UnfedVisionEncoderLayer(vision_config)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from node.runtime.text_blocks import get_activation


# ---------------------------------------------------------------------------
# UnfedPatchEmbedding — Conv2D + Learned Position Embeddings
# ---------------------------------------------------------------------------

class UnfedPatchEmbedding(nn.Module):
    """Patch embedding for vision transformers.

    Splits an image into non-overlapping patches using Conv2D, then adds
    learned position embeddings.

    Config params:
        hidden_size: int — output embedding dimension
        patch_size: int — size of each square patch (e.g. 16)
        image_size: int — expected image size (e.g. 512)
        num_channels: int — input channels (default 3)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.patch_size = config["patch_size"]
        self.image_size = config["image_size"]
        self.num_channels = config.get("num_channels", 3)

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2

        # Conv2D projects each patch to hidden_size
        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        # Learned position embeddings
        self.position_embedding = nn.Embedding(self.num_patches, self.hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [num_tiles, channels, height, width]
            patch_attention_mask: [num_tiles, patches_h, patches_w] — bool mask
                indicating which patches are real (not padding).

        Returns:
            [num_tiles, num_patches, hidden_size] patch embeddings
        """
        batch_size = pixel_values.shape[0]

        # Conv2D: [B, C, H, W] → [B, hidden, patches_h, patches_w]
        embeddings = self.patch_embedding(pixel_values)

        # Flatten spatial: [B, hidden, ph, pw] → [B, num_patches, hidden]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        # Add position embeddings
        position_ids = torch.arange(self.num_patches, device=embeddings.device)
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


# ---------------------------------------------------------------------------
# UnfedVisionAttention — Bidirectional Multi-Head Attention
# ---------------------------------------------------------------------------

class UnfedVisionAttention(nn.Module):
    """Bidirectional multi-head attention for vision encoders.

    Key differences from text attention:
      - No causal mask (bidirectional — each patch attends to all patches)
      - No RoPE (uses learned position embeddings from PatchEmbedding)
      - No KV cache (single-pass, not autoregressive)

    Config params:
        hidden_size: int
        num_attention_heads: int
    """

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: optional mask for padded patches

        Returns:
            [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Bidirectional attention (no causal mask)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(attn_output)


# ---------------------------------------------------------------------------
# UnfedVisionMLP — Standard MLP (not gated)
# ---------------------------------------------------------------------------

class UnfedVisionMLP(nn.Module):
    """Standard MLP for vision encoder layers.

    Most vision encoders use a simple up → activation → down pattern
    (not gated SwiGLU like text decoders).

    Config params:
        hidden_size: int
        intermediate_size: int
        hidden_act: str (default "gelu_pytorch_tanh")
    """

    def __init__(self, config: dict):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=True)
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=True)
        self.act_fn = get_activation(config.get("hidden_act", "gelu_pytorch_tanh"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


# ---------------------------------------------------------------------------
# UnfedVisionEncoderLayer
# ---------------------------------------------------------------------------

class UnfedVisionEncoderLayer(nn.Module):
    """Vision transformer encoder layer.

    Structure (pre-norm):
        residual = hidden_states
        hidden_states = layer_norm1(hidden_states)
        hidden_states = self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer_norm2(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states

    Config params:
        hidden_size, num_attention_heads, intermediate_size, hidden_act,
        layer_norm_eps (default 1e-6)
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config.get("layer_norm_eps", 1e-6)

        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = UnfedVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = UnfedVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            [batch, seq_len, hidden_size] — output hidden states
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# UnfedPixelShuffleConnector — Vision → Text Projection
# ---------------------------------------------------------------------------


class _SimpleMLP(nn.Module):
    """Single-layer linear projection matching Idefics3SimpleMLP naming."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class UnfedPixelShuffleConnector(nn.Module):
    """Pixel shuffle connector that projects vision features to text dimension.

    Used by SmolVLM/Idefics3.  Takes vision encoder output and:
      1. Pixel shuffle: reduces spatial resolution by scale_factor^2, increasing
         channel depth (reduces token count by scale_factor^2)
      2. MLP: projects from (vision_hidden * scale_factor^2) to text_hidden

    Config params:
        scale_factor: int — spatial downsampling factor (e.g. 4)
        vision_hidden_size: int — input dimension from vision encoder
        text_hidden_size: int — output dimension for text decoder
        mlp_depth: int — number of linear layers (1 for SmolVLM, 2 for others)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.scale_factor = config["scale_factor"]
        vision_hidden = config["vision_hidden_size"]
        text_hidden = config["text_hidden_size"]
        mlp_depth = config.get("mlp_depth", 1)

        # After pixel shuffle: channel dim = vision_hidden * scale_factor^2
        shuffle_dim = vision_hidden * (self.scale_factor ** 2)

        if mlp_depth == 1:
            # Single-layer projection (SmolVLM/Idefics3SimpleMLP)
            self.modality_projection = _SimpleMLP(shuffle_dim, text_hidden)
        else:
            # Two-layer MLP: shuffle_dim → text_hidden → text_hidden
            self.modality_projection = nn.Sequential(
                nn.Linear(shuffle_dim, text_hidden, bias=False),
                nn.GELU(),
                nn.Linear(text_hidden, text_hidden, bias=False),
            )

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_hidden_states: [num_tiles, num_patches, vision_hidden]
                Output from the vision encoder.

        Returns:
            [num_tiles, reduced_patches, text_hidden]
                Reduced token count by scale_factor^2, projected to text dim.
        """
        batch, seq_len, hidden = image_hidden_states.shape

        # Compute spatial dimensions for pixel shuffle
        # num_patches = patches_h * patches_w, assume square
        patches_per_side = int(math.isqrt(seq_len))
        if patches_per_side * patches_per_side != seq_len:
            raise ValueError(
                f"Pixel shuffle requires square patch grid, got {seq_len} patches "
                f"(sqrt={math.sqrt(seq_len):.2f})"
            )

        # Reshape to spatial: [B, patches_h, patches_w, hidden]
        h = image_hidden_states.view(batch, patches_per_side, patches_per_side, hidden)

        # Pixel shuffle: group adjacent patches
        # [B, pH, pW, C] → [B, pH/s, pW/s, C*s*s]
        s = self.scale_factor
        new_h = patches_per_side // s
        new_w = patches_per_side // s

        h = h.view(batch, new_h, s, new_w, s, hidden)
        h = h.permute(0, 1, 3, 2, 4, 5)  # [B, new_h, new_w, s, s, C]
        h = h.reshape(batch, new_h * new_w, hidden * s * s)  # [B, reduced, C*s*s]

        # MLP projection: [B, reduced, shuffle_dim] → [B, reduced, text_hidden]
        return self.modality_projection(h)


# ---------------------------------------------------------------------------
# UnfedPatchMerger — Vision → Text Projection (Qwen2-VL style)
# ---------------------------------------------------------------------------

class UnfedPatchMerger(nn.Module):
    """Patch merger connector used by Qwen2-VL.

    Merges spatial_merge_size^2 adjacent patches, then projects to text dim.

    Config params:
        spatial_merge_size: int (e.g. 2)
        vision_hidden_size: int
        text_hidden_size: int
    """

    def __init__(self, config: dict):
        super().__init__()
        self.merge_size = config.get("spatial_merge_size", 2)
        vision_hidden = config["vision_hidden_size"]
        text_hidden = config["text_hidden_size"]

        merge_dim = vision_hidden * (self.merge_size ** 2)
        self.mlp = nn.Sequential(
            nn.LayerNorm(merge_dim),
            nn.Linear(merge_dim, text_hidden, bias=True),
            nn.GELU(),
            nn.Linear(text_hidden, text_hidden, bias=True),
        )

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_hidden_states: [total_patches, vision_hidden]

        Returns:
            [total_merged_patches, text_hidden]
        """
        # This is a simplified version — full implementation would need
        # the grid_thw to properly merge spatial neighbors.
        # For now, assume patches are in row-major order and merge groups.
        total, hidden = image_hidden_states.shape
        m = self.merge_size
        groups = total // (m * m)
        merged = image_hidden_states.view(groups, m * m * hidden)
        return self.mlp(merged)
