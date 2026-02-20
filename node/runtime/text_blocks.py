"""
UNFED Text Decoder Building Blocks — pure PyTorch, no external model libraries.

Every modern LLM text decoder (Llama, Qwen, Mistral, Gemma, SmolVLM, etc.)
is assembled from these same primitives.  The only differences are the
*numbers* (hidden_size, num_heads, etc.) which come from the manifest config.

Components:
  UnfedRMSNorm       — Root-mean-square layer normalization
  UnfedRoPE          — Rotary position embeddings (cos/sin precomputation)
  UnfedAttention     — Grouped-query attention (MHA/GQA/MQA) with RoPE
  UnfedMLP           — Gated MLP (SwiGLU, GELU, etc.)
  UnfedDecoderLayer  — Full transformer block (norm → attn → residual → norm → mlp → residual)
  UnfedEmbedding     — Token embedding lookup
  UnfedLMHead        — Linear projection to vocabulary logits

Usage:
    config = {
        "hidden_size": 576,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "intermediate_size": 1536,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "vocab_size": 49280,
        "max_position_embeddings": 2048,
    }
    layer = UnfedDecoderLayer(config)
    layer.load_state_dict(weights)
    output = layer(hidden_states, position_embeddings, cache, cache_position)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_fast": lambda x: F.gelu(x, approximate="tanh"),
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "tanh": torch.tanh,
}


def get_activation(name: str):
    """Look up activation function by name."""
    fn = _ACTIVATIONS.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Supported: {', '.join(_ACTIVATIONS)}"
        )
    return fn


# ---------------------------------------------------------------------------
# UnfedRMSNorm
# ---------------------------------------------------------------------------

class UnfedRMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Used by Llama, Qwen, Mistral, SmolVLM, Gemma, and most modern LLMs.

    Formula:
        y = x * rsqrt(mean(x^2) + eps) * weight

    Config params:
        hidden_size: int — dimension of the input
        rms_norm_eps: float — epsilon for numerical stability (default 1e-5)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


# ---------------------------------------------------------------------------
# UnfedRoPE — Rotary Position Embeddings
# ---------------------------------------------------------------------------

class UnfedRoPE(nn.Module):
    """Rotary position embeddings (RoPE).

    Precomputes cos/sin tables and returns position embeddings for given
    input length.  Works for all standard 1D-RoPE models (Llama, Qwen,
    Mistral, SmolVLM, etc.).

    Config params:
        head_dim: int — dimension per attention head
        rope_theta: float — base frequency (default 10000.0)
        max_position_embeddings: int — max sequence length (default 8192)
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache will be built on first forward
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin cache up to seq_len."""
        if seq_len <= self._cached_seq_len and self._cos_cached is not None:
            return
        self._cached_seq_len = max(seq_len, self.max_position_embeddings)
        t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [seq_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) position embeddings for the given positions.

        Args:
            x: Input tensor, used only for dtype/device inference.
               Shape: [batch, seq_len, hidden] or similar.
            position_ids: Position indices, shape [batch, seq_len].

        Returns:
            (cos, sin) each of shape [batch, seq_len, head_dim].
        """
        seq_len = int(position_ids.max()) + 1
        self._build_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cached[position_ids]  # [batch, seq_len, head_dim]
        sin = self._sin_cached[position_ids]
        return cos, sin


# ---------------------------------------------------------------------------
# RoPE application helpers
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim] or broadcastable
        sin: [batch, seq_len, head_dim] or broadcastable

    Returns:
        (q_embed, k_embed) with rotary embeddings applied.
    """
    # Unsqueeze for head dimension: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# UnfedAttention — Grouped-Query Attention with RoPE
# ---------------------------------------------------------------------------

class UnfedAttention(nn.Module):
    """Multi-head / grouped-query / multi-query attention with RoPE.

    Supports all standard attention configurations:
      - MHA: num_key_value_heads == num_attention_heads
      - GQA: num_key_value_heads < num_attention_heads (groups)
      - MQA: num_key_value_heads == 1

    Uses PyTorch's scaled_dot_product_attention (SDPA) for efficient
    computation with Flash Attention / memory-efficient backends.

    Config params:
        hidden_size: int
        num_attention_heads: int
        num_key_value_heads: int
        head_dim: int (optional, default = hidden_size // num_attention_heads)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) from UnfedRoPE
            past_key_value: (cached_k, cached_v) or None
            cache_position: [seq_len] — absolute positions for cache update
            attention_mask: optional causal/padding mask

        Returns:
            (output, (new_k_cache, new_v_cache))
        """
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache update
        if past_key_value is not None:
            cached_k, cached_v = past_key_value
            if cache_position is not None:
                # Scatter new KV into cache at the right positions
                cached_k[:, :, cache_position] = k
                cached_v[:, :, cache_position] = v
                # Slice to valid range only (cache may be pre-allocated larger)
                valid_len = int(cache_position[-1].item()) + 1
                k = cached_k[:, :, :valid_len]
                v = cached_v[:, :, :valid_len]
            else:
                # Append mode (fallback)
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)

        new_cache = (past_key_value[0] if past_key_value is not None else k,
                     past_key_value[1] if past_key_value is not None else v)

        # GQA: repeat KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Causal mask for SDPA
        is_causal = (attention_mask is None and seq_len > 1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )

        # Reshape: [batch, num_heads, seq_len, head_dim] → [batch, seq_len, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_output)

        return output, new_cache


# ---------------------------------------------------------------------------
# UnfedMLP — Gated MLP (SwiGLU / GELU / etc.)
# ---------------------------------------------------------------------------

class UnfedMLP(nn.Module):
    """Gated MLP used in modern LLMs.

    For SwiGLU (most common):
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    For standard GELU:
        output = down_proj(gelu(up_proj(x)))
        (gate_proj is not used)

    Config params:
        hidden_size: int
        intermediate_size: int
        hidden_act: str — "silu", "gelu", "relu", etc.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.act_fn = get_activation(config.get("hidden_act", "silu"))

        # Gated MLP: gate_proj + up_proj → down_proj
        # Non-gated: just up_proj → down_proj (gate_proj unused)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Whether to use gated activation (SwiGLU pattern)
        # Most modern LLMs use gated — detect from activation name
        self._gated = config.get("hidden_act", "silu") in ("silu", "swiglu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._gated:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act_fn(self.up_proj(x)))


# ---------------------------------------------------------------------------
# UnfedDecoderLayer — Full Transformer Decoder Block
# ---------------------------------------------------------------------------

class UnfedDecoderLayer(nn.Module):
    """Standard transformer decoder layer.

    Structure:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        hidden_states = self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states

    Config params: (forwarded to sub-components)
        hidden_size, num_attention_heads, num_key_value_heads,
        intermediate_size, hidden_act, rms_norm_eps
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config.get("rms_norm_eps", 1e-5)

        self.input_layernorm = UnfedRMSNorm(hidden_size, eps)
        self.self_attn = UnfedAttention(config)
        self.post_attention_layernorm = UnfedRMSNorm(hidden_size, eps)
        self.mlp = UnfedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            (output_hidden_states, new_kv_cache)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


# ---------------------------------------------------------------------------
# UnfedEmbedding — Token Embedding
# ---------------------------------------------------------------------------

class UnfedEmbedding(nn.Module):
    """Token embedding lookup table.

    Config params:
        vocab_size: int
        hidden_size: int
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] integer token IDs

        Returns:
            [batch, seq_len, hidden_size] embeddings
        """
        return self.embed_tokens(token_ids)


# ---------------------------------------------------------------------------
# UnfedLMHead — Language Model Head
# ---------------------------------------------------------------------------

class UnfedLMHead(nn.Module):
    """Linear projection from hidden states to vocabulary logits.

    Config params:
        hidden_size: int
        vocab_size: int
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            [batch, seq_len, vocab_size] logits
        """
        return self.lm_head(hidden_states)
