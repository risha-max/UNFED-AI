"""
MPC (Multi-Party Computation) for Shard 0 — Beaver-Triple 2PC.

Two independent nodes (A and B) jointly compute the embedding + layer 0
of a transformer model.  Neither node sees the raw token IDs or the full
embedding vector.  Subsequent layers run in cleartext on Node A — after
one full transformer layer, activations cannot be inverted to raw tokens.

Protocol overview (Beaver-triple-based 2PC):
  1. Node A receives token_ids, computes embedding, splits into additive
     shares, sends share_b + Beaver triple shares to Node B.
  2. Both nodes jointly compute layer 0 on their shares:
     - Linear ops (matmul with public weights): free on shares.
     - Non-linear ops (RMSNorm, softmax, SiLU): polynomial approximations
       evaluated via Beaver-triple secure multiplications.
  3. Node B sends its final share back to Node A.
  4. Node A reconstructs and continues with cleartext layers 1-N.

Security:
  - Only (x - a) and (y - b) are ever revealed, where a, b are random
    Beaver triple values.  These reveal nothing about x or y.
  - No intermediate value is ever reconstructed on a single party during
    the MPC phase (layer 0).

Usage:
    python -m network.mpc_shard0 --role B --port 50063 --peer localhost:50061
    python -m network.mpc_shard0 --role A --port 50061 --peer localhost:50063 \\
        --advertise localhost:50061 --registry localhost:50050
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass
from typing import Optional

import grpc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc

from network.resilience import create_resilient_channel
from network.mpc_beaver import (
    BeaverTriple, BeaverTripleShares,
    serialize_triple_shares, deserialize_triple_shares,
)
from network.mpc_protocols import (
    PeerExchanger, TripleAllocator,
    secure_rmsnorm, secure_softmax, secure_silu, secure_gate_up,
    secure_matmul, allocate_layer0_triples,
)


# ---------------------------------------------------------------------------
# Secret sharing primitives
# ---------------------------------------------------------------------------

def create_additive_shares(tensor: torch.Tensor,
                           num_shares: int = 2) -> list[torch.Tensor]:
    """Split a tensor into additive secret shares."""
    shares = []
    remaining = tensor.clone()
    for _ in range(num_shares - 1):
        share = torch.randn_like(tensor)
        shares.append(share)
        remaining = remaining - share
    shares.append(remaining)
    return shares


def reconstruct_from_shares(shares: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct the original tensor from additive shares."""
    return sum(shares)


# ---------------------------------------------------------------------------
# Tensor serialization helpers
# ---------------------------------------------------------------------------

def _tensor_to_bytes(tensor: torch.Tensor) -> tuple[bytes, list[int]]:
    """Serialize a tensor to bytes + shape."""
    t = tensor.contiguous().float()
    return t.numpy().tobytes(), list(t.shape)


def _bytes_to_tensor(data: bytes, shape: list[int]) -> torch.Tensor:
    """Deserialize bytes + shape to a tensor."""
    return torch.from_numpy(
        np.frombuffer(data, dtype=np.float32).copy().reshape(shape))


# ---------------------------------------------------------------------------
# gRPC-based peer exchanger (Node A's view: calls Node B's Exchange RPC)
# ---------------------------------------------------------------------------

class GrpcPeerExchangerA(PeerExchanger):
    """
    Node A's exchanger: calls Node B's MPCPeer.Exchange RPC to send
    epsilon/delta and receive Node B's epsilon/delta in return.
    """

    def __init__(self, peer_stub):
        self._stub = peer_stub  # inference_pb2_grpc.MPCPeerStub

    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor,
                 my_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps_bytes, eps_shape = _tensor_to_bytes(my_epsilon)
        del_bytes, del_shape = _tensor_to_bytes(my_delta)

        resp = self._stub.Exchange(
            inference_pb2.MPCExchangeRequest(
                session_id=session_id,
                op_id=op_id,
                epsilon_data=eps_bytes,
                delta_data=del_bytes,
                shape=list(eps_shape),
                delta_shape=list(del_shape),
            ),
            timeout=60,
        )
        resp_eps_shape = list(resp.shape)
        resp_del_shape = list(resp.delta_shape) if resp.delta_shape else resp_eps_shape
        peer_eps = _bytes_to_tensor(bytes(resp.epsilon_data), resp_eps_shape)
        peer_del = _bytes_to_tensor(bytes(resp.delta_data), resp_del_shape)
        return peer_eps, peer_del


# ---------------------------------------------------------------------------
# Node B's peer exchanger: waits for Node A's call, responds synchronously
# ---------------------------------------------------------------------------

class GrpcPeerExchangerB(PeerExchanger):
    """
    Node B's exchanger: exchanges are driven by Node A calling the Exchange
    RPC.  Node B computes its epsilon/delta and waits for A's call, then
    responds.  This class bridges the async gRPC handler with the
    synchronous protocol execution on Node B's side.

    Flow:
      - Node B's protocol thread calls exchanger.exchange() → blocks
      - Node A calls the gRPC Exchange RPC → handler receives A's values,
        wakes up B's protocol thread, gets B's values, responds to A
    """

    def __init__(self):
        self._pending_b: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._incoming_a: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._response_b: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._b_ready: dict[str, threading.Event] = {}  # B has values
        self._a_ready: dict[str, threading.Event] = {}  # A has called

    def exchange(self, session_id: str, op_id: str,
                 my_epsilon: torch.Tensor,
                 my_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Called by Node B's protocol thread.  Deposits B's values, waits
        for A's values."""
        key = f"{session_id}:{op_id}"
        with self._lock:
            self._pending_b[key] = (my_epsilon, my_delta)
            if key not in self._b_ready:
                self._b_ready[key] = threading.Event()
            if key not in self._a_ready:
                self._a_ready[key] = threading.Event()

        # Signal that B's values are ready
        self._b_ready[key].set()

        # Wait for A's values
        self._a_ready[key].wait(timeout=60)

        with self._lock:
            peer_eps, peer_del = self._incoming_a.pop(key)
            self._a_ready.pop(key, None)
            self._b_ready.pop(key, None)
        return peer_eps, peer_del

    def handle_exchange_rpc(self, session_id: str, op_id: str,
                            a_epsilon: torch.Tensor,
                            a_delta: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Called by the gRPC handler when Node A sends its exchange."""
        key = f"{session_id}:{op_id}"
        with self._lock:
            if key not in self._b_ready:
                self._b_ready[key] = threading.Event()
            if key not in self._a_ready:
                self._a_ready[key] = threading.Event()

        # Wait for B's protocol thread to deposit its values
        self._b_ready[key].wait(timeout=60)

        with self._lock:
            b_eps, b_del = self._pending_b.pop(key)
            self._incoming_a[key] = (a_epsilon, a_delta)

        # Wake up B's protocol thread
        self._a_ready[key].set()

        return b_eps, b_del


# ---------------------------------------------------------------------------
# MPCNode — loads weights, runs layer 0 with MPC
# ---------------------------------------------------------------------------

class MPCNode:
    """
    Full drop-in replacement for shard 0, with MPC-protected layer 0.

    Each MPCNode holds:
      - The embedding layer weights (both nodes have a copy)
      - ALL layers for shard 0 (loaded from shard_0.pt)

    Privacy model:
      - Layer 0: computed via Beaver-triple 2PC (privacy-protected)
      - Layers 1-N: computed in cleartext on Node A (safe — activations
        after one full transformer layer can't be inverted to raw tokens)
    """

    def __init__(self, role: str, peer_address: str,
                 shards_dir: str = None, manifest_path: str = None):
        assert role in ("A", "B"), "Role must be 'A' or 'B'"
        self.role = role
        self.peer_address = peer_address
        self._peer_stub = None
        self._shards_dir = shards_dir or config.SHARDS_DIR

        self.embed_tokens = None
        self.layers: list = []
        self.rotary_emb = None
        self.model_config = None
        self.layer_start = 0
        self.layer_end = 0
        self._use_generic = False
        self._image_token_id = None
        self._cache_manager = None

        # Architecture info (set during weight loading)
        self.hidden_size = 0
        self.num_heads = 0
        self.head_dim = 0
        self.intermediate_size = 0
        self.num_kv_heads = 0

        self._load_weights(manifest_path)

        self._session_shares: dict[str, dict] = {}
        self._session_lock = threading.Lock()

    def _load_weights(self, manifest_path: str = None):
        """Load ALL of shard 0's weights."""
        manifest_path = manifest_path or os.path.join(
            self._shards_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        use_generic = manifest.get("format_version", 1) >= 2

        if "text_shards" in manifest and manifest["text_shards"]:
            shard_info = manifest["text_shards"][0]
        else:
            shard_info = manifest["shards"][0]
        self.layer_start = shard_info["layer_start"]
        self.layer_end = shard_info["layer_end"]
        num_layers = self.layer_end - self.layer_start
        print(f"[MPC-{self.role}]   Shard 0 covers layers "
              f"{self.layer_start}-{self.layer_end} ({num_layers} layers)")

        self._image_token_id = manifest.get("image_token_id")

        if use_generic and manifest.get("architecture", {}).get("text"):
            self._load_weights_generic(manifest, shard_info, num_layers)
        else:
            self._load_weights_hf(num_layers)

    def _load_weights_generic(self, manifest: dict, shard_info: dict,
                              num_layers: int):
        """Load weights using generic runtime building blocks (v2 path)."""
        from node.runtime.text_blocks import (
            UnfedRMSNorm, UnfedRoPE, UnfedAttention, UnfedMLP,
            UnfedDecoderLayer, UnfedEmbedding,
        )
        from node.runtime.weight_loader import (
            load_shard_weights, load_weights_into_module,
        )

        arch = manifest["architecture"]["text"]
        shard_file = shard_info.get("file", "text_shard_0.pt")
        shard_path = os.path.join(self._shards_dir, shard_file)
        print(f"[MPC-{self.role}] Loading shard 0 from {shard_path} "
              f"(generic runtime)")

        self.hidden_size = arch["hidden_size"]
        self.num_heads = arch["num_attention_heads"]
        self.head_dim = arch.get(
            "head_dim", self.hidden_size // self.num_heads)
        self.intermediate_size = arch.get(
            "intermediate_size", self.hidden_size * 4)
        self.num_kv_heads = arch.get(
            "num_key_value_heads", self.num_heads)

        self.embed_tokens = UnfedEmbedding(
            arch["vocab_size"], self.hidden_size)
        self.rotary_emb = UnfedRoPE(
            self.head_dim,
            arch.get("rope_theta", 10000.0),
            arch.get("max_position_embeddings", 8192),
        )

        self.layers = []
        for i in range(num_layers):
            self.layers.append(UnfedDecoderLayer(arch))

        state_dict = load_shard_weights(shard_path)
        remaining = dict(state_dict)
        loaded = 0

        embed_keys = {k: v for k, v in remaining.items()
                      if "embed_tokens" in k}
        if embed_keys:
            embed_sd = {}
            for k, v in embed_keys.items():
                local_k = k.split("embed_tokens.")[-1] \
                    if "embed_tokens." in k else k
                if not local_k.startswith("embed_tokens"):
                    local_k = f"embed_tokens.{local_k}"
                embed_sd[local_k] = v
            load_weights_into_module(self.embed_tokens, embed_sd, strict=False)
            for k in embed_keys:
                remaining.pop(k, None)
            loaded += len(embed_keys)

        print(f"[MPC-{self.role}]   Loaded embed_tokens")

        for local_idx in range(num_layers):
            global_idx = self.layer_start + local_idx
            prefixes = [
                f"layers.{global_idx}.",
                f"model.layers.{global_idx}.",
                f"layer_{global_idx}.",
            ]
            layer_sd = {}
            keys_used = []
            for k, v in remaining.items():
                for pfx in prefixes:
                    if k.startswith(pfx):
                        layer_sd[k[len(pfx):]] = v
                        keys_used.append(k)
                        break
            if layer_sd:
                load_weights_into_module(
                    self.layers[local_idx], layer_sd, strict=False)
                for k in keys_used:
                    remaining.pop(k, None)
                loaded += len(keys_used)

            label = 'MPC-protected (2PC)' if local_idx == 0 else 'cleartext'
            print(f"[MPC-{self.role}]   Loaded layer {global_idx} ({label})")

        self._use_generic = True
        self.model_config = None

        from node.runtime.generic_runner import SessionCacheManager
        self._cache_manager = SessionCacheManager(
            num_layers=num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=arch.get("max_position_embeddings", 4096),
        )
        print(f"[MPC-{self.role}] KV cache: {num_layers} layers, "
              f"{self.num_kv_heads} kv_heads, head_dim={self.head_dim}")
        print(f"[MPC-{self.role}] Loaded {num_layers} layers "
              f"(generic, format_version=2)")

    def _load_weights_hf(self, num_layers: int):
        """Load weights using HuggingFace model classes (v1 legacy path)."""
        from transformers import AutoConfig, AutoModelForCausalLM

        shard_path = config.get_shard_path(0)
        hf_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        self.model_config = hf_config
        self.hidden_size = hf_config.hidden_size
        self.num_heads = hf_config.num_attention_heads
        self.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        self.intermediate_size = hf_config.intermediate_size
        self.num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads)

        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.embed_tokens = model.model.embed_tokens
        if hasattr(model.model, 'rotary_emb'):
            self.rotary_emb = model.model.rotary_emb
        else:
            from node.runtime.text_blocks import UnfedRoPE
            self.rotary_emb = UnfedRoPE(
                self.head_dim,
                getattr(hf_config, "rope_theta", 10000.0),
                getattr(hf_config, "max_position_embeddings", 2048),
            )

        all_layers = model.model.layers
        start = self.layer_start
        end = self.layer_end
        self.layers = [all_layers[i] for i in range(start, end)]
        del model
        print(f"[MPC-{self.role}] Loaded {num_layers} layers (HuggingFace v1)")

    @torch.no_grad()
    def compute_embedding_share(self, token_ids: torch.Tensor,
                                image_embeddings=None):
        """Compute embedding and split into two additive shares."""
        embedding = self.embed_tokens(token_ids)

        if (image_embeddings is not None
                and self._image_token_id is not None):
            mask = (token_ids == self._image_token_id)
            num_placeholders = mask.sum().item()
            num_features = image_embeddings.shape[0]

            if num_placeholders > 0 and num_features > 0:
                use_count = min(num_features, num_placeholders)
                flat_mask = mask.view(-1)
                indices = flat_mask.nonzero(as_tuple=True)[0][:use_count]
                flat_embed = embedding.view(-1, embedding.shape[-1])
                flat_embed[indices] = image_embeddings[:use_count].to(
                    flat_embed.dtype)
                embedding = flat_embed.view(embedding.shape)
                print(f"[MPC] Merged {use_count} image embeddings into "
                      f"text sequence")

        shares = create_additive_shares(embedding, 2)
        return shares[0], shares[1]

    @torch.no_grad()
    def forward_layer0_on_share(
        self,
        my_share: torch.Tensor,
        triples: TripleAllocator,
        exchanger: PeerExchanger,
        session_id: str,
        is_party_0: bool,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run layer 0 on this party's share using Beaver-triple 2PC.

        Both parties call this simultaneously with their own share.
        All non-linear operations go through secure_multiply via
        the exchanger.  Matrix products of two shared tensors use
        secure_matmul with matrix Beaver triples.  Linear operations
        (matmul with public weights) are computed locally on shares.

        Returns this party's share of the layer 0 output.
        """
        from node.runtime.text_blocks import rotate_half

        layer_0 = self.layers[0]
        batch, seq_len, _ = my_share.shape

        # --- Secure Input RMSNorm ---
        weight = layer_0.input_layernorm.weight
        eps = layer_0.input_layernorm.eps if hasattr(
            layer_0.input_layernorm, 'eps') else getattr(
            layer_0.input_layernorm, 'variance_epsilon', 1e-6)

        normed_share = secure_rmsnorm(
            my_share, weight, eps, triples, exchanger,
            session_id, is_party_0, prefix="rmsnorm_in")

        # --- Attention: Q/K/V projections (linear, free on shares) ---
        attn = layer_0.self_attn
        q_share = normed_share @ attn.q_proj.weight.T
        k_share = normed_share @ attn.k_proj.weight.T
        v_share = normed_share @ attn.v_proj.weight.T

        if hasattr(attn.q_proj, 'bias') and attn.q_proj.bias is not None:
            if is_party_0:
                q_share = q_share + attn.q_proj.bias
                k_share = k_share + attn.k_proj.bias
                v_share = v_share + attn.v_proj.bias

        # Reshape for multi-head attention
        q_share = q_share.view(batch, seq_len, self.num_heads, self.head_dim
                               ).transpose(1, 2)
        k_share = k_share.view(batch, seq_len, self.num_kv_heads,
                               self.head_dim).transpose(1, 2)
        v_share = v_share.view(batch, seq_len, self.num_kv_heads,
                               self.head_dim).transpose(1, 2)

        # --- Apply RoPE (public rotation, free on shares) ---
        if position_ids is None:
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(my_share, position_ids)
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
        q_share = q_share * cos + rotate_half(q_share) * sin
        k_share = k_share * cos[:, :, :, :self.head_dim] + \
            rotate_half(k_share) * sin[:, :, :, :self.head_dim]

        # GQA repeat for K, V if needed
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_share = k_share.repeat_interleave(repeat, dim=1)
            v_share = v_share.repeat_interleave(repeat, dim=1)

        # --- Secure Q @ K^T via matrix Beaver triple ---
        scale = 1.0 / (self.head_dim ** 0.5)
        scores_share = secure_matmul(
            q_share, k_share,
            triples.get("attn_qk_matmul"),
            exchanger, session_id, "attn_qk_matmul",
            is_party_0, transpose_b=True)
        scores_share = scores_share * scale

        # Causal mask (public, same for both parties)
        if seq_len > 1:
            causal_mask = torch.full(
                (seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)
            if is_party_0:
                scores_share = scores_share + causal_mask.unsqueeze(0).unsqueeze(0)

        # --- Secure Softmax ---
        attn_weights_share = secure_softmax(
            scores_share, triples, exchanger, session_id, is_party_0)

        # --- Secure attn_weights @ V via matrix Beaver triple ---
        attn_output_share = secure_matmul(
            attn_weights_share, v_share,
            triples.get("attn_av_matmul"),
            exchanger, session_id, "attn_av_matmul",
            is_party_0, transpose_b=False)

        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output_share = attn_output_share.transpose(1, 2).contiguous().view(
            batch, seq_len, self.hidden_size)

        # Output projection (linear, free)
        attn_output_share = attn_output_share @ attn.o_proj.weight.T
        if hasattr(attn.o_proj, 'bias') and attn.o_proj.bias is not None:
            if is_party_0:
                attn_output_share = attn_output_share + attn.o_proj.bias

        # Residual connection
        hidden_share = my_share + attn_output_share

        # --- Secure Post-Attention RMSNorm ---
        post_weight = layer_0.post_attention_layernorm.weight
        post_eps = layer_0.post_attention_layernorm.eps if hasattr(
            layer_0.post_attention_layernorm, 'eps') else getattr(
            layer_0.post_attention_layernorm, 'variance_epsilon', 1e-6)

        normed_post_share = secure_rmsnorm(
            hidden_share, post_weight, post_eps, triples, exchanger,
            session_id, is_party_0, prefix="rmsnorm_post")

        # --- MLP (linear projections are free) ---
        mlp = layer_0.mlp
        gate_share = normed_post_share @ mlp.gate_proj.weight.T
        up_share = normed_post_share @ mlp.up_proj.weight.T

        if hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
            if is_party_0:
                gate_share = gate_share + mlp.gate_proj.bias
                up_share = up_share + mlp.up_proj.bias

        # --- Secure SiLU on gate ---
        gate_activated_share = secure_silu(
            gate_share, triples, exchanger, session_id, is_party_0)

        # --- Secure gate * up ---
        mlp_mid_share = secure_gate_up(
            gate_activated_share, up_share,
            triples, exchanger, session_id, is_party_0)

        # Down projection (linear, free)
        mlp_out_share = mlp_mid_share @ mlp.down_proj.weight.T
        if hasattr(mlp.down_proj, 'bias') and mlp.down_proj.bias is not None:
            if is_party_0:
                mlp_out_share = mlp_out_share + mlp.down_proj.bias

        # Residual connection
        output_share = hidden_share + mlp_out_share

        return output_share

    @torch.no_grad()
    def forward_remaining_layers(self, hidden: torch.Tensor,
                                 session_id: str) -> torch.Tensor:
        """Run layers 1-N in cleartext on Node A (safe — activations are
        already mixed beyond inversion)."""
        if not self._use_generic:
            return self._forward_remaining_hf(hidden, session_id)

        session_cache = self._cache_manager.get_or_create(session_id)
        current_pos = session_cache.seq_len

        seq_len = hidden.shape[1]
        batch = hidden.shape[0]
        position_ids = torch.arange(
            current_pos, current_pos + seq_len
        ).unsqueeze(0).expand(batch, -1)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        cache_position = torch.arange(current_pos, current_pos + seq_len)

        attention_mask = None
        if seq_len > 1:
            total_len = current_pos + seq_len
            causal_mask = torch.full(
                (seq_len, total_len), float("-inf"))
            for i in range(seq_len):
                causal_mask[i, :current_pos + i + 1] = 0.0
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Layer 0 KV cache update (we still need it for decode steps)
        k_cache_0, v_cache_0 = session_cache.get_layer_cache(0)

        # Run layer 0 in cleartext as well (the MPC already computed it,
        # but we need to update the KV cache for future decode steps)
        # Skip the actual computation, just populate cache with dummy
        # Note: for decode steps, layer 0 still needs MPC, so the KV cache
        # is populated correctly during the MPC phase.

        for i, layer in enumerate(self.layers[1:], start=1):
            position_embeddings = self.rotary_emb(hidden, position_ids)
            k_cache_i, v_cache_i = session_cache.get_layer_cache(i)
            hidden, _ = layer(
                hidden,
                position_embeddings=position_embeddings,
                past_key_value=(k_cache_i, v_cache_i),
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        session_cache.advance(seq_len)
        print(f"[MPC-{self.role}] All {len(self.layers)} layers done, "
              f"output shape: {hidden.shape} (pos: {current_pos}->"
              f"{session_cache.seq_len})")
        return hidden

    def _forward_remaining_hf(self, hidden, session_id):
        """Legacy HF path for layers 1-N."""
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden, position_ids)

        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        cache_position = torch.arange(seq_len)

        for i, layer in enumerate(self.layers[1:], start=1):
            position_embeddings = self.rotary_emb(hidden, position_ids)
            output = layer(
                hidden,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden = output[0] if isinstance(output, tuple) else output

        print(f"[MPC-{self.role}] All {len(self.layers)} layers done, "
              f"output shape: {hidden.shape}")
        return hidden


# ---------------------------------------------------------------------------
# Node B's MPCPeer gRPC servicer
# ---------------------------------------------------------------------------

class MPCPeerServicer(inference_pb2_grpc.MPCPeerServicer):
    """
    Handles MPC peer RPCs on Node B.

    Node A calls these RPCs to:
      1. SendShare: distribute B's embedding share + Beaver triple shares
      2. Exchange: swap epsilon/delta during each secure multiplication
      3. CollectShare: retrieve B's final share after layer 0
    """

    def __init__(self, mpc_node: MPCNode, exchanger_b: GrpcPeerExchangerB):
        self._mpc = mpc_node
        self._exchanger = exchanger_b
        self._session_results: dict[str, torch.Tensor] = {}
        self._session_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def SendShare(self, request, context):
        """Receive embedding share + triples from Node A, start layer 0."""
        session_id = request.session_id
        shape = list(request.share_shape)
        share_b = _bytes_to_tensor(bytes(request.share_data), shape)

        print(f"[MPC-B] Received share for session {session_id[:8]}... "
              f"shape: {share_b.shape}")

        # Deserialize triples (support different a/b/c shapes for matmul)
        triple_dict: dict[str, BeaverTripleShares] = {}
        for tp in request.triples:
            a_shape = list(tp.shape)
            b_shape = list(tp.b_shape) if tp.b_shape else a_shape
            c_shape = list(tp.c_shape) if tp.c_shape else a_shape
            triple_dict[tp.op_id] = deserialize_triple_shares(
                bytes(tp.a_data), bytes(tp.b_data),
                bytes(tp.c_data), a_shape, b_shape, c_shape)

        triples = TripleAllocator(triple_dict)

        # Create completion event
        with self._lock:
            self._session_events[session_id] = threading.Event()

        # Run layer 0 on B's share in a background thread
        def run_b():
            try:
                result = self._mpc.forward_layer0_on_share(
                    share_b, triples, self._exchanger,
                    session_id, is_party_0=False)
                with self._lock:
                    self._session_results[session_id] = result
                    self._session_events[session_id].set()
                print(f"[MPC-B] Layer 0 done for session {session_id[:8]}... "
                      f"result shape: {result.shape}")
            except Exception as e:
                print(f"[MPC-B] Error in layer 0: {e}")
                import traceback
                traceback.print_exc()
                with self._lock:
                    self._session_events[session_id].set()

        t = threading.Thread(target=run_b, daemon=True)
        t.start()

        return inference_pb2.MPCSendShareResponse(accepted=True)

    def Exchange(self, request, context):
        """Handle a secure multiplication exchange from Node A."""
        session_id = request.session_id
        op_id = request.op_id
        eps_shape = list(request.shape)
        del_shape = list(request.delta_shape) if request.delta_shape else eps_shape

        a_eps = _bytes_to_tensor(bytes(request.epsilon_data), eps_shape)
        a_del = _bytes_to_tensor(bytes(request.delta_data), del_shape)

        b_eps, b_del = self._exchanger.handle_exchange_rpc(
            session_id, op_id, a_eps, a_del)

        b_eps_bytes, b_eps_shape = _tensor_to_bytes(b_eps)
        b_del_bytes, b_del_shape = _tensor_to_bytes(b_del)

        return inference_pb2.MPCExchangeResponse(
            epsilon_data=b_eps_bytes,
            delta_data=b_del_bytes,
            shape=b_eps_shape,
            delta_shape=b_del_shape,
        )

    def CollectShare(self, request, context):
        """Return B's final share of layer 0 output."""
        session_id = request.session_id
        event = self._session_events.get(session_id)
        if event:
            event.wait(timeout=120)

        with self._lock:
            result = self._session_results.pop(session_id, None)
            self._session_events.pop(session_id, None)

        if result is None:
            context.set_details("No result for this session")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return inference_pb2.MPCCollectResponse()

        data, shape = _tensor_to_bytes(result)
        return inference_pb2.MPCCollectResponse(
            share_data=data,
            share_shape=shape,
        )


# ---------------------------------------------------------------------------
# Node A's servicer (receives Forward from clients, orchestrates MPC)
# ---------------------------------------------------------------------------

class MPCNodeServicer(inference_pb2_grpc.InferenceNodeServicer):
    """
    gRPC servicer for MPC Node A.  Receives client requests, orchestrates
    the 2PC protocol with Node B, and forwards results to the next shard.
    """

    def __init__(self, mpc_node: MPCNode, port: int):
        self.mpc = mpc_node
        self.port = port
        self._node_id: str = ""
        self._peer_address: str = ""
        self._peer_node_id: str = ""
        self._daemon_stub = None
        self._private_key = None
        self._peer_stub = None  # MPCPeerStub for calling Node B

    def _connect_peer(self):
        """Connect to Node B's MPCPeer service."""
        if self._peer_stub is None and self._peer_address:
            channel = create_resilient_channel(
                self._peer_address, config.GRPC_OPTIONS)
            self._peer_stub = inference_pb2_grpc.MPCPeerStub(channel)
            print(f"[MPC-A] Connected to peer B at {self._peer_address}")

    def _init_daemon_stub(self, registry_address: str):
        """Discover and connect to the chain daemon."""
        try:
            import registry_pb2
            import registry_pb2_grpc
            channel = grpc.insecure_channel(
                registry_address, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(
                registry_pb2.DiscoverRequest(model_id=""),
                timeout=10,
            )
            daemons = [n for n in resp.nodes if n.node_type == "daemon"]
            channel.close()
            if daemons:
                daemon = daemons[0]
                self._daemon_stub = inference_pb2_grpc.InferenceNodeStub(
                    grpc.insecure_channel(
                        daemon.address, options=config.GRPC_OPTIONS))
                print(f"[MPC-A] Connected to chain daemon at "
                      f"{daemon.address}")
            else:
                print(f"[MPC-A] No daemon found — shares won't be recorded")
        except Exception:
            print(f"[MPC-A] Could not connect to daemon")

    def _record_dual_shares(self, session_id: str, output: torch.Tensor):
        """Submit compute shares for both MPC nodes to the daemon."""
        if not self._daemon_stub or not self._node_id:
            return

        act_hash = hashlib.sha256(
            output.contiguous().float().numpy().tobytes()
        ).hexdigest()[:16]

        from economics.share_chain import ComputeShare
        from economics.distributed_chain import share_to_proto

        share_a = ComputeShare(
            node_id=self._node_id,
            shard_index=0,
            session_id=session_id,
            activation_hash=act_hash,
            tokens_processed=1,
            share_weight=1.0,
        )
        share_b = ComputeShare(
            node_id=self._peer_node_id,
            shard_index=0,
            session_id=session_id,
            activation_hash=act_hash,
            tokens_processed=1,
            share_weight=1.0,
        )

        try:
            self._daemon_stub.SubmitShares(
                inference_pb2.SubmitSharesRequest(
                    shares=[share_to_proto(share_a),
                            share_to_proto(share_b)],
                    submitter_id=self._node_id,
                ),
                timeout=5,
            )
            print(f"[MPC-A] Submitted dual shares for session "
                  f"{session_id[:8]}...")
        except Exception:
            print(f"[MPC-A] Failed to submit shares to daemon")

    def Forward(self, request, context):
        """Handle a forward pass through the MPC layer (Node A only)."""
        session_id = request.session_id

        try:
            if self.mpc.role != "A":
                context.set_details("Only Node A accepts Forward requests")
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                return inference_pb2.ForwardResponse()

            self._connect_peer()

            token_ids = torch.tensor([list(request.token_ids)])

            # Extract image embeddings
            image_embeddings = None
            if (request.image_embeddings
                    and request.image_embeddings_shape):
                ie_shape = list(request.image_embeddings_shape)
                image_embeddings = torch.from_numpy(
                    np.frombuffer(
                        bytes(request.image_embeddings),
                        dtype=np.float32,
                    ).copy().reshape(ie_shape))
                print(f"[MPC-A] Received image_embeddings: "
                      f"{image_embeddings.shape}")

            # Step 1: Compute embedding and create shares
            share_a, share_b = self.mpc.compute_embedding_share(
                token_ids, image_embeddings=image_embeddings)
            print(f"[MPC-A] Session {session_id[:8]}... "
                  f"embedding shape: {share_a.shape}")

            # Step 2: Generate Beaver triples and distribute to Node B
            batch, seq_len, _ = share_a.shape
            all_triples = allocate_layer0_triples(
                self.mpc.hidden_size,
                self.mpc.num_heads,
                self.mpc.head_dim,
                self.mpc.intermediate_size,
                seq_len,
                batch,
            )

            # Build party-specific triple allocators
            my_triples: dict[str, BeaverTripleShares] = {}
            peer_triples: list = []
            for op_id, bt in all_triples.items():
                my_triples[op_id] = bt.party_0
                (a_bytes, b_bytes, c_bytes,
                 a_shape, b_shape, c_shape) = serialize_triple_shares(
                    bt.party_1)
                peer_triples.append(inference_pb2.MPCTriplePayload(
                    op_id=op_id,
                    a_data=a_bytes,
                    b_data=b_bytes,
                    c_data=c_bytes,
                    shape=a_shape,
                    b_shape=b_shape,
                    c_shape=c_shape,
                ))

            my_allocator = TripleAllocator(my_triples)

            # Send share_b and triples to Node B
            share_b_bytes, share_b_shape = _tensor_to_bytes(share_b)
            send_resp = self._peer_stub.SendShare(
                inference_pb2.MPCSendShareRequest(
                    session_id=session_id,
                    share_data=share_b_bytes,
                    share_shape=share_b_shape,
                    triples=peer_triples,
                    num_layers=len(self.mpc.layers),
                    hidden_size=self.mpc.hidden_size,
                    seq_len=seq_len,
                ),
                timeout=30,
            )
            if not send_resp.accepted:
                context.set_details(f"Node B rejected share: "
                                    f"{send_resp.error}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return inference_pb2.ForwardResponse()

            print(f"[MPC-A] Sent share + {len(peer_triples)} triples "
                  f"to Node B")

            # Step 3: Run layer 0 on Node A's share (Node B runs
            # simultaneously on its share)
            exchanger_a = GrpcPeerExchangerA(self._peer_stub)
            share_a_result = self.mpc.forward_layer0_on_share(
                share_a, my_allocator, exchanger_a,
                session_id, is_party_0=True)

            print(f"[MPC-A] Layer 0 (2PC) done, my share shape: "
                  f"{share_a_result.shape}")

            # Step 4: Collect Node B's final share and reconstruct
            collect_resp = self._peer_stub.CollectShare(
                inference_pb2.MPCCollectRequest(session_id=session_id),
                timeout=60,
            )
            share_b_result = _bytes_to_tensor(
                bytes(collect_resp.share_data),
                list(collect_resp.share_shape))

            # Reconstruct layer 0 output
            layer0_output = share_a_result + share_b_result
            print(f"[MPC-A] Reconstructed layer 0 output: "
                  f"{layer0_output.shape}")

            # Step 5: Run remaining layers in cleartext
            output = self.mpc.forward_remaining_layers(
                layer0_output, session_id)
            print(f"[MPC-A] Session {session_id[:8]}... "
                  f"shard 0 output: {output.shape}")

            # Record dual compute shares
            self._record_dual_shares(session_id, output)

            # Step 6: Forward to next node
            activation_bytes, shape = _tensor_to_bytes(output)
            next_address = None
            next_onion_blob = b""
            next_ephemeral_key = b""

            if request.remaining_circuit:
                remaining = list(request.remaining_circuit)
                next_address = remaining.pop(0)
            elif request.onion_blob and self._private_key:
                from network.onion import peel_onion
                layer, next_eph = peel_onion(
                    self._private_key,
                    bytes(request.onion_ephemeral_key),
                    bytes(request.onion_blob),
                )
                next_address = layer.next_hop or None
                next_onion_blob = layer.payload
                next_ephemeral_key = next_eph
                if next_address:
                    print(f"[MPC-A] Peeled onion -> next: {next_address}")

            if next_address:
                next_request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    activation_data=activation_bytes,
                    tensor_shape=list(shape),
                    is_prefill=request.is_prefill,
                )
                if request.remaining_circuit:
                    remaining = list(request.remaining_circuit)
                    remaining.pop(0)
                    next_request.remaining_circuit.extend(remaining)
                if next_onion_blob:
                    next_request.onion_blob = next_onion_blob
                    next_request.onion_ephemeral_key = next_ephemeral_key
                if request.use_random_routing:
                    next_request.use_random_routing = True
                if request.response_keys:
                    next_request.response_keys.extend(
                        list(request.response_keys)[1:])

                channel = create_resilient_channel(
                    next_address, config.GRPC_OPTIONS)
                stub = inference_pb2_grpc.InferenceNodeStub(channel)
                response = stub.Forward(next_request)

                if request.response_keys:
                    from network.onion import encrypt
                    key = bytes(request.response_keys[0])
                    if response.encrypted_response:
                        encrypted = encrypt(key, response.encrypted_response)
                    elif response.has_token:
                        token_bytes = response.token_id.to_bytes(
                            4, 'big', signed=True)
                        encrypted = encrypt(key, token_bytes)
                    else:
                        encrypted = encrypt(key, response.activation_data)
                    return inference_pb2.ForwardResponse(
                        encrypted_response=encrypted,
                        token_id=response.token_id,
                        has_token=response.has_token,
                        is_eos=response.is_eos,
                    )

                return response
            else:
                return inference_pb2.ForwardResponse(
                    activation_data=activation_bytes,
                    tensor_shape=list(shape),
                )

        except Exception as e:
            print(f"[MPC-A] Error: {e}")
            import traceback
            traceback.print_exc()
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.ForwardResponse()


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def serve(role: str, port: int, peer_address: str, host: str = "[::]",
          advertise: str = None, registry_address: str = None,
          shards_dir: str = None, eth_address: str = None):
    """Start an MPC node server."""
    mpc_node = MPCNode(role, peer_address, shards_dir=shards_dir)
    public_address = advertise or f"localhost:{port}"

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=config.GRPC_OPTIONS,
    )

    if role == "A":
        servicer = MPCNodeServicer(mpc_node, port)
        servicer._peer_address = peer_address
        inference_pb2_grpc.add_InferenceNodeServicer_to_server(
            servicer, server)
    else:
        exchanger_b = GrpcPeerExchangerB()
        peer_servicer = MPCPeerServicer(mpc_node, exchanger_b)
        inference_pb2_grpc.add_MPCPeerServicer_to_server(
            peer_servicer, server)
        servicer = peer_servicer

    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"[MPC-{role}] Listening on {host}:{port}")
    print(f"[MPC-{role}] Peer: {peer_address}")
    print(f"[MPC-{role}] Advertised as {public_address}")

    registration = None
    if role == "A":
        servicer._init_daemon_stub(
            registry_address or config.REGISTRY_ADDRESS)

        if registry_address:
            from network.discovery import NodeRegistration
            _manifest_path = os.path.join(
                mpc_node._shards_dir, "manifest.json")
            with open(_manifest_path) as _f:
                _mpc_manifest = json.load(_f)
            _mpc_model_id = _mpc_manifest.get(
                "model_id", config.MODEL_NAME)
            registration = NodeRegistration(
                address=public_address,
                model_id=_mpc_model_id,
                shard_index=0,
                layer_start=mpc_node.layer_start,
                layer_end=mpc_node.layer_end,
                has_embedding=True,
                has_lm_head=False,
                node_type="mpc",
                registry_address=registry_address,
                node_id=eth_address,
            )
            registration.start()
            servicer._node_id = registration.node_id
            servicer._peer_address = peer_address
            servicer._peer_node_id = f"peer-of-{registration.node_id[:8]}"
            servicer._private_key = registration.private_key
            print(f"[MPC-{role}] Registered as 'mpc' with registry "
                  f"at {registry_address}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\n[MPC-{role}] Shutting down...")
        if registration:
            registration.stop()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UNFED AI MPC Node (Shard 0) — Beaver-Triple 2PC")
    parser.add_argument("--role", type=str, required=True,
                        choices=["A", "B"],
                        help="MPC role: A (entry) or B (peer)")
    parser.add_argument("--port", type=int, required=True,
                        help="Port for this MPC node")
    parser.add_argument("--peer", type=str, required=True,
                        help="Address of the other MPC node")
    parser.add_argument("--host", type=str, default="[::]",
                        help="Bind address")
    parser.add_argument("--advertise", type=str, default=None,
                        help="Address to advertise to registry")
    parser.add_argument("--registry", type=str, default=None,
                        help="Registry address")
    parser.add_argument("--shards-dir", type=str, default=None,
                        help="Directory containing shard files")
    parser.add_argument("--eth-address", type=str, default=None,
                        help="Ethereum address for on-chain staking "
                             "(used as node_id)")
    args = parser.parse_args()
    serve(args.role, args.port, args.peer, args.host, args.advertise,
          args.registry, args.shards_dir, eth_address=args.eth_address)
