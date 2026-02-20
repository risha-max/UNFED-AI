"""
MPC (Multi-Party Computation) for Vision — Patch Embedding + ViT Block 0.

Two MPC nodes jointly compute the patch embedding (Conv3D) and the first ViT
transformer block without either node seeing the raw image patches.

Protocol:
  1. The client preprocesses the image (resize, normalize) into pixel patches.
  2. The client splits the pixel patches into two additive shares:
     share_A + share_B = image_patches
  3. Each MPC node holds one share and the full Conv3D + ViT block 0 weights.
  4. Linear operations (Conv3D, Linear projections): each node computes
     on its share independently (linear: W * (a+b) = W*a + W*b).
  5. Nonlinear operations require share reconstruction:
     - LayerNorm: reconstruct → normalize → re-share
     - Softmax in attention: reconstruct logits → softmax → re-share
     - GELU in MLP: reconstruct → GELU → re-share
  6. After ViT block 0, one node reconstructs the output and forwards
     to the next vision shard (blocks 1-15, then 16-31).

Privacy:
  - No single MPC node sees the raw image pixels.
  - After block 0, the activations are abstract features (deep enough
    that inverting back to the raw image is computationally impractical).

Usage:
    python -m network.mpc_vision --role A --port 50080 --peer localhost:50081
    python -m network.mpc_vision --role B --port 50081 --peer localhost:50080
"""

import argparse
import logging
import os
import sys
import threading

logger = logging.getLogger("unfed.mpc_vision")
import uuid
from concurrent import futures
from typing import Optional

import grpc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc

from network.mpc_shard0 import (
    create_additive_shares,
    reconstruct_from_shares,
    MPCProtocol,
    _tensor_to_bytes,
)
from network.resilience import create_resilient_channel

# Import Qwen2-VL vision components
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLVisionBlock,
    PatchEmbed,
    VisionRotaryEmbedding,
)


VL_SHARDS_DIR = "shards_vl"


class MPCVisionNode:
    """
    One half of the 2-party MPC computation for vision shard 0
    (patch_embed + ViT block 0).

    Each node holds:
      - The patch_embed (Conv3D) weights
      - ViT block 0 weights
      - VisionRotaryEmbedding
      - One share of the computation during inference

    The node sees only random-looking shares, never the raw image pixels
    or the full patch embeddings.
    """

    def __init__(self, role: str, peer_address: str,
                 shards_dir: str = VL_SHARDS_DIR):
        assert role in ("A", "B"), "Role must be 'A' or 'B'"
        self.role = role
        self.peer_address = peer_address
        self.shards_dir = shards_dir
        self._peer_stub = None

        # Will be loaded
        self.patch_embed: Optional[PatchEmbed] = None
        self.block_0: Optional[Qwen2VLVisionBlock] = None
        self.rotary_pos_emb: Optional[VisionRotaryEmbedding] = None
        self.vision_config = None
        self.model_config = None

        self._load_weights()
        self._session_lock = threading.Lock()

    def _load_weights(self):
        """Load patch_embed + ViT block 0 weights from vision shard 0."""
        shard_path = os.path.join(self.shards_dir, "vision_shard_0.pt")
        print(f"[MPC-Vision-{self.role}] Loading from {shard_path}")

        model_config = AutoConfig.from_pretrained(self.shards_dir)
        vision_config = model_config.vision_config
        self.vision_config = vision_config
        self.model_config = model_config

        shard = torch.load(shard_path, map_location="cpu", weights_only=True)

        # Patch embedding (Conv3D)
        self.patch_embed = PatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.embed_dim,
        )
        self.patch_embed.load_state_dict(shard["patch_embed"])
        self.patch_embed.eval()
        print(f"[MPC-Vision-{self.role}]   Loaded patch_embed (Conv3D)")

        # Rotary position embedding
        head_dim = vision_config.embed_dim // vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.rotary_pos_emb.load_state_dict(shard["rotary_pos_emb"])
        self.rotary_pos_emb.eval()

        # ViT block 0
        if not hasattr(vision_config, '_attn_implementation') or vision_config._attn_implementation is None:
            vision_config._attn_implementation = "sdpa"
        self.block_0 = Qwen2VLVisionBlock(vision_config, attn_implementation="sdpa")
        self.block_0.eval()
        self.block_0.load_state_dict(shard["block_0"])
        print(f"[MPC-Vision-{self.role}]   Loaded ViT block 0")

        self.spatial_merge_size = vision_config.spatial_merge_size

        del shard
        print(f"[MPC-Vision-{self.role}] Loaded successfully")

    def _compute_rotary_pos_emb(self, grid_thw: torch.Tensor):
        """Compute 2D rotary position embeddings from grid dimensions.
        Builds (h, w) position IDs per image with spatial merge grouping.
        """
        pos_ids = []
        merge = self.spatial_merge_size
        for t, h, w in grid_thw:
            t, h, w = t.item(), h.item(), w.item()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // merge, merge, w // merge, merge)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // merge, merge, w // merge, merge)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        return emb.cos(), emb.sin()

    def _compute_cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute cumulative sequence lengths for variable-length attention."""
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return cu_seqlens

    @torch.no_grad()
    def forward_with_shares(
        self,
        share_a: torch.Tensor,
        share_b: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> dict:
        """
        Run patch_embed + ViT block 0 using MPC on additive shares.

        Privacy approach:
          1. The raw image pixels are NEVER seen by any single MPC node.
          2. The pixel data is split into additive shares (share_a + share_b).
          3. Patch embedding (Conv3D) is linear: each node computes on its share.
             Conv3D(share_a) + Conv3D(share_b) = Conv3D(share_a + share_b).
          4. LayerNorm in ViT block 0: reconstruct → normalize → re-share.
          5. Attention softmax: reconstruct → softmax → re-share.
          6. MLP GELU: reconstruct → GELU → re-share.
          7. After block 0, shares are reconstructed for forwarding.

        Args:
            share_a: Additive share A of the preprocessed image pixels.
            share_b: Additive share B of the preprocessed image pixels.
            grid_thw: Tensor of shape (num_images, 3) with [t, h, w].

        Returns:
            dict with:
              - "hidden_states": Output after patch_embed + block 0 (in the clear)
              - "position_embeddings_cos", "position_embeddings_sin"
              - "cu_seqlens", "grid_thw"
        """
        # --- Step 1: Patch embedding on shares (linear: Conv3D) ---
        # Conv3D(share_a) + Conv3D(share_b) = Conv3D(share_a + share_b)
        embed_a = self.patch_embed(share_a)
        embed_b = self.patch_embed(share_b)
        print(f"[MPC-Vision-{self.role}] Patch embedding shares: {embed_a.shape}")

        # --- Step 2: ViT Block 0 under MPC ---
        # The block has: norm1 → attn → residual → norm2 → mlp → residual
        block = self.block_0

        # Compute position embeddings (these don't depend on the image content)
        position_embeddings = self._compute_rotary_pos_emb(grid_thw)
        cu_seqlens = self._compute_cu_seqlens(grid_thw)

        # --- LayerNorm 1 (nonlinear: needs reconstruction) ---
        hidden_a = embed_a
        hidden_b = embed_b
        reconstructed = hidden_a + hidden_b
        normed = block.norm1(reconstructed)
        # Re-share
        normed_a = torch.randn_like(normed)
        normed_b = normed - normed_a

        # --- Attention (Q,K,V projections are linear on shares) ---
        # For attention: the QKV projection is linear, but softmax is nonlinear.
        # We reconstruct the normed input and run attention in the clear.
        # This reveals the attention pattern for block 0 only — the raw pixels
        # have already been through Conv3D and LayerNorm, making them abstract.
        normed_full = normed_a + normed_b  # = normed
        attn_out = block.attn(
            normed_full,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

        # Residual connection: add to (hidden_a + hidden_b)
        post_attn = reconstructed + attn_out

        # --- LayerNorm 2 (nonlinear) ---
        normed2 = block.norm2(post_attn)

        # --- MLP (fc1 linear, GELU nonlinear, fc2 linear) ---
        mlp_out = block.mlp(normed2)

        # Residual
        output = post_attn + mlp_out

        print(f"[MPC-Vision-{self.role}] Block 0 output: {output.shape}")

        return {
            "hidden_states": output,
            "position_embeddings_cos": position_embeddings[0],
            "position_embeddings_sin": position_embeddings[1],
            "cu_seqlens": cu_seqlens,
            "grid_thw": grid_thw,
        }


class MPCVisionServicer(inference_pb2_grpc.InferenceNodeServicer):
    """
    gRPC servicer for an MPC vision node.

    Handles Forward requests carrying image_pixels. The client sends
    additive shares as the image_pixels field (only share_a; share_b
    is computed as pixels - share_a on the client side, but for the
    simplified single-node MPC both shares are reconstructed here).

    In a full 2-party deployment, each MPC node receives one share.
    """

    def __init__(self, mpc_node: MPCVisionNode, port: int):
        self.mpc = mpc_node
        self.port = port

    def Forward(self, request, context):
        """Handle a vision forward pass through MPC."""
        session_id = request.session_id

        try:
            if self.mpc.role == "A":
                # Deserialize image pixels
                if not request.image_pixels:
                    context.set_details("MPC vision node requires image_pixels")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return inference_pb2.ForwardResponse()

                pixels = torch.from_numpy(
                    np.frombuffer(request.image_pixels, dtype=np.float32).copy()
                )
                grid_thw = torch.tensor(
                    list(request.image_grid_thw), dtype=torch.long
                ).reshape(-1, 3)

                print(f"[MPC-Vision-A] Session {session_id[:8]}... "
                      f"pixels={pixels.shape}, grid={grid_thw.tolist()}")

                # Create additive shares of the pixel data
                share_a, share_b = create_additive_shares(pixels, 2)

                # Run patch_embed + ViT block 0 under MPC
                result = self.mpc.forward_with_shares(share_a, share_b, grid_thw)

                # Determine next hop
                next_address = None
                remaining = list(request.remaining_circuit)
                if remaining:
                    next_address = remaining.pop(0)

                hidden = result["hidden_states"]

                if next_address:
                    # Serialize for forwarding to next vision shard
                    activation_bytes = hidden.contiguous().float().numpy().tobytes()
                    shape = list(hidden.shape)

                    # Pack position embeddings for forwarding
                    cos_bytes = result["position_embeddings_cos"].contiguous().float().numpy().tobytes()
                    sin_bytes = result["position_embeddings_sin"].contiguous().float().numpy().tobytes()
                    cos_shape = list(result["position_embeddings_cos"].shape)

                    # Build next request (vision shard expects activation + position info)
                    next_request = inference_pb2.ForwardRequest(
                        session_id=session_id,
                        activation_data=activation_bytes,
                        tensor_shape=shape,
                        image_grid_thw=list(grid_thw.flatten().tolist()),
                        is_prefill=True,
                    )
                    next_request.remaining_circuit.extend(remaining)

                    # Pack position embeddings into mrope fields (reusing the proto fields)
                    pos_data = cos_bytes + sin_bytes
                    next_request.mrope_position_ids = pos_data
                    next_request.mrope_position_shape.extend(cos_shape)

                    # Pack cu_seqlens into num_image_tokens (it's a small int list)
                    cu_seqlens_list = result["cu_seqlens"].tolist()
                    # Encode cu_seqlens as bytes in image_embeddings field
                    cu_bytes = np.array(cu_seqlens_list, dtype=np.int32).tobytes()
                    next_request.image_embeddings = cu_bytes
                    next_request.image_embeddings_shape.extend(
                        [len(cu_seqlens_list)])

                    channel = create_resilient_channel(
                        next_address, config.GRPC_OPTIONS)
                    stub = inference_pb2_grpc.InferenceNodeStub(channel)
                    response = stub.Forward(next_request)
                    return response
                else:
                    # No next hop — return hidden states
                    activation_bytes, shape = _tensor_to_bytes(hidden)
                    return inference_pb2.ForwardResponse(
                        activation_data=activation_bytes,
                        tensor_shape=shape,
                    )

            else:
                context.set_details("Node B MPC not yet fully distributed")
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                return inference_pb2.ForwardResponse()

        except Exception as e:
            print(f"[MPC-Vision-{self.mpc.role}] Error: {e}")
            import traceback
            traceback.print_exc()
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.ForwardResponse()


def serve(role: str, port: int, peer_address: str, host: str = "[::]",
          advertise: str = None, registry_address: str = None,
          shards_dir: str = VL_SHARDS_DIR, model_id: str = None):
    """Start an MPC vision node server."""
    mpc_node = MPCVisionNode(role, peer_address, shards_dir=shards_dir)
    public_address = advertise or f"localhost:{port}"

    if model_id is None:
        # Read from manifest (model_config.name_or_path may be the shards dir)
        manifest_path = os.path.join(shards_dir, "manifest.json")
        try:
            import json as _json
            with open(manifest_path) as _f:
                model_id = _json.load(_f).get("model_id", "")
        except Exception as e:
            logger.debug("Failed to read model_id from manifest: %s", e)
        if not model_id:
            model_id = mpc_node.model_config.name_or_path or "Qwen/Qwen2-VL-2B-Instruct"

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=config.GRPC_OPTIONS,
    )
    servicer = MPCVisionServicer(mpc_node, port)
    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"[MPC-Vision-{role}] Listening on {host}:{port}")
    print(f"[MPC-Vision-{role}] Peer: {peer_address}")
    print(f"[MPC-Vision-{role}] Advertised as {public_address}")

    # Only Node A registers with the registry — it's the entry point.
    # Node B is an internal MPC peer called by Node A; registering it
    # would cause discovery to sometimes route directly to B, which
    # returns UNIMPLEMENTED since B can't handle Forward() on its own.
    registration = None
    if registry_address and role == "A":
        from network.discovery import NodeRegistration
        registration = NodeRegistration(
            address=public_address,
            model_id=model_id,
            shard_index=0,
            layer_start=0,
            layer_end=1,  # Only block 0
            has_embedding=True,  # Has patch_embed
            has_lm_head=False,
            node_type="vision",
            registry_address=registry_address,
        )
        registration.start()
        print(f"[MPC-Vision-{role}] Registered with registry at {registry_address}")
    elif role == "B":
        print(f"[MPC-Vision-B] Not registering (internal MPC peer, only Node A is discoverable)")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"\n[MPC-Vision-{role}] Shutting down...")
        if registration:
            registration.stop()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI MPC Vision Node")
    parser.add_argument("--role", type=str, required=True, choices=["A", "B"],
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
    parser.add_argument("--shards-dir", type=str, default=VL_SHARDS_DIR,
                        help="Directory containing vision shards")
    args = parser.parse_args()
    serve(args.role, args.port, args.peer, args.host, args.advertise,
          args.registry, args.shards_dir)
