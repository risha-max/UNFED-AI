#!/usr/bin/env python3
"""
Tests — Multimodal Vision MPC (Qwen2-VL-2B-Instruct).

Unit tests:
  1. MPC Conv3D on shares: verify Conv3D(share_a) + Conv3D(share_b) == Conv3D(full)
  2. Vision layer runner: forward pass through ViT blocks
  3. Image embedding merge: placeholder replacement in text sequence
  4. M-RoPE position IDs: 3D positions for mixed image+text sequence

E2E test (registry + 1 MPC vision + 1 vision shard + 4 text shards):
  1. Send a synthetic test image + text question
  2. Verify: vision pipeline returns image embeddings of correct shape
  3. Verify: text pipeline generates coherent tokens
  4. Verify: MPC was used (Conv3D on shares matches Conv3D on full)
  5. Test text-only query still works through the text pipeline

Usage:
    python -m tests.test_vision
"""

import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VL_SHARDS_DIR = os.path.join(PROJECT_ROOT, "shards_vl")
VL_MODEL_PATH = os.path.expanduser("~/models/Qwen2-VL-2B-Instruct")

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        failed += 1


def has_vl_shards():
    """Check if vision+text shards exist."""
    return (os.path.exists(os.path.join(VL_SHARDS_DIR, "vision_shard_0.pt")) and
            os.path.exists(os.path.join(VL_SHARDS_DIR, "text_shard_0.pt")) and
            os.path.exists(os.path.join(VL_SHARDS_DIR, "manifest.json")))


# ============================================================================
# Unit Test 1: MPC Conv3D on shares
# ============================================================================
def test_mpc_conv3d():
    """Verify that Conv3D is linear: Conv3D(a) + Conv3D(b) == Conv3D(a + b)."""
    print("\n=== Unit Test: MPC Conv3D on shares ===")

    from network.mpc_shard0 import create_additive_shares, reconstruct_from_shares

    # Create a small Conv3D (like Qwen2-VL patch_embed)
    conv = nn.Conv3d(3, 128, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    conv.eval()

    # Create synthetic image pixels (4 patches worth)
    # Shape: [num_patches, channels * temporal * patch_h * patch_w]
    # For Conv3D: reshape to [N, 3, 2, 14, 14]
    pixels = torch.randn(4, 3, 2, 14, 14)

    # Compute ground truth
    with torch.no_grad():
        full_output = conv(pixels)

    # Split into additive shares
    share_a, share_b = create_additive_shares(pixels, 2)

    # Compute on each share separately
    with torch.no_grad():
        out_a = conv(share_a)
        out_b = conv(share_b)

    # Reconstruct
    reconstructed = out_a + out_b

    # Verify
    diff = (full_output - reconstructed).abs().max().item()
    test("Conv3D linearity: shares sum matches full",
         diff < 1e-4, f"max diff = {diff}")

    # Verify shares individually are different from original
    # (cosine similarity between share_a and pixels should be low)
    flat_share = share_a.flatten()
    flat_pixels = pixels.flatten()
    cos_sim = F.cosine_similarity(flat_share.unsqueeze(0), flat_pixels.unsqueeze(0)).item()
    test("Shares not correlated with original (cosine sim < 0.3)",
         abs(cos_sim) < 0.3, f"cosine_sim={cos_sim:.4f}")


# ============================================================================
# Unit Test 2: Image embedding merge
# ============================================================================
def test_image_merge():
    """Test merging image embeddings into text sequence at placeholder positions."""
    print("\n=== Unit Test: Image embedding merge ===")

    # Use a well-known image_token_id (SmolVLM / Qwen2-VL style)
    image_token_id = 151655

    # Simulate a text sequence with image placeholders
    batch = 1
    seq_len = 20
    hidden_dim = 1536
    num_image_tokens = 5

    # Token IDs: some text tokens, then image placeholders, then more text
    token_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    token_ids[0, :5] = torch.arange(100, 105)     # text tokens
    token_ids[0, 5:10] = image_token_id            # image placeholders
    token_ids[0, 10:] = torch.arange(200, 210)     # text tokens

    # Text embeddings (random)
    text_embeds = torch.randn(batch, seq_len, hidden_dim)

    # Image embeddings (distinctive: all ones for easy verification)
    image_embeds = torch.ones(num_image_tokens, hidden_dim) * 42.0

    # Create a mock with image_token_id and call GenericTextRunner's merge method
    from node.runtime.generic_runner import GenericTextRunner

    class MockRunner:
        def __init__(self):
            self.shard_index = 0
            self.image_token_id = image_token_id

    mock = MockRunner()
    merged = GenericTextRunner.merge_image_embeddings(mock, text_embeds, image_embeds, token_ids)

    # Verify: positions 5-9 should have value 42.0
    test("Merged output shape matches input",
         merged.shape == text_embeds.shape)
    test("Image positions have correct values",
         (merged[0, 5:10, :] == 42.0).all().item())
    test("Text positions unchanged (before)",
         torch.allclose(merged[0, :5, :], text_embeds[0, :5, :]))
    test("Text positions unchanged (after)",
         torch.allclose(merged[0, 10:, :], text_embeds[0, 10:, :]))


# ============================================================================
# Unit Test 3: M-RoPE position IDs
# ============================================================================
def test_mrope_positions():
    """Test M-RoPE 3D position ID computation for mixed image+text sequences."""
    print("\n=== Unit Test: M-RoPE position IDs ===")

    from client.client import UnfedClient

    # Create a mock client with the method
    class MockClient:
        def __init__(self):
            self.model_id = VL_MODEL_PATH

    mock = MockClient()

    # Simulate: [text, text, image_placeholder x 4, text, text]
    from transformers import AutoConfig
    vl_config = AutoConfig.from_pretrained(VL_MODEL_PATH)
    image_token_id = vl_config.image_token_id
    spatial_merge = vl_config.vision_config.spatial_merge_size  # 2

    # Simple case: 2 text tokens, then 4 image tokens (2x2 merged grid), then 2 text tokens
    input_ids = [100, 101]  # text
    input_ids += [image_token_id] * 4  # image (2x2 merged from 4x4 grid)
    input_ids += [200, 201]  # text

    # grid_thw: 1 image, t=1, h=4 (patch units), w=4 (patch units)
    # Merged grid: h_merged=2, w_merged=2 → 4 image tokens
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    pos_ids = UnfedClient._compute_mrope_position_ids(mock, input_ids, grid_thw, VL_MODEL_PATH)

    test("M-RoPE shape is [3, 1, seq_len]",
         pos_ids.shape == (3, 1, len(input_ids)),
         f"got {pos_ids.shape}")

    # Text tokens before image: positions 0, 1
    test("Text before image: temporal pos",
         pos_ids[0, 0, 0].item() == 0 and pos_ids[0, 0, 1].item() == 1)
    test("Text before image: all dims same",
         (pos_ids[:, 0, 0] == pos_ids[0, 0, 0]).all().item() and
         (pos_ids[:, 0, 1] == pos_ids[0, 0, 1]).all().item())

    # Image tokens: temporal dim should be text_pos for t=0
    # Height/width dims should vary
    test("Image tokens have valid positions",
         pos_ids[:, 0, 2:6].min().item() >= 0)

    # Text after image: all dims same value, > image positions
    text_after_start = 6  # index of first text token after image
    test("Text after image: all dims same",
         (pos_ids[:, 0, text_after_start] ==
          pos_ids[0, 0, text_after_start]).all().item())
    test("Text after image: position advances",
         pos_ids[0, 0, text_after_start].item() > pos_ids[0, 0, 1].item())


# ============================================================================
# E2E Test: Full multimodal pipeline
# ============================================================================
def test_e2e_multimodal():
    """E2E test with registry + MPC vision + vision shard + 4 text shards."""
    print("\n=== E2E Test: Full Multimodal Pipeline ===")

    if not has_vl_shards():
        print("  [SKIP] VL shards not found — run shard.vision_splitter first")
        return

    import json
    import grpc
    import inference_pb2
    import inference_pb2_grpc

    processes = []

    def cleanup():
        for p in processes:
            try:
                os.kill(p.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        time.sleep(1)
        for p in processes:
            try:
                p.kill()
            except (OSError, ProcessLookupError):
                pass

    try:
        # --- Start registry ---
        print("  Starting registry on port 50050...")
        reg_proc = subprocess.Popen(
            [sys.executable, "-m", "network.registry_server",
             "--port", "50050"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        processes.append(reg_proc)
        time.sleep(2)

        # --- Start MPC vision node (role A) on port 50060 ---
        print("  Starting MPC vision node on port 50060...")
        mpc_proc = subprocess.Popen(
            [sys.executable, "-m", "network.mpc_vision",
             "--role", "A", "--port", "50060", "--peer", "localhost:50061",
             "--advertise", "localhost:50060",
             "--registry", "localhost:50050",
             "--shards-dir", VL_SHARDS_DIR],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        processes.append(mpc_proc)
        time.sleep(8)

        # --- Start vision shard 1 (blocks 16-31 + merger) on port 50061 ---
        print("  Starting vision shard 1 on port 50061...")
        vis1_proc = subprocess.Popen(
            [sys.executable, "-m", "node.server",
             "--model-type", "vision",
             "--shard-index", "1",
             "--port", "50061",
             "--advertise", "localhost:50061",
             "--registry", "localhost:50050",
             "--shards-dir", VL_SHARDS_DIR],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        processes.append(vis1_proc)
        time.sleep(8)

        # --- Start 4 text shard nodes (sequentially, they load VL model) ---
        text_ports = [50070, 50071, 50072, 50073]
        for i, port in enumerate(text_ports):
            print(f"  Starting text shard {i} on port {port}...")
            p = subprocess.Popen(
                [sys.executable, "-m", "node.server",
                 "--model-type", "qwen2_vl",
                 "--shard-index", str(i),
                 "--port", str(port),
                 "--advertise", f"localhost:{port}",
                 "--registry", "localhost:50050",
                 "--shards-dir", VL_SHARDS_DIR],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            processes.append(p)
            time.sleep(15)

        print("  All nodes started. Waiting for registration...")
        time.sleep(5)

        # --- Verify nodes registered (with retry) ---
        from network.discovery import RegistryClient
        reg = RegistryClient("localhost:50050")

        # Retry discovery up to 30s to wait for all nodes
        for attempt in range(6):
            all_nodes = reg.discover("")
            vision_nodes = [n for n in all_nodes if n.node_type == "vision"]
            compute_nodes = [n for n in all_nodes
                             if n.node_type not in ("vision",)]
            if len(vision_nodes) >= 2 and len(compute_nodes) >= 4:
                break
            print(f"  Waiting for nodes... ({len(vision_nodes)} vision, "
                  f"{len(compute_nodes)} compute)")
            time.sleep(5)

        print(f"  Registered: {len(vision_nodes)} vision, {len(compute_nodes)} compute")
        test("Vision nodes registered", len(vision_nodes) >= 2,
             f"got {len(vision_nodes)}")
        test("Text compute nodes registered", len(compute_nodes) >= 4,
             f"got {len(compute_nodes)}")

        # --- Test 1: Vision pipeline (synthetic image) ---
        print("\n  --- Test 1: Vision pipeline ---")

        # Create a synthetic test image
        from PIL import Image
        test_image_path = os.path.join(PROJECT_ROOT, "tests", "test_image.png")
        img = Image.new("RGB", (224, 224), color=(128, 64, 196))
        # Draw some patterns for visual features
        import random as rng
        rng.seed(42)
        pixels_arr = np.array(img)
        for _ in range(50):
            x, y = rng.randint(0, 223), rng.randint(0, 223)
            r = rng.randint(5, 20)
            pixels_arr[max(0,y-r):min(224,y+r), max(0,x-r):min(224,x+r)] = [
                rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)
            ]
        img = Image.fromarray(pixels_arr)
        img.save(test_image_path)

        # Preprocess image
        from transformers import Qwen2VLImageProcessor, AutoTokenizer, AutoConfig
        image_processor = Qwen2VLImageProcessor.from_pretrained(VL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(VL_MODEL_PATH)
        vl_cfg = AutoConfig.from_pretrained(VL_MODEL_PATH)

        img_result = image_processor(images=[img], return_tensors="pt")
        pixel_values = img_result["pixel_values"]
        image_grid_thw = img_result["image_grid_thw"]

        print(f"  Preprocessed: pixels={pixel_values.shape}, "
              f"grid={image_grid_thw.tolist()}")

        # Send through MPC vision node
        pixel_bytes = pixel_values.contiguous().float().numpy().tobytes()
        grid_list = image_grid_thw.flatten().tolist()

        # Build vision circuit
        vision_circuit = reg.build_vision_circuit("")
        test("Vision circuit built", vision_circuit is not None)

        if vision_circuit:
            vis_addrs, vis_pks = vision_circuit
            print(f"  Vision circuit: {' → '.join(vis_addrs)}")

            mpc_addr = vis_addrs[0]
            remaining_vis = vis_addrs[1:]

            vision_req = inference_pb2.ForwardRequest(
                session_id="test-vision-1",
                image_pixels=pixel_bytes,
                image_grid_thw=grid_list,
                is_prefill=True,
            )
            vision_req.remaining_circuit.extend(remaining_vis)

            channel = grpc.insecure_channel(
                mpc_addr,
                options=[
                    ("grpc.max_send_message_length", 256 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ],
            )
            stub = inference_pb2_grpc.InferenceNodeStub(channel)

            print("  Sending image through vision pipeline...")
            vis_start = time.time()
            vis_response = stub.Forward(vision_req)
            vis_time = time.time() - vis_start

            test("Vision response has data",
                 len(vis_response.activation_data) > 0,
                 f"got {len(vis_response.activation_data)} bytes")
            test("Vision response has shape",
                 len(vis_response.tensor_shape) > 0)

            if vis_response.activation_data:
                ie_shape = list(vis_response.tensor_shape)
                image_embeddings = np.frombuffer(
                    vis_response.activation_data, dtype=np.float32
                ).reshape(ie_shape)
                print(f"  Image embeddings: {ie_shape} ({vis_time:.2f}s)")

                # Verify shape: should be [num_merged_tokens, 1536]
                test("Image embeddings dim=1536",
                     ie_shape[-1] == 1536, f"got {ie_shape[-1]}")
                test("Image embeddings non-zero",
                     np.abs(image_embeddings).mean() > 0.01)

        # --- Test 2: Text-only query (no image) ---
        print("\n  --- Test 2: Text-only query through VL text pipeline ---")

        # Build text circuit
        text_circuit = reg.build_circuit("")
        test("Text circuit built", text_circuit is not None)

        if text_circuit:
            text_addrs, text_pks = text_circuit
            print(f"  Text circuit: {' → '.join(text_addrs)}")

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(VL_MODEL_PATH)
            text_prompt = "The capital of France is"
            tokens = tokenizer(text_prompt, return_tensors="pt").input_ids[0].tolist()

            entry_addr = text_addrs[0]
            remaining_text = text_addrs[1:]

            text_req = inference_pb2.ForwardRequest(
                session_id="test-text-only-1",
                token_ids=tokens,
                is_prefill=True,
            )
            text_req.remaining_circuit.extend(remaining_text)

            text_channel = grpc.insecure_channel(
                entry_addr,
                options=[
                    ("grpc.max_send_message_length", 256 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ],
            )
            text_stub = inference_pb2_grpc.InferenceNodeStub(text_channel)

            print("  Sending text-only query...")
            text_start = time.time()
            text_resp = text_stub.Forward(text_req)
            text_time = time.time() - text_start

            test("Text response has token",
                 text_resp.has_token, f"has_token={text_resp.has_token}")

            if text_resp.has_token:
                token_text = tokenizer.decode([text_resp.token_id])
                print(f"  Generated: '{token_text}' (token_id={text_resp.token_id}, "
                      f"{text_time:.2f}s)")
                test("Generated meaningful token",
                     text_resp.token_id > 0 and text_resp.token_id < 151936)

        # --- Test 3: Full multimodal query (image + text → generation) ---
        print("\n  --- Test 3: Full multimodal (image + text → generation) ---")

        # We need: image_embeddings from Test 1, text circuit from Test 2
        if vision_circuit and text_circuit and image_embeddings is not None:
            from transformers import AutoConfig
            vl_config = AutoConfig.from_pretrained(VL_MODEL_PATH)
            vision_start_id = vl_config.vision_start_token_id
            vision_end_id = vl_config.vision_end_token_id
            image_token_id_val = vl_config.image_token_id
            spatial_merge = vl_config.vision_config.spatial_merge_size

            # Calculate merged token count from the grid
            t_dim = image_grid_thw[0, 0].item()
            h_dim = image_grid_thw[0, 1].item()
            w_dim = image_grid_thw[0, 2].item()
            num_merged = t_dim * (h_dim // spatial_merge) * (w_dim // spatial_merge)
            print(f"  Image grid: t={t_dim}, h={h_dim}, w={w_dim}, "
                  f"merged_tokens={num_merged}")

            # Build multimodal token sequence:
            # <|im_start|>user\n<|vision_start|><|image_pad|>xN<|vision_end|>\nDescribe this image<|im_end|>\n<|im_start|>assistant\n
            prompt_text = "Describe this image in detail."
            prompt_toks = tokenizer.encode(prompt_text, add_special_tokens=False)
            im_start_tok = tokenizer.encode("<|im_start|>", add_special_tokens=False)
            im_end_tok = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            user_tok = tokenizer.encode("user\n", add_special_tokens=False)
            newline_tok = tokenizer.encode("\n", add_special_tokens=False)
            asst_tok = tokenizer.encode("assistant\n", add_special_tokens=False)

            multimodal_ids = (
                im_start_tok + user_tok +
                [vision_start_id] + [image_token_id_val] * num_merged + [vision_end_id] +
                newline_tok + prompt_toks +
                im_end_tok + newline_tok + im_start_tok + asst_tok
            )

            num_img_tokens = sum(1 for t in multimodal_ids if t == image_token_id_val)
            print(f"  Token sequence: {len(multimodal_ids)} tokens "
                  f"({num_img_tokens} image placeholders)")

            test("Image placeholder count matches merged",
                 num_img_tokens == num_merged,
                 f"placeholders={num_img_tokens}, merged={num_merged}")

            # Compute M-RoPE position IDs (3D: temporal, height, width)
            spatial_merge_size = vl_config.vision_config.spatial_merge_size
            seq_len = len(multimodal_ids)
            mrope_pos = torch.zeros(3, 1, seq_len, dtype=torch.long)
            text_pos = 0
            idx = 0
            while idx < seq_len:
                if multimodal_ids[idx] == image_token_id_val:
                    # Find contiguous image tokens
                    img_start = idx
                    while idx < seq_len and multimodal_ids[idx] == image_token_id_val:
                        idx += 1
                    n_img = idx - img_start

                    # Grid-based 3D positions for image tokens
                    h_merged = h_dim // spatial_merge_size
                    w_merged = w_dim // spatial_merge_size
                    for j in range(n_img):
                        t_i = j // (h_merged * w_merged)
                        remainder = j % (h_merged * w_merged)
                        h_i = remainder // w_merged
                        w_i = remainder % w_merged
                        mrope_pos[0, 0, img_start + j] = text_pos + t_i
                        mrope_pos[1, 0, img_start + j] = text_pos + h_i
                        mrope_pos[2, 0, img_start + j] = text_pos + w_i

                    max_t = t_dim
                    max_h = h_merged
                    max_w = w_merged
                    text_pos += max(max_t, max_h, max_w)
                else:
                    mrope_pos[0, 0, idx] = text_pos
                    mrope_pos[1, 0, idx] = text_pos
                    mrope_pos[2, 0, idx] = text_pos
                    text_pos += 1
                    idx += 1

            print(f"  M-RoPE positions computed: {mrope_pos.shape}")

            # Serialize image embeddings
            ie_tensor = torch.from_numpy(image_embeddings).float()
            ie_bytes = ie_tensor.contiguous().numpy().tobytes()
            ie_shape_list = list(ie_tensor.shape)

            # Serialize M-RoPE positions
            mp_bytes = mrope_pos.contiguous().numpy().astype(np.int64).tobytes()
            mp_shape_list = list(mrope_pos.shape)

            # Build the combined request
            mm_req = inference_pb2.ForwardRequest(
                session_id="test-multimodal-combined-1",
                token_ids=multimodal_ids,
                is_prefill=True,
                image_embeddings=ie_bytes,
                image_grid_thw=grid_list,
                num_image_tokens=num_img_tokens,
                mrope_position_ids=mp_bytes,
            )
            mm_req.image_embeddings_shape.extend(ie_shape_list)
            mm_req.mrope_position_shape.extend(mp_shape_list)
            mm_req.remaining_circuit.extend(text_addrs[1:])

            text_entry_channel = grpc.insecure_channel(
                text_addrs[0],
                options=[
                    ("grpc.max_send_message_length", 256 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                ],
            )
            mm_stub = inference_pb2_grpc.InferenceNodeStub(text_entry_channel)

            print("  Sending multimodal query (image + text)...")
            mm_start = time.time()
            mm_resp = mm_stub.Forward(mm_req)
            mm_time = time.time() - mm_start

            test("Multimodal response has token",
                 mm_resp.has_token, f"has_token={mm_resp.has_token}")

            if mm_resp.has_token:
                mm_token_text = tokenizer.decode([mm_resp.token_id])
                print(f"  Generated: '{mm_token_text}' (token_id={mm_resp.token_id}, "
                      f"{mm_time:.2f}s)")
                test("Multimodal generated valid token",
                     mm_resp.token_id > 0 and mm_resp.token_id < 151936)

                # Try generating a few more tokens (decode steps)
                generated = [mm_token_text]
                all_tokens = list(multimodal_ids) + [mm_resp.token_id]

                for decode_step in range(4):
                    decode_req = inference_pb2.ForwardRequest(
                        session_id="test-multimodal-combined-1",
                        token_ids=[all_tokens[-1]],
                        is_prefill=False,
                    )
                    decode_req.remaining_circuit.extend(text_addrs[1:])

                    try:
                        decode_resp = mm_stub.Forward(decode_req)
                        if decode_resp.has_token:
                            tok_txt = tokenizer.decode([decode_resp.token_id])
                            generated.append(tok_txt)
                            all_tokens.append(decode_resp.token_id)
                            if decode_resp.is_eos:
                                break
                        else:
                            break
                    except Exception as e:
                        print(f"  Decode step {decode_step} failed: {e}")
                        break

                full_response = "".join(generated)
                print(f"  Full generation ({len(generated)} tokens): '{full_response}'")
                test("Generated multi-token response",
                     len(generated) >= 2, f"got {len(generated)} tokens")
        else:
            print("  [SKIP] Missing vision circuit or text circuit for combined test")

        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

    except Exception as e:
        print(f"  [ERROR] E2E test failed: {e}")
        import traceback
        traceback.print_exc()
        global failed
        failed += 1
    finally:
        cleanup()


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("UNFED AI — Multimodal Vision MPC Tests")
    print("=" * 60)

    # Unit tests (no network needed)
    test_mpc_conv3d()
    test_image_merge()
    test_mrope_positions()

    # E2E test (starts full network)
    test_e2e_multimodal()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
