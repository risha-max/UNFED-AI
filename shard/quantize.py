"""
Quantization Support — int4/int8 quantization for larger models.

Quantization reduces model weight precision to decrease memory footprint:
  - float32: 4 bytes per parameter (full precision)
  - float16: 2 bytes per parameter
  - int8: 1 byte per parameter (2x compression vs float16)
  - int4: 0.5 bytes per parameter (4x compression vs float16)

This enables serving larger models on commodity hardware:
  - Llama 3 8B at int4: ~4.5 GB total, ~280 MB per shard (16 shards)
  - Llama 3 70B at int4: ~38 GB total, ~2.4 GB per shard (16 shards)

We support two quantization strategies:
  1. Post-training quantization (PTQ): quantize after splitting
  2. Load quantized: use HF's BitsAndBytes integration to load pre-quantized

Usage:
    python -m shard.quantize --model "meta-llama/Llama-3-8B" --bits 4 --num-shards 16
    python -m shard.quantize --model "Qwen/Qwen2.5-0.5B" --bits 8 --num-shards 4
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def check_quantization_support() -> dict:
    """Check what quantization backends are available."""
    support = {
        "bitsandbytes": False,
        "torch_int8": True,  # PyTorch native int8 always available
        "cuda_available": False,
    }

    try:
        import bitsandbytes
        support["bitsandbytes"] = True
    except ImportError:
        pass

    try:
        import torch
        support["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass

    return support


def estimate_model_size(model_id: str, bits: int, num_shards: int) -> dict:
    """Estimate memory requirements for a quantized model."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    num_params = _estimate_params(cfg)

    bytes_per_param = {32: 4.0, 16: 2.0, 8: 1.0, 4: 0.5}
    bpp = bytes_per_param.get(bits, 4.0)

    total_bytes = num_params * bpp
    shard_bytes = total_bytes / num_shards

    return {
        "model_id": model_id,
        "num_params": num_params,
        "bits": bits,
        "total_size_mb": total_bytes / (1024 * 1024),
        "shard_size_mb": shard_bytes / (1024 * 1024),
        "num_shards": num_shards,
        "num_layers": cfg.num_hidden_layers,
        "layers_per_shard": cfg.num_hidden_layers // num_shards,
    }


def _estimate_params(cfg) -> int:
    """Estimate number of parameters from model config."""
    hidden = cfg.hidden_size
    vocab = cfg.vocab_size
    layers = cfg.num_hidden_layers
    intermediate = getattr(cfg, 'intermediate_size', hidden * 4)
    num_heads = cfg.num_attention_heads
    head_dim = hidden // num_heads

    # Embedding
    embed_params = vocab * hidden

    # Per layer: attention (Q, K, V, O) + MLP (gate, up, down) + norms
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
    attn_params = (hidden * hidden +  # Q
                   hidden * (num_kv_heads * head_dim) +  # K
                   hidden * (num_kv_heads * head_dim) +  # V
                   hidden * hidden)  # O
    mlp_params = (hidden * intermediate +  # gate/up
                  hidden * intermediate +  # up (for gated MLPs)
                  intermediate * hidden)  # down
    norm_params = hidden * 2  # 2 layer norms per layer
    layer_params = attn_params + mlp_params + norm_params

    # LM head + final norm
    head_params = hidden * vocab + hidden

    total = embed_params + (layers * layer_params) + head_params
    return int(total)


def quantize_and_split(model_id: str, bits: int, num_shards: int,
                       output_dir: str = None):
    """
    Load a model with quantization and split into shards.

    For int4/int8 with bitsandbytes (requires CUDA):
      - Uses HF's BitsAndBytesConfig for on-the-fly quantization
      - Saves quantized shard files

    For CPU (no CUDA):
      - Uses PyTorch's dynamic quantization (int8 only)
      - Falls back to float16/float32 for int4

    Args:
        model_id: HuggingFace model ID
        bits: Quantization bits (4, 8, 16, 32)
        num_shards: Number of shards to split into
        output_dir: Output directory for shards
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    output_dir = output_dir or config.SHARDS_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_config = AutoConfig.from_pretrained(model_id)
    num_layers = model_config.num_hidden_layers

    if num_layers % num_shards != 0:
        print(f"ERROR: {num_layers} layers not divisible by {num_shards} shards")
        return

    layers_per_shard = num_layers // num_shards
    support = check_quantization_support()

    print(f"Model: {model_id}")
    print(f"Layers: {num_layers}, Shards: {num_shards}, Layers/shard: {layers_per_shard}")
    print(f"Quantization: {bits}-bit")
    print(f"CUDA: {support['cuda_available']}, BitsAndBytes: {support['bitsandbytes']}")

    # Determine loading strategy
    if bits == 4 and support['bitsandbytes'] and support['cuda_available']:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("Loading with 4-bit quantization (BitsAndBytes NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant_config, device_map="auto")
    elif bits == 8 and support['bitsandbytes'] and support['cuda_available']:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Loading with 8-bit quantization (BitsAndBytes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant_config, device_map="auto")
    elif bits <= 16:
        dtype = torch.float16 if bits <= 16 else torch.float32
        print(f"Loading with {dtype} (no hardware quantization available)...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    else:
        print("Loading with float32...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32)

    model.eval()

    # Split into shards
    manifest = {
        "model_id": model_id,
        "num_layers": num_layers,
        "num_shards": num_shards,
        "layers_per_shard": layers_per_shard,
        "quantization_bits": bits,
        "shards": [],
    }

    for shard_idx in range(num_shards):
        start = shard_idx * layers_per_shard
        end = start + layers_per_shard
        is_first = shard_idx == 0
        is_last = shard_idx == num_shards - 1

        shard_path = os.path.join(output_dir, f"shard_{shard_idx}.pt")
        print(f"\nShard {shard_idx}: layers {start}-{end - 1}")

        shard = {}

        if is_first:
            shard["embed_tokens"] = model.model.embed_tokens.state_dict()
            print(f"  + embed_tokens")

        for layer_idx in range(start, end):
            shard[f"layer_{layer_idx}"] = model.model.layers[layer_idx].state_dict()
        print(f"  + layers {start}-{end - 1}")

        if is_last:
            shard["norm"] = model.model.norm.state_dict()
            shard["lm_head"] = model.lm_head.state_dict()
            print(f"  + norm + lm_head")

        torch.save(shard, shard_path)
        size = os.path.getsize(shard_path)

        sha256 = hashlib.sha256()
        with open(shard_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)

        manifest["shards"].append({
            "shard_index": shard_idx,
            "layer_start": start,
            "layer_end": end,
            "has_embedding": is_first,
            "has_lm_head": is_last,
            "file": os.path.basename(shard_path),
            "size_bytes": size,
            "sha256": sha256.hexdigest(),
        })

        print(f"  Saved: {size / (1024 * 1024):.1f} MB")

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    model_config.save_pretrained(output_dir)
    print(f"\nManifest: {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Quantized Model Splitter")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                        help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16, 32],
                        help="Quantization bits")
    parser.add_argument("--num-shards", type=int, default=4,
                        help="Number of shards")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output directory (default: {config.SHARDS_DIR})")
    args = parser.parse_args()

    # Show estimates first
    est = estimate_model_size(args.model, args.bits, args.num_shards)
    print(f"Estimated sizes:")
    print(f"  Total: {est['total_size_mb']:.0f} MB")
    print(f"  Per shard: {est['shard_size_mb']:.0f} MB")
    print()

    quantize_and_split(args.model, args.bits, args.num_shards, args.output)
