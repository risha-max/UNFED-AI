"""
UNFED AI Verification — confirms token-for-token correctness of the distributed pipeline
by comparing output against the full (non-sharded) model.

Usage:
    python verify.py                      # default: compare sharded vs full model
    python verify.py --prompt "Hello"     # custom prompt
    python verify.py --max-tokens 20      # generate more tokens
"""

import argparse
import os
import sys
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from node.runtime.generic_runner import GenericTextRunner


def load_full_model(model_name: str):
    """Load the full (non-sharded) HuggingFace model for reference."""
    print(f"[Verify] Loading full model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


def generate_reference(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Generate tokens using the full HuggingFace model."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
        )
    return output[0].tolist()[input_ids.shape[1]:]


def generate_sharded(manifest_path: str, shards_dir: str,
                     prompt: str, max_tokens: int, model_name: str) -> list[int]:
    """Generate tokens using the sharded GenericTextRunner pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    arch_config = manifest.get("architecture", {}).get("text")
    if not arch_config:
        raise ValueError("Manifest missing 'architecture.text'")

    shards_info = manifest.get("shards", [])
    if not shards_info:
        shards_info = manifest.get("text_shards", [])

    runners = []
    for i, shard_info in enumerate(shards_info):
        shard_file = shard_info.get("file", f"text_shard_{i}.pt")
        shard_path = os.path.join(shards_dir, shard_file)
        runner = GenericTextRunner(
            config=arch_config,
            shard_info=shard_info,
            shard_path=shard_path,
        )
        runners.append(runner)

    print(f"[Verify] Loaded {len(runners)} shards")

    session_id = "verify"
    generated = []
    tokens = list(input_ids)

    for step in range(max_tokens):
        hidden = None
        sampled_token = None

        for i, runner in enumerate(runners):
            if i == 0:
                if step == 0:
                    t = torch.tensor([tokens])
                else:
                    t = torch.tensor([[tokens[-1]]])
                hidden, sampled_token = runner.forward(
                    token_ids=t, session_id=session_id)
            else:
                hidden, sampled_token = runner.forward(
                    hidden_states=hidden, session_id=session_id)

        if sampled_token is not None:
            generated.append(sampled_token)
            tokens.append(sampled_token)
            eos = manifest.get("eos_token_id", 151643)
            if sampled_token == eos:
                break

    return generated


def main():
    parser = argparse.ArgumentParser(description="UNFED AI Verification")
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is",
                        help="Prompt for verification")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Tokens to generate (default: 10)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name (default: {config.MODEL_NAME})")
    parser.add_argument("--shards-dir", type=str, default=None,
                        help="Shards directory")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Manifest path")
    args = parser.parse_args()

    model_name = args.model or config.MODEL_NAME
    shards_dir = args.shards_dir or config.SHARDS_DIR
    manifest_path = args.manifest or config.MANIFEST_PATH

    if not os.path.exists(manifest_path):
        print(f"[Verify] Manifest not found at {manifest_path}")
        print("[Verify] Run the model splitter first, or provide --manifest and --shards-dir")
        sys.exit(1)

    print(f"[Verify] Prompt: {args.prompt!r}")
    print(f"[Verify] Max tokens: {args.max_tokens}")
    print()

    tokenizer, full_model = load_full_model(model_name)

    print("[Verify] Generating reference (full model)...")
    ref_tokens = generate_reference(full_model, tokenizer, args.prompt, args.max_tokens)
    ref_text = tokenizer.decode(ref_tokens)
    print(f"  Reference: {ref_text!r}")
    print(f"  Tokens: {ref_tokens}")

    del full_model

    print("\n[Verify] Generating sharded output...")
    sharded_tokens = generate_sharded(
        manifest_path, shards_dir, args.prompt, args.max_tokens, model_name)
    sharded_text = tokenizer.decode(sharded_tokens)
    print(f"  Sharded:   {sharded_text!r}")
    print(f"  Tokens: {sharded_tokens}")

    print()
    if ref_tokens == sharded_tokens:
        print("[PASS] Token-for-token match!")
        sys.exit(0)
    else:
        n_match = sum(1 for a, b in zip(ref_tokens, sharded_tokens) if a == b)
        print(f"[FAIL] Mismatch — {n_match}/{max(len(ref_tokens), len(sharded_tokens))} "
              f"tokens match")
        for i, (a, b) in enumerate(zip(ref_tokens, sharded_tokens)):
            if a != b:
                print(f"  First diff at position {i}: ref={a} vs sharded={b}")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
