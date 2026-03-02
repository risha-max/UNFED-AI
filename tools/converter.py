"""
UNFED Weight Converter — convert between weight formats.

Supported conversions:
  - PyTorch .pt/.bin → safetensors (secure, no code execution risk)
  - safetensors → PyTorch .pt (legacy compatibility)

Usage:
    python -m tools convert model.bin -o model.safetensors
    python -m tools convert model.pt -o model.safetensors --to safetensors
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
from tools.gguf_utils import GGUFError, gguf_to_state_dict


def convert_file(
    input_path: str,
    output_path: str,
    target_format: str = "safetensors",
) -> dict:
    """Convert a weight file between formats.

    Args:
        input_path: Source weight file.
        output_path: Destination file path.
        target_format: "safetensors" or "pt".

    Returns:
        Dict with conversion stats.
    """
    start_time = time.time()
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    input_ext = input_file.suffix.lower()
    if not input_file.exists() or not input_file.is_file():
        raise ValueError(f"Input file not found: {input_path}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    input_size = os.path.getsize(input_file)

    print("=" * 60)
    print("  UNFED Weight Converter")
    print("=" * 60)
    print()
    print(f"  Input:  {input_file} ({input_size / (1024*1024):.1f} MB)")
    print(f"  Output: {output_file}")
    print(f"  Format: {input_ext} → .{target_format}")
    print()

    # Load source
    print("  Loading weights...")
    gguf_meta = None
    if input_ext == ".safetensors":
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(input_file), device="cpu")
        except ImportError:
            print("  ERROR: safetensors package not installed.")
            sys.exit(1)
    elif input_ext in (".pt", ".pth", ".bin"):
        state_dict = torch.load(str(input_file), map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            print("  ERROR: Expected a state dict, got:", type(state_dict))
            sys.exit(1)
    elif input_ext == ".gguf":
        try:
            state_dict, gguf_meta = gguf_to_state_dict(str(input_file))
            print(f"  GGUF architecture: {gguf_meta.get('architecture', 'unknown')}")
            print(f"  GGUF tensors:      {gguf_meta.get('tensor_count', 0)}")
            print(f"  GGUF tensor bytes: {gguf_meta.get('total_tensor_bytes', 0):,}")
        except GGUFError as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)
    else:
        print(f"  ERROR: Unsupported input format: {input_ext}")
        sys.exit(1)

    num_tensors = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"  Loaded {num_tensors} tensors ({total_params:,} parameters)")

    # Save in target format
    print(f"  Saving as {target_format}...")
    if target_format == "safetensors":
        try:
            from safetensors.torch import save_file
            # Ensure all tensors are contiguous (required by safetensors)
            clean = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(clean, str(output_file))
        except ImportError:
            print("  ERROR: safetensors package not installed. "
                  "Install with: pip install safetensors")
            sys.exit(1)
    elif target_format == "pt":
        torch.save(state_dict, str(output_file))
    else:
        print(f"  ERROR: Unsupported target format: {target_format}")
        sys.exit(1)

    output_size = os.path.getsize(output_file)
    elapsed = time.time() - start_time

    ratio = output_size / input_size if input_size > 0 else 0

    print()
    print("=" * 60)
    print(f"  Conversion complete in {elapsed:.1f}s")
    print(f"  Input:  {input_size / (1024*1024):.1f} MB")
    print(f"  Output: {output_size / (1024*1024):.1f} MB ({ratio:.2f}x)")
    print(f"  Tensors: {num_tensors}, Parameters: {total_params:,}")
    print("=" * 60)

    return {
        "input_path": input_path,
        "output_path": str(output_file),
        "num_tensors": num_tensors,
        "total_params": total_params,
        "input_size": input_size,
        "output_size": output_size,
        "elapsed": elapsed,
        "source_format": input_ext.lstrip("."),
        "gguf_metadata": gguf_meta,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_convert(args):
    """CLI handler for 'unfed convert'."""
    convert_file(
        input_path=args.input_path,
        output_path=args.output,
        target_format=args.to,
    )
