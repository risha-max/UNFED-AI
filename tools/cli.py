"""
UNFED CLI — unified entry point for the pool creator toolchain.

Usage:
    python -m tools.cli inspect /path/to/model
    python -m tools.cli split /path/to/model -o ./shards --text-shards 2
    python -m tools.cli split /path/to/model -o ./shards --shards text_decoder=30 vision_encoder=12
    python -m tools.cli verify ./shards/manifest.json
    python -m tools.cli convert model.pt -o model.safetensors
    python -m tools.cli publish ./shards --registry http://localhost:8765
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="unfed",
        description="UNFED AI — Pool Creator Toolchain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
commands:
  inspect   Analyse a model: detect format, architecture, layer count
  split     Split a model into shards for distributed inference
  verify    Validate shard integrity and correctness
  convert   Convert weight formats (pt → safetensors, etc.)
  publish   Upload shards + manifest to an UNFED registry
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Tool to run")

    # ---------------------------------------------------------------
    # inspect
    # ---------------------------------------------------------------
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Analyse a model: detect format, infer architecture, report compatibility",
    )
    p_inspect.add_argument(
        "model_path",
        help="Path to model directory or weight file",
    )
    p_inspect.add_argument(
        "--format", choices=["auto", "hf", "safetensors", "pt", "gguf"],
        default="auto",
        help="Force a specific format (default: auto-detect)",
    )
    p_inspect.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed weight tensor information",
    )

    # ---------------------------------------------------------------
    # split
    # ---------------------------------------------------------------
    p_split = subparsers.add_parser(
        "split",
        help="Split a model into shards for distributed inference",
    )
    p_split.add_argument(
        "model_path",
        help="Path to model directory or weight file",
    )
    p_split.add_argument(
        "-o", "--output", default="./shards",
        help="Output directory for shards (default: ./shards)",
    )
    p_split.add_argument(
        "--shards", nargs="*", default=None,
        help="Per-stack shard counts, e.g. text_decoder=30 vision_encoder=12. "
             "Takes precedence over --text-shards/--vision-shards.",
    )
    p_split.add_argument(
        "--text-shards", type=int, default=2,
        help="Number of text decoder shards (default: 2). "
             "Legacy alias — prefer --shards for multi-stack control.",
    )
    p_split.add_argument(
        "--vision-shards", type=int, default=1,
        help="Number of vision encoder shards (default: 1). "
             "Legacy alias — prefer --shards for multi-stack control.",
    )
    p_split.add_argument(
        "--format", choices=["safetensors", "pt"], default="safetensors",
        help="Output weight format (default: safetensors)",
    )
    p_split.add_argument(
        "--verify", action="store_true", default=True,
        help="Generate verification vectors (default: True)",
    )

    # ---------------------------------------------------------------
    # verify
    # ---------------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Validate shard integrity and correctness",
    )
    p_verify.add_argument(
        "manifest_path",
        help="Path to manifest.json",
    )
    p_verify.add_argument(
        "--deep", action="store_true",
        help="Run deep verification (compare against source model if available)",
    )
    p_verify.add_argument(
        "--source", default=None,
        help="Path to original model for deep comparison",
    )

    # ---------------------------------------------------------------
    # convert
    # ---------------------------------------------------------------
    p_convert = subparsers.add_parser(
        "convert",
        help="Convert weight formats (pt → safetensors, GGUF → safetensors)",
    )
    p_convert.add_argument(
        "input_path",
        help="Path to input weight file",
    )
    p_convert.add_argument(
        "-o", "--output", required=True,
        help="Output file path",
    )
    p_convert.add_argument(
        "--to", choices=["safetensors", "pt"], default="safetensors",
        help="Target format (default: safetensors)",
    )

    # ---------------------------------------------------------------
    # publish
    # ---------------------------------------------------------------
    p_publish = subparsers.add_parser(
        "publish",
        help="Upload shards + manifest to an UNFED registry",
    )
    p_publish.add_argument(
        "shards_dir",
        help="Directory containing shards and manifest.json",
    )
    p_publish.add_argument(
        "--registry", default="localhost:50050",
        help="Registry gRPC address (default: localhost:50050)",
    )
    p_publish.add_argument(
        "--pool-name", default=None,
        help="Pool name (default: derived from model ID)",
    )

    # ---------------------------------------------------------------
    # Parse and dispatch
    # ---------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "inspect":
        from tools.inspector import run_inspect
        run_inspect(args)
    elif args.command == "split":
        from tools.splitter import run_split
        run_split(args)
    elif args.command == "verify":
        from tools.verifier import run_verify
        run_verify(args)
    elif args.command == "convert":
        from tools.converter import run_convert
        run_convert(args)
    elif args.command == "publish":
        from tools.publisher import run_publish
        run_publish(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
