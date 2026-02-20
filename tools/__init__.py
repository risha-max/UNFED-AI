"""
UNFED Tools — Pool creator toolchain for model preparation and publishing.

Tools:
  inspect   — Analyse a model: detect format, architecture, layer structure
  split     — Split a model into shards with declarative manifest
  verify    — Validate shards: hash checks, verification vectors, source compare
  convert   — Convert between weight formats (pickle→safetensors, GGUF→safetensors)
  publish   — Upload shards + manifest to an UNFED registry

Usage:
  python -m tools.cli inspect /path/to/model
  python -m tools.cli split /path/to/model -o ./shards --text-shards 2
  python -m tools.cli split /path/to/model --shards text_decoder=30 vision_encoder=12
  python -m tools.cli verify ./shards/manifest.json
  python -m tools.cli convert model.pt -o model.safetensors
  python -m tools.cli publish ./shards --registry http://localhost:8765
"""
