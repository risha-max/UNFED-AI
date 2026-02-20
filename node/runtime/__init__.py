"""
UNFED Generic Model Runtime — platform-agnostic transformer building blocks.

Assembles standard building blocks (attention, MLP, norm, RoPE) from a
declarative config + weight tensors.  No model-specific imports needed at
runtime — the architecture is defined entirely by JSON config numbers and
the weights are just named tensors.

Building blocks:
  text_blocks   — RMSNorm, RoPE, Attention, MLP, DecoderLayer, Embedding, LMHead
  vision_blocks — PatchEmbedding, VisionAttention, VisionEncoderLayer, Connector
  weight_loader — Load weights from safetensors/pt with key remapping
  generic_runner — GenericTextRunner, GenericVisionRunner
"""
