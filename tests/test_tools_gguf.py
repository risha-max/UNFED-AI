from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter
from safetensors.torch import load_file

from tools.converter import convert_file
from tools.splitter import split_model
from tools.verifier import verify_shards


def _write_test_gguf(path: Path, tensors: dict[str, np.ndarray], arch: str = "qwen2"):
    writer = GGUFWriter(str(path), arch=arch)
    writer.add_name("test-model")
    writer.add_block_count(2)
    writer.add_embedding_length(8)
    writer.add_feed_forward_length(16)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_key_length(4)
    writer.add_value_length(4)
    writer.add_context_length(64)
    writer.add_file_type(0)
    for key, array in tensors.items():
        writer.add_tensor(key, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_convert_gguf_to_safetensors(tmp_path: Path):
    gguf_path = tmp_path / "mini.gguf"
    out_path = tmp_path / "mini.safetensors"
    _write_test_gguf(
        gguf_path,
        {
            "model.embed_tokens.weight": np.random.randn(32, 8).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float32),
        },
    )

    stats = convert_file(str(gguf_path), str(out_path), target_format="safetensors")
    assert out_path.exists()
    assert stats["source_format"] == "gguf"
    sd = load_file(str(out_path), device="cpu")
    assert "model.embed_tokens.weight" in sd
    assert "model.layers.0.self_attn.q_proj.weight" in sd


def test_convert_gguf_rejects_unsupported_dtype(tmp_path: Path):
    gguf_path = tmp_path / "quantized_like.gguf"
    out_path = tmp_path / "out.safetensors"
    _write_test_gguf(
        gguf_path,
        {
            "model.embed_tokens.weight": np.random.randint(-8, 8, size=(8, 8), dtype=np.int8),
        },
    )

    with pytest.raises(SystemExit):
        convert_file(str(gguf_path), str(out_path), target_format="safetensors")


def test_split_gguf_text_only_success(tmp_path: Path):
    gguf_path = tmp_path / "split_ok.gguf"
    out_dir = tmp_path / "shards"
    _write_test_gguf(
        gguf_path,
        {
            "model.embed_tokens.weight": np.random.randn(32, 8).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float32),
            "model.layers.1.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float32),
            "model.norm.weight": np.random.randn(8).astype(np.float32),
            "lm_head.weight": np.random.randn(32, 8).astype(np.float32),
        },
    )

    manifest = split_model(
        model_path=str(gguf_path),
        output_dir=str(out_dir),
        num_text_shards=2,
        output_format="safetensors",
        generate_verify=False,
    )

    assert (out_dir / "manifest.json").exists()
    assert len(manifest["shards"]) == 2
    assert manifest["shards"][0]["has_embedding"] is True
    assert manifest["shards"][-1]["has_head"] is True


def test_split_gguf_rejects_noncanonical_layer_names(tmp_path: Path):
    gguf_path = tmp_path / "unsupported_keys.gguf"
    _write_test_gguf(
        gguf_path,
        {
            "token_embd.weight": np.random.randn(32, 8).astype(np.float32),
            "blk.0.attn_q.weight": np.random.randn(8, 8).astype(np.float32),
        },
    )

    with pytest.raises(ValueError, match="supported transformer layer key names"):
        split_model(
            model_path=str(gguf_path),
            output_dir=str(tmp_path / "out"),
            num_text_shards=2,
            output_format="safetensors",
            generate_verify=False,
        )


def test_verifier_blocks_manifest_path_escape(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "model_id": "x",
        "model_type": "qwen2",
        "shards": [
            {
                "shard_index": 0,
                "layer_start": 0,
                "layer_end": 1,
                "file": "../secret.bin",
                "sha256": "0" * 64,
                "chunk_hashes": [],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_shards(str(manifest_path))
    assert not result.ok
    assert any("unsafe path" in c["detail"] for c in result.checks)
