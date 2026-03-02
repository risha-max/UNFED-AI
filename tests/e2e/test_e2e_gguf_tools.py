import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON = os.environ.get("UNFED_E2E_PYTHON", sys.executable)


def _write_cli_test_gguf(path: Path):
    writer = GGUFWriter(str(path), arch="qwen2")
    writer.add_name("e2e-gguf")
    writer.add_block_count(2)
    writer.add_embedding_length(8)
    writer.add_feed_forward_length(16)
    writer.add_head_count(2)
    writer.add_head_count_kv(2)
    writer.add_key_length(4)
    writer.add_value_length(4)
    writer.add_context_length(64)
    writer.add_file_type(0)
    writer.add_tensor("model.embed_tokens.weight", np.random.randn(32, 8).astype(np.float32))
    writer.add_tensor("model.layers.0.self_attn.q_proj.weight", np.random.randn(8, 8).astype(np.float32))
    writer.add_tensor("model.layers.1.self_attn.q_proj.weight", np.random.randn(8, 8).astype(np.float32))
    writer.add_tensor("model.norm.weight", np.random.randn(8).astype(np.float32))
    writer.add_tensor("lm_head.weight", np.random.randn(32, 8).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.e2e
def test_cli_convert_and_split_gguf(tmp_path: Path):
    gguf_path = tmp_path / "e2e.gguf"
    safetensors_path = tmp_path / "e2e.safetensors"
    shards_dir = tmp_path / "shards"
    _write_cli_test_gguf(gguf_path)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    convert = subprocess.run(
        [PYTHON, "-m", "tools.cli", "convert", str(gguf_path), "-o", str(safetensors_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert convert.returncode == 0, convert.stdout + "\n" + convert.stderr
    assert safetensors_path.exists()

    split = subprocess.run(
        [
            PYTHON,
            "-m",
            "tools.cli",
            "split",
            str(gguf_path),
            "-o",
            str(shards_dir),
            "--text-shards",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert split.returncode == 0, split.stdout + "\n" + split.stderr
    assert (shards_dir / "manifest.json").exists()


@pytest.mark.e2e
def test_cli_rejects_truncated_gguf(tmp_path: Path):
    bad_gguf = tmp_path / "bad.gguf"
    bad_gguf.write_bytes(b"GGUF")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(
        [PYTHON, "-m", "tools.cli", "convert", str(bad_gguf), "-o", str(tmp_path / "bad.safetensors")],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


@pytest.mark.e2e
def test_cli_inspect_real_downloaded_gguf_if_available():
    model_path = os.environ.get("UNFED_GGUF_TEST_MODEL", "").strip()
    if not model_path:
        pytest.skip("UNFED_GGUF_TEST_MODEL not set")
    if not os.path.isfile(model_path):
        pytest.skip(f"UNFED_GGUF_TEST_MODEL not found: {model_path}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(
        [PYTHON, "-m", "tools.cli", "inspect", model_path],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "Format:     gguf" in proc.stdout
