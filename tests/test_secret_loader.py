import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.secret_loader import load_secret_from_env


def test_load_secret_prefers_direct_env(monkeypatch, tmp_path):
    secret_file = tmp_path / "stake.key"
    secret_file.write_text("from-file", encoding="utf-8")
    monkeypatch.setenv("UNFED_STAKE_EVM_PRIVATE_KEY", "from-env")
    monkeypatch.setenv("UNFED_STAKE_EVM_PRIVATE_KEY_FILE", str(secret_file))
    value = load_secret_from_env(
        "UNFED_STAKE_EVM_PRIVATE_KEY",
        file_env_var="UNFED_STAKE_EVM_PRIVATE_KEY_FILE",
    )
    assert value == "from-env"


def test_load_secret_falls_back_to_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "stake.key"
    secret_file.write_text("  from-file  \n", encoding="utf-8")
    monkeypatch.delenv("UNFED_STAKE_EVM_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("UNFED_STAKE_EVM_PRIVATE_KEY_FILE", str(secret_file))
    value = load_secret_from_env(
        "UNFED_STAKE_EVM_PRIVATE_KEY",
        file_env_var="UNFED_STAKE_EVM_PRIVATE_KEY_FILE",
    )
    assert value == "from-file"
