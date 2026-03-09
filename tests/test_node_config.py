import pytest

from node.node_config import DaemonConfig, ComputeConfig, load_config


def test_compute_defaults_autosize_workers():
    cfg = load_config(
        config_path=None,
        cli_overrides={
            "role": "compute",
            "shard_index": 0,
        },
    )
    assert isinstance(cfg, ComputeConfig)
    assert cfg.grpc_max_workers >= 4


def test_base_validation_rejects_invalid_backoff_bounds():
    with pytest.raises(ValueError, match="retry_backoff_max must be >= retry_backoff_base"):
        load_config(
            config_path=None,
            cli_overrides={
                "role": "compute",
                "shard_index": 0,
                "retry_backoff_base": 2.0,
                "retry_backoff_max": 1.0,
            },
        )


def test_daemon_validation_rejects_invalid_fee_range():
    with pytest.raises(ValueError, match="fee_max must be >= fee_min"):
        load_config(
            config_path=None,
            cli_overrides={
                "role": "daemon",
                "fee_min": 0.01,
                "fee_max": 0.001,
            },
        )


def test_daemon_config_loads_with_valid_fee_settings():
    cfg = load_config(
        config_path=None,
        cli_overrides={
            "role": "daemon",
            "fee_base": 0.002,
            "fee_min": 0.001,
            "fee_max": 0.01,
        },
    )
    assert isinstance(cfg, DaemonConfig)
    assert cfg.fee_min <= cfg.fee_base <= cfg.fee_max
