from economics.cluster_config import ClusterConfig


def test_registry_runtime_defaults_are_valid():
    cfg = ClusterConfig(name="local")
    errors = [e for e in cfg.validate() if not e.startswith("WARNING:")]
    assert not errors


def test_registry_runtime_validation_rejects_invalid_workers():
    cfg = ClusterConfig(name="local", registry_grpc_max_workers=0)
    errors = cfg.validate()
    assert any("registry_grpc_max_workers must be >= 1" in e for e in errors)


def test_registry_runtime_validation_rejects_invalid_timeouts():
    cfg = ClusterConfig(
        name="local",
        registry_node_timeout_seconds=0,
        registry_cleanup_interval_seconds=0.0,
        registry_gossip_interval_seconds=0.0,
        registry_peer_exchange_timeout_seconds=0.0,
        registry_daemon_poll_timeout_seconds=0.0,
    )
    errors = cfg.validate()
    assert any("registry_node_timeout_seconds must be > 0" in e for e in errors)
    assert any("registry_cleanup_interval_seconds must be > 0" in e for e in errors)
    assert any("registry_gossip_interval_seconds must be > 0" in e for e in errors)
    assert any("registry_peer_exchange_timeout_seconds must be > 0" in e for e in errors)
    assert any("registry_daemon_poll_timeout_seconds must be > 0" in e for e in errors)
