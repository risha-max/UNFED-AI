from types import SimpleNamespace

from network.admission import preflight_model_admission


def _node(shard_index: int, node_type: str) -> SimpleNamespace:
    return SimpleNamespace(shard_index=shard_index, node_type=node_type)


class _FakeDiscovery:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def discover(self, model_id: str):
        if not model_id:
            return list(self._nodes) + [_node(-1, "daemon")]
        return list(self._nodes)

    def get_verifier_health(self):
        return SimpleNamespace(
            healthy_verifier_count=1,
            required_verifier_count=1,
            healthy=True,
        )


def test_preflight_rejects_missing_mpc_by_default(monkeypatch):
    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    discovery = _FakeDiscovery([
        _node(0, "compute"),
        _node(1, "compute"),
    ])
    result = preflight_model_admission(discovery, "model-x")
    assert result.ok is False
    assert result.reason == "missing_mpc"


def test_preflight_accepts_full_coverage_when_mpc_override_disabled(monkeypatch):
    monkeypatch.setenv("UNFED_REQUIRE_MPC", "0")
    discovery = _FakeDiscovery([
        _node(0, "compute"),
        _node(1, "compute"),
    ])
    result = preflight_model_admission(discovery, "model-x")
    assert result.ok is True
    assert result.text.covered_shards == 2
    assert result.text.total_shards == 2


def test_preflight_rejects_multimodal_when_vision_incomplete(monkeypatch):
    monkeypatch.setenv("UNFED_REQUIRE_MPC", "0")
    discovery = _FakeDiscovery([
        _node(0, "compute"),
        _node(1, "compute"),
    ])
    result = preflight_model_admission(
        discovery,
        "model-x",
        require_vision=True,
    )
    assert result.ok is False
    assert result.reason == "incomplete_vision_coverage"


def test_preflight_rejects_missing_daemon_by_default(monkeypatch):
    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    monkeypatch.delenv("UNFED_REQUIRE_DAEMON", raising=False)
    discovery = _FakeDiscovery([_node(0, "mpc"), _node(1, "compute")])
    # Force discovery() with empty model_id to return no daemon.
    discovery.discover = lambda model_id: [_node(0, "mpc"), _node(1, "compute")]
    result = preflight_model_admission(discovery, "model-x")
    assert result.ok is False
    assert result.reason == "missing_daemon"


def test_preflight_accepts_without_daemon_when_override_disabled(monkeypatch):
    monkeypatch.setenv("UNFED_REQUIRE_DAEMON", "0")
    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    discovery = _FakeDiscovery([_node(0, "mpc"), _node(1, "compute")])
    discovery.discover = lambda model_id: [_node(0, "mpc"), _node(1, "compute")]
    result = preflight_model_admission(discovery, "model-x")
    assert result.ok is True
