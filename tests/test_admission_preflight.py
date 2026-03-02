from types import SimpleNamespace

from network.admission import preflight_model_admission


def _node(shard_index: int, node_type: str) -> SimpleNamespace:
    return SimpleNamespace(shard_index=shard_index, node_type=node_type)


class _FakeDiscovery:
    def __init__(self, nodes):
        self._nodes = list(nodes)

    def discover(self, model_id: str):
        return list(self._nodes)


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
