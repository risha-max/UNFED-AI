from types import SimpleNamespace

from network.admission import preflight_model_admission


def _node(shard_index: int, node_type: str) -> SimpleNamespace:
    return SimpleNamespace(shard_index=shard_index, node_type=node_type)


class _DiscoveryWithVerifier:
    def __init__(self, healthy: bool):
        self._healthy = healthy

    def discover(self, model_id: str):
        nodes = [_node(0, "mpc"), _node(1, "compute")]
        if not model_id:
            nodes.append(_node(-1, "daemon"))
        return nodes

    def get_verifier_health(self):
        return SimpleNamespace(
            healthy_verifier_count=1 if self._healthy else 0,
            required_verifier_count=1,
            healthy=self._healthy,
        )


def test_preflight_rejects_when_verifier_missing(monkeypatch):
    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    monkeypatch.delenv("UNFED_REQUIRE_VERIFIER", raising=False)
    discovery = _DiscoveryWithVerifier(healthy=False)
    result = preflight_model_admission(discovery, "model-z")
    assert result.ok is False
    assert result.reason == "missing_verifier"


def test_preflight_accepts_when_verifier_healthy(monkeypatch):
    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    monkeypatch.delenv("UNFED_REQUIRE_VERIFIER", raising=False)
    discovery = _DiscoveryWithVerifier(healthy=True)
    result = preflight_model_admission(discovery, "model-z")
    assert result.ok is True
