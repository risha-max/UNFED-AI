import os
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import web.server as web_server


class _FakeDiscovery:
    def get_pool_health(self, model_id):
        return SimpleNamespace(
            model_id=model_id or "m",
            total_shards=2,
            overall_status="healthy",
            can_serve=True,
            shards=[
                SimpleNamespace(shard_index=0, node_count=1, status="healthy"),
                SimpleNamespace(shard_index=1, node_count=1, status="healthy"),
            ],
        )

    def discover(self, model_id):
        return [
            SimpleNamespace(node_type="daemon"),
            SimpleNamespace(node_type="compute"),
        ]

    def get_verifier_health(self):
        return SimpleNamespace(healthy_verifier_count=2, required_verifier_count=2)

    def get_infra_telemetry(self):
        return SimpleNamespace(
            healthy_daemon_count=1,
            required_daemon_count=1,
            selected_daemon_recipient="0xdaemon",
            selected_verifier_recipient="0xverifier",
            daemon_work_window_json='{"0xdaemon": 4.0}',
            verifier_work_window_json='{"ver-1": 3.0}',
            daemon_payout_share_json='{"0xdaemon": 1.0}',
            verifier_payout_share_json='{"0xverifier": 1.0}',
        )


@pytest.mark.anyio
async def test_network_health_exposes_infra_telemetry(monkeypatch):
    monkeypatch.setattr(web_server, "get_discovery", lambda: _FakeDiscovery())
    payload = await web_server.get_health("demo-model")
    assert payload["healthy_daemon_count"] == 1
    assert payload["required_daemon_count"] == 1
    assert payload["selected_daemon_recipient"] == "0xdaemon"
    assert payload["selected_verifier_recipient"] == "0xverifier"
    assert payload["daemon_work_window"] == {"0xdaemon": 4.0}
    assert payload["verifier_payout_share"] == {"0xverifier": 1.0}


@pytest.mark.anyio
async def test_chain_fees_includes_candidate_utilization_on_no_daemon(monkeypatch):
    monkeypatch.setattr(
        web_server,
        "_discover_daemon_with_utilization",
        lambda: (None, {"d1:50070": 0.2, "d2:50070": 0.8}),
    )
    payload = await web_server.chain_fees()
    assert payload["daemon_available"] is False
    assert payload["selected_daemon"] == ""
    assert payload["daemon_candidates"] == {"d1:50070": 0.2, "d2:50070": 0.8}
