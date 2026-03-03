import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from network.infra_routing import select_least_loaded_daemon


def _daemon(address: str):
    return SimpleNamespace(address=address, node_type="daemon")


def test_select_least_loaded_daemon_prefers_lowest_utilization():
    daemons = [
        _daemon("d1:50070"),
        _daemon("d2:50070"),
        _daemon("d3:50070"),
    ]
    utilizations = {"d1:50070": 0.7, "d2:50070": 0.2, "d3:50070": 0.4}

    selected, by_addr = select_least_loaded_daemon(
        daemons,
        lambda d: utilizations[d.address],
    )

    assert selected is not None
    assert selected.address == "d2:50070"
    assert by_addr == utilizations


def test_select_least_loaded_daemon_tie_breaks_by_address():
    daemons = [_daemon("z:1"), _daemon("a:1")]
    selected, by_addr = select_least_loaded_daemon(daemons, lambda _: 0.3)

    assert selected is not None
    assert selected.address == "a:1"
    assert by_addr == {"z:1": 0.3, "a:1": 0.3}


def test_select_least_loaded_daemon_fallbacks_on_probe_error():
    daemons = [_daemon("d1:1"), _daemon("d2:1")]

    def flaky_probe(d):
        if d.address == "d1:1":
            raise RuntimeError("unreachable")
        return 0.8

    selected, by_addr = select_least_loaded_daemon(daemons, flaky_probe)

    # Probe failures are treated as util=1.0, so d2 wins with 0.8.
    assert selected is not None
    assert selected.address == "d2:1"
    assert by_addr["d1:1"] == 1.0
    assert by_addr["d2:1"] == 0.8
