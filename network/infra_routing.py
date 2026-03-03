"""
Infra routing helpers for daemon selection and telemetry.
"""

from __future__ import annotations

from typing import Any, Callable


def select_least_loaded_daemon(
    daemons: list[Any],
    utilization_probe: Callable[[Any], float],
) -> tuple[Any | None, dict[str, float]]:
    """
    Pick the daemon with the lowest utilization.

    Returns:
      - selected daemon object (or None)
      - address -> utilization map for observability
    """
    best = None
    best_key = None
    utilization_by_address: dict[str, float] = {}
    for daemon in daemons or []:
        address = str(getattr(daemon, "address", "") or "")
        util = 1.0
        try:
            util = float(utilization_probe(daemon))
        except Exception:
            util = 1.0
        if util < 0.0:
            util = 0.0
        utilization_by_address[address] = util
        key = (util, address)
        if best is None or key < best_key:
            best = daemon
            best_key = key
    return best, utilization_by_address
