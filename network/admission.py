"""
Admission policy for inference entrypoints.

Hard-default behavior:
  - Require full text shard coverage.
  - Require MPC shard-0 availability unless UNFED_REQUIRE_MPC=0.
  - For multimodal requests, also require full vision coverage.
"""

from dataclasses import dataclass
import os


def resolve_mpc_required_flag() -> bool:
    """Default true unless UNFED_REQUIRE_MPC is explicitly false-like."""
    raw = os.environ.get("UNFED_REQUIRE_MPC", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def resolve_verifier_required_flag() -> bool:
    """Default true unless UNFED_REQUIRE_VERIFIER is explicitly false-like."""
    raw = os.environ.get("UNFED_REQUIRE_VERIFIER", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def resolve_daemon_required_flag() -> bool:
    """Default true unless UNFED_REQUIRE_DAEMON is explicitly false-like."""
    raw = os.environ.get("UNFED_REQUIRE_DAEMON", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


@dataclass(frozen=True)
class Coverage:
    covered_shards: int
    total_shards: int

    @property
    def complete(self) -> bool:
        return self.total_shards > 0 and self.covered_shards == self.total_shards


@dataclass(frozen=True)
class AdmissionResult:
    ok: bool
    reason: str
    message: str
    model_id: str
    mpc_required: bool
    mpc_available: bool
    text: Coverage
    vision: Coverage
    verifier_required: bool = True
    healthy_verifier_count: int = 0
    required_verifier_count: int = 1
    daemon_required: bool = True
    healthy_daemon_count: int = 0
    required_daemon_count: int = 1


def _coverage_for(nodes: list) -> Coverage:
    shard_indexes = {int(n.shard_index) for n in nodes if int(n.shard_index) >= 0}
    if not shard_indexes:
        return Coverage(covered_shards=0, total_shards=0)
    return Coverage(
        covered_shards=len(shard_indexes),
        total_shards=max(shard_indexes) + 1,
    )


def preflight_model_admission(
    discovery,
    model_id: str,
    *,
    require_vision: bool = False,
    require_mpc: bool | None = None,
    require_verifier: bool | None = None,
    require_daemon: bool | None = None,
) -> AdmissionResult:
    selected_model = (model_id or "").strip()
    if not selected_model:
        return AdmissionResult(
            ok=False,
            reason="invalid_model",
            message="Model selection is required.",
            model_id=selected_model,
            mpc_required=resolve_mpc_required_flag() if require_mpc is None else bool(require_mpc),
            mpc_available=False,
            text=Coverage(0, 0),
            vision=Coverage(0, 0),
            verifier_required=resolve_verifier_required_flag()
            if require_verifier is None else bool(require_verifier),
            daemon_required=resolve_daemon_required_flag()
            if require_daemon is None else bool(require_daemon),
        )

    mpc_required = resolve_mpc_required_flag() if require_mpc is None else bool(require_mpc)
    verifier_required = (
        resolve_verifier_required_flag()
        if require_verifier is None
        else bool(require_verifier)
    )
    daemon_required = (
        resolve_daemon_required_flag()
        if require_daemon is None
        else bool(require_daemon)
    )
    all_nodes = discovery.discover(selected_model)
    if not all_nodes:
        return AdmissionResult(
            ok=False,
            reason="no_nodes",
            message=f"Model '{selected_model}' has no registered nodes.",
            model_id=selected_model,
            mpc_required=mpc_required,
            mpc_available=False,
            text=Coverage(0, 0),
            vision=Coverage(0, 0),
            verifier_required=verifier_required,
            daemon_required=daemon_required,
        )

    text_nodes = [n for n in all_nodes if n.node_type in ("compute", "mpc")]
    vision_nodes = [n for n in all_nodes if n.node_type == "vision"]
    mpc_available = any(n.node_type == "mpc" and int(n.shard_index) == 0 for n in all_nodes)
    text = _coverage_for(text_nodes)
    vision = _coverage_for(vision_nodes)

    if not text.complete:
        return AdmissionResult(
            ok=False,
            reason="incomplete_text_coverage",
            message=(
                f"Model '{selected_model}' has incomplete text coverage "
                f"({text.covered_shards}/{text.total_shards} shards)."
            ),
            model_id=selected_model,
            mpc_required=mpc_required,
            mpc_available=mpc_available,
            text=text,
            vision=vision,
            verifier_required=verifier_required,
            daemon_required=daemon_required,
        )

    if mpc_required and not mpc_available:
        return AdmissionResult(
            ok=False,
            reason="missing_mpc",
            message=(
                f"Model '{selected_model}' is missing MPC shard-0 entry while "
                "UNFED_REQUIRE_MPC=1."
            ),
            model_id=selected_model,
            mpc_required=mpc_required,
            mpc_available=mpc_available,
            text=text,
            vision=vision,
            verifier_required=verifier_required,
            daemon_required=daemon_required,
        )

    if require_vision and not vision.complete:
        return AdmissionResult(
            ok=False,
            reason="incomplete_vision_coverage",
            message=(
                f"Model '{selected_model}' has incomplete vision coverage "
                f"({vision.covered_shards}/{vision.total_shards} shards)."
            ),
            model_id=selected_model,
            mpc_required=mpc_required,
            mpc_available=mpc_available,
            text=text,
            vision=vision,
            verifier_required=verifier_required,
            daemon_required=daemon_required,
        )

    healthy_verifier_count = 0
    required_verifier_count = 1
    if verifier_required:
        health = None
        get_health = getattr(discovery, "get_verifier_health", None)
        if callable(get_health):
            health = get_health()
        if health is not None:
            healthy_verifier_count = int(
                getattr(health, "healthy_verifier_count", 0) or 0
            )
            required_verifier_count = int(
                getattr(health, "required_verifier_count", 1) or 1
            )
            is_healthy = bool(getattr(health, "healthy", False))
        else:
            is_healthy = False
        if not is_healthy:
            return AdmissionResult(
                ok=False,
                reason="missing_verifier",
                message=(
                    "No healthy verifier quorum available for admission "
                    f"({healthy_verifier_count}/{required_verifier_count})."
                ),
                model_id=selected_model,
                mpc_required=mpc_required,
                mpc_available=mpc_available,
                text=text,
                vision=vision,
                verifier_required=verifier_required,
                healthy_verifier_count=healthy_verifier_count,
                required_verifier_count=required_verifier_count,
                daemon_required=daemon_required,
            )

    healthy_daemon_count = 0
    required_daemon_count = 1
    if daemon_required:
        daemon_nodes = []
        try:
            daemon_nodes = [
                n for n in discovery.discover("")
                if getattr(n, "node_type", "") == "daemon"
            ]
        except Exception:
            daemon_nodes = [
                n for n in all_nodes
                if getattr(n, "node_type", "") == "daemon"
            ]
        healthy_daemon_count = len(daemon_nodes)
        if healthy_daemon_count < required_daemon_count:
            return AdmissionResult(
                ok=False,
                reason="missing_daemon",
                message=(
                    "No healthy daemon available for admission "
                    f"({healthy_daemon_count}/{required_daemon_count})."
                ),
                model_id=selected_model,
                mpc_required=mpc_required,
                mpc_available=mpc_available,
                text=text,
                vision=vision,
                verifier_required=verifier_required,
                healthy_verifier_count=healthy_verifier_count,
                required_verifier_count=required_verifier_count,
                daemon_required=daemon_required,
                healthy_daemon_count=healthy_daemon_count,
                required_daemon_count=required_daemon_count,
            )

    return AdmissionResult(
        ok=True,
        reason="ok",
        message="Admission preflight passed.",
        model_id=selected_model,
        mpc_required=mpc_required,
        mpc_available=mpc_available,
        text=text,
        vision=vision,
        verifier_required=verifier_required,
        healthy_verifier_count=healthy_verifier_count,
        required_verifier_count=required_verifier_count,
        daemon_required=daemon_required,
        healthy_daemon_count=healthy_daemon_count,
        required_daemon_count=required_daemon_count,
    )


def pick_first_eligible_model(
    discovery,
    *,
    require_vision: bool = False,
    require_mpc: bool | None = None,
    require_verifier: bool | None = None,
    require_daemon: bool | None = None,
) -> tuple[str | None, AdmissionResult | None]:
    model_ids = sorted({m.model_id for m in discovery.list_models() if getattr(m, "model_id", "")})
    first_failure = None
    for mid in model_ids:
        result = preflight_model_admission(
            discovery,
            mid,
            require_vision=require_vision,
            require_mpc=require_mpc,
            require_verifier=require_verifier,
            require_daemon=require_daemon,
        )
        if result.ok:
            return mid, result
        if first_failure is None:
            first_failure = result
    return None, first_failure
