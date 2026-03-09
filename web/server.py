"""
UNFED AI Web Dashboard — FastAPI server.

Provides REST + WebSocket endpoints for:
  - Chat with streaming (text and multimodal)
  - Network node visualization (Tor-style circuit view)
  - Mini-chain block explorer

Usage:
    python -m web.server
    python -m web.server --port 8080 --registry localhost:50050
"""

import asyncio
import base64
from collections import Counter, deque
from functools import lru_cache
import grpc
import ipaddress
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import uuid

logger = logging.getLogger("unfed.web")
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoConfig

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import config as app_config
import inference_pb2
import inference_pb2_grpc
from network.admission import (
    pick_first_eligible_model,
    preflight_model_admission,
    resolve_daemon_required_flag,
    resolve_mpc_required_flag,
)
from network.discovery import RegistryClient, RegistryPool
from network.infra_routing import select_least_loaded_daemon
import registry_pb2
import registry_pb2_grpc
from web.auth import WalletAuth

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="UNFED AI Dashboard", version="1.0")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_registry_address: str = app_config.REGISTRY_ADDRESS
_discovery: Optional[RegistryPool] = None
_temp_dir = tempfile.mkdtemp(prefix="unfed_uploads_")
_active_sessions: dict[str, dict] = {}
_wallet_auth = WalletAuth()
_onchain_escrow = None  # Lazily initialized from registry config
_MAX_BILLING_INPUT_TOKENS = 200_000
_MAX_BILLING_OUTPUT_TOKENS = 50_000
_ETH_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._/@:+-]{1,200}$")
_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9._-]+:\d{1,5}$")
_DEV_AUTH_BYPASS = (
    os.environ.get("UNFED_DEV_AUTH_BYPASS", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
_HE_OUTPUT_ENABLED = (
    os.environ.get("UNFED_HE_OUTPUT_ENABLED", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
_HE_OUTPUT_STRICT = (
    os.environ.get("UNFED_HE_OUTPUT_STRICT", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_ALLOWED_CLUSTER_ENDPOINTS = {
    ep.strip()
    for ep in os.environ.get("UNFED_ALLOWED_CLUSTER_ENDPOINTS", "").split(",")
    if ep.strip()
}
_FAUCET_ENABLED = (
    os.environ.get("UNFED_FAUCET_ENABLED", "0").strip().lower()
    in ("1", "true", "yes", "on")
)
_FAUCET_REQUIRE_AUTH = (
    os.environ.get("UNFED_FAUCET_REQUIRE_AUTH", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
_FAUCET_STATE_DB_PATH = os.path.expanduser(
    os.environ.get("UNFED_FAUCET_STATE_DB", "~/.unfed/faucet_state.db")
)


def _env_int(name: str, default: int, min_value: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return max(default, min_value)
    return max(parsed, min_value)


_REGISTRY_METRICS_WINDOW_SECONDS = _env_int(
    "UNFED_REGISTRY_METRICS_WINDOW_SECONDS", 900, 60
)
_REGISTRY_METRICS_MAX_SAMPLES = _env_int(
    "UNFED_REGISTRY_METRICS_MAX_SAMPLES", 1000, 50
)
_registry_perf_samples = deque(maxlen=max(_REGISTRY_METRICS_MAX_SAMPLES, 50))
_REGISTRY_SNAPSHOT_TTL_SECONDS = max(
    1.0, float(os.environ.get("UNFED_REGISTRY_SNAPSHOT_TTL_SECONDS", "2"))
)
_registry_snapshot_cache_lock = threading.Lock()
_registry_snapshot_cache: dict[tuple, tuple[float, object]] = {}
_grpc_pool_lock = threading.Lock()
_grpc_channels: dict[tuple[str, str], grpc.Channel] = {}
_grpc_stubs: dict[tuple[str, str], object] = {}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pct = min(max(float(percentile), 0.0), 1.0)
    idx = int(round((len(ordered) - 1) * pct))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _trim_perf_samples(window_seconds: Optional[int] = None) -> None:
    window = int(window_seconds or _REGISTRY_METRICS_WINDOW_SECONDS)
    now = time.time()
    while _registry_perf_samples and (now - _registry_perf_samples[0]["ts"]) > window:
        _registry_perf_samples.popleft()


def _record_perf_sample(
    model_id: str,
    output_tokens: int,
    total_time_s: float,
    ttft_ms: Optional[float],
    token_intervals_ms: list[float],
) -> None:
    if total_time_s <= 0:
        return
    tps = float(output_tokens) / float(total_time_s)
    sample = {
        "ts": time.time(),
        "model_id": model_id or "",
        "output_tokens": int(output_tokens),
        "total_time_s": float(total_time_s),
        "tokens_per_sec": float(tps),
        "ttft_ms": float(ttft_ms) if ttft_ms is not None else 0.0,
        "avg_token_interval_ms": (
            sum(token_intervals_ms) / len(token_intervals_ms)
            if token_intervals_ms else 0.0
        ),
        "p95_token_interval_ms": (
            _percentile(token_intervals_ms, 0.95)
            if token_intervals_ms else 0.0
        ),
    }
    _registry_perf_samples.append(sample)
    _trim_perf_samples()


def _perf_summary(model_id: str = "", window_seconds: Optional[int] = None) -> dict:
    _trim_perf_samples(window_seconds=window_seconds)
    requested_model = (model_id or "").strip()
    samples = list(_registry_perf_samples)
    if requested_model:
        samples = [s for s in samples if s.get("model_id") == requested_model]
    if not samples:
        return {
            "sample_count": 0,
            "avg_tps": 0.0,
            "p50_tps": 0.0,
            "p95_tps": 0.0,
            "p50_ttft_ms": 0.0,
            "p95_ttft_ms": 0.0,
            "p50_total_latency_ms": 0.0,
            "p95_total_latency_ms": 0.0,
            "window_seconds": int(window_seconds or _REGISTRY_METRICS_WINDOW_SECONDS),
        }

    tps_values = [float(s.get("tokens_per_sec", 0.0)) for s in samples]
    ttft_values = [float(s.get("ttft_ms", 0.0)) for s in samples]
    total_latency_values = [float(s.get("total_time_s", 0.0)) * 1000.0 for s in samples]

    return {
        "sample_count": len(samples),
        "avg_tps": round(sum(tps_values) / max(len(tps_values), 1), 2),
        "p50_tps": round(_percentile(tps_values, 0.50), 2),
        "p95_tps": round(_percentile(tps_values, 0.95), 2),
        "p50_ttft_ms": round(_percentile(ttft_values, 0.50), 1),
        "p95_ttft_ms": round(_percentile(ttft_values, 0.95), 1),
        "p50_total_latency_ms": round(_percentile(total_latency_values, 0.50), 1),
        "p95_total_latency_ms": round(_percentile(total_latency_values, 0.95), 1),
        "window_seconds": int(window_seconds or _REGISTRY_METRICS_WINDOW_SECONDS),
    }


def _is_loopback_host(value: str) -> bool:
    raw = (value or "").strip().lower()
    if raw in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def _faucet_db_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_FAUCET_STATE_DB_PATH), exist_ok=True)
    return sqlite3.connect(_FAUCET_STATE_DB_PATH)


def _faucet_state_init() -> None:
    with _faucet_db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faucet_drips (
                address TEXT PRIMARY KEY,
                last_drip_ts REAL NOT NULL
            )
            """
        )
        conn.commit()


def _faucet_get_last_drip(address: str) -> float:
    with _faucet_db_conn() as conn:
        row = conn.execute(
            "SELECT last_drip_ts FROM faucet_drips WHERE address = ?",
            (address,),
        ).fetchone()
    return float(row[0]) if row else 0.0


def _faucet_set_last_drip(address: str, ts: float) -> None:
    with _faucet_db_conn() as conn:
        conn.execute(
            """
            INSERT INTO faucet_drips(address, last_drip_ts)
            VALUES(?, ?)
            ON CONFLICT(address)
            DO UPDATE SET last_drip_ts = excluded.last_drip_ts
            """,
            (address, float(ts)),
        )
        conn.commit()


def get_discovery() -> RegistryPool:
    global _discovery
    if _discovery is None:
        _discovery = RegistryPool([_registry_address])
    return _discovery


def _normalize_eth_address(value: str) -> str:
    """Validate and normalize EVM addresses."""
    from web3 import Web3

    raw = (value or "").strip()
    if not raw or not _ETH_ADDRESS_RE.fullmatch(raw):
        raise ValueError("Invalid wallet address format. Expected 0x + 40 hex chars.")
    return Web3.to_checksum_address(raw)


def _validate_model_id(model_id: str) -> str:
    raw = (model_id or "").strip()
    if not raw:
        raise ValueError("model_id is required")
    if not _MODEL_ID_RE.fullmatch(raw):
        raise ValueError("Invalid model_id format")
    return raw


def _resolve_registry_endpoint(cluster_endpoint: str) -> str:
    endpoint = (cluster_endpoint or "").strip()
    if not endpoint:
        return _registry_address
    if not _ENDPOINT_RE.fullmatch(endpoint):
        raise ValueError("Invalid cluster_endpoint format. Expected host:port.")
    if endpoint == _registry_address or endpoint in _ALLOWED_CLUSTER_ENDPOINTS:
        return endpoint
    raise ValueError(
        "cluster_endpoint is not allowed. "
        "Use the local registry endpoint or configure UNFED_ALLOWED_CLUSTER_ENDPOINTS."
    )


def _close_grpc_endpoint(kind: str, endpoint: str) -> None:
    key = (str(kind), str(endpoint))
    with _grpc_pool_lock:
        _grpc_stubs.pop(key, None)
        ch = _grpc_channels.pop(key, None)
    if ch is not None:
        try:
            ch.close()
        except Exception:
            pass


def _get_grpc_stub(kind: str, endpoint: str):
    key = (str(kind), str(endpoint))
    with _grpc_pool_lock:
        stub = _grpc_stubs.get(key)
        if stub is not None:
            return stub
        channel = grpc.insecure_channel(endpoint, options=app_config.GRPC_OPTIONS)
        if kind == "registry":
            stub = registry_pb2_grpc.RegistryStub(channel)
        elif kind == "inference":
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
        else:
            raise ValueError(f"Unsupported gRPC stub kind: {kind}")
        _grpc_channels[key] = channel
        _grpc_stubs[key] = stub
        return stub


def _snapshot_cache_key(
    discovery: RegistryPool,
    *,
    model_id: str,
    include_nodes: bool,
    include_models: bool,
    include_pricing: bool,
    include_pool_health: bool,
) -> tuple:
    endpoint = getattr(discovery, "active_registry", None) or _registry_address
    return (
        str(endpoint),
        str(model_id or ""),
        bool(include_nodes),
        bool(include_models),
        bool(include_pricing),
        bool(include_pool_health),
    )


def _get_registry_snapshot_cached(
    discovery: RegistryPool,
    *,
    model_id: str = "",
    include_nodes: bool = True,
    include_models: bool = True,
    include_pricing: bool = False,
    include_pool_health: bool = False,
):
    fetch_snapshot = getattr(discovery, "get_registry_snapshot", None)
    if not callable(fetch_snapshot):
        return None

    key = _snapshot_cache_key(
        discovery,
        model_id=model_id,
        include_nodes=include_nodes,
        include_models=include_models,
        include_pricing=include_pricing,
        include_pool_health=include_pool_health,
    )
    now = time.time()
    with _registry_snapshot_cache_lock:
        cached = _registry_snapshot_cache.get(key)
        if cached and now < cached[0]:
            return cached[1]

    snapshot = fetch_snapshot(
        model_id=model_id,
        include_nodes=include_nodes,
        include_models=include_models,
        include_pricing=include_pricing,
        include_pool_health=include_pool_health,
    )
    if snapshot is None:
        return None

    with _registry_snapshot_cache_lock:
        _registry_snapshot_cache[key] = (now + _REGISTRY_SNAPSHOT_TTL_SECONDS, snapshot)
    return snapshot


def _parse_mpc_capability_json(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _node_has_mpc_capability(node, capability: str) -> bool:
    data = _parse_mpc_capability_json(getattr(node, "capability_json", ""))
    caps = data.get("mpc_capabilities")
    if isinstance(caps, list) and caps:
        normalized = {str(x).strip().lower() for x in caps}
        return capability in normalized
    return True


def _node_mpc_role(node) -> str:
    data = _parse_mpc_capability_json(getattr(node, "capability_json", ""))
    role = str(data.get("mpc_role", "")).strip().upper()
    return role if role in ("A", "B") else "A"


def _has_mpc_pair(nodes: list, capability: str) -> bool:
    scoped = [
        n
        for n in nodes
        if str(getattr(n, "node_type", "") or "") == "mpc"
        and int(getattr(n, "shard_index", -1) or -1) == 0
        and _node_has_mpc_capability(n, capability)
    ]
    has_a = any(_node_mpc_role(n) == "A" for n in scoped)
    has_b = any(_node_mpc_role(n) == "B" for n in scoped)
    return has_a and has_b


def _lookup_models(discovery: RegistryPool) -> dict[str, dict]:
    models = list(discovery.list_models())
    snapshot = _get_registry_snapshot_cached(
        discovery,
        include_nodes=True,
        include_models=False,
        include_pricing=False,
        include_pool_health=False,
    )
    all_nodes = (
        list(getattr(snapshot, "nodes", []))
        if snapshot is not None
        else list(discovery.discover(""))
    )

    daemon_required = resolve_daemon_required_flag()
    daemon_count = len(
        [n for n in all_nodes if str(getattr(n, "node_type", "") or "") == "daemon"]
    )
    mpc_required = resolve_mpc_required_flag()

    by_model_nodes: dict[str, list] = {}
    for node in all_nodes:
        node_type = str(getattr(node, "node_type", "") or "")
        model_id = str(getattr(node, "model_id", "") or "")
        if not model_id or node_type not in ("compute", "mpc"):
            continue
        by_model_nodes.setdefault(model_id, []).append(node)

    out = {}
    for m in models:
        model_id = str(getattr(m, "model_id", "") or "")
        text_nodes = by_model_nodes.get(model_id, [])
        shard_indexes = {
            int(getattr(n, "shard_index", -1))
            for n in text_nodes
            if int(getattr(n, "shard_index", -1)) >= 0
        }
        covered_shards = len(shard_indexes)
        total_shards = (max(shard_indexes) + 1) if shard_indexes else 0
        mpc_available = _has_mpc_pair(text_nodes, "input") and _has_mpc_pair(
            text_nodes, "output"
        )
        text_ready = total_shards > 0 and covered_shards == total_shards
        daemon_ready = (not daemon_required) or daemon_count >= 1
        can_serve = text_ready and ((not mpc_required) or mpc_available) and daemon_ready
        out[model_id] = {
            "can_serve": bool(can_serve),
            "covered_shards": int(covered_shards),
            "total_shards": int(total_shards),
            "mpc_available": bool(mpc_available),
            "mpc_required": bool(mpc_required),
        }
    return out


def _validate_selected_model(discovery: RegistryPool, model_id: str) -> str:
    selected = _validate_model_id(model_id)
    catalog = _lookup_models(discovery)
    if selected not in catalog:
        available = ", ".join(sorted(catalog.keys())[:8]) or "none"
        raise ValueError(
            f"Model '{selected}' is not available in this registry. "
            f"Available models: {available}"
        )
    preflight = preflight_model_admission(discovery, selected)
    if not preflight.ok:
        raise ValueError(preflight.message)
    return selected


@lru_cache(maxsize=64)
def _model_context_window(model_id: str) -> int:
    """Best-effort context window lookup for a selected model."""
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        max_pos = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        if max_pos > 0:
            return max_pos
    except Exception:
        pass
    return int(getattr(app_config, "MAX_NEW_TOKENS", 100))


def _resolve_max_new_tokens(
    model_id: str,
    input_tokens: int,
    requested_max_tokens: Optional[int],
) -> int:
    """Resolve output budget with safe defaults bounded by model context."""
    context_window = _model_context_window(model_id)
    available_budget = max(1, context_window - max(0, int(input_tokens)))
    if requested_max_tokens is not None and requested_max_tokens > 0:
        return max(1, min(int(requested_max_tokens), available_budget))

    # No UI token knob: use server default budget, but never exceed context.
    default_budget = int(getattr(app_config, "MAX_NEW_TOKENS", 100) or 100)
    return max(1, min(default_budget, available_budget))


# ---------------------------------------------------------------------------
# HTML entry point
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())


# ---------------------------------------------------------------------------
# REST: Models
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def list_models():
    """List available models from the registry."""
    try:
        discovery = get_discovery()
        models = discovery.list_models()
        admission_map = _lookup_models(discovery)
        result = []
        for m in models:
            model_info = admission_map.get(m.model_id, {})
            can_serve = bool(model_info.get("can_serve", False))
            covered_shards = int(model_info.get("covered_shards", 0))
            result.append({
                "model_id": m.model_id,
                "total_nodes": m.total_nodes,
                "total_shards": int(model_info.get("total_shards", 0)),
                "covered_shards": covered_shards,
                "can_serve": can_serve,
                "is_healthy": can_serve,  # backward compatibility
                "mpc_required": bool(model_info.get("mpc_required", True)),
                "mpc_available": bool(model_info.get("mpc_available", False)),
            })
        return {"models": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# REST: Network / Nodes
# ---------------------------------------------------------------------------
@app.get("/api/network/nodes")
async def get_nodes():
    """Get all registered nodes from the registry."""
    try:
        discovery = get_discovery()
        snapshot = _get_registry_snapshot_cached(
            discovery,
            include_nodes=True,
            include_models=False,
            include_pricing=False,
            include_pool_health=False,
        )
        all_nodes = (
            list(getattr(snapshot, "nodes", []))
            if snapshot is not None
            else list(discovery.discover(""))
        )
        nodes = []

        def _node_function(node_type: str, has_embedding: bool, has_lm_head: bool) -> str:
            t = (node_type or "").strip().lower()
            if t == "mpc":
                return "MPC shard-0 entry and share-secure compute"
            if t == "daemon":
                return "Collect shares, build blocks, and relay chain state"
            if t == "vision":
                return "Vision encoder and image feature extraction"
            # Compute fallback
            flags = []
            if has_embedding:
                flags.append("embedding")
            if has_lm_head:
                flags.append("lm_head")
            if flags:
                return f"Transformer shard ({', '.join(flags)})"
            return "Transformer shard compute"

        for n in all_nodes:
            nodes.append({
                "node_id": n.node_id,
                "address": n.address,
                "model_id": n.model_id,
                "shard_index": n.shard_index,
                "layer_start": n.layer_start,
                "layer_end": n.layer_end,
                "has_embedding": n.has_embedding,
                "has_lm_head": n.has_lm_head,
                "node_type": n.node_type,
                "public_key": base64.b64encode(n.public_key).decode()
                    if n.public_key else "",
                "node_function": _node_function(
                    n.node_type, bool(n.has_embedding), bool(n.has_lm_head)
                ),
            })

        return {"nodes": nodes}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/network/circuit")
async def get_circuit(model_id: str = ""):
    """Build and return the current circuit for a model."""
    try:
        discovery = get_discovery()

        # MPC nodes for shard 0
        mpc_nodes = discovery.discover_mpc(model_id)

        # Text circuit — replace shard 0 with MPC if available
        text_result = discovery.build_circuit(model_id)
        text_circuit = None
        if text_result:
            addrs, pks = text_result
            all_compute = discovery.discover_compute(model_id)
            # Build node info map for layer details
            node_info = {}
            for n in all_compute:
                node_info[n.address] = n

            text_circuit = []
            for i, addr in enumerate(addrs):
                entry = {
                    "address": addr,
                    "shard_index": i,
                    "node_type": "compute",
                }
                if addr in node_info:
                    entry["layer_start"] = node_info[addr].layer_start
                    entry["layer_end"] = node_info[addr].layer_end
                text_circuit.append(entry)

            # If MPC nodes exist, replace shard 0 with an MPC entry
            if mpc_nodes:
                mpc_entry = mpc_nodes[0]
                text_circuit[0] = {
                    "address": mpc_entry.address,
                    "shard_index": 0,
                    "node_type": "mpc",
                    "layer_start": 0,
                    "layer_end": 1,
                }

        # Vision circuit
        vision_result = discovery.build_vision_circuit(model_id)
        vision_circuit = None
        if vision_result:
            addrs, pks = vision_result
            vision_circuit = [{"address": a, "shard_index": i}
                              for i, a in enumerate(addrs)]

        return {
            "text_circuit": text_circuit,
            "vision_circuit": vision_circuit,
            "mpc_enabled": len(mpc_nodes) > 0,
            "mpc_entries": len(mpc_nodes),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/network/health")
async def get_health(model_id: str = ""):
    """Get pool health info."""
    try:
        discovery = get_discovery()
        health = discovery.get_pool_health(model_id)
        daemon_count = len([
            n for n in discovery.discover("")
            if getattr(n, "node_type", "") == "daemon"
        ])
        infra = None
        get_infra_telemetry = getattr(discovery, "get_infra_telemetry", None)
        if callable(get_infra_telemetry):
            infra = get_infra_telemetry()
        daemon_work_window = {}
        daemon_payout_share = {}
        winner_bonus_window = {}
        recent_winner_receipts = []
        if infra is not None:
            try:
                daemon_work_window = json.loads(
                    getattr(infra, "daemon_work_window_json", "{}") or "{}"
                )
            except Exception:
                daemon_work_window = {}
            try:
                daemon_payout_share = json.loads(
                    getattr(infra, "daemon_payout_share_json", "{}") or "{}"
                )
            except Exception:
                daemon_payout_share = {}
            try:
                winner_bonus_window = json.loads(
                    getattr(infra, "winner_bonus_window_json", "{}") or "{}"
                )
            except Exception:
                winner_bonus_window = {}
            try:
                recent_winner_receipts = json.loads(
                    getattr(infra, "recent_winner_receipts_json", "[]") or "[]"
                )
            except Exception:
                recent_winner_receipts = []
        if health:
            return {
                "model_id": health.model_id,
                "total_shards": health.total_shards,
                "is_healthy": health.overall_status == "healthy",
                "overall_status": health.overall_status,
                "can_serve": health.can_serve,
                "shards": [
                    {
                        "shard_index": s.shard_index,
                        "node_count": s.node_count,
                        "status": s.status,
                    }
                    for s in health.shards
                ],
                "daemon_required": (
                    os.environ.get("UNFED_REQUIRE_DAEMON", "1").strip().lower()
                    not in ("0", "false", "no", "off")
                ),
                "daemon_count": daemon_count,
                "healthy_daemon_count": (
                    int(getattr(infra, "healthy_daemon_count", daemon_count) or daemon_count)
                    if infra is not None else daemon_count
                ),
                "required_daemon_count": (
                    int(getattr(infra, "required_daemon_count", 1) or 1)
                    if infra is not None else 1
                ),
                "selected_daemon_recipient": (
                    getattr(infra, "selected_daemon_recipient", "") if infra is not None else ""
                ),
                "daemon_work_window": daemon_work_window,
                "daemon_payout_share": daemon_payout_share,
                "winner_bonus_window": winner_bonus_window,
                "recent_winner_receipts": recent_winner_receipts,
                "winner_receipt_count": (
                    int(getattr(infra, "winner_receipt_count", 0) or 0)
                    if infra is not None else 0
                ),
            }
        return {"error": "No health data available"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/registry/summary")
async def get_registry_summary(model_id: str = "", window_seconds: int = 0):
    """Aggregate registry topology, pricing, and recent generation performance."""
    try:
        selected_model = (model_id or "").strip()
        discovery = get_discovery()
        snapshot = _get_registry_snapshot_cached(
            discovery,
            model_id=selected_model,
            include_nodes=True,
            include_models=True,
            include_pricing=True,
            include_pool_health=False,
        )
        all_nodes = (
            list(getattr(snapshot, "nodes", []))
            if snapshot is not None
            else list(discovery.discover(""))
        )
        models = (
            list(getattr(snapshot, "models", []))
            if snapshot is not None
            else list(discovery.list_models())
        )
        admission_map = _lookup_models(discovery)

        type_counts = Counter()
        model_node_counts = Counter()
        for node in all_nodes:
            node_type = str(getattr(node, "node_type", "") or "unknown")
            node_model = str(getattr(node, "model_id", "") or "")
            type_counts[node_type] += 1
            if node_model:
                model_node_counts[node_model] += 1

        model_rows = []
        healthy_models = 0
        for model in models:
            model_info = admission_map.get(model.model_id, {})
            can_serve = bool(model_info.get("can_serve", False))
            if can_serve:
                healthy_models += 1
            model_rows.append({
                "model_id": model.model_id,
                "total_nodes": int(model.total_nodes or 0),
                "registered_nodes": int(model_node_counts.get(model.model_id, 0)),
                "can_serve": can_serve,
                "covered_shards": int(model_info.get("covered_shards", 0)),
                "total_shards": int(model_info.get("total_shards", 0)),
                "mpc_available": bool(model_info.get("mpc_available", False)),
                "mpc_required": bool(model_info.get("mpc_required", False)),
            })

        selected_health = None
        if selected_model:
            selected_info = admission_map.get(selected_model, {})
            selected_health = {
                "model_id": selected_model,
                "can_serve": bool(selected_info.get("can_serve", False)),
                "covered_shards": int(selected_info.get("covered_shards", 0)),
                "total_shards": int(selected_info.get("total_shards", 0)),
                "registered_nodes": int(model_node_counts.get(selected_model, 0)),
                "mpc_available": bool(selected_info.get("mpc_available", False)),
                "mpc_required": bool(selected_info.get("mpc_required", False)),
            }

        snapshot_pricing = getattr(snapshot, "pricing", None) if snapshot is not None else None
        if snapshot_pricing and (
            getattr(snapshot_pricing, "currency", "")
            or float(getattr(snapshot_pricing, "price_per_input_token", 0.0) or 0.0) > 0.0
            or float(getattr(snapshot_pricing, "price_per_output_token", 0.0) or 0.0) > 0.0
        ):
            pricing = {
                "price_per_input_token": float(
                    getattr(snapshot_pricing, "price_per_input_token", 0.0) or 0.0
                ),
                "price_per_output_token": float(
                    getattr(snapshot_pricing, "price_per_output_token", 0.0) or 0.0
                ),
                "currency": str(getattr(snapshot_pricing, "currency", "UNFED") or "UNFED"),
                "model_id": str(getattr(snapshot_pricing, "model_id", "") or ""),
            }
        else:
            pricing = await get_pricing(model_id=selected_model)
        perf = _perf_summary(
            model_id=selected_model,
            window_seconds=(window_seconds if window_seconds > 0 else None),
        )

        return {
            "timestamp": int(time.time()),
            "registry": {
                "total_nodes": len(all_nodes),
                "by_type": dict(type_counts),
                "healthy_models": healthy_models,
                "total_models": len(models),
            },
            "pricing": pricing,
            "performance": perf,
            "model_health": selected_health,
            "models": model_rows,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# REST: Mini-Chain (syncs from network like a Monero wallet)
# ---------------------------------------------------------------------------
def _get_chain():
    """Get or create a local ShareChain (light-wallet mode).

    Like a Monero wallet — we don't produce blocks, we just receive
    them from the daemon via SubscribeBlocks or GetBlocks.
    """
    from economics.share_chain import ShareChain
    if not hasattr(_get_chain, "_chain"):
        _get_chain._chain = ShareChain(block_interval=10.0, settlement_blocks=6)
    return _get_chain._chain


def _discover_daemon_with_utilization():
    """Find least-loaded daemon and return (daemon, utilization map)."""
    try:
        discovery = get_discovery()
        all_nodes = discovery.discover("")
        daemons = [n for n in all_nodes if n.node_type == "daemon"]
        if daemons:
            def _probe(daemon):
                try:
                    stub = _get_grpc_stub("inference", daemon.address)
                    fee = stub.GetLoad(
                        inference_pb2.FeeEstimateRequest(estimated_tokens=1),
                        timeout=2,
                    )
                    return float(getattr(fee, "utilization", 1.0))
                except Exception:
                    _close_grpc_endpoint("inference", daemon.address)
                    stub = _get_grpc_stub("inference", daemon.address)
                    fee = stub.GetLoad(
                        inference_pb2.FeeEstimateRequest(estimated_tokens=1),
                        timeout=2,
                    )
                    return float(getattr(fee, "utilization", 1.0))

            selected, utilization = select_least_loaded_daemon(daemons, _probe)
            if selected is None:
                return daemons[0], utilization
            return selected, utilization
    except Exception as e:
        logger.debug("Daemon discovery failed: %s", e)
    return None, {}


def _discover_daemon():
    daemon, _ = _discover_daemon_with_utilization()
    return daemon


def _rebuild_settlements_from_blocks(blocks: list, settlement_blocks: int) -> list:
    """Reconstruct settlement summaries from imported daemon blocks.

    This is used by dashboard trusted-sync mode, where blocks are accepted as
    canonical from the daemon snapshot.
    """
    if settlement_blocks <= 0 or not blocks:
        return []
    try:
        from economics.share_chain import SettlementSummary
    except Exception:
        return []

    by_index = {int(getattr(b, "index", -1)): b for b in blocks}
    max_index = max(by_index.keys(), default=0)
    settlements = []

    for end_idx in range(settlement_blocks, max_index + 1, settlement_blocks):
        start_idx = max(1, end_idx - settlement_blocks + 1)
        node_shares: dict[str, float] = {}
        total = 0.0
        total_tokens = 0
        earliest = float("inf")
        latest = 0.0

        for i in range(start_idx, end_idx + 1):
            block = by_index.get(i)
            if block is None:
                continue
            for share in getattr(block, "shares", []) or []:
                if not getattr(share, "validated", False):
                    continue
                node_id = str(getattr(share, "node_id", "") or "")
                if not node_id:
                    continue
                weight = float(getattr(share, "share_weight", 1.0) or 1.0)
                node_shares[node_id] = node_shares.get(node_id, 0.0) + weight
                total += weight
                total_tokens += int(getattr(share, "tokens_processed", 0) or 0)
                ts = float(getattr(share, "timestamp", 0.0) or 0.0)
                if ts > 0:
                    earliest = min(earliest, ts)
                    latest = max(latest, ts)

        now = time.time()
        if earliest == float("inf"):
            earliest = now
        if latest == 0.0:
            latest = now

        settlement = SettlementSummary(
            period_start=earliest,
            period_end=latest,
            block_range=(start_idx, end_idx),
            node_shares=node_shares,
            total_shares=total,
            total_tokens=total_tokens,
        )
        settlement.finalize()
        settlements.append(settlement)

    return settlements


def _sync_chain():
    """Pull latest blocks from the daemon (light-wallet mode).

    Discovers a daemon node via registry, then calls GetBlocks gRPC
    to fetch any blocks we don't have yet. Falls back to compute nodes
    if no daemon is available.
    """
    import grpc
    import inference_pb2
    import inference_pb2_grpc
    from economics.distributed_chain import proto_to_block

    chain = _get_chain()
    my_height = chain.get_tip_height()

    def _trusted_replace_from_daemon(stub) -> int:
        """Fallback sync path for dashboard light-wallet mode.

        Some chain hashes can differ after proto float round-trips, which makes
        strict local validation reject all remote blocks. For UI visibility, we
        accept the daemon's canonical chain snapshot as source of truth.
        """
        full = stub.GetBlocks(
            inference_pb2.GetBlocksRequest(from_height=0),
            timeout=10,
        )
        imported = [proto_to_block(bm) for bm in full.blocks]
        if not imported:
            return 0
        rebuilt_settlements = _rebuild_settlements_from_blocks(
            imported,
            settlement_blocks=int(getattr(chain, "settlement_blocks", 6) or 6),
        )
        with chain._lock:  # dashboard-local cache only
            chain._chain = imported
            chain._pending_shares = []
            chain._settlements = rebuilt_settlements
        return len(imported)

    # Try daemon first (preferred)
    daemon = _discover_daemon()
    if daemon:
        try:
            channel = grpc.insecure_channel(
                daemon.address, options=app_config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
            resp = stub.GetBlocks(
                inference_pb2.GetBlocksRequest(from_height=my_height + 1),
                timeout=10,
            )

            synced = 0
            invalid_hash_count = 0
            for block_msg in resp.blocks:
                block = proto_to_block(block_msg)
                accepted, reason = chain.receive_external_block(block)
                if accepted:
                    synced += 1
                elif reason == "invalid_hash":
                    invalid_hash_count += 1

            if synced > 0:
                print(f"[Dashboard] Synced {synced} block(s) from daemon "
                      f"{daemon.address} "
                      f"(height: {my_height} -> {chain.get_tip_height()})")
                return synced
            if invalid_hash_count > 0:
                imported = _trusted_replace_from_daemon(stub)
                if imported > 0:
                    print(f"[Dashboard] Synced daemon snapshot ({imported} block(s)) "
                          f"from {daemon.address} via trusted import")
                    return imported
            return 0
        except Exception:
            pass  # Daemon unreachable, try compute nodes as fallback

    # Fallback: try compute nodes (legacy behavior)
    try:
        discovery = get_discovery()
        nodes = discovery.discover_compute("")
        if not nodes:
            return 0
    except Exception:
        return 0

    for node in nodes[:3]:  # try first 3
        try:
            current_height = chain.get_tip_height()
            channel = grpc.insecure_channel(
                node.address, options=app_config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
            resp = stub.GetBlocks(
                inference_pb2.GetBlocksRequest(from_height=current_height + 1),
                timeout=10,
            )

            synced = 0
            for block_msg in resp.blocks:
                block = proto_to_block(block_msg)
                accepted, reason = chain.receive_external_block(block)
                if accepted:
                    synced += 1
            if synced > 0:
                return synced
        except Exception:
            continue

    return 0


@app.get("/api/chain/info")
async def chain_info():
    """Get chain summary (syncs from network first)."""
    try:
        _sync_chain()
        chain = _get_chain()
        return chain.get_chain_info()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/chain/blocks")
async def chain_blocks(start: int = 0, limit: int = 20):
    """Get paginated blocks (syncs from network first)."""
    try:
        _sync_chain()
        chain = _get_chain()
        blocks = chain.get_blocks_from(start)
        result = []
        for b in blocks[:limit]:
            result.append(b.to_dict())
        return {"blocks": result, "total": chain.height}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/chain/settlements")
async def chain_settlements():
    """Get all settlement summaries (syncs from network first)."""
    try:
        _sync_chain()
        chain = _get_chain()
        settlements = chain.get_settlements()
        return {"settlements": [s.to_dict() for s in settlements]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/chain/node-totals")
async def chain_node_totals():
    """Get per-node share counts (syncs from network first)."""
    try:
        _sync_chain()
        chain = _get_chain()
        totals = chain.get_node_totals()
        return {"node_totals": totals}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/chain/fees")
async def chain_fees():
    """Get current fee market data from the daemon's fee oracle."""
    try:
        daemon, candidates = _discover_daemon_with_utilization()
        if not daemon:
            return {
                "base_fee": 0.001,
                "utilization": 0.0,
                "estimated_cost_100": 0.1,
                "suggested_tip": 0.0,
                "daemon_available": False,
                "selected_daemon": "",
                "daemon_candidates": candidates,
                "fee_history": [],
            }

        try:
            stub = _get_grpc_stub("inference", daemon.address)
            resp = stub.GetLoad(
                inference_pb2.FeeEstimateRequest(estimated_tokens=100),
                timeout=5,
            )
        except Exception:
            _close_grpc_endpoint("inference", daemon.address)
            stub = _get_grpc_stub("inference", daemon.address)
            resp = stub.GetLoad(
                inference_pb2.FeeEstimateRequest(estimated_tokens=100),
                timeout=5,
            )

        return {
            "base_fee": resp.base_fee,
            "utilization": resp.utilization,
            "estimated_cost_100": resp.estimated_cost,
            "suggested_tip": resp.suggested_tip,
            "daemon_available": True,
            "selected_daemon": daemon.address,
            "daemon_candidates": candidates,
            "daemon_required": (
                os.environ.get("UNFED_REQUIRE_DAEMON", "1").strip().lower()
                not in ("0", "false", "no", "off")
            ),
        }
    except Exception as e:
        return {
            "base_fee": 0.001,
            "utilization": 0.0,
            "estimated_cost_100": 0.1,
            "suggested_tip": 0.0,
            "daemon_available": False,
            "selected_daemon": "",
            "daemon_candidates": {},
            "error": str(e),
        }


@app.get("/api/chain/payout-ledger")
async def chain_payout_ledger(limit: int = 8):
    """Return payout ledger rows (estimated/finalized-like) for the dashboard."""
    try:
        _sync_chain()
        chain = _get_chain()
        settlements = chain.get_settlements()
        if not settlements:
            return {"rows": [], "meta": {"settlements": 0}}

        max_settlements = max(1, min(int(limit or 8), 30))
        selected = settlements[-max_settlements:]

        discovery = get_discovery()
        all_nodes = discovery.discover("")
        daemon_count = len([n for n in all_nodes if getattr(n, "node_type", "") == "daemon"])

        # Best-effort fee/pricing snapshot for payout estimation.
        base_fee = 0.001
        utilization = 0.0
        selected_daemon = ""
        daemon, _ = _discover_daemon_with_utilization()
        if daemon is not None:
            selected_daemon = str(getattr(daemon, "address", "") or "")
            try:
                stub = _get_grpc_stub("inference", daemon.address)
                fee = stub.GetLoad(
                    inference_pb2.FeeEstimateRequest(estimated_tokens=100),
                    timeout=5,
                )
                base_fee = float(getattr(fee, "base_fee", base_fee) or base_fee)
                utilization = float(getattr(fee, "utilization", 0.0) or 0.0)
            except Exception:
                _close_grpc_endpoint("inference", daemon.address)

        pricing = await get_pricing(model_id="")
        output_price = float(pricing.get("price_per_output_token", 0.0) or 0.0)
        currency = str(pricing.get("currency", "UNFED") or "UNFED")

        daemon_work_window = {}
        daemon_payout_share = {}
        winner_bonus_window = {}
        infra = None
        get_infra_telemetry = getattr(discovery, "get_infra_telemetry", None)
        if callable(get_infra_telemetry):
            infra = get_infra_telemetry()
        if infra is not None:
            try:
                daemon_work_window = json.loads(
                    getattr(infra, "daemon_work_window_json", "{}") or "{}"
                )
            except Exception:
                daemon_work_window = {}
            try:
                daemon_payout_share = json.loads(
                    getattr(infra, "daemon_payout_share_json", "{}") or "{}"
                )
            except Exception:
                daemon_payout_share = {}
            try:
                winner_bonus_window = json.loads(
                    getattr(infra, "winner_bonus_window_json", "{}") or "{}"
                )
            except Exception:
                winner_bonus_window = {}

        rows: list[dict] = []
        for idx, settlement in enumerate(reversed(selected), start=1):
            total_shares = float(getattr(settlement, "total_shares", 0.0) or 0.0)
            total_tokens = int(getattr(settlement, "total_tokens", 0) or 0)
            if output_price > 0.0 and total_tokens > 0:
                pool_amount = float(total_tokens) * output_price
                pool_basis = "token_pricing"
            else:
                pool_amount = total_shares * base_fee
                pool_basis = "base_fee"

            node_shares = dict(getattr(settlement, "node_shares", {}) or {})
            for node_id, weight in sorted(node_shares.items(), key=lambda kv: float(kv[1]), reverse=True):
                w = float(weight or 0.0)
                share_ratio = (w / total_shares) if total_shares > 0 else 0.0
                rows.append(
                    {
                        "kind": "compute",
                        "status": "estimated",
                        "settlement_index": len(settlements) - idx + 1,
                        "settlement_hash": str(getattr(settlement, "settlement_hash", "") or ""),
                        "recipient": str(node_id),
                        "weight": round(w, 6),
                        "share_ratio": round(share_ratio, 8),
                        "estimated_amount": round(pool_amount * share_ratio, 8),
                        "currency": currency,
                        "pool_basis": pool_basis,
                        "block_start": int(getattr(settlement, "block_range", (0, 0))[0]),
                        "block_end": int(getattr(settlement, "block_range", (0, 0))[1]),
                    }
                )

        for recipient, units in sorted(daemon_work_window.items(), key=lambda kv: float(kv[1]), reverse=True):
            rows.append(
                {
                    "kind": "daemon_pending",
                    "status": "pending",
                    "recipient": str(recipient),
                    "weight": round(float(units or 0.0), 6),
                    "share_ratio": round(float(daemon_payout_share.get(recipient, 0.0) or 0.0), 8),
                    "estimated_amount": None,
                    "currency": currency,
                    "pool_basis": "infra_window",
                }
            )

        for winner, bonus_units in sorted(winner_bonus_window.items(), key=lambda kv: float(kv[1]), reverse=True):
            rows.append(
                {
                    "kind": "winner_bonus_pending",
                    "status": "pending",
                    "recipient": str(winner),
                    "weight": round(float(bonus_units or 0.0), 6),
                    "share_ratio": 0.0,
                    "estimated_amount": None,
                    "currency": currency,
                    "pool_basis": "winner_bonus_window",
                }
            )

        rows.sort(
            key=lambda r: (
                0 if r.get("kind") == "compute" else (1 if r.get("kind") == "daemon_pending" else 2),
                -float(r.get("estimated_amount") or 0.0),
                -float(r.get("weight") or 0.0),
            )
        )
        rows = rows[:300]
        return {
            "rows": rows,
            "meta": {
                "settlements": len(settlements),
                "selected_settlements": len(selected),
                "daemon_count": daemon_count,
                "selected_daemon": selected_daemon,
                "base_fee": base_fee,
                "utilization": utilization,
                "currency": currency,
                "price_per_output_token": output_price,
            },
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
@app.get("/api/pricing")
async def get_pricing(model_id: str = ""):
    """Get per-token pricing from the registry."""
    try:
        stub = _get_grpc_stub("registry", _registry_address)
        requested_model_id = (model_id or "").strip()
        try:
            resp = stub.GetPricing(
                registry_pb2.GetPricingRequest(model_id=model_id),
                timeout=5,
            )
        except Exception:
            _close_grpc_endpoint("registry", _registry_address)
            stub = _get_grpc_stub("registry", _registry_address)
            resp = stub.GetPricing(
                registry_pb2.GetPricingRequest(model_id=model_id),
                timeout=5,
            )
        # Backward-compatible fallback: if model-scoped pricing lookup fails in the
        # registry, retry cluster-level defaults instead of returning zeros.
        if (
            requested_model_id
            and float(getattr(resp, "price_per_input_token", 0.0) or 0.0) <= 0.0
            and float(getattr(resp, "price_per_output_token", 0.0) or 0.0) <= 0.0
        ):
            try:
                resp = stub.GetPricing(
                    registry_pb2.GetPricingRequest(model_id=""),
                    timeout=5,
                )
            except Exception:
                pass
        return {
            "price_per_input_token": resp.price_per_input_token,
            "price_per_output_token": resp.price_per_output_token,
            "currency": resp.currency,
            "model_id": resp.model_id,
        }
    except Exception as e:
        return {
            "price_per_input_token": 0,
            "price_per_output_token": 0,
            "currency": "UNFED",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Escrow / balance helpers
# ---------------------------------------------------------------------------

def _get_escrow():
    """Lazily initialize on-chain escrow from deployed.env if available."""
    global _onchain_escrow
    if _onchain_escrow is not None:
        return _onchain_escrow

    env_path = os.path.join(PROJECT_ROOT, "deployed.env")
    if os.path.exists(env_path):
        try:
            from economics.onchain import load_from_env
            _onchain_escrow = load_from_env(env_path)
            print(f"[Web] On-chain escrow loaded from {env_path}")
        except Exception as e:
            print(f"[Web] Could not load on-chain escrow: {e}")
    return _onchain_escrow


def _check_client_balance(client_address: str,
                          estimated_cost_wei: int = 0) -> tuple[bool, float]:
    """Check if a client has sufficient escrow balance.

    Returns (has_sufficient_balance, balance_in_tokens).
    If no escrow is configured, always returns True (permissive mode).
    """
    escrow = _get_escrow()
    if escrow is None:
        return True, 0.0

    try:
        balance = escrow.get_client_balance(client_address)
        balance_tokens = balance / 1e18
        if estimated_cost_wei > 0:
            return balance >= estimated_cost_wei, balance_tokens
        return balance > 0, balance_tokens
    except Exception as e:
        print(f"[Web] Balance check failed for {client_address}: {e}")
        return False, 0.0


@app.get("/api/client/balance")
async def get_client_balance(address: str = ""):
    """Get a client's escrow balance."""
    if not address:
        address = _wallet_auth.default_address
    if not address:
        return JSONResponse(
            status_code=400,
            content={"error": "address is required when demo auth is disabled."},
        )
    try:
        address = _normalize_eth_address(address)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    escrow = _get_escrow()
    if escrow is None:
        return {
            "address": address,
            "balance_wei": 0,
            "balance_tokens": 0.0,
            "escrow_enabled": False,
        }
    try:
        balance_wei = escrow.get_client_balance(address)
        return {
            "address": address,
            "balance_wei": balance_wei,
            "balance_tokens": balance_wei / 1e18,
            "escrow_enabled": True,
        }
    except Exception as e:
        return {
            "address": address,
            "balance_wei": 0,
            "balance_tokens": 0.0,
            "escrow_enabled": True,
            "error": str(e),
        }


class ClientAuthVerifyRequest(BaseModel):
    challenge: str
    signature: str
    address: str = ""


@app.post("/api/client/auth")
async def client_auth():
    """Get a challenge for wallet signature authentication."""
    challenge = _wallet_auth.generate_challenge()
    return {"challenge": challenge}


@app.get("/api/client/auth/mode")
async def client_auth_mode():
    """Expose wallet auth mode to frontend (strict vs dev bypass)."""
    return {
        "dev_auth_bypass": _DEV_AUTH_BYPASS,
    }


@app.get("/api/security/modes")
async def security_modes():
    """Expose enabled security transport modes for frontend/backends."""
    return {
        "he_output_enabled": _HE_OUTPUT_ENABLED,
        "he_output_strict": _HE_OUTPUT_STRICT,
        "he_compute_mode": app_config.HE_COMPUTE_MODE,
        "he_sidecar_configured": False,
        "registry_adjudication_enabled": True,
        "return_path_encryption_supported": True,
    }


@app.post("/api/client/auth/verify")
async def client_auth_verify(body: ClientAuthVerifyRequest):
    """Verify wallet signature and issue a short-lived session token."""
    challenge = (body.challenge or "").strip()
    signature = (body.signature or "").strip()
    claimed = (body.address or "").strip()
    if not challenge or not signature:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing challenge or signature."},
        )

    recovered = _wallet_auth.consume_signed_challenge(challenge, signature)
    if not recovered:
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or expired wallet challenge/signature."},
        )

    try:
        recovered_addr = _normalize_eth_address(recovered)
        if claimed:
            claimed_addr = _normalize_eth_address(claimed)
            if claimed_addr != recovered_addr:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Signed address does not match claimed wallet."},
                )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    session_token = _wallet_auth.create_session(recovered_addr)
    return {
        "success": True,
        "address": recovered_addr,
        "session_token": session_token,
        "expires_in_seconds": _wallet_auth.session_ttl_seconds,
    }


# ---------------------------------------------------------------------------
# Faucet (testnet token distribution)
# ---------------------------------------------------------------------------
class FaucetRequest(BaseModel):
    address: str
    session_token: str = ""

@app.post("/api/faucet")
async def faucet_drip(body: FaucetRequest, authorization: str = Header(default="")):
    """Drip test tokens into a client's escrow balance."""
    if not _FAUCET_ENABLED:
        return JSONResponse(status_code=403, content={
            "error": "Faucet is disabled. Set UNFED_FAUCET_ENABLED=1 to enable."
        })

    escrow = _get_escrow()
    if escrow is None:
        return JSONResponse(status_code=503, content={
            "error": "Escrow not configured. Deploy contracts and provide deployed.env."
        })

    address = body.address.strip()
    if not address:
        return JSONResponse(status_code=400, content={
            "error": "Missing 'address' field."
        })
    try:
        address = _normalize_eth_address(address)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    if _FAUCET_REQUIRE_AUTH:
        bearer = ""
        auth_header = (authorization or "").strip()
        if auth_header.lower().startswith("bearer "):
            bearer = auth_header[7:].strip()
        session_token = (body.session_token or "").strip() or bearer
        session_addr = _wallet_auth.get_session_address(session_token)
        if not session_addr:
            return JSONResponse(status_code=401, content={
                "error": "Authenticated session required for faucet."
            })
        try:
            session_addr = _normalize_eth_address(session_addr)
        except ValueError:
            return JSONResponse(status_code=401, content={
                "error": "Invalid session identity."
            })
        if session_addr != address:
            return JSONResponse(status_code=403, content={
                "error": "Faucet requests must target the authenticated wallet."
            })

    now = time.time()
    cooldown = escrow.FAUCET_COOLDOWN
    last = _faucet_get_last_drip(address)
    remaining = cooldown - (now - last)
    if remaining > 0:
        return JSONResponse(status_code=429, content={
            "error": "Cooldown active.",
            "retry_after_seconds": int(remaining),
        })

    try:
        tx_hash = escrow.faucet_drip(address)
        _faucet_set_last_drip(address, time.time())
        _, balance = _check_client_balance(address)
        return {
            "success": True,
            "amount": escrow.FAUCET_DRIP_AMOUNT,
            "tx_hash": tx_hash,
            "balance": round(balance, 4),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------
@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for multimodal queries."""
    try:
        ext = os.path.splitext(file.filename or "image.png")[1] or ".png"
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(_temp_dir, filename)

        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        return {"path": filepath, "filename": filename, "size": len(contents)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------------------------
# WebSocket: Chat (streaming generation)
# ---------------------------------------------------------------------------
@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """
    WebSocket for streaming chat generation.

    Query params:
        session: Wallet-auth session token from /api/client/auth/verify.
        wallet: accepted only when UNFED_DEV_AUTH_BYPASS=1.

    Client sends:
        {
            "prompt": "...",
            "image_path": "..." (optional),
            "model_type": "qwen2" | "qwen2_vl",
            "max_tokens": 100,
            "use_voting": false,
            "model_id": "" (optional),
            "cluster_endpoint": "" (optional — override registry)
        }

    Server sends:
        {type: "status", message: "..."}
        {type: "circuit", nodes: [...], vision_nodes: [...]}
        {type: "hop", shard_index: int, address: "...", phase: "vision"|"text"}
        {type: "token", text: "...", token_id: int, step_time: float, step: int}
        {type: "done", total_tokens: int, total_time: float, tokens_per_sec: float}
        {type: "error", message: "..."}
    """
    await websocket.accept()

    if _DEV_AUTH_BYPASS:
        remote_host = (websocket.client.host if websocket.client else "") or ""
        if not _is_loopback_host(remote_host):
            await websocket.send_json({
                "type": "error",
                "message": (
                    "Dev auth bypass is restricted to loopback clients. "
                    "Disable UNFED_DEV_AUTH_BYPASS for public deployments."
                ),
            })
            await websocket.close(code=1008)
            return
        wallet_param = (websocket.query_params.get("wallet", "") or "").strip()
        if not wallet_param:
            await websocket.send_json({
                "type": "error",
                "message": "Dev bypass enabled: provide wallet query param.",
            })
            await websocket.close(code=1008)
            return
        try:
            client_address = _normalize_eth_address(wallet_param)
        except ValueError as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1008)
            return
    else:
        # Strict wallet auth: session token is required for chat websocket.
        session_token = (websocket.query_params.get("session", "") or "").strip()
        client_address = _wallet_auth.get_session_address(session_token)
        if not client_address:
            await websocket.send_json({
                "type": "error",
                "message": "Wallet authentication required. Sign in and retry.",
            })
            await websocket.close(code=1008)
            return

    # Notify the client which address is active for this session
    await websocket.send_json({
        "type": "wallet",
        "address": client_address,
    })

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            image_path = data.get("image_path")
            model_type = data.get("model_type", "qwen2")
            max_tokens_raw = data.get("max_tokens")
            try:
                max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None
            except (TypeError, ValueError):
                max_tokens = None
            use_voting = data.get("use_voting", False)
            model_id = data.get("model_id", "")
            cluster_endpoint = data.get("cluster_endpoint", "")
            he_output_requested = bool(data.get("he_output", False))
            he_output_enabled = _HE_OUTPUT_ENABLED or he_output_requested

            await _run_generation(
                websocket, prompt, image_path, model_type,
                max_tokens, use_voting, model_id,
                cluster_endpoint=cluster_endpoint,
                client_address=client_address,
                he_output_enabled=he_output_enabled,
            )

    except WebSocketDisconnect:
        logger.debug("Chat WebSocket disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception as send_err:
            logger.debug("Failed to send error to client: %s (original: %s)", send_err, e)


async def _run_generation(
    websocket: WebSocket,
    prompt: str,
    image_path: Optional[str],
    model_type: str,
    max_tokens: Optional[int],
    use_voting: bool,
    model_id: str,
    cluster_endpoint: str = "",
    client_address: str = "",
    he_output_enabled: bool = False,
):
    """Run generation and stream results to the WebSocket."""
    from client.client import UnfedClient

    await websocket.send_json({
        "type": "status",
        "message": f"Initializing client ({model_type})..."
    })

    if not client_address:
        await websocket.send_json({
            "type": "error",
            "message": "Wallet session missing for this request.",
        })
        return

    try:
        # --- Client deposit balance check ---
        has_balance, balance_tokens = _check_client_balance(client_address)
        if not has_balance:
            await websocket.send_json({
                "type": "error",
                "message": (
                    f"Insufficient escrow balance. "
                    f"Your balance: {balance_tokens:.4f} UNFED. "
                    f"Please deposit funds to your escrow account "
                    f"({client_address}) before making requests."
                ),
            })
            return

        registry_addr = _resolve_registry_endpoint(cluster_endpoint)
        discovery = (get_discovery()
                     if registry_addr == _registry_address
                     else RegistryPool([registry_addr]))

        # Resolve selected model strictly against registry catalog.
        require_vision = model_type in ("qwen2_vl", "smolvlm") and bool(image_path)
        if model_id:
            model_id = _validate_selected_model(discovery, model_id)
        else:
            model_id, first_failure = pick_first_eligible_model(
                discovery,
                require_vision=require_vision,
            )
            if not model_id:
                if first_failure is not None:
                    raise RuntimeError(first_failure.message)
                raise RuntimeError("No eligible models available in registry.")

        preflight = preflight_model_admission(
            discovery,
            model_id,
            require_vision=require_vision,
        )
        if not preflight.ok:
            raise RuntimeError(preflight.message)

        # Create client (use cluster-specific endpoint if selected)
        client = UnfedClient(
            registry_address=registry_addr,
            use_voting=use_voting,
            model_id=model_id,
            use_he_output=he_output_enabled,
        )
        if he_output_enabled:
            await websocket.send_json({
                "type": "status",
                "message": "HE output artifact mode enabled.",
            })

        # For image-based multimodal requests, enforce a deterministic default
        # prompt before token accounting and generation.
        if model_type in ("qwen2_vl", "smolvlm") and image_path and not prompt.strip():
            prompt = "Describe this image."

        # Discover and send circuit info

        text_circuit = discovery.build_circuit(model_id)
        vision_circuit = None

        if model_type in ("qwen2_vl", "smolvlm"):
            vision_circuit = discovery.build_vision_circuit(model_id)

        # --- Validate circuit availability ---
        if text_circuit is None:
            # No compute nodes found for this model — tell the user clearly
            available = discovery.discover_compute("")
            registered_models = set()
            for n in available:
                if n.model_id:
                    registered_models.add(n.model_id)

            hint = ""
            if registered_models:
                hint = (f" Registered models: {', '.join(registered_models)}."
                        f" The '{model_id}' model needs dedicated nodes to be"
                        f" running and registered with the registry.")
            else:
                hint = " No compute nodes are currently registered."

            await websocket.send_json({
                "type": "error",
                "message": (
                    f"No compute nodes found for model '{model_id}'.{hint}"
                    f" Start nodes with: python -m node.server --model-id {model_id}"
                ),
            })
            return

        if model_type in ("qwen2_vl", "smolvlm") and vision_circuit is None:
            await websocket.send_json({
                "type": "error",
                "message": (
                    f"No vision nodes found for {model_type} model. "
                    "The multimodal pipeline requires dedicated vision shard "
                    "nodes to be running. "
                ),
            })
            return

        if model_type == "qwen2_vl" and image_path is None:
            await websocket.send_json({
                "type": "error",
                "message": "Qwen2-VL model requires an image. Please attach an image.",
            })
            return

        # SmolVLM can do text-only (skip vision pipeline) or multimodal
        if model_type == "smolvlm" and image_path is None:
            await websocket.send_json({
                "type": "status",
                "message": "SmolVLM text-only mode (no image attached)"
            })

        input_tokens = _count_request_input_tokens(
            client=client,
            model_type=model_type,
            model_id=model_id,
            prompt=prompt,
            image_path=image_path,
        )
        if input_tokens < 0 or input_tokens > _MAX_BILLING_INPUT_TOKENS:
            raise RuntimeError(
                f"Invalid input token count: {input_tokens}. "
                f"Allowed range: 0..{_MAX_BILLING_INPUT_TOKENS}."
            )
        logger.debug(
            "[Billing] exact input_tokens=%s model_type=%s model_id=%s",
            input_tokens, model_type, model_id
        )

        effective_max_tokens = _resolve_max_new_tokens(
            model_id=model_id,
            input_tokens=input_tokens,
            requested_max_tokens=max_tokens,
        )

        # Send circuit info to frontend
        circuit_msg = {
            "type": "circuit",
            "text_nodes": [],
            "vision_nodes": [],
        }
        text_hop_addresses: list[str] = []
        vision_hop_addresses: list[str] = []

        # Check for MPC nodes — use them for shard 0 if available
        mpc_nodes = discovery.discover_mpc(model_id)
        using_mpc = len(mpc_nodes) > 0

        if text_circuit:
            addrs, pks = text_circuit
            text_hop_addresses = list(addrs)
            # Get full node info for each address
            all_nodes = discovery.discover("")
            node_map = {n.address: n for n in all_nodes}

            # If MPC available, swap shard 0 with MPC entry
            if using_mpc:
                mpc_entry = mpc_nodes[0]
                addrs[0] = mpc_entry.address
                text_hop_addresses[0] = mpc_entry.address
                node_map[mpc_entry.address] = mpc_entry

                # Discover the MPC peer (Node B) — convention: port - 1
                mpc_a_addr = mpc_entry.address
                try:
                    host, port_str = mpc_a_addr.rsplit(":", 1)
                    mpc_b_addr = f"{host}:{int(port_str) - 1}"
                except Exception:
                    mpc_b_addr = "unknown"

                circuit_msg["mpc_pair"] = {
                    "node_a": mpc_a_addr,
                    "node_b": mpc_b_addr,
                }

                await websocket.send_json({
                    "type": "status",
                    "message": f"MPC enabled — shard 0 uses secret-shared pair (A={mpc_a_addr}, B={mpc_b_addr})"
                })

            for i, addr in enumerate(addrs):
                info = node_map.get(addr)
                circuit_msg["text_nodes"].append({
                    "address": addr,
                    "shard_index": i,
                    "layer_start": info.layer_start if info else 0,
                    "layer_end": info.layer_end if info else 0,
                    "has_embedding": info.has_embedding if info else False,
                    "has_lm_head": info.has_lm_head if info else False,
                    "node_type": info.node_type if info else "compute",
                })

        if vision_circuit:
            addrs, pks = vision_circuit
            vision_hop_addresses = list(addrs)
            all_nodes = discovery.discover("")
            node_map = {n.address: n for n in all_nodes}
            for i, addr in enumerate(addrs):
                info = node_map.get(addr)
                circuit_msg["vision_nodes"].append({
                    "address": addr,
                    "shard_index": i,
                    "node_type": info.node_type if info else "vision",
                })

        await websocket.send_json(circuit_msg)

        # Emit live hop events so UI can animate actual selected route.
        for i, addr in enumerate(vision_hop_addresses):
            await websocket.send_json({
                "type": "hop",
                "phase": "vision",
                "shard_index": i,
                "address": addr,
            })
            await asyncio.sleep(0)

        for i, addr in enumerate(text_hop_addresses):
            await websocket.send_json({
                "type": "hop",
                "phase": "text",
                "shard_index": i,
                "address": addr,
            })
            await asyncio.sleep(0)

        # Run generation
        gen_start = time.time()
        first_token_ts = None
        prev_token_ts = None
        token_intervals_ms: list[float] = []
        step = 0

        if model_type in ("qwen2_vl", "smolvlm") and image_path:
            await websocket.send_json({
                "type": "status",
                "message": f"Processing image through {model_type} vision pipeline..."
            })

            # Multimodal generation — dispatch to the correct pipeline
            if model_type == "smolvlm":
                generator = client.generate_multimodal_smolvlm(
                    prompt=prompt,
                    image_path=image_path,
                    max_new_tokens=effective_max_tokens,
                    verbose=False,
                    model_id=model_id,
                )
            else:
                generator = client.generate_multimodal(
                    prompt=prompt,
                    image_path=image_path,
                    max_new_tokens=effective_max_tokens,
                    verbose=False,
                    model_id=model_id,
                )
        else:
            # Text-only generation
            generator = client.generate(
                prompt=prompt,
                max_new_tokens=effective_max_tokens,
                verbose=False,
            )

        # Stream tokens
        for token_text in generator:
            now_ts = time.time()
            step_time = now_ts - gen_start
            if first_token_ts is None:
                first_token_ts = now_ts
            if prev_token_ts is not None:
                token_intervals_ms.append((now_ts - prev_token_ts) * 1000.0)
            prev_token_ts = now_ts
            await websocket.send_json({
                "type": "token",
                "text": token_text,
                "step": step,
                "step_time": round(step_time, 3),
            })
            step += 1
            # Yield control to event loop
            await asyncio.sleep(0)

        total_time = time.time() - gen_start
        tps = step / total_time if total_time > 0 else 0
        ttft_ms = ((first_token_ts - gen_start) * 1000.0) if first_token_ts else 0.0
        _record_perf_sample(
            model_id=model_id,
            output_tokens=step,
            total_time_s=total_time,
            ttft_ms=ttft_ms,
            token_intervals_ms=token_intervals_ms,
        )

        # Report token usage to registry for settlement accounting
        output_tokens = step
        if output_tokens < 0 or output_tokens > _MAX_BILLING_OUTPUT_TOKENS:
            raise RuntimeError(
                f"Invalid output token count: {output_tokens}. "
                f"Allowed range: 0..{_MAX_BILLING_OUTPUT_TOKENS}."
            )
        usage_cost = 0.0
        try:
            usage_cost = _report_usage(
                registry_addr=registry_addr,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_id=model_id,
                session_id="",
            )
        except Exception as e:
            print(f"[Web] ReportUsage failed: {e}")

        # Apply immediate per-request debit in on-chain escrow mode.
        try:
            _charge_client_escrow(client_address, usage_cost)
        except Exception as e:
            print(f"[Web] Escrow debit failed: {e}")

        # Report remaining balance after deduction
        _, remaining_balance = _check_client_balance(client_address)

        await websocket.send_json({
            "type": "done",
            "total_tokens": step,
            "output_tokens": output_tokens,
            "total_time": round(total_time, 2),
            "tokens_per_sec": round(tps, 1),
            "cost": round(usage_cost, 6),
            "input_tokens": input_tokens,
            "remaining_balance": round(remaining_balance, 4),
        })

    except Exception as e:
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })


def _count_request_input_tokens(client, model_type: str, model_id: str,
                                prompt: str, image_path: Optional[str]) -> int:
    """Strict exact token counting for billing. No heuristic fallback."""
    if model_type == "qwen2_vl" and image_path:
        return client.count_qwen2_vl_input_tokens(
            prompt=prompt,
            image_path=image_path,
            model_id=model_id,
        )
    if model_type == "smolvlm" and image_path:
        return client.count_smolvlm_input_tokens(
            prompt=prompt,
            image_path=image_path,
            model_id=model_id,
        )
    return client.count_text_input_tokens(prompt=prompt, model_id=model_id)


def _report_usage(registry_addr: str, input_tokens: int, output_tokens: int,
                  model_id: str, session_id: str = "") -> float:
    """Report usage to registry and return computed cost."""
    try:
        stub = _get_grpc_stub("registry", registry_addr)
        resp = stub.ReportUsage(registry_pb2.ReportUsageRequest(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            session_id=session_id,
        ), timeout=3)
        if not resp.accepted:
            raise RuntimeError("Usage report rejected by registry")
        return float(resp.cost)
    except Exception:
        _close_grpc_endpoint("registry", registry_addr)
        stub = _get_grpc_stub("registry", registry_addr)
        resp = stub.ReportUsage(registry_pb2.ReportUsageRequest(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_id=model_id,
            session_id=session_id,
        ), timeout=3)
        if not resp.accepted:
            raise RuntimeError("Usage report rejected by registry")
        return float(resp.cost)


def _charge_client_escrow(client_address: str, cost_tokens: float) -> None:
    """Apply immediate per-request debit to on-chain client escrow balance."""
    escrow = _get_escrow()
    if escrow is None:
        return
    if cost_tokens <= 0.0:
        return
    amount_wei = int(round(float(cost_tokens) * 1e18))
    if amount_wei <= 0:
        return
    escrow.charge_client(client_address, amount_wei)


# ---------------------------------------------------------------------------
# WebSocket: Chain updates
# ---------------------------------------------------------------------------
@app.websocket("/ws/chain")
async def ws_chain(websocket: WebSocket):
    """WebSocket for live chain updates — subscribes to daemon or polls."""
    await websocket.accept()
    chain = _get_chain()
    last_height = chain.height

    # Try to use daemon's SubscribeBlocks stream (preferred)
    daemon = _discover_daemon()
    if daemon:
        try:
            import grpc
            import inference_pb2
            import inference_pb2_grpc
            from economics.distributed_chain import proto_to_block

            channel = grpc.insecure_channel(
                daemon.address, options=app_config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)

            # Subscribe starting from current height
            stream = stub.SubscribeBlocks(
                inference_pb2.SubscribeBlocksRequest(from_height=last_height + 1)
            )

            async def _push_from_stream():
                """Read from gRPC stream in a thread and push to WebSocket."""
                nonlocal last_height
                import asyncio
                loop = asyncio.get_event_loop()

                def _iter_blocks():
                    for block_msg in stream:
                        yield block_msg

                for block_msg in await loop.run_in_executor(None, lambda: list(_iter_blocks())):
                    block = proto_to_block(block_msg)
                    chain.receive_external_block(block)
                    await websocket.send_json({
                        "type": "new_block",
                        "block": block.to_dict(),
                    })
                    last_height = max(last_height, block.index)

            # For now, fall through to polling if stream setup fails
            # (SubscribeBlocks is a long-lived stream, we need to handle it
            #  in a background thread — use polling for reliability)
            print(f"[Dashboard] Daemon found at {daemon.address}, "
                  f"using polling with daemon sync")
            channel.close()
        except Exception as e:
            logger.debug("Daemon stream setup failed, falling back to polling: %s", e)

    # Polling mode (works with or without daemon)
    try:
        while True:
            await asyncio.sleep(3)

            # Sync from daemon (or fallback to compute nodes)
            _sync_chain()

            current_height = chain.height
            if current_height > last_height:
                new_blocks = chain.get_blocks_from(last_height + 1)
                for block in new_blocks:
                    await websocket.send_json({
                        "type": "new_block",
                        "block": block.to_dict(),
                    })
                last_height = current_height

            # Send chain info periodically
            await websocket.send_json({
                "type": "chain_info",
                "info": chain.get_chain_info(),
            })

    except WebSocketDisconnect:
        logger.debug("Chain WebSocket disconnected")
    except Exception as e:
        logger.debug("Chain WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="UNFED AI Web Dashboard")
    parser.add_argument("--port", type=int, default=8080,
                        help="HTTP port (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--registry", type=str,
                        default=app_config.REGISTRY_ADDRESS,
                        help="Registry address (default: localhost:50050)")
    args = parser.parse_args()

    global _registry_address
    _registry_address = args.registry

    if _DEV_AUTH_BYPASS and not _is_loopback_host(args.host):
        raise RuntimeError(
            "Refusing startup: UNFED_DEV_AUTH_BYPASS=1 requires --host to be loopback "
            "(127.0.0.1/localhost/::1)."
        )
    _faucet_state_init()

    print(f"UNFED AI Dashboard starting on http://{args.host}:{args.port}")
    print(f"Registry: {_registry_address}")
    print(f"Static files: {STATIC_DIR}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
