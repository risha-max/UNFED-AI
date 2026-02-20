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
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import uuid

logger = logging.getLogger("unfed.web")
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import config as app_config
from network.discovery import RegistryClient, RegistryPool
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


def get_discovery() -> RegistryPool:
    global _discovery
    if _discovery is None:
        _discovery = RegistryPool([_registry_address])
    return _discovery


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
        result = []
        for m in models:
            result.append({
                "model_id": m.model_id,
                "total_nodes": m.total_nodes,
                "total_shards": m.total_shards,
                "is_healthy": m.is_healthy,
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
        all_nodes = discovery.discover("")
        nodes = []
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
        if health:
            return {
                "model_id": health.model_id,
                "total_nodes": health.total_nodes,
                "total_shards": health.total_shards,
                "is_healthy": health.is_healthy,
                "shard_coverage": list(health.shard_coverage),
            }
        return {"error": "No health data available"}
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


def _discover_daemon():
    """Find a daemon node via the registry."""
    try:
        discovery = get_discovery()
        all_nodes = discovery.discover("")
        daemons = [n for n in all_nodes if n.node_type == "daemon"]
        if daemons:
            return daemons[0]
    except Exception as e:
        logger.debug("Daemon discovery failed: %s", e)
    return None


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
            for block_msg in resp.blocks:
                block = proto_to_block(block_msg)
                accepted, reason = chain.receive_external_block(block)
                if accepted:
                    synced += 1

            if synced > 0:
                print(f"[Dashboard] Synced {synced} block(s) from daemon "
                      f"{daemon.address} "
                      f"(height: {my_height} -> {chain.get_tip_height()})")
            return synced
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
    import grpc as _grpc
    try:
        daemon = _discover_daemon()
        if not daemon:
            return {
                "base_fee": 0.001,
                "utilization": 0.0,
                "estimated_cost_100": 0.1,
                "suggested_tip": 0.0,
                "daemon_available": False,
                "fee_history": [],
            }

        channel = _grpc.insecure_channel(daemon.address, options=config.GRPC_OPTIONS)
        stub = inference_pb2_grpc.InferenceNodeStub(channel)

        resp = stub.GetFeeEstimate(
            inference_pb2.FeeEstimateRequest(estimated_tokens=100),
            timeout=5,
        )
        channel.close()

        return {
            "base_fee": resp.base_fee,
            "utilization": resp.utilization,
            "estimated_cost_100": resp.estimated_cost,
            "suggested_tip": resp.suggested_tip,
            "daemon_available": True,
        }
    except Exception as e:
        return {
            "base_fee": 0.001,
            "utilization": 0.0,
            "estimated_cost_100": 0.1,
            "suggested_tip": 0.0,
            "daemon_available": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
@app.get("/api/pricing")
async def get_pricing(model_id: str = ""):
    """Get per-token pricing from the registry."""
    import grpc as _grpc
    import registry_pb2 as _rpb
    import registry_pb2_grpc as _rgrpc
    try:
        channel = _grpc.insecure_channel(
            _registry_address, options=app_config.GRPC_OPTIONS)
        stub = _rgrpc.RegistryStub(channel)
        resp = stub.GetPricing(
            _rpb.GetPricingRequest(model_id=model_id),
            timeout=5,
        )
        channel.close()
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
        return True, 0.0  # fail-open if chain is unreachable


@app.get("/api/client/balance")
async def get_client_balance(address: str = ""):
    """Get a client's escrow balance."""
    if not address:
        address = _wallet_auth.default_address
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


@app.post("/api/client/auth")
async def client_auth():
    """Get a challenge for wallet signature authentication."""
    challenge = _wallet_auth.generate_challenge()
    return {"challenge": challenge}


# ---------------------------------------------------------------------------
# Faucet (testnet token distribution)
# ---------------------------------------------------------------------------
_faucet_last_drip: dict[str, float] = {}


class FaucetRequest(BaseModel):
    address: str

@app.post("/api/faucet")
async def faucet_drip(body: FaucetRequest):
    """Drip test tokens into a client's escrow balance."""
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

    now = time.time()
    cooldown = escrow.FAUCET_COOLDOWN
    last = _faucet_last_drip.get(address, 0)
    remaining = cooldown - (now - last)
    if remaining > 0:
        return JSONResponse(status_code=429, content={
            "error": "Cooldown active.",
            "retry_after_seconds": int(remaining),
        })

    try:
        tx_hash = escrow.faucet_drip(address)
        _faucet_last_drip[address] = time.time()
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
        wallet: Ethereum address for escrow billing (optional).
                If omitted, falls back to demo default address.

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

    # Resolve client wallet from query param or fall back to demo default
    wallet_param = websocket.query_params.get("wallet", "")
    client_address = _wallet_auth.authenticate(wallet_address=wallet_param)

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
            max_tokens = data.get("max_tokens", 100)
            use_voting = data.get("use_voting", False)
            model_id = data.get("model_id", "")
            cluster_endpoint = data.get("cluster_endpoint", "")

            await _run_generation(
                websocket, prompt, image_path, model_type,
                max_tokens, use_voting, model_id,
                cluster_endpoint=cluster_endpoint,
                client_address=client_address,
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
    max_tokens: int,
    use_voting: bool,
    model_id: str,
    cluster_endpoint: str = "",
    client_address: str = "",
):
    """Run generation and stream results to the WebSocket."""
    from client.client import UnfedClient

    await websocket.send_json({
        "type": "status",
        "message": f"Initializing client ({model_type})..."
    })

    # Fall back to demo address if none provided (shouldn't happen — ws_chat resolves it)
    if not client_address:
        client_address = _wallet_auth.default_address

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

        # If a specific cluster was selected, use its endpoint
        registry_addr = _registry_address
        if cluster_endpoint:
            registry_addr = cluster_endpoint

        # Determine model ID — try HF repo name first (v2 manifests use it),
        # then fall back to local path for v1 compatibility.
        if not model_id:
            discovery = (RegistryPool([cluster_endpoint])
                         if cluster_endpoint else get_discovery())
            if model_type == "smolvlm":
                candidates = [
                    "HuggingFaceTB/SmolVLM-256M-Instruct",
                    "/home/oasis/models/SmolVLM-256M-Instruct",
                ]
            elif model_type == "qwen2_vl":
                candidates = [
                    "Qwen/Qwen2-VL-2B-Instruct",
                    "/home/oasis/models/Qwen2-VL-2B-Instruct",
                ]
            else:
                candidates = [app_config.MODEL_NAME]

            # Pick the first model_id that has registered nodes
            model_id = candidates[0]
            for cand in candidates:
                nodes = discovery.discover(cand)
                if nodes:
                    model_id = cand
                    break

        # Create client (use cluster-specific endpoint if selected)
        client = UnfedClient(
            registry_address=registry_addr,
            use_voting=use_voting,
            model_id=model_id,
        )

        # Discover and send circuit info
        discovery = get_discovery()

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

        # Send circuit info to frontend
        circuit_msg = {
            "type": "circuit",
            "text_nodes": [],
            "vision_nodes": [],
        }

        # Check for MPC nodes — use them for shard 0 if available
        mpc_nodes = discovery.discover_mpc(model_id)
        using_mpc = len(mpc_nodes) > 0

        if text_circuit:
            addrs, pks = text_circuit
            # Get full node info for each address
            all_nodes = discovery.discover("")
            node_map = {n.address: n for n in all_nodes}

            # If MPC available, swap shard 0 with MPC entry
            if using_mpc:
                mpc_entry = mpc_nodes[0]
                addrs[0] = mpc_entry.address
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

        # Run generation
        gen_start = time.time()
        step = 0

        if model_type in ("qwen2_vl", "smolvlm") and image_path:
            # Default prompt when image sent without text
            if not prompt.strip():
                prompt = "Describe this image."

            await websocket.send_json({
                "type": "status",
                "message": f"Processing image through {model_type} vision pipeline..."
            })

            # Multimodal generation — dispatch to the correct pipeline
            if model_type == "smolvlm":
                generator = client.generate_multimodal_smolvlm(
                    prompt=prompt,
                    image_path=image_path,
                    max_new_tokens=max_tokens,
                    verbose=False,
                    model_id=model_id,
                )
            else:
                generator = client.generate_multimodal(
                    prompt=prompt,
                    image_path=image_path,
                    max_new_tokens=max_tokens,
                    verbose=False,
                    model_id=model_id,
                )
        else:
            # Text-only generation
            generator = client.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                verbose=False,
            )

        # Stream tokens
        for token_text in generator:
            step_time = time.time() - gen_start
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

        # Report token usage to registry for settlement accounting
        output_tokens = step
        # Estimate input tokens from prompt length (rough: 1 token ≈ 4 chars)
        input_tokens = max(1, len(prompt) // 4)
        usage_cost = 0.0
        try:
            import grpc as _grpc
            import registry_pb2 as _rpb
            import registry_pb2_grpc as _rgrpc
            _ch = _grpc.insecure_channel(registry_addr)
            _stub = _rgrpc.RegistryStub(_ch)
            resp = _stub.ReportUsage(_rpb.ReportUsageRequest(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_id=model_id,
            ), timeout=3)
            usage_cost = resp.cost
            _ch.close()
        except Exception as e:
            print(f"[Web] ReportUsage failed: {e}")

        # Report remaining balance after deduction
        _, remaining_balance = _check_client_balance(client_address)

        await websocket.send_json({
            "type": "done",
            "total_tokens": step,
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
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--registry", type=str,
                        default=app_config.REGISTRY_ADDRESS,
                        help="Registry address (default: localhost:50050)")
    args = parser.parse_args()

    global _registry_address
    _registry_address = args.registry

    print(f"UNFED AI Dashboard starting on http://{args.host}:{args.port}")
    print(f"Registry: {_registry_address}")
    print(f"Static files: {STATIC_DIR}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
