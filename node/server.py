"""
Node Server — gRPC server that loads a model shard and serves Forward() RPCs.

Each node:
  - Loads its weight shard on startup
  - Registers with the registry (for peer discovery)
  - Listens on its gRPC port
  - On Forward(): runs its layers, caches KV, forwards to next node in the circuit
  - Manages KV cache per session_id
  - Sends periodic heartbeats to the registry
  - Unregisters on shutdown

Usage:
    python -m node.server --shard-index 0 --port 50051
    python -m node.server --shard-index 1 --port 50052
    ...
"""

import argparse
import hashlib
import io
import json
import os
import random
import signal
import sys
import time
import threading
from concurrent import futures

import grpc
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Fix import path for generated protobuf code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import inference_pb2
import inference_pb2_grpc
import registry_pb2
import registry_pb2_grpc

from network.discovery import NodeRegistration
from network.resilience import create_resilient_channel, with_retry
from network.onion import peel_onion, encrypt_response
from network.randomness import compute_activation_hash, select_next_node, SessionCircuit
from network.verification import TicketCollector
from network.zk_verification import create_commitment
from economics.distributed_chain import (
    share_to_proto, block_to_proto, proto_to_block, proto_to_share,
)
from economics.share_chain import ComputeShare


def tensor_to_bytes(tensor: torch.Tensor,
                    compress: bool = None,
                    wire_dtype: str = None) -> tuple[bytes, list[int], bool]:
    """Serialize a tensor to bytes + shape for gRPC transfer.

    Args:
        tensor: Input tensor.
        compress: Override compression (None = use config default).
        wire_dtype: "float16" to halve transfer size, "float32" or None for full precision.

    Returns:
        (data_bytes, shape, is_compressed)
    """
    import gzip as _gzip

    target_dtype = torch.float16 if wire_dtype == "float16" else torch.float32

    if tensor.dtype == target_dtype and tensor.is_contiguous():
        t = tensor
    else:
        t = tensor.contiguous().to(target_dtype)

    raw = t.numpy().tobytes()
    shape = list(t.shape)

    should_compress = compress if compress is not None else config.COMPRESS_ACTIVATIONS
    threshold = config.COMPRESS_THRESHOLD

    if should_compress and len(raw) > threshold:
        compressed = _gzip.compress(raw, compresslevel=1)
        if len(compressed) < len(raw) * 0.95:
            return compressed, shape, True

    return raw, shape, False


def bytes_to_tensor(data: bytes, shape: list[int],
                    compressed: bool = False,
                    wire_dtype: str = None) -> torch.Tensor:
    """Deserialize bytes + shape back to a tensor.

    Args:
        data: Raw or gzip-compressed bytes.
        shape: Tensor shape dimensions.
        compressed: If True, data is gzip-compressed.
        wire_dtype: "float16" if data was serialized in half precision.
    """
    import numpy as np
    import gzip as _gzip

    if compressed:
        data = _gzip.decompress(data)

    np_dtype = np.float16 if wire_dtype == "float16" else np.float32
    arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    t = torch.from_numpy(arr.copy())
    if wire_dtype == "float16":
        t = t.float()
    return t


class InferenceNodeServicer(inference_pb2_grpc.InferenceNodeServicer):
    """gRPC service implementation for a single inference node."""

    def __init__(self, shard_index: int, port: int, registration: NodeRegistration = None,
                 node_config=None, model_type: str = "qwen2",
                 shards_dir: str = None, cheat_rate: float = 0.0,
                 sampling_rate: float = 0.05):
        self.shard_index = shard_index
        self.port = port
        self.registration = registration  # holds private key for onion routing
        self._node_config = node_config   # NodeConfig (optional, for tunable values)
        self.model_type = model_type
        self._cheat_rate = cheat_rate     # fraction of requests to corrupt (0.0 = honest)
        self._default_sampling_rate = sampling_rate
        self._shards_dir = shards_dir or config.SHARDS_DIR

        # Resolve configurable values (config > global defaults)
        self._manifest_path = (node_config.manifest_path
                               if node_config else config.MANIFEST_PATH)
        # Override manifest path for VL models
        if model_type == "qwen2_vl":
            self._manifest_path = os.path.join(
                shards_dir or "shards_vl", "manifest.json")
        elif model_type == "smolvlm":
            self._manifest_path = os.path.join(
                shards_dir or "shards_smolvlm", "manifest.json")
        # EOS token: read from config or detect from model
        if node_config and hasattr(node_config, 'eos_token_id') and node_config.eos_token_id:
            self._eos_token_id = node_config.eos_token_id
        elif model_type == "smolvlm":
            self._eos_token_id = 49279  # <end_of_utterance>
        else:
            self._eos_token_id = 151643  # Qwen default
        sampling_rate = (node_config.verification_sampling_rate
                         if node_config else self._default_sampling_rate)

        # --- P2P bandwidth controls ---
        self._serve_shards_enabled = (node_config.serve_shards_enabled
                                      if node_config else True)
        self._max_upload_rate_mbps = (node_config.max_upload_rate_mbps
                                      if node_config else 0.0)
        max_transfers = (node_config.max_concurrent_transfers
                         if node_config else 0)
        self._inference_priority = (node_config.inference_priority
                                    if node_config else True)

        # Semaphore to limit concurrent shard transfers (0 = unlimited)
        self._transfer_semaphore = (threading.Semaphore(max_transfers)
                                    if max_transfers > 0 else None)
        # Counter for active inference requests (used by inference_priority)
        self._active_inferences = 0
        self._inference_lock = threading.Lock()

        # Load shard metadata from manifest
        manifest = self._load_manifest()

        # VL models have separate vision_shards and text_shards in manifest
        if model_type in ("qwen2_vl", "smolvlm"):
            shard_info = manifest["text_shards"][shard_index]
            self.num_shards = len(manifest["text_shards"])
        else:
            shard_info = manifest["shards"][shard_index]
            self.num_shards = len(manifest["shards"])
        self.layer_start = shard_info["layer_start"]
        self.layer_end = shard_info["layer_end"]
        self.has_embedding = shard_info.get("has_embedding", False)
        self.has_lm_head = shard_info.get("has_lm_head", False) or shard_info.get("has_head", False)

        # Shard weights hash (for ZK commitments)
        self.shard_weights_hash = shard_info.get("sha256", "")

        # --- KV cache config ---
        kv_quantize = node_config.kv_quantize if node_config else "none"
        kv_offload_enabled = node_config.kv_offload_enabled if node_config else False
        kv_offload_after_seconds = (node_config.kv_offload_after_seconds
                                    if node_config else 30.0)
        max_kv_memory_gb = node_config.max_kv_memory_gb if node_config else 0.0
        prefill_chunk_size = node_config.prefill_chunk_size if node_config else 0

        # Load the shard and initialize the layer runner (generic runtime).
        print(f"[Node] Shard {shard_index}: layers {self.layer_start}-{self.layer_end - 1} "
              f"(model_type={model_type})")

        from node.runtime.generic_runner import GenericTextRunner

        arch_config = manifest.get("architecture", {}).get("text")
        if not arch_config:
            raise ValueError(
                f"Manifest has no 'architecture.text' config — "
                f"only v2+ manifests with generic runtime are supported. "
                f"Re-split the model with the latest splitter."
            )

        shard_file = shard_info.get("file", f"text_shard_{shard_index}.pt")
        shard_path = os.path.join(self._shards_dir, shard_file)
        image_token_id = manifest.get("image_token_id")

        device_str = node_config.device if node_config else "cpu"
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)

        self.runner = GenericTextRunner(
            config=arch_config,
            shard_info=shard_info,
            shard_path=shard_path,
            image_token_id=image_token_id,
            device=self._device,
        )
        print(f"[Node] Using generic runtime (format_version="
              f"{manifest.get('format_version', '?')}) on {self._device}")

        # Cache of gRPC stubs to other nodes (lazily created)
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}

        # Computational randomness routing: session -> circuit cache
        self._session_circuits: dict[str, SessionCircuit] = {}
        self._discovery = None  # lazy-initialized RegistryClient

        # Verification: sample forward passes (prefill only)
        self._ticket_collector = TicketCollector(sampling_rate=sampling_rate)
        self._seen_sessions: set[str] = set()  # track sessions to identify prefill

        # Daemon stub for submitting shares (replaces embedded chain)
        self._daemon_stub = None
        self._daemon_address: str = ""
        self._share_buffer: list[ComputeShare] = []
        self._share_buffer_lock = threading.Lock()

    def init_daemon_connection(self, node_id: str, registry_address: str = None,
                               self_address: str = ""):
        """Discover and connect to the chain daemon for share submission.

        Compute nodes no longer manage their own chain — they submit
        shares to the daemon (like miners submitting work to monerod).
        """
        self._node_id_for_shares = node_id
        registry_addr = registry_address or config.REGISTRY_ADDRESS

        # Discover daemon node from registry
        try:
            channel = grpc.insecure_channel(registry_addr, options=config.GRPC_OPTIONS)
            stub = registry_pb2_grpc.RegistryStub(channel)
            resp = stub.Discover(
                registry_pb2.DiscoverRequest(model_id=""),
                timeout=10,
            )
            daemons = [n for n in resp.nodes if n.node_type == "daemon"]
            channel.close()

            if daemons:
                daemon = daemons[0]
                self._daemon_address = daemon.address
                self._daemon_stub = inference_pb2_grpc.InferenceNodeStub(
                    grpc.insecure_channel(daemon.address, options=config.GRPC_OPTIONS)
                )
                print(f"[Node] Connected to chain daemon at {daemon.address}")
            else:
                print(f"[Node] No daemon found — shares will be buffered locally")
                # Start background thread to retry daemon discovery
                t = threading.Thread(target=self._daemon_discovery_loop,
                                     args=(registry_addr,), daemon=True)
                t.start()
        except grpc.RpcError:
            print(f"[Node] Registry unreachable — shares will be buffered locally")

    def _daemon_discovery_loop(self, registry_addr: str):
        """Background loop to discover daemon if not found on startup."""
        while self._daemon_stub is None:
            time.sleep(10)
            try:
                channel = grpc.insecure_channel(registry_addr, options=config.GRPC_OPTIONS)
                stub = registry_pb2_grpc.RegistryStub(channel)
                resp = stub.Discover(
                    registry_pb2.DiscoverRequest(model_id=""),
                    timeout=10,
                )
                daemons = [n for n in resp.nodes if n.node_type == "daemon"]
                channel.close()
                if daemons:
                    daemon = daemons[0]
                    self._daemon_address = daemon.address
                    self._daemon_stub = inference_pb2_grpc.InferenceNodeStub(
                        grpc.insecure_channel(daemon.address, options=config.GRPC_OPTIONS)
                    )
                    print(f"[Node] Connected to chain daemon at {daemon.address}")
                    # Flush buffered shares
                    self._flush_share_buffer()
            except grpc.RpcError:
                pass

    def _submit_share_to_daemon(self, share: ComputeShare):
        """Submit a compute share to the chain daemon.

        If the daemon is unreachable, buffer locally and retry later.
        """
        if self._daemon_stub is None:
            with self._share_buffer_lock:
                self._share_buffer.append(share)
            return

        try:
            self._daemon_stub.SubmitShares(
                inference_pb2.SubmitSharesRequest(
                    shares=[share_to_proto(share)],
                    submitter_id=share.node_id,
                ),
                timeout=5,
            )
        except grpc.RpcError:
            # Daemon unreachable — buffer locally
            with self._share_buffer_lock:
                self._share_buffer.append(share)

    def _flush_share_buffer(self):
        """Send any buffered shares to the daemon."""
        with self._share_buffer_lock:
            if not self._share_buffer:
                return
            shares = self._share_buffer
            self._share_buffer = []

        if self._daemon_stub is None:
            # Put them back
            with self._share_buffer_lock:
                self._share_buffer.extend(shares)
            return

        try:
            protos = [share_to_proto(s) for s in shares]
            self._daemon_stub.SubmitShares(
                inference_pb2.SubmitSharesRequest(
                    shares=protos,
                    submitter_id=shares[0].node_id if shares else "",
                ),
                timeout=10,
            )
            print(f"[Node] Flushed {len(shares)} buffered shares to daemon")
        except grpc.RpcError:
            with self._share_buffer_lock:
                self._share_buffer.extend(shares)

    def _load_manifest(self) -> dict:
        """Load the model manifest."""
        with open(self._manifest_path, "r") as f:
            return json.load(f)

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        """Get or create a gRPC stub for the given address (resilient channel)."""
        if address not in self._stubs:
            channel = create_resilient_channel(address, config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def _get_discovery(self):
        """Lazy-initialize a registry client for random routing."""
        if self._discovery is None:
            from network.discovery import RegistryClient
            registry_addr = self.registration._client.registry_address if self.registration else None
            self._discovery = RegistryClient(registry_addr)
        return self._discovery

    def _select_next_node_random(self, session_id: str, activation: torch.Tensor,
                                 next_shard_index: int) -> str | None:
        """
        Select the next node using computational randomness routing.
        Caches the decision per session for KV cache stability.
        """
        # Check if we already have a route for this session
        if session_id not in self._session_circuits:
            self._session_circuits[session_id] = SessionCircuit()
        circuit = self._session_circuits[session_id]

        if circuit.has_route(next_shard_index):
            return circuit.get_route(next_shard_index)

        # First token: compute hash and select
        discovery = self._get_discovery()
        nodes = discovery.discover(self.registration.model_id if self.registration else "")
        candidates = [n for n in nodes if n.shard_index == next_shard_index]

        if not candidates:
            return None

        act_hash = compute_activation_hash(activation)
        chosen_idx = select_next_node(act_hash, len(candidates))
        chosen = candidates[chosen_idx]

        circuit.set_route(next_shard_index, chosen.address)
        print(f"[Node shard={self.shard_index}] Random routing: "
              f"session={session_id[:8]}... → shard {next_shard_index} "
              f"→ {chosen.address} (hash={act_hash[:8]}...)")

        return chosen.address

    def Forward(self, request, context):
        """Handle a forward pass request."""
        session_id = request.session_id

        # Track active inferences for bandwidth priority
        with self._inference_lock:
            self._active_inferences += 1
        try:
            # --- Run layers ---
            input_for_ticket = None
            input_is_tokens = False

            # Extract prefix caching fields
            prefix_session_id = request.prefix_session_id or ""
            prefix_length = request.prefix_length or 0

            # --- Deserialize image embeddings (VL text pipeline) ---
            image_embeddings = None
            if (request.image_embeddings and request.image_embeddings_shape
                    and self.model_type in ("qwen2_vl", "smolvlm")
                    and self.has_embedding):
                import numpy as np
                ie_shape = list(request.image_embeddings_shape)
                image_embeddings = torch.from_numpy(
                    np.frombuffer(request.image_embeddings, dtype=np.float32)
                    .copy().reshape(ie_shape)
                )

            # --- Deserialize M-RoPE position IDs (VL text pipeline) ---
            mrope_position_ids = None
            if (request.mrope_position_ids and request.mrope_position_shape
                    and self.model_type == "qwen2_vl"):
                import numpy as np
                mp_shape = list(request.mrope_position_shape)
                mrope_position_ids = torch.from_numpy(
                    np.frombuffer(request.mrope_position_ids, dtype=np.int64)
                    .copy().reshape(mp_shape)
                )

            wire_dtype = request.wire_dtype or config.WIRE_DTYPE

            if self.has_embedding and not request.activation_data:
                token_ids = torch.tensor([list(request.token_ids)])
                input_for_ticket = token_ids
                input_is_tokens = True
                hidden, sampled_token = self.runner.forward(
                    token_ids=token_ids,
                    session_id=session_id,
                    prefix_session_id=prefix_session_id,
                    prefix_length=prefix_length,
                    image_embeddings=image_embeddings,
                    mrope_position_ids=mrope_position_ids,
                )
            else:
                hidden_states = bytes_to_tensor(
                    request.activation_data,
                    list(request.tensor_shape),
                    compressed=request.compressed,
                    wire_dtype=wire_dtype,
                )
                input_for_ticket = hidden_states
                hidden, sampled_token = self.runner.forward(
                    hidden_states=hidden_states,
                    session_id=session_id,
                    prefix_session_id=prefix_session_id,
                    prefix_length=prefix_length,
                    mrope_position_ids=mrope_position_ids,
                )

            # --- Cheat mode: corrupt output BEFORE ticket collection ---
            # The ticket will contain the corrupted output, so the verifier
            # will detect the mismatch when it re-computes the correct result.
            if self._cheat_rate > 0 and random.random() < self._cheat_rate:
                noise = torch.randn_like(hidden) * hidden.abs().mean() * 0.5
                hidden = hidden + noise
                if not hasattr(self, '_cheat_count'):
                    self._cheat_count = 0
                self._cheat_count += 1
                print(f"[CHEAT] Shard {self.shard_index}: corrupted output "
                      f"(cheat #{self._cheat_count})")

            # --- Verification sampling (prefill only) ---
            is_prefill = session_id not in self._seen_sessions
            self._seen_sessions.add(session_id)
            if is_prefill and self._ticket_collector.should_sample():
                # Generate ZK commitment for this computation
                zk_commit = create_commitment(
                    shard_index=self.shard_index,
                    input_tensor=input_for_ticket,
                    output_tensor=hidden,
                    shard_weights_hash=self.shard_weights_hash,
                )
                # Collect full ticket (for spot-check verification)
                self._ticket_collector.collect(
                    shard_index=self.shard_index,
                    input_tensor=input_for_ticket,
                    output_tensor=hidden,
                    input_is_tokens=input_is_tokens,
                    sampled_token=sampled_token,
                    is_prefill=True,
                )

            # --- Record compute share on the distributed chain ---
            if self.registration:
                share = ComputeShare(
                    node_id=self.registration.node_id,
                    shard_index=self.shard_index,
                    session_id=session_id,
                    activation_hash="",
                    tokens_processed=1,
                    share_weight=1.0,
                )
                self._submit_share_to_daemon(share)

            # --- Determine next hop ---
            next_address = None
            next_onion_blob = b""
            next_ephemeral_key = b""
            use_random_routing = request.use_random_routing
            activation_commitment = ""

            if self.has_lm_head:
                # We are the last shard — no next hop needed
                next_address = None
            elif request.onion_blob and self.registration:
                # Onion routing mode: peel one layer
                layer, next_eph = peel_onion(
                    self.registration.private_key,
                    bytes(request.onion_ephemeral_key),
                    bytes(request.onion_blob),
                )
                next_address = layer.next_hop or None
                next_onion_blob = layer.payload
                next_ephemeral_key = next_eph
            elif use_random_routing and self.registration:
                # Computational randomness routing
                next_shard = self.shard_index + 1
                activation_commitment = compute_activation_hash(hidden)
                next_address = self._select_next_node_random(
                    session_id, hidden, next_shard
                )
            elif request.remaining_circuit:
                # Plain circuit mode (fallback)
                remaining = list(request.remaining_circuit)
                next_address = remaining.pop(0) if remaining else None

            # --- Return-path encryption ---
            # Pop this node's response key (index 0) from the list.
            # Remaining keys are passed to the next node.
            my_response_key = None
            remaining_response_keys = []
            if request.response_keys:
                all_keys = list(request.response_keys)
                my_response_key = bytes(all_keys[0])
                remaining_response_keys = all_keys[1:]

            # --- Route ---
            if next_address is None and self.has_lm_head:
                # Last node: return the sampled token
                is_eos = sampled_token == self._eos_token_id

                resp = inference_pb2.ForwardResponse(
                    token_id=sampled_token,
                    has_token=True,
                    is_eos=is_eos,
                    activation_commitment=activation_commitment,
                )

                # Encrypt the response for return-path privacy
                if my_response_key:
                    token_bytes = sampled_token.to_bytes(4, 'big', signed=True)
                    is_eos_byte = b'\x01' if is_eos else b'\x00'
                    plaintext = token_bytes + is_eos_byte
                    resp.encrypted_response = encrypt_response(
                        my_response_key, plaintext)

                return resp

            elif next_address is None:
                # Intermediate node, direct mode (used by per-shard racing):
                # Return activation tensor to the caller instead of forwarding.
                activation_bytes, shape, is_compressed = tensor_to_bytes(
                    hidden.cpu(), wire_dtype=wire_dtype)

                resp = inference_pb2.ForwardResponse(
                    activation_data=activation_bytes,
                    tensor_shape=shape,
                    activation_commitment=activation_commitment,
                    compressed=is_compressed,
                )

                # Encrypt activation if return-path key is present
                if my_response_key:
                    resp.encrypted_response = encrypt_response(
                        my_response_key, activation_bytes)

                return resp

            else:
                # Forward to next node
                activation_bytes, shape, is_compressed = tensor_to_bytes(
                    hidden.cpu(), wire_dtype=wire_dtype)

                next_request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    activation_data=activation_bytes,
                    tensor_shape=shape,
                    prefix_session_id=prefix_session_id,
                    prefix_length=prefix_length,
                    compressed=is_compressed,
                    wire_dtype=wire_dtype or "",
                )

                # Forward VL fields (image_embeddings, mrope, etc.)
                if self.model_type in ("qwen2_vl", "smolvlm"):
                    # Forward image embeddings (only to next shard if it
                    # needs them, i.e. the shard after the embedding shard)
                    if request.image_embeddings and not self.has_embedding:
                        # Pass through — this shard didn't consume them
                        next_request.image_embeddings = request.image_embeddings
                        next_request.image_embeddings_shape.extend(
                            list(request.image_embeddings_shape))
                    # Forward M-RoPE position IDs to all shards (Qwen2-VL only)
                    if request.mrope_position_ids:
                        next_request.mrope_position_ids = request.mrope_position_ids
                        next_request.mrope_position_shape.extend(
                            list(request.mrope_position_shape))
                    if request.image_grid_thw:
                        next_request.image_grid_thw.extend(
                            list(request.image_grid_thw))
                    next_request.num_image_tokens = request.num_image_tokens

                # Attach routing info
                if next_onion_blob:
                    next_request.onion_blob = next_onion_blob
                    next_request.onion_ephemeral_key = next_ephemeral_key
                elif use_random_routing:
                    next_request.use_random_routing = True
                elif request.remaining_circuit:
                    remaining = list(request.remaining_circuit)
                    remaining.pop(0)
                    next_request.remaining_circuit.extend(remaining)

                # Pass remaining response keys to the next node
                if remaining_response_keys:
                    next_request.response_keys.extend(remaining_response_keys)

                stub = self._get_stub(next_address)
                response = stub.Forward(next_request)

                # Encrypt the response before returning it upstream
                if my_response_key and response.encrypted_response:
                    encrypted = encrypt_response(
                        my_response_key, response.encrypted_response)
                    return inference_pb2.ForwardResponse(
                        encrypted_response=encrypted,
                        token_id=response.token_id,
                        has_token=response.has_token,
                        is_eos=response.is_eos,
                        activation_commitment=response.activation_commitment,
                    )

                return response

        except Exception as e:
            print(f"[Node shard={self.shard_index}] Error in Forward: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.ForwardResponse()
        finally:
            with self._inference_lock:
                self._active_inferences -= 1

    def Commit(self, request, context):
        """
        Commit RPC: compute forward pass and return only the output hash.

        Used for redundant voting — the client sends the same input to 2 nodes
        serving the same shard. Each node commits to hash(output) without
        revealing the actual output. The client compares the two hashes.

        The node caches the output in case the client asks it to proceed
        with a full Forward call.
        """
        session_id = request.session_id
        try:
            if self.has_embedding and request.token_ids:
                token_ids = torch.tensor([list(request.token_ids)])
                hidden, sampled_token = self.runner.forward(
                    token_ids=token_ids,
                    session_id=f"commit-{session_id}",
                )
            else:
                hidden_states = bytes_to_tensor(
                    request.activation_data,
                    list(request.tensor_shape),
                    compressed=request.compressed,
                )
                hidden, sampled_token = self.runner.forward(
                    hidden_states=hidden_states,
                    session_id=f"commit-{session_id}",
                )

            output_bytes, _, _ = tensor_to_bytes(hidden.cpu(), compress=False)
            import hashlib
            output_hash = hashlib.sha256(output_bytes).hexdigest()
            if sampled_token is not None:
                # Include the token in the hash for last-shard nodes
                token_bytes = sampled_token.to_bytes(4, 'big', signed=True)
                output_hash = hashlib.sha256(
                    output_bytes + token_bytes).hexdigest()

            # Cache the output for potential follow-up
            commit_id = f"commit-{session_id}"
            # (The KV cache is already stored under commit_id from the forward call)

            return inference_pb2.CommitResponse(
                output_hash=output_hash,
                commit_id=commit_id,
                token_id=sampled_token if sampled_token is not None else 0,
                has_token=sampled_token is not None,
            )

        except Exception as e:
            print(f"[Node shard={self.shard_index}] Error in Commit: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.CommitResponse()

    # --- Share-Chain Gossip RPCs ---

    def GossipBlock(self, request, context):
        """Compute nodes no longer hold the chain — redirect to daemon."""
        return inference_pb2.GossipResponse(
            accepted=False,
            reason="use_daemon",
            current_height=0,
        )

    def GetBlocks(self, request, context):
        """Compute nodes no longer hold the chain — redirect to daemon."""
        return inference_pb2.GetBlocksResponse(blocks=[], chain_height=0)

    def GetShard(self, request, context):
        """Stream a shard file to a requesting node.

        Bandwidth controls (configured via NodeConfig):
        - serve_shards_enabled: opt out entirely (refuse all transfers)
        - max_concurrent_transfers: limit parallel uploads via semaphore
        - inference_priority: pause transfer while inference is active
        - max_upload_rate_mbps: throttle upload speed

        Supports range serving: if request.offset and/or request.length are set,
        only the requested byte range is streamed. Each streamed chunk includes
        its own SHA256 hash for piece-level verification.

        If both offset and length are 0 (default), the full file is streamed
        (backward compatible with old clients).
        """
        # --- Bandwidth control: opt-out ---
        if not self._serve_shards_enabled:
            context.set_details("This node has shard serving disabled")
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return

        # --- Bandwidth control: concurrent transfer limit ---
        if self._transfer_semaphore is not None:
            acquired = self._transfer_semaphore.acquire(blocking=False)
            if not acquired:
                context.set_details("Too many concurrent transfers")
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                return

        try:
            yield from self._get_shard_impl(request, context)
        finally:
            if self._transfer_semaphore is not None:
                self._transfer_semaphore.release()

    def _get_shard_impl(self, request, context):
        """Internal implementation of GetShard (after bandwidth gate checks)."""
        # Resolve shard path from this node's shards directory + manifest.
        # Use the model_type-appropriate shard list to avoid collisions
        # between text and vision shards that share the same shard_index.
        manifest = self._load_manifest()
        if self.model_type in ("qwen2_vl", "smolvlm"):
            shards_list = manifest.get("text_shards", [])
        else:
            shards_list = manifest.get("shards", [])
        # Also check the combined list as fallback
        if not shards_list:
            shards_list = (manifest.get("text_shards", [])
                           + manifest.get("vision_shards", []))
        shard_file = None
        for s in shards_list:
            if s.get("shard_index") == request.shard_index:
                shard_file = s.get("file")
                break
        if shard_file:
            shard_path = os.path.join(self._shards_dir, shard_file)
        else:
            shard_path = config.get_shard_path(request.shard_index)

        if not os.path.exists(shard_path):
            context.set_details(f"Shard {request.shard_index} not found")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return

        total_size = os.path.getsize(shard_path)

        # Determine byte range to serve
        start_offset = request.offset if request.offset > 0 else 0
        serve_length = request.length if request.length > 0 else (total_size - start_offset)

        # Clamp to file bounds
        if start_offset >= total_size:
            context.set_details(f"Offset {start_offset} beyond file size {total_size}")
            context.set_code(grpc.StatusCode.OUT_OF_RANGE)
            return
        serve_length = min(serve_length, total_size - start_offset)

        is_full_file = (start_offset == 0 and serve_length == total_size)

        # Compute full-file hash only for full-file requests (avoid expensive I/O for ranges)
        shard_hash = ""
        if is_full_file:
            sha256 = hashlib.sha256()
            with open(shard_path, "rb") as f:
                while True:
                    block = f.read(config.SHARD_CHUNK_SIZE)
                    if not block:
                        break
                    sha256.update(block)
            shard_hash = sha256.hexdigest()

        if is_full_file:
            print(f"[Node] Serving shard {request.shard_index} "
                  f"({total_size / 1024 / 1024:.1f} MB) to peer")
        else:
            print(f"[Node] Serving shard {request.shard_index} "
                  f"range [{start_offset}..{start_offset + serve_length}) "
                  f"({serve_length / 1024 / 1024:.1f} MB) to peer")

        # --- Rate-limit calculation ---
        # Convert max_upload_rate_mbps to bytes-per-second for throttling.
        # 0 = unlimited (no sleep between chunks).
        bytes_per_second = 0.0
        if self._max_upload_rate_mbps > 0:
            bytes_per_second = self._max_upload_rate_mbps * 1024 * 1024 / 8

        # Stream the requested range in gRPC-sized chunks, computing a hash
        # over the entire served range for chunk-level verification.
        range_hasher = hashlib.sha256()
        offset = start_offset
        remaining = serve_length
        first_chunk = True

        with open(shard_path, "rb") as f:
            f.seek(start_offset)
            while remaining > 0:
                # --- Inference priority: pause while inference is active ---
                if self._inference_priority:
                    while self._active_inferences > 0:
                        time.sleep(0.05)  # 50ms poll, yield to inference

                read_size = min(config.SHARD_CHUNK_SIZE, remaining)
                data = f.read(read_size)
                if not data:
                    break
                range_hasher.update(data)
                remaining -= len(data)

                # Include chunk_hash only on the last gRPC chunk of this range
                # (once the full range hash is computed).
                is_last = remaining <= 0
                yield inference_pb2.ShardChunk(
                    data=data,
                    offset=offset,
                    total_size=total_size,
                    shard_hash=shard_hash if first_chunk else "",
                    chunk_hash=range_hasher.hexdigest() if is_last else "",
                )
                offset += len(data)
                first_chunk = False

                # --- Rate limiting: sleep to cap upload speed ---
                if bytes_per_second > 0 and not is_last:
                    sleep_time = len(data) / bytes_per_second
                    time.sleep(sleep_time)

    def VerifyCapacity(self, request, context):
        """Capacity verification: prove reported resources by allocating memory."""
        from node.capacity import respond_to_challenge

        proof_hash, alloc_time = respond_to_challenge(
            request.challenge_bytes, request.expected_device)

        passed = proof_hash != "" and alloc_time >= 0
        return registry_pb2.VerifyCapacityResponse(
            passed=passed,
            proof_hash=proof_hash,
            allocation_time_ms=alloc_time,
        )


def serve(shard_index: int, port: int, host: str = "[::]",
          advertise_address: str = None, registry_address: str = None,
          node_config=None, model_type: str = "qwen2",
          shards_dir: str = None, eth_address: str = None,
          cheat_rate: float = 0.0, sampling_rate: float = 0.05,
          tls_cert: str = None, tls_key: str = None):
    """Start the gRPC server for a node.

    Args:
        shard_index: Which shard this node serves.
        port: Port to listen on.
        host: Bind address (default [::] = all interfaces).
        advertise_address: Address to register with the registry.
            Defaults to localhost:<port> for local testing.
            For cross-machine, set to the machine's reachable IP/hostname.
        registry_address: Registry server address.
        node_config: Optional NodeConfig for tunable values.
        model_type: "qwen2" for text-only, "qwen2_vl" for VL text decoder.
        shards_dir: Override shards directory for VL models.
          eth_address: Ethereum address to use as node_id (for on-chain staking).
        cheat_rate: Fraction of requests to corrupt (0.0=honest, for testing).
        sampling_rate: Verification ticket sampling rate (0.05=5%).
    """
    # Resolve configurable values
    manifest_path = (node_config.manifest_path
                     if node_config else config.MANIFEST_PATH)
    if model_type == "qwen2_vl":
        shards_dir = shards_dir or "shards_vl"
        manifest_path = os.path.join(shards_dir, "manifest.json")
    elif model_type == "smolvlm":
        shards_dir = shards_dir or "shards_smolvlm"
        manifest_path = os.path.join(shards_dir, "manifest.json")
    grpc_max_workers = (node_config.grpc_max_workers
                        if node_config else 4)
    grpc_options = (node_config.grpc_options
                    if node_config else config.GRPC_OPTIONS)
    ticket_interval = (node_config.ticket_submit_interval_seconds
                       if node_config else 10)

    # Load manifest to get shard info
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if model_type in ("qwen2_vl", "smolvlm"):
        shard_info = manifest["text_shards"][shard_index]
    else:
        shard_info = manifest["shards"][shard_index]
    layer_start = shard_info["layer_start"]
    layer_end = shard_info["layer_end"]
    has_embedding = shard_info.get("has_embedding", False)
    has_lm_head = shard_info.get("has_lm_head", False) or shard_info.get("has_head", False)

    # Create registration first (generates key pair)
    public_address = advertise_address or f"localhost:{port}"
    registration = NodeRegistration(
        address=public_address,
        model_id=manifest["model_id"],
        shard_index=shard_index,
        layer_start=layer_start,
        layer_end=layer_end,
        has_embedding=has_embedding,
        has_lm_head=has_lm_head,
        registry_address=registry_address,
        node_id=eth_address,
    )

    # Start the gRPC server (pass registration so servicer has the private key)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_max_workers),
        options=grpc_options,
    )
    servicer = InferenceNodeServicer(shard_index, port, registration,
                                     node_config=node_config,
                                     model_type=model_type,
                                     shards_dir=shards_dir,
                                     cheat_rate=cheat_rate,
                                     sampling_rate=sampling_rate)
    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)
    from network.tls import configure_server_port
    bind_address = f"{host}:{port}"
    configure_server_port(server, host, port, tls_cert, tls_key)
    server.start()

    print(f"[Node] Server listening on {bind_address} "
          f"(shard {shard_index}, layers {layer_start}-{layer_end - 1})")

    # Register with the registry (after server is up)
    registration.start()

    # Connect to the chain daemon for share submission
    servicer.init_daemon_connection(
        node_id=registration.node_id,
        registry_address=registry_address,
        self_address=public_address,
    )

    print(f"[Node] Advertised as {public_address}")
    print(f"[Node] Ready to serve requests.")

    # Start background KV cache cleanup thread
    session_timeout = (node_config.session_timeout_seconds
                       if node_config else config.SESSION_TIMEOUT_SECONDS)

    def _cache_cleanup_loop():
        """Periodically clean up stale KV cache sessions."""
        while True:
            time.sleep(60)  # check every minute
            cleaned = servicer.runner.cleanup_stale_sessions(session_timeout)

    cleanup_thread = threading.Thread(target=_cache_cleanup_loop, daemon=True)
    cleanup_thread.start()

    # Start background ticket submission thread
    import registry_pb2 as reg_pb2
    import registry_pb2_grpc as reg_pb2_grpc

    def _submit_tickets_loop():
        """Periodically drain collected tickets and submit them to the registry."""
        reg_channel = grpc.insecure_channel(
            registry_address or config.REGISTRY_ADDRESS)
        reg_stub = reg_pb2_grpc.RegistryStub(reg_channel)
        while True:
            time.sleep(ticket_interval)
            tickets = servicer._ticket_collector.drain_tickets()
            if not tickets:
                continue
            # Convert to proto messages
            proto_tickets = []
            for t in tickets:
                proto_tickets.append(reg_pb2.VerificationTicketProto(
                    ticket_id=t.ticket_id,
                    shard_index=t.shard_index,
                    input_data=t.input_data,
                    input_shape=list(t.input_shape),
                    input_is_tokens=t.input_is_tokens,
                    expected_output_data=t.expected_output_data,
                    expected_output_shape=list(t.expected_output_shape),
                    expected_token=t.expected_token if t.expected_token is not None else 0,
                    has_expected_token=t.expected_token is not None,
                    timestamp=t.timestamp,
                ))
            try:
                resp = reg_stub.SubmitTickets(reg_pb2.SubmitTicketsRequest(
                    node_id=registration.node_id,
                    tickets=proto_tickets,
                ))
                print(f"[Node] Submitted {resp.accepted} verification ticket(s)")
            except grpc.RpcError:
                pass  # registry may be temporarily unreachable

    ticket_thread = threading.Thread(target=_submit_tickets_loop, daemon=True)
    ticket_thread.start()

    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        print(f"\n[Node] Shutting down...")
        registration.stop()
        server.stop(grace=5)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    shutdown_event.wait()
    print("[Node] Stopped.")


class VisionNodeServicer(inference_pb2_grpc.InferenceNodeServicer):
    """gRPC service implementation for a vision shard node (ViT blocks).

    Supports two vision architectures:
      - Qwen2-VL: Conv3D patch embed, ViT blocks with rotary embeddings,
        PatchMerger, cu_seqlens for variable-length attention.
      - SmolVLM: SigLIP embeddings, standard encoder layers, pixel shuffle
        connector. Much simpler — no position_embeddings/cu_seqlens.

    The model_type is auto-detected from the manifest.
    """

    def __init__(self, shard_index: int, port: int,
                 registration: NodeRegistration = None,
                 shards_dir: str = "shards_vl"):
        self.shard_index = shard_index
        self.port = port
        self.registration = registration

        # Load manifest for shard metadata and model type
        manifest_path = os.path.join(shards_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.model_type = manifest.get("model_type", "qwen2_vl")
        vs = manifest["vision_shards"][shard_index]
        self.num_vision_shards = len(manifest["vision_shards"])

        # Extract common shard metadata (names may differ between v1/v2)
        self.layer_start = vs.get("layer_start", vs.get("block_start", 0))
        self.layer_end = vs.get("layer_end", vs.get("block_end", 0))
        self.has_embeddings = vs.get("has_embeddings", vs.get("has_patch_embed", False))
        self.has_connector = vs.get("has_connector", vs.get("has_merger", False))
        # Registration compatibility aliases
        self.has_patch_embed = self.has_embeddings
        self.has_merger = self.has_connector
        # Legacy Qwen2-VL aliases (block_start/block_end)
        self.block_start = self.layer_start
        self.block_end = self.layer_end

        from node.runtime.generic_runner import GenericVisionRunner

        vision_config = manifest.get("architecture", {}).get("vision")
        if not vision_config:
            raise ValueError(
                f"Manifest has no 'architecture.vision' config — "
                f"only v2+ manifests with generic runtime are supported. "
                f"Re-split the model with the latest splitter."
            )

        connector_config = manifest["architecture"].get("connector")
        shard_file = vs.get("file", f"vision_shard_{shard_index}.pt")
        shard_path = os.path.join(shards_dir, shard_file)

        self.runner = GenericVisionRunner(
            vision_config=vision_config,
            connector_config=connector_config,
            shard_info=vs,
            shard_path=shard_path,
        )
        print(f"[Vision] Using generic runtime (format_version="
              f"{manifest.get('format_version', '?')})")

        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        if address not in self._stubs:
            channel = create_resilient_channel(address, config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def Forward(self, request, context):
        """Handle a vision forward pass.

        Dispatches to the appropriate handler based on model_type:
          - SmolVLM: simpler path (pixel_values + patch_attention_mask only)
          - Qwen2-VL: complex path (position_embeddings, cu_seqlens, grid_thw)
        """
        if self.model_type == "smolvlm":
            return self._forward_smolvlm(request, context)
        else:
            return self._forward_qwen2vl(request, context)

    def _forward_smolvlm(self, request, context):
        """SmolVLM vision forward — simpler pipeline.

        Receives pixel_values (in image_pixels) or hidden_states (in
        activation_data). Runs through SigLIP encoder layers, optionally
        through connector. Returns image_features to caller.
        """
        import numpy as np

        session_id = request.session_id
        try:
            pixel_values = None
            hidden_states = None
            patch_attention_mask = None

            if request.image_pixels:
                # First shard: receiving raw pixel data from client
                # pixel_values packed as float32 bytes
                pixel_values = torch.from_numpy(
                    np.frombuffer(request.image_pixels, dtype=np.float32)
                    .copy()
                    .reshape(list(request.tensor_shape))
                )
                # patch_attention_mask packed in image_grid_thw field as int32
                if request.image_grid_thw:
                    mask_data = list(request.image_grid_thw)
                    # Decode mask shape from mrope_position_shape
                    if request.mrope_position_shape:
                        mask_shape = list(request.mrope_position_shape)
                        patch_attention_mask = torch.tensor(
                            mask_data, dtype=torch.bool
                        ).reshape(mask_shape)

            elif request.activation_data:
                # Intermediate shard: receiving hidden_states
                hidden_states = bytes_to_tensor(
                    request.activation_data, list(request.tensor_shape),
                    compressed=request.compressed)

            result = self.runner.forward(
                pixel_values=pixel_values,
                hidden_states=hidden_states,
                patch_attention_mask=patch_attention_mask,
            )

            # Determine next hop
            remaining = list(request.remaining_circuit)
            next_address = remaining.pop(0) if remaining else None

            if self.has_merger and "image_features" in result:
                # Last vision shard: return image features to caller
                feat = result["image_features"].cpu()
                feat_bytes = feat.contiguous().float().numpy().tobytes()
                feat_shape = list(feat.shape)

                return inference_pb2.ForwardResponse(
                    activation_data=feat_bytes,
                    tensor_shape=feat_shape,
                )

            elif next_address:
                # Forward to next vision shard
                h = result["hidden_states"].cpu()
                activation_bytes = h.contiguous().float().numpy().tobytes()
                shape = list(h.shape)

                next_request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    activation_data=activation_bytes,
                    tensor_shape=shape,
                    is_prefill=True,
                )
                next_request.remaining_circuit.extend(remaining)

                stub = self._get_stub(next_address)
                return stub.Forward(next_request)
            else:
                h = result["hidden_states"].cpu()
                activation_bytes, shape, is_compressed = tensor_to_bytes(h)
                return inference_pb2.ForwardResponse(
                    activation_data=activation_bytes,
                    tensor_shape=shape,
                    compressed=is_compressed,
                )

        except Exception as e:
            print(f"[SmolVLM Vision shard={self.shard_index}] Error: {e}")
            import traceback
            traceback.print_exc()
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.ForwardResponse()

    def _forward_qwen2vl(self, request, context):
        """Qwen2-VL vision forward — full pipeline with position embeddings.

        For subsequent vision shards (not MPC): receives activation_data
        + position_embeddings (packed in mrope_position_ids) + cu_seqlens
        (packed in image_embeddings).
        """
        import numpy as np

        session_id = request.session_id
        try:
            # Deserialize activations from the previous shard (or MPC node)
            hidden_states = None
            if request.activation_data:
                hidden_states = bytes_to_tensor(
                    request.activation_data, list(request.tensor_shape),
                    compressed=request.compressed)

            # Deserialize grid_thw
            grid_thw = None
            if request.image_grid_thw:
                grid_thw = torch.tensor(
                    list(request.image_grid_thw), dtype=torch.long
                ).reshape(-1, 3)

            # Deserialize position embeddings (packed in mrope fields by MPC node)
            position_embeddings = None
            if request.mrope_position_ids and request.mrope_position_shape:
                pe_shape = list(request.mrope_position_shape)
                total = 1
                for d in pe_shape:
                    total *= d
                all_data = np.frombuffer(
                    request.mrope_position_ids, dtype=np.float32).copy()
                # MPC node packed cos + sin concatenated
                half = len(all_data) // 2
                cos_data = all_data[:half].reshape(pe_shape)
                sin_data = all_data[half:].reshape(pe_shape)
                position_embeddings = (
                    torch.from_numpy(cos_data),
                    torch.from_numpy(sin_data),
                )

            # Deserialize cu_seqlens (packed in image_embeddings by MPC node)
            cu_seqlens = None
            if request.image_embeddings and request.image_embeddings_shape:
                cu_len = list(request.image_embeddings_shape)[0]
                cu_seqlens = torch.from_numpy(
                    np.frombuffer(request.image_embeddings, dtype=np.int32)
                    .copy()[:cu_len]
                )

            result = self.runner.forward(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
            )

            # Determine next hop
            remaining = list(request.remaining_circuit)
            next_address = remaining.pop(0) if remaining else None

            if self.has_merger and "image_embeddings" in result:
                # Last vision shard: return image embeddings to caller
                ie = result["image_embeddings"].cpu()
                ie_bytes = ie.contiguous().float().numpy().tobytes()
                ie_shape = list(ie.shape)

                resp = inference_pb2.ForwardResponse(
                    activation_data=ie_bytes,
                    tensor_shape=ie_shape,
                )
                return resp

            elif next_address:
                # Forward to next vision shard
                h = result["hidden_states"].cpu()
                activation_bytes = h.contiguous().float().numpy().tobytes()
                shape = list(h.shape)

                next_request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    activation_data=activation_bytes,
                    tensor_shape=shape,
                    is_prefill=True,
                )
                next_request.remaining_circuit.extend(remaining)

                # Forward grid_thw
                if grid_thw is not None:
                    next_request.image_grid_thw.extend(
                        grid_thw.flatten().tolist())

                # Forward position embeddings
                if "position_embeddings_cos" in result:
                    cos_bytes = result["position_embeddings_cos"].cpu().contiguous().float().numpy().tobytes()
                    sin_bytes = result["position_embeddings_sin"].cpu().contiguous().float().numpy().tobytes()
                    cos_shape = list(result["position_embeddings_cos"].shape)
                    next_request.mrope_position_ids = cos_bytes + sin_bytes
                    next_request.mrope_position_shape.extend(cos_shape)

                # Forward cu_seqlens
                if "cu_seqlens" in result:
                    cu_bytes = result["cu_seqlens"].cpu().numpy().astype(
                        np.int32).tobytes()
                    next_request.image_embeddings = cu_bytes
                    next_request.image_embeddings_shape.extend(
                        [len(result["cu_seqlens"])])

                stub = self._get_stub(next_address)
                response = stub.Forward(next_request)
                return response
            else:
                h = result["hidden_states"].cpu()
                activation_bytes, shape, is_compressed = tensor_to_bytes(h)
                return inference_pb2.ForwardResponse(
                    activation_data=activation_bytes,
                    tensor_shape=shape,
                    compressed=is_compressed,
                )

        except Exception as e:
            print(f"[Vision Node shard={self.shard_index}] Error in Forward: {e}")
            import traceback
            traceback.print_exc()
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return inference_pb2.ForwardResponse()

    def VerifyCapacity(self, request, context):
        """Capacity verification for vision nodes."""
        from node.capacity import respond_to_challenge

        proof_hash, alloc_time = respond_to_challenge(
            request.challenge_bytes, request.expected_device)

        passed = proof_hash != "" and alloc_time >= 0
        return registry_pb2.VerifyCapacityResponse(
            passed=passed,
            proof_hash=proof_hash,
            allocation_time_ms=alloc_time,
        )


def serve_vision(shard_index: int, port: int, host: str = "[::]",
                 advertise_address: str = None, registry_address: str = None,
                 shards_dir: str = "shards_vl", eth_address: str = None,
                 tls_cert: str = None, tls_key: str = None):
    """Start a vision shard gRPC server.

    Auto-detects model type from manifest (Qwen2-VL or SmolVLM).
    """
    manifest_path = os.path.join(shards_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    model_type = manifest.get("model_type", "qwen2_vl")
    vs = manifest["vision_shards"][shard_index]

    # SmolVLM uses layer_start/layer_end and has_embeddings/has_connector
    # Qwen2-VL uses block_start/block_end and has_patch_embed/has_merger
    if model_type == "smolvlm":
        layer_start = vs["layer_start"]
        layer_end = vs["layer_end"]
        has_first = vs.get("has_embeddings", False)
        has_last = vs.get("has_connector", False)
    else:
        layer_start = vs["block_start"]
        layer_end = vs["block_end"]
        has_first = vs.get("has_patch_embed", False)
        has_last = vs.get("has_merger", False)

    public_address = advertise_address or f"localhost:{port}"
    registration = NodeRegistration(
        address=public_address,
        model_id=manifest["model_id"],
        shard_index=shard_index,
        layer_start=layer_start,
        layer_end=layer_end,
        has_embedding=has_first,
        has_lm_head=has_last,
        node_type="vision",
        registry_address=registry_address,
        node_id=eth_address,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=config.GRPC_OPTIONS,
    )
    servicer = VisionNodeServicer(shard_index, port, registration,
                                  shards_dir=shards_dir)
    inference_pb2_grpc.add_InferenceNodeServicer_to_server(servicer, server)
    from network.tls import configure_server_port
    configure_server_port(server, host, port, tls_cert, tls_key)
    server.start()

    print(f"[Vision Node] Server listening on {host}:{port} "
          f"(vision shard {shard_index}, layers {layer_start}-{layer_end - 1})")

    registration.start()
    print(f"[Vision Node] Advertised as {public_address}")
    print(f"[Vision Node] Ready to serve requests.")

    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        print(f"\n[Vision Node] Shutting down...")
        registration.stop()
        server.stop(grace=5)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    shutdown_event.wait()
    print("[Vision Node] Stopped.")


def _download_shard_from_peers(
    peers: list[str], shard_index: int, model_id: str,
    shard_file: str, shards_dir: str,
) -> bool:
    """Download a shard weight file from explicit peers.

    Uses single-peer streaming via GetShard RPC.  For PoC this is
    sequential; the MultiPeerDownloader in shard/downloader.py can be
    used for parallel chunk downloads when chunk_hashes are available.
    """
    output_path = os.path.join(shards_dir, shard_file)
    if os.path.exists(output_path):
        print(f"[Auto] Shard file {shard_file} already exists locally")
        return True

    os.makedirs(shards_dir, exist_ok=True)

    for peer in peers:
        print(f"[Auto] Downloading {shard_file} from {peer}...")
        try:
            channel = grpc.insecure_channel(peer, options=config.GRPC_OPTIONS)
            stub = inference_pb2_grpc.InferenceNodeStub(channel)
            request = inference_pb2.GetShardRequest(
                model_id=model_id,
                shard_index=shard_index,
            )
            tmp_path = output_path + ".tmp"
            total_written = 0
            with open(tmp_path, "wb") as f:
                for chunk in stub.GetShard(request):
                    f.write(chunk.data)
                    total_written += len(chunk.data)

            channel.close()

            if total_written > 0:
                os.rename(tmp_path, output_path)
                mb = total_written / (1024 * 1024)
                print(f"[Auto] Downloaded {shard_file} ({mb:.1f} MB)")
                return True
            else:
                os.remove(tmp_path)
        except Exception as e:
            print(f"[Auto] Download from {peer} failed: {e}")
            if os.path.exists(output_path + ".tmp"):
                os.remove(output_path + ".tmp")

    print(f"[Auto] Could not download {shard_file} from any peer")
    return False


def serve_auto(port: int, host: str = "[::]",
               advertise_address: str = None, registry_address: str = None,
               shards_dir: str = "shards_smolvlm", eth_address: str = None,
               willing_to_mpc: bool = False, preferred_model: str = ""):
    """Auto-assignment startup: probe hardware, get assignment, serve.

    The node does NOT specify shard-index or model-type. Instead:
    1. Probe local hardware capacity
    2. List available shard files
    3. Call registry RequestAssignment RPC
    4. Download shard from peers if needed
    5. Launch as the assigned role (compute, vision, mpc, guard)
    """
    import uuid
    from node.capacity import probe_capacity, list_available_shards

    node_id = eth_address or str(uuid.uuid4())
    public_address = advertise_address or f"localhost:{port}"
    registry_addr = registry_address or config.REGISTRY_ADDRESS

    # 1. Probe hardware
    cap = probe_capacity()
    gpu_info = ""
    if cap["has_gpu"]:
        gpu_info = f" ({cap['gpu_vram_gb']:.1f} GB VRAM)"
    print(f"[Auto] Hardware: {cap['total_ram_gb']:.1f} GB RAM "
          f"({cap['available_ram_gb']:.1f} GB free), "
          f"{cap['cpu_cores']} CPU cores, "
          f"GPU={'yes' if cap['has_gpu'] else 'no'}{gpu_info}")

    # 2. List local shard files
    available = list_available_shards(shards_dir)
    print(f"[Auto] Local shards: {len(available)} files in {shards_dir}/")

    # 3. Request assignment from registry (with retry for MPC queuing)
    channel = grpc.insecure_channel(registry_addr, options=config.GRPC_OPTIONS)
    stub = registry_pb2_grpc.RegistryStub(channel)

    cap_proto = registry_pb2.NodeCapacity(
        total_ram_gb=cap["total_ram_gb"],
        available_ram_gb=cap["available_ram_gb"],
        cpu_cores=cap["cpu_cores"],
        has_gpu=cap["has_gpu"],
        gpu_vram_gb=cap["gpu_vram_gb"],
        available_gpu_vram_gb=cap["available_gpu_vram_gb"],
    )
    req = registry_pb2.RequestAssignmentRequest(
        node_id=node_id,
        capacity=cap_proto,
        preferred_model_id=preferred_model,
        available_shards=available,
        willing_to_mpc=willing_to_mpc,
        address=public_address,
    )

    assignment = None
    max_polls = 60
    for attempt in range(max_polls):
        try:
            resp = stub.RequestAssignment(req, timeout=10)
        except grpc.RpcError as e:
            print(f"[Auto] Registry unreachable ({e.code()}), retrying in 5s...")
            time.sleep(5)
            continue

        if resp.success:
            assignment = resp.assignment
            print(f"[Auto] Assigned: role={assignment.role} "
                  f"shard={assignment.shard_index} stack={assignment.stack} "
                  f"model={assignment.model_id}")
            break
        else:
            print(f"[Auto] {resp.message}")
            if "Queued for MPC" in resp.message:
                print(f"[Auto] Polling for MPC partner... (attempt {attempt+1})")
                time.sleep(5)
                continue
            else:
                print("[Auto] No assignment available. Retrying in 10s...")
                time.sleep(10)

    channel.close()

    if assignment is None:
        print("[Auto] Failed to get assignment after all retries. Exiting.")
        return

    # 4. Download shard if needed
    shard_file = assignment.shard_file
    if shard_file and shard_file not in available:
        if assignment.download_peers:
            ok = _download_shard_from_peers(
                list(assignment.download_peers),
                assignment.shard_index,
                assignment.model_id,
                shard_file,
                shards_dir,
            )
            if not ok:
                print("[Auto] Cannot proceed without shard file. Exiting.")
                return
        else:
            shard_path = os.path.join(shards_dir, shard_file)
            if not os.path.exists(shard_path):
                print(f"[Auto] Shard {shard_file} not found locally and no "
                      f"download peers. Exiting.")
                return

    # 5. Determine model_type from assignment stack
    if assignment.role == "guard":
        print("[Auto] Assigned as guard relay — starting guard server")
        _serve_guard(port, host, public_address, registry_addr,
                     node_id, assignment.model_id)
        return

    if assignment.role == "mpc":
        print(f"[Auto] Assigned as MPC role {assignment.mpc_role} "
              f"with peer {assignment.mpc_peer_address}")
        _serve_mpc(port, host, public_address, registry_addr,
                   node_id, assignment, shards_dir)
        return

    if assignment.stack == "vision_encoder":
        serve_vision(
            shard_index=assignment.shard_index,
            port=port, host=host,
            advertise_address=public_address,
            registry_address=registry_addr,
            shards_dir=shards_dir,
            eth_address=node_id,
        )
    else:
        model_type = _detect_model_type(shards_dir)
        serve(
            shard_index=assignment.shard_index,
            port=port, host=host,
            advertise_address=public_address,
            registry_address=registry_addr,
            model_type=model_type,
            shards_dir=shards_dir,
            eth_address=node_id,
        )


def _detect_model_type(shards_dir: str) -> str:
    """Detect model_type from the manifest in the shards directory."""
    manifest_path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return "qwen2"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    mt = manifest.get("model_type", "")
    if "smolvlm" in mt.lower() or "smolv" in mt.lower():
        return "smolvlm"
    if "vl" in mt.lower() or "vision" in mt.lower():
        return "qwen2_vl"
    return "qwen2"


def _serve_guard(port, host, public_address, registry_addr, node_id, model_id,
                 tls_cert=None, tls_key=None):
    """Start a lightweight guard relay node."""
    registration = NodeRegistration(
        address=public_address,
        model_id=model_id,
        shard_index=0,
        layer_start=0,
        layer_end=0,
        has_embedding=False,
        has_lm_head=False,
        node_type="guard",
        registry_address=registry_addr,
        node_id=node_id,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=config.GRPC_OPTIONS,
    )
    # Guard relay uses the same servicer framework but only relays
    inference_pb2_grpc.add_InferenceNodeServicer_to_server(
        inference_pb2_grpc.InferenceNodeServicer(), server)
    from network.tls import configure_server_port
    configure_server_port(server, host, port, tls_cert, tls_key)
    server.start()
    print(f"[Guard] Listening on {host}:{port}")

    registration.start()
    print(f"[Guard] Registered as guard relay for {model_id}")

    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        print("\n[Guard] Shutting down...")
        registration.stop()
        server.stop(grace=5)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    shutdown_event.wait()
    print("[Guard] Stopped.")


def _serve_mpc(port, host, public_address, registry_addr,
               node_id, assignment, shards_dir):
    """Start an MPC node with the assigned role and peer."""
    model_type = _detect_model_type(shards_dir)
    role_flag = "--mpc-role-a" if assignment.mpc_role == "A" else "--mpc-role-b"

    print(f"[Auto MPC] Launching MPC shard 0 as role {assignment.mpc_role}")
    print(f"[Auto MPC] Peer: {assignment.mpc_peer_address}")

    # Run as standard compute node with shard 0 + MPC subprocess
    serve(
        shard_index=0,
        port=port, host=host,
        advertise_address=public_address,
        registry_address=registry_addr,
        model_type=model_type,
        shards_dir=shards_dir,
        eth_address=node_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNFED AI Inference Node")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Shard index to load (from manifest)")
    parser.add_argument("--port", type=int, required=True,
                        help="Port for this node's gRPC server")
    parser.add_argument("--host", type=str, default="[::]",
                        help="Bind address (default: [::] = all interfaces)")
    parser.add_argument("--advertise", type=str, default=None,
                        help="Address to advertise to registry (default: localhost:<port>)")
    parser.add_argument("--registry", type=str, default=None,
                        help=f"Registry address (default: {config.REGISTRY_ADDRESS})")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to node_config.json (for tunable settings)")
    parser.add_argument("--model-type", type=str, default="qwen2",
                        choices=["qwen2", "qwen2_vl", "vision", "smolvlm",
                                 "smolvlm_vision"],
                        help="Model type: qwen2 (text), qwen2_vl (VL text decoder), "
                             "vision (VL vision shard), smolvlm (SmolVLM text decoder), "
                             "smolvlm_vision (SmolVLM vision shard)")
    parser.add_argument("--shards-dir", type=str, default=None,
                        help="Shards directory (default: shards/ or shards_vl/)")
    parser.add_argument("--eth-address", type=str, default=None,
                        help="Ethereum address for on-chain staking (used as node_id)")
    parser.add_argument("--cheat", type=float, default=0.0,
                        help="Cheat rate: fraction of requests to corrupt (0.0=honest, 1.0=always cheat)")
    parser.add_argument("--sampling-rate", type=float, default=0.05,
                        help="Verification ticket sampling rate (0.05=5%%)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-assignment mode: let the registry assign role and shard")
    parser.add_argument("--willing-to-mpc", action="store_true",
                        help="Accept MPC pairing in auto mode")
    parser.add_argument("--preferred-model", type=str, default="",
                        help="Preferred model ID in auto mode (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: cpu, cuda, cuda:0, or auto (default: cpu)")
    parser.add_argument("--tls-cert", type=str, default=None,
                        help="Path to TLS certificate PEM file")
    parser.add_argument("--tls-key", type=str, default=None,
                        help="Path to TLS private key PEM file")
    args = parser.parse_args()

    if args.auto:
        serve_auto(
            port=args.port,
            host=args.host,
            advertise_address=args.advertise,
            registry_address=args.registry,
            shards_dir=args.shards_dir or "shards_smolvlm",
            eth_address=args.eth_address,
            willing_to_mpc=args.willing_to_mpc,
            preferred_model=args.preferred_model,
        )
    else:
        if args.shard_index is None:
            parser.error("--shard-index is required unless --auto is used")

        # Load ComputeConfig from file or CLI args
        nc = None
        if args.config or args.device:
            from node.node_config import load_config
            cli_overrides = {
                "role": "compute",
                "shard_index": args.shard_index,
                "port": args.port,
                "host": args.host,
                "advertise": args.advertise,
                "registry": args.registry,
                "model_type": args.model_type,
                "shards_dir": args.shards_dir,
                "device": args.device,
            }
            nc = load_config(args.config, cli_overrides)

        if args.cheat > 0:
            print(f"\n*** WARNING: CHEAT MODE ENABLED ({args.cheat*100:.0f}% of requests) ***\n")

        if args.model_type in ("vision", "smolvlm_vision"):
            default_dir = "shards_smolvlm" if "smolvlm" in args.model_type else "shards_vl"
            serve_vision(args.shard_index, args.port, args.host, args.advertise,
                         args.registry, shards_dir=args.shards_dir or default_dir,
                         eth_address=args.eth_address,
                         tls_cert=args.tls_cert, tls_key=args.tls_key)
        else:
            serve(args.shard_index, args.port, args.host, args.advertise,
                  args.registry, node_config=nc, model_type=args.model_type,
                  shards_dir=args.shards_dir, eth_address=args.eth_address,
                  cheat_rate=args.cheat, sampling_rate=args.sampling_rate,
                  tls_cert=args.tls_cert, tls_key=args.tls_key)
