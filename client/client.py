"""
UNFED AI Client — discovers nodes, builds circuits, and runs distributed inference.

Privacy features:
  - Onion routing: layered encryption so no node sees the full circuit
  - Return-path encryption: response is encrypted so no node reads the answer
  - Redundant voting: double-check one random shard per query for correctness

Multimodal (Qwen2-VL):
  - Two-phase pipeline: vision (image → embeddings) then text (tokens → generation)
  - MPC for vision: raw image patches secret-shared to MPC vision nodes
  - Image embeddings merged into text sequence at placeholder token positions

Usage:
    python -m client.client --prompt "The capital of France is"
    python -m client.client --prompt "Once upon a time" --max-tokens 200
    python -m client.client  # interactive mode
    python -m client.client --use-voting --prompt "Hello"
    python -m client.client --model-type qwen2_vl --image photo.jpg --prompt "What is this?"
"""

import argparse
import json
import os
import random
import sys
import time
import uuid

import grpc
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
from network.discovery import RegistryClient, RegistryPool
from network.resilience import create_resilient_channel, with_retry
from network.onion import (
    build_onion, public_key_from_bytes,
    generate_response_keys, decrypt_response_layers,
)
from network.voting import VotingCoordinator
from network.racing import RacingCoordinator

class UnfedClient:
    """Client for the UNFED AI distributed inference pipeline."""

    def __init__(self, registry_address: str = None,
                 use_voting: bool = False, use_return_encryption: bool = False,
                 use_racing: bool = False, model_id: str = None,
                 tls_ca: str = None):
        self.model_id = model_id or config.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self._tls_credentials = None
        if tls_ca:
            from network.tls import make_channel_credentials
            self._tls_credentials = make_channel_credentials(tls_ca)

        if registry_address:
            self.discovery = RegistryPool([registry_address])
        else:
            self.discovery = RegistryPool()

        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._channels: dict[str, grpc.Channel] = {}

        # Privacy features
        self.use_voting = use_voting
        self.use_return_encryption = use_return_encryption
        self.use_racing = use_racing

        # Voting coordinator
        self._voter = VotingCoordinator() if use_voting else None

        # Racing coordinator (per-shard fault tolerance)
        self._racer = RacingCoordinator() if use_racing else None

        # Prefix caching: track previous session for KV cache reuse
        self._prev_session_id: str = ""
        self._prev_token_ids: list[int] = []

    def _compute_prefix(self, new_token_ids: list[int]) -> tuple[str, int]:
        """Compare new token_ids with the previous session's tokens.

        Returns (prefix_session_id, prefix_length) if there's a reusable prefix,
        or ("", 0) if there's nothing to reuse.
        """
        if not self._prev_session_id or not self._prev_token_ids:
            return "", 0

        # Find the longest common prefix
        min_len = min(len(self._prev_token_ids), len(new_token_ids))
        prefix_len = 0
        for i in range(min_len):
            if self._prev_token_ids[i] == new_token_ids[i]:
                prefix_len += 1
            else:
                break

        # Only reuse if the prefix is at least 10 tokens (not worth it otherwise)
        if prefix_len < 10:
            return "", 0

        return self._prev_session_id, prefix_len

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        """Get or create a gRPC stub for the given address (resilient channel)."""
        if address not in self._stubs:
            channel = create_resilient_channel(
                address, config.GRPC_OPTIONS,
                channel_credentials=self._tls_credentials,
            )
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
            self._channels[address] = channel
        return self._stubs[address]

    def _warmup_channels(self, addresses: list[str], timeout: float = 5.0):
        """Pre-establish gRPC connections to eliminate cold-start latency."""
        from concurrent.futures import ThreadPoolExecutor
        def _warmup_one(addr):
            self._get_stub(addr)
            try:
                grpc.channel_ready_future(self._channels[addr]).result(timeout=timeout)
            except Exception:
                pass
        with ThreadPoolExecutor(max_workers=len(addresses)) as pool:
            list(pool.map(_warmup_one, addresses))

    def get_fee_estimate(self, estimated_tokens: int = 100) -> dict:
        """Query the daemon's fee oracle for current pricing.

        Returns a dict with base_fee, utilization, estimated_cost, suggested_tip.
        Returns a default estimate if no daemon is reachable.
        """
        try:
            all_nodes = self.discovery.discover("")
            daemons = [n for n in all_nodes if n.node_type == "daemon"]
            if daemons:
                daemon = daemons[0]
                stub = self._get_stub(daemon.address)
                resp = stub.GetFeeEstimate(
                    inference_pb2.FeeEstimateRequest(
                        estimated_tokens=estimated_tokens),
                    timeout=5,
                )
                return {
                    "base_fee": resp.base_fee,
                    "utilization": resp.utilization,
                    "estimated_cost": resp.estimated_cost,
                    "suggested_tip": resp.suggested_tip,
                }
        except Exception:
            pass
        # Fallback: return default estimate
        return {
            "base_fee": 0.001,
            "utilization": 0.0,
            "estimated_cost": 0.001 * estimated_tokens,
            "suggested_tip": 0.0,
        }

    def generate(self, prompt: str, max_new_tokens: int = 100,
                 verbose: bool = False, use_onion: bool = True,
                 use_random_routing: bool = False,
                 fee_per_token: float = 0.0,
                 tip: float = 0.0):
        """
        Generate text from a prompt using the distributed pipeline.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            verbose: If True, print timing info
            use_onion: If True, use onion-encrypted routing (default)
            use_random_routing: If True, use hash-based routing
            fee_per_token: Max fee per token (0 = use oracle default)
            tip: Priority tip per token (0 = no tip)

        Yields:
            Generated text tokens as they arrive
        """
        # Dispatch to racing mode if enabled
        if self.use_racing and self._racer:
            self._fee_per_token = fee_per_token
            self._tip = tip
            yield from self._generate_racing(prompt, max_new_tokens, verbose)
            return

        session_id = str(uuid.uuid4())
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

        # Prefix caching: check if the new prompt shares a prefix with the previous one
        prefix_session_id, prefix_length = self._compute_prefix(input_ids)

        # Build circuit: discover nodes, pick one per shard.
        # If MPC nodes are available, use one for shard 0 (privacy default).
        using_mpc = False
        mpc_nodes = self.discovery.discover_mpc(self.model_id)
        result = self.discovery.build_circuit(self.model_id)

        if not result:
            raise RuntimeError(
                f"Cannot build circuit for model '{self.model_id}' — "
                f"no compute nodes found or pool is incomplete. "
                f"Ensure nodes are running and registered."
            )

        addresses, public_key_bytes = result
        num_shards = len(addresses)

        if mpc_nodes:
            # Replace shard-0 entry with an MPC node
            mpc_node = random.choice(mpc_nodes)
            addresses[0] = mpc_node.address
            public_key_bytes[0] = bytes(mpc_node.public_key)
            using_mpc = True

        if use_random_routing:
            routing_mode = "random"
        elif use_onion:
            routing_mode = "onion"
        else:
            routing_mode = "plain"

        if verbose:
            print(f"[Session {session_id[:8]}...] Prompt tokens: {len(input_ids)}")
            mpc_label = " + MPC shard 0" if using_mpc else ""
            print(f"[Routing] {routing_mode}{mpc_label}")
            if self.use_return_encryption:
                print(f"[Return Encryption] Enabled for {num_shards} nodes")
            if self.use_voting:
                print(f"[Voting] Enabled (1 random shard per query)")
            if prefix_length > 0:
                print(f"[Prefix Cache] Reusing {prefix_length} tokens from "
                      f"session {prefix_session_id[:8]}...")

        # For random routing, we only need the entry node (shard 0)
        entry_address = addresses[0]

        # Build onion or plain circuit
        onion_blob = b""
        onion_ephemeral_key = b""
        remaining_circuit = []

        if routing_mode == "onion" and all(pk for pk in public_key_bytes):
            pub_keys = [public_key_from_bytes(pk) for pk in public_key_bytes]
            onion_blob, onion_ephemeral_key = build_onion(addresses, pub_keys)
            if verbose:
                print(f"[Circuit] {' → '.join(addresses)}")
                print(f"[Onion] {len(onion_blob)} bytes, {len(addresses)} layers")
        elif routing_mode == "plain":
            remaining_circuit = addresses[1:]
            if verbose:
                print(f"[Circuit] {' → '.join(addresses)}")
        elif routing_mode == "random":
            if verbose:
                print(f"[Entry] {entry_address} (next hops determined by hash(output))")

        # Generate return-path encryption keys
        response_keys = []
        response_key_list = []
        if self.use_return_encryption:
            response_key_list, _ = generate_response_keys(num_shards)
            response_keys = response_key_list

        # Voting: select which shard to double-check
        voted_shard = None
        if self.use_voting and self._voter and num_shards > 0:
            voted_shard = self._voter.select_voted_shard(num_shards)
            if verbose:
                print(f"[Voting] Will double-check shard {voted_shard}")

        self._warmup_channels(addresses)
        entry_stub = self._get_stub(entry_address)
        generated_tokens = list(input_ids)
        total_start = time.time()

        for step in range(max_new_tokens):
            step_start = time.time()

            _wire_dtype = config.WIRE_DTYPE or ''
            if step == 0:
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=input_ids,
                    is_prefill=True,
                    prefix_session_id=prefix_session_id,
                    prefix_length=prefix_length,
                    fee_per_token=fee_per_token,
                    tip=tip,
                    wire_dtype=_wire_dtype,
                )
            else:
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=[generated_tokens[-1]],
                    is_prefill=False,
                    fee_per_token=fee_per_token,
                    tip=tip,
                    wire_dtype=_wire_dtype,
                )

            # Attach routing
            if routing_mode == "random":
                request.use_random_routing = True
            elif onion_blob:
                request.onion_blob = onion_blob
                request.onion_ephemeral_key = onion_ephemeral_key
            else:
                request.remaining_circuit.extend(remaining_circuit)

            # Attach return-path encryption keys
            if response_keys:
                request.response_keys.extend(response_keys)

            # --- Voting: double-check one shard (prefill only) ---
            if (voted_shard is not None and step == 0 and self._voter):
                all_nodes = self.discovery.discover_compute(self.model_id)
                primary_addr = addresses[voted_shard]
                candidates = self._voter.find_candidates(
                    all_nodes, voted_shard, exclude_address=primary_addr)

                if candidates:
                    voter_node = random.choice(candidates)
                    vote_result = self._voter.vote(
                        node_a_address=primary_addr,
                        node_b_address=voter_node.address,
                        shard_index=voted_shard,
                        session_id=session_id,
                        token_ids=input_ids if voted_shard == 0 else None,
                        is_prefill=True,
                    )
                    if not vote_result.match:
                        vote_result = self._voter.resolve_conflict(
                            all_nodes, voted_shard, session_id, vote_result,
                            token_ids=input_ids if voted_shard == 0 else None,
                            is_prefill=True,
                        )
                        print(f"[Voting] Conflict resolved, winner: "
                              f"{vote_result.winner_address}")
                else:
                    if verbose:
                        print(f"[Voting] No second node for shard {voted_shard}, "
                              f"skipping vote")

            try:
                response = entry_stub.Forward(request)
            except grpc.RpcError as e:
                print(f"\n[Error] gRPC call failed: {e.details()}")
                break

            step_time = time.time() - step_start

            # Decrypt return-path encryption if active
            if self.use_return_encryption and response.encrypted_response and response_key_list:
                try:
                    plaintext = decrypt_response_layers(
                        response_key_list, response.encrypted_response)
                    # Parse token from decrypted data
                    token_id = int.from_bytes(plaintext[:4], 'big', signed=True)
                    is_eos = plaintext[4:5] == b'\x01'
                    generated_tokens.append(token_id)
                    token_text = self.tokenizer.decode([token_id])

                    if verbose:
                        print(f"  [{step_time:.3f}s] Token {step}: {token_id} "
                              f"-> {token_text!r} (decrypted)")

                    yield token_text

                    if is_eos:
                        break
                    continue
                except Exception as e:
                    if verbose:
                        print(f"  [Decrypt] Failed: {e}, falling back to plaintext")

            if response.has_token:
                token_id = response.token_id
                generated_tokens.append(token_id)
                token_text = self.tokenizer.decode([token_id])

                if verbose:
                    print(f"  [{step_time:.3f}s] Token {step}: {token_id} -> {token_text!r}")

                yield token_text

                if response.is_eos:
                    break
            else:
                break

        # Save session state for future prefix reuse
        self._prev_session_id = session_id
        self._prev_token_ids = input_ids  # save the prompt tokens (not the generated ones)

        if verbose:
            total_time = time.time() - total_start
            num_generated = len(generated_tokens) - len(input_ids)
            tokens_per_sec = num_generated / total_time if total_time > 0 else 0
            print(f"\n[Stats] {num_generated} tokens in {total_time:.2f}s "
                  f"({tokens_per_sec:.1f} tok/s)")

    def _generate_racing(self, prompt: str, max_new_tokens: int = 100,
                         verbose: bool = False):
        """
        Generate text using per-shard racing for fault tolerance.

        Client-driven hop-by-hop routing: at each shard, the client races
        N replicas and takes the first response. All replicas process
        every token to keep their KV caches in sync.

        Yields:
            Generated text tokens as they arrive
        """
        import numpy as np

        session_id = str(uuid.uuid4())
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

        # Prefix caching
        prefix_session_id, prefix_length = self._compute_prefix(input_ids)

        # Build racing circuit — prefer MPC for shard 0 (privacy default)
        using_mpc = False
        mpc_result = self.discovery.build_mpc_racing_circuit(
            self.model_id, replicas=config.RACING_REPLICAS)

        if mpc_result:
            # MPC available: shard 0 = MPC entries, shards 1+ = regular compute
            mpc_entries, compute_shard_map = mpc_result
            shard_addresses: dict[int, list[str]] = {
                0: [addr for addr, _pk in mpc_entries],
            }
            for shard_idx, nodes in compute_shard_map.items():
                shard_addresses[shard_idx] = [addr for addr, _pk in nodes]
            num_shards = max(shard_addresses.keys()) + 1
            using_mpc = True
        else:
            # No MPC nodes — fall back to regular racing
            shard_map = self.discovery.build_racing_circuit(
                self.model_id, replicas=config.RACING_REPLICAS)
            if not shard_map:
                print("[Error] Cannot build racing circuit — pool may be incomplete")
                return
            num_shards = max(shard_map.keys()) + 1
            shard_addresses: dict[int, list[str]] = {}
            for shard_idx, nodes in shard_map.items():
                shard_addresses[shard_idx] = [addr for addr, _pk in nodes]

        if verbose:
            print(f"[Session {session_id[:8]}...] Prompt tokens: {len(input_ids)}")
            mode_label = "MPC+racing" if using_mpc else "racing"
            print(f"[Routing] {mode_label} ({config.RACING_REPLICAS} replicas/shard)")
            if prefix_length > 0:
                print(f"[Prefix Cache] Reusing {prefix_length} tokens from "
                      f"session {prefix_session_id[:8]}...")
            for i in range(num_shards):
                addrs = shard_addresses.get(i, [])
                label = "MPC" if (i == 0 and using_mpc) else f"Shard {i}"
                print(f"  {label}: {', '.join(addrs)}")

        all_addrs = []
        for addrs in shard_addresses.values():
            all_addrs.extend(addrs)
        self._warmup_channels(all_addrs)

        generated_tokens = list(input_ids)
        total_start = time.time()

        # Pipelined prefill for long prompts (uses primary node per shard)
        prefill_min_tokens = config.PREFILL_PIPELINE_MIN_TOKENS
        use_pipelined = len(input_ids) >= prefill_min_tokens and num_shards > 1
        prefill_done = False

        if use_pipelined:
            primary_addrs = {
                s: [addrs[0]] for s, addrs in shard_addresses.items() if addrs
            }
            if verbose:
                print(f"[Pipelined Prefill] {len(input_ids)} tokens, "
                      f"{num_shards} shards")
            self._pipelined_prefill(
                session_id=session_id,
                input_ids=input_ids,
                shard_addresses=primary_addrs,
                num_shards=num_shards,
                prefix_session_id=prefix_session_id,
                prefix_length=prefix_length,
                chunk_size=config.PREFILL_CHUNK_SIZE,
                verbose=verbose,
            )
            prefill_done = True

        for step in range(max_new_tokens):
            step_start = time.time()

            current_activation = None
            current_shape = None

            for shard_idx in range(num_shards):
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    fee_per_token=getattr(self, '_fee_per_token', 0.0),
                    tip=getattr(self, '_tip', 0.0),
                    wire_dtype=config.WIRE_DTYPE or '',
                )

                if shard_idx == 0:
                    if step == 0 and not prefill_done:
                        request.token_ids.extend(input_ids)
                        request.prefix_session_id = prefix_session_id
                        request.prefix_length = prefix_length
                    else:
                        request.token_ids.extend([generated_tokens[-1]])
                else:
                    request.activation_data = current_activation
                    request.tensor_shape.extend(current_shape)
                    if step == 0 and not prefill_done:
                        request.prefix_session_id = prefix_session_id
                        request.prefix_length = prefix_length

                nodes = shard_addresses.get(shard_idx, [])
                if not nodes:
                    print(f"[Error] No nodes for shard {shard_idx}")
                    return

                try:
                    race_result = self._racer.race_shard(
                        shard_index=shard_idx,
                        node_addresses=nodes,
                        request=request,
                        guard_address=None,
                    )
                except (TimeoutError, RuntimeError) as e:
                    print(f"\n[Error] Racing failed at shard {shard_idx}: {e}")
                    return

                response = race_result.response

                if verbose and shard_idx < num_shards - 1:
                    print(f"    Shard {shard_idx}: won by {race_result.winner_address} "
                          f"({race_result.latency_ms:.0f}ms)")

                if response.has_token:
                    break
                elif response.activation_data:
                    current_activation = bytes(response.activation_data)
                    current_shape = list(response.tensor_shape)
                else:
                    print(f"[Error] Shard {shard_idx} returned empty response")
                    return

            step_time = time.time() - step_start

            if response.has_token:
                token_id = response.token_id
                generated_tokens.append(token_id)
                token_text = self.tokenizer.decode([token_id])

                if verbose:
                    print(f"  [{step_time:.3f}s] Token {step}: {token_id} "
                          f"-> {token_text!r} "
                          f"(won by {race_result.winner_address})")

                yield token_text

                if response.is_eos:
                    break
            else:
                print(f"[Error] Final shard did not return a token")
                break

        # Save session state for future prefix reuse
        self._prev_session_id = session_id
        self._prev_token_ids = input_ids

        if verbose:
            total_time = time.time() - total_start
            num_generated = len(generated_tokens) - len(input_ids)
            tokens_per_sec = num_generated / total_time if total_time > 0 else 0
            print(f"\n[Stats] {num_generated} tokens in {total_time:.2f}s "
                  f"({tokens_per_sec:.1f} tok/s)")
            mismatches = self._racer.mismatches
            if mismatches:
                print(f"[Racing] {len(mismatches)} mismatch(es) detected!")
                for m in mismatches:
                    print(f"  Shard {m['shard_index']}: "
                          f"{m['winner']} vs {m['other']}")
            else:
                print(f"[Racing] All verifications passed")

    def _pipelined_prefill(
        self,
        session_id: str,
        input_ids: list[int],
        shard_addresses: dict[int, list[str]],
        num_shards: int,
        prefix_session_id: str = "",
        prefix_length: int = 0,
        chunk_size: int = 64,
        verbose: bool = False,
    ) -> tuple[bytes, list[int]]:
        """Run pipelined prefill: overlap chunk processing across shards.

        Returns the final activation (bytes, shape) from the last non-final shard,
        or (None, None) if the pipeline ends with a token.
        After this returns, all shards have fully populated KV caches for the prompt.
        """
        import concurrent.futures

        wire_dtype = config.WIRE_DTYPE or ''

        tokens = list(input_ids)
        n_chunks = max(1, (len(tokens) + chunk_size - 1) // chunk_size)
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(tokens))
            chunks.append(tokens[start:end])

        if verbose:
            print(f"[Pipelined Prefill] {len(tokens)} tokens → "
                  f"{n_chunks} chunks of ~{chunk_size}")

        # chunk_activations[chunk_idx][shard_idx] = (activation_bytes, shape) or None
        # For shard 0, activation is generated from token_ids.
        chunk_activations: dict[int, dict[int, tuple]] = {
            c: {} for c in range(n_chunks)
        }

        # Track the last token response from the final shard
        final_response = None

        def _process_chunk_at_shard(chunk_idx: int, shard_idx: int):
            """Process one chunk at one shard. Returns (activation_bytes, shape) or token response."""
            chunk_tokens = chunks[chunk_idx]
            addr_list = shard_addresses.get(shard_idx, [])
            if not addr_list:
                raise RuntimeError(f"No nodes for shard {shard_idx}")
            addr = addr_list[0]

            request = inference_pb2.ForwardRequest(
                session_id=session_id,
                is_prefill=True,
                wire_dtype=wire_dtype,
            )

            if shard_idx == 0:
                request.token_ids.extend(chunk_tokens)
                if chunk_idx == 0:
                    request.prefix_session_id = prefix_session_id
                    request.prefix_length = prefix_length
            else:
                prev_act, prev_shape = chunk_activations[chunk_idx][shard_idx - 1]
                request.activation_data = prev_act
                request.tensor_shape.extend(prev_shape)
                if chunk_idx == 0:
                    request.prefix_session_id = prefix_session_id
                    request.prefix_length = prefix_length

            stub = self._get_stub(addr)
            response = stub.Forward(request)

            if response.has_token:
                return ("token", response)
            elif response.activation_data:
                return ("activation", (bytes(response.activation_data),
                                       list(response.tensor_shape)))
            else:
                raise RuntimeError(
                    f"Empty response from shard {shard_idx} chunk {chunk_idx}")

        # Pipeline execution: process chunks through shards with overlap.
        # We process diagonal wavefronts: at wavefront W, we can process
        # all (chunk_idx, shard_idx) pairs where chunk_idx + shard_idx == W.
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(num_shards, 4)) as pool:
            for wavefront in range(n_chunks + num_shards - 1):
                futures = {}
                for shard_idx in range(num_shards):
                    chunk_idx = wavefront - shard_idx
                    if chunk_idx < 0 or chunk_idx >= n_chunks:
                        continue
                    if shard_idx > 0 and (shard_idx - 1) not in chunk_activations.get(chunk_idx, {}):
                        continue
                    futures[(chunk_idx, shard_idx)] = pool.submit(
                        _process_chunk_at_shard, chunk_idx, shard_idx)

                for (ci, si), fut in futures.items():
                    result_type, result_data = fut.result()
                    if result_type == "activation":
                        chunk_activations[ci][si] = result_data
                    elif result_type == "token":
                        final_response = result_data

                if verbose and futures:
                    processed = ", ".join(
                        f"c{ci}@s{si}" for (ci, si) in sorted(futures.keys()))
                    print(f"  Wavefront {wavefront}: {processed}")

        last_shard = num_shards - 1
        last_chunk = n_chunks - 1
        if last_shard in chunk_activations.get(last_chunk, {}):
            return chunk_activations[last_chunk][last_shard]
        return None, None

    def generate_multimodal(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: int = 100,
        verbose: bool = False,
        model_id: str = None,
    ):
        """
        Generate text from a prompt + image using the two-phase VL pipeline.

        Phase 1 (Vision): preprocess image → secret-share patches → send to
        MPC vision node → forward through vision shards → receive image embeddings.

        Phase 2 (Text): tokenize text with image placeholders → send token_ids
        + image_embeddings + M-RoPE position IDs → autoregressive text generation.

        Args:
            prompt: Text prompt (e.g., "What is in this image?")
            image_path: Path to the image file.
            max_new_tokens: Maximum new tokens to generate.
            verbose: Print detailed timing info.
            model_id: Override the model ID for the VL model.

        Yields:
            Generated text tokens as they arrive.
        """
        from network.mpc_shard0 import create_additive_shares

        vl_model_id = model_id or self.model_id
        session_id = str(uuid.uuid4())

        # === Phase 1: Vision ===
        if verbose:
            print(f"[Vision] Processing image: {image_path}")
            vision_start = time.time()

        # Preprocess image using Qwen2VLImageProcessor (avoid AutoProcessor video bug)
        from transformers import Qwen2VLImageProcessor, AutoConfig
        image_proc = Qwen2VLImageProcessor.from_pretrained(vl_model_id)

        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        # Process image to get pixel values and grid dimensions
        img_result = image_proc(images=[image], return_tensors="pt")
        pixel_values = img_result["pixel_values"]    # Preprocessed pixel tensor
        image_grid_thw = img_result["image_grid_thw"]  # [num_images, 3]

        # Build token IDs with image placeholder tokens
        vl_config = AutoConfig.from_pretrained(vl_model_id)
        vision_start_id = vl_config.vision_start_token_id
        vision_end_id = vl_config.vision_end_token_id
        image_token_id = vl_config.image_token_id
        spatial_merge = vl_config.vision_config.spatial_merge_size

        # Calculate number of merged image tokens
        t = image_grid_thw[0, 0].item()
        h = image_grid_thw[0, 1].item()
        w = image_grid_thw[0, 2].item()
        num_merged = t * (h // spatial_merge) * (w // spatial_merge)

        # Build the chat-style token sequence:
        # <|im_start|>user\n<|vision_start|><|image_pad|>...<|vision_end|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        im_start = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        user_tok = self.tokenizer.encode("user\n", add_special_tokens=False)
        newline = self.tokenizer.encode("\n", add_special_tokens=False)
        asst_tok = self.tokenizer.encode("assistant\n", add_special_tokens=False)

        input_ids = (im_start + user_tok +
                     [vision_start_id] + [image_token_id] * num_merged + [vision_end_id] +
                     newline + prompt_tokens +
                     im_end + newline + im_start + asst_tok)

        if verbose:
            print(f"[Vision] Pixels shape: {pixel_values.shape}")
            print(f"[Vision] Grid THW: {image_grid_thw.tolist()}")
            print(f"[Vision] Token count: {len(input_ids)}")

        # Discover vision circuit (MPC node + vision shards)
        vision_result = self.discovery.build_vision_circuit(vl_model_id)

        if vision_result:
            vision_addresses, vision_pub_keys = vision_result
            if verbose:
                print(f"[Vision] Circuit: {' → '.join(vision_addresses)}")

            # Send image to MPC vision node (first in vision circuit)
            mpc_address = vision_addresses[0]
            remaining_vision = vision_addresses[1:]

            # Serialize pixel data for the MPC node
            pixel_bytes = pixel_values.contiguous().float().numpy().tobytes()
            grid_thw_list = image_grid_thw.flatten().tolist()

            vision_request = inference_pb2.ForwardRequest(
                session_id=session_id,
                image_pixels=pixel_bytes,
                image_grid_thw=grid_thw_list,
                is_prefill=True,
            )
            vision_request.remaining_circuit.extend(remaining_vision)

            mpc_stub = self._get_stub(mpc_address)
            vision_response = mpc_stub.Forward(vision_request)

            # Extract image embeddings from vision pipeline
            image_embeddings_bytes = bytes(vision_response.activation_data)
            ie_shape = list(vision_response.tensor_shape)
            image_embeddings = torch.from_numpy(
                np.frombuffer(image_embeddings_bytes, dtype=np.float32)
                .copy().reshape(ie_shape)
            )

            if verbose:
                vision_time = time.time() - vision_start
                print(f"[Vision] Image embeddings: {image_embeddings.shape} "
                      f"({vision_time:.2f}s)")
        else:
            raise RuntimeError(
                "No vision nodes found for the multimodal pipeline. "
                "The Qwen2-VL model requires dedicated vision shard nodes "
                "and MPC vision nodes to be running."
            )

        # === Phase 2: Text ===
        if verbose:
            print(f"[Text] Starting autoregressive generation")
            text_start = time.time()

        # Build text circuit
        text_result = self.discovery.build_circuit(vl_model_id)
        if not text_result:
            raise RuntimeError(
                f"No compute nodes found for model '{vl_model_id}'. "
                f"The Qwen2-VL text pipeline needs dedicated compute nodes "
                f"registered for this model ID."
            )

        text_addresses, text_pub_keys = text_result
        num_shards = len(text_addresses)

        if verbose:
            print(f"[Text] Circuit: {' → '.join(text_addresses)}")

        # Compute M-RoPE position IDs
        mrope_position_ids = self._compute_mrope_position_ids(
            input_ids, image_grid_thw, vl_model_id
        )

        entry_address = text_addresses[0]
        remaining_text_circuit = text_addresses[1:]
        entry_stub = self._get_stub(entry_address)

        generated_tokens = list(input_ids)

        # Count image tokens for verification
        num_image_tokens = sum(1 for t in input_ids if t == image_token_id)

        for step in range(max_new_tokens):
            step_start = time.time()

            if step == 0:
                # Prefill: send full token sequence + image embeddings + M-RoPE
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=input_ids,
                    is_prefill=True,
                    num_image_tokens=num_image_tokens,
                )
                request.remaining_circuit.extend(remaining_text_circuit)

                # Attach image embeddings
                if image_embeddings is not None:
                    ie_bytes = image_embeddings.contiguous().float().numpy().tobytes()
                    request.image_embeddings = ie_bytes
                    request.image_embeddings_shape.extend(
                        list(image_embeddings.shape))

                # Attach M-RoPE position IDs
                if mrope_position_ids is not None:
                    mp_bytes = mrope_position_ids.contiguous().numpy().astype(
                        np.int64).tobytes()
                    request.mrope_position_ids = mp_bytes
                    request.mrope_position_shape.extend(
                        list(mrope_position_ids.shape))
            else:
                # Decode: single token
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=[generated_tokens[-1]],
                    is_prefill=False,
                )
                request.remaining_circuit.extend(remaining_text_circuit)

            try:
                response = entry_stub.Forward(request)
            except grpc.RpcError as e:
                print(f"\n[Error] gRPC call failed: {e.details()}")
                break

            step_time = time.time() - step_start

            if response.has_token:
                token_id = response.token_id
                generated_tokens.append(token_id)
                token_text = self.tokenizer.decode([token_id])

                if verbose:
                    print(f"  [{step_time:.3f}s] Token {step}: {token_id} -> {token_text!r}")

                yield token_text

                if response.is_eos:
                    break
            else:
                break

        if verbose:
            total_time = time.time() - text_start
            num_generated = len(generated_tokens) - len(input_ids)
            tokens_per_sec = num_generated / total_time if total_time > 0 else 0
            print(f"\n[Stats] {num_generated} tokens in {total_time:.2f}s "
                  f"({tokens_per_sec:.1f} tok/s)")

    def generate_multimodal_smolvlm(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: int = 100,
        verbose: bool = False,
        model_id: str = None,
    ):
        """
        Generate text from a prompt + image using SmolVLM.

        Simpler than Qwen2-VL — no M-RoPE, no cu_seqlens.

        Phase 1 (Vision): preprocess image → send pixel tiles to vision node
        → receive image_features (after connector).

        Phase 2 (Text): tokenize with <image> placeholders → send token_ids
        + image_features → autoregressive text generation.

        Args:
            prompt: Text prompt (e.g., "What is in this image?")
            image_path: Path to the image file.
            max_new_tokens: Maximum new tokens to generate.
            verbose: Print detailed timing info.
            model_id: Override the model ID.

        Yields:
            Generated text tokens as they arrive.
        """
        vl_model_id = model_id or self.model_id
        session_id = str(uuid.uuid4())

        # === Phase 1: Vision ===
        if verbose:
            print(f"[SmolVLM Vision] Processing image: {image_path}")
            vision_start = time.time()

        from transformers import AutoProcessor, AutoConfig
        from PIL import Image

        processor = AutoProcessor.from_pretrained(vl_model_id)
        vl_config = AutoConfig.from_pretrained(vl_model_id)
        image_token_id = vl_config.image_token_id

        # Build chat template
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

        # Process through processor to get pixel_values and input_ids
        result = processor(text=[text], images=[image], return_tensors="pt")
        pixel_values = result["pixel_values"]            # [1, num_tiles, 3, H, W]
        pixel_attention_mask = result["pixel_attention_mask"]  # [1, num_tiles, H, W]
        input_ids = result["input_ids"][0].tolist()       # [seq_len]

        if verbose:
            print(f"[SmolVLM Vision] pixel_values: {pixel_values.shape}")
            print(f"[SmolVLM Vision] Token count: {len(input_ids)}")

        # Flatten tiles: [1, num_tiles, C, H, W] → [num_tiles, C, H, W]
        # Send ALL tiles (including padding) so feature count matches
        # the <image> token count in input_ids.  The patch_attention_mask
        # tells the vision encoder which patches are real.
        batch_size, num_images, channels, height, width = pixel_values.shape
        pv_flat = pixel_values.reshape(
            batch_size * num_images, channels, height, width
        ).contiguous()

        # Compute per-tile patch attention mask
        # pixel_attention_mask: [1, num_tiles, H, W] → [num_tiles, H, W]
        mask_flat = pixel_attention_mask.reshape(
            batch_size * num_images, height, width)
        patch_size = vl_config.vision_config.patch_size
        patches_subgrid = mask_flat.unfold(1, patch_size, patch_size)
        patches_subgrid = patches_subgrid.unfold(2, patch_size, patch_size)
        patch_attn_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
        # [num_tiles, patches_h, patches_w]

        if verbose:
            print(f"[SmolVLM Vision] Tiles: {pv_flat.shape[0]}")
            print(f"[SmolVLM Vision] patch_attn_mask: {patch_attn_mask.shape}")

        # Discover vision circuit
        vision_result = self.discovery.build_vision_circuit(vl_model_id)

        if vision_result:
            vision_addresses, vision_pub_keys = vision_result
            if verbose:
                print(f"[SmolVLM Vision] Circuit: {' → '.join(vision_addresses)}")

            entry_address = vision_addresses[0]
            remaining_vision = vision_addresses[1:]

            # Pack pixel_values as bytes
            pv_bytes = pv_flat.contiguous().float().numpy().tobytes()
            pv_shape = list(pv_flat.shape)

            # Pack patch_attention_mask into image_grid_thw (repurposed field)
            mask_list = patch_attn_mask.flatten().int().tolist()
            mask_shape = list(patch_attn_mask.shape)

            vision_request = inference_pb2.ForwardRequest(
                session_id=session_id,
                image_pixels=pv_bytes,
                tensor_shape=pv_shape,
                is_prefill=True,
            )
            vision_request.image_grid_thw.extend(mask_list)
            vision_request.mrope_position_shape.extend(mask_shape)
            vision_request.remaining_circuit.extend(remaining_vision)

            vision_stub = self._get_stub(entry_address)
            vision_response = vision_stub.Forward(vision_request)

            # Extract image features — vision returns [num_tiles, tokens_per_tile, hidden]
            feat_bytes = bytes(vision_response.activation_data)
            feat_shape = list(vision_response.tensor_shape)
            image_features = torch.from_numpy(
                np.frombuffer(feat_bytes, dtype=np.float32)
                .copy().reshape(feat_shape)
            )

            if verbose:
                vision_time = time.time() - vision_start
                print(f"[SmolVLM Vision] Image features (raw): {image_features.shape} "
                      f"({vision_time:.2f}s)")

            # Flatten to 2D: [num_tiles, tok_per_tile, hidden] → [total_img_tokens, hidden]
            # This must match the number of <image> placeholder tokens in input_ids
            if image_features.dim() == 3:
                image_features = image_features.reshape(-1, image_features.shape[-1])

            num_img_tokens = sum(1 for t in input_ids if t == image_token_id)
            if verbose:
                print(f"[SmolVLM Vision] Image features (flat): {image_features.shape}, "
                      f"<image> placeholders: {num_img_tokens}")
        else:
            raise RuntimeError(
                "No vision nodes found for SmolVLM. "
                "Start vision nodes with: python -m node.server "
                "--model-type smolvlm_vision --shard-index 0 --port <port>"
            )

        # === Phase 2: Text ===
        if verbose:
            print(f"[SmolVLM Text] Starting autoregressive generation")
            text_start = time.time()

        # Build text circuit
        text_result = self.discovery.build_circuit(vl_model_id)
        if not text_result:
            raise RuntimeError(
                f"No compute nodes found for model '{vl_model_id}'. "
                f"Start nodes with: python -m node.server "
                f"--model-type smolvlm --shard-index <n> --port <port>"
            )

        text_addresses, text_pub_keys = text_result
        num_shards = len(text_addresses)

        if verbose:
            print(f"[SmolVLM Text] Circuit: {' → '.join(text_addresses)}")

        entry_address = text_addresses[0]
        remaining_text_circuit = text_addresses[1:]
        entry_stub = self._get_stub(entry_address)

        generated_tokens = list(input_ids)
        num_image_tokens = sum(1 for t in input_ids if t == image_token_id)

        for step in range(max_new_tokens):
            step_start = time.time()

            if step == 0:
                # Prefill: full token sequence + image features
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=input_ids,
                    is_prefill=True,
                    num_image_tokens=num_image_tokens,
                )
                request.remaining_circuit.extend(remaining_text_circuit)

                # Attach image features
                ie_bytes = image_features.contiguous().float().numpy().tobytes()
                request.image_embeddings = ie_bytes
                request.image_embeddings_shape.extend(
                    list(image_features.shape))

            else:
                # Decode: single token
                request = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=[generated_tokens[-1]],
                    is_prefill=False,
                )
                request.remaining_circuit.extend(remaining_text_circuit)

            try:
                response = entry_stub.Forward(request)
            except grpc.RpcError as e:
                print(f"\n[Error] gRPC call failed: {e.details()}")
                break

            step_time = time.time() - step_start

            if response.has_token:
                token_id = response.token_id
                generated_tokens.append(token_id)
                token_text = self.tokenizer.decode([token_id])

                if verbose:
                    print(f"  [{step_time:.3f}s] Token {step}: "
                          f"{token_id} -> {token_text!r}")

                yield token_text

                if response.is_eos:
                    break
            else:
                break

        if verbose:
            total_time = time.time() - text_start
            num_generated = len(generated_tokens) - len(input_ids)
            tokens_per_sec = num_generated / total_time if total_time > 0 else 0
            print(f"\n[Stats] {num_generated} tokens in {total_time:.2f}s "
                  f"({tokens_per_sec:.1f} tok/s)")

    def _compute_mrope_position_ids(
        self,
        input_ids: list[int],
        image_grid_thw: torch.Tensor,
        model_id: str,
    ) -> torch.Tensor:
        """Compute 3D M-RoPE position IDs for a multimodal sequence.

        For text tokens: all 3 dimensions (temporal, height, width) have the
        same value (standard 1D positions).

        For image tokens: temporal, height, width positions come from the
        image grid layout.

        Args:
            input_ids: Token IDs (with image placeholders at image_token_id).
            image_grid_thw: Image grid [num_images, 3] with [t, h, w].
            model_id: Model ID for loading config.

        Returns:
            position_ids: Tensor of shape [3, 1, seq_len].
        """
        from transformers import AutoConfig
        vl_config = AutoConfig.from_pretrained(model_id)
        image_token_id = vl_config.image_token_id
        vision_start_id = vl_config.vision_start_token_id
        spatial_merge_size = vl_config.vision_config.spatial_merge_size

        seq_len = len(input_ids)
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long)

        # Walk through the token sequence, tracking position
        text_pos = 0
        image_idx = 0
        i = 0

        while i < seq_len:
            if input_ids[i] == image_token_id:
                # This is an image placeholder token
                # Count how many consecutive image tokens
                img_start = i
                while i < seq_len and input_ids[i] == image_token_id:
                    i += 1
                num_img_tokens = i - img_start

                if image_idx < image_grid_thw.shape[0]:
                    t = image_grid_thw[image_idx, 0].item()
                    h = image_grid_thw[image_idx, 1].item()
                    w = image_grid_thw[image_idx, 2].item()
                    merge = spatial_merge_size

                    # Compute merged grid positions
                    h_merged = h // merge
                    w_merged = w // merge

                    # Generate position IDs for image tokens
                    for ti in range(t):
                        for hi in range(h_merged):
                            for wi in range(w_merged):
                                idx = img_start + ti * h_merged * w_merged + hi * w_merged + wi
                                if idx < i:
                                    position_ids[0, 0, idx] = text_pos + ti
                                    position_ids[1, 0, idx] = text_pos + hi
                                    position_ids[2, 0, idx] = text_pos + wi

                    # Text position advances by max vision position + 1
                    text_pos = text_pos + max(t, h_merged, w_merged)
                    image_idx += 1
                else:
                    # No grid info — just use text position
                    for j in range(img_start, i):
                        position_ids[:, 0, j] = text_pos
                        text_pos += 1
            else:
                # Regular text token: all 3 dims same value
                position_ids[0, 0, i] = text_pos
                position_ids[1, 0, i] = text_pos
                position_ids[2, 0, i] = text_pos
                text_pos += 1
                i += 1

        return position_ids

    def close(self):
        """Close connections."""
        self.discovery.close()
        if self._voter:
            self._voter.close()
        if self._racer:
            self._racer.close()


def list_models_command(registry_address: str = None):
    """Query registries and print available models."""
    if registry_address:
        pool = RegistryPool([registry_address])
    else:
        pool = RegistryPool()

    healthy = pool.find_healthy_registry()
    if not healthy:
        print("No reachable registries found.")
        print(f"Tried: {', '.join(pool._addresses)}")
        return

    models = pool.list_models()
    if not models:
        print("No models currently available on the network.")
        return

    print(f"Available models ({len(models)}):\n")
    for m in sorted(models, key=lambda x: x.model_id):
        status = "READY" if m.can_serve else "INCOMPLETE"
        coverage = f"{m.covered_shards}/{m.total_shards} shards"
        print(f"  {m.model_id}")
        print(f"    Nodes: {m.total_nodes}  |  Shards: {coverage}  |  Status: {status}")
    print()
    pool.close()


def main():
    parser = argparse.ArgumentParser(description="UNFED AI Client")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=config.MAX_NEW_TOKENS,
                        help="Max tokens to generate")
    parser.add_argument("--verbose", action="store_true", help="Print timing info")
    parser.add_argument("--no-onion", action="store_true",
                        help="Disable onion routing (use plain circuit)")
    parser.add_argument("--random-routing", action="store_true",
                        help="Use computational randomness routing")
    parser.add_argument("--use-voting", action="store_true",
                        help="Enable redundant voting (double-check 1 random shard)")
    parser.add_argument("--use-return-encryption", action="store_true",
                        help="Enable return-path encryption")
    parser.add_argument("--racing", action="store_true",
                        help="Use per-shard racing for fault tolerance "
                             "(client-driven hop-by-hop with N replicas)")
    parser.add_argument("--registry", type=str, default=None,
                        help=f"Registry address (default: seed list)")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model to use (default: {config.MODEL_NAME})")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models on the network and exit")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file (enables multimodal VL mode)")
    parser.add_argument("--pipelined-prefill", action="store_true",
                        help="Enable pipelined prefill for long prompts (racing mode)")
    parser.add_argument("--prefill-chunk-size", type=int, default=None,
                        help=f"Chunk size for pipelined prefill (default: {config.PREFILL_CHUNK_SIZE})")
    parser.add_argument("--tls-ca", type=str, default=None,
                        help="Path to CA certificate PEM for TLS connections")
    args = parser.parse_args()

    if args.prefill_chunk_size is not None:
        config.PREFILL_CHUNK_SIZE = args.prefill_chunk_size
    if args.pipelined_prefill:
        config.PREFILL_PIPELINE_MIN_TOKENS = 1

    # List models mode
    if args.list_models:
        list_models_command(args.registry)
        return

    model_id = args.model or config.MODEL_NAME

    client = UnfedClient(
        registry_address=args.registry,
        use_voting=args.use_voting,
        use_return_encryption=args.use_return_encryption,
        tls_ca=args.tls_ca,
        use_racing=args.racing,
        model_id=model_id,
    )

    use_onion = not args.no_onion and not args.random_routing
    use_random = args.random_routing

    # Multimodal mode
    if args.image and args.prompt:
        print(f"[Image: {args.image}]")
        print(f"Q: {args.prompt}")
        print("A: ", end="", flush=True)

        # Detect model type to choose correct multimodal pipeline
        is_smolvlm = "smolvlm" in model_id.lower() or "smol" in model_id.lower()
        if is_smolvlm:
            gen = client.generate_multimodal_smolvlm(
                args.prompt, args.image,
                max_new_tokens=args.max_tokens,
                verbose=args.verbose, model_id=model_id)
        else:
            gen = client.generate_multimodal(
                args.prompt, args.image,
                max_new_tokens=args.max_tokens,
                verbose=args.verbose, model_id=model_id)
        for token_text in gen:
            print(token_text, end="", flush=True)
        print()
        client.close()
        return

    if args.prompt:
        # Single prompt mode
        print(f"{args.prompt}", end="", flush=True)
        for token_text in client.generate(args.prompt, max_new_tokens=args.max_tokens,
                                          verbose=args.verbose, use_onion=use_onion,
                                          use_random_routing=use_random):
            print(token_text, end="", flush=True)
        print()
    else:
        # Interactive mode
        print("UNFED AI — Distributed Inference")
        print(f"Model: {client.model_id}")
        print(f"Registry: {args.registry or 'seed list'}")

        features = []
        if args.use_voting:
            features.append("voting")
        if args.use_return_encryption:
            features.append("return-encryption")
        if args.racing:
            features.append("per-shard-racing")
        if features:
            print(f"Privacy: {', '.join(features)}")

        # Show pool health
        health = client.discovery.get_pool_health(client.model_id)
        if health:
            print(f"Pool: {health.overall_status} ({health.total_shards} shards, "
                  f"{'can serve' if health.can_serve else 'CANNOT serve'})")
            for s in health.shards:
                print(f"  Shard {s.shard_index} (layers {s.layer_start}-{s.layer_end - 1}): "
                      f"{s.node_count} nodes [{s.status}]")
        else:
            print("Pool: unknown (registry not reachable)")

        print("\nType 'quit' to exit.\n")

        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if prompt.strip().lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if not prompt.strip():
                continue

            for token_text in client.generate(prompt, max_new_tokens=args.max_tokens,
                                              verbose=args.verbose, use_onion=use_onion,
                                              use_random_routing=use_random):
                print(token_text, end="", flush=True)
            print()

    client.close()


if __name__ == "__main__":
    main()
