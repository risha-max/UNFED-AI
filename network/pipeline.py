"""
Pipeline Parallelism — batch queries from multiple users so all nodes
stay busy simultaneously.

The problem:
  In sequential inference, only one node is active at a time.
  While Node 0 processes tokens, Nodes 1-3 are idle.
  This wastes 75% of the pipeline's capacity.

The solution:
  Pipeline parallelism overlaps different stages:
  - While Node 0 processes query B's prefill,
    Node 1 is processing query A's activations from its prefill,
    Node 2 processes an earlier query's activations, etc.

Implementation:
  The PipelineScheduler manages multiple concurrent sessions,
  feeding them through the pipeline in a staggered fashion.

  For N nodes and M concurrent queries:
  - Step 1: Node 0 processes Query 1
  - Step 2: Node 0 processes Query 2, Node 1 processes Query 1
  - Step 3: Node 0 processes Query 3, Node 1 processes Query 2, Node 2 processes Query 1
  - ... etc.

  Once the pipeline is full, all N nodes are busy every step,
  achieving N-fold throughput improvement.

  This is orchestrated client-side: the client sends multiple sessions
  in parallel, and the pipeline naturally fills up because gRPC calls
  are concurrent.
"""

import sys
import os
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import grpc
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "proto"))
import config
import inference_pb2
import inference_pb2_grpc
from network.discovery import RegistryClient
from network.resilience import create_resilient_channel


class PipelineScheduler:
    """
    Schedules multiple inference sessions through the pipeline concurrently.

    Instead of processing one query at a time (leaving most nodes idle),
    the scheduler overlaps queries so all pipeline stages stay busy.
    """

    def __init__(self, registry_address: str = None, max_concurrent: int = 4):
        """
        Args:
            registry_address: Registry server address
            max_concurrent: Maximum number of concurrent queries in the pipeline
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.discovery = RegistryClient(registry_address)
        self.max_concurrent = max_concurrent
        self._stubs: dict[str, inference_pb2_grpc.InferenceNodeStub] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent * 2)

    def _get_stub(self, address: str) -> inference_pb2_grpc.InferenceNodeStub:
        if address not in self._stubs:
            channel = create_resilient_channel(address, config.GRPC_OPTIONS)
            self._stubs[address] = inference_pb2_grpc.InferenceNodeStub(channel)
        return self._stubs[address]

    def generate_single(self, prompt: str, max_new_tokens: int = 100,
                        circuit: list[str] = None) -> Iterator[str]:
        """Generate text for a single prompt (used internally)."""
        session_id = str(uuid.uuid4())
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

        if circuit is None:
            result = self.discovery.build_circuit(config.MODEL_NAME)
            if not result:
                return
            circuit, _ = result

        entry = circuit[0]
        remaining = circuit[1:]
        stub = self._get_stub(entry)
        generated = list(input_ids)

        for step in range(max_new_tokens):
            if step == 0:
                req = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=input_ids,
                    is_prefill=True,
                    remaining_circuit=remaining,
                )
            else:
                req = inference_pb2.ForwardRequest(
                    session_id=session_id,
                    token_ids=[generated[-1]],
                    is_prefill=False,
                    remaining_circuit=remaining,
                )

            try:
                resp = stub.Forward(req)
            except grpc.RpcError:
                break

            if resp.has_token:
                generated.append(resp.token_id)
                yield self.tokenizer.decode([resp.token_id])
                if resp.is_eos:
                    break
            else:
                break

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 100,
                       callback=None) -> list[str]:
        """
        Generate text for multiple prompts concurrently.

        Overlaps queries in the pipeline for maximum throughput.

        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens per prompt
            callback: Optional function called with (prompt_index, token_text)
                      as tokens are generated

        Returns:
            List of complete generated texts
        """
        results = [""] * len(prompts)
        locks = [threading.Lock() for _ in prompts]

        # Build a shared circuit for all queries
        result = self.discovery.build_circuit(config.MODEL_NAME)
        if not result:
            print("[Pipeline] Cannot build circuit")
            return results
        circuit, _ = result

        def _run_one(idx: int, prompt: str):
            tokens = []
            for token_text in self.generate_single(
                    prompt, max_new_tokens, circuit):
                tokens.append(token_text)
                if callback:
                    callback(idx, token_text)
            with locks[idx]:
                results[idx] = prompt + "".join(tokens)

        # Submit all prompts concurrently (up to max_concurrent at a time)
        futures = []
        for i, prompt in enumerate(prompts):
            future = self._executor.submit(_run_one, i, prompt)
            futures.append(future)

        # Wait for all to complete
        for future in as_completed(futures):
            future.result()  # raises any exception

        return results

    def benchmark(self, prompts: list[str], max_new_tokens: int = 20) -> dict:
        """
        Benchmark pipeline throughput with concurrent queries.

        Returns performance statistics.
        """
        start = time.time()
        total_tokens = [0]

        def count_callback(idx, token_text):
            total_tokens[0] += 1

        results = self.generate_batch(prompts, max_new_tokens, count_callback)

        elapsed = time.time() - start
        tokens = total_tokens[0]
        tok_per_sec = tokens / elapsed if elapsed > 0 else 0

        return {
            "num_prompts": len(prompts),
            "total_tokens": tokens,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(tok_per_sec, 1),
            "avg_tokens_per_prompt": round(tokens / max(len(prompts), 1), 1),
        }

    def close(self):
        self.discovery.close()
        self._executor.shutdown(wait=False)
