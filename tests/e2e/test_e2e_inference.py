"""E2E tests for the full inference pipeline."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.mark.e2e
class TestInference:

    def test_single_token_generation(self, client):
        tokens = []
        for token in client.generate("Hello", max_new_tokens=1):
            tokens.append(token)
        assert len(tokens) >= 1
        assert tokens[0] != ""

    def test_multi_token_generation(self, client):
        tokens = list(client.generate(
            "The capital of France is", max_new_tokens=20))
        assert len(tokens) >= 10

    def test_streaming_output(self, client):
        first_time = None
        last_time = None
        count = 0
        for token in client.generate("Once upon a time", max_new_tokens=10):
            now = time.time()
            if first_time is None:
                first_time = now
            last_time = now
            count += 1
        assert count >= 5
        assert last_time > first_time, "Tokens should arrive over time, not all at once"

    def test_concurrent_sessions(self, client, cluster):
        from client.client import UnfedClient

        prompts = [
            "The sky is",
            "Water boils at",
            "Python is a",
        ]
        results = [None] * len(prompts)

        def run_one(idx):
            c = UnfedClient(registry_address=cluster.registry_addr)
            try:
                results[idx] = list(c.generate(prompts[idx], max_new_tokens=5))
            finally:
                c.close()

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run_one, range(len(prompts))))

        for i, r in enumerate(results):
            assert r is not None, f"Session {i} returned None"
            assert len(r) >= 1, f"Session {i} generated no tokens"

    def test_model_listing(self, client):
        from network.discovery import RegistryPool
        pool = RegistryPool([client.discovery._registries[0]])
        models = pool.list_models()
        assert len(models) >= 1
        model_ids = [m.model_id for m in models]
        assert any("Qwen" in mid for mid in model_ids)
