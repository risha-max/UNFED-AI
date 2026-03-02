import pytest
from types import SimpleNamespace

import network.pipeline as pipeline_mod


class _DummyTokenizer:
    def __call__(self, prompt, return_tensors="pt"):
        class _Ids:
            def __getitem__(self, _idx):
                return SimpleNamespace(tolist=lambda: [1, 2, 3])
        return SimpleNamespace(input_ids=_Ids())

    def decode(self, token_ids):
        return "x"


class _FakeDiscoveryNoMpc:
    def discover(self, model_id: str):
        nodes = [
            SimpleNamespace(shard_index=0, node_type="compute"),
            SimpleNamespace(shard_index=1, node_type="compute"),
        ]
        if not model_id:
            nodes.append(SimpleNamespace(shard_index=-1, node_type="daemon"))
        return nodes

    def build_circuit(self, model_id: str):
        return (["node-a:50051", "node-b:50052"], [b"pk-a", b"pk-b"])

    def get_verifier_health(self):
        return SimpleNamespace(
            healthy_verifier_count=1,
            required_verifier_count=1,
            healthy=True,
        )

    def close(self):
        return None


def test_pipeline_rejects_without_mpc_when_required(monkeypatch):
    monkeypatch.delenv("UNFED_REQUIRE_MPC", raising=False)
    monkeypatch.setattr(
        pipeline_mod.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(pipeline_mod, "RegistryClient", lambda *_: _FakeDiscoveryNoMpc())
    scheduler = pipeline_mod.PipelineScheduler(max_concurrent=1)
    with pytest.raises(RuntimeError, match="missing MPC shard-0 entry"):
        list(scheduler.generate_single("hello"))
