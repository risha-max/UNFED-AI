import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import registry_pb2
from network.registry_server import RegistryServicer
from network.share_auth import (
    generate_signing_keypair,
    registration_pop_payload,
    sign_bytes,
)


def _register_node(
    svc: RegistryServicer,
    *,
    node_id: str,
    model_id: str,
    shard_index: int,
    node_type: str,
):
    priv, pub = generate_signing_keypair()
    address = f"localhost:{50050 + shard_index}"
    req = registry_pb2.RegisterRequest(
        node_id=node_id,
        address=address,
        model_id=model_id,
        shard_index=shard_index,
        layer_start=shard_index,
        layer_end=shard_index + 1,
        has_embedding=(shard_index == 0),
        has_lm_head=False,
        node_type=node_type,
        share_signing_public_key=pub,
        share_signing_pop=sign_bytes(
            priv,
            registration_pop_payload(
                node_id=node_id,
                address=address,
                model_id=model_id,
                shard_index=shard_index,
                node_type=node_type,
            ),
        ),
    )
    resp = svc.Register(req, None)
    assert resp.success is True


def test_list_models_requires_mpc_when_policy_enabled(monkeypatch):
    monkeypatch.setenv("UNFED_REQUIRE_MPC", "1")
    svc = RegistryServicer(no_chain=True)
    _register_node(
        svc, node_id="0x1000000000000000000000000000000000000001",
        model_id="model-x", shard_index=0, node_type="compute",
    )
    _register_node(
        svc, node_id="0x1000000000000000000000000000000000000002",
        model_id="model-x", shard_index=1, node_type="compute",
    )
    resp = svc.ListModels(registry_pb2.ListModelsRequest(), None)
    model = next(m for m in resp.models if m.model_id == "model-x")
    assert model.covered_shards == 2
    assert model.total_shards == 2
    assert model.can_serve is False


def test_list_models_can_serve_with_mpc_present(monkeypatch):
    monkeypatch.setenv("UNFED_REQUIRE_MPC", "1")
    svc = RegistryServicer(no_chain=True)
    _register_node(
        svc, node_id="0x2000000000000000000000000000000000000001",
        model_id="model-y", shard_index=0, node_type="mpc",
    )
    _register_node(
        svc, node_id="0x2000000000000000000000000000000000000002",
        model_id="model-y", shard_index=1, node_type="compute",
    )
    resp = svc.ListModels(registry_pb2.ListModelsRequest(), None)
    model = next(m for m in resp.models if m.model_id == "model-y")
    assert model.covered_shards == 2
    assert model.total_shards == 2
    assert model.can_serve is True
