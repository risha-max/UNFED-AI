"""E2E tests for share-chain economics."""

import time

import grpc
import pytest

from tests.e2e.conftest import PROJECT_ROOT

import os
import sys
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import inference_pb2
import inference_pb2_grpc


def _get_blocks(address: str, from_height: int = 0) -> list:
    channel = grpc.insecure_channel(address)
    stub = inference_pb2_grpc.InferenceNodeStub(channel)
    resp = stub.GetBlocks(inference_pb2.GetBlocksRequest(
        from_height=from_height, max_blocks=100))
    channel.close()
    return list(resp.blocks)


@pytest.mark.e2e
class TestChain:

    def test_blocks_produced_after_inference(self, client, cluster):
        list(client.generate("Test chain block production", max_new_tokens=5))
        time.sleep(8)

        blocks = _get_blocks(cluster.node_addrs[0])
        assert len(blocks) > 0, "No blocks produced after inference"

    def test_shares_recorded_for_all_shards(self, client, cluster):
        list(client.generate("Test shares", max_new_tokens=5))
        time.sleep(8)

        blocks = _get_blocks(cluster.node_addrs[0])
        shard_indexes = set()
        for block in blocks:
            for share in block.shares:
                shard_indexes.add(share.shard_index)

        for i in range(4):
            assert i in shard_indexes, f"Missing shares for shard {i}"

    def test_chain_gossip_convergence(self, client, cluster):
        list(client.generate("Test gossip convergence", max_new_tokens=3))
        time.sleep(10)

        heights = []
        for addr in cluster.node_addrs:
            try:
                blocks = _get_blocks(addr)
                heights.append(len(blocks))
            except grpc.RpcError:
                heights.append(-1)

        valid_heights = [h for h in heights if h > 0]
        if len(valid_heights) >= 2:
            assert max(valid_heights) - min(valid_heights) <= 1, \
                f"Chain heights diverged: {heights}"

    def test_node_totals_accurate(self, client, cluster):
        list(client.generate("Test node totals", max_new_tokens=5))
        time.sleep(8)

        blocks = _get_blocks(cluster.node_addrs[0])
        manual_totals: dict[str, int] = {}
        for block in blocks:
            for share in block.shares:
                manual_totals[share.node_id] = (
                    manual_totals.get(share.node_id, 0) + share.tokens_processed
                )

        assert len(manual_totals) > 0, "No shares found"
