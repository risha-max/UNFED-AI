"""
Shared pytest fixtures for UNFED AI unit tests.
"""

import os
import sys
import time

import pytest
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hidden_size():
    return 64


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def head_dim(hidden_size, num_heads):
    return hidden_size // num_heads


@pytest.fixture
def intermediate_size(hidden_size):
    return hidden_size * 4


@pytest.fixture
def seq_len():
    return 8


@pytest.fixture
def batch():
    return 1


@pytest.fixture
def sample_hidden(batch, seq_len, hidden_size):
    """Random hidden state tensor."""
    return torch.randn(batch, seq_len, hidden_size)


@pytest.fixture
def sample_attn_scores(batch, num_heads, seq_len):
    """Random attention score tensor."""
    return torch.randn(batch, num_heads, seq_len, seq_len)


@pytest.fixture
def sample_mlp(batch, seq_len, intermediate_size):
    """Random MLP intermediate tensor."""
    return torch.randn(batch, seq_len, intermediate_size)


# ---------------------------------------------------------------------------
# Share chain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_shares():
    """Generate sample compute shares for chain tests."""
    from economics.share_chain import ComputeShare

    shares = []
    for i in range(5):
        shares.append(ComputeShare(
            node_id=f"node_{i % 3}",
            shard_index=i % 4,
            session_id=f"session_{i}",
            activation_hash=f"hash_{i}",
            tokens_processed=10,
            timestamp=time.time(),
            share_weight=1.0,
        ))
    return shares


# ---------------------------------------------------------------------------
# Payment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stake_manager():
    from economics.payments import StakeManager
    return StakeManager(min_stake=100.0, slash_fraction=0.5)


@pytest.fixture
def payment_contract(stake_manager):
    from economics.payments import PaymentContract
    return PaymentContract(
        stake_manager,
        challenge_window=0.1,  # fast for tests
        price_per_input_token=0.0001,
        price_per_output_token=0.001,
    )
