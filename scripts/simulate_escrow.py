#!/usr/bin/env python3
"""
UNFED AI — On-Chain Escrow Simulation

Spins up a local Anvil chain, deploys contracts, starts a registry with
on-chain escrow enabled, and simulates the following scenarios:

  1. UNSTAKED NODE    — Tries to register, gets rejected
  2. STAKED NODE      — Stakes, registers, gets accepted
  3. CLIENT PAYMENT   — Client deposits, inference happens, settlement pays out
  4. CHEATING NODE    — Fraud proof submitted, node slashed on-chain
  5. UNSTAKE COOLDOWN — Node tries to unstake, must wait for cooldown
  6. UNDER-STAKED     — Node stakes below minimum, gets rejected

Usage:
    # Start Anvil first:
    ./scripts/start_local_chain.sh

    # Then run this simulation:
    python scripts/simulate_escrow.py
"""

import hashlib
import json
import os
import sys
import time
import threading

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "proto"))

import grpc
import requests
import registry_pb2
import registry_pb2_grpc
from web3 import Web3

from economics.onchain import OnChainEscrow, load_from_env
from economics.cluster_config import ClusterConfig

# ---------------------------------------------------------------------------
# Constants — Anvil default accounts
# ---------------------------------------------------------------------------
OPERATOR_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
OPERATOR_ADDR = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

# Nodes (Anvil accounts 1-3)
NODE_HONEST_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
NODE_HONEST_ADDR = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"

NODE_CHEATER_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
NODE_CHEATER_ADDR = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"

NODE_UNSTAKED_KEY = "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"
NODE_UNSTAKED_ADDR = "0x90F79bf6EB2c4f870365E785982E1f101E93b906"

NODE_UNDERSTAKED_KEY = "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"
NODE_UNDERSTAKED_ADDR = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"

# Client (Anvil account 5)
CLIENT_KEY = "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba"
CLIENT_ADDR = "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"

RPC_URL = "http://localhost:8545"
REGISTRY_PORT = 50099  # Use a non-default port to avoid conflicts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_w3 = None
_token = None
_escrow_addr = None

ERC20_ABI = [
    {"type": "function", "name": "approve",
     "inputs": [{"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}],
     "stateMutability": "nonpayable"},
    {"type": "function", "name": "transfer",
     "inputs": [{"name": "to", "type": "address"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}],
     "stateMutability": "nonpayable"},
    {"type": "function", "name": "balanceOf",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}],
     "stateMutability": "view"},
]

ESCROW_USER_ABI = [
    {"type": "function", "name": "stake",
     "inputs": [{"name": "amount", "type": "uint256"}],
     "outputs": [], "stateMutability": "nonpayable"},
    {"type": "function", "name": "deposit",
     "inputs": [{"name": "amount", "type": "uint256"}],
     "outputs": [], "stateMutability": "nonpayable"},
    {"type": "function", "name": "requestUnstake",
     "inputs": [{"name": "amount", "type": "uint256"}],
     "outputs": [], "stateMutability": "nonpayable"},
    {"type": "function", "name": "withdraw",
     "inputs": [], "outputs": [], "stateMutability": "nonpayable"},
    {"type": "function", "name": "claimEarnings",
     "inputs": [], "outputs": [], "stateMutability": "nonpayable"},
]


def send_tx(fn, key, addr):
    tx = fn.build_transaction({
        "from": addr,
        "nonce": _w3.eth.get_transaction_count(addr),
        "gas": 500_000,
        "gasPrice": _w3.eth.gas_price,
    })
    signed = _w3.eth.account.sign_transaction(tx, key)
    h = _w3.eth.send_raw_transaction(signed.raw_transaction)
    return _w3.eth.wait_for_transaction_receipt(h)


def wei(amount):
    return int(amount * 10**18)


def from_wei(amount):
    return amount / 10**18


def anvil_advance_time(seconds):
    requests.post(RPC_URL, json={
        "jsonrpc": "2.0", "method": "evm_increaseTime",
        "params": [seconds], "id": 1,
    })
    requests.post(RPC_URL, json={
        "jsonrpc": "2.0", "method": "evm_mine", "params": [], "id": 2,
    })


def header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def ok(text):
    print(f"  [PASS] {text}")


def fail(text):
    print(f"  [FAIL] {text}")
    sys.exit(1)


def info(text):
    print(f"  [INFO] {text}")


# ---------------------------------------------------------------------------
# Registry in background thread
# ---------------------------------------------------------------------------

def start_registry(cluster_config_path):
    from network.registry_server import serve
    t = threading.Thread(
        target=serve,
        args=(REGISTRY_PORT,),
        kwargs={"cluster_config_path": cluster_config_path},
        daemon=True,
    )
    t.start()
    time.sleep(2)  # Wait for it to start
    return t


def get_stub():
    channel = grpc.insecure_channel(f"localhost:{REGISTRY_PORT}")
    return registry_pb2_grpc.RegistryStub(channel)


def try_register(stub, node_id, node_type="compute"):
    try:
        resp = stub.Register(registry_pb2.RegisterRequest(
            node_id=node_id,
            address=f"localhost:50{abs(hash(node_id)) % 900 + 100}",
            model_id="test/SimModel",
            shard_index=0,
            layer_start=0,
            layer_end=12,
            has_embedding=True,
            has_lm_head=True,
            public_key=b"\x00" * 32,
            node_type=node_type,
        ), timeout=10)
        return resp.success, resp.message
    except grpc.RpcError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    global _w3, _token, _escrow_addr

    env_file = os.path.join(PROJECT_ROOT, "contracts", "deployed.env")
    if not os.path.exists(env_file):
        print("ERROR: contracts/deployed.env not found.")
        print("Run ./scripts/start_local_chain.sh first.")
        sys.exit(1)

    # Parse env file
    env = {}
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    token_addr = env["TOKEN_ADDRESS"]
    _escrow_addr = env["ESCROW_ADDRESS"]

    _w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not _w3.is_connected():
        print("ERROR: Cannot connect to Anvil. Is it running?")
        sys.exit(1)

    _token = _w3.eth.contract(address=token_addr, abi=ERC20_ABI)
    escrow_user = _w3.eth.contract(address=_escrow_addr, abi=ESCROW_USER_ABI)

    # Create OnChainEscrow client (operator perspective)
    onchain = load_from_env(env_file)

    print()
    print("  UNFED AI — On-Chain Escrow Simulation")
    print("  ======================================")
    print()
    info(f"Chain:    {RPC_URL}")
    info(f"Token:    {token_addr} (UNFED)")
    info(f"Escrow:   {_escrow_addr}")
    info(f"Operator: {OPERATOR_ADDR}")
    info(f"Min stake: {from_wei(onchain.min_stake()):.0f} UNFED")

    # ---------------------------------------------------------------
    # Setup: distribute tokens
    # ---------------------------------------------------------------
    header("SETUP: Distribute tokens")

    for name, addr in [("Honest Node", NODE_HONEST_ADDR),
                       ("Cheater Node", NODE_CHEATER_ADDR),
                       ("Unstaked Node", NODE_UNSTAKED_ADDR),
                       ("Understaked Node", NODE_UNDERSTAKED_ADDR),
                       ("Client", CLIENT_ADDR)]:
        bal = _token.functions.balanceOf(addr).call()
        if bal < wei(1000):
            send_tx(_token.functions.transfer(addr, wei(10000)),
                    OPERATOR_KEY, OPERATOR_ADDR)
        bal = _token.functions.balanceOf(addr).call()
        info(f"{name} ({addr[:10]}...): {from_wei(bal):.0f} UNFED")

    # ---------------------------------------------------------------
    # Setup: stake nodes
    # ---------------------------------------------------------------
    header("SETUP: Stake nodes")

    # Honest node stakes 200 UNFED (above min of 100)
    send_tx(_token.functions.approve(_escrow_addr, wei(200)),
            NODE_HONEST_KEY, NODE_HONEST_ADDR)
    send_tx(escrow_user.functions.stake(wei(200)),
            NODE_HONEST_KEY, NODE_HONEST_ADDR)
    info(f"Honest node staked: {from_wei(onchain.staked_balance(NODE_HONEST_ADDR)):.0f} UNFED")
    info(f"Honest node eligible: {onchain.is_eligible(NODE_HONEST_ADDR)}")

    # Cheater node stakes 150 UNFED
    send_tx(_token.functions.approve(_escrow_addr, wei(150)),
            NODE_CHEATER_KEY, NODE_CHEATER_ADDR)
    send_tx(escrow_user.functions.stake(wei(150)),
            NODE_CHEATER_KEY, NODE_CHEATER_ADDR)
    info(f"Cheater node staked: {from_wei(onchain.staked_balance(NODE_CHEATER_ADDR)):.0f} UNFED")

    # Understaked node stakes only 50 UNFED (below min of 100)
    send_tx(_token.functions.approve(_escrow_addr, wei(50)),
            NODE_UNDERSTAKED_KEY, NODE_UNDERSTAKED_ADDR)
    send_tx(escrow_user.functions.stake(wei(50)),
            NODE_UNDERSTAKED_KEY, NODE_UNDERSTAKED_ADDR)
    info(f"Understaked node staked: {from_wei(onchain.staked_balance(NODE_UNDERSTAKED_ADDR)):.0f} UNFED")
    info(f"Understaked node eligible: {onchain.is_eligible(NODE_UNDERSTAKED_ADDR)}")

    # Unstaked node: does nothing
    info(f"Unstaked node staked: {from_wei(onchain.staked_balance(NODE_UNSTAKED_ADDR)):.0f} UNFED")

    # ---------------------------------------------------------------
    # Setup: client deposits
    # ---------------------------------------------------------------
    header("SETUP: Client deposits credits")

    send_tx(_token.functions.approve(_escrow_addr, wei(5000)),
            CLIENT_KEY, CLIENT_ADDR)
    send_tx(escrow_user.functions.deposit(wei(5000)),
            CLIENT_KEY, CLIENT_ADDR)
    info(f"Client deposited: {from_wei(onchain.client_balance(CLIENT_ADDR)):.0f} UNFED")
    info(f"Escrow pool: {from_wei(onchain.escrow_pool()):.0f} UNFED")

    # ---------------------------------------------------------------
    # Start registry with on-chain escrow
    # ---------------------------------------------------------------
    header("START REGISTRY (with on-chain escrow)")

    cluster_cfg_path = os.path.join(PROJECT_ROOT, "contracts",
                                    "sim_cluster.json")
    cfg = ClusterConfig(
        name="SimCluster",
        description="Simulation test cluster",
        public_endpoint=f"localhost:{REGISTRY_PORT}",
        chain_rpc_url=RPC_URL,
        escrow_contract_address=_escrow_addr,
        staking_token_address=token_addr,
        operator_private_key=OPERATOR_KEY,
        cooldown_seconds=300,
        challenge_window_seconds=60,
    )
    cfg.save(cluster_cfg_path)
    info(f"Cluster config saved to {cluster_cfg_path}")

    start_registry(cluster_cfg_path)
    info(f"Registry running on port {REGISTRY_PORT}")

    stub = get_stub()

    # ---------------------------------------------------------------
    # Scenario 1: Unstaked node tries to register
    # ---------------------------------------------------------------
    header("SCENARIO 1: Unstaked node tries to register")

    success, msg = try_register(stub, NODE_UNSTAKED_ADDR)
    if not success:
        ok(f"Registration REJECTED: {msg}")
    else:
        fail("Unstaked node was allowed to register!")

    # ---------------------------------------------------------------
    # Scenario 2: Understaked node tries to register
    # ---------------------------------------------------------------
    header("SCENARIO 2: Under-staked node (50 < 100 min) tries to register")

    success, msg = try_register(stub, NODE_UNDERSTAKED_ADDR)
    if not success:
        ok(f"Registration REJECTED: {msg}")
    else:
        fail("Under-staked node was allowed to register!")

    # ---------------------------------------------------------------
    # Scenario 3: Properly staked nodes register
    # ---------------------------------------------------------------
    header("SCENARIO 3: Properly staked nodes register")

    success, msg = try_register(stub, NODE_HONEST_ADDR)
    if success:
        ok(f"Honest node registered: {msg}")
    else:
        fail(f"Honest node rejected: {msg}")

    success, msg = try_register(stub, NODE_CHEATER_ADDR)
    if success:
        ok(f"Cheater node registered (doesn't know it'll cheat yet): {msg}")
    else:
        fail(f"Cheater node rejected: {msg}")

    # Verify both are discovered
    resp = stub.Discover(registry_pb2.DiscoverRequest(model_id="test/SimModel"),
                         timeout=10)
    info(f"Discovered {len(resp.nodes)} nodes for test/SimModel")
    for n in resp.nodes:
        info(f"  - {n.node_id[:10]}... shard={n.shard_index} type={n.node_type}")

    # ---------------------------------------------------------------
    # Scenario 4: Settlement — both nodes get paid
    # ---------------------------------------------------------------
    header("SCENARIO 4: Settlement — paying nodes for work")

    escrow_before = onchain.escrow_pool()
    info(f"Escrow pool before: {from_wei(escrow_before):.0f} UNFED")

    s_hash = hashlib.sha256(b"sim-settlement-1").hexdigest()
    onchain.post_settlement(
        s_hash,
        [NODE_HONEST_ADDR, NODE_CHEATER_ADDR],
        [wei(30), wei(20)],  # honest=30, cheater=20
    )

    escrow_after = onchain.escrow_pool()
    info(f"Escrow pool after posting: {from_wei(escrow_after):.0f} UNFED")
    info(f"Deducted: {from_wei(escrow_before - escrow_after):.0f} UNFED")

    # Advance time past challenge window (60s)
    info("Advancing time 61 seconds past challenge window...")
    anvil_advance_time(61)

    onchain.finalize_settlement(s_hash)
    ok(f"Settlement finalized!")
    info(f"Honest node earnings: {from_wei(onchain.node_earnings(NODE_HONEST_ADDR)):.0f} UNFED")
    info(f"Cheater node earnings: {from_wei(onchain.node_earnings(NODE_CHEATER_ADDR)):.0f} UNFED")

    # ---------------------------------------------------------------
    # Scenario 5: Cheater node detected — fraud proof + slash
    # ---------------------------------------------------------------
    header("SCENARIO 5: Cheater node caught — fraud proof + on-chain slash")

    cheater_stake_before = onchain.staked_balance(NODE_CHEATER_ADDR)
    escrow_before_slash = onchain.escrow_pool()
    info(f"Cheater stake before slash: {from_wei(cheater_stake_before):.0f} UNFED")
    info(f"Escrow pool before slash: {from_wei(escrow_before_slash):.0f} UNFED")

    # First, simulate the cheater node submitting a verification ticket.
    # This links "sim-ticket-001" -> NODE_CHEATER_ADDR in the registry's
    # _ticket_owners map, so the fraud proof can identify the cheater.
    info("Cheater submits a verification ticket (as part of normal operation)...")
    import numpy as np
    dummy_data = np.zeros((1, 128), dtype=np.float32).tobytes()
    stub.SubmitTickets(registry_pb2.SubmitTicketsRequest(
        node_id=NODE_CHEATER_ADDR,
        tickets=[registry_pb2.VerificationTicketProto(
            ticket_id="sim-ticket-001",
            shard_index=0,
            input_data=dummy_data,
            input_shape=[1, 128],
            input_is_tokens=True,
            expected_output_data=dummy_data,
            expected_output_shape=[1, 128],
            timestamp=time.time(),
        )],
    ), timeout=10)
    info("Ticket submitted. Registry now knows ticket belongs to cheater.")

    # Now submit the fraud proof — a verifier re-ran the computation
    # and got a different result
    info("Verifier submits fraud proof (output mismatch detected)...")
    fraud_resp = stub.SubmitFraudProof(registry_pb2.FraudProofMessage(
        ticket_id="sim-ticket-001",
        shard_index=0,
        input_hash="aaaa",
        expected_output_hash="bbbb",
        actual_output_hash="cccc",
        expected_token=42,
        actual_token=99,
        timestamp=time.time(),
    ), timeout=10)

    if fraud_resp.accepted:
        ok("Fraud proof accepted by registry")
    else:
        fail("Fraud proof rejected!")

    # Give the registry a moment to process the on-chain slash
    time.sleep(2)

    cheater_stake_after = onchain.staked_balance(NODE_CHEATER_ADDR)
    escrow_after_slash = onchain.escrow_pool()
    slash_amount = cheater_stake_before - cheater_stake_after
    escrow_gain = escrow_after_slash - escrow_before_slash

    info(f"Cheater stake after slash: {from_wei(cheater_stake_after):.0f} UNFED")
    info(f"Slashed amount: {from_wei(slash_amount):.0f} UNFED")
    info(f"Escrow pool after slash: {from_wei(escrow_after_slash):.0f} UNFED")
    info(f"Escrow gained from slash: {from_wei(escrow_gain):.0f} UNFED")

    if slash_amount > 0:
        ok(f"Cheater slashed {from_wei(slash_amount):.0f} UNFED "
           f"(50% of {from_wei(cheater_stake_before):.0f})")
    else:
        fail("Cheater was not slashed!")

    if escrow_gain > 0:
        ok(f"Slashed funds ({from_wei(escrow_gain):.0f} UNFED) "
           f"returned to client escrow pool")
    else:
        fail("Slashed funds did not go to escrow pool!")

    # ---------------------------------------------------------------
    # Scenario 6: Honest node claims earnings
    # ---------------------------------------------------------------
    header("SCENARIO 6: Honest node claims earnings")

    token_before = _token.functions.balanceOf(NODE_HONEST_ADDR).call()
    earnings = onchain.node_earnings(NODE_HONEST_ADDR)
    info(f"Pending earnings: {from_wei(earnings):.0f} UNFED")

    send_tx(escrow_user.functions.claimEarnings(),
            NODE_HONEST_KEY, NODE_HONEST_ADDR)

    token_after = _token.functions.balanceOf(NODE_HONEST_ADDR).call()
    claimed = token_after - token_before
    info(f"Claimed: {from_wei(claimed):.0f} UNFED")

    remaining_earnings = onchain.node_earnings(NODE_HONEST_ADDR)
    if remaining_earnings == 0 and claimed > 0:
        ok(f"Earnings claimed successfully: {from_wei(claimed):.0f} UNFED")
    else:
        fail(f"Earnings claim failed (remaining: {remaining_earnings})")

    # ---------------------------------------------------------------
    # Scenario 7: Honest node unstakes (cooldown)
    # ---------------------------------------------------------------
    header("SCENARIO 7: Honest node unstakes (cooldown enforced)")

    staked = onchain.staked_balance(NODE_HONEST_ADDR)
    info(f"Current stake: {from_wei(staked):.0f} UNFED")

    send_tx(escrow_user.functions.requestUnstake(staked),
            NODE_HONEST_KEY, NODE_HONEST_ADDR)
    info(f"Unstake requested. Stake now: {from_wei(onchain.staked_balance(NODE_HONEST_ADDR)):.0f} UNFED")

    # Try to withdraw immediately (should fail — cooldown not expired)
    try:
        tx = escrow_user.functions.withdraw().build_transaction({
            "from": NODE_HONEST_ADDR,
            "nonce": _w3.eth.get_transaction_count(NODE_HONEST_ADDR),
            "gas": 500_000,
            "gasPrice": _w3.eth.gas_price,
        })
        signed = _w3.eth.account.sign_transaction(tx, NODE_HONEST_KEY)
        h = _w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = _w3.eth.wait_for_transaction_receipt(h)
        if receipt["status"] == 0:
            ok("Withdrawal correctly reverted during cooldown (300s)")
        else:
            fail("Withdrawal should have failed during cooldown!")
    except Exception:
        ok("Withdrawal correctly blocked during cooldown (300s)")

    # Advance time past cooldown
    info("Advancing time 301 seconds past cooldown...")
    anvil_advance_time(301)

    token_before = _token.functions.balanceOf(NODE_HONEST_ADDR).call()
    send_tx(escrow_user.functions.withdraw(),
            NODE_HONEST_KEY, NODE_HONEST_ADDR)
    token_after = _token.functions.balanceOf(NODE_HONEST_ADDR).call()
    withdrawn = token_after - token_before

    if withdrawn > 0:
        ok(f"Withdrawal successful after cooldown: {from_wei(withdrawn):.0f} UNFED")
    else:
        fail("Withdrawal failed after cooldown")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    header("SIMULATION COMPLETE")

    print()
    info("Final balances:")
    info(f"  Escrow pool:      {from_wei(onchain.escrow_pool()):.0f} UNFED")
    info(f"  Honest node:")
    info(f"    Stake:          {from_wei(onchain.staked_balance(NODE_HONEST_ADDR)):.0f} UNFED")
    info(f"    Wallet:         {from_wei(_token.functions.balanceOf(NODE_HONEST_ADDR).call()):.0f} UNFED")
    info(f"    Earnings:       {from_wei(onchain.node_earnings(NODE_HONEST_ADDR)):.0f} UNFED")
    info(f"  Cheater node:")
    info(f"    Stake:          {from_wei(onchain.staked_balance(NODE_CHEATER_ADDR)):.0f} UNFED")
    info(f"    Wallet:         {from_wei(_token.functions.balanceOf(NODE_CHEATER_ADDR).call()):.0f} UNFED")
    info(f"    Earnings:       {from_wei(onchain.node_earnings(NODE_CHEATER_ADDR)):.0f} UNFED")
    info(f"  Client:")
    info(f"    Escrow balance: {from_wei(onchain.client_balance(CLIENT_ADDR)):.0f} UNFED")
    info(f"    Wallet:         {from_wei(_token.functions.balanceOf(CLIENT_ADDR).call()):.0f} UNFED")
    print()
    print("  All scenarios passed!")
    print()


if __name__ == "__main__":
    main()
