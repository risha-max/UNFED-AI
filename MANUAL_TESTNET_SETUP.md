# Manual Testnet Setup (Step-by-Step)

This guide starts a local UNFED testnet manually, in this order:

1. Funding + contracts
2. Registry
3. Daemon
4. Nodes
5. Web UI

Run all commands from project root:

```bash
cd /home/oasis/programming/UNFED_AI
source .venv/bin/activate
```

## 1) Start chain and deploy contracts

Set operator credentials (Anvil default account 0):

```bash
export OPERATOR_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
export OPERATOR_ADDRESS=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
```

Start local chain + deploy token/escrow:

```bash
./scripts/start_local_chain.sh
```

This writes `deployed.env` with:
- `CHAIN_RPC_URL`
- `TOKEN_ADDRESS`
- `ESCROW_ADDRESS`
- operator values

## 2) Fund and stake node wallets

```bash
source deployed.env
export CHAIN_RPC_URL TOKEN_ADDRESS ESCROW_ADDRESS
export MNEMONIC="test test test test test test test test test test test junk"
export STAKE_WEI=100000000000000000000
export FUND_WEI=200000000000000000000
```

Fund accounts 1..3 from account 0 if needed:

```bash
OP_KEY="$(cast wallet private-key "$MNEMONIC" 0)"
for i in 1 2 3; do
  KEY="$(cast wallet private-key "$MNEMONIC" "$i")"
  ADDR="$(cast wallet address --private-key "$KEY")"
  BAL="$(cast call "$TOKEN_ADDRESS" "balanceOf(address)(uint256)" "$ADDR" --rpc-url "$CHAIN_RPC_URL")"
  if [[ "$BAL" -lt "$FUND_WEI" ]]; then
    cast send "$TOKEN_ADDRESS" "transfer(address,uint256)" "$ADDR" "$FUND_WEI" \
      --rpc-url "$CHAIN_RPC_URL" --private-key "$OP_KEY"
  fi
done
```

Approve + stake for accounts 0..3:

```bash
for i in 0 1 2 3; do
  KEY="$(cast wallet private-key "$MNEMONIC" "$i")"
  ADDR="$(cast wallet address --private-key "$KEY")"
  STAKED="$(cast call "$ESCROW_ADDRESS" "stakedBalance(address)(uint256)" "$ADDR" --rpc-url "$CHAIN_RPC_URL")"
  if [[ "$STAKED" -lt "$STAKE_WEI" ]]; then
    cast send "$TOKEN_ADDRESS" "approve(address,uint256)" "$ESCROW_ADDRESS" "$STAKE_WEI" \
      --rpc-url "$CHAIN_RPC_URL" --private-key "$KEY"
    cast send "$ESCROW_ADDRESS" "stake(uint256)" "$STAKE_WEI" \
      --rpc-url "$CHAIN_RPC_URL" --private-key "$KEY"
  fi
  echo "$ADDR staked => $(cast call "$ESCROW_ADDRESS" "stakedBalance(address)(uint256)" "$ADDR" --rpc-url "$CHAIN_RPC_URL")"
done
```

## 3) Create `cluster_config.json`

Create this file at project root (`/home/oasis/programming/UNFED_AI/cluster_config.json`):

```json
{
  "name": "UNFED Local MPC Testnet",
  "chain_rpc_url": "http://localhost:8545",
  "escrow_contract_address": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
  "staking_token_address": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
  "operator_private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
}
```

If your addresses differ, use values from `deployed.env`.

## 4) Start registry

```bash
python -m network.registry_server --port 50050 --cluster-config cluster_config.json
```

## 5) Start daemon

In a new terminal:

```bash
cd /home/oasis/programming/UNFED_AI
source .venv/bin/activate
python -m network.daemon_node --port 50070 --registry localhost:50050
```

Recommended in on-chain mode (stake-gated registration): run daemon with a staked
identity so it can register and receive infra rewards:

```bash
python -m network.daemon_node --port 50070 --registry localhost:50050 \
  --eth-address 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
```

## 6) Start nodes

Use these 4 wallets:
- shard 0: `0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266`
- shard 1: `0x70997970C51812dc3A010C7d01b50e0d17dc79C8`
- shard 2: `0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC`
- shard 3: `0x90F79bf6EB2c4f870365E785982E1f101E93b906`

Start one terminal per node:

```bash
python -m node.server --shard-index 0 --port 50051 --registry localhost:50050 --shards-dir shards --eth-address 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 --willing-to-mpc
python -m node.server --shard-index 1 --port 50052 --registry localhost:50050 --shards-dir shards --eth-address 0x70997970C51812dc3A010C7d01b50e0d17dc79C8
python -m node.server --shard-index 2 --port 50053 --registry localhost:50050 --shards-dir shards --eth-address 0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC
python -m node.server --shard-index 3 --port 50054 --registry localhost:50050 --shards-dir shards --eth-address 0x90F79bf6EB2c4f870365E785982E1f101E93b906
```

## 7) Start web UI

In a new terminal:

```bash
cd /home/oasis/programming/UNFED_AI
source .venv/bin/activate
python -m web.server --port 8080 --registry localhost:50050
```

Open:

```text
http://localhost:8080
```

## 8) Quick checks

- List models:

```bash
python -m client.client --list-models --registry localhost:50050
```

- Check wallet balance from web backend:

```bash
curl "http://localhost:8080/api/client/balance?address=0xa0Ee7A142d267C1f36714E4a8F75612F20a79720"
```

- Request faucet drip:

```bash
curl -X POST "http://localhost:8080/api/faucet" \
  -H "Content-Type: application/json" \
  -d '{"address":"0xa0Ee7A142d267C1f36714E4a8F75612F20a79720"}'
```
