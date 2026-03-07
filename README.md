# UNFED AI — Testnet Guide

How to run, join, and operate on the UNFED AI test network.

## Prerequisites

- Python 3.12+
- ~2 GB RAM per shard (Qwen2.5-0.5B, 4 shards)
- Model weights: run `python -m tools.cli split Qwen/Qwen2.5-0.5B -o shards --text-shards 4` or download pre-split shards

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Local)

Start a full local stack:

```bash
# Terminal 1: Registry
python -m network.registry_server --port 50050

# Terminals 2-5: Compute nodes (one per shard)
python -m node.server --shard-index 0 --port 50051 --shards-dir shards
python -m node.server --shard-index 1 --port 50052 --shards-dir shards
python -m node.server --shard-index 2 --port 50053 --shards-dir shards
python -m node.server --shard-index 3 --port 50054 --shards-dir shards

# Terminal 6: Client
python -m client.client --prompt "Hello, world"

# Terminal 7 (optional): Web dashboard
python -m web.server --port 8080
```

By default, the dashboard binds to `127.0.0.1`. Use `--host 0.0.0.0` only when you
intend to expose it behind network controls.

Run preflight before exposing services:

```bash
python -m scripts.testnet_preflight web --host 127.0.0.1
python -m scripts.testnet_preflight node --advertise "203.0.113.42:50051" --tls-cert /path/to/server.crt --tls-key /path/to/server.key
```

## Using GGUF / Ollama Models

UNFED helper scripts now support GGUF inspection and conversion. This is useful
when you downloaded a model with Ollama and want to validate compatibility
before testnet rollout.

Install/update dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Inspect a GGUF file:

```bash
python -m tools.cli inspect /path/to/model.gguf
```

Convert GGUF to a local safetensors file:

```bash
python -m tools.cli convert /path/to/model.gguf -o ./model.safetensors
```

Split directly from GGUF (supported subset):

```bash
python -m tools.cli split /path/to/model.gguf -o ./shards --text-shards 4
```

Current limitations:

- Split from GGUF is text-only and expects transformer-style tensor names (`model.layers.<idx>.*`)
- Quantized GGUF tensors (for example Q4/Q5/Q8 packed tensors) are rejected
- For unsupported GGUF key layouts, use original Hugging Face weights for splitting

Troubleshooting:

- `Unsupported GGUF tensor dtype`: your file is quantized; use non-quantized export or HF weights
- `supported transformer layer key names`: GGUF uses llama.cpp naming; split from HF source instead
- `GGUF support requires the 'gguf' package`: run `pip install gguf`

## Hosting on the Internet

To expose your node to other participants over the internet:

### 1. Open firewall ports

Each service needs its port accessible from the internet:

| Service  | Default Port | Protocol |
|----------|-------------|----------|
| Registry | 50050       | TCP/gRPC |
| Node     | 50051+      | TCP/gRPC |
| Web UI   | 8080        | TCP/HTTP |
| Daemon   | 50080       | TCP/gRPC |

Example (ufw):
```bash
sudo ufw allow 50051/tcp
```

### 2. Use `--advertise` with your public IP

Nodes register with the registry using their advertised address. By default this is `localhost:<port>`, which only works locally.

```bash
python -m node.server \
  --shard-index 0 \
  --port 50051 \
  --advertise "203.0.113.42:50051" \
  --registry "registry.unfed.ai:50050" \
  --shards-dir shards
```

Or via config file:

```bash
python -m node.run --config node_config.json
```

See `node_config.testnet.json.example` for a template.

### 3. Configure seed registries

Edit `network/seeds.json` (or set `UNFED_SEEDS` env var) to point at the public registry:

```bash
export UNFED_SEEDS="registry.unfed.ai:50050,registry2.unfed.ai:50050"
```

Or copy `network/seeds.testnet.json.example` to `network/seeds.json` and fill in real addresses.

### 4. TLS (recommended for public networks)

Generate or obtain a TLS certificate, then start with:

```bash
python -m network.registry_server \
  --port 50050 \
  --tls-cert /path/to/server.crt \
  --tls-key /path/to/server.key

python -m node.server \
  --shard-index 0 --port 50051 \
  --tls-cert /path/to/server.crt \
  --tls-key /path/to/server.key \
  --registry "registry.unfed.ai:50050"
```

Publicly advertised compute nodes now require TLS by default. If a node advertises
a non-local endpoint and no TLS cert/key is provided, startup fails closed.

Clients connect with:
```bash
python -m client.client --tls-ca /path/to/ca.crt
```

For testing, generate a self-signed cert:
```bash
openssl req -x509 -newkey rsa:2048 -keyout server.key -out server.crt \
  -days 365 -nodes -subj "/CN=your.domain.com" \
  -addext "subjectAltName=DNS:your.domain.com,IP:203.0.113.42"
```

## On-Chain Escrow and Token

The registry can enforce staking, escrow, and settlement on-chain via an ERC-20 token and the UnfedEscrow contract. This is optional — without it the registry runs in-memory simulation mode (`--no-chain`).

### Deploy contracts to a testnet

You need [Foundry](https://book.getfoundry.sh/getting-started/installation) installed and a wallet funded with testnet ETH (get Sepolia ETH from [sepoliafaucet.com](https://sepoliafaucet.com)).

```bash
export CHAIN_RPC_URL="https://rpc.sepolia.org"
export OPERATOR_PRIVATE_KEY="0x<your-deployer-key>"
export OPERATOR_ADDRESS="0x<your-deployer-address>"

./scripts/deploy_testnet.sh
```

This deploys `UnfedToken` (1M test UNFED) and `UnfedEscrow`, then writes the addresses to `deployed.env` at the project root.

For local development with Anvil instead:

```bash
export OPERATOR_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
export OPERATOR_ADDRESS=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266

./scripts/start_local_chain.sh
```

### Start the registry with on-chain escrow

Create a `cluster_config.json`:

```json
{
  "name": "My UNFED Pool",
  "chain_rpc_url": "https://rpc.sepolia.org",
  "escrow_contract_address": "0x<from deployed.env>",
  "staking_token_address": "0x<from deployed.env>",
  "operator_private_key": "0x<your-operator-key>"
}
```

You do not need to provide a token name in config; the registry resolves token
metadata from the ERC-20 contract at `staking_token_address`.

Then start the registry:

```bash
python -m network.registry_server --port 50050 --cluster-config cluster_config.json
```

If you also want the browser UI for easier querying, start the web server
separately (registry does not launch it automatically):

```bash
python -m web.server --port 8080 --registry localhost:50050
```

The registry will verify node stake eligibility on-chain, post settlements, and handle slashing.

## Docs Site

The static docs are in `site/` and include:

- Getting started: `site/getting-started.html`
- Architecture and protocol flow: `site/architecture.html`, `site/protocol.html`
- Economics and staking/slashing model: `site/economics.html`
- Full config reference by node type: `site/config-reference.html`

Community links:

- GitHub: [https://github.com/risha-max/UNFED-AI](https://github.com/risha-max/UNFED-AI)
- Discord: [https://discord.gg/7N73crmnbu](https://discord.gg/7N73crmnbu)

### Using your own ERC-20 token

The deploy script creates a test `UnfedToken`, but the escrow contract accepts any ERC-20. To use your own token:

1. Deploy `UnfedEscrow` with your token's address (edit `Deploy.s.sol` or pass it as an env var)
2. Set `staking_token_address` in `cluster_config.json` to your token's address
3. Nodes must hold and stake your token to be eligible

### How staking works

- Nodes call `stake()` on the escrow contract with the required minimum (default: 100 tokens)
- The registry checks `isEligible()` on-chain before accepting a node
- Clients deposit tokens into escrow; inference fees are deducted per-request
- The operator posts settlement batches via `postSettlement()`; after the challenge window, `finalizeSettlement()` unlocks claimable payouts
- Misbehaving nodes can be slashed (50% of stake, configurable)

### Test token faucet

On a testnet the operator holds all minted tokens. Clients can request free test
tokens from the web dashboard:

1. Open the dashboard and enter your wallet address in the Settings sidebar.
2. Click **Get Test Tokens**.
3. The server calls `depositFor(address, amount)` on the escrow contract, which
   transfers tokens from the operator and credits them directly to the client's
   escrow balance.

Rate limit: one drip (100 tokens) per address per hour. The amount and cooldown
are configurable via `OnChainEscrow.FAUCET_DRIP_AMOUNT` and
`OnChainEscrow.FAUCET_COOLDOWN`.

Faucet is disabled by default and must be explicitly enabled with:

```bash
export UNFED_FAUCET_ENABLED=1
```

By default, faucet requests also require a wallet-authenticated session
(`UNFED_FAUCET_REQUIRE_AUTH=1`). For local test-only automation, you can disable
that check with `UNFED_FAUCET_REQUIRE_AUTH=0`.

API: `POST /api/faucet` with body `{"address": "0x...", "session_token": "<optional when auth disabled>"}`. Returns
`{"success": true, "amount": 100, "tx_hash": "0x...", "balance": 200.0}` on
success, or an error with the appropriate HTTP status.

Address validation is strict: invalid/non-EVM addresses are rejected with
HTTP `400`. Faucet cooldown is keyed on normalized checksum addresses, so case
changes do not bypass rate limits.

### Strict token billing

Billing is tokenizer-based and model-specific (no character-length heuristics):

- Text requests: input tokens are counted with the selected model tokenizer.
- Vision-language requests: input tokens are counted from the same model
  processor/tokenizer path used to build multimodal inputs (including image
  placeholder tokens).
- Output tokens are counted from streamed generation events.

Cost is reported as:

`(input_tokens * price_per_input_token) + (output_tokens * price_per_output_token)`

Strict mode behavior:

- If model tokenizer/processor cannot be loaded, the request fails (no fallback
  estimate and no usage report).
- Usage reports with invalid token counts are rejected by the registry.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UNFED_REGISTRY` | Registry address | `localhost:50050` |
| `UNFED_SEEDS` | Comma-separated seed registries | from `network/seeds.json` |
| `UNFED_WIRE_DTYPE` | Wire format for activations | `float16` |
| `UNFED_COMPRESS_THRESHOLD` | Compression threshold (bytes) | `16384` |
| `UNFED_PREFILL_MIN` | Min tokens for pipelined prefill | `64` |
| `UNFED_REQUIRE_MPC` | Require MPC shard-0 for admission | `1` |
| `CHAIN_RPC_URL` | Ethereum RPC endpoint | — |
| `OPERATOR_PRIVATE_KEY` | Operator wallet key (deploy/settlement) | — |
| `OPERATOR_ADDRESS` | Operator wallet address | — |

`UNFED_REQUIRE_MPC=1` is the default fail-closed policy across web, client,
and pipeline entrypoints. Set `UNFED_REQUIRE_MPC=0` only for local debugging.

## Manual MPC Local Setup Script

Use the orchestration script to launch chain/funding/registry/daemon/nodes/web
in separate `gnome-terminal` windows:

```bash
export OPERATOR_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
export OPERATOR_ADDRESS=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
./scripts/manual_setup_mpc_testnet.sh
```

Useful flags:

- `--dry-run`: print commands without starting processes
- `--yes-reuse-ports`: continue even if expected ports are already occupied

## Running as a Node Operator

1. Split or download the model shards
2. Pick a shard index (coordinate with other operators or use `--auto`)
3. Open firewall ports
4. Start your node with `--advertise` pointing to your public IP
5. Verify registration via the web dashboard or `python -m client.client --list-models`

## Running the Client

**CLI**:

```bash
python -m client.client \
  --registry "registry.unfed.ai:50050" \
  --prompt "Explain quantum computing" \
  --max-tokens 100
```

Add `--tls-ca ca.crt` if the network uses TLS.

**Web interface**: Every registry hosts a web dashboard. Open the registry's
HTTP address in a browser to chat, inspect the network, and explore the
share-chain — no separate client install needed.

## Web Dashboard

The registry operator starts the dashboard with:

```bash
python -m web.server --host 0.0.0.0 --port 8080 --registry "localhost:50050"
```

Open `http://<registry-ip>:8080` — tabs for Chat, Network topology, and Chain
explorer. Clients visiting the URL can send inference requests directly from
the browser, get test tokens via the faucet, and monitor node activity.

Model selection is strict and registry-driven:

- The dashboard loads model options from `GET /api/models` only (no frontend hardcoded list).
- A request is accepted only for a model that passes admission preflight:
  - full text shard coverage
  - MPC shard-0 entry available (default policy)
  - full vision shard coverage for multimodal requests
- If preflight fails, the request is rejected with a model-scoped error.

Security restrictions:

- Websocket `wallet` and faucet `address` inputs must be valid EVM addresses.
- `cluster_endpoint` overrides are blocked by default; only the co-located registry
  (or entries from `UNFED_ALLOWED_CLUSTER_ENDPOINTS`) are accepted.

Troubleshooting:

- **No models discovered**: ensure compute nodes are registered with the intended
  `model_id` and shard coverage is complete.
- **Model rejected by policy (MPC missing)**: register an MPC shard-0 node, or
  set `UNFED_REQUIRE_MPC=0` for local debugging only.
- **Model rejected by policy (coverage incomplete)**: verify all required shards
  are registered for that `model_id`.

## Known Limitations

- **Partial persistent identity**: compute nodes use their Ethereum address as ID (`--eth-address`, required when escrow is active), but daemon nodes still generate a random ID per session
- **Testnet economics**: on-chain escrow works but uses test tokens — no real value at stake
- **Single model**: the testnet runs one model (Qwen2.5-0.5B); multi-model pools are implemented but untested at scale
- **No NAT traversal**: nodes behind NAT must configure port forwarding manually
- **TLS is optional**: without it, traffic is plaintext — do not send sensitive data
