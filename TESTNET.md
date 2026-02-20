# UNFED AI — Testnet Guide

How to run, join, and operate on the UNFED AI test network.

## Prerequisites

- Python 3.12+
- ~2 GB RAM per shard (Qwen2.5-0.5B, 4 shards)
- Model weights: run `python -m tools.cli split --model Qwen/Qwen2.5-0.5B --num-shards 4 --output-dir shards` or download pre-split shards

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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UNFED_REGISTRY` | Registry address | `localhost:50050` |
| `UNFED_SEEDS` | Comma-separated seed registries | from `network/seeds.json` |
| `UNFED_WIRE_DTYPE` | Wire format for activations | `float16` |
| `UNFED_COMPRESS_THRESHOLD` | Compression threshold (bytes) | `16384` |
| `UNFED_PREFILL_MIN` | Min tokens for pipelined prefill | `64` |

## Running as a Node Operator

1. Split or download the model shards
2. Pick a shard index (coordinate with other operators or use `--auto`)
3. Open firewall ports
4. Start your node with `--advertise` pointing to your public IP
5. Verify registration via the web dashboard or `python -m client.client --list-models`

## Running the Client

```bash
python -m client.client \
  --registry "registry.unfed.ai:50050" \
  --prompt "Explain quantum computing" \
  --max-tokens 100
```

Add `--tls-ca ca.crt` if the network uses TLS.

## Web Dashboard

```bash
python -m web.server --host 0.0.0.0 --port 8080 --registry "registry.unfed.ai:50050"
```

Open `http://localhost:8080` — tabs for Chat, Network topology, and Chain explorer.

## Known Limitations

- **No persistent identity**: node IDs are generated per session (no wallet integration yet)
- **No incentive enforcement**: the share-chain tracks contributions but settlement is not on-chain yet
- **Single model**: the testnet runs one model (Qwen2.5-0.5B); multi-model pools are implemented but untested at scale
- **No NAT traversal**: nodes behind NAT must configure port forwarding manually
- **TLS is optional**: without it, traffic is plaintext — do not send sensitive data
