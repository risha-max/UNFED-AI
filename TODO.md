# UNFED AI — Post-PoC Roadmap

> PoC is complete: distributed inference across 4 gRPC nodes with KV cache management, verified token-for-token against full model output.

---

## Completed

### Phase 13 — Long-Context KV Cache Optimizations

- [x] **Prefix caching** — reuses KV cache from a previous session sharing a common prompt prefix; client tracks `prev_session_id` / `prev_token_ids`, computes longest common prefix, sets `prefix_session_id` + `prefix_length` on `ForwardRequest`; each node clones the KV prefix instead of recomputing (`node/kv_cache.py`, `client/client.py`, `proto/inference.proto`)
- [x] **KV cache int8 quantization** — `CacheEntry.quantize()` converts KV tensors to int8 with per-tensor absmax scaling (~4x memory savings vs float32); automatically applied after each forward pass when `kv_quantize="int8"`; handles None placeholder layers for non-first shards (`node/kv_cache.py`)
- [x] **KV cache CPU offloading** — inactive sessions are automatically moved to CPU RAM after a configurable inactivity threshold; restored transparently on the next cache hit; LRU eviction when memory limit is exceeded (`node/kv_cache.py`)
- [x] **Chunked prefill** — long prefill inputs are split into configurable chunks (e.g. 32 tokens) to limit peak memory; explicit causal attention mask ensures correct attention for non-first chunks; verified bit-identical to non-chunked prefill (`node/layer_runner.py`)
- [x] **Node config KV fields** — 5 new config fields: `kv_quantize`, `kv_offload_enabled`, `kv_offload_after_seconds`, `max_kv_memory_gb`, `prefill_chunk_size` (`node/node_config.py`)
- [x] **Server `--config` support** — `node.server` accepts `--config` flag for JSON config files; periodic KV cache cleanup thread (`node/server.py`)

### Phase 12 — Multi-Registry Model Discovery

- [x] **ListModels RPC** — new `ListModels` RPC on the registry returns a model catalog with per-model health summary (total nodes, shard coverage, can_serve flag) (`proto/registry.proto`, `network/registry_server.py`)
- [x] **Seed registry list** — `network/seeds.json` provides bootstrap registry addresses; `config.SEED_REGISTRIES` loader with fallback to `REGISTRY_ADDRESS` (`network/seeds.json`, `config.py`)
- [x] **RegistryPool** — multi-registry client with automatic fallback; tries registries in order until one responds; merges model catalogs across registries; `list_models()`, `discover()`, `find_healthy_registry()` (`network/discovery.py`)
- [x] **Dynamic model selection** — client accepts `--model` flag to choose any model on the network; `--list-models` prints available models and exits; `UnfedClient` no longer hardcodes `config.MODEL_NAME` (`client/client.py`)

### Phase 11 — Distributed Mini-Chain

- [x] **P2P share-chain gossip** — share-chain is now fully distributed: every node holds a copy, proposes blocks on a 10s timer, and gossips them directly to peers; deterministic conflict resolution (lowest block_hash wins at same height); `GossipBlock` / `GetBlocks` RPCs on `InferenceNode` service (`proto/inference.proto`, `economics/distributed_chain.py`, `node/server.py`)
- [x] **Registry becomes passive peer** — registry no longer produces blocks; receives gossipped blocks from nodes; still runs settlements from its local chain copy (`network/registry_server.py`)
- [x] **Peer sync on startup** — new or recovering nodes fetch missed blocks from a peer via `GetBlocks` RPC before starting block production
- [x] **Local compute share recording** — each node records its own compute shares locally on every forward pass (no more centralized ticket-to-share pipeline for chain recording) (`node/server.py`)
- [x] **Fraud proof → slashing** — `SubmitFraudProof` now identifies the fraudulent node and calls `challenge_settlement()` on the payment contract, triggering stake slashing (`network/registry_server.py`)

### Phase 10 — P2P Upgrade

- [x] **Multi-peer parallel shard download** — BitTorrent-style P2P weight distribution; 4 MB chunk-level SHA256 hashes in manifest; range-based `GetShard` RPC (offset/length); `MultiPeerDownloader` downloads chunks from multiple peers in parallel, verifies each independently, reassigns failed chunks, resumes on interruption (`shard/downloader.py`)
- [x] **Chunk hashes in manifest** — splitter and gen_manifest compute per-chunk SHA256 hashes; backward compatible (old manifests fall back to single-peer download) (`shard/splitter.py`, `shard/gen_manifest.py`)
- [x] **Registry serves manifest** — `GetManifest` / `PutManifest` RPCs on registry; splitter publishes manifest via `--registry` flag; downloader fetches manifest via `--from-registry` flag; eliminates out-of-band manifest distribution (`network/registry_server.py`, `proto/registry.proto`)
- [x] **Bandwidth controls** — nodes choose how much to serve: `serve_shards_enabled` (opt out), `max_upload_rate_mbps` (throttle), `max_concurrent_transfers` (limit parallel uploads), `inference_priority` (pause transfers during inference); downloader ranks peers by observed speed and favors fast peers on retry (`node/server.py`, `node/node_config.py`, `shard/downloader.py`)

### Phase 9 — Operations

- [x] **Node config file** — JSON config (`node_config.json`) for all node types (compute, guard, verifier); 30+ tunable fields covering gRPC workers, timeouts, sampling rates, keepalive, shard paths, EOS token, ZK params; CLI args override config; unified entry point `python -m node.run --config node_config.json` (`node/node_config.py`, `node/run.py`)

### Phase 8 — Fault Tolerance

- [x] **Per-shard racing** — client-driven hop-by-hop routing races N replicas per shard; first response wins; all racers keep KV caches in sync; free verification via background hash comparison; seamless fault tolerance if a node disconnects (`network/racing.py`, `client/client.py --racing`)
- [x] **Intermediate node direct return** — nodes without LM head return activation tensor to caller when no routing info is present (enables client-driven routing)
- [x] **Racing circuit discovery** — `build_racing_circuit()` returns multiple nodes per shard from the registry (`network/discovery.py`)

### Phase 7 — Privacy Hardening

- [x] **Guard relay node** — lightweight gRPC relay between client and compute pipeline; client persists guard selection (Tor strategy); guard sees client IP but cannot read encrypted payload (`network/guard_node.py`)
- [x] **MPC for embedding + layer 0** — 2-party additive secret sharing for shard 0; MPC protocols for RMSNorm, softmax, SiLU; no single node sees raw tokens (`network/mpc_shard0.py`)
- [x] **Return-path encryption** — each node wraps the response in an AES-256-GCM layer; client decrypts all layers; no intermediate node reads the generated token (`network/onion.py`)
- [x] **Redundant voting** — client double-checks 1 random shard per query; commit-then-reveal protocol; 3rd-node tiebreaker on mismatch; catches cheating before the answer reaches the client (`network/voting.py`)

### Phase 6 — Scale

- [x] **int4 quantization** — support for larger models via BitsAndBytes (NF4/int8); `shard/quantize.py` with model size estimation and on-the-fly quantized splitting
- [x] **Pipeline parallelism** — `PipelineScheduler` batches concurrent queries through the pipeline; 1.7x throughput on localhost, approaches Nx on separate machines

### Phase 5 — Economics

- [x] **Share-chain** — P2Pool-style side-chain: compute shares recorded per forward pass, 10s block time, periodic settlement summaries (`economics/share_chain.py`)
- [x] **Optimistic rollup payments** — simulated smart contract with escrow, staking, challenge windows, fraud-based slashing (`economics/payments.py`)
- [x] **Model pool system** — pool registry with per-shard health tracking and self-balancing payout multipliers for underserved shards (`economics/model_pools.py`)

### Phase 4 — Verification

- [x] **Spot-check verifier nodes** — context-free verification tickets (~5% sampling, prefill only); standalone `verifier_node.py` pulls tickets from registry, re-executes, submits fraud proofs
- [x] **ZK verification tickets** — hash commitments + challenge-response protocol; verifiers check correctness without seeing full activations (`network/zk_verification.py`)

### Phase 3 — Privacy

- [x] **Tor-style anonymous routing** — X25519 key exchange + AES-256-GCM onion encryption; each node peels one layer, sees only next hop (`network/onion.py`)
- [x] **Computational randomness routing** — next node = `hash(activation_output) % candidates`; commit-reveal protocol prevents manipulation; session-stable for KV cache (`network/randomness.py`)

### Phase 2 — Networking

- [x] **P2P node discovery** — gRPC registry service with registration, heartbeat, and discovery; replaces hardcoded addresses (`network/registry_server.py`, `network/discovery.py`)
- [x] **BitTorrent-style weight distribution** — `GetShard` RPC streams shard files with hash verification; `shard/downloader.py` fetches from peers (upgraded in Phase 10: multi-peer parallel + chunk verification)
- [x] **Cross-machine deployment** — resilient gRPC channels with keepalive, retry, and reconnect; `--host` and `--advertise` flags for WAN (`network/resilience.py`)

### PoC

- [x] **Model splitter** — Qwen2.5-0.5B split into 4 shards of 6 layers each
- [x] **Layer runner** — loads shard, runs layers with rotary embeddings and DynamicCache KV management
- [x] **gRPC service** — protobuf schema + node server with Forward() RPC
- [x] **Pipeline stitching** — 4 nodes on ports 50051-50054, verified identical output to full model
- [x] **Client** — tokenize, send to Node 0, stream tokens back (~9.5 tok/s on localhost)
- [x] **KV cache management** — per-session DynamicCache, stable circuits, stale session cleanup
- [x] **Correctness verification** — `verify.py` and `test_pipeline.py` confirm token-for-token match

Lever 3: Eliminate Per-Request Overhead; Lever 4: GPU Compute
Lever 6: Connection Optimization
Lever 5: Pipelined Prefill
