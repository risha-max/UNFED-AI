# UNFED AI — Decentralized Private AI Inference

> A protocol design for privacy-preserving AI inference over a decentralized network, where no single party knows both **who you are** and **what you asked**.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Design Principles](#design-principles)
- [Architecture Overview](#architecture-overview)
- [Model Weights & Distribution](#model-weights--distribution)
- [Model Pools](#model-pools)
- [Layer 1 — Structural Privacy](#layer-1--structural-privacy)
- [Layer 2 — Decentralized Routing](#layer-2--decentralized-routing)
- [Layer 3 — Verification](#layer-3--verification)
- [Layer 4 — Cryptographic Privacy](#layer-4--cryptographic-privacy)
- [Performance Optimizations](#performance-optimizations)
- [Estimated Performance](#estimated-performance)
- [Economics & Payment](#economics--payment)
- [KV Cache Management](#kv-cache-management)
- [Open Problems](#open-problems)
- [Prior Art & References](#prior-art--references)
- [Project Status](#project-status)

---

## Problem Statement

When you send a query to a centralized AI provider (OpenAI, Google, Anthropic), that provider can:

1. **Link all your queries** to your account, API key, or IP address
2. **Build a detailed profile** of your interests, beliefs, health concerns, politics, vulnerabilities
3. **Store this indefinitely** and be compelled to share it (subpoena, breach, government request)

**Goal:** Build a decentralized inference protocol where:
- The AI model runs at near-full speed on unencrypted data (no HE/MPC overhead on the main pipeline)
- No single node sees the full query AND knows the user's identity
- No central authority controls or observes the full pipeline
- Computation is verifiable without compromising privacy
- The system is economically incentivized (nodes earn rewards for honest work)

---

## Design Principles

| Principle | Approach |
|-----------|----------|
| **Identity privacy** | Tor-like anonymous routing — compute nodes never know who sent the query |
| **Content privacy** | Layer-based partitioning — each node runs a complete slice of layers, but only sees a fraction of the full pipeline (activations are deeply transformed between slices) |
| **No central authority** | Computational randomness routing + P2Pool-style side-chain |
| **Verifiability** | Spot-check verifier nodes with context-free encrypted/ZK verification tickets |
| **Performance** | Batched layers + pipeline parallelism |
| **Accessibility** | Quantized weights (int4) + P2P weight distribution — any modern computer can participate |

---

## Architecture Overview

```
┌──────────────────┐
│   User Device    │  Connects anonymously to the network
│   + Tor/Anon     │
└────────┬─────────┘
         │ Anonymous connection (identity stripped)
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED COMPUTE NETWORK                   │
│                                                                  │
│  Each node runs ALL operations for its assigned layer slice      │
│  (complete self-contained computation, no intra-layer hops)      │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Node A  │───▶│ Node B  │───▶│ Node C  │───▶│ Node D  │──▶...│
│  │Layers   │    │Layers   │    │Layers   │    │Layers   │      │
│  │ 0-4     │    │ 5-9     │    │ 10-14   │    │ 15-19   │      │
│  │(int4    │    │(int4    │    │(int4    │    │(int4    │      │
│  │ ~2 GB)  │    │ ~2 GB)  │    │ ~2 GB)  │    │ ~2 GB)  │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│       │                                                          │
│       │ Next node determined by computational randomness         │
│       ▼                                                          │
│  ┌─────────┐    ┌─────────┐                                     │
│  │Verifier │    │Verifier │  Context-free spot checks            │
│  │  V1     │    │  V2     │  (encrypted tickets or ZK proofs)    │
│  └─────────┘    └─────────┘                                     │
│                                                                  │
│  ┌──────────────────────────────┐                                │
│  │  P2Pool Side-Chain           │  Tracks contributions,         │
│  │  (no central coordinator)    │  rewards, stake, penalties     │
│  └──────────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Weights & Distribution

### How the Model Is Structured

A transformer model (e.g., Llama 3) is a stack of identical layer blocks. Each layer contains 9 weight tensors:

```
model.layers.N.self_attn.q_proj.weight      — Q projection
model.layers.N.self_attn.k_proj.weight      — K projection
model.layers.N.self_attn.v_proj.weight      — V projection
model.layers.N.self_attn.o_proj.weight      — Output projection
model.layers.N.mlp.gate_proj.weight         — Feed-forward gate
model.layers.N.mlp.up_proj.weight           — Feed-forward up
model.layers.N.mlp.down_proj.weight         — Feed-forward down
model.layers.N.input_layernorm.weight       — LayerNorm 1
model.layers.N.post_attention_layernorm.weight — LayerNorm 2
```

Plus a shared embedding layer and output head. The layers are the natural partition boundary.

### Quantization: Making Shards Small Enough

Full precision (float16) weights are too large for volunteer nodes. **4-bit quantization** (int4) reduces size by ~4x with minimal quality loss:

```
Model            Layers  Full (fp16)  int4     Per node (16 nodes)
Llama 3 8B       32      16 GB        4.5 GB   280 MB
Llama 3 70B      80      141 GB       38 GB    2.4 GB
Llama 3 405B     128     810 GB       220 GB   13.7 GB
```

**Target for v1:** Llama 3 8B in int4 — each of 16 nodes stores **~280 MB**. Downloadable in under a minute.

**Target for v2:** Llama 3 70B in int4 — each of 16 nodes stores **~2.4 GB**. Downloadable in ~5 minutes.

### P2P Weight Distribution (BitTorrent-Style)

Nodes don't download weights from a central server — they download from each other, like BitTorrent:

```
1. Model manifest published (list of layer shards + their hashes):
   {
     model: "llama3-70b-int4",
     shards: [
       { layers: "0-4",   hash: "0xabc...", size: "2.4 GB" },
       { layers: "5-9",   hash: "0xdef...", size: "2.4 GB" },
       { layers: "10-14", hash: "0x123...", size: "2.4 GB" },
       ...
     ]
   }

2. New node announces: "I want to serve layers 10-14"

3. Network responds: "Nodes 0x7a, 0x3f, 0x91 already have that shard"

4. New node downloads from MULTIPLE peers in parallel:
   From 1 peer at 50 Mbps:   2.4 GB in ~6 minutes
   From 5 peers in parallel:  2.4 GB in ~1.2 minutes
   From 20 peers:             2.4 GB in ~20 seconds

5. Verify: hash(downloaded_shard) == expected_hash from manifest
   → Ensures nobody served tampered weights

6. Node announces readiness to the network
```

The more popular a shard, the more peers seed it, the faster new nodes bootstrap. Popular layers (early and late layers with higher demand) naturally have more seeders.

### Node Hardware Requirements

```
Tier 1 — Lightweight (v1 target):
  Model: Llama 3 8B (int4)
  Weight shard: ~280 MB
  RAM: 1-2 GB
  Hardware: any computer from the last ~10 years, including phones

Tier 2 — Mid-range (v2 target):
  Model: Llama 3 70B (int4)
  Weight shard: ~2.4 GB
  RAM: 4-8 GB
  Hardware: modern laptop or desktop with 8+ GB RAM

Tier 3 — Power node:
  Model: Llama 3 70B (int4), multiple shards
  Weight shard: 5-10 GB (covers more layers, fewer hops)
  RAM: 16+ GB, preferably with GPU
  Hardware: gaming PC, workstation, or cloud server
```

---

## Model Pools

The network supports multiple open-source models simultaneously. Nodes are grouped into **model pools** — protocol-level groupings of all nodes serving the same model. A pool has no central operator; it's an emergent structure.

### Pool Structure

Each pool is defined by a **model manifest** published to an on-chain registry:

```
Model Manifest:
{
  model_id:      "llama3-70b-int4",
  model_hash:    "0xabc123...",        // hash of the full model
  total_layers:  80,
  shards: [
    { shard_id: 0,  layers: "0-4",   hash: "0xdef..." },
    { shard_id: 1,  layers: "5-9",   hash: "0x456..." },
    ...
    { shard_id: 15, layers: "75-79", hash: "0xfff..." }
  ],
  tokenizer_hash: "0x321...",
  config_hash:    "0x654..."
}
```

A node joins a pool by downloading its shard, verifying the hash against the manifest, and announcing to the network with its `model_id` and `shard_ids`.

### Pool Discovery (DNS Seeds + P2P Gossip)

Initial discovery uses DNS seeds, then transitions to fully decentralized P2P gossip:

```
Bootstrap:
  1. Client queries hardcoded DNS seeds:
     seed1.unfed.ai → returns list of active pools + node counts
     seed2.unfed.ai → same (redundancy)
     seed3.unfed.ai → same

  2. Client receives pool directory:
     [
       { model: "llama3-8b-int4",    nodes: 200, status: "healthy"  },
       { model: "llama3-70b-int4",   nodes: 80,  status: "healthy"  },
       { model: "mistral-7b-int4",   nodes: 150, status: "healthy"  },
       { model: "deepseek-r1-int4",  nodes: 40,  status: "degraded" },
       { model: "codellama-34b-int4", nodes: 60,  status: "healthy"  }
     ]

  3. Client connects to peers in the chosen pool via P2P gossip
     From this point on, no DNS seeds needed — fully decentralized.
```

DNS seeds are only used for initial bootstrap (like Bitcoin). They can't censor, manipulate, or track queries — they only provide a starting list of peers. Multiple independent seed operators ensure no single point of failure.

### Pool Health and Coverage

A pool is only usable if it has **full layer coverage** — at least one node for every shard:

```
llama3-70b pool health:
  Shard 0  (layers 0-4):   12 nodes  ✓
  Shard 1  (layers 5-9):    8 nodes  ✓
  ...
  Shard 12 (layers 60-64):  1 node   ⚠ fragile
  Shard 13 (layers 65-69):  5 nodes  ✓
  Shard 14 (layers 70-74):  4 nodes  ✓
  Shard 15 (layers 75-79):  0 nodes  ✗ INCOMPLETE

Status:
  All shards covered     → "healthy"
  Some shards < 3 nodes  → "degraded" (functional but fragile)
  Any shard has 0 nodes  → "incomplete" (cannot serve queries)
```

### Self-Balancing Incentives

Underserved shards automatically become more profitable, attracting nodes:

```
Normal shard reward:           1x base rate
Shard with < 3 nodes:          2x base rate (scarcity bonus)
Shard with only 1 node:        5x base rate (critical scarcity)
```

This creates a self-balancing market — underserved positions attract nodes, which rebalances the pool, which normalizes the reward.

### Multi-Pool Nodes

A single node can serve multiple models simultaneously:

```
Node 0x7a3f:
  llama3-8b-int4,     shard 3 (layers 6-7):     280 MB
  mistral-7b-int4,    shard 5 (layers 10-11):    260 MB
  codellama-34b-int4, shard 2 (layers 4-7):    1.1 GB

  Total: 1.64 GB storage, ~3 GB RAM
  Earns shares from all three pools simultaneously.
```

### User Experience

```
$ unfed chat

Available models:
  [1] Llama 3 8B      — 200 nodes — ~0.0001 XMR/query — fast
  [2] Llama 3 70B     — 80 nodes  — ~0.001 XMR/query  — high quality
  [3] Mistral 7B      — 150 nodes — ~0.0001 XMR/query  — fast
  [4] DeepSeek R1     — 40 nodes  — ~0.002 XMR/query  — reasoning
  [5] CodeLlama 34B   — 60 nodes  — ~0.0005 XMR/query — code

Select model: 2

> What are the symptoms of early-stage pancreatic cancer?

[connecting to llama3-70b pool... 16 nodes selected... circuit built]

Early-stage pancreatic cancer can include...
```

---

## Layer 1 — Structural Privacy

### Layer-Based Partitioning

The model is split by **layer** — each node runs the **complete computation** (MatMul, Softmax, LayerNorm, activation) for its assigned slice of layers. This is the natural partition because:

1. **Operations within a layer are tightly coupled** — attention requires Q, K, V to be on the same machine; splitting by operation type would cause constant back-and-forth between nodes
2. **98% of weights are in MatMul** — a "Softmax-only" node would have almost no weights while the "MatMul node" has everything
3. **The model file is already organized by layer** — `model.layers.0.*`, `model.layers.1.*`, etc.

Each node runs the full inner loop for its layers:

```
Node receives activation tensor → for each assigned layer:
  LayerNorm → Q,K,V projections → Attention (Softmax) → 
  Output projection → Residual → LayerNorm → 
  Feed-forward (MatMul → Activation → MatMul) → Residual
→ Send output activation to next node
```

Only the activation tensor crosses the network between nodes — one hop per layer batch, not 14 hops per layer.

### What Each Node Sees vs. Doesn't See

```
Node handling layers 20-24:

  ✓ Input activation at layer 20 (deeply transformed from original query)
  ✓ All intermediate computations for layers 20-24
  ✓ Attention patterns for its layers (which tokens attend to which)
  ✓ Output activation at layer 24

  ✗ The original query tokens (transformed beyond recognition by layers 0-19)
  ✗ The final output (will be further transformed by remaining layers)
  ✗ Who sent this query (anonymous routing)
  ✗ What other nodes are in the circuit
```

**Privacy by depth:** By the time activations reach deeper layers, they are heavily transformed through many nonlinear operations. Inverting 20+ layers to recover the original input is extremely difficult.

```
Privacy risk by position:
  Layers 0-9:   HIGHEST (closest to raw input embeddings)
  Layers 10-19: High
  Layers 20-39: Medium (activations are increasingly abstract)
  Layers 40-59: Lower
  Layers 60-79: LOW (activations represent output predictions)
```

### Circuit Randomization

Each query uses a **different set of nodes**, determined by computational randomness. No node is guaranteed to see the same position across multiple queries:

```
Query 1: Node A (layers 0-4) → Node D (5-9) → Node B (10-14) → ...
Query 2: Node C (layers 0-4) → Node A (5-9) → Node F (10-14) → ...
Query 3: Node F (layers 0-4) → Node B (5-9) → Node D (10-14) → ...
```

Even if a node handles early layers (high risk), it only does so for a fraction of queries, preventing it from building profiles.

---

## Layer 2 — Decentralized Routing

### The Problem

Any central authority that assigns roles, orchestrates the pipeline, and routes data effectively knows everything. **You've replaced one God with another God.**

### Solution: Computational Randomness

The next node assignment is derived from the **output of the previous computation**:

```
next_node = hash(layer_output) mod (available_nodes)
```

**Properties:**
- Unpredictable until computation finishes (output-dependent)
- No central beacon or coordinator needed
- Adversary cannot pre-position nodes to capture specific layers

### Commit-Reveal Protocol (Anti-Manipulation)

Prevents a node from altering its output to steer routing:

```
Step 1: Node A computes layer output
Step 2: Node A publishes commitment = hash(output)
Step 3: Next node selected using hash(output) — but output not yet revealed
Step 4: Node A reveals the actual output
Step 5: Network verifies: hash(revealed_output) == commitment
        Match    → proceed
        Mismatch → Node A is penalized, stake slashed
```

### Client-Side Circuit Building (Tor Model)

As a fallback, the user's device can select its own circuit from a public node directory:

```
User device randomly selects:
  Node 7  → MatMul for layers 1-12
  Node 23 → Softmax for layers 1-12
  Node 41 → GELU for layers 13-24
  ... etc
```

No central authority involved. The client builds its own path, like Tor circuit construction.

---

## Layer 3 — Verification

### The Problem

Neural network operations are **symmetric** — equally hard to compute and to verify. Unlike mining (where verification is instant), checking `softmax(input) == output` requires re-doing the computation.

### Solution: Spot-Check Verifier Nodes

Dedicated verifier nodes randomly check a small percentage of operations:

```
96 layer batches processed per query
Verifier checks 5 random ones per query (~5% overhead)

Probability cheating node is caught:
  Per query:       5/96 ≈ 5%
  Within 20 queries:     ≈ 64%
  Within 100 queries:    ≈ 99.4%
```

The **threat of verification** keeps nodes honest (same principle as tax audits).

### Context-Free Verification Tickets

Verifiers receive tickets **stripped of all context**:

```json
{
  "operation": "softmax",
  "input": [0.342, -0.871, 0.156, ...],
  "output": [0.518, -0.102, 0.887, ...],
  "ticket_id": "random_hash"
}
```

**What's missing from the ticket:**
- No layer number
- No user ID
- No query ID
- No information about what comes before or after

The verifier is a **stateless calculator** — it checks `softmax(input) == output?` without knowing what those numbers mean.

### Multi-Verifier Conflict Resolution

```
Verifier V1 checks operation → "valid"
Verifier V2 checks operation → "invalid"
Conflict → escalate to full re-verification by N independent verifiers
Majority rules. Dishonest verifier's stake is slashed.
```

### Stake-Based Accountability

Both compute nodes and verifiers put up a stake (deposit):
- **Honest work** → earn rewards from the pool
- **Caught cheating or falsely verifying** → lose stake
- Aligns incentives without requiring trust

---

## Layer 4 — Cryptographic Privacy

### Encrypted Verification Tickets

Instead of sending raw activation values to verifiers, encrypt them:

```
Compute node creates verification ticket:
  Enc(input), Enc(output), operation_type

Verifier uses Homomorphic Encryption to check:
  HE_compute(softmax, Enc(input)) == Enc(output)?

Verifier never sees raw numbers.
```

**Why this is feasible:** Unlike full encrypted inference (96 layers of accumulated HE noise), verification is a **single operation check** — one layer of noise, manageable.

### Zero-Knowledge Proof Alternative

Compute nodes generate ZK proofs of correct computation:

```
Node B computes: output = softmax(input)
Node B generates: ZK_proof("I know input such that softmax(input) = output")
Node B publishes: { proof: ZK_proof }

Verifier checks proof validity:
  - Doesn't see input OR output
  - Doesn't redo computation
  - Just validates the proof math
  - Fast to verify, slow to generate (compute node's cost)
```

### Privacy Stack Summary

```
Layer 1 — Structural:     Functional partitioning, no node sees full pipeline
Layer 2 — Routing:        Computational randomness, no predictable paths
Layer 3 — Verification:   Context-free tickets, verifiers see no metadata
Layer 4 — Cryptographic:  Encrypted tickets or ZK proofs, verifiers see nothing
```

Each layer addresses a different attack vector. Expensive cryptography (Layer 4) is only applied to the small verification step, not the full inference pipeline.

---

## Performance Optimizations

### Strategy 1: Layer Batching

Each node hosts a complete slice of consecutive layers. Fewer nodes = fewer hops:

```
Llama 3 70B (80 layers):
  80 nodes × 1 layer  = 80 hops → too slow
  16 nodes × 5 layers = 16 hops → usable
  8 nodes  × 10 layers = 8 hops → fast, but larger shards

Llama 3 8B (32 layers):
  16 nodes × 2 layers = 16 hops
  8 nodes  × 4 layers = 8 hops
```

### Strategy 2: Pipeline Parallelism

Across multiple users, keep all nodes busy simultaneously:

```
Time:    T1      T2      T3      T4      T5      T6
Node A: [User1] [User2] [User3] [User4] [User5] [User6]
Node B:   -     [User1] [User2] [User3] [User4] [User5]
Node C:   -       -     [User1] [User2] [User3] [User4]
Node D:   -       -       -     [User1] [User2] [User3]
```

For prompt processing (all input tokens known), tokens from the same user can also be pipelined.

---

## Estimated Performance

### v1 Configuration (Llama 3 8B)

```
Model:           Llama 3 8B (32 layers, int4 quantized)
Nodes:           16 compute nodes, each hosting 2 layers
Weight/node:     ~280 MB
Compute/batch:   10ms per 2-layer batch (int4 on CPU)
Network hop:     25ms between nodes
```

```
Per token: 16 hops × (10ms compute + 25ms network) = 560ms

Prompt processing (12 tokens, pipelined):
  Token 1 exits at:  560ms
  Token 12 exits at: 560 + (11 × 35ms) = 945ms

Generation (200 tokens, sequential):
  200 × 560ms = 112,000ms

Total: ~113 seconds ≈ 1 min 53 sec for 200 tokens
First token appears:  ~560ms
Streaming speed:      ~1.79 tokens/sec
```

### v2 Configuration (Llama 3 70B)

```
Model:           Llama 3 70B (80 layers, int4 quantized)
Nodes:           16 compute nodes, each hosting 5 layers
Weight/node:     ~2.4 GB
Compute/batch:   30ms per 5-layer batch (int4 on GPU)
Network hop:     25ms between nodes
```

```
Per token: 16 hops × (30ms compute + 25ms network) = 880ms

Generation (200 tokens, sequential):
  200 × 880ms = 176,000ms

Total: ~176 seconds ≈ 2 min 56 sec for 200 tokens
First token appears:  ~880ms
Streaming speed:      ~1.14 tokens/sec
```

### Comparison

```
                                    Total     Tokens/sec   First token
No optimizations (80 nodes, 70B):  8.7 min    0.38         2.6s
v1: 8B int4 (16 nodes):           1m 53s     1.79         560ms
v2: 70B int4 (16 nodes):          2m 56s     1.14         880ms
+ Pipeline parallelism:            same*      same         same

OpenAI (data center):              3-5 sec    50-100       ~200ms

* Pipeline improves system throughput (more users served),
  not single-user latency
```

---

## Economics & Payment

### The Problem: Proof of Correct Inference

In crypto mining, proof is irrefutable — `hash(block + nonce) < target` is instantly verifiable by anyone, and payment is automatic. In AI inference, there is no cheap on-chain way to verify that `softmax(input) == output`. A blockchain can verify a hash in a smart contract; it **cannot** run neural network operations to check results.

Without irrefutable proof, how do you ensure nodes get paid fairly and only for honest work?

### Solution: Optimistic Rollup Model

Adapted from Ethereum L2s (Arbitrum, Optimism) — assume all work is honest, but allow anyone to challenge dishonest claims within a time window.

```
1. Node A completes operations and submits claims to the share-chain:
   {
     node_id:        "0x7a3f...",
     operation_type: "softmax",
     input_hash:     hash(input activations),
     output_hash:    hash(output activations),
     commitment:     hash(output)  (from commit-reveal protocol)
   }

2. Payment is ESCROWED, not immediately released

3. Challenge window opens (e.g., 24 hours)

4. During the window, any verifier can submit a FRAUD PROOF:
   - Verifier re-computes the operation
   - If Node A's committed output doesn't match: submit evidence
   - Smart contract checks: does the commitment match the fraud proof?
   - Hashes ARE verifiable on-chain (cheap)

5. Outcomes:
   - Fraud proven   → Node A's stake slashed, challenger rewarded
   - No challenge    → payment released automatically after window
```

**Why this works:** The system doesn't prove everything was correct on-chain. It proves nothing was **provably wrong** within the challenge period. The irrefutable part is the **fraud proof** — comparing hashes is cheap and on-chain verifiable.

### Escrow: No Third Party, Just Code

The escrow is a **smart contract** — a program deployed on the blockchain that holds and releases funds automatically. No person, no company, no server. Just immutable, self-executing code.

```
Traditional escrow:
  Buyer → gives money to Lawyer → Lawyer holds it → releases when conditions met
  Trust required: you trust the lawyer

Smart contract escrow:
  User → sends funds to Contract Address → code holds it → releases when conditions met
  Trust required: ZERO — code is public, immutable, and self-executing
```

The smart contract is a blockchain address with code attached. When you send funds to it, they're locked at that address. Nobody "has" the money — it sits on-chain, governed by rules that no one can change (not even the creator).

```
Step 1: User sends 0.001 XMR to contract address 0xESCROW
        Contract records: "0.001 XMR held for query #4521"

Step 2: Nodes complete work, share-chain records contributions

Step 3: Challenge window (24 hours) — contract waits
        If challenge() called with valid fraud proof → auto-slash cheater
        If no challenge within 24 hours → auto-release funds

Step 4: Contract executes release():
        → 0.0006 XMR to Node A's wallet
        → 0.0003 XMR to Node B's wallet
        → 0.0001 XMR to Verifier V's wallet
        Automatic. No approval. No human involved.
```

**Note on Monero:** Monero does not natively support smart contracts like Ethereum. Options include:
- **Time-locked transactions** — Monero supports outputs spendable only after a certain block height, with refund paths (limited but functional for simple escrow)
- **Sidechain with smart contract support** — run escrow logic on a sidechain that bridges to Monero for final settlement
- **Multi-signature schemes** — Monero supports multisig; a 2-of-3 multisig between user, node, and a decentralized arbitration protocol can simulate escrow

### The Share-Chain (P2Pool Model)

A lightweight side-chain tracks contributions in near real-time:

```
Main chain (slow, secure):         Side-chain / share-chain (fast):
  - Final settlement of rewards      - Records every compute contribution
  - Stake deposits/withdrawals       - Produces a block every ~10 seconds
  - Fraud proof resolution           - Tallies shares per node
  - One block every ~2 minutes       - Periodically settles to main chain
```

Each share-chain block records:

```
Block #4521
  Previous: hash(Block #4520)
  Timestamp: 1707500010
  
  Contributions:
    Node 0x7a3f: 12 MatMul ops,  3 Softmax ops  → 69 shares
    Node 0xb2c1: 8 GELU ops, 5 verifications     → 13 shares
    Node 0x91d4: 15 LayerNorm ops                 → 15 shares
  
  Total shares this block: 97
  Running total this epoch: 4,823
```

**Privacy note:** The share-chain records operation hashes and types only — no query content, no user IDs, no layer numbers.

### Share Weighting

Operations have different computational costs and are weighted accordingly:

```
Operation rates (per 12-layer batch):
  MatMul:       5 shares  (most compute-intensive)
  Softmax:      2 shares
  LayerNorm:    1 share
  GELU:         1 share
  Verification: 1 share per spot-check
```

### Payment Flow

```
┌─────────┐   pays per query    ┌──────────────┐
│  User   │ ──────────────────▶ │  Fee Escrow  │
└─────────┘                     └──────┬───────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  Share-Chain     │
                              │  (tracks work)   │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                   ▼
             ┌────────────┐   ┌────────────┐      ┌────────────┐
             │ Compute    │   │ Compute    │      │ Verifier   │
             │ Node A     │   │ Node B     │      │ Node V     │
             │ earns      │   │ earns      │      │ earns      │
             │ shares     │   │ shares     │      │ shares     │
             └─────┬──────┘   └─────┬──────┘      └─────┬──────┘
                   │                │                    │
                   └────────────────┼────────────────────┘
                                    ▼
                        ┌────────────────────┐
                        │  Challenge Window  │
                        │  (24 hours)        │
                        │  No fraud proof?   │
                        │  → Release payment │
                        └────────┬───────────┘
                                 ▼
                          ┌──────────────┐
                          │  Settlement  │
                          │  (main chain)│
                          │  Payouts     │
                          └──────────────┘
```

### Staking and Penalties

Nodes must stake tokens to participate:

```
To join as compute node:  stake 1 XMR (or equivalent)
To join as verifier:      stake 0.5 XMR

Penalty events:
  Commit-reveal mismatch (manipulated output):  lose 100% of stake
  Failed verification (wrong computation):       lose 50% of stake
  Timeout (didn't respond in time):              lose 5% of stake
  Verifier caught colluding:                     lose 100% of stake

Slashed stakes:
  50% → rewarded to the challenger who proved fraud
  50% → burned (deflationary)
```

**Economic irrationality of cheating:** A node staking 1 XMR to earn 0.001 XMR per query has no rational incentive to cheat — the potential loss (entire stake) far exceeds any single query's reward.

### Future Path: ZK Proofs of Computation

As ZK proof technology matures, nodes could generate irrefutable mathematical proofs per operation — equivalent to mining's `hash < target`. This would eliminate the challenge window entirely:

```
Current:  Optimistic model → assume honest, challenge if not (24hr delay)
Future:   ZK proof per operation → mathematically proven, instant settlement
```

Projects like EZKL, Modulus Labs, and Giza are actively working on making ZK proofs for neural network operations practical.

---

## KV Cache Management

### The Problem

Autoregressive LLM generation requires **KV (Key-Value) caches** — each transformer layer stores attention key and value tensors from all previous tokens. Without a KV cache, the model would need to reprocess the entire prompt + all previously generated tokens at every single decode step, making generation O(n²) in sequence length.

### Why This Matters for Distributed Inference

In a centralized model, the KV cache lives in one process's memory. In a distributed pipeline, each node stores KV caches **only for its own layers** — but these caches must persist across token steps for the same query session.

```
Token step 1 (prefill "The capital of France is"):
  Node 0 (layers 0-5):   KV cache stores 5 tokens × 6 layers
  Node 1 (layers 6-11):  KV cache stores 5 tokens × 6 layers
  Node 2 (layers 12-17): KV cache stores 5 tokens × 6 layers
  Node 3 (layers 18-23): KV cache stores 5 tokens × 6 layers → outputs "Paris"

Token step 2 (decode "Paris"):
  Node 0: extends its KV cache to 6 tokens, runs only the new token
  Node 1: extends its KV cache to 6 tokens
  Node 2: extends its KV cache to 6 tokens
  Node 3: extends its KV cache to 6 tokens → outputs "."
  
  ...and so on for each token
```

### Design Constraints

This introduces critical requirements for the distributed protocol:

1. **Stable circuits** — the same set of nodes must handle the same session for its entire duration (can't randomly reassign mid-generation)
2. **Session identity** — each node needs a `session_id` to associate KV cache state with the correct request
3. **Memory management** — nodes must clean up stale sessions (timeout after inactivity) to prevent memory exhaustion
4. **No KV cache transfer** — transferring KV caches between nodes would be prohibitively expensive (they grow with sequence length); the circuit must remain fixed

### PoC Implementation

The PoC uses `DynamicCache` from Hugging Face Transformers, keyed by `session_id`:

```
Per node:
  session "abc123" → DynamicCache (layers N to M, growing with each token)
  session "def456" → DynamicCache (different query, independent state)
  
Cleanup: sessions unused for 5 minutes are evicted.
```

---

## Open Problems

### 1. Latency vs Privacy Tradeoff
Fewer hops = faster but each node sees more layers. More hops = more private but slower. Finding the optimal batch size is an engineering challenge.

### 2. Model Weight Distribution
Largely mitigated by int4 quantization (~280 MB to ~2.4 GB per shard) and P2P BitTorrent-style distribution. Remaining challenge: nodes can't switch layer assignments instantly (requires downloading a different shard). Dynamic rebalancing when nodes join/leave needs a protocol for shard reassignment.

### 3. Sybil Attacks
If an adversary controls enough nodes, they can capture multiple pipeline positions and reconstruct queries. Mitigations: stake requirements, reputation systems, diversity requirements per circuit.

### 4. Activation Inversion
Intermediate activations (even without layer context) may leak statistical information about the input. Mitigations: noise injection (differential privacy), role rotation, anonymous routing.

### 5. Incentive Design
With layer-based partitioning, all nodes do the same type of work (complete layer computation), simplifying pricing. Remaining challenge: nodes with GPUs are faster than CPU-only nodes — how to price speed vs. just completion.

### 6. Mixnet Cold Start
The anonymous mixer approach requires enough simultaneous users for a meaningful anonymity set. Mitigation: generate synthetic/dummy queries to pad the set when user volume is low.

---

## Prior Art & References

| Project | Relevance |
|---------|-----------|
| **ZKANN** (Ross Ulbricht, 2020) | Zero-Knowledge Artificial Neural Network — distributing ANN neurons across independent volunteer nodes so no single node sees the full content |
| **Petals / Hivemind** | Decentralized LLM inference by splitting model layers across volunteer nodes (BitTorrent for AI) |
| **Tor Project** | Onion routing — separating identity from content through multi-hop anonymous circuits |
| **Monero P2Pool** | Decentralized mining pool with no central operator, using a peer-to-peer side-chain for contribution tracking |
| **Nillion** | Blind computation — splitting data across nodes using MPC so no single node can reconstruct the full picture |
| **Bittensor (TAO)** | Blockchain-based decentralized AI network with economic incentives for compute contribution |
| **Zama (TFHE)** | Fully Homomorphic Encryption library making encrypted computation practical for ML |

---

## Project Status

**Phase: Working Proof of Concept**

The core distributed inference pipeline is implemented and verified:

### What Works (PoC)

- **Model splitting** — Qwen2.5-0.5B (24 layers) split into 4 shards of 6 layers each
- **Layer runner** — loads a shard, runs its assigned layers with KV cache management
- **gRPC pipeline** — 4 independent node processes communicate over gRPC, forwarding activation tensors
- **Autoregressive generation** — client sends a prompt, tokens stream back through the pipeline
- **Correctness verified** — distributed output matches full model output token-for-token (greedy decoding)
- **Performance** — ~9.5 tokens/sec on localhost (100ms per decode step across 4 nodes)

### Tech Stack

- Python, PyTorch, Hugging Face Transformers
- gRPC (protobuf) for node-to-node activation transfer
- DynamicCache for per-session KV cache management

### Running the PoC

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Split the model into shards
python -m shard.splitter

# 3. Launch 4 nodes (each in a separate terminal)
python -m node.server --node-index 0
python -m node.server --node-index 1
python -m node.server --node-index 2
python -m node.server --node-index 3

# 4. Run the client
python -m client.client --prompt "The capital of France is" --max-tokens 50

# Or interactive mode:
python -m client.client

# Verify correctness against full model:
python verify.py
```

### What's NOT in the PoC

- No anonymous routing (nodes are hardcoded)
- No P2P discovery (addresses in config file)
- No verification or spot-checks
- No share-chain or payments
- No quantization (Qwen2.5-0.5B is small enough in float32)
- No computational randomness routing (fixed circuit)
- Nodes on same machine (different ports)

### Post-PoC Roadmap

1. **Networking** — cross-machine deployment, P2P node discovery, BitTorrent-style weight distribution
2. **Privacy** — Tor-style anonymous routing, computational randomness routing with commit-reveal
3. **Verification** — spot-check verifier nodes, encrypted/ZK verification tickets
4. **Economics** — share-chain, optimistic rollup payments, smart contract escrow
5. **Scale** — int4 quantization for larger models (Llama 3 8B/70B), pipeline parallelism

### Architecture

This document captures the full architectural design of the protocol. The core ideas combine:

1. Layer-based partitioning — each node runs a complete slice of model layers, no node sees the full pipeline
2. Tor-style anonymous routing for identity privacy
3. P2Pool-style decentralized coordination (no central authority)
4. Computational randomness for unpredictable, manipulation-resistant routing
5. Context-free verification with encrypted/ZK tickets
6. Quantized weights (int4) with P2P BitTorrent-style distribution for accessibility
7. Optimistic rollup model with smart contract escrow for trustless payment
8. Pipeline parallelism for system throughput
