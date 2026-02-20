# Privacy-Preserving Decentralized AI Inference: Market & Technology Analysis

A research synthesis on technology approaches, UNFED AI differentiators, success factors, and risks. Data and citations are from published research and industry sources (2024–2025 where noted).

---

## 1. Technology Approaches for Private AI Inference

### 1.1 Homomorphic Encryption (HE) for AI

**Current state:** HE allows computation on encrypted data without decryption. Recent work extends HE beyond CNNs to **transformers**: polynomial-form transformers enable secure inference on language and vision models that were previously considered impractical for HE ([Converting Transformers to Polynomial Form for Secure Inference Over Homomorphic Encryption](https://proceedings.mlr.press/v235/zimerman24a.html)).

**Performance overhead:** HE still incurs large cost. Computation overhead is **at least two orders of magnitude** slower than plaintext ([Towards Fast and Scalable Private Inference](https://export.arxiv.org/pdf/2307.04077v1.pdf)). Optimizations are closing the gap: HTCNN reports ~6.9× throughput improvement for CNN inference (163 images in 10.4s, 98.9% accuracy on MNIST) ([HTCNN: High-Throughput Batch CNN Inference with Homomorphic Encryption](https://eprint.iacr.org/2024/1753)); MOFHEI reduces latency and memory by factors of 9.63 and 4.04 via block pruning ([MOFHEI](https://www.arxiv.org/pdf/2412.07954)); Equivariant Encryption (2025) aims for **near-zero overhead** by encrypting only critical internal representations ([Encrypted Large Model Inference: The Equivariant Encryption Paradigm](https://arxiv.org/abs/2502.01013)). HE remains non-interactive (no extra communication rounds), which is an advantage over MPC in some deployments ([Comparison of FHE and Garbled Circuit in Privacy-Preserving ML](https://arxiv.org/html/2510.07457v1)).

**Summary:** HE is moving from research to early practicality for transformers, but overhead is still 100×+ in many settings; selective encryption and pruning are the main levers to reduce it.

---

### 1.2 Multi-Party Computation (MPC) for AI

**Who uses it:** **Meta’s CrypTen** is a prominent framework that exposes MPC primitives via standard ML abstractions (tensors, autodiff, modular networks), targeting adoption by ML practitioners ([CrypTen: Secure Multi-Party Computation Meets Machine Learning](https://ai.meta.com/research/publications/crypten-secure-multi-party-computation-meets-machine-learning)). It demonstrates two-party secure inference **faster than real-time** on speech tasks with GPU and high-performance communication ([NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html)).

**Performance:** MPC has **lower compute overhead** than HE but adds **multiple communication rounds** and serialization, causing GPU idle time during transfers ([Towards Fast and Scalable Private Inference](https://export.arxiv.org/pdf/2307.04077v1.pdf)). PIGEON improves ReLU by two orders of magnitude and supports large-batch ImageNet (e.g. 192-image batches) via CPU/GPU switching ([PIGEON](https://eprint.iacr.org/2024/1371)). Mixed secret-sharing can reduce communication by **3.6–6.1×** and runtime by **1.5–4.3×** in WAN settings ([Scalable Honest-majority MPC for ML from Mixed Secret Sharings](https://eprint.iacr.org/2026/038)). MPC-Pipe cuts latency by **12.6–14.48%** by pipelining compute and communication ([ibid.](https://export.arxiv.org/pdf/2307.04077v1.pdf)).

**Summary:** MPC is used in research and by large labs (e.g. Meta); frameworks are maturing and performance is viable for selected workloads, with communication and pipelining as main optimization axes.

---

### 1.3 Trusted Execution Environments (TEEs: Intel SGX, AMD SEV)

**Technologies:** AMD SEV is VM-level (“lift-and-shift”); Intel SGX is process-level with higher integration complexity; Intel TDX is Intel’s VM-level answer ([Confidential VMs: AMD SEV-SNP and Intel TDX](https://dl.acm.org/doi/10.1145/3700418)). AMD SEV is built into EPYC 7000/8000/9000 ([AMD Confidential Computing](https://www.amd.com/en/products/processors/server/epyc/confidential-computing.html)).

**Adoption for AI:** **Microsoft Azure** is pushing confidential AI inference across VMs, containers, and attestation, treating confidential computing as part of responsible AI ([Azure AI Confidential Inferencing](https://techcommunity.microsoft.com/blog/azureconfidentialcomputingblog/azure-ai-confidential-inferencing-technical-deep-dive/4253150)). **Phala Network** offers decentralized confidential inference using CPU TEEs (Intel SGX/TDX, AMD SEV) and **GPU TEEs** (e.g. H100/H200), claiming up to **99% efficiency** vs native for larger models ([Phala Confidential AI](https://docs.phala.network/confidential-ai-inference/benchmark); [Phala Overview](https://docs.phala.network/overview/phala-network/confidential-ai-inference)).

**Summary:** TEEs are in production in major clouds and in decentralized stacks (e.g. Phala); VM-based (SEV/TDX) is easing adoption compared to enclave-based SGX.

---

### 1.4 Model Partitioning / Splitting Across Nodes

**Who does this:** Several systems split models across parties for privacy and scale:

- **Cascade:** Token-sharded private LLM inference; uses **sequence-dimension sharding** instead of heavy crypto to get large speedups ([Cascade: Token-Sharded Private LLM Inference](https://arxiv.org/html/2507.05228v1)).
- **Fission:** Two-network design—MPC for linear layers, separate evaluators for nonlinearities on shuffled values; **~8× faster** inference and **~8×** less bandwidth than prior methods, weights stay private ([Fission: Distributed Privacy-Preserving LLM Inference](https://eprint.iacr.org/2025/653)).
- **PFID:** Client–server model sharding with SVD; client keeps compressed hidden states instead of sending raw prompts ([PFID](https://arxiv.org/abs/2406.12238)).
- **PlanetServe (GenTorrent):** P2P overlay for decentralized LLM serving—routing, privacy, forwarding, verification ([PlanetServe](https://arxiv.org/abs/2504.20101)).
- **Petals:** Layers split across geodistributed devices; BLOOM-176B runs up to **10× faster** than offloading with fault tolerance and load balancing ([Petals](https://openreview.net/pdf?id=XmN7ZNbUAe)).

**Summary:** Layer-wise and token-wise partitioning is an active design pattern; combined with MPC or shuffling it yields privacy and scalability without full HE/MPC on every layer.

---

### 1.5 Differential Privacy in Inference

**Role:** DP bounds the influence of individual inputs (e.g. tokens or users) on outputs. It is used in inference to protect sensitive context or user data.

- **DP-Fusion:** Token-level DP for LLMs; labels sensitive tokens, runs with/without them, blends distributions to bound influence; **~6× lower perplexity** than related methods for document privatization ([DP-Fusion](https://arxiv.org/abs/2507.04531)).
- **Split-and-Denoise (SnD):** Local DP; client does embedding on-device, adds noise before server; **>10%** gain vs baselines under same privacy budget ([Split-and-Denoise](https://arxiv.org/abs/2310.09130)).
- **InferDPT:** Black-box LLM inference via perturbed prompts and RANTEXT ([InferDPT](https://arxiv.org/abs/2310.12214)).
- **Privacy auditing:** Tight auditing for DP in in-context learning (membership inference, Gaussian DP) ([Tight and Practical Privacy Auditing](https://arxiv.org/abs/2511.13502)).

**Summary:** DP in inference is used for token- or user-level guarantees; trade-off is controlled by ε (stronger privacy, lower utility).

---

### 1.6 Zero-Knowledge Proofs for AI Verification

**Training:** zkPoT proves correct training on committed data without revealing model or data; ~15 min prover time per iteration (24× faster than generic recursive), ~1.63 MB proofs, ~130 ms verify ([Zero-Knowledge Proofs of Training for DNNs](https://eprint.iacr.org/2024/162)).

**Inference:** ZK-SNARKs verify that inference was performed correctly. Work on trustless DNN inference reaches **ImageNet-scale**, ~79% top-5, **~1 s** verification ([Scaling up Trustless DNN Inference with ZK](https://openreview.net/forum?id=kmnQYA8snK)). zkSNARKs can attest that a fixed-weights model meets stated performance/fairness ([Verifiable evaluations of ML models using zkSNARKs](https://arxiv.org/abs/2402.02675)).

**Summary:** ZK is used for **verification** (correctness/benchmarks), not primarily for input privacy during inference; useful for decentralized or closed-source model attestation.

---

## 2. What UNFED AI Offers vs Competitors

### UNFED AI’s Stack (from project docs/code)

- **Layer splitting:** LLM layers split across nodes; **no single node sees the full prompt** ([architecture](site/architecture.html)).
- **Tor-style anonymous routing:** X25519 + AES-256-GCM onion encryption; each node peels one layer and sees only the next hop; optional **guard** sees client IP but not payload; compute nodes do not learn client identity ([architecture](site/architecture.html), [TODO](TODO.md)).
- **MPC for embedding (Shard 0):** Token IDs secret-shared between two nodes; neither can reconstruct tokens; only activations leave the MPC pair ([architecture](site/architecture.html)).
- **Spot-check verification:** Random tickets (e.g. 5%) with shard index, input activations, output activation hash; **no user identity or prompt** in tickets; verifier cannot link to who asked what ([protocol](site/protocol.html)).
- **Share-chain economics:** P2Pool-style side-chain recording compute shares (~10 s blocks); proportional payouts from escrow; on-chain staking and slashing ([protocol](site/protocol.html), [economics](site/economics.html)).

### Comparison to Other Approaches

| Dimension | UNFED AI | Bittensor | Gensyn | Phala | Cascade / Fission / Petals |
|-----------|----------|-----------|--------|-------|----------------------------|
| **Privacy model** | No single node sees prompt; MPC on embedding; onion routing | Validators/miners may see prompts; focus on incentive alignment | Training; proofs of learning | TEE: operator cannot read memory | Cascade: token sharding; Fission: MPC+shuffling; Petals: layer split, no crypto privacy |
| **Identity vs query** | Unlinkable (guard + onion: no node has both) | Not designed for anonymity | N/A (training) | TEE hides from operator; identity can be known to gateway | Varies |
| **Verification** | Spot-check re-execution + fraud proofs | Validator scoring (Yuma consensus) | Probabilistic proofs of learning | TEE attestation | Varies |
| **Economics** | Share-chain + escrow + UNFED token; per-token pricing | TAO; proof-of-intelligence; subnets | Gensyn token; training rewards | Phala token; pay-per-request / dedicated GPU | N/A or project-specific |
| **Primary use** | Private, censorship-resistant inference | Decentralized model marketplace / subnets | Decentralized training | Confidential inference (centralized or decentralized) | Research / P2P serving |

**Differentiators for UNFED:**

1. **Prompt never in one place:** Layer sharding + MPC on the embedding layer means the raw prompt exists only as secret shares at the first hop; downstream shards see only activations. Few competitors combine layer splitting with **cryptographic** protection of the embedding step.
2. **Identity–query separation:** Onion routing + guard ensures no node learns both “who” and “what”; TEE-based (e.g. Phala) hides data from the operator but the gateway can still link identity and query.
3. **Lightweight verification:** Spot-checks (e.g. 5%) keep cost low while retaining fraud detection; no per-request ZK (unlike some zkML stacks).
4. **Familiar economics:** Per-token pricing and share-chain feel similar to existing API and mining-pool mental models; integrates with on-chain escrow and staking.

**Gaps vs competitors:** No TEE (so no hardware attestation like Phala); no ZK proof of correctness per request (unlike some verifiable inference projects); no validator-based quality ranking like Bittensor. UNFED’s strength is **privacy and anonymity by architecture**, not maximal verification or hardware trust.

---

## 3. Key Success Factors

### 3.1 What Users/Enterprises Care About

From private-AI and confidential-computing messaging:

- **Prompt and data exposure:** Avoiding sensitive queries and business intelligence leaking to cloud or operators ([Phala](https://phala.com/solutions/private-ai-inference); [Petronella](https://petronellatech.com/blog/enterprise-private-ai-confidential-computing-zero-trust-llms-data/)).
- **Model IP and inference logs:** Protecting weights and usage patterns from providers and admins ([Phala](https://phala.com/solutions/private-ai-inference)).
- **Control and compliance:** Data residency, zero-trust, and “AI on your data” ([SambaNova](https://sambanova.ai/solutions/developers-enterprise); [VMware](https://www.vmware.com/solutions/cloud-infrastructure/private-ai)).

So: **privacy/confidentiality**, **latency/cost**, and **ease of use** all matter; for enterprises, compliance and control often lead.

### 3.2 Latency Overhead of Privacy Approaches

- **HE:** Often **100×+** slower than plaintext; optimizations (pruning, equivariant encryption) aim to reduce this ([Towards Fast and Scalable Private Inference](https://export.arxiv.org/pdf/2307.04077v1.pdf); [Equivariant Encryption](https://arxiv.org/abs/2502.01013)).
- **MPC:** Lower compute overhead than HE but **round-trip and serialization** add latency; pipelining (e.g. 12–14% gain) and mixed secret-sharing (1.5–4.3× in WAN) help ([Fast and Scalable Private Inference](https://export.arxiv.org/pdf/2307.04077v1.pdf); [Mixed Secret Sharings](https://eprint.iacr.org/2026/038)).
- **TEE:** Phala reports **~99%** of native performance for large models on GPU TEE ([Phala Benchmark](https://docs.phala.network/confidential-ai-inference/benchmark)).
- **Partitioning (no crypto):** Petals/Cascade-style can be **faster** than single-node offload (e.g. 10×) by better parallelism; UNFED adds MPC only at embedding and onion routing, so most of the path is “just” distributed layers plus one MPC step.

**Takeaway:** TEE and smart partitioning have the smallest latency tax; full HE is the heaviest; MPC is in between and improvable with pipelining and protocol design.

### 3.3 Real Paying Demand for Private AI Inference

- **Vendors and use cases:** SambaNova lists enterprise and managed inference (e.g. Maitai, Hume.ai, Aion Labs) ([SambaNova](https://sambanova.ai/solutions/developers-enterprise)). Phala offers confidential LLM serving with zero-logging and OpenAI-compatible API ([Phala](https://phala.com/solutions/private-ai-inference)). Azure and others are pushing confidential AI as part of responsible AI ([Azure](https://techcommunity.microsoft.com/blog/azureconfidentialcomputingblog/azure-ai-confidential-inferencing-technical-deep-dive/4253150)).
- **Drivers:** Regulation, IP, and “no prompt logging” are concrete demands; the market is still early but enterprises are paying for confidential and private AI today.

### 3.4 Centralized vs Decentralized Inference: Quality/Speed Gap

- Decentralized inference adds **network hops**, **coordination**, and possibly **MPC/onion** cost; quality depends on model and node behavior.
- Bittensor/Gensyn-style networks focus on **incentives and verification** (validators, proofs) rather than matching a single data center’s tail latency.
- UNFED’s design (local layer execution, one MPC stage, onion) aims to keep the gap small vs a single data center, at the cost of complexity and multi-hop latency.

**Summary:** There is real demand for private/confidential inference; latency and ease of use are differentiators; decentralized systems can compete where privacy, censorship resistance, or distribution matter more than minimal latency.

---

## 4. Risks and Challenges

### 4.1 Technical Barriers

- **Complexity:** MPC, onion routing, and share-chain require operational and integration effort; debugging and monitoring are harder than with a single API.
- **Performance:** Any crypto (MPC, HE) or extra hops add latency; adoption depends on keeping overhead acceptable (e.g. MPC only at embedding, rest in plain distributed layers).
- **Correctness and security:** Protocol bugs, weak spot-check rates, or misconfigured MPC could leak data or allow fraud; formal analysis and audits help.

### 4.2 Regulatory Risks for Crypto-Based AI

- **SEC and “AI + crypto”:** Projects that combine AI with tokens can face scrutiny under both crypto and AI narratives; former SEC Chair Gensler has highlighted AI’s impact on markets and “bad actors,” implying **double scrutiny** for AI tokens ([The Block](https://www.theblock.co/post/245414/ai-tokens-may-risk-double-scrutiny-after-gary-gensler-takes-aim-at-artificial-intelligence)).
- **Disclosure:** AI disclosure under securities law is tricky (probabilistic, time-varying, third-party systems); frameworks like “Reasoning Claim Tokens” are being proposed ([Zenodo](https://zenodo.org/records/18113674)).
- **Uncertainty:** Broader crypto regulation (e.g. “Project Crypto,” Howey-based taxonomy) affects any token used for staking, escrow, or rewards ([SEC](https://www.sec.gov/newsroom/speeches-statements/atkins-111225-securities-exchange-commissions-approach-digital-assets-inside-project-crypto)).

**Implication for UNFED:** Token (UNFED) used for staking, escrow, and payments may be subject to securities and AI-related disclosure; design choices (utility vs investment, jurisdiction) matter.

### 4.3 Competition from Big Tech Privacy Features

- **Confidential computing:** Azure, AMD, Intel are making TEE-based confidential AI a standard option ([Azure](https://techcommunity.microsoft.com/blog/azureconfidentialcomputingblog/azure-ai-confidential-inferencing-technical-deep-dive/4253150); [AMD](https://www.amd.com/en/products/processors/server/epyc/confidential-computing.html)).
- **Enterprise trust:** Many enterprises will prefer “private AI” from incumbent clouds (familiar contracts, compliance, SLAs) over a decentralized token-based network.
- **Differentiation:** UNFED’s edge is **no single party** (including cloud) seeing prompts, **anonymity** (onion + guard), and **censorship resistance**, not only “encrypted in use” in one provider’s TEE.

### 4.4 Network Bootstrapping (Chicken-and-Egg)

- Decentralized AI needs **supply** (nodes, models) and **demand** (clients); each side waits on the other ([BlockEden](https://blockeden.xyz/blog/2025/07/28/decentralized-ai-inference-markets)).
- **Token incentives** are a common fix: reward early nodes/suppliers until usage and liquidity grow; Helium is cited as an example (e.g. 390k+ nodes) ([Future.com](https://future.com/the-web3-playbook-using-token-incentives-to-bootstrap-new-networks/)).
- UNFED’s share-chain and staking are a **reward mechanism** for compute; liquidity bootstrapping (e.g. LBP-style as with Olas) or partnerships could help ([Olas FAQ](https://olas.network/faq)).

**Summary:** Bootstrapping is a known problem; UNFED’s economics (shares, escrow, staking) are aligned with incentivizing supply; demand depends on clear privacy/anonymity value prop and distribution.

---

## Summary Table: Privacy Technologies at a Glance

| Approach | Latency/overhead | Adoption / users | Best for |
|----------|------------------|------------------|----------|
| **HE** | 100×+ (improving with pruning/equivariant) | Research; early transformers | Non-interactive, single-party encrypted inference |
| **MPC** | Moderate (rounds, pipelining helps) | Meta (CrypTen); research; UNFED (embedding) | Multi-party private inference without full HE |
| **TEE** | ~1% (e.g. Phala 99% native) | Azure, Phala, AMD/Intel ecosystem | “Encrypted in use” in cloud or decentralized TEE |
| **Partitioning** | Can be faster (parallelism) | Cascade, Fission, Petals, UNFED | Scale + privacy when combined with crypto or shuffling |
| **DP (inference)** | Depends on ε and method | Research; document/PII redaction | Bounded influence of sensitive tokens/users |
| **ZK** | Prover cost high; verify ~1 s | Research; verifiable inference/benchmarks | Correctness attestation, not input privacy |

---

## Sources

- [AMD Confidential Computing](https://www.amd.com/en/products/processors/server/epyc/confidential-computing.html)
- [Azure AI Confidential Inferencing: Technical Deep-Dive](https://techcommunity.microsoft.com/blog/azureconfidentialcomputingblog/azure-ai-confidential-inferencing-technical-deep-dive/4253150)
- [Cascade: Token-Sharded Private LLM Inference](https://arxiv.org/html/2507.05228v1)
- [Comparison of Fully Homomorphic Encryption and Garbled Circuit Techniques in Privacy-Preserving ML](https://arxiv.org/html/2510.07457v1)
- [Confidential VMs: AMD SEV-SNP and Intel TDX](https://dl.acm.org/doi/10.1145/3700418)
- [CrypTen: Secure Multi-Party Computation Meets Machine Learning | Meta](https://ai.meta.com/research/publications/crypten-secure-multi-party-computation-meets-machine-learning/)
- [CrypTen NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html)
- [Decentralized AI Inference Markets: Bittensor, Gensyn, Cuckoo AI](https://blockeden.xyz/blog/2025/07/28/decentralized-ai-inference-markets)
- [DP-Fusion: Token-Level Differentially Private Inference for LLMs](https://arxiv.org/abs/2507.04531)
- [Encrypted Large Model Inference: Equivariant Encryption Paradigm](https://arxiv.org/abs/2502.01013)
- [Enterprise Private AI: Confidential Computing, Zero-Trust LLMs](https://petronellatech.com/blog/enterprise-private-ai-confidential-computing-zero-trust-llms-data-residency/)
- [Fission: Distributed Privacy-Preserving LLM Inference](https://eprint.iacr.org/2025/653)
- [Future.com – Token incentives to bootstrap networks](https://future.com/the-web3-playbook-using-token-incentives-to-bootstrap-new-networks/)
- [HTCNN: High-Throughput Batch CNN Inference with Homomorphic Encryption](https://eprint.iacr.org/2024/1753)
- [Lightchain AI vs Bittensor vs Gensyn](https://cryptoweekly.co/lightchain-vs-bittensor-vs-gensyn/)
- [Converting Transformers to Polynomial Form for Secure Inference Over HE](https://proceedings.mlr.press/v235/zimerman24a.html)
- [PIGEON: High Throughput Framework for Private Inference using MPC](https://eprint.iacr.org/2024/1371)
- [Phala Confidential AI Inference](https://docs.phala.network/overview/phala-network/confidential-ai-inference)
- [Phala Benchmark](https://docs.phala.network/confidential-ai-inference/benchmark)
- [Phala Private AI](https://phala.com/solutions/private-ai-inference)
- [Phala GPU TEE Deep Dive](https://phala.network/posts/Phala-GPU-TEE-Deep-Dive)
- [PlanetServe: Decentralized, Privacy-Preserving LLM Serving](https://arxiv.org/abs/2504.20101)
- [SambaNova – Private AI for Enterprise](https://sambanova.ai/solutions/developers-enterprise)
- [Scalable Honest-majority MPC for ML from Mixed Secret Sharings](https://eprint.iacr.org/2026/038)
- [Scaling up Trustless DNN Inference with ZK](https://openreview.net/forum?id=kmnQYA8snK)
- [SEC Approach to Digital Assets: Project Crypto](https://www.sec.gov/newsroom/speeches-statements/atkins-111225-securities-exchange-commissions-approach-digital-assets-inside-project-crypto)
- [Split-and-Denoise: Protect LLM inference with local DP](https://arxiv.org/abs/2310.09130)
- [The Block – AI tokens double scrutiny](https://www.theblock.co/post/245414/ai-tokens-may-risk-double-scrutiny-after-gary-gensler-takes-aim-at-artificial-intelligence)
- [Third Earth – Decentralized TEE and Phala](https://thirdearth3.medium.com/decentralized-tee-cpus-gpus-for-ai-agents-advantages-challenges-and-phala-networks-approach-16e63bcc3a2d)
- [Towards Fast and Scalable Private Inference](https://export.arxiv.org/pdf/2307.04077v1.pdf)
- [Verifiable evaluations of ML models using zkSNARKs](https://arxiv.org/abs/2402.02675)
- [VMware Private AI](https://www.vmware.com/solutions/cloud-infrastructure/private-ai)
- [Zero-Knowledge Proofs of Training for Deep Neural Networks](https://eprint.iacr.org/2024/162)
- [Zenodo – Reasoning Claim Tokens and AI Disclosure](https://zenodo.org/records/18113674)
- UNFED AI project: `site/architecture.html`, `site/protocol.html`, `site/economics.html`, `site/about.html`, `TODO.md`
