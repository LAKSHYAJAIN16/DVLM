# State of the Art: Decentralized / Distributed VLM Inference

_Survey date: September 2026._

## 1. Problem statement

A vision-language model (VLM) has three parts:

1. **Vision encoder** (a ViT, e.g. SigLIP or a CLIP-style model): turns pixels into patch features.
   It is compute-bound and has no state between requests.
2. **Projector / modality adapter**: maps the patch features into the LLM's embedding space.
   It is small.
3. **LLM decoder**: does **prefill** over text + visual tokens (compute-bound; images add hundreds to
   thousands of tokens each) and then autoregressive **decode** (memory-bandwidth-bound, with a
   KV cache that grows as it goes).

The goal is to run this on many independently owned, heterogeneous machines connected over ordinary
networks (the public internet, with 10–200 ms RTT and tens of Mbit/s), in parallel. Two
properties should come out of that:

- **Capacity:** a model too large for any single participant can still run.
- **Throughput:** many requests run at once across the swarm.

## 2. What exists today

### 2.1 Decentralized LLM inference (text-only)

| System | Parallelism | Scheduling / routing | Notes |
|---|---|---|---|
| **Petals** (BigScience/HF, 2022–) | Pipeline: each server holds a contiguous span of transformer blocks | Servers announce their blocks in a DHT (hivemind). A new server picks the span with the worst throughput and servers periodically rebalance. The client builds a latency graph and picks the fastest chain with shortest-path routing, taking free KV-cache memory into account. | The client keeps the activations it sent to each server so it can replay them to a replacement when a server dies. Reported ~6 tok/s single-batch on Llama-2-70B. Public-swarm abuse is a known problem. |
| **Exo** | Pipeline, with layers partitioned by each device's memory share | Automatic discovery on the LAN, topology-aware placement | Aimed at *your own* devices (Apple Silicon/MLX). Not an adversarial setting. |
| **Parallax** (Gradient, 2025) | Pipeline | Two-phase scheduler: (i) *model allocation* places layers of each replica across heterogeneous GPUs under memory and link-bandwidth limits; (ii) *request-time pipeline selection* stitches layers from **different replicas** into a chain that balances load. | Reported lower latency and higher throughput than decentralized baselines (Petals). The current best published scheduler for this setting. |
| **Prime Intellect** (prime-iroh / prime-vllm / prime-pipeline, 2025) | Pipeline over the public internet | Micro-batching designed for **batched decoding** to keep pipeline stages busy despite 100 ms+ latency | P2P transport is iroh (QUIC with NAT traversal). Ran SYNTHETIC-2 with 1,250+ GPUs joining within 3 days. |
| **prima.cpp** (2025) | Pipeline across home devices | Handles heterogeneous, low-resource clusters | Runs 30–70B models on home clusters. |

**Consensus:** over slow, unreliable links, every serious system uses **pipeline parallelism**
(layer sharding). Tensor parallelism needs an all-reduce every layer, which only works with
NVLink or InfiniBand, so it's out. The open problems are scheduling and placement across
heterogeneous nodes, hiding latency, fault tolerance, and trust.

### 2.2 Disaggregated VLM serving (datacenter)

- **EPD disaggregation** (Encode / Prefill / Decode) is now in vLLM (Dec 2025), SGLang (Jan
  2026, plus a heterogeneous CPU+GPU variant in Jun 2026), NVIDIA Dynamo, and llm-d. The vision
  encoder runs on its **own pool of workers**, and embeddings are shipped to the prefill workers.
  Reported gains: up to ~5× TTFT and ~7× end-to-end on image-heavy prompts; ~31–42% TTFT
  reduction in mixed traffic because it removes head-of-line blocking.
- Encoding and decoding have *opposite* hardware profiles (compute-bound vs. bandwidth-bound).
  That makes it natural to run them on different hardware, including cheaper GPUs or CPUs for
  the encoder.
- Visual tokens dominate prefill: for example, 20 images at 480p come to ~8.3k LLM input tokens.
  Visual token pruning and compression is an active area of work that reduces both compute and
  the size of the activations that have to be sent over the network.

**Gap:** we found no decentralized, peer-to-peer system that treats VLMs as a first-class
workload. Petals, Parallax, Exo and Prime Intellect all shard *LLM* layers. EPD disaggregation
exists only inside datacenters on RDMA. **Combining EPD-style encoder disaggregation with
Petals/Parallax-style pipeline sharding over a P2P swarm is the gap this project targets.**

### 2.3 Hiding latency

- **Micro-batching / continuous batching across sessions** keeps every pipeline stage busy
  (Prime Intellect, gLLM).
- **Speculative decoding in pipelines:** SpecInfer (tree verification), PipeInfer (asynchronous
  speculation fills pipeline bubbles, up to 2.15×), FlowSpec, SpecPipe, and *Decentralized
  Speculative Decoding* (2025), which verifies several candidate tokens per network round trip
  and so turns latency into useful work.
- **Prefix-cache-aware routing** in P2P swarms (2026): each node keeps a radix tree of cached
  prefixes, and requests go to the node with the longest match. Stale metadata only causes
  cache misses, never wrong output.

### 2.4 Bandwidth: compressing activations

Hidden states are `seq × d_model` per hop. That's large during prefill (thousands of visual
tokens) and small during decode (one token). Current work: TAH-Quant (tile-wise quantization
plus Hadamard rotation to handle outliers), reference-aware activation compression (RAC, 2026),
learned subspace compression, and ResBM residual bottlenecks. Naive int8 on activations loses
quality because of outliers. fp16/bf16 is the safe default, and compression is an opt-in
optimization.

### 2.5 Trust and verification

- **TOPLOC** (ICML 2025): locality-sensitive hashes of intermediate activations, about 258 bytes
  per 32 tokens. In their evaluations it detects a swapped model, prompt or precision with no
  false positives or negatives, and it tolerates differences between GPUs and reordered
  arithmetic.
- **Canary activation-drift test** (Jul 2026): secret canary requests whose correct activations
  are precomputed are mixed into normal traffic, and the verifier measures drift at each hop.
  Reported AUROC 1.0 at identifying the node that tampered.
- **VeriLLM** (2025): lightweight, publicly verifiable decentralized inference.
- Privacy: in pipeline sharding, intermediate nodes see hidden states, which can be partly
  inverted back to the input. The first stage sees raw input unless the client runs it. Petals
  lets the client run embeddings and the LM head locally, and we'll do the same.

## 3. Takeaways that drive the design

1. Use **pipeline parallelism** for the LLM, the same foundation as all prior decentralized systems.
2. **Disaggregate the vision encoder** (EPD). It's stateless, so it's easy to replicate, retry,
   spot-check and cache by content hash. It can also run on weaker nodes.
3. **The client owns the embedding layer and the LM head** (Petals). This protects the prompt and
   output text from servers and lets the client splice visual embeddings in locally.
4. **Routing and placement** start Petals-style (announce spans, client picks the fastest chain)
   and evolve toward Parallax's two-phase scheduler.
5. **Fault tolerance** comes from client-side replay of each hop's inputs.
6. Later: batching across sessions, speculative decoding, activation compression, and
   TOPLOC/canary verification.

## Sources

- Petals: [arXiv 2209.01188](https://arxiv.org/pdf/2209.01188), [arXiv 2312.08361](https://arxiv.org/html/2312.08361), [v2.0 release notes](https://github.com/bigscience-workshop/petals/releases/tag/v2.0.0.post1)
- [SharedLLM vs Petals vs Exo vs Kalavai (2026)](https://sharedllm.org/blog/sharedllm-vs-petals-vs-exo.html)
- Prime Intellect: [Planetary-Scale Inference](https://www.primeintellect.ai/blog/inference), [prime-pipeline](https://github.com/PrimeIntellect-ai/prime-pipeline), [prime-vllm](https://github.com/PrimeIntellect-ai/prime-vllm), [SYNTHETIC-2](https://www.primeintellect.ai/blog/synthetic-2-release)
- Parallax: [arXiv 2509.26182](https://arxiv.org/abs/2509.26182), [GitHub](https://github.com/GradientHQ/parallax)
- [prima.cpp, arXiv 2504.08791](https://arxiv.org/pdf/2504.08791)
- [gLLM, arXiv 2504.14775](https://arxiv.org/pdf/2504.14775)
- EPD: [arXiv 2501.05460](https://arxiv.org/pdf/2501.05460), [vLLM blog](https://vllm.ai/blog/2025-12-15-vllm-epd), [LMSYS SGLang EPD](https://www.lmsys.org/blog/2026-01-12-epd/), [LMSYS hetero EPD](https://www.lmsys.org/blog/2026-06-01-hetero-epd/), [NVIDIA](https://developer.nvidia.com/blog/when-to-use-encode-prefill-decode-disaggregation-to-accelerate-multimodal-model-serving/), [Dynamo docs](https://docs.nvidia.com/dynamo/v1.2.1/user-guides/multimodal/encoder-disaggregation)
- VLM efficiency surveys: [arXiv 2604.05546](https://arxiv.org/html/2604.05546v2), [arXiv 2603.27960](https://arxiv.org/html/2603.27960)
- Speculative decoding: [SpecInfer](https://arxiv.org/pdf/2305.09781), [PipeInfer](https://arxiv.org/abs/2407.11798), [FlowSpec](https://arxiv.org/html/2507.02620v1), [SpecPipe](https://arxiv.org/abs/2504.04104), [Decentralized Speculative Decoding](https://arxiv.org/abs/2511.11733)
- [Prefix-cache-aware P2P routing, arXiv 2606.17059](https://arxiv.org/abs/2606.17059)
- Activation compression: [TAH-Quant](https://arxiv.org/html/2506.01352v2), [RAC](https://arxiv.org/pdf/2608.04991), [Learned Subspace Compression](https://arxiv.org/pdf/2606.05484), [ResBM](https://arxiv.org/pdf/2604.11947), [Internet-scale comm. optimization](https://arxiv.org/pdf/2604.21072)
- Verification: [TOPLOC](https://arxiv.org/abs/2501.16007), [Canary integrity, arXiv 2607.19490](https://arxiv.org/abs/2607.19490), [VeriLLM](https://arxiv.org/pdf/2509.24257), [Equilibrium: State of Verifiable Inference](https://equilibrium.co/writing/state-of-verifiable-inference)
