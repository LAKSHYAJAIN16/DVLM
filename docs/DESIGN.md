# DVLM Design: Decisions and Roadmap

See [RESEARCH.md](RESEARCH.md) for the survey behind these decisions.

## Decisions

| # | Question | Decision | Why |
|---|---|---|---|
| D1 | Who runs the nodes? | **An open swarm of heterogeneous peers.** Developed and tested on localhost/LAN first. | "Decentralized" is the goal. A design that works over the internet also works on a LAN, but not the other way round. |
| D2 | Workload | **Inference** (training is out of scope) | Every decentralized system that has shipped does inference first. Training over the internet is a separate research problem. |
| D3 | How to split the model | **Hybrid EPD + pipeline:** vision encoder+projector as a *stateless, replicated* stage; LLM decoder as *pipeline-sharded* layer spans. | The encoder is stateless and compute-bound, so it's cheap to replicate, retry and verify. PP is the only form of LLM sharding that survives internet latency. No existing P2P system does this for VLMs. |
| D4 | What the client runs | **Token embeddings, splicing visual embeddings into the sequence, final norm + LM head, sampling.** | Prompt and output text never leave the client (Petals approach). The client controls sampling and positions. Embeddings and LM head are cheap. |
| D5 | Discovery | **A pluggable `Registry` interface.** v0: an HTTP tracker. Later: a DHT (Kademlia / libp2p / iroh). | Get correctness first. Swap in a DHT without touching the routing code. |
| D6 | Routing | **Client-side:** servers announce `(model, span, throughput, free cache, address)`, and the client picks the minimum-cost chain of spans covering `[0, L)`. | Petals/Parallax phase-2. Later: Parallax-style allocation for phase 1 (where each server places itself). |
| D7 | Transport | **asyncio TCP with length-prefixed msgpack frames and raw tensor bytes**, behind a `Transport` interface. Later: QUIC/iroh for NAT traversal. | Keeps dependencies low and connections persistent for each session, which matters because decode has one round trip per token. |
| D8 | State and fault tolerance | Each server keeps a **KV cache per session** for its span. The **client keeps the inputs it sent to each hop**, so it can replay the history to a replacement server when one fails. | Petals approach, and proven. |
| D9 | Wire format | fp16/bf16 hidden states. Activation compression (int8 + Hadamard) is opt-in later. | Correctness first. The literature shows naive int8 loses quality. |
| D10 | Reference models | **Dev/CI: `HuggingFaceTB/SmolVLM-256M-Instruct`** (Idefics3 with a Llama-style decoder; runs on CPU). **Target: the Qwen2.5-VL / LLaVA family**, through a per-architecture `ModelAdapter`. | CI needs something small enough to run on CPU. The adapter keeps architecture-specific details (M-RoPE position IDs in Qwen-VL, image token layout) out of the core. |
| D11 | Correctness criterion | Distributed greedy generation must produce **the same tokens** as running the whole model in one process, and logits must match within fp tolerance. | This is the one test that proves the sharding is right. |
| D12 | Trust | v0 assumes honest nodes. Plan: encoder spot-recompute, TOPLOC-style activation hashes, canary requests, reputation. | Hard to add safely before the data path is stable, but the protocol reserves fields for proofs now. |

## Architecture

```
            ┌──────────────┐  announce / lookup   ┌───────────────┐
            │   Registry   │◄────────────────────►│ Encoder nodes │  (stateless, replicated)
            │ (tracker→DHT)│                      │ ViT+projector │
            └──────▲───────┘                      └───────▲───────┘
                   │ lookup                               │ pixels → visual embeds
                   │                                      │ (cached by image hash)
┌──────────────────┴──────────────────────────────────────┴───────────┐
│ Client: tokenize → embed text → splice visual embeds → [route]      │
│         → final norm → LM head → sample  (keeps per-hop replay log) │
└──────┬──────────────────────────────────────────────────────▲───────┘
       │ hidden states (session, positions)                   │
       ▼                                                      │
  ┌──────────┐   hidden   ┌──────────┐   hidden   ┌──────────┐│
  │ Decoder  │──────────► │ Decoder  │──────────► │ Decoder  ├┘
  │ layers   │            │ layers   │            │ layers   │
  │ [0,k)    │            │ [k,m)    │            │ [m,L)    │   each keeps a KV cache per session
  └──────────┘            └──────────┘            └──────────┘
```

v0 sends each hop's result back to the client, which forwards it to the next hop (star
topology). This makes replay trivial. Petals v2-style **server-to-server forwarding** saves one
round trip per hop and comes in M3.

## Roadmap

Status: M0–M2 are done. M3 is partly done: replay failover, TTL expiry of dead servers and
automatic span placement all work.

- **M0: Research and design** (this document).
- **M1: Correct sharding, single process.** Split the model into client / encoder / decoder
  spans in memory, and test that the result matches the reference model token for token.
- **M2: Networked swarm on localhost.** Registry, encoder server, span server, transport,
  routing, and a CLI to launch N workers. The same equivalence test runs across processes.
- **M3: Robustness.** ~~Failover and replay~~ (done), ~~heartbeat and expiry~~ (done), server-to-server forwarding, and rebalancing.
- **M4: Throughput.** Batching across sessions and micro-batching, an encoder output cache,
  and prefix-cache-aware routing.
- **M5: Latency.** Speculative decoding (the client drafts, the swarm verifies) and opt-in
  activation compression.
- **M6: Open-swarm readiness.** DHT discovery, QUIC/NAT traversal, TOPLOC/canary
  verification, and reputation.
