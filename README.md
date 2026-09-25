# DVLM: Decentralized Vision-Language Models

Run VLM inference across a swarm of independently owned, heterogeneous machines:

- **Vision encoders** (ViT + projector) are *stateless, replicated* peers. Images in a
  prompt are spread across them in parallel, and results are cached by image content.
- **The language decoder is pipeline-sharded.** Each peer serves a contiguous span of layers
  and keeps a KV cache for each session.
- **The client** keeps the token embeddings, LM head and sampling, so prompt text and output
  tokens never leave it. It routes hidden states through the cheapest chain of peers, and
  when a peer dies it **re-routes and replays** that hop's history to the replacements.

**DreamerV3** is also supported as a second model family on the same swarm. The CNN
observation encoder runs on encoder peers, with a batch of environments split across them. The
RSSM's recurrent latent state lives on `rssm` peers. The actor, critic, reward and continue
heads run on the client. All sampling is seeded, so if an RSSM peer dies, replaying the logged
ops to a replacement reproduces its state exactly.

See [docs/RESEARCH.md](docs/RESEARCH.md) for the state-of-the-art survey and
[docs/DESIGN.md](docs/DESIGN.md) for the design decisions and roadmap.

## Quick start

```bash
pip install -e ".[dev]"

# Whole swarm in one process (tiny random model, no downloads)
dvlm demo --encoders 2 --spans 3

# Real multi-process / multi-machine swarm
dvlm make-tiny ./tiny --layers 12                    # or use a HF checkpoint, e.g. HuggingFaceTB/SmolVLM-256M-Instruct
dvlm registry --port 7700
dvlm serve --model ./tiny --registry HOST:7700 --role encoder
dvlm serve --model ./tiny --registry HOST:7700 --num-layers 6   # picks the least-served layers automatically
dvlm serve --model ./tiny --registry HOST:7700 --layers 6:12    # or choose explicitly
dvlm status --registry HOST:7700
dvlm generate --model ./tiny --registry HOST:7700 --input-ids 1,7,8,9,100,100,100,100,10,11 --random-images 1

# With a real checkpoint + processor:
dvlm generate --model HuggingFaceTB/SmolVLM-256M-Instruct --registry HOST:7700 --image cat.jpg --prompt "What is this?"
```

DreamerV3:

```bash
dvlm demo --arch dreamerv3 --batch 8 --steps 20 --horizon 15
dvlm make-tiny ./dreamer --arch dreamerv3
dvlm serve --model ./dreamer --registry HOST:7700 --role encoder
dvlm serve --model ./dreamer --registry HOST:7700 --role rssm
```

```python
from dvlm.dreamer.client import DistributedDreamer

agent = DistributedDreamer.from_checkpoint("./dreamer", "HOST:7700")
policy = agent.policy(batch_size=8, seed=0)
action = await policy.act(obs_uint8)          # [B, C, H, W] -> [B, A]
dream = await policy.imagine(horizon=15)      # features, actions, rewards, continues, values
```

Use `--public-host` when peers are on different machines. Every node loads only the tensors it
serves (for Hub checkpoints, only the shards that hold them).

## Layout

| Module | Role |
|---|---|
| `dvlm/parts.py` | `DecoderSpan` (layers `[start, end)` + RoPE, causal mask over the KV cache) and `ClientHead` (embed, splice image embeddings, norm + LM head) |
| `dvlm/adapters/` | Per-architecture split. Currently SmolVLM / Idefics3 (Llama decoder) |
| `dvlm/checkpoint.py` | Loads only the safetensors keys a node needs |
| `dvlm/wire.py`, `dvlm/rpc.py` | Length-prefixed msgpack frames with raw tensor bytes; multiplexed async RPC |
| `dvlm/registry.py` | Soft-state tracker (announce with TTL); can be swapped for a DHT |
| `dvlm/routing.py` | Dijkstra over layer boundaries (RTT + compute / throughput), automatic span placement |
| `dvlm/server.py` | `SpanWorker`, `EncoderWorker`, `Server` (RPC + heartbeat) |
| `dvlm/dreamer/` | DreamerV3 model, adapter, `RSSMWorker`, `Policy` with local/swarm backends and op-log replay failover |
| `dvlm/client.py` | `DistributedVLM.generate`, `InferenceSession` (replay failover), parallel encoding |

## Tests

`pytest` covers:

- Sharded parts vs. the monolithic HF model: exact encoder output, prefill logits for several
  split points, and greedy tokens.
- The same equivalence across a real localhost TCP swarm.
- DreamerV3: the swarm matches the in-process model exactly (discrete and continuous actions,
  mid-batch episode resets, imagination). RSSM failover during observe or imagine replays
  exactly.
- Failover: when a node dies, generation re-routes (including onto different layer
  boundaries), survives two failures in a row, fails over between encoders, and reuses the
  encoder cache.
