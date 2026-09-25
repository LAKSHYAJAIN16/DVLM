"""Command line: `dvlm registry | serve | generate | status | demo | make-tiny`.

Roles: `span` (VLM decoder layers), `encoder` (vision / observation encoder), `rssm` (DreamerV3 dynamics)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import torch

from .adapters import get_adapter
from .checkpoint import Checkpoint
from .client import DistributedVLM, Swarm, default_model_name
from .registry import RegistryServer
from .routing import choose_span
from .server import EncoderWorker, Server, SpanWorker

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def build_worker(ckpt: Checkpoint, role: str, start: int = 0, end: int = 0, dtype=torch.float32):
    adapter = get_adapter(ckpt.config)
    if role == "encoder":
        return EncoderWorker(adapter.build_encoder(ckpt, dtype))
    if role == "rssm":
        from .dreamer.server import RSSMWorker

        return RSSMWorker(adapter.build_rssm(ckpt, dtype))
    return SpanWorker(adapter.build_span(ckpt, start, end, dtype))


async def start_server(
    ckpt: Checkpoint, model: str, registry: str, role: str, start: int = 0, end: int = 0, dtype=torch.float32, **kw
) -> Server:
    worker = build_worker(ckpt, role, start, end, dtype)
    throughput = worker.measure_throughput() if role in ("span", "rssm") else 1.0
    server = Server(worker, model, registry, throughput=throughput, **kw)
    await server.start()
    return server


async def _forever():
    await asyncio.Event().wait()


# ---------------------------------------------------------------- commands


async def cmd_registry(args):
    reg = RegistryServer(args.host, args.port)
    await reg.start()
    print(f"registry listening on {reg.address}", flush=True)
    await _forever()


async def cmd_serve(args):
    ckpt = Checkpoint(args.model)
    model = args.model_name or default_model_name(args.model)
    start = end = 0
    if args.role == "span":
        n_layers = get_adapter(ckpt.config).num_layers
        if args.layers:
            start, end = (int(x) for x in args.layers.split(":"))
        else:
            swarm = Swarm(args.registry, model)
            start, end = choose_span(await swarm.servers("span"), n_layers, args.num_layers or n_layers)
            await swarm.close()
    server = await start_server(
        ckpt, model, args.registry, args.role, start, end, DTYPES[args.dtype],
        host=args.host, port=args.port, public_host=args.public_host,
    )
    what = {"span": f"layers [{start}, {end})", "encoder": "encoder", "rssm": "RSSM dynamics"}[args.role]
    print(f"serving {model} {what} at {server.info.address} (peer {server.info.peer_id})", flush=True)
    try:
        await _forever()
    finally:
        await server.stop()


def _load_inputs(args, vlm: DistributedVLM):
    """Returns (input_ids, pixel_values, pixel_attention_mask, decode_fn)."""
    if args.input_ids:
        ids = torch.tensor([[int(t) for t in args.input_ids.split(",")]])
        pixels = None
        if args.random_images:
            size = vlm.adapter.config.vision_config.image_size
            pixels = torch.randn(1, args.random_images, 3, size, size, generator=torch.Generator().manual_seed(0))
        return ids, pixels, None, lambda toks: ",".join(map(str, toks))

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    content = [{"type": "image"} for _ in args.image] + [{"type": "text", "text": args.prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in args.image] or None
    inputs = processor(text=text, images=images, return_tensors="pt")
    return (
        inputs["input_ids"],
        inputs.get("pixel_values"),
        inputs.get("pixel_attention_mask"),
        lambda toks: processor.decode(toks, skip_special_tokens=True),
    )


async def cmd_generate(args):
    vlm = DistributedVLM.from_checkpoint(args.model, args.registry, args.model_name, DTYPES[args.dtype])
    try:
        ids, pixels, mask, decode = _load_inputs(args, vlm)
        eos = vlm.adapter.text_config.eos_token_id
        t0 = time.perf_counter()
        out = await vlm.generate(
            ids, pixels, mask, max_new_tokens=args.max_new_tokens, do_sample=args.sample,
            temperature=args.temperature, top_p=args.top_p, eos_token_id=eos, seed=args.seed,
        )
        new = out[0, ids.shape[1]:].tolist()
        dt = time.perf_counter() - t0
        print(decode(new))
        print(f"[{len(new)} tokens in {dt:.2f}s, {len(new) / dt:.1f} tok/s]", file=sys.stderr)
    finally:
        await vlm.close()


async def cmd_status(args):
    swarm = Swarm(args.registry, args.model_name)
    for kind in ("encoder", "span", "rssm"):
        servers = sorted(await swarm.servers(kind), key=lambda s: (s.model, s.start))
        print(f"{kind} servers: {len(servers)}")
        for s in servers:
            span = f"[{s.start:>3}, {s.end:>3})" if kind == "span" else ""
            print(f"  {s.peer_id}  {s.address:<22} {s.model:<30} {span}  {s.throughput:8.1f}/s")
    await swarm.close()


async def cmd_demo(args):
    """Whole swarm in one process: registry + encoders + span servers, then generate."""
    import tempfile

    from .tiny import make_tiny_smolvlm

    if args.arch == "dreamerv3":
        return await demo_dreamer(args)

    path = args.model or str(make_tiny_smolvlm(tempfile.mkdtemp(prefix="dvlm-tiny-")))
    ckpt = Checkpoint(path)
    model = default_model_name(path)
    n = get_adapter(ckpt.config).num_layers
    reg = RegistryServer()
    await reg.start()
    servers = [await start_server(ckpt, model, reg.address, "encoder") for _ in range(args.encoders)]
    bounds = [round(i * n / args.spans) for i in range(args.spans + 1)]
    for a, b in zip(bounds, bounds[1:]):
        servers.append(await start_server(ckpt, model, reg.address, "span", a, b))
    print(f"swarm up: registry {reg.address}, {args.encoders} encoder(s), spans {list(zip(bounds, bounds[1:]))}")
    args.model, args.registry, args.model_name = path, reg.address, model
    if not args.prompt:
        args.input_ids = args.input_ids or "1,7,8,9," + ",".join(["100"] * 4) + ",10,11,12"
        args.random_images = args.random_images or 1
    try:
        await cmd_generate(args)
    finally:
        for s in servers:
            await s.stop()
        await reg.stop()


async def demo_dreamer(args):
    """Registry + encoders + RSSM servers; run a policy on a batch of synthetic episodes and dream ahead."""
    import tempfile

    from .dreamer.client import DistributedDreamer
    from .tiny import make_tiny_dreamer

    path = args.model or str(make_tiny_dreamer(tempfile.mkdtemp(prefix="dvlm-dreamer-")))
    ckpt = Checkpoint(path)
    model = default_model_name(path)
    reg = RegistryServer()
    await reg.start()
    servers = [await start_server(ckpt, model, reg.address, "encoder") for _ in range(args.encoders)]
    servers += [await start_server(ckpt, model, reg.address, "rssm") for _ in range(args.rssm)]
    print(f"swarm up: registry {reg.address}, {args.encoders} encoder(s), {args.rssm} rssm server(s)")
    agent = DistributedDreamer.from_checkpoint(path, reg.address, model)
    policy = agent.policy(batch_size=args.batch, seed=args.seed or 0)
    gen = torch.Generator().manual_seed(0)
    try:
        t0 = time.perf_counter()
        for _ in range(args.steps):
            obs = torch.randint(0, 256, (args.batch, *ckpt.config.obs_shape), dtype=torch.uint8, generator=gen)
            action = await policy.act(obs)
        dt = time.perf_counter() - t0
        print(f"acted {args.steps} steps x {args.batch} envs in {dt:.2f}s ({args.steps * args.batch / dt:.0f} env-steps/s)")
        print("last actions:", action.argmax(-1).tolist() if ckpt.config.discrete_actions else action.tolist())
        dream = await policy.imagine(args.horizon)
        print(f"imagined {args.horizon} steps: predicted return per env",
              [round(x, 3) for x in dream["rewards"].sum(0).tolist()])
        await policy.close()
    finally:
        await agent.close()
        for s in servers:
            await s.stop()
        await reg.stop()


def cmd_make_tiny(args):
    from .tiny import make_tiny_dreamer, make_tiny_smolvlm

    if args.arch == "dreamerv3":
        print(make_tiny_dreamer(args.out))
    else:
        print(make_tiny_smolvlm(args.out, num_layers=args.layers))


# ---------------------------------------------------------------- parser


def _add_generate_args(p):
    p.add_argument("--prompt", default="Describe this image.")
    p.add_argument("--image", action="append", default=[], help="image path (repeatable)")
    p.add_argument("--input-ids", help="comma-separated token ids (skip the processor, e.g. for tiny models)")
    p.add_argument("--random-images", type=int, default=0, help="with --input-ids: number of random images")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--dtype", choices=DTYPES, default="float32")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dvlm", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("registry", help="run the peer registry (tracker)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7700)

    p = sub.add_parser("serve", help="serve a vision encoder or a span of decoder layers")
    p.add_argument("--model", required=True, help="local checkpoint dir or HF Hub repo id")
    p.add_argument("--model-name", help="swarm-wide model name (default: dir name / repo id)")
    p.add_argument("--registry", required=True)
    p.add_argument("--role", choices=["span", "encoder", "rssm"], default="span")
    p.add_argument("--layers", help="START:END; default: pick the least-served window automatically")
    p.add_argument("--num-layers", type=int, help="window size when choosing automatically")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--public-host", help="address other peers use to reach this server")
    p.add_argument("--dtype", choices=DTYPES, default="float32")

    p = sub.add_parser("generate", help="generate text through the swarm")
    p.add_argument("--model", required=True)
    p.add_argument("--model-name")
    p.add_argument("--registry", required=True)
    _add_generate_args(p)

    p = sub.add_parser("status", help="list servers in the swarm")
    p.add_argument("--registry", required=True)
    p.add_argument("--model-name")

    p = sub.add_parser("demo", help="run a full local swarm in one process and generate")
    p.add_argument("--model", help="checkpoint (default: a tiny random SmolVLM)")
    p.add_argument("--encoders", type=int, default=2)
    p.add_argument("--spans", type=int, default=3)
    p.add_argument("--arch", choices=["smolvlm", "dreamerv3"], default="smolvlm")
    p.add_argument("--rssm", type=int, default=2, help="dreamerv3: number of RSSM servers")
    p.add_argument("--batch", type=int, default=8, help="dreamerv3: parallel environments")
    p.add_argument("--steps", type=int, default=20, help="dreamerv3: environment steps")
    p.add_argument("--horizon", type=int, default=15, help="dreamerv3: imagination horizon")
    _add_generate_args(p)
    p.set_defaults(prompt=None)

    p = sub.add_parser("make-tiny", help="write a tiny random SmolVLM checkpoint")
    p.add_argument("out")
    p.add_argument("--arch", choices=["smolvlm", "dreamerv3"], default="smolvlm")
    p.add_argument("--layers", type=int, default=6)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "make-tiny":
        return cmd_make_tiny(args)
    commands = {"registry": cmd_registry, "serve": cmd_serve, "generate": cmd_generate, "status": cmd_status, "demo": cmd_demo}
    try:
        asyncio.run(commands[args.cmd](args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
