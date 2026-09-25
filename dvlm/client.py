"""Client: owns embeddings + LM head, routes hidden states through the swarm, survives node failures."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch

from .adapters import ModelAdapter, get_adapter
from .checkpoint import Checkpoint
from .parts import ClientHead
from .registry import ServerInfo
from .routing import Hop, NoRouteError, find_route
from .rpc import ConnectionPool, RemoteError

log = logging.getLogger(__name__)

PEER_FAILURES = (ConnectionError, OSError, asyncio.TimeoutError, RemoteError)


def default_model_name(path_or_repo: str) -> str:
    path = Path(path_or_repo)
    return path.resolve().name if path.is_dir() else str(path_or_repo)


class Swarm:
    """The client's view of the network: registry lookups plus pooled peer connections."""

    def __init__(self, registry_address: str, model: str, timeout: float = 60.0):
        self.registry_address = registry_address
        self.model = model
        self.timeout = timeout
        self.pool = ConnectionPool()

    async def servers(self, kind: str) -> list[ServerInfo]:
        rows = await self.pool.call(self.registry_address, "list", model=self.model, kind=kind)
        return [ServerInfo.from_dict(r) for r in rows]

    async def call(self, address: str, method: str, **params):
        return await self.pool.call(address, method, timeout=self.timeout, **params)

    async def close(self) -> None:
        await self.pool.close()


class HopFailed(Exception):
    def __init__(self, hop: Hop, cause: Exception):
        super().__init__(f"{hop.address} [{hop.start}, {hop.end}) failed: {cause!r}")
        self.hop, self.cause = hop, cause


class InferenceSession:
    """One autoregressive session through a chain of span servers.

    Every input a hop receives is kept in `history[hop.start]`. When a server fails, the client
    plans a new route for that hop's layer range and replays the history to the replacements so
    their KV caches catch up, then continues. Servers on other hops are untouched.
    """

    def __init__(self, swarm: Swarm, num_layers: int, max_failures: int = 8):
        self.swarm = swarm
        self.num_layers = num_layers
        self.max_failures = max_failures
        self.session_id = uuid.uuid4().hex
        self.route: list[Hop] | None = None
        self.history: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.exclude: set[str] = set()
        self.failures = 0
        self._hop_sid: dict[Hop, str] = {}
        self._opened: list[tuple[Hop, str]] = []
        self._gen = itertools.count()

    async def _plan(self, start: int, end: int) -> list[Hop]:
        hops = find_route(await self.swarm.servers("span"), start, end, exclude=self.exclude)
        for hop in hops:
            # Fresh server-side session per assignment, so a half-finished replay never pollutes a cache.
            self._hop_sid[hop] = f"{self.session_id}.{next(self._gen)}"
            self._opened.append((hop, self._hop_sid[hop]))
        return hops

    async def _forward(self, hop: Hop, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        try:
            return await self.swarm.call(
                hop.address,
                "forward",
                session_id=self._hop_sid[hop],
                hidden=hidden,
                position_ids=position_ids,
                start=hop.start,
                end=hop.end,
            )
        except PEER_FAILURES as e:
            raise HopFailed(hop, e) from e

    def _record_failure(self, err: HopFailed) -> None:
        self.failures += 1
        self.exclude.add(err.hop.peer_id)
        log.warning("%s; re-routing (failure %d/%d)", err, self.failures, self.max_failures)
        if self.failures > self.max_failures:
            raise RuntimeError(f"too many peer failures in session {self.session_id}") from err

    async def _recover(self, start: int, end: int) -> list[Hop]:
        hops = await self._plan(start, end)
        past = self.history.get(start)
        if past:
            x = torch.cat([h for h, _ in past], dim=1)
            p = torch.cat([p for _, p in past], dim=-1)
            for hop in hops:
                y = await self._forward(hop, x, p)
                self.history[hop.start] = [(x, p)]
                x = y
        return hops

    async def step(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if self.route is None:
            self.route = await self._plan(0, self.num_layers)
        i = 0
        while i < len(self.route):
            hop = self.route[i]
            try:
                out = await self._forward(hop, hidden, position_ids)
            except HopFailed as err:
                self._record_failure(err)
                while True:
                    try:
                        self.route[i : i + 1] = await self._recover(hop.start, hop.end)
                        break
                    except HopFailed as err2:
                        self._record_failure(err2)
                continue
            self.history.setdefault(hop.start, []).append((hidden, position_ids))
            hidden = out
            i += 1
        return hidden

    async def close(self) -> None:
        for hop, sid in self._opened:
            if hop.peer_id in self.exclude:
                continue
            try:
                await self.swarm.call(hop.address, "close", session_id=sid)
            except PEER_FAILURES:
                pass


class DistributedVLM:
    """Generate with a VLM whose vision encoder and decoder layers live on swarm peers."""

    def __init__(self, adapter: ModelAdapter, head: ClientHead, swarm: Swarm, encoder_cache_size: int = 256):
        self.adapter = adapter
        self.head = head
        self.swarm = swarm
        self._encoder_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._encoder_cache_size = encoder_cache_size

    @classmethod
    def from_checkpoint(
        cls, path_or_repo: str, registry_address: str, model_name: str | None = None, dtype: torch.dtype = torch.float32
    ) -> "DistributedVLM":
        ckpt = Checkpoint(path_or_repo)
        adapter = get_adapter(ckpt.config)
        swarm = Swarm(registry_address, model_name or default_model_name(path_or_repo))
        return cls(adapter, adapter.build_client(ckpt, dtype), swarm)

    async def close(self) -> None:
        await self.swarm.close()

    # ---- vision: data-parallel over encoder peers, cached by image content ----

    @staticmethod
    def _image_key(pixels: torch.Tensor, mask: torch.Tensor | None) -> str:
        h = hashlib.sha256(pixels.float().contiguous().numpy().tobytes())
        if mask is not None:
            h.update(mask.to(torch.uint8).contiguous().numpy().tobytes())
        return h.hexdigest()

    async def encode_images(
        self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """[1, n_images, C, H, W] -> [n_images, tokens_per_image, hidden]."""
        if pixel_values.shape[0] != 1:
            raise NotImplementedError("batch size 1 only for now")
        n = pixel_values.shape[1]
        masks = [None if pixel_attention_mask is None else pixel_attention_mask[0, j] for j in range(n)]
        keys = [self._image_key(pixel_values[0, j], masks[j]) for j in range(n)]
        missing = [j for j in range(n) if keys[j] not in self._encoder_cache]

        if missing:
            encoders = sorted(await self.swarm.servers("encoder"), key=lambda s: -s.throughput)
            if not encoders:
                raise NoRouteError("no encoder servers available")
            # Round-robin images over encoders; each image retries on the next encoder if one fails.
            results = await asyncio.gather(
                *(self._encode_one(pixel_values[:, j : j + 1], masks[j], encoders, offset=k) for k, j in enumerate(missing))
            )
            for j, emb in zip(missing, results):
                self._encoder_cache[keys[j]] = emb
                if len(self._encoder_cache) > self._encoder_cache_size:
                    self._encoder_cache.popitem(last=False)
        for key in keys:
            self._encoder_cache.move_to_end(key)
        return torch.cat([self._encoder_cache[k] for k in keys], dim=0)

    async def _encode_one(
        self, pixels: torch.Tensor, mask: torch.Tensor | None, encoders: list[ServerInfo], offset: int
    ) -> torch.Tensor:
        mask = None if mask is None else mask[None, None]
        last_err = None
        for i in range(len(encoders)):
            enc = encoders[(offset + i) % len(encoders)]
            try:
                return await self.swarm.call(enc.address, "encode", pixel_values=pixels, pixel_attention_mask=mask)
            except PEER_FAILURES as e:
                log.warning("encoder %s failed: %r", enc.address, e)
                last_err = e
        raise NoRouteError("all encoder servers failed") from last_err

    # ---- generation ----

    @staticmethod
    def _sample(logits: torch.Tensor, do_sample: bool, temperature: float, top_p: float, gen: torch.Generator | None):
        if not do_sample:
            return logits.argmax(-1, keepdim=True)
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        if top_p < 1.0:
            sorted_p, idx = probs.sort(dim=-1, descending=True)
            drop = sorted_p.cumsum(-1) - sorted_p > top_p
            sorted_p[drop] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        return torch.multinomial(probs, 1, generator=gen)

    async def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | list[int] | None = None,
        seed: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise NotImplementedError("batch size 1 only for now")
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        image_embeds = None
        if pixel_values is not None:
            image_embeds = await self.encode_images(pixel_values, pixel_attention_mask)

        session = InferenceSession(self.swarm, self.adapter.num_layers)
        tokens = input_ids
        try:
            hidden = self.head.embed(input_ids, image_embeds)
            position_ids = self.adapter.position_ids(input_ids, 0)
            past = input_ids.shape[1]
            for _ in range(max_new_tokens):
                out = await session.step(hidden, position_ids)
                next_token = self._sample(self.head.logits(out[:, -1]), do_sample, temperature, top_p, gen)
                tokens = torch.cat([tokens, next_token], dim=1)
                if on_token is not None:
                    on_token(int(next_token))
                if int(next_token) in eos:
                    break
                hidden = self.head.embed(next_token)
                position_ids = self.adapter.position_ids(next_token, past)
                past += 1
        finally:
            await session.close()
        return tokens
