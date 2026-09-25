"""Swarm servers: a worker (model part + per-session state) exposed over RPC and announced to the registry."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from .parts import DecoderSpan
from .registry import ServerInfo
from .rpc import ConnectionPool, RpcServer

log = logging.getLogger(__name__)


@dataclass
class _Session:
    cache: DynamicCache
    start: int
    end: int
    length: int = 0
    last_used: float = field(default_factory=time.monotonic)


class SpanWorker:
    """Serves decoder layers [span.start, span.end) with one KV cache per (session, entry layer)."""

    kind = "span"

    def __init__(self, span: DecoderSpan, max_sessions: int = 64, session_timeout: float = 600.0):
        self.span = span
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: dict[str, _Session] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="span")

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def end(self) -> int:
        return self.span.end

    def handlers(self):
        return {"forward": self.forward, "close": self.close, "info": self.info}

    async def forward(
        self, session_id: str, hidden: torch.Tensor, position_ids: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        # A client may use one server for two disjoint layer ranges after re-routing, so the
        # cache is keyed by (session, entry layer).
        key = f"{session_id}:{start}"
        sess = self.sessions.get(key)
        if sess is None:
            if len(self.sessions) >= self.max_sessions:
                self.gc()
                if len(self.sessions) >= self.max_sessions:
                    raise RuntimeError("server is at capacity")
            sess = self.sessions[key] = _Session(self.span.new_cache(), start, end)
        elif (sess.start, sess.end) != (start, end):
            raise ValueError(f"session uses layers [{sess.start}, {sess.end}), got [{start}, {end})")
        if hidden.shape[1] != position_ids.shape[-1]:
            raise ValueError("hidden states and position ids disagree on sequence length")
        sess.last_used = time.monotonic()
        past = sess.length

        def run():
            return self.span(hidden, position_ids, sess.cache, past, start, end)

        out = await asyncio.get_running_loop().run_in_executor(self._executor, run)
        sess.length += hidden.shape[1]
        return out

    async def close(self, session_id: str) -> None:
        for key in [k for k in self.sessions if k.split(":", 1)[0] == session_id]:
            del self.sessions[key]

    async def info(self) -> dict:
        return {"start": self.start, "end": self.end, "sessions": len(self.sessions)}

    def gc(self) -> None:
        cutoff = time.monotonic() - self.session_timeout
        for key in [k for k, s in self.sessions.items() if s.last_used < cutoff]:
            del self.sessions[key]

    def measure_throughput(self, steps: int = 8) -> float:
        """Decode steps per second through the whole span (announced for routing)."""
        hidden_size = self.span.config.hidden_size
        cache = self.span.new_cache()
        x = torch.zeros(1, 1, hidden_size)
        self.span(x, torch.zeros(1, 1, dtype=torch.long), cache, 0)  # warm-up
        t0 = time.perf_counter()
        for i in range(1, steps + 1):
            self.span(x, torch.tensor([[i]]), cache, i)
        return steps / (time.perf_counter() - t0)


class EncoderWorker:
    """Stateless vision tower + projector: pixels in, LLM-space visual embeddings out."""

    kind = "encoder"
    start = end = 0

    def __init__(self, encoder: torch.nn.Module):
        self.encoder = encoder
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="encoder")

    def handlers(self):
        return {"encode": self.encode, "info": self.info}

    async def encode(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, lambda: self.encoder(pixel_values, pixel_attention_mask)
        )

    async def info(self) -> dict:
        return {}

    def gc(self) -> None:
        pass


class Server:
    """Hosts a worker on a TCP port and keeps it announced in the registry."""

    def __init__(
        self,
        worker: SpanWorker | EncoderWorker,
        model: str,
        registry_address: str,
        host: str = "127.0.0.1",
        port: int = 0,
        public_host: str | None = None,
        throughput: float = 1.0,
        ttl: float = 15.0,
    ):
        self.worker = worker
        self.registry_address = registry_address
        self.public_host = public_host or host
        self.ttl = ttl
        self.rpc = RpcServer(worker.handlers(), host=host, port=port)
        self.info = ServerInfo(
            peer_id=uuid.uuid4().hex[:12],
            address="",
            model=model,
            kind=worker.kind,
            start=worker.start,
            end=worker.end,
            throughput=throughput,
        )
        self._pool = ConnectionPool()
        self._heartbeat: asyncio.Task | None = None

    async def start(self) -> None:
        await self.rpc.start()
        self.info.address = f"{self.public_host}:{self.rpc.port}"
        await self._announce()
        self._heartbeat = asyncio.create_task(self._heartbeat_loop())
        log.info("serving %s %s [%d, %d) at %s", self.info.model, self.info.kind, self.info.start, self.info.end, self.info.address)

    async def _announce(self) -> None:
        await self._pool.call(self.registry_address, "announce", info=self.info.to_dict(), ttl=self.ttl)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.ttl / 3)
            self.worker.gc()
            try:
                await self._announce()
            except Exception as e:
                log.warning("registry announce failed: %r", e)

    async def crash(self) -> None:
        """Stop abruptly without deregistering (simulates a node dying)."""
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        await self.rpc.stop()
        await self._pool.close()

    async def stop(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        try:
            await self._pool.call(self.registry_address, "remove", peer_id=self.info.peer_id, timeout=5)
        except Exception:
            pass
        await self.rpc.stop()
        await self._pool.close()
