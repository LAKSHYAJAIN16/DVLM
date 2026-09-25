"""RSSM worker: holds one recurrent latent state per (episode batch | imagination fork)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .model import RSSM


class RSSMWorker:
    kind = "rssm"
    start = end = 0

    def __init__(self, rssm: RSSM, max_sessions: int = 1024, session_timeout: float = 600.0):
        self.rssm = rssm
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: dict[str, list] = {}  # sid -> [h, z, last_used]
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rssm")

    def handlers(self):
        return {
            "observe": self.observe,
            "imagine": self.imagine,
            "fork": self.fork,
            "close": self.close,
            "info": self.info,
        }

    async def _run(self, fn):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn)

    def _get(self, session_id: str) -> list:
        try:
            state = self.sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session {session_id}") from None
        state[2] = time.monotonic()
        return state

    async def observe(
        self, session_id: str, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor, seed: int
    ) -> torch.Tensor:
        state = self.sessions.get(session_id)
        if state is None:
            if len(self.sessions) >= self.max_sessions:
                self.gc()
                if len(self.sessions) >= self.max_sessions:
                    raise RuntimeError("server is at capacity")
            h, z = self.rssm.initial(embed.shape[0])
            state = self.sessions[session_id] = [h, z, time.monotonic()]
        state[2] = time.monotonic()
        h, z = await self._run(lambda: self.rssm.obs_step(state[0], state[1], action, embed, is_first, seed))
        state[0], state[1] = h, z
        return RSSM.features(h, z)

    async def imagine(self, session_id: str, action: torch.Tensor, seed: int) -> torch.Tensor:
        state = self._get(session_id)
        h, z = await self._run(lambda: self.rssm.img_step(state[0], state[1], action, seed))
        state[0], state[1] = h, z
        return RSSM.features(h, z)

    async def fork(self, session_id: str, new_session_id: str) -> None:
        h, z, _ = self._get(session_id)
        self.sessions[new_session_id] = [h.clone(), z.clone(), time.monotonic()]

    async def close(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    async def info(self) -> dict:
        return {"sessions": len(self.sessions)}

    def gc(self) -> None:
        cutoff = time.monotonic() - self.session_timeout
        for sid in [s for s, st in self.sessions.items() if st[2] < cutoff]:
            del self.sessions[sid]

    def measure_throughput(self, steps: int = 8) -> float:
        """Batch-1 imagination steps per second."""
        cfg = self.rssm.cfg
        h, z = self.rssm.initial(1)
        action = torch.zeros(1, cfg.num_actions)
        self.rssm.img_step(h, z, action, 0)
        t0 = time.perf_counter()
        for i in range(steps):
            h, z = self.rssm.img_step(h, z, action, i)
        return steps / (time.perf_counter() - t0)
