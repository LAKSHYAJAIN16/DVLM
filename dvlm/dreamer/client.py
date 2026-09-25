"""Acting and imagining with a DreamerV3 whose encoder and RSSM live on swarm peers.

`Policy` holds the agent logic and talks to a backend:
- `LocalBackend`: everything in-process (the reference).
- `SwarmBackend`: observations encoded on encoder peers (batch split across them),
  latent state kept on an rssm peer, with op-log replay if that peer dies.
Both must produce identical actions and imagined trajectories for the same seed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

import torch

from ..checkpoint import Checkpoint
from ..client import PEER_FAILURES, Swarm, default_model_name, encode_on_pool
from ..routing import NoRouteError
from .adapter import DreamerAdapter
from .model import RSSM, DreamerV3, Heads, seeded, step_seed

log = logging.getLogger(__name__)


class LocalBackend:
    def __init__(self, model: DreamerV3):
        self.model = model
        self.states: dict[str, list[torch.Tensor]] = {}

    async def encode(self, obs):
        return self.model.encoder(obs)

    async def new_session(self):
        return uuid.uuid4().hex

    async def observe(self, sid, embed, action, is_first, seed):
        if sid not in self.states:
            self.states[sid] = list(self.model.rssm.initial(embed.shape[0]))
        h, z = self.model.rssm.obs_step(*self.states[sid], action, embed, is_first, seed)
        self.states[sid] = [h, z]
        return RSSM.features(h, z)

    async def imagine(self, sid, action, seed):
        h, z = self.model.rssm.img_step(*self.states[sid], action, seed)
        self.states[sid] = [h, z]
        return RSSM.features(h, z)

    async def fork(self, sid):
        new = uuid.uuid4().hex
        self.states[new] = [t.clone() for t in self.states[sid]]
        return new

    async def close(self, sid):
        self.states.pop(sid, None)


class StatefulSession:
    """A session pinned to one stateful peer. Every successful op is logged; if the peer
    fails, a replacement is chosen and the log is replayed so its state catches up exactly
    (all sampling is seeded, so the replay is deterministic)."""

    def __init__(self, swarm: Swarm, kind: str, ops: list | None = None, max_failures: int = 8):
        self.swarm, self.kind = swarm, kind
        self.ops: list[tuple[str, dict]] = list(ops or [])
        self.max_failures = max_failures
        self.failures = 0
        self.exclude: set[str] = set()
        self.address: str | None = None
        self.sid: str | None = None

    async def _attach(self) -> None:
        servers = [s for s in await self.swarm.servers(self.kind) if s.peer_id not in self.exclude]
        if not servers:
            raise NoRouteError(f"no {self.kind} servers available")
        server = max(servers, key=lambda s: s.throughput)
        self.address, self.peer_id, self.sid = server.address, server.peer_id, uuid.uuid4().hex
        for method, params in self.ops:
            await self.swarm.call(self.address, method, session_id=self.sid, **params)

    def _fail(self, err: Exception) -> None:
        self.failures += 1
        log.warning("%s peer %s failed: %r; re-attaching (failure %d/%d)", self.kind, self.address, err, self.failures, self.max_failures)
        if self.failures > self.max_failures:
            raise RuntimeError(f"too many {self.kind} peer failures") from err
        self.exclude.add(self.peer_id)
        self.address = None

    async def call(self, method: str, **params):
        while True:
            try:
                if self.address is None:
                    await self._attach()
                result = await self.swarm.call(self.address, method, session_id=self.sid, **params)
            except PEER_FAILURES as e:
                self._fail(e)
                continue
            self.ops.append((method, params))
            return result

    async def fork(self) -> "StatefulSession":
        child = StatefulSession(self.swarm, self.kind, self.ops, self.max_failures)
        child.exclude = set(self.exclude)
        if self.address is not None:
            new_sid = uuid.uuid4().hex
            try:  # cheap server-side copy; otherwise the child replays lazily
                await self.swarm.call(self.address, "fork", session_id=self.sid, new_session_id=new_sid)
                child.address, child.peer_id, child.sid = self.address, self.peer_id, new_sid
            except PEER_FAILURES:
                pass
        return child

    async def close(self) -> None:
        if self.address is not None:
            try:
                await self.swarm.call(self.address, "close", session_id=self.sid)
            except PEER_FAILURES:
                pass


class SwarmBackend:
    def __init__(self, swarm: Swarm):
        self.swarm = swarm

    async def encode(self, obs):
        encoders = sorted(await self.swarm.servers("encoder"), key=lambda s: -s.throughput)
        if not encoders:
            raise NoRouteError("no encoder servers available")
        chunks = obs.tensor_split(min(len(encoders), obs.shape[0]))
        parts = await asyncio.gather(*(encode_on_pool(self.swarm, encoders, c, None, offset=i) for i, c in enumerate(chunks)))
        return torch.cat(parts, 0)

    async def new_session(self):
        return StatefulSession(self.swarm, "rssm")

    async def observe(self, sess, embed, action, is_first, seed):
        return await sess.call("observe", embed=embed, action=action, is_first=is_first, seed=seed)

    async def imagine(self, sess, action, seed):
        return await sess.call("imagine", action=action, seed=seed)

    async def fork(self, sess):
        return await sess.fork()

    async def close(self, sess):
        await sess.close()


class Policy:
    """One batch of environments driven by a DreamerV3 agent."""

    def __init__(self, heads: Heads, backend, batch_size: int, seed: int = 0, greedy: bool = False):
        self.heads, self.backend = heads, backend
        self.batch_size, self.seed, self.greedy = batch_size, seed, greedy
        self.cfg = heads.cfg
        self.t = 0
        self.session = None
        self.feat: torch.Tensor | None = None
        self.prev_action = torch.zeros(batch_size, self.cfg.num_actions)
        self._imaginations = 0

    async def act(self, obs: torch.Tensor, is_first: torch.Tensor | None = None) -> torch.Tensor:
        """obs [B, C, H, W] -> action [B, A] (one-hot for discrete actions)."""
        if self.session is None:
            self.session = await self.backend.new_session()
        if is_first is None:
            is_first = torch.full((self.batch_size,), self.t == 0)
        embed = await self.backend.encode(obs)
        self.feat = await self.backend.observe(
            self.session, embed, self.prev_action, is_first.float(), step_seed(self.seed, self.t, 0)
        )
        action = self.heads.act(self.feat, seeded(step_seed(self.seed, self.t, 1)), self.greedy)
        self.prev_action = action
        self.t += 1
        return action

    async def imagine(self, horizon: int = 15) -> dict[str, torch.Tensor]:
        """Dream `horizon` steps ahead from the current state without touching it."""
        if self.feat is None:
            raise RuntimeError("call act() at least once before imagining")
        base = step_seed(self.seed, self.t, 2 + self._imaginations)
        self._imaginations += 1
        fork = await self.backend.fork(self.session)
        feats, actions = [self.feat], []
        try:
            for j in range(horizon):
                a = self.heads.act(feats[-1], seeded(step_seed(base, j, 1)), self.greedy)
                feats.append(await self.backend.imagine(fork, a, step_seed(base, j, 0)))
                actions.append(a)
        finally:
            await self.backend.close(fork)
        feats = torch.stack(feats)
        return {
            "features": feats,
            "actions": torch.stack(actions),
            "rewards": self.heads.predict_reward(feats[1:]),
            "continues": self.heads.predict_continue(feats[1:]),
            "values": self.heads.value(feats),
        }

    async def close(self) -> None:
        if self.session is not None:
            await self.backend.close(self.session)
            self.session = None


class DistributedDreamer:
    def __init__(self, heads: Heads, swarm: Swarm):
        self.heads, self.swarm = heads, swarm
        self.backend = SwarmBackend(swarm)

    @classmethod
    def from_checkpoint(cls, path: str, registry_address: str, model_name: str | None = None, dtype=torch.float32):
        ckpt = Checkpoint(path)
        heads = DreamerAdapter(ckpt.config).build_client(ckpt, dtype)
        return cls(heads, Swarm(registry_address, model_name or default_model_name(path)))

    def policy(self, batch_size: int, seed: int = 0, greedy: bool = False) -> Policy:
        return Policy(self.heads, self.backend, batch_size, seed, greedy)

    async def close(self) -> None:
        await self.swarm.close()
