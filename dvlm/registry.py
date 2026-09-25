"""Peer discovery. v0 is a soft-state tracker: servers re-announce periodically, entries expire.

Kept behind a small interface (`announce` / `remove` / `list`) so it can be swapped for a DHT.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .rpc import RpcServer


@dataclass
class ServerInfo:
    peer_id: str
    address: str
    model: str
    kind: str  # "span" or "encoder"
    start: int = 0
    end: int = 0
    throughput: float = 1.0  # span: full-span decode steps/s; encoder: images/s
    expires: float = field(default=0.0, compare=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ServerInfo":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class Registry:
    def __init__(self, clock=time.monotonic):
        self._servers: dict[str, ServerInfo] = {}
        self._clock = clock

    def announce(self, info: ServerInfo, ttl: float) -> None:
        info.expires = self._clock() + ttl
        self._servers[info.peer_id] = info

    def remove(self, peer_id: str) -> None:
        self._servers.pop(peer_id, None)

    def list(self, model: str | None = None, kind: str | None = None) -> list[ServerInfo]:
        now = self._clock()
        for peer_id in [p for p, s in self._servers.items() if s.expires < now]:
            del self._servers[peer_id]
        return [
            s for s in self._servers.values() if (model is None or s.model == model) and (kind is None or s.kind == kind)
        ]


class RegistryServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.registry = Registry()
        self.rpc = RpcServer(
            {"announce": self._announce, "remove": self._remove, "list": self._list}, host=host, port=port
        )

    @property
    def address(self) -> str:
        return f"{self.rpc.host}:{self.rpc.port}"

    async def start(self) -> None:
        await self.rpc.start()

    async def stop(self) -> None:
        await self.rpc.stop()

    async def _announce(self, info: dict, ttl: float) -> None:
        self.registry.announce(ServerInfo.from_dict(info), ttl)

    async def _remove(self, peer_id: str) -> None:
        self.registry.remove(peer_id)

    async def _list(self, model: str | None = None, kind: str | None = None) -> list[dict]:
        return [s.to_dict() for s in self.registry.list(model, kind)]
