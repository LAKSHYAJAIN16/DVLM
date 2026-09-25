"""Client-side routing (which servers compute a request) and server-side span placement."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .registry import ServerInfo

DEFAULT_RTT = 0.05  # seconds, assumed for peers we have not measured yet


class NoRouteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hop:
    peer_id: str
    address: str
    start: int
    end: int


def find_route(
    servers: list[ServerInfo],
    start: int,
    end: int,
    exclude: set[str] = frozenset(),
    rtt: dict[str, float] | None = None,
) -> list[Hop]:
    """Cheapest chain of hops covering layers [start, end).

    A hop may enter a server's span mid-way (any layer in [s.start, s.end)) and runs to the
    end of that span, capped at `end`. Cost per hop = RTT + fraction of span computed / throughput,
    i.e. the estimated wall time of one decode step. Dijkstra over layer boundaries.
    """
    rtt = rtt or {}
    candidates = [s for s in servers if s.kind == "span" and s.peer_id not in exclude and s.end > s.start]
    best = {start: 0.0}
    prev: dict[int, tuple[int, ServerInfo]] = {}
    heap = [(0.0, start)]
    while heap:
        cost, layer = heapq.heappop(heap)
        if layer == end:
            break
        if cost > best.get(layer, float("inf")):
            continue
        for s in candidates:
            if not s.start <= layer < s.end:
                continue
            nxt = min(s.end, end)
            step = rtt.get(s.address, DEFAULT_RTT) + (nxt - layer) / (s.end - s.start) / max(s.throughput, 1e-6)
            if cost + step < best.get(nxt, float("inf")):
                best[nxt] = cost + step
                prev[nxt] = (layer, s)
                heapq.heappush(heap, (cost + step, nxt))
    if end not in best:
        covered = sorted({(s.start, s.end) for s in candidates})
        raise NoRouteError(f"no chain of servers covers layers [{start}, {end}); available spans: {covered}")
    hops = []
    layer = end
    while layer != start:
        from_layer, s = prev[layer]
        hops.append(Hop(s.peer_id, s.address, from_layer, layer))
        layer = from_layer
    return hops[::-1]


def choose_span(servers: list[ServerInfo], num_layers: int, span_len: int) -> tuple[int, int]:
    """Pick the contiguous window of `span_len` layers with the least serving throughput (Petals-style)."""
    span_len = min(span_len, num_layers)
    coverage = [0.0] * num_layers
    for s in servers:
        if s.kind == "span":
            for layer in range(max(s.start, 0), min(s.end, num_layers)):
                coverage[layer] += s.throughput
    best_start = min(
        range(num_layers - span_len + 1),
        key=lambda i: (min(coverage[i : i + span_len]), sum(coverage[i : i + span_len]), i),
    )
    return best_start, best_start + span_len
