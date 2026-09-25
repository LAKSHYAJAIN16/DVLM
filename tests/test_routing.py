import pytest

from dvlm.registry import Registry, ServerInfo
from dvlm.routing import NoRouteError, choose_span, find_route


def span(pid, start, end, tp=1.0):
    return ServerInfo(pid, f"h:{pid}", "m", "span", start, end, tp)


def test_prefers_fewer_hops_when_equal_speed():
    servers = [span("a", 0, 4), span("b", 4, 8), span("c", 0, 2), span("d", 2, 4)]
    route = find_route(servers, 0, 8)
    assert [h.peer_id for h in route] == ["a", "b"]


def test_prefers_faster_servers():
    servers = [span("slow", 0, 8, tp=0.1), span("f1", 0, 4, tp=100), span("f2", 4, 8, tp=100)]
    assert [h.peer_id for h in find_route(servers, 0, 8)] == ["f1", "f2"]


def test_enters_span_midway_and_caps_at_end():
    servers = [span("a", 0, 3), span("b", 2, 8)]
    route = find_route(servers, 0, 8, exclude={"x"})
    assert [(h.peer_id, h.start, h.end) for h in route] == [("a", 0, 3), ("b", 3, 8)]
    sub = find_route(servers, 2, 5, exclude={"a"})
    assert [(h.peer_id, h.start, h.end) for h in sub] == [("b", 2, 5)]


def test_exclusion_and_no_route():
    servers = [span("a", 0, 4), span("b", 4, 8)]
    with pytest.raises(NoRouteError):
        find_route(servers, 0, 8, exclude={"b"})


def test_choose_span_fills_gaps():
    servers = [span("a", 0, 4)]
    assert choose_span(servers, 8, 4) == (4, 8)
    assert choose_span([], 8, 3) == (0, 3)
    assert choose_span(servers + [span("b", 4, 8, tp=5)], 8, 4) == (0, 4)


def test_registry_expiry():
    now = [0.0]
    reg = Registry(clock=lambda: now[0])
    reg.announce(span("a", 0, 4), ttl=10)
    reg.announce(ServerInfo("e", "h:e", "m", "encoder"), ttl=30)
    assert {s.peer_id for s in reg.list("m")} == {"a", "e"}
    assert [s.peer_id for s in reg.list("m", "encoder")] == ["e"]
    assert reg.list("other") == []
    now[0] = 20
    assert [s.peer_id for s in reg.list("m")] == ["e"]
