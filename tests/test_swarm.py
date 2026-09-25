"""M2/M3: generation over a real localhost swarm, including node failures."""

import pytest
import torch

from dvlm.client import DistributedVLM
from dvlm.routing import NoRouteError
from dvlm.tiny import tiny_inputs

from .conftest import EOS, reference_generate

N_NEW = 12


def make_client(tiny_path, swarm):
    return DistributedVLM.from_checkpoint(tiny_path, swarm.registry.address, model_name=swarm.model)


async def generate(vlm, ids, pixels, n=N_NEW):
    return await vlm.generate(ids, pixels, max_new_tokens=n, eos_token_id=EOS)


def crash_on_forward(vlm, server, nth):
    """Crash `server` right before the client's nth forward call to it."""
    calls = {"n": 0, "hops": []}
    real_call = vlm.swarm.call

    async def call(address, method, **params):
        if method == "forward":
            calls["hops"].append((address, params["start"], params["end"]))
        if address == server.info.address and method == "forward":
            calls["n"] += 1
            if calls["n"] == nth:
                await server.crash()
        return await real_call(address, method, **params)

    vlm.swarm.call = call
    return calls


async def test_swarm_matches_reference(swarm, tiny_path, reference):
    await swarm.add("encoder")
    for a, b in [(0, 2), (2, 4), (4, 6)]:
        await swarm.add("span", a, b)
    ids, pixels = tiny_inputs(n_images=2)
    vlm = make_client(tiny_path, swarm)
    try:
        out = await generate(vlm, ids, pixels)
    finally:
        await vlm.close()
    assert out.tolist() == reference_generate(reference, ids, pixels, N_NEW).tolist()
    for s in swarm.servers:  # sessions are closed after generation
        assert getattr(s.worker, "sessions", {}) == {}


async def test_failover_same_boundaries(swarm, tiny_path, reference):
    await swarm.add("encoder")
    primary = await swarm.add("span", 0, 3, throughput=100)
    await swarm.add("span", 0, 3, throughput=1)  # backup
    await swarm.add("span", 3, 6)
    ids, pixels = tiny_inputs()
    vlm = make_client(tiny_path, swarm)
    calls = crash_on_forward(vlm, primary, nth=5)
    try:
        out = await generate(vlm, ids, pixels)
    finally:
        await vlm.close()
    assert calls["n"] == 5
    assert out.tolist() == reference_generate(reference, ids, pixels, N_NEW).tolist()


async def test_failover_reroutes_with_new_boundaries(swarm, tiny_path, reference):
    """Replacement servers split the failed range differently: [0,3) -> [0,2) + [2,3)."""
    await swarm.add("encoder")
    first = await swarm.add("span", 0, 3, throughput=100)
    second = await swarm.add("span", 3, 6, throughput=100)
    low_a = await swarm.add("span", 0, 2, throughput=1)
    low_b = await swarm.add("span", 2, 6, throughput=1)
    ids, pixels = tiny_inputs()
    vlm = make_client(tiny_path, swarm)
    calls = crash_on_forward(vlm, first, nth=4)
    try:
        out = await generate(vlm, ids, pixels)
    finally:
        await vlm.close()
    assert out.tolist() == reference_generate(reference, ids, pixels, N_NEW).tolist()
    hops = set(calls["hops"])
    assert (low_a.info.address, 0, 2) in hops and (low_b.info.address, 2, 3) in hops
    # The surviving [3, 6) server kept its cache and served every step; [3, 6) was never re-routed.
    assert (low_b.info.address, 2, 6) not in hops
    assert sum(1 for h in calls["hops"] if h == (second.info.address, 3, 6)) == N_NEW
    assert second.worker.sessions == {}  # closed at the end


async def test_two_sequential_failures(swarm, tiny_path, reference):
    await swarm.add("encoder")
    a = await swarm.add("span", 0, 6, throughput=100)
    b = await swarm.add("span", 0, 6, throughput=10)
    await swarm.add("span", 0, 6, throughput=1)
    ids, pixels = tiny_inputs()
    vlm = make_client(tiny_path, swarm)
    crash_on_forward(vlm, a, nth=3)
    try:
        # b will be chosen after a dies; kill it too, mid-generation.
        real_call = vlm.swarm.call
        seen = {"n": 0}

        async def call(address, method, **params):
            if address == b.info.address and method == "forward":
                seen["n"] += 1
                if seen["n"] == 4:
                    await b.crash()
            return await real_call(address, method, **params)

        vlm.swarm.call = call
        out = await generate(vlm, ids, pixels)
    finally:
        await vlm.close()
    assert seen["n"] >= 4
    assert out.tolist() == reference_generate(reference, ids, pixels, N_NEW).tolist()


async def test_encoder_failover_and_cache(swarm, tiny_path, reference):
    enc_a = await swarm.add("encoder", throughput=10)
    enc_b = await swarm.add("encoder", throughput=1)
    await swarm.add("span", 0, 6)
    ids, pixels = tiny_inputs(n_images=3)
    vlm = make_client(tiny_path, swarm)
    try:
        await enc_a.crash()
        out = await generate(vlm, ids, pixels)
        assert out.tolist() == reference_generate(reference, ids, pixels, N_NEW).tolist()
        # Second request with the same images is served from the client's encoder cache.
        await enc_b.crash()
        out2 = await generate(vlm, ids, pixels)
        assert out2.tolist() == out.tolist()
    finally:
        await vlm.close()


async def test_images_spread_across_encoders(swarm, tiny_path):
    encoders = [await swarm.add("encoder") for _ in range(3)]
    await swarm.add("span", 0, 6)
    counts = {e.info.address: 0 for e in encoders}
    vlm = make_client(tiny_path, swarm)
    real_call = vlm.swarm.call

    async def call(address, method, **params):
        if method == "encode":
            counts[address] += 1
        return await real_call(address, method, **params)

    vlm.swarm.call = call
    try:
        _, pixels = tiny_inputs(n_images=3)
        await vlm.encode_images(pixels)
    finally:
        await vlm.close()
    assert sorted(counts.values()) == [1, 1, 1]


async def test_no_route(swarm, tiny_path):
    await swarm.add("span", 0, 3)
    vlm = make_client(tiny_path, swarm)
    try:
        with pytest.raises(NoRouteError):
            await vlm.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2)
    finally:
        await vlm.close()
