"""DreamerV3 on the swarm must act and imagine exactly like the in-process model."""

import pytest
import torch
from safetensors.torch import load_file

from dvlm.checkpoint import Checkpoint
from dvlm.dreamer.client import DistributedDreamer, LocalBackend, Policy
from dvlm.dreamer.model import DreamerV3, symexp, symlog, twohot_mean
from dvlm.tiny import make_tiny_dreamer

from .conftest import LocalSwarm

B, STEPS = 4, 8


def load_reference(path):
    ckpt = Checkpoint(path)
    model = DreamerV3(ckpt.config)
    model.load_state_dict(load_file(f"{path}/model.safetensors"))
    return model.eval()


@pytest.fixture(scope="module", params=[True, False], ids=["discrete", "continuous"])
def dreamer_path(request, tmp_path_factory):
    return str(make_tiny_dreamer(tmp_path_factory.mktemp("dreamer"), discrete=request.param))


@pytest.fixture
async def dswarm(dreamer_path):
    s = await LocalSwarm(Checkpoint(dreamer_path), model="dreamer").start()
    yield s
    await s.stop()


def observations(cfg, steps=STEPS):
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 256, (B, *cfg.obs_shape), dtype=torch.uint8, generator=g) for _ in range(steps)]


def resets(t):
    """Env 1 starts a new episode at step 5 (tests mid-batch is_first handling)."""
    first = torch.zeros(B)
    if t == 0:
        first[:] = 1
    if t == 5:
        first[1] = 1
    return first


async def rollout(policy, obs_seq, imagine_at=(), horizon=5):
    actions, dreams = [], []
    for t, obs in enumerate(obs_seq):
        actions.append(await policy.act(obs, resets(t)))
        if t in imagine_at:
            dreams.append(await policy.imagine(horizon))
    feat = policy.feat
    await policy.close()
    return torch.stack(actions), dreams, feat


def test_symlog_and_twohot():
    x = torch.tensor([-100.0, -1.0, 0.0, 0.5, 1e4])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-5)
    bins = torch.linspace(-20, 20, 255)
    logits = torch.full((255,), -1e9)
    logits[127] = 0  # all mass on bin value 0
    assert twohot_mean(logits, bins).abs() < 1e-6


def test_encoder_uint8_equals_float(dreamer_path):
    enc = load_reference(dreamer_path).encoder
    obs = torch.randint(0, 256, (2, *enc_shape(dreamer_path)), dtype=torch.uint8)
    assert torch.allclose(enc(obs), enc(obs.float() / 255), atol=1e-5)


def enc_shape(path):
    return Checkpoint(path).config.obs_shape


async def test_swarm_matches_local(dswarm, dreamer_path):
    ref = load_reference(dreamer_path)
    obs_seq = observations(ref.cfg)
    local = await rollout(Policy(ref.heads, LocalBackend(ref), B, seed=3), obs_seq, imagine_at={2, 6})

    await dswarm.add("encoder")
    await dswarm.add("encoder")  # batch is split across both
    await dswarm.add("rssm")
    agent = DistributedDreamer.from_checkpoint(dreamer_path, dswarm.registry.address, "dreamer")
    try:
        remote = await rollout(agent.policy(B, seed=3), obs_seq, imagine_at={2, 6})
    finally:
        await agent.close()

    assert torch.allclose(local[0], remote[0], atol=1e-5)  # actions
    assert torch.allclose(local[2], remote[2], atol=1e-5)  # final features
    for a, b in zip(local[1], remote[1]):
        for key in a:
            assert torch.allclose(a[key], b[key], atol=1e-4), key
    rssm = next(s for s in dswarm.servers if s.info.kind == "rssm")
    assert rssm.worker.sessions == {}  # main session and imagination forks are closed


async def test_imagination_does_not_disturb_acting(dreamer_path):
    ref = load_reference(dreamer_path)
    obs_seq = observations(ref.cfg)
    plain = await rollout(Policy(ref.heads, LocalBackend(ref), B, seed=5), obs_seq)
    dreaming = await rollout(Policy(ref.heads, LocalBackend(ref), B, seed=5), obs_seq, imagine_at={1, 3, 4})
    assert torch.equal(plain[0], dreaming[0])


@pytest.mark.parametrize("crash_method, nth", [("observe", 4), ("imagine", 3)])
async def test_rssm_failover_replays_exactly(dswarm, dreamer_path, crash_method, nth):
    ref = load_reference(dreamer_path)
    obs_seq = observations(ref.cfg)
    local = await rollout(Policy(ref.heads, LocalBackend(ref), B, seed=9), obs_seq, imagine_at={3})

    await dswarm.add("encoder")
    primary = await dswarm.add("rssm", throughput=100)
    await dswarm.add("rssm", throughput=1)
    agent = DistributedDreamer.from_checkpoint(dreamer_path, dswarm.registry.address, "dreamer")
    calls = {"n": 0}
    real_call = agent.swarm.call

    async def call(address, method, **params):
        if address == primary.info.address and method == crash_method:
            calls["n"] += 1
            if calls["n"] == nth:
                await primary.crash()
        return await real_call(address, method, **params)

    agent.swarm.call = call
    try:
        remote = await rollout(agent.policy(B, seed=9), obs_seq, imagine_at={3})
    finally:
        await agent.close()
    assert calls["n"] == nth
    assert torch.allclose(local[0], remote[0], atol=1e-5)
    assert torch.allclose(local[2], remote[2], atol=1e-5)
    for key in local[1][0]:
        assert torch.allclose(local[1][0][key], remote[1][0][key], atol=1e-4), key
