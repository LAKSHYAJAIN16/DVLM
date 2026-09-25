"""DreamerV3 (Hafner et al., 2023) in PyTorch: the pieces needed to act and imagine.

Split for the swarm:
- `Encoder` (CNN over image observations): stateless -> encoder servers.
- `RSSM` (recurrent state-space model): per-episode state -> rssm servers.
- `Heads` (actor, critic, reward, continue): small -> client.

All stochastic sampling takes an explicit seed, so a replayed episode reproduces the
same latent trajectory bit for bit (this is what makes failover exact).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DreamerConfig:
    model_type: str = "dreamerv3"
    obs_shape: tuple[int, int, int] = (3, 64, 64)
    num_actions: int = 6
    discrete_actions: bool = True
    deter: int = 512
    stoch: int = 32
    classes: int = 32
    hidden: int = 512
    cnn_depth: int = 32
    mlp_layers: int = 2
    unimix: float = 0.01
    bins: int = 255
    extra: dict = field(default_factory=dict)

    @property
    def feat_size(self) -> int:
        return self.deter + self.stoch * self.classes

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DreamerConfig":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "obs_shape" in d:
            d["obs_shape"] = tuple(d["obs_shape"])
        return cls(**d)


# ---------------------------------------------------------------- utilities


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value of a two-hot distribution over symlog-spaced bins, in real space."""
    return symexp((torch.softmax(logits, -1) * bins).sum(-1))


def unimix_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    probs = torch.softmax(logits, -1)
    return (1 - unimix) * probs + unimix / logits.shape[-1]


def sample_onehot(probs: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, 1, generator=gen)
    return F.one_hot(idx.squeeze(-1), probs.shape[-1]).to(probs.dtype).reshape(probs.shape)


def seeded(seed: int | None) -> torch.Generator | None:
    return None if seed is None else torch.Generator().manual_seed(int(seed))


class NormLinear(nn.Sequential):
    def __init__(self, i: int, o: int):
        super().__init__(nn.Linear(i, o, bias=False), nn.LayerNorm(o), nn.SiLU())


def mlp(i: int, hidden: int, layers: int) -> nn.Sequential:
    return nn.Sequential(*[NormLinear(i if k == 0 else hidden, hidden) for k in range(layers)])


class ChannelNorm(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------- modules


class Encoder(nn.Module):
    """[B, C, H, W] images (uint8 in [0, 255] or float in [0, 1]) -> [B, embed]. Halves resolution down to 4x4."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        ch, size = cfg.obs_shape[0], cfg.obs_shape[1]
        if cfg.obs_shape[1] != cfg.obs_shape[2] or size < 4 or size & (size - 1):
            raise ValueError("obs_shape must be square with a power-of-two side >= 4")
        layers, depth = [], cfg.cnn_depth
        while size > 4:
            layers += [nn.Conv2d(ch, depth, 4, 2, 1, bias=False), ChannelNorm(depth), nn.SiLU()]
            ch, depth, size = depth, depth * 2, size // 2
        self.net = nn.Sequential(*layers)
        self.embed_size = ch * size * size

    @torch.inference_mode()
    def forward(self, obs: torch.Tensor, _mask=None) -> torch.Tensor:
        scale = 255.0 if obs.dtype == torch.uint8 else 1.0
        obs = obs.to(next(self.parameters()).dtype) / scale
        return self.net(obs - 0.5).flatten(1)


class LayerNormGRU(nn.Module):
    """DreamerV3's GRU: one linear over [x, h], LayerNorm, update gate biased towards keeping h."""

    def __init__(self, i: int, size: int):
        super().__init__()
        self.linear = nn.Linear(i + size, 3 * size, bias=False)
        self.norm = nn.LayerNorm(3 * size)

    def forward(self, x, h):
        reset, cand, update = self.norm(self.linear(torch.cat([x, h], -1))).chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * h


class RSSM(nn.Module):
    """State = (deter h [B, deter], stoch z [B, stoch, classes] one-hot)."""

    def __init__(self, cfg: DreamerConfig, embed_size: int):
        super().__init__()
        self.cfg = cfg
        z = cfg.stoch * cfg.classes
        self.h0 = nn.Parameter(torch.zeros(cfg.deter))
        self.img_in = NormLinear(z + cfg.num_actions, cfg.hidden)
        self.gru = LayerNormGRU(cfg.hidden, cfg.deter)
        self.prior = nn.Sequential(NormLinear(cfg.deter, cfg.hidden), nn.Linear(cfg.hidden, z))
        self.post = nn.Sequential(NormLinear(cfg.deter + embed_size, cfg.hidden), nn.Linear(cfg.hidden, z))

    def _logits(self, head, x):
        return head(x).reshape(*x.shape[:-1], self.cfg.stoch, self.cfg.classes)

    def initial(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.h0).expand(batch, -1)
        mode = self._logits(self.prior, h).argmax(-1)
        return h, F.one_hot(mode, self.cfg.classes).to(h.dtype)

    @staticmethod
    def features(h, z) -> torch.Tensor:
        return torch.cat([h, z.flatten(1)], -1)

    @torch.inference_mode()
    def img_step(self, h, z, action, seed: int | None = None):
        """Prior step: imagine the next state from the current one and an action."""
        x = self.img_in(torch.cat([z.flatten(1), action.to(z.dtype)], -1))
        h = self.gru(x, h)
        probs = unimix_probs(self._logits(self.prior, h), self.cfg.unimix)
        return h, sample_onehot(probs, seeded(seed))

    @torch.inference_mode()
    def obs_step(self, h, z, action, embed, is_first, seed: int | None = None):
        """Posterior step: advance with the previous action, then correct with the observation embedding."""
        is_first = is_first.bool().reshape(-1, 1)
        if is_first.any():
            h0, z0 = self.initial(h.shape[0])
            h = torch.where(is_first, h0, h)
            z = torch.where(is_first[..., None], z0, z)
            action = torch.where(is_first, torch.zeros_like(action), action)
        x = self.img_in(torch.cat([z.flatten(1), action.to(z.dtype)], -1))
        h = self.gru(x, h)
        probs = unimix_probs(self._logits(self.post, torch.cat([h, embed.to(h.dtype)], -1)), self.cfg.unimix)
        return h, sample_onehot(probs, seeded(seed))


class Heads(nn.Module):
    """Client-side heads over RSSM features."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        f, hid, n = cfg.feat_size, cfg.hidden, cfg.mlp_layers
        out = cfg.num_actions if cfg.discrete_actions else 2 * cfg.num_actions
        self.actor = nn.Sequential(mlp(f, hid, n), nn.Linear(hid, out))
        self.critic = nn.Sequential(mlp(f, hid, n), nn.Linear(hid, cfg.bins))
        self.reward = nn.Sequential(mlp(f, hid, n), nn.Linear(hid, cfg.bins))
        self.cont = nn.Sequential(mlp(f, hid, n), nn.Linear(hid, 1))
        self.register_buffer("bins", torch.linspace(-20, 20, cfg.bins), persistent=False)

    @torch.inference_mode()
    def act(self, feat: torch.Tensor, gen: torch.Generator | None = None, greedy: bool = False) -> torch.Tensor:
        out = self.actor(feat)
        if self.cfg.discrete_actions:
            probs = unimix_probs(out, self.cfg.unimix)
            if greedy:
                return F.one_hot(probs.argmax(-1), self.cfg.num_actions).to(feat.dtype)
            return sample_onehot(probs, gen)
        mean, std = out.chunk(2, -1)
        mean, std = torch.tanh(mean), 0.9 * torch.sigmoid(std + 2.0) + 0.1
        if greedy:
            return mean
        noise = torch.randn(mean.shape, generator=gen, dtype=mean.dtype)
        return (mean + std * noise).clamp(-1, 1)

    @torch.inference_mode()
    def value(self, feat):
        return twohot_mean(self.critic(feat), self.bins)

    @torch.inference_mode()
    def predict_reward(self, feat):
        return twohot_mean(self.reward(feat), self.bins)

    @torch.inference_mode()
    def predict_continue(self, feat):
        return torch.sigmoid(self.cont(feat)).squeeze(-1)


class DreamerV3(nn.Module):
    """Monolithic model: the single-process reference the swarm must reproduce."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.rssm = RSSM(cfg, self.encoder.embed_size)
        self.heads = Heads(cfg)

    def save(self, out_dir: str | Path) -> Path:
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(self.cfg.to_json())
        save_file({k: v.contiguous() for k, v in self.state_dict().items()}, out_dir / "model.safetensors")
        return out_dir


def step_seed(base: int, step: int, stream: int = 0) -> int:
    """Deterministic per-step seed shared by the client and the reference implementation."""
    return (base * 1_000_003 + step * 7_919 + stream) % (2**62) or 1

