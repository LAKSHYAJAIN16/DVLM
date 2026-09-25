"""How a DreamerV3 checkpoint is cut into swarm parts."""

from __future__ import annotations

import torch

from ..checkpoint import Checkpoint
from .model import RSSM, DreamerConfig, Encoder, Heads


class DreamerAdapter:
    model_types = ("dreamerv3",)

    def __init__(self, config: DreamerConfig):
        self.config = config
        with torch.device("meta"):
            self.embed_size = Encoder(config).embed_size

    def build_encoder(self, ckpt: Checkpoint, dtype: torch.dtype) -> Encoder:
        enc = Encoder(self.config)
        ckpt.load_into(enc, "encoder.", dtype)
        return enc.to(dtype).eval()

    def build_rssm(self, ckpt: Checkpoint, dtype: torch.dtype) -> RSSM:
        rssm = RSSM(self.config, self.embed_size)
        ckpt.load_into(rssm, "rssm.", dtype)
        return rssm.to(dtype).eval()

    def build_client(self, ckpt: Checkpoint, dtype: torch.dtype) -> Heads:
        heads = Heads(self.config)
        ckpt.load_into(heads, "heads.", dtype)
        return heads.to(dtype).eval()
