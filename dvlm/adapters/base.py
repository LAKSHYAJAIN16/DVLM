"""Per-architecture adapters: how to cut a VLM into encoder / span / client parts."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers import AutoModel

from ..checkpoint import Checkpoint
from ..parts import ClientHead, DecoderSpan


def text_model_classes(text_config) -> tuple[type, type, type]:
    """Discover (decoder layer, final norm, rotary embedding) classes without allocating weights."""
    with torch.device("meta"):
        text_model = AutoModel.from_config(text_config)
    return type(text_model.layers[0]), type(text_model.norm), type(text_model.rotary_emb)


class ModelAdapter(ABC):
    """Knows the checkpoint layout and the vision path of one VLM family."""

    model_types: tuple[str, ...] = ()
    #: prefix of the text decoder inside the checkpoint, e.g. "model.text_model."
    text_prefix: str
    lm_head_key = "lm_head.weight"

    def __init__(self, config):
        self.config = config
        self.text_config = config.get_text_config(decoder=True)
        # Parts are built directly from config, so pick an attention kernel explicitly.
        self.text_config._attn_implementation = "sdpa"

    @property
    def num_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    @abstractmethod
    def image_token_id(self) -> int: ...

    @abstractmethod
    def build_encoder(self, ckpt: Checkpoint, dtype: torch.dtype) -> nn.Module:
        """Vision tower + projector. forward(pixel_values, pixel_attention_mask) -> [n_images, n_tokens, hidden]."""

    def position_ids(self, input_ids: torch.Tensor, past_len: int) -> torch.Tensor:
        """Position ids for new tokens. Override for multimodal RoPE (e.g. Qwen2-VL M-RoPE)."""
        n = input_ids.shape[1]
        return torch.arange(past_len, past_len + n).unsqueeze(0).expand(input_ids.shape[0], n)

    def build_span(self, ckpt: Checkpoint, start: int, end: int, dtype: torch.dtype) -> DecoderSpan:
        layer_cls, _, rotary_cls = text_model_classes(self.text_config)
        span = DecoderSpan(self.text_config, start, end, layer_cls, rotary_cls)
        for local, layer in enumerate(span.layers):
            ckpt.load_into(layer, f"{self.text_prefix}layers.{start + local}.", dtype)
        return span.to(dtype).eval()

    def build_client(self, ckpt: Checkpoint, dtype: torch.dtype) -> ClientHead:
        _, norm_cls, _ = text_model_classes(self.text_config)
        cfg = self.text_config
        embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        ckpt.load_into(embed, f"{self.text_prefix}embed_tokens.", dtype)
        norm = norm_cls(cfg.hidden_size, eps=cfg.rms_norm_eps)
        ckpt.load_into(norm, f"{self.text_prefix}norm.", dtype)
        lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if ckpt.has(self.lm_head_key):
            lm_head.weight.data = ckpt.load(self.lm_head_key, dtype)[""]
        else:  # tied embeddings
            lm_head.weight = embed.weight
        return ClientHead(embed, norm, lm_head, self.image_token_id).to(dtype).eval()
