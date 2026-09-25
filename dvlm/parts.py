"""Model-agnostic building blocks that nodes execute.

- `DecoderSpan`: a contiguous range of decoder layers `[start, end)` plus rotary
  embeddings. Runs on span servers; KV caches live beside it, one per session.
- `ClientHead`: token embeddings, final norm and LM head. Runs on the client so
  prompt text and sampled tokens never leave it.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import DynamicCache


def causal_mask(q_len: int, past_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    """Additive mask of shape [1, 1, q_len, past_len + q_len]; None when q_len == 1."""
    if q_len == 1:
        return None
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + past_len
    k_pos = torch.arange(past_len + q_len, device=device).unsqueeze(0)
    mask = torch.zeros(q_len, past_len + q_len, dtype=dtype, device=device)
    mask.masked_fill_(k_pos > q_pos, torch.finfo(dtype).min)
    return mask[None, None]


def span_config(text_config, start: int, end: int):
    """A copy of `text_config` describing only layers [start, end), indexed locally from 0."""
    cfg = copy.deepcopy(text_config)
    cfg.num_hidden_layers = end - start
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(cfg.layer_types[start:end])
    return cfg


class DecoderSpan(nn.Module):
    def __init__(self, text_config, start: int, end: int, layer_cls: type, rotary_cls: type):
        super().__init__()
        if not 0 <= start < end <= text_config.num_hidden_layers:
            raise ValueError(f"invalid span [{start}, {end}) for {text_config.num_hidden_layers} layers")
        if getattr(text_config, "sliding_window", None) and getattr(text_config, "use_sliding_window", True):
            # TODO: sliding-window masks (Mistral, some Qwen2 configs).
            raise NotImplementedError("sliding-window attention is not supported yet")
        self.start, self.end = start, end
        self.config = span_config(text_config, start, end)
        self.layers = nn.ModuleList(layer_cls(self.config, i) for i in range(end - start))
        self.rotary_emb = rotary_cls(config=self.config)

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.config)

    @torch.inference_mode()
    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DynamicCache,
        past_len: int,
        start: int | None = None,
        end: int | None = None,
    ) -> torch.Tensor:
        """Run layers [start, end) (defaulting to the whole span) over `hidden`.

        `past_len` is how many positions this cache has already seen for these layers.
        """
        start = self.start if start is None else start
        end = self.end if end is None else end
        if not self.start <= start < end <= self.end:
            raise ValueError(f"[{start}, {end}) is outside span [{self.start}, {self.end})")
        dtype = next(self.layers.parameters()).dtype
        hidden = hidden.to(dtype)
        mask = causal_mask(hidden.shape[1], past_len, dtype, hidden.device)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        for layer in self.layers[start - self.start : end - self.start]:
            out = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


class ClientHead(nn.Module):
    def __init__(self, embed_tokens: nn.Embedding, norm: nn.Module, lm_head: nn.Linear, image_token_id: int):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head
        self.image_token_id = image_token_id

    @torch.inference_mode()
    def embed(self, input_ids: torch.Tensor, image_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Embed tokens and splice visual embeddings over the image placeholder tokens, in order."""
        embeds = self.embed_tokens(input_ids)
        if image_embeds is None:
            return embeds
        image_mask = input_ids == self.image_token_id
        n_slots = int(image_mask.sum())
        flat = image_embeds.reshape(-1, image_embeds.shape[-1])
        if flat.shape[0] != n_slots:
            raise ValueError(f"prompt has {n_slots} image tokens but got {flat.shape[0]} image embeddings")
        embeds = embeds.clone()
        embeds[image_mask] = flat.to(embeds.dtype)
        return embeds

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(self.lm_head.weight.dtype)
        return self.lm_head(self.norm(hidden)).float()
