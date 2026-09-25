"""SmolVLM / Idefics3: SigLIP-style vision tower + pixel-shuffle connector + Llama decoder."""

from __future__ import annotations

import torch
from torch import nn

from ..checkpoint import Checkpoint
from .base import ModelAdapter


class Idefics3StyleEncoder(nn.Module):
    """Mirrors `SmolVLMModel.get_image_features`, standalone so it can live on its own node."""

    def __init__(self, vision_model: nn.Module, connector: nn.Module, patch_size: int):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector
        self.patch_size = patch_size

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        batch, n_images = pixel_values.shape[:2]
        pixel_values = pixel_values.to(dtype).view(batch * n_images, *pixel_values.shape[2:])

        # Drop all-zero padding images.
        per_image = pixel_values.shape[1:].numel()
        real = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != per_image
        real[0] |= ~torch.any(real)
        pixel_values = pixel_values[real].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                [pixel_values.shape[i] for i in (0, 2, 3)], dtype=torch.bool, device=pixel_values.device
            )
        else:
            pixel_attention_mask = pixel_attention_mask.view(batch * n_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real].contiguous()

        p = self.patch_size
        subgrid = pixel_attention_mask.unfold(1, p, p).unfold(2, p, p)
        patch_attention_mask = (subgrid.sum(dim=(-1, -2)) > 0).bool()

        hidden = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask).last_hidden_state
        return self.connector(hidden)


class SmolVLMAdapter(ModelAdapter):
    model_types = ("smolvlm", "idefics3")
    text_prefix = "model.text_model."

    def __init__(self, config):
        super().__init__(config)
        self.config.vision_config._attn_implementation = "sdpa"

    @property
    def image_token_id(self) -> int:
        return self.config.image_token_id

    def _classes(self):
        if self.config.model_type == "smolvlm":
            from transformers.models.smolvlm import modeling_smolvlm as m

            return m.SmolVLMVisionTransformer, m.SmolVLMConnector
        from transformers.models.idefics3 import modeling_idefics3 as m

        return m.Idefics3VisionTransformer, m.Idefics3Connector

    def build_encoder(self, ckpt: Checkpoint, dtype: torch.dtype) -> nn.Module:
        vision_cls, connector_cls = self._classes()
        vision = vision_cls._from_config(self.config.vision_config)
        ckpt.load_into(vision, "model.vision_model.", dtype)
        connector = connector_cls(self.config)
        ckpt.load_into(connector, "model.connector.", dtype)
        encoder = Idefics3StyleEncoder(vision, connector, self.config.vision_config.patch_size)
        return encoder.to(dtype).eval()
