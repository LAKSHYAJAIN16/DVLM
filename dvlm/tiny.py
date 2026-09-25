"""Tiny randomly initialised VLM checkpoints for tests, demos and CI (no downloads needed)."""

from __future__ import annotations

from pathlib import Path

import torch

IMAGE_TOKEN_ID = 100


def make_tiny_smolvlm(out_dir: str | Path, num_layers: int = 6, seed: int = 0) -> Path:
    from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

    config = SmolVLMConfig(
        text_config=dict(
            model_type="llama",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            tie_word_embeddings=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=64, patch_size=16
        ),
        scale_factor=2,
        image_token_id=IMAGE_TOKEN_ID,
        pad_token_id=0,
    )
    torch.manual_seed(seed)
    model = SmolVLMForConditionalGeneration(config).eval()
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)
    return out_dir


def tiny_inputs(n_images: int = 1, image_tokens: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A prompt with `n_images` image placeholders and matching random pixel values."""
    g = torch.Generator().manual_seed(seed)
    ids = [1, 7, 8]
    for _ in range(n_images):
        ids += [9] + [IMAGE_TOKEN_ID] * image_tokens + [10]
    ids += [11, 12, 13]
    pixel_values = torch.randn(1, n_images, 3, 64, 64, generator=g)
    return torch.tensor([ids]), pixel_values
