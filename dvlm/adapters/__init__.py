from .base import ModelAdapter
from ..dreamer.adapter import DreamerAdapter
from .smolvlm import SmolVLMAdapter

ADAPTERS: list[type] = [SmolVLMAdapter, DreamerAdapter]


def get_adapter(config):
    for cls in ADAPTERS:
        if config.model_type in cls.model_types:
            return cls(config)
    supported = sorted(t for cls in ADAPTERS for t in cls.model_types)
    raise NotImplementedError(f"model_type {config.model_type!r} is not supported yet (supported: {supported})")


__all__ = ["ModelAdapter", "get_adapter", "ADAPTERS"]
