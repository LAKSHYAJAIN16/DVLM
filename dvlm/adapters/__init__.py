from .base import ModelAdapter
from .smolvlm import SmolVLMAdapter

ADAPTERS: list[type[ModelAdapter]] = [SmolVLMAdapter]


def get_adapter(config) -> ModelAdapter:
    for cls in ADAPTERS:
        if config.model_type in cls.model_types:
            return cls(config)
    supported = sorted(t for cls in ADAPTERS for t in cls.model_types)
    raise NotImplementedError(f"model_type {config.model_type!r} is not supported yet (supported: {supported})")


__all__ = ["ModelAdapter", "get_adapter", "ADAPTERS"]
