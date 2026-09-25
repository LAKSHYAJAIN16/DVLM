"""Selective checkpoint loading: each node reads only the tensors it serves."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"


class Checkpoint:
    """A safetensors checkpoint on local disk or the Hugging Face Hub.

    For Hub repos, shards are downloaded lazily, so a node serving layers 20-30
    only fetches the shards containing those layers.
    """

    def __init__(self, path_or_repo: str):
        self.source = str(path_or_repo)
        self._local = Path(path_or_repo) if Path(path_or_repo).is_dir() else None
        self.config = AutoConfig.from_pretrained(self.source)
        self.weight_map = self._read_weight_map()

    def _file(self, name: str) -> Path:
        if self._local is not None:
            return self._local / name
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.source, name))

    def _read_weight_map(self) -> dict[str, str]:
        try:
            index = self._file(INDEX_FILE)
            if index.exists():
                return json.loads(index.read_text())["weight_map"]
        except Exception:  # hub repo without an index file
            pass
        with safe_open(self._file(SINGLE_FILE), "pt") as f:
            return {key: SINGLE_FILE for key in f.keys()}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def load(self, prefix: str, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        """Load all tensors under `prefix`, returned with the prefix stripped."""
        by_file: dict[str, list[str]] = defaultdict(list)
        for key, file in self.weight_map.items():
            if key.startswith(prefix):
                by_file[file].append(key)
        if not by_file:
            raise KeyError(f"no tensors with prefix {prefix!r} in {self.source}")
        out = {}
        for file, keys in by_file.items():
            with safe_open(self._file(file), "pt") as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    out[key[len(prefix):]] = tensor
        return out

    def load_into(self, module: torch.nn.Module, prefix: str, dtype: torch.dtype | None = None) -> None:
        module.load_state_dict(self.load(prefix, dtype), strict=True)
