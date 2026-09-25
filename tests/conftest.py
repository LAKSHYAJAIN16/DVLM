import contextlib

import pytest
import torch
from transformers import SmolVLMForConditionalGeneration

from dvlm.checkpoint import Checkpoint
from dvlm.cli import build_worker
from dvlm.registry import RegistryServer
from dvlm.server import Server
from dvlm.tiny import make_tiny_smolvlm

EOS = 2


@pytest.fixture(scope="session")
def tiny_path(tmp_path_factory):
    return str(make_tiny_smolvlm(tmp_path_factory.mktemp("tiny"), num_layers=6))


@pytest.fixture(scope="session")
def ckpt(tiny_path):
    return Checkpoint(tiny_path)


@pytest.fixture(scope="session")
def reference(tiny_path):
    return SmolVLMForConditionalGeneration.from_pretrained(tiny_path, attn_implementation="sdpa").eval()


def reference_generate(model, ids, pixel_values, n):
    with torch.no_grad():
        return model.generate(
            input_ids=ids, pixel_values=pixel_values, max_new_tokens=n, do_sample=False, eos_token_id=EOS
        )


class LocalSwarm:
    """Registry + servers on localhost, all in this event loop."""

    def __init__(self, ckpt, model="tiny"):
        self.ckpt, self.model = ckpt, model
        self.registry = RegistryServer()
        self.servers: list[Server] = []

    async def start(self):
        await self.registry.start()
        return self

    async def add(self, role, start=0, end=0, throughput=1.0) -> Server:
        worker = build_worker(self.ckpt, role, start, end)
        server = Server(worker, self.model, self.registry.address, throughput=throughput)
        await server.start()
        self.servers.append(server)
        return server

    async def stop(self):
        for s in self.servers:
            await s.stop()
        await self.registry.stop()


@pytest.fixture
async def swarm(ckpt):
    s = await LocalSwarm(ckpt).start()
    yield s
    await s.stop()
