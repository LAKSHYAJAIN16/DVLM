"""Wire format: length-prefixed msgpack frames with raw tensor bytes (any torch dtype, incl. bf16)."""

from __future__ import annotations

import asyncio
import struct

import msgpack
import torch

MAX_FRAME = 1 << 31
_HEADER = struct.Struct(">I")
_TENSOR_KEY = "__tensor__"
_DTYPES = {
    str(dt).removeprefix("torch."): dt
    for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.uint8, torch.bool)
}


def _encode(obj):
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        return {
            _TENSOR_KEY: str(t.dtype).removeprefix("torch."),
            "shape": list(t.shape),
            "data": t.view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"",
        }
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def _decode(obj):
    if _TENSOR_KEY in obj:
        dtype = _DTYPES[obj[_TENSOR_KEY]]
        if not obj["data"]:
            return torch.empty(obj["shape"], dtype=dtype)
        raw = torch.frombuffer(bytearray(obj["data"]), dtype=torch.uint8)
        return raw.view(dtype).reshape(obj["shape"])
    return obj


def pack(obj) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode, raw=False, strict_map_key=False)


async def write_frame(writer: asyncio.StreamWriter, obj) -> None:
    payload = pack(obj)
    writer.write(_HEADER.pack(len(payload)) + payload)
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader):
    (size,) = _HEADER.unpack(await reader.readexactly(_HEADER.size))
    if size > MAX_FRAME:
        raise ValueError(f"frame of {size} bytes exceeds limit")
    return unpack(await reader.readexactly(size))
