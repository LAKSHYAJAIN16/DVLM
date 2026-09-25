import pytest
import torch

from dvlm.wire import pack, unpack


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_tensor_roundtrip(dtype):
    t = (torch.randn(2, 3, 5) * 10).to(dtype)
    out = unpack(pack({"x": t, "meta": [1, "a", None]}))
    assert out["meta"] == [1, "a", None]
    assert out["x"].dtype == dtype and torch.equal(out["x"], t)


def test_empty_and_noncontiguous():
    empty = torch.empty(0, 4)
    assert unpack(pack(empty)).shape == (0, 4)
    t = torch.randn(4, 6).t()
    assert torch.equal(unpack(pack(t)), t)
