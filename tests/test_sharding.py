"""M1: sharded parts must reproduce the monolithic model exactly."""

import pytest
import torch

from dvlm.adapters import get_adapter
from dvlm.tiny import tiny_inputs

from .conftest import reference_generate


@pytest.fixture(scope="module")
def parts(ckpt):
    adapter = get_adapter(ckpt.config)
    return adapter, adapter.build_encoder(ckpt, torch.float32), adapter.build_client(ckpt, torch.float32)


def run_local(adapter, head, spans, ids, image_embeds, n_new):
    caches = [s.new_cache() for s in spans]

    def through(x, pos, past):
        for s, c in zip(spans, caches):
            x = s(x, pos, c, past)
        return x

    x = through(head.embed(ids, image_embeds), adapter.position_ids(ids, 0), 0)
    past, out = ids.shape[1], []
    for _ in range(n_new):
        tok = head.logits(x[:, -1:]).argmax(-1)
        out.append(int(tok))
        x = through(head.embed(tok), adapter.position_ids(tok, past), past)
        past += 1
    return out


def test_encoder_matches_reference(parts, reference):
    _, encoder, _ = parts
    _, pixels = tiny_inputs(n_images=3)
    with torch.no_grad():
        expected = reference.model.get_image_features(pixels).pooler_output
    assert torch.allclose(encoder(pixels), expected, atol=1e-6)


@pytest.mark.parametrize("bounds", [[0, 6], [0, 1, 6], [0, 2, 4, 6], [0, 1, 2, 3, 4, 5, 6]])
def test_prefill_logits_match(parts, ckpt, reference, bounds):
    adapter, encoder, head = parts
    ids, pixels = tiny_inputs(n_images=2)
    spans = [adapter.build_span(ckpt, a, b, torch.float32) for a, b in zip(bounds, bounds[1:])]
    x = head.embed(ids, encoder(pixels))
    pos = adapter.position_ids(ids, 0)
    for s in spans:
        x = s(x, pos, s.new_cache(), 0)
    with torch.no_grad():
        expected = reference(input_ids=ids, pixel_values=pixels).logits
    assert torch.allclose(head.logits(x), expected, atol=1e-5)


def test_greedy_generation_matches(parts, ckpt, reference):
    adapter, encoder, head = parts
    ids, pixels = tiny_inputs(n_images=1)
    spans = [adapter.build_span(ckpt, a, b, torch.float32) for a, b in [(0, 2), (2, 5), (5, 6)]]
    got = run_local(adapter, head, spans, ids, encoder(pixels), 16)
    expected = reference_generate(reference, ids, pixels, 16)[0, ids.shape[1]:].tolist()
    assert got[: len(expected)] == expected


def test_chunked_prefill_equals_single_prefill(parts, ckpt):
    """Replay after failover sends history in a different chunking; caches must end up equivalent."""
    adapter, encoder, head = parts
    ids, pixels = tiny_inputs(n_images=1)
    span = adapter.build_span(ckpt, 0, 6, torch.float32)
    x = head.embed(ids, encoder(pixels))
    pos = adapter.position_ids(ids, 0)
    full = span(x, pos, span.new_cache(), 0)
    cache, k = span.new_cache(), 5
    a = span(x[:, :k], pos[:, :k], cache, 0)
    b = span(x[:, k:], pos[:, k:], cache, k)
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-5)


def test_span_subrange_and_validation(ckpt):
    adapter = get_adapter(ckpt.config)
    span = adapter.build_span(ckpt, 2, 5, torch.float32)
    assert (span.start, span.end, len(span.layers)) == (2, 5, 3)
    with pytest.raises(ValueError):
        span(torch.zeros(1, 1, 64), torch.zeros(1, 1, dtype=torch.long), span.new_cache(), 0, start=1, end=3)
    with pytest.raises(ValueError):
        adapter.build_span(ckpt, 4, 9, torch.float32)
