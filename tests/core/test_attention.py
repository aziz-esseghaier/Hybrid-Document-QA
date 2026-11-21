"""Tests for SDPA with repeated key/value heads (needs torch; skipped without it)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from doc_qa.core.attention import sdpa_repeat_kv  # noqa: E402


def reference(query, key, value, mask=None, causal=False):
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    scores = query @ key.transpose(-1, -2) / query.shape[-1] ** 0.5
    if causal:
        scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool).triu(1), float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return (scores.softmax(-1) @ value).transpose(1, 2)


@pytest.mark.parametrize("causal", [True, False])
def test_matches_grouped_query_attention(causal):
    torch.manual_seed(0)
    query, key, value = torch.randn(1, 8, 5, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    output, weights = sdpa_repeat_kv(SimpleNamespace(is_causal=causal), query, key, value, None)
    assert weights is None and output.shape == (1, 5, 8, 4)
    assert torch.allclose(output, reference(query, key, value, causal=causal), atol=1e-5)


def test_uses_the_mask_and_cuts_it_to_the_key_length():
    torch.manual_seed(0)
    query, key, value = torch.randn(1, 4, 3, 4), torch.randn(1, 4, 3, 4), torch.randn(1, 4, 3, 4)
    mask = torch.ones(1, 1, 3, 5, dtype=torch.bool).tril()
    output, _ = sdpa_repeat_kv(SimpleNamespace(is_causal=True), query, key, value, mask)
    assert torch.allclose(output, reference(query, key, value, mask=mask[..., :3]), atol=1e-5)


def test_one_query_token_is_never_causal():
    query, key, value = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4)
    output, _ = sdpa_repeat_kv(SimpleNamespace(is_causal=True), query, key, value, None)
    assert torch.allclose(output, reference(query, key, value), atol=1e-5)
