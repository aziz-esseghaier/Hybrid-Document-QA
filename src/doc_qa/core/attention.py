"""SDPA attention that repeats shared key/value heads itself, for GPUs without FlashAttention (e.g. T4).

PyTorch runs grouped-query attention (`enable_gqa`, used by `transformers`' "sdpa" when there is no mask) with
FlashAttention or the math kernel only. Without FlashAttention it falls back to the math kernel, whose memory grows
with the square of the prompt length. Repeating the key/value heads first keeps the memory-efficient kernel usable.
Requires `torch` and `transformers`.
"""

from __future__ import annotations

from typing import Any

SDPA_REPEAT_KV = "sdpa_repeat_kv"


def sdpa_repeat_kv(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Scaled dot-product attention with key/value heads repeated to the number of query heads.

    Args:
        module (Any): Attention layer; its `is_causal` attribute applies when `is_causal` is None.
        query (Any): (batch, heads, query length, head dim) tensor.
        key (Any): (batch, key/value heads, key length, head dim) tensor.
        value (Any): Same shape as `key`.
        attention_mask (Any): Boolean or additive mask, or None.
        dropout (float): Dropout probability.
        scaling (float | None): Score scale; None for 1 / sqrt(head dim).
        is_causal (bool | None): Causal masking; None to follow the module.
        **kwargs (Any): Other arguments from `transformers`, unused.

    Returns:
        tuple[Any, None]: (batch, query length, heads, head dim) output, and no attention weights.
    """
    import torch

    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    is_causal = bool(query.shape[2] > 1 and attention_mask is None and is_causal)
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal
    )
    return output.transpose(1, 2).contiguous(), None


def register_sdpa_repeat_kv() -> str:
    """Make `sdpa_repeat_kv` available as an `attn_implementation`, with the masks "sdpa" uses.

    Returns:
        str: The implementation name.
    """
    import transformers

    transformers.AttentionInterface.register(SDPA_REPEAT_KV, sdpa_repeat_kv)
    try:
        from transformers.masking_utils import sdpa_mask

        transformers.AttentionMaskInterface.register(SDPA_REPEAT_KV, sdpa_mask)
    except (ImportError, AttributeError):
        pass  # older versions build "sdpa"-style masks for registered functions anyway
    return SDPA_REPEAT_KV
