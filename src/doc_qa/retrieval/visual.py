"""Multi-vector embeddings of page images and queries (one unit vector per token), and MaxSim scoring.

Loading and encoding require `torch`, `transformers` and `datasets` (not base dependencies).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from doc_qa.core.parallel import batched


def load_visual_model(
    name: str, revision: str, dtype: str, device: str, attn_implementation: str
) -> tuple[Any, Any, list[str]]:
    """Load the visual retriever and its processor.

    Args:
        name (str): Hugging Face repo id.
        revision (str): Commit hash; the repo's custom code is loaded at this commit.
        dtype (str): Torch dtype name, e.g. "float16".
        device (str): Torch device.
        attn_implementation (str): Attention backend, e.g. "sdpa".

    Returns:
        tuple[Any, Any, list[str]]: Model, processor, and weights that did not load cleanly.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(name, revision=revision, trust_remote_code=True)
    model, info = AutoModel.from_pretrained(
        name,
        revision=revision,
        trust_remote_code=True,
        dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
        output_loading_info=True,
    )
    problems = [
        str(key)
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys")
        for key in info.get(field, ())
        if not str(key).startswith("lm_head")  # tied to the input embeddings, unused here
    ]
    return model.to(device).eval(), processor, sorted(problems)


def decode_image(data: bytes) -> Image.Image:
    """Decode an encoded page image to RGB.

    Args:
        data (bytes): Encoded image.

    Returns:
        Image.Image: RGB image.
    """
    with Image.open(BytesIO(data)) as image:
        return image.convert("RGB")


def trim_padding(vectors: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Keep each sequence's real tokens.

    Args:
        vectors (np.ndarray): (batch, tokens, dim).
        mask (np.ndarray): (batch, tokens), nonzero for real tokens.

    Returns:
        list[np.ndarray]: One (real tokens, dim) array per sequence.
    """
    return [row[keep.astype(bool)] for row, keep in zip(vectors, mask)]


def _encode(model: Any, inputs: Any) -> list[np.ndarray]:
    """Run one processed batch through the model.

    Args:
        model (Any): Visual retriever.
        inputs (Any): Processor output.

    Returns:
        list[np.ndarray]: Unit vectors per token (float32), one array per item.
    """
    import torch

    parameter = next(model.parameters())
    inputs = {key: value.to(parameter.device) for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device_type=parameter.device.type, dtype=parameter.dtype):
        hidden = model(**inputs).last_hidden_state
        vectors = torch.nn.functional.normalize(hidden.float(), dim=-1)  # float32: an fp16 norm can overflow
    items = trim_padding(vectors.cpu().numpy(), inputs["attention_mask"].cpu().numpy())
    if not all(np.isfinite(item).all() for item in items):
        raise FloatingPointError("Non-finite embeddings (fp16 overflow?)")
    return items


def embed_images(model: Any, processor: Any, images: list[Image.Image], batch_size: int) -> list[np.ndarray]:
    """Embed page images.

    Args:
        model (Any): Visual retriever.
        processor (Any): Its processor.
        images (list[Image.Image]): RGB pages.
        batch_size (int): Pages per forward pass.

    Returns:
        list[np.ndarray]: One (tokens, dim) array per page.
    """
    vectors: list[np.ndarray] = []
    for batch in batched(images, batch_size):
        vectors.extend(_encode(model, processor.process_documents([{"image": i, "text": ""} for i in batch])))
    return vectors


def embed_queries(model: Any, processor: Any, queries: list[str], batch_size: int) -> list[np.ndarray]:
    """Embed query texts.

    Args:
        model (Any): Visual retriever.
        processor (Any): Its processor.
        queries (list[str]): Queries.
        batch_size (int): Queries per forward pass.

    Returns:
        list[np.ndarray]: One (tokens, dim) array per query.
    """
    vectors: list[np.ndarray] = []
    for batch in batched(queries, batch_size):
        vectors.extend(_encode(model, processor.process_queries(batch)))
    return vectors


def maxsim(query: np.ndarray, page: np.ndarray) -> float:
    """Score a page: each query token takes its best page token, and the similarities are summed.

    Args:
        query (np.ndarray): (query tokens, dim), unit vectors.
        page (np.ndarray): (page tokens, dim), unit vectors.

    Returns:
        float: MaxSim score.
    """
    return float((query @ page.T).max(axis=1).sum())


def token_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compare two embeddings of the same sequence, token by token.

    Args:
        first (np.ndarray): (tokens, dim), unit vectors.
        second (np.ndarray): (tokens, dim), unit vectors.

    Returns:
        np.ndarray: Cosine similarity per token.
    """
    if first.shape != second.shape:
        raise ValueError(f"Different shapes: {first.shape} and {second.shape}")
    return (first * second).sum(axis=1)
