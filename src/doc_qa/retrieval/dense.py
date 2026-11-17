"""Dense text embeddings with sentence-transformers, and their storage as .npz files.

Loading and encoding require `torch` and `sentence-transformers` (not base dependencies).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_embedder(model: str, revision: str, max_tokens: int) -> Any:
    """Load a sentence-transformers model: fp16 + SDPA on GPU, fp32 on CPU.

    Args:
        model (str): Hugging Face repo id.
        revision (str): Commit hash.
        max_tokens (int): Input limit; longer inputs are truncated.

    Returns:
        Any: SentenceTransformer.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    cuda = torch.cuda.is_available()
    embedder = SentenceTransformer(
        model,
        revision=revision,
        device="cuda" if cuda else "cpu",
        model_kwargs={"dtype": torch.float16 if cuda else torch.float32, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    embedder.max_seq_length = max_tokens
    return embedder


def embed_texts(embedder: Any, texts: list[str], batch_size: int, prompt_name: str | None = None) -> np.ndarray:
    """Embed texts as unit vectors.

    Args:
        embedder (Any): SentenceTransformer.
        texts (list[str]): Texts.
        batch_size (int): Texts per forward pass.
        prompt_name (str | None): Name of a prompt in the model config (e.g. "query"); None = no prompt.

    Returns:
        np.ndarray: (len(texts), dim) float32.
    """
    vectors = embedder.encode(
        texts, batch_size=batch_size, prompt_name=prompt_name, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise FloatingPointError("Non-finite embeddings (fp16 overflow?)")
    return vectors


def save_vectors(path: Path, ids: list[str], vectors: np.ndarray) -> None:
    """Save ids and vectors (as float16) to an .npz file.

    Args:
        path (Path): Output file.
        ids (list[str]): One id per row.
        vectors (np.ndarray): (len(ids), dim).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids, dtype=str), vectors=vectors.astype(np.float16))


def load_vectors(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Load vectors saved by `save_vectors`.

    Args:
        paths (Iterable[Path]): .npz files.

    Returns:
        dict[str, np.ndarray]: float32 vector per id.
    """
    vectors = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            vectors.update(zip(data["ids"].tolist(), data["vectors"].astype(np.float32)))
    return vectors
