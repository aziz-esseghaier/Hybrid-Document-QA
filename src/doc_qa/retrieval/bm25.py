"""BM25 sparse vectors with fastembed: stopwords and stemming per language, IDF applied by Qdrant.

Requires `fastembed`.
"""

from __future__ import annotations

from typing import Any


def load_bm25(language: str, k: float = 1.2, b: float = 0.75, avg_len: float = 256.0) -> Any:
    """Load fastembed's BM25 model.

    Args:
        language (str): Stemmer and stopword language, e.g. "english", "french".
        k (float): Term frequency saturation.
        b (float): Length normalization (0 = none, 1 = full).
        avg_len (float): Mean document length in BM25 tokens.

    Returns:
        Any: fastembed SparseTextEmbedding.
    """
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding("Qdrant/bm25", language=language, k=k, b=b, avg_len=avg_len)


def mean_length(texts: list[str], language: str) -> float:
    """Mean document length in BM25 tokens (after stopwords and stemming).

    Args:
        texts (list[str]): Documents.
        language (str): Stemmer and stopword language.

    Returns:
        float: Mean length.
    """
    # With b=0 and a huge k, each term weight equals its count, so a vector's sum is the document length.
    counter = load_bm25(language, k=1e9, b=0.0)
    lengths = [float(e.values.sum()) for e in counter.embed(texts)]
    return sum(lengths) / len(lengths)
