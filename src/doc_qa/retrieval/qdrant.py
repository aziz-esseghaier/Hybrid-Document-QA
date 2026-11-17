"""Qdrant collections of text chunks: a dense and a BM25 sparse vector per chunk. Requires `qdrant-client`."""

from __future__ import annotations

from typing import Any

import numpy as np

DENSE = "dense"
SPARSE = "bm25"


def text_collection(dataset: str) -> str:
    """Return the name of a dataset's text collection.

    Args:
        dataset (str): Dataset key.

    Returns:
        str: Collection name.
    """
    return f"text_{dataset}"


def create_text_collection(client: Any, name: str, dim: int) -> None:
    """Create (or replace) a collection with a cosine dense vector and an IDF-weighted sparse vector.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        dim (int): Dense vector size.
    """
    from qdrant_client import models

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def sparse_vector(indices: np.ndarray, values: np.ndarray) -> Any:
    """Build a Qdrant sparse vector.

    Args:
        indices (np.ndarray): Term ids.
        values (np.ndarray): Term weights.

    Returns:
        Any: models.SparseVector.
    """
    from qdrant_client import models

    return models.SparseVector(indices=indices.tolist(), values=values.tolist())


def upsert_chunks(
    client: Any, name: str, payloads: list[dict], dense: np.ndarray, sparse: list[Any], batch_size: int = 256
) -> None:
    """Insert chunks as points with ids 0..n-1.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        payloads (list[dict]): Payload per chunk.
        dense (np.ndarray): (n, dim) dense vectors.
        sparse (list[Any]): Sparse vector per chunk (from `sparse_vector`).
        batch_size (int): Points per request.
    """
    from qdrant_client import models

    for start in range(0, len(payloads), batch_size):
        points = [
            models.PointStruct(id=i, vector={DENSE: dense[i].tolist(), SPARSE: sparse[i]}, payload=payloads[i])
            for i in range(start, min(start + batch_size, len(payloads)))
        ]
        client.upsert(name, points=points, wait=True)


def search(client: Any, name: str, using: str, query: Any, limit: int) -> list[tuple[dict, float]]:
    """Search one vector of a collection. Dense search is exact (no HNSW); sparse search always is.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        using (str): DENSE or SPARSE.
        query (Any): Dense vector (list of floats) or sparse vector.
        limit (int): Max hits.

    Returns:
        list[tuple[dict, float]]: (payload, score), best first.
    """
    from qdrant_client import models

    params = models.SearchParams(exact=True) if using == DENSE else None
    result = client.query_points(name, query=query, using=using, limit=limit, with_payload=True, search_params=params)
    return [(point.payload, point.score) for point in result.points]
