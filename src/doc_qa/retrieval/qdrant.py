"""Qdrant collections: text chunks (dense + BM25 vectors) and pages (multi-vectors). Requires `qdrant-client`."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

DENSE = "dense"
SPARSE = "bm25"
VISUAL = "page"
MAX_MULTIVECTOR_VALUES = 1_048_576  # Qdrant's limit on vectors x dimensions in one point; not configurable


def visual_collection(name: str, dataset: str) -> str:
    """Return the name of a dataset's page collection.

    Args:
        name (str): Visual config name, one per retriever (e.g. "visual", "visual_tomoro").
        dataset (str): Dataset key.

    Returns:
        str: Collection name, `<name>_<dataset>`.
    """
    return f"{name}_{dataset}"


def fits_in_point(tokens: int, dim: int) -> bool:
    """Check that a multi-vector is within Qdrant's size limit.

    Args:
        tokens (int): Vectors.
        dim (int): Dimensions per vector.

    Returns:
        bool: True if a point can hold it.
    """
    return tokens * dim < MAX_MULTIVECTOR_VALUES


def create_visual_collection(client: Any, name: str, dim: int) -> None:
    """Create (or replace) a MaxSim multi-vector collection, stored as float16 on disk, without HNSW.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        dim (int): Dimensions per vector.
    """
    from qdrant_client import models

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            VISUAL: models.VectorParams(
                size=dim,
                distance=models.Distance.DOT,  # the vectors are already unit length
                datatype=models.Datatype.FLOAT16,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                hnsw_config=models.HnswConfigDiff(m=0),  # search is exact
            )
        },
    )


def upsert_page(client: Any, name: str, point_id: int, page_id: str, vectors: np.ndarray) -> None:
    """Insert one page. A page is close to the request size limit, so it goes alone.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        point_id (int): Point id.
        page_id (str): Page id, stored as payload.
        vectors (np.ndarray): (tokens, dim) unit vectors.
    """
    from qdrant_client import models

    point = models.PointStruct(id=point_id, vector={VISUAL: vectors.tolist()}, payload={"page_id": page_id})
    client.upsert(name, points=[point], wait=True)


def wait_until_ready(client: Any, name: str, timeout: float = 3600.0) -> None:
    """Wait for Qdrant to finish reorganizing a collection.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        timeout (float): Seconds before giving up.
    """
    from qdrant_client import models

    deadline = time.monotonic() + timeout
    while client.get_collection(name).status != models.CollectionStatus.GREEN:
        if time.monotonic() > deadline:
            raise TimeoutError(f"{name} is still being optimized after {timeout:.0f} s.")
        time.sleep(2)


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
    """Search one vector of a collection. Dense and multi-vector search skip HNSW; sparse search is always exact.

    Args:
        client (Any): QdrantClient.
        name (str): Collection name.
        using (str): DENSE, SPARSE or VISUAL.
        query (Any): Dense vector, sparse vector, or list of vectors for VISUAL.
        limit (int): Max hits.

    Returns:
        list[tuple[dict, float]]: (payload, score), best first.
    """
    from qdrant_client import models

    params = models.SearchParams(exact=True) if using != SPARSE else None
    result = client.query_points(name, query=query, using=using, limit=limit, with_payload=True, search_params=params)
    return [(point.payload, point.score) for point in result.points]


def file_sha256(path: Path) -> str:
    """Hash a file.

    Args:
        path (Path): File.

    Returns:
        str: SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_snapshot(client: Any, url: str, name: str, path: Path, timeout: float = 3600.0) -> str:
    """Snapshot a collection into a local file, then delete the snapshot from the server.

    Args:
        client (Any): QdrantClient.
        url (str): Qdrant base URL.
        name (str): Collection name.
        path (Path): Output file.
        timeout (float): Download timeout in seconds.

    Returns:
        str: SHA-256 of the file.
    """
    import httpx

    snapshot = client.create_snapshot(name, wait=True)
    try:
        address = f"{url}/collections/{quote(name)}/snapshots/{quote(snapshot.name)}"
        with httpx.stream("GET", address, timeout=timeout) as response:
            response.raise_for_status()
            with path.open("wb") as f:
                for block in response.iter_bytes():
                    f.write(block)
    finally:
        client.delete_snapshot(name, snapshot.name, wait=True)
    sha256 = file_sha256(path)
    if snapshot.checksum and snapshot.checksum != sha256:
        raise RuntimeError(f"Snapshot of {name} doesn't match the server's checksum.")
    return sha256


def restore_snapshot(url: str, name: str, path: Path, sha256: str, timeout: float = 3600.0) -> None:
    """Create a collection from a snapshot file. Qdrant verifies the checksum.

    Args:
        url (str): Qdrant base URL.
        name (str): Collection name.
        path (Path): Snapshot file.
        sha256 (str): Expected SHA-256 of the file.
        timeout (float): Request timeout in seconds.
    """
    import httpx

    with path.open("rb") as f:
        response = httpx.post(
            f"{url}/collections/{quote(name)}/snapshots/upload",
            params={"wait": "true", "priority": "snapshot", "checksum": sha256},
            files={"snapshot": (path.name, f, "application/octet-stream")},
            timeout=timeout,
        )
    if response.is_error:
        raise RuntimeError(f"Restoring {name} failed ({response.status_code}): {response.text[:500]}")
