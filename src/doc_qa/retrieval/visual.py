"""Multi-vector embeddings of page images and queries (one unit vector per token): encoding, merging, storage, MaxSim.

Loading and encoding require `torch` and `transformers` (not base dependencies).
"""

from __future__ import annotations

import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage

from doc_qa.core.parallel import batched


def fetch_visual_model(name: str, revision: str) -> None:
    """Download the model and its custom code, so parallel workers find them ready.

    Args:
        name (str): Hugging Face repo id.
        revision (str): Commit hash.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor

    snapshot_download(name, revision=revision)
    AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=True)
    AutoProcessor.from_pretrained(name, revision=revision, trust_remote_code=True)


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
        if "lm_head" not in str(key).split(".")  # tied to the input embeddings, unused here
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


def page_inputs(processor: Any, images: list[Image.Image]) -> Any:
    """Prepare page images for the model. Nemotron and ColQwen3 processors name this step differently.

    Args:
        processor (Any): The model's processor.
        images (list[Image.Image]): RGB pages.

    Returns:
        Any: Model inputs.
    """
    if hasattr(processor, "process_documents"):  # nemotron
        return processor.process_documents([{"image": image, "text": ""} for image in images])
    return processor.process_images(images=images)  # ColQwen3


def query_inputs(processor: Any, queries: list[str]) -> Any:
    """Prepare query texts for the model.

    Args:
        processor (Any): The model's processor.
        queries (list[str]): Queries.

    Returns:
        Any: Model inputs.
    """
    if hasattr(processor, "process_documents"):  # nemotron
        return processor.process_queries(queries)
    return processor.process_texts(texts=queries)  # ColQwen3


def check_unit_vectors(items: Sequence[np.ndarray]) -> None:
    """Fail if any token vector is not unit length, e.g. after an fp16 overflow.

    Args:
        items (Sequence[np.ndarray]): (tokens, dim) per item.
    """
    for item in items:
        norms = np.linalg.norm(item, axis=1)
        if not np.isfinite(norms).all() or np.abs(norms - 1).max(initial=0) > 1e-3:
            raise FloatingPointError("Embeddings that are not unit vectors (fp16 overflow?)")


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
    inputs = {key: value.to(parameter.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    low_precision = parameter.dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.autocast(parameter.device.type, parameter.dtype, enabled=low_precision):
        output = model(**inputs)
        hidden = output.embeddings if "embeddings" in output else output.last_hidden_state  # ColQwen3 or nemotron
        vectors = torch.nn.functional.normalize(hidden.float(), dim=-1)  # float32: an fp16 norm can overflow
    items = trim_padding(vectors.cpu().numpy(), inputs["attention_mask"].cpu().numpy())
    check_unit_vectors(items)
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
        vectors.extend(_encode(model, page_inputs(processor, batch)))
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
        vectors.extend(_encode(model, query_inputs(processor, batch)))
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


def maxsim_matrix(queries: Sequence[np.ndarray], pages: Sequence[np.ndarray]) -> np.ndarray:
    """Score every page for every query with MaxSim.

    Args:
        queries (Sequence[np.ndarray]): (query tokens, dim) per query, unit vectors.
        pages (Sequence[np.ndarray]): (page tokens, dim) per page, unit vectors.

    Returns:
        np.ndarray: (queries, pages) scores.
    """
    query_starts = np.cumsum([0, *(len(query) for query in queries[:-1])])
    page_starts = np.cumsum([0, *(len(page) for page in pages[:-1])])
    similarities = np.concatenate(queries) @ np.concatenate(pages).T
    best_per_page = np.maximum.reduceat(similarities, page_starts, axis=1)
    return np.add.reduceat(best_per_page, query_starts, axis=0)


def pool_vectors(vectors: np.ndarray, factor: int) -> np.ndarray:
    """Merge a page's most similar vectors with Ward clustering, keeping about 1/factor of them.

    Args:
        vectors (np.ndarray): (tokens, dim) unit vectors.
        factor (int): Reduction factor.

    Returns:
        np.ndarray: (at most ceil(tokens / factor), dim) float32 unit vectors, one per cluster.
    """
    vectors = vectors.astype(np.float32)
    count, target = len(vectors), math.ceil(len(vectors) / factor)
    if target >= count:
        return vectors
    squared = np.clip(2 - 2 * (vectors @ vectors.T), 0, None)  # unit vectors: |a - b|^2 = 2 - 2 a.b
    distances = np.sqrt(squared[np.triu_indices(count, k=1)])
    labels = fcluster(linkage(distances, method="ward"), target, criterion="maxclust") - 1
    merged = np.zeros((labels.max() + 1, vectors.shape[1]), dtype=np.float32)
    np.add.at(merged, labels, vectors)
    return merged / np.linalg.norm(merged, axis=1, keepdims=True)


def spread(values: Sequence[float]) -> dict[str, float]:
    """Summarize values.

    Args:
        values (Sequence[float]): Values.

    Returns:
        dict[str, float]: Min, mean and max.
    """
    return {"min": min(values), "mean": round(float(np.mean(values)), 1), "max": max(values)}


def visual_dir(data_root: Path, dataset: str, model: str) -> Path:
    """Locate a dataset's saved visual embeddings.

    Args:
        data_root (Path): Data root.
        dataset (str): Dataset key.
        model (str): Hugging Face repo id.

    Returns:
        Path: Folder holding `pages`, `queries` and `pooled_<factor>`.
    """
    return data_root / "derived" / "visual" / dataset / model.split("/")[-1]


def pooled_dir(root: Path, factor: int) -> Path:
    """Locate a dataset's merged page vectors; with factor 1 nothing is merged.

    Args:
        root (Path): Folder from `visual_dir`.
        factor (int): Reduction factor.

    Returns:
        Path: `pages` for factor 1, else `pooled_<factor>`.
    """
    return root / "pages" if factor == 1 else root / f"pooled_{factor}"


def part_files(folder: Path) -> list[Path]:
    """List a folder's finished embedding files.

    Args:
        folder (Path): Folder; may not exist.

    Returns:
        list[Path]: Sorted .npz files.
    """
    return sorted(folder.glob("*.npz"))


def save_multivectors(path: Path, ids: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
    """Save multi-vectors as float16. The file appears only once complete.

    Args:
        path (Path): Output .npz file.
        ids (Sequence[str]): One id per item.
        vectors (Sequence[np.ndarray]): (tokens, dim) per item.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez(
            f,
            ids=np.array(ids, dtype=str),
            lengths=np.array([len(item) for item in vectors], dtype=np.int32),
            vectors=np.concatenate(vectors).astype(np.float16),
        )
    os.replace(temporary, path)


def iter_multivectors(paths: Iterable[Path]) -> Iterator[tuple[list[str], list[np.ndarray]]]:
    """Read saved multi-vectors one file at a time.

    Args:
        paths (Iterable[Path]): Files from `save_multivectors`.

    Returns:
        Iterator[tuple[list[str], list[np.ndarray]]]: Ids and float32 (tokens, dim) arrays, per file.
    """
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            stops = np.cumsum(data["lengths"])[:-1]
            yield data["ids"].tolist(), np.split(data["vectors"].astype(np.float32), stops)


def multivector_lengths(paths: Iterable[Path]) -> dict[str, int]:
    """Read how many vectors each saved item has, without loading the vectors.

    Args:
        paths (Iterable[Path]): Files from `save_multivectors`.

    Returns:
        dict[str, int]: Vector count per id.
    """
    lengths: dict[str, int] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            lengths.update(zip(data["ids"].tolist(), data["lengths"].tolist()))
    return lengths


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
