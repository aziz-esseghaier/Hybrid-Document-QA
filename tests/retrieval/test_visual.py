"""Tests for the visual retriever's helpers."""

import io

import numpy as np
import pytest
from PIL import Image

from doc_qa.retrieval.visual import (
    decode_image,
    iter_multivectors,
    maxsim,
    maxsim_matrix,
    multivector_lengths,
    part_files,
    pool_vectors,
    save_multivectors,
    token_cosines,
    trim_padding,
)


def unit_vectors(count, dim, seed=0):
    rows = np.random.default_rng(seed).normal(size=(count, dim)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_maxsim_matrix_matches_maxsim():
    queries = [unit_vectors(3, 8, 1), unit_vectors(5, 8, 2)]
    pages = [unit_vectors(7, 8, 3), unit_vectors(1, 8, 4), unit_vectors(4, 8, 5)]
    scores = maxsim_matrix(queries, pages)
    assert scores.shape == (2, 3)
    for i, query in enumerate(queries):
        for j, page in enumerate(pages):
            assert scores[i, j] == pytest.approx(maxsim(query, page), abs=1e-5)


def test_pool_vectors_halves_the_count_and_keeps_unit_length():
    pooled = pool_vectors(unit_vectors(101, 32), 2)
    assert len(pooled) <= 51
    assert np.linalg.norm(pooled, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_pool_vectors_merges_the_closest_vectors():
    centers = unit_vectors(3, 16, 1)
    noise = unit_vectors(6, 16, 2) * 0.01
    vectors = np.concatenate([centers, centers]) + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pooled = pool_vectors(vectors, 2)
    assert len(pooled) == 3
    assert (pooled @ centers.T).max(axis=0) == pytest.approx(1.0, abs=1e-3)


def test_pool_vectors_factor_one_changes_nothing():
    vectors = unit_vectors(5, 8)
    assert np.array_equal(pool_vectors(vectors, 1), vectors)


def test_multivectors_round_trip(tmp_path):
    first, second = unit_vectors(2, 4, 1), unit_vectors(3, 4, 2)
    path = tmp_path / "pages" / "w0_00000.npz"
    save_multivectors(path, ["a", "b"], [first, second])
    assert part_files(path.parent) == [path]
    assert not list(path.parent.glob("*.tmp"))
    assert multivector_lengths([path]) == {"a": 2, "b": 3}
    ids, vectors = next(iter_multivectors([path]))
    assert ids == ["a", "b"]
    assert vectors[0].dtype == np.float32
    assert vectors[1] == pytest.approx(second, abs=1e-3)


def test_part_files_of_a_missing_folder(tmp_path):
    assert part_files(tmp_path / "missing") == []


def test_decode_image_gives_rgb():
    buffer = io.BytesIO()
    Image.new("L", (4, 3), 200).save(buffer, "PNG")
    image = decode_image(buffer.getvalue())
    assert (image.mode, image.size) == ("RGB", (4, 3))


def test_trim_padding_handles_either_side():
    vectors = np.arange(12.0).reshape(2, 3, 2)
    mask = np.array([[1, 1, 0], [0, 1, 1]])
    first, second = trim_padding(vectors, mask)
    assert first.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert second.tolist() == [[8.0, 9.0], [10.0, 11.0]]


def test_maxsim_sums_each_query_tokens_best_match():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    page = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])
    assert maxsim(query, page) == pytest.approx(1.0 + 0.8)


def test_token_cosines():
    first = np.array([[1.0, 0.0], [0.0, 1.0]])
    second = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert token_cosines(first, second).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError):
        token_cosines(first, second[:1])
