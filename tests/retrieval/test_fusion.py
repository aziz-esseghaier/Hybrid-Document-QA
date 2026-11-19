"""Tests for page aggregation, rank fusion and vector storage."""

import numpy as np
import pytest

from doc_qa.retrieval.dense import load_vectors, save_vectors
from doc_qa.retrieval.fusion import page_ranking, rrf


def test_page_ranking_keeps_best_chunk():
    hits = [("p1", 0.5), ("p2", 0.9), ("p1", 0.95), ("p3", 0.9)]
    assert page_ranking(hits) == [("p1", 0.95), ("p2", 0.9), ("p3", 0.9)]
    assert page_ranking(hits, limit=1) == [("p1", 0.95)]


def test_rrf():
    merged = rrf([["a", "b", "c"], ["b", "d"]], k=60)
    assert [i for i, _ in merged] == ["b", "a", "d", "c"]
    assert merged[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert len(rrf([["a", "b"], ["c"]], limit=2)) == 2


def test_weighted_rrf():
    lists = [["a", "b"], ["b", "c"]]
    assert rrf(lists, weights=[1, 1]) == rrf(lists)
    assert [i for i, _ in rrf([["a"], ["b"]])] == ["a", "b"]  # tie, broken by id
    heavy = rrf([["a"], ["b"]], k=60, weights=[1, 3])
    assert [i for i, _ in heavy] == ["b", "a"]
    assert heavy[0][1] == pytest.approx(3 / 61)
    assert [i for i, _ in rrf(lists, weights=[1, 0])] == ["a", "b"]  # weight 0 drops the list's pages
    with pytest.raises(ValueError):
        rrf(lists, weights=[1])


def test_vectors_roundtrip(tmp_path):
    vectors = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    save_vectors(tmp_path / "w0.npz", ["a", "b"], vectors[:2])
    save_vectors(tmp_path / "w1.npz", ["c"], vectors[2:])
    loaded = load_vectors(sorted(tmp_path.glob("w*.npz")))
    assert list(loaded) == ["a", "b", "c"]
    assert loaded["c"].dtype == np.float32
    np.testing.assert_allclose(loaded["c"], vectors[2], atol=1e-3)
