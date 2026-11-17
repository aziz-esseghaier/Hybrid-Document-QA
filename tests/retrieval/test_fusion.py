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


def test_vectors_roundtrip(tmp_path):
    vectors = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    save_vectors(tmp_path / "w0.npz", ["a", "b"], vectors[:2])
    save_vectors(tmp_path / "w1.npz", ["c"], vectors[2:])
    loaded = load_vectors(sorted(tmp_path.glob("w*.npz")))
    assert list(loaded) == ["a", "b", "c"]
    assert loaded["c"].dtype == np.float32
    np.testing.assert_allclose(loaded["c"], vectors[2], atol=1e-3)
