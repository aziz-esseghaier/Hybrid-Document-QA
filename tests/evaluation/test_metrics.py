"""Tests for retrieval metrics."""

import math

import pytest

from doc_qa.evaluation.metrics import evaluate, ndcg_at_k, recall_at_k


def test_ndcg_linear_gain():
    relevance = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], relevance) == pytest.approx(1.0)
    expected = (1 / math.log2(2) + 2 / math.log2(3)) / (2 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], relevance) == pytest.approx(expected)
    assert ndcg_at_k(["x"] * 10 + ["a"], relevance) == 0.0  # beyond the cutoff
    assert ndcg_at_k(["a"], {}) == 0.0


def test_recall():
    assert recall_at_k(["a", "x"], {"a": 1, "b": 1}) == 0.5
    assert recall_at_k(["a", "b"], {"a": 1, "b": 0}) == 1.0


def test_evaluate_groups_by_language():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    rankings = {"q1": ["a"], "q2": ["x"]}
    result = evaluate(rankings, qrels, {"q1": "en", "q2": "fr"})
    assert result["en"]["ndcg@10"] == 1.0
    assert result["fr"]["ndcg@10"] == 0.0
    assert result["all"] == {"queries": 2, "ndcg@10": 0.5, "recall@100": 0.5}
