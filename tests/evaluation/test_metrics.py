"""Tests for retrieval metrics."""

import math

import pytest

from doc_qa.evaluation.metrics import (
    cluster_bootstrap_ci,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    text_coverage,
    token_f1,
)


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


def test_token_f1():
    assert token_f1("Scope 1 emissions: 12,5 t", "scope 1 EMISSIONS 12 5 t") == 1.0
    assert token_f1("a a b", "a c") == pytest.approx(2 * 1 / 5)
    assert token_f1("", "") == 1.0
    assert token_f1("text", "") == 0.0


def test_text_coverage():
    texts = {"a": "enough text on this page here", "b": " tiny ", "c": "", "d": "another page with enough text"}
    qrels = {"q1": {"b": 1, "c": 1}, "q2": {"a": 1, "b": 1}, "q3": {"d": 1, "c": 0}}
    assert text_coverage(qrels, texts) == {
        "pages": 4,
        "near_empty_pages": 2,
        "relevant_pages": 4,
        "near_empty_relevant_pages": 2,
        "queries": 3,
        "queries_only_near_empty": 1,
    }


def test_cluster_bootstrap_ci():
    mean, low, high = cluster_bootstrap_ci([1.0, 1.0, 0.0, 0.0], ["a", "a", "b", "b"], n_resamples=2000)
    assert mean == 0.5 and low == 0.0 and high == 1.0
    assert cluster_bootstrap_ci([0.2, 0.2, 0.2], ["a", "b", "c"], n_resamples=100) == pytest.approx((0.2, 0.2, 0.2))


def test_evaluate_groups_by_language():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    rankings = {"q1": ["a"], "q2": ["x"]}
    result = evaluate(rankings, qrels, {"q1": "en", "q2": "fr"})
    assert result["en"]["ndcg@10"] == 1.0
    assert result["fr"]["ndcg@10"] == 0.0
    assert result["all"] == {"queries": 2, "ndcg@10": 0.5, "recall@100": 0.5}
