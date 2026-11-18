"""Tests for the visual model check."""

import importlib.util
from pathlib import Path

import numpy as np

from doc_qa.core.schemas import Qrel, Query

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_visual_model", ROOT / "scripts" / "check_visual_model.py")
check_visual_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_visual_model)

PAGES = [f"p{i:02d}" for i in range(30)]
QUERIES = {
    "a-en": Query("a-en", "question a", "en"),
    "a-fr": Query("a-fr", "question a", "fr"),
    "b-en": Query("b-en", "question b", "en"),
    "c-en": Query("c-en", "question c", "en"),
}
QRELS = [
    Qrel("a-en", "p01", 1),
    Qrel("a-fr", "p01", 1),
    Qrel("b-en", "p02", 1),
    Qrel("b-en", "p03", 1),
    Qrel("c-en", "p04", 0),
    Qrel("c-en", "missing", 1),
]


def test_sample_keeps_translations_together_and_a_relevant_page():
    query_ids, pages, relevant = check_visual_model.choose_sample(QUERIES, QRELS, PAGES, 2, 10, 0)
    assert query_ids == ["a-en", "a-fr", "b-en"]  # c-en has no usable relevant page
    assert len(pages) == 10 and set(pages) <= set(PAGES)
    assert all(relevant[query_id] for query_id in query_ids)
    assert relevant["a-en"] == relevant["a-fr"] == {"p01"}


def test_sample_is_reproducible_and_limited_by_the_data():
    first = check_visual_model.choose_sample(QUERIES, QRELS, PAGES, 1, 5, 3)
    assert first == check_visual_model.choose_sample(QUERIES, QRELS, PAGES, 1, 5, 3)
    query_ids, pages, _ = check_visual_model.choose_sample(QUERIES, QRELS, PAGES[:4], 9, 99, 0)
    assert query_ids == ["a-en", "a-fr", "b-en"]
    assert pages == PAGES[:4]


def test_retrieval_check():
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.2, 0.9], [0.3, 0.9, 0.1]])
    result = check_visual_model.retrieval_check(
        ["q1", "q2", "q3"], ["en", "en", "fr"], {"q1": {"a"}, "q2": {"b"}, "q3": {"b", "c"}}, ["a", "b", "c"], scores
    )
    assert result["en"] == {"queries": 2, "top1": 0.5, "mrr": 0.75, "top1_by_chance": 0.333}
    assert result["fr"] == {"queries": 1, "top1": 1.0, "mrr": 1.0, "top1_by_chance": 0.667}


def test_cosine_stats_and_spread():
    same = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    other = [np.array([[1.0, 0.0]]), np.array([[0.6, 0.8]])]
    assert check_visual_model.cosine_stats(same, other) == {"mean": 0.9, "min": 0.8}
    assert check_visual_model.spread([2, 4]) == {"min": 2, "mean": 3.0, "max": 4}
