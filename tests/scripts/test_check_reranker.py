"""Tests for the reranker check's sampling and quality measure."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_reranker", ROOT / "scripts" / "check_reranker.py")
check_reranker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_reranker)


def test_pick_queries_keeps_translations_together():
    qrels = {"en1": {"a": 1}, "fr1": {"a": 1}, "en2": {"b": 1, "x": 0}, "fr2": {"b": 1}, "en3": {"c": 1}}
    picked = check_reranker.pick_queries(qrels, 2, seed=0)
    assert len(picked) in (3, 4)
    for pair in (("en1", "fr1"), ("en2", "fr2")):
        assert (pair[0] in picked) == (pair[1] in picked)
    assert picked == check_reranker.pick_queries(qrels, 2, seed=0)


def test_rerank_quality_measures_before_and_after():
    candidates = {"q1": ["x", "a"], "q2": ["y", "z"]}
    scores = {("q1", "x"): 0.1, ("q1", "a"): 2.0, ("q2", "y"): 1.0, ("q2", "z"): 1.0}
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    quality = check_reranker.rerank_quality(["q1", "q2"], {"q1": "en", "q2": "en"}, candidates, scores, qrels)
    assert quality["en"]["with_relevant_candidate"] == 0.5
    assert quality["en"]["ndcg@10_after"] == 0.5  # q1 moves its relevant page to the top
    assert quality["en"]["ndcg@10_before"] < 0.5
