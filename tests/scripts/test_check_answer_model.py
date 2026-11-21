"""Tests for the answer model check's question sampling."""

import importlib.util
from pathlib import Path

from doc_qa.core.schemas import Qrel, Query

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_answer_model", ROOT / "scripts" / "check_answer_model.py")
check_answer_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_answer_model)

QUERIES = {q: Query(q, "question", q[-2:]) for q in ("a-en", "a-fr", "b-en", "c-en", "d-en")}
QRELS = [Qrel("a-en", "p1", 1), Qrel("a-fr", "p1", 1), Qrel("b-en", "p2", 2), Qrel("c-en", "p3", 0)]


def test_translations_stay_together_and_questions_need_a_relevant_page():
    assert check_answer_model.choose_questions(QUERIES, QRELS, 5, 0) == ["a-en", "a-fr", "b-en"]
    one = check_answer_model.choose_questions(QUERIES, QRELS, 1, 3)
    assert one in (["a-en", "a-fr"], ["b-en"])
    assert one == check_answer_model.choose_questions(QUERIES, QRELS, 1, 3)


def test_gpu_gb():
    assert check_answer_model.gpu_gb([1_234_000_000, 0]) == [1.23, 0.0]
