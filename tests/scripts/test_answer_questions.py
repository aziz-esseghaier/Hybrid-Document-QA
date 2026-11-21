"""Tests for summarizing saved answer records."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("answer_questions", ROOT / "scripts" / "answer_questions.py")
answer_questions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer_questions)


def record(query_id, language="en", found=True, cited=("p1",), error=None, prompt_tokens=5000):
    return {
        "query_id": query_id, "language": language, "given": ["p1", "p2"], "found": found, "cited": list(cited),
        "attempts": 1, "seconds": 20.0, "prompt_tokens": prompt_tokens, "new_tokens": 100, "finite": True,
        "error": error,
    }


def test_setting_name():
    assert answer_questions.setting_name(5, 640) == "5x640"


def test_summary_counts_failures_and_leaves_them_out_of_the_cost():
    records = [
        record("q1"),
        record("q2", language="fr", cited=("p2",)),
        record("q3", found=None, cited=(), error="out of memory", prompt_tokens=0),
    ]
    summary = answer_questions.summarize_setting(records, {"q1": {"p1": 1}, "q2": {"p1": 1}})
    assert summary["answers"]["all"]["questions"] == 3
    assert summary["answers"]["all"]["failed"] == 0.3333
    assert summary["answers"]["all"]["citation_precision"] == 0.5
    assert summary["prompt_tokens"] == {"min": 5000, "mean": 5000.0, "max": 5000}
    assert (summary["out_of_memory"], summary["replies_with_nan_or_inf"]) == (1, 0)


def test_summary_of_an_empty_setting():
    summary = answer_questions.summarize_setting([], {})
    assert summary["answers"]["all"]["questions"] == 0 and summary["prompt_tokens"] is None
