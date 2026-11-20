"""Tests for the answer metrics."""

from doc_qa.evaluation.answers import AnswerRecord, citation_precision, summarize_answers


def record(language="en", given=("a", "b"), found=True, cited=("a",), attempts=1, correct=None):
    return AnswerRecord(language, given, {"a": 2, "z": 1}, found, cited, attempts, 10.0, correct)


def test_citation_precision():
    assert citation_precision(["a", "b"], {"a": 1, "b": 0}) == 0.5
    assert citation_precision([], {"a": 1}) is None


def test_summary_counts_outcomes_per_language():
    records = [
        record(correct=True),
        record(found=False, cited=(), given=("b",), correct=False),  # nothing relevant given: abstaining is right
        record(found=None, cited=(), attempts=3),  # no usable reply
        record(language="fr", cited=("a", "b"), attempts=2, correct=False),
    ]
    summary = summarize_answers(records)
    assert list(summary) == ["all", "en", "fr"]
    everything = summary["all"]
    assert everything["questions"] == 4 and everything["relevant_page_given"] == 0.75
    assert (everything["found"], everything["not_found"], everything["failed"]) == (0.5, 0.25, 0.25)
    assert everything["not_found_when_no_relevant_page"] == 1.0
    assert everything["citation_precision"] == 0.75  # mean of 1.0 and 0.5
    assert everything["retried"] == 0.5 and everything["seconds_per_question"] == 10.0
    assert (everything["judged"], everything["accuracy"]) == (3, 0.3333)
    assert summary["fr"]["questions"] == 1 and summary["fr"]["not_found_when_no_relevant_page"] is None
