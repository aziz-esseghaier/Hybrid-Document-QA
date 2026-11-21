"""Tests for the judge prompt, reply checks and retries, and kappa."""

import pytest

from doc_qa.answer.output import InvalidAnswer
from doc_qa.evaluation.judge import judge_messages, judge_with_retries, parse_verdict, with_system_prompt
from doc_qa.evaluation.metrics import cohen_kappa


def test_prompt_holds_question_references_and_answer():
    [message] = judge_messages("How much?", ["7 %", "7 % et 23 %"], "Seven percent.")
    text = message["content"][0]["text"]
    assert message["role"] == "user"
    assert "Question: How much?" in text and "- 7 %\n- 7 % et 23 %" in text and "Seven percent." in text


@pytest.mark.parametrize(
    "reply, label",
    [
        ('{"explanation": "Same figure.", "judgment": "Correct"}', "correct"),
        ('```json\n{"explanation": "x", "judgment": "Partially Correct"}\n```', "partially correct"),
        ('{"judgment": "partially_correct"}', "partially correct"),
        ('{"explanation": "Wrong year.", "judgment": " INCORRECT "}', "incorrect"),
    ],
)
def test_labels_are_normalized(reply, label):
    assert parse_verdict(reply)[0] == label


@pytest.mark.parametrize("reply", ['{"explanation": "x"}', '{"judgment": "Mostly right"}', "Correct", '{"judgment": 1}'])
def test_invalid_replies(reply):
    with pytest.raises(InvalidAnswer):
        parse_verdict(reply)


def test_retry_then_give_up():
    replies = iter(["Correct!", '{"judgment": "Correct", "explanation": "ok"}'])
    verdict = judge_with_retries(lambda messages: next(replies), judge_messages("q", ["r"], "a"), retries=2)
    assert (verdict.label, verdict.explanation, verdict.attempts) == ("correct", "ok", 2)
    failed = judge_with_retries(lambda messages: "no", judge_messages("q", ["r"], "a"), retries=1)
    assert failed.label is None and failed.attempts == 2 and len(failed.errors) == 2


def test_fixed_system_prompt_goes_first():
    messages = judge_messages("q", ["r"], "a")
    assert with_system_prompt(messages, None) == messages
    first, *rest = with_system_prompt(messages, "Grade carefully.")
    assert first == {"role": "system", "content": [{"type": "text", "text": "Grade carefully."}]} and rest == messages


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0  # one label throughout
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0  # chance level
    assert cohen_kappa(["a", "b"], ["b", "a"]) == pytest.approx(-1.0)
