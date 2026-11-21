"""Tests for parsing and checking the answer model's reply."""

import pytest

from doc_qa.answer.output import Answer, InvalidAnswer, cited_page_ids, parse_answer


def test_parses_json_inside_fences_and_reasoning():
    reply = '<think>look at page 2</think>```json\n{"answer": " 42 % ", "pages": [2, 1, 2], "found": true}\n```'
    assert parse_answer(reply, 3) == Answer("42 %", (2, 1), True)


def test_not_found_answer():
    reply = '{"answer": "Les pages ne le disent pas.", "pages": [], "found": false}'
    assert parse_answer(reply, 5) == Answer("Les pages ne le disent pas.", (), False)


@pytest.mark.parametrize(
    "reply, reason",
    [
        ("The answer is 42.", "no JSON object"),
        ('{"answer": "42", "pages": [1], "found": true', "no JSON object"),
        ('{"answer": "42" "pages": [1]}', "malformed"),
        ("[1, 2]", "no JSON object"),
        ('{"answer": "42", "pages": [1]}', "keys missing: found"),
        ('{"answer": 42, "pages": [1], "found": true}', '"answer" must be a string'),
        ('{"answer": "42", "pages": [1], "found": "true"}', '"found" must be true or false'),
        ('{"answer": "42", "pages": ["one"], "found": true}', '"pages" must be a list'),
        ('{"answer": "42", "pages": [true], "found": true}', '"pages" must be a list'),
        ('{"answer": "42", "pages": 1, "found": true}', '"pages" must be a list'),
        ('{"answer": "42", "pages": [0, 4], "found": true}', r"pages \[0, 4\] do not exist"),
        ('{"answer": " ", "pages": [1], "found": true}', '"answer" is empty'),
        ('{"answer": "42", "pages": [], "found": true}', "must cite at least one page"),
        ('{"answer": "No.", "pages": [1], "found": false}', "must be empty"),
    ],
)
def test_rejects_replies_that_break_the_rules(reply, reason):
    with pytest.raises(InvalidAnswer, match=reason):
        parse_answer(reply, 3)


def test_labels_written_as_text_are_accepted():
    reply = '{"answer": "42", "pages": ["2", "Page 3", "[Page 1]", 2.0], "found": true}'
    assert parse_answer(reply, 3).pages == (2, 3, 1)


def test_cited_labels_map_back_to_page_ids():
    assert cited_page_ids(Answer("x", (3, 1), True), ["d/9", "d/4", "d/7"]) == ["d/7", "d/9"]
