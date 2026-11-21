"""Tests for the hand-grading sheet."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("label_sheet", ROOT / "scripts" / "label_sheet.py")
label_sheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_sheet)


def row(query_id, language, found=True):
    return {
        "query_id": query_id, "language": language, "question": f"Q {query_id}?", "reference_answers": ["ref"],
        "answer": "ans", "found": found,
    }


ANSWERS = {
    "5x640": {f"q{i}": row(f"q{i}", "en" if i % 2 else "fr") for i in range(10)},
    "10x640": {f"q{i}": row(f"q{i}", "en" if i % 2 else "fr", found=None if i == 3 else True) for i in range(10)},
}


def test_sample_balances_languages_spreads_settings_and_skips_failed_replies():
    picks = label_sheet.sample(ANSWERS, 6, 0)
    assert len(picks) == 6 and len({q for q, _ in picks}) == 6
    assert sum(ANSWERS["5x640"][q]["language"] == "en" for q, _ in picks) == 3
    assert {s for _, s in picks} == {"5x640", "10x640"}
    assert all(ANSWERS[s][q]["found"] is not None for q, s in picks)
    assert picks == label_sheet.sample(ANSWERS, 6, 0)


def test_sheet_round_trip():
    picks = label_sheet.sample(ANSWERS, 4, 1)
    sheet = label_sheet.make_sheet(ANSWERS, picks)
    for letter in ("c", "i", "P", "p"):  # lower case is accepted
        sheet = sheet.replace("Label: \n", f"Label: {letter}\n", 1)
    labels = label_sheet.read_sheet(sheet)
    assert [(label["query_id"], label["setting"]) for label in labels] == picks
    assert [label["grade"] for label in labels] == ["correct", "incorrect", "partially correct", "partially correct"]


def test_ids_list_the_sheet_answers_without_grades():
    picks = label_sheet.sample(ANSWERS, 4, 2)
    rows = label_sheet.sheet_ids(label_sheet.make_sheet(ANSWERS, picks))
    assert [(row["query_id"], row["setting"]) for row in rows] == picks and rows[0]["number"] == 1


def test_missing_grade_is_reported():
    sheet = label_sheet.make_sheet(ANSWERS, label_sheet.sample(ANSWERS, 2, 0))
    with pytest.raises(ValueError, match="Answer 1"):
        label_sheet.read_sheet(sheet)
