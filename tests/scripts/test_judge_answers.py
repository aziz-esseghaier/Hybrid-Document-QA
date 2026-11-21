"""Tests for judging saved answers and comparing judges with hand grades."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


judge_answers = load("judge_answers")
compare_judges = load("compare_judges")


def test_answers_keep_the_last_row_per_question(tmp_path):
    rows = [{"query_id": "q1", "answer": "old"}, {"query_id": "q1", "answer": "new"}, {"query_id": "q2", "answer": "x"}]
    (tmp_path / "answers_5x640.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    answers = judge_answers.load_answers(tmp_path)
    assert list(answers) == ["5x640"] and answers["5x640"]["q1"]["answer"] == "new"


def test_summary_counts_labels_per_language():
    verdicts = [
        {"language": "en", "label": "correct"},
        {"language": "en", "label": "incorrect"},
        {"language": "fr", "label": "partially correct"},
        {"language": "fr", "label": None},
    ]
    summary = judge_answers.summarize(verdicts)
    assert summary["all"] == {
        "questions": 4, "accuracy": 0.25, "correct": 1, "partially correct": 1, "incorrect": 1, "unjudged": 1,
    }
    assert summary["en"]["accuracy"] == 0.5 and summary["fr"]["accuracy"] == 0.0


def test_agreement_with_hand_grades():
    human = ["correct", "correct", "incorrect", "partially correct"]
    judge = ["correct", "partially correct", "incorrect", None]
    result = compare_judges.agreement(human, judge)
    assert (result["answers"], result["agreement"], result["agreement_correct_or_not"]) == (4, 0.5, 0.75)
    assert (result["accuracy_by_hand"], result["accuracy_by_judge"]) == (0.5, 0.25)
    assert result["confusion"]["correct"] == {"correct": 1, "partially correct": 1}
    assert result["confusion"]["partially correct"] == {"unjudged": 1}
