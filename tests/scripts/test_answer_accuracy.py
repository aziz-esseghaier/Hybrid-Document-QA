"""Tests for the dev answer accuracy summary."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("answer_accuracy", ROOT / "scripts" / "answer_accuracy.py")
answer_accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer_accuracy)


def row(dataset, language, cluster, value, relevant_given=True, lenient=None):
    return {
        "dataset": dataset, "group": f"{dataset}/{language}", "language": language, "cluster": (dataset, cluster),
        "value": float(value), "lenient": bool(value) if lenient is None else lenient, "relevant_given": relevant_given,
    }


ROWS = [
    row("a", "en", "q1", 1), row("a", "fr", "q1", 1), row("a", "en", "q2", 0), row("a", "fr", "q2", 1),
    row("b", "fr", "q3", 0, relevant_given=False, lenient=True), row("b", "fr", "q4", 1),
]


def test_macro_weighs_each_group_the_same():
    result = answer_accuracy.macro_interval(ROWS, ["a/en", "a/fr", "b/fr"], 500, 0)
    assert result["mean"] == round((0.5 + 1.0 + 0.5) / 3, 4)
    assert 0 <= result["low"] <= result["mean"] <= result["high"] <= 1
    assert answer_accuracy.macro_interval(ROWS, ["a/fr", "b/fr"], 200, 0)["mean"] == 0.75


def test_group_summary_splits_by_relevant_page_given():
    summary = answer_accuracy.group_summary([r for r in ROWS if r["group"] == "b/fr"], 200, 0)
    assert summary["questions"] == 2 and summary["accuracy"]["mean"] == 0.5
    assert summary["correct_or_partial"] == 1.0 and summary["relevant_page_given"] == 0.5
    assert (summary["accuracy_when_relevant_given"], summary["accuracy_when_none_given"]) == (1.0, 0.0)


def test_rows_join_verdicts_records_and_judgments(tmp_path):
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    write(tmp_path / "judge_j" / "d" / "verdicts_5x1280.jsonl", [
        {"query_id": "q1", "language": "en", "label": "correct"},
        {"query_id": "q2", "language": "fr", "label": "partially correct"},
    ])
    write(tmp_path / "answer_answers" / "d" / "records_5x1280.jsonl", [
        {"query_id": "q1", "given": ["p1", "p9"]}, {"query_id": "q2", "given": ["p9"]},
    ])
    write(tmp_path / "qrels" / "d.jsonl", [
        {"query_id": "q1", "pages": {"p1": 2, "p3": 0}}, {"query_id": "q2", "pages": {"p1": 2}},
    ])
    first, second = answer_accuracy.load_rows(tmp_path, "d", "j", "5x1280")
    assert (first["value"], first["relevant_given"], first["cluster"]) == (1.0, True, ("d", ("p1",)))
    assert (second["value"], second["lenient"], second["relevant_given"]) == (0.0, True, False)
    assert first["cluster"] == second["cluster"]  # translations of one question
