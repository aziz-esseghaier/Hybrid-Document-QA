"""Tests for the final report."""

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("final_report", ROOT / "scripts" / "final_report.py")
final_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final_report)


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return str(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def accuracy(mean):
    return {
        "questions": 10, "accuracy": {"mean": mean, "low": mean - 0.1, "high": mean + 0.1},
        "correct_or_partial": mean + 0.1, "relevant_page_given": 0.9,
    }


def interval(mean):
    return {"mean": mean, "low": mean - 0.05, "high": mean + 0.05}


def test_ndcg5_counts_english_queries_only(tmp_path):
    write_rows(tmp_path / "qrels.jsonl", [
        {"query_id": "q1", "language": "en", "pages": {"p1": 1}},
        {"query_id": "q2", "language": "fr", "pages": {"p1": 1}},
    ])
    write_rows(tmp_path / "rankings.jsonl", [{"query_id": "q1", "pages": [["p2", 0.9], ["p1", 0.8]]}])
    value, count = final_report.ndcg5_english("t", str(tmp_path / "qrels.jsonl"), str(tmp_path / "rankings.jsonl"))
    assert (value, count) == (0.6309, 1)


def test_report_puts_test_dev_and_published_numbers_side_by_side(tmp_path, monkeypatch):
    methods = list(final_report.METHODS)
    datasets = tmp_path / "datasets.yaml"
    datasets.write_text(yaml.safe_dump({"datasets": {"t": {"role": "test"}, "d": {"role": "dev"}}}))
    text_metrics = {m: {"all": {"ndcg@10": 0.5}, "en": {"ndcg@10": 0.4}, "fr": {"ndcg@10": 0.6}}
                    for m in ("bm25", "dense", "rrf")}
    write_json(tmp_path / "text_d.json", text_metrics)
    write_rows(tmp_path / "qrels_t.jsonl", [{"query_id": "q1", "language": "en", "pages": {"p1": 1}}])
    write_rows(tmp_path / "rank_t.jsonl", [{"query_id": "q1", "pages": [["p1", 0.9]]}])
    cfg = {
        "datasets_config": str(datasets),
        "test": {
            "retrieval": write_json(tmp_path / "test_retrieval.json", {
                "macro_ndcg@10": {m: 0.6 for m in methods},
                "ndcg@10": {m: {"t/en": 0.7, "t/fr": 0.5} for m in methods},
                "best_minus": {m: {"all": {"delta": 0.02, "low": 0.01, "high": 0.03}} for m in methods if m != "best"},
            }),
            "answers": write_json(tmp_path / "test_answers.json", {
                "groups": {"t/en": accuracy(0.8), "t/fr": accuracy(0.6)},
                "macro": {"all": interval(0.7), "en": interval(0.8), "fr": interval(0.6)},
            }),
            "qrels": str(tmp_path / "qrels_{dataset}.jsonl"),
            "visual_rankings": str(tmp_path / "rank_{dataset}.jsonl"),
        },
        "dev": {
            "retrieval": write_json(tmp_path / "dev_retrieval.json", {
                "macro_ndcg@10": {"visual": 0.708, "rerank": 0.7, "best": 0.723},
            }),
            "text": str(tmp_path / "text_{dataset}.json"),
            "answers": write_json(tmp_path / "dev_answers.json", {"macro": {"all": interval(0.7)}}),
        },
        "paper": {"answer_accuracy": {"t": 65.0}, "answer_accuracy_mean": 65.7, "answer_accuracy_hard": 54.7},
        "model_card": {"ndcg@5_en": {"t": 0.9}},
    }
    config = tmp_path / "report.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(sys, "argv", ["final_report.py", "--config", str(config)])
    final_report.main()

    report = (tmp_path / "out" / "final_report" / "report.md").read_text(encoding="utf-8")
    assert "| t/en | 0.700 |" in report and "Mean, dev | 0.500 | 0.500 | 0.500 | 0.708 | 0.700 | 0.723 |" in report
    assert "| t | 80.0 | 60.0 | **70.0** | 80.0 | 90.0 | 65.0 |" in report and "**65.7**" in report
    assert "| t | 1 | 1.000 | 0.900 | +0.100 |" in report
    summary = json.loads((tmp_path / "out" / "final_report" / "summary.json").read_text())
    assert summary["answers"]["per_dataset"]["t"]["accuracy"] == 0.7
