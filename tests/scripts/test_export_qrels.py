"""Tests for exporting relevance judgments."""

import importlib.util
from pathlib import Path

from doc_qa.core.schemas import Benchmark, Qrel, Query

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("export_qrels", ROOT / "scripts" / "export_qrels.py")
export_qrels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_qrels)


def test_query_records_group_judgments_without_text():
    queries = {"d/q2": Query("d/q2", "secret question", "fr"), "d/q1": Query("d/q1", "question", "en")}
    qrels = [Qrel("d/q2", "d/p9", 1), Qrel("d/q1", "d/p2", 2), Qrel("d/q1", "d/p1", 0)]
    records = export_qrels.query_records(Benchmark("d", "dev", {}, {}, queries, qrels))
    assert records == [
        {"query_id": "d/q1", "language": "en", "pages": {"d/p1": 0, "d/p2": 2}},
        {"query_id": "d/q2", "language": "fr", "pages": {"d/p9": 1}},
    ]
    assert "question" not in str(records)
