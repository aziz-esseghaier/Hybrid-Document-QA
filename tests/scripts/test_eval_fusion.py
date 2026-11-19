"""Tests for choosing fusion weights."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eval_fusion", ROOT / "scripts" / "eval_fusion.py")
eval_fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_fusion)

RULE = eval_fusion.LANGUAGE_RULE


def query(cross_language, group="d/fr", relevance=None):
    return {
        "dataset": group.split("/")[0],
        "query_id": "q",
        "group": group,
        "cross_language": cross_language,
        "relevance": relevance or {"v": 1},
        "lists": {"visual": ["v", "b"], "dense": ["d"], "bm25": ["b"]},
        "cluster": (group, ("v",)),
    }


def test_grid_skips_visual_alone_and_rules_that_change_nothing():
    grid = {"visual": [1, 2], "dense": [0, 1], "bm25": [0, 1], RULE: [False, True]}
    settings = eval_fusion.grid_settings(grid)
    assert len(settings) == 10
    assert all(s["dense"] or s["bm25"] for s in settings)
    assert not any(s[RULE] and not s["bm25"] for s in settings)


def test_grid_without_the_language_rule():
    settings = eval_fusion.grid_settings({"visual": [1], "rerank": [0.5, 2]})
    assert [eval_fusion.label(s) for s in settings] == ["visual 1 + rerank 0.5", "visual 1 + rerank 2"]


def test_language_rule_drops_bm25_only_across_languages():
    setting = {"visual": 1, "bm25": 5, RULE: True}
    assert [p for p, _ in eval_fusion.fuse(query(cross_language=True), setting, 60, 10)] == ["v", "b"]
    assert eval_fusion.fuse(query(cross_language=False), setting, 60, 10)[0][0] == "b"  # BM25 counts, weight 5
    assert eval_fusion.fuse(query(cross_language=True), {**setting, RULE: False}, 60, 10)[0][0] == "b"


def test_macro_and_leave_one_dataset_out():
    means = [
        {"a/en": 0.9, "a/fr": 0.7, "b/en": 0.1},  # best on a
        {"a/en": 0.5, "a/fr": 0.5, "b/en": 0.6},  # best on b
    ]
    assert eval_fusion.macro(means[0], include=["a"]) == pytest.approx(0.8)
    assert eval_fusion.macro(means[0], exclude="a") == pytest.approx(0.1)
    settings = [{"visual": 1, "dense": 1}, {"visual": 2, "dense": 1}]
    held = eval_fusion.leave_one_dataset_out(settings, means, {"a/en": 0.4, "a/fr": 0.4, "b/en": 0.4}, ["a", "b"])
    assert held["a"] == {"setting": "visual 2 + dense 1", "ndcg@10": 0.5, "reference": 0.4}  # chosen on b
    assert held["b"]["ndcg@10"] == 0.1  # chosen on a


def test_compare_reports_each_group_and_all():
    queries = [query(True, "d/fr"), query(False, "d/en")]
    result = eval_fusion.compare(queries, np.array([1.0, 0.5]), np.array([0.5, 0.5]), 200, 0)
    assert result["d/fr"]["delta"] == 0.5 and result["d/en"]["delta"] == 0.0
    assert result["all"]["queries"] == 2


def test_load_queries_reads_judgments_and_rankings(tmp_path):
    (tmp_path / "qrels.jsonl").write_text(json.dumps({"query_id": "q", "language": "fr", "pages": {"v": 1, "x": 0}}))
    for source in ("visual", "dense"):
        (tmp_path / f"{source}.jsonl").write_text(json.dumps({"query_id": "q", "pages": [["v", 0.9], ["x", 0.1]]}))
    cfg = {
        "datasets": ["d"],
        "qrels": str(tmp_path / "qrels.jsonl"),
        "rankings": {source: str(tmp_path / f"{source}.jsonl") for source in ("visual", "dense")},
    }
    [loaded] = eval_fusion.load_queries(cfg, {"d": {"doc_language": "en"}})
    assert loaded["group"] == "d/fr" and loaded["cross_language"]
    assert loaded["lists"] == {"visual": ["v", "x"], "dense": ["v", "x"]}
    assert loaded["cluster"] == ("d", ("v",))
