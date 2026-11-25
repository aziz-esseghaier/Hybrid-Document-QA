"""Tests for locating saved results."""

from doc_qa.core.config import REPO_ROOT, result_path, role_name


def test_result_path_prefers_the_copy_under_results_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    assert result_path("results/x/rows.jsonl") == REPO_ROOT / "results" / "x" / "rows.jsonl"
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "rows.jsonl").write_text("")
    assert result_path("results/x/rows.jsonl") == tmp_path / "x" / "rows.jsonl"
    assert result_path("configs/answer.yaml") == REPO_ROOT / "configs" / "answer.yaml"
    assert result_path(tmp_path / "x" / "rows.jsonl") == tmp_path / "x" / "rows.jsonl"


def test_role_name():
    assert (role_name("qrels", "dev"), role_name("qrels", "test")) == ("qrels", "qrels_test")
