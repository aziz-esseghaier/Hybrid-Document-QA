"""Tests for the OCR store manifest."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from doc_qa.core.parallel import append_jsonl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ocr_store.py"
spec = importlib.util.spec_from_file_location("ocr_store", SCRIPT)
ocr_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocr_store)


def make_run(tmp_path: Path, pages_total: int) -> tuple[list[Path], Path]:
    shard = tmp_path / "shard0.jsonl"
    append_jsonl(shard, [{"page_id": f"p{i}", "text": "x", "seconds": 1.0} for i in range(3)])
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"pages_total": pages_total}))
    (run / "config.yaml").write_text(yaml.safe_dump({"pipeline_version": "v1.6", "batch_size": 4}))
    (run / "run_info.json").write_text(json.dumps({
        "git_commit": "abc1234def", "git_dirty": False,
        "dataset": {"repo": "vidore/x", "revision": "r1"},
        "packages": ["numpy==2.0", "paddleocr==3.6.0", "PaddlePaddle-GPU==3.2.1"],
    }))
    return [shard], run


def test_manifest(tmp_path):
    shards, run = make_run(tmp_path, pages_total=3)
    m = ocr_store.build_manifest("v2_x", shards, run)
    assert (m["pages"], m["pages_total"], m["pipeline_version"]) == (3, 3, "v1.6")
    assert m["source"] == {"repo": "vidore/x", "revision": "r1"}
    assert m["packages"] == ["paddleocr==3.6.0", "PaddlePaddle-GPU==3.2.1"]


def test_manifest_rejects_partial_run(tmp_path):
    shards, run = make_run(tmp_path, pages_total=5)
    with pytest.raises(SystemExit):
        ocr_store.build_manifest("v2_x", shards, run)
    assert ocr_store.build_manifest("v2_x", shards, run, allow_partial=True)["pages"] == 3


def test_handle(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    assert ocr_store.dataset_handle("v2_esg_reports") == "me/ocr-v2-esg-reports"
