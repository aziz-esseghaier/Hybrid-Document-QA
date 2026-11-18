"""Tests for data-parallel helpers."""

import csv
import sys

from doc_qa.core.parallel import append_jsonl, batched, launch_workers, read_jsonl, write_csv


def test_batched():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(batched([], 3)) == []


def test_jsonl_survives_truncated_line(tmp_path):
    path = tmp_path / "shard0.jsonl"
    append_jsonl(path, [{"page_id": "a"}])
    with path.open("a", encoding="utf-8") as f:
        f.write('{"page_id": "b", "te')  # interrupted write
    append_jsonl(path, [{"page_id": "c"}])
    assert [r["page_id"] for r in read_jsonl([path, tmp_path / "missing.jsonl"])] == ["a", "c"]


def test_write_csv(tmp_path):
    path = tmp_path / "sub" / "rows.csv"
    write_csv(path, [{"a": 1, "b": None}, {"a": 2, "b": "x"}])
    with path.open(encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f)) == [{"a": "1", "b": ""}, {"a": "2", "b": "x"}]


def test_launch_workers_sets_gpu_per_worker(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        "import os, sys\n"
        "i, n = sys.argv[sys.argv.index('--worker-index') + 1], sys.argv[sys.argv.index('--num-workers') + 1]\n"
        f"open(r'{tmp_path}' + f'/w{{i}}.txt', 'w').write(os.environ['CUDA_VISIBLE_DEVICES'] + ' ' + n)\n",
        encoding="utf-8",
    )
    launch_workers([sys.executable, str(script)], 2)
    assert (tmp_path / "w0.txt").read_text() == "0 2"
    assert (tmp_path / "w1.txt").read_text() == "1 2"
