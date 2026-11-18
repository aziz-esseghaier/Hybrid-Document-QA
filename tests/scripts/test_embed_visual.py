"""Tests for the visual embedding script."""

import importlib.util
from pathlib import Path

import numpy as np

from doc_qa.retrieval.visual import iter_multivectors, part_files, save_multivectors

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("embed_visual", ROOT / "scripts" / "embed_visual.py")
embed_visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed_visual)


def test_workers_share_all_items_and_skip_finished_ones():
    ids = ["d", "b", "a", "c", "e"]
    first = embed_visual.todo_ids(ids, {"c"}, 0, 2)
    second = embed_visual.todo_ids(ids, {"c"}, 1, 2)
    assert first == ["a", "e"]
    assert second == ["b", "d"]


def test_pool_pages_merges_each_file_once(tmp_path):
    pages, pooled = tmp_path / "pages", tmp_path / "pooled_2"
    rows = np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32)
    save_multivectors(pages / "w0_00000.npz", ["a"], [rows / np.linalg.norm(rows, axis=1, keepdims=True)])
    embed_visual.pool_pages(pages, pooled, 2)
    target = part_files(pooled)[0]
    ids, vectors = next(iter_multivectors([target]))
    assert ids == ["a"] and len(vectors[0]) <= 5

    before = target.stat().st_mtime_ns
    embed_visual.pool_pages(pages, pooled, 2)
    assert target.stat().st_mtime_ns == before
