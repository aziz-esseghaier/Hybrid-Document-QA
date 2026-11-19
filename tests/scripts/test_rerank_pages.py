"""Tests for the reranking script."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rerank_pages", ROOT / "scripts" / "rerank_pages.py")
rerank_pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerank_pages)


def test_rerank_order_sorts_by_score_and_keeps_ties_in_visual_order():
    order = rerank_pages.rerank_order(["a", "b", "c", "d"], [0.5, 2.0, 0.5, -1.0])
    assert order == [["b", 2.0], ["a", 0.5], ["c", 0.5], ["d", -1.0]]
