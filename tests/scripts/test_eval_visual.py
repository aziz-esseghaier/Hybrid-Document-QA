"""Tests for the visual evaluation script."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from doc_qa.retrieval.visual import save_multivectors

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("eval_visual", ROOT / "scripts" / "eval_visual.py")
eval_visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_visual)


def test_rank_pages_keeps_the_best_and_breaks_ties_by_id():
    ranking = eval_visual.rank_pages(["c", "a", "b"], [1.0, 1.0, 2.0], 2)
    assert ranking == [("b", 2.0), ("a", 1.0)]


def test_mean_overlap():
    first = {"q1": [("a", 1.0), ("b", 0.5)], "q2": [("c", 1.0), ("d", 0.5)]}
    second = {"q1": [("b", 1.0), ("a", 0.5)], "q2": [("c", 1.0), ("e", 0.5)]}
    assert eval_visual.mean_overlap(first, second, k=2) == 0.75


def test_exact_rankings_cover_every_file(tmp_path):
    x, y = np.array([[1.0, 0.0]], dtype=np.float32), np.array([[0.0, 1.0]], dtype=np.float32)
    save_multivectors(tmp_path / "w0.npz", ["p1"], [x])
    save_multivectors(tmp_path / "w1.npz", ["p2"], [np.concatenate([y, y])])
    rankings = eval_visual.exact_rankings(["q"], {"q": y}, [tmp_path / "w0.npz", tmp_path / "w1.npz"], 5)
    assert [page for page, _ in rankings["q"]] == ["p2", "p1"]
    assert rankings["q"][0][1] == pytest.approx(1.0)
