"""Tests for choosing which OCR a page can reuse."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "reuse_ocr.py"
spec = importlib.util.spec_from_file_location("reuse_ocr", SCRIPT)
reuse_ocr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse_ocr)


def plan(target, source, matches=None, ocr=None):
    return reuse_ocr.plan_reuse(target, source, matches or {}, set(source) if ocr is None else ocr, 0.98, 0.02)


def test_identical_pixels_are_reused():
    assert plan({"t1": "h1"}, {"s2": "h1", "s1": "h1"}) == {"t1": ("s1", "exact")}  # duplicates: first id


def test_exact_match_needs_source_ocr():
    assert plan({"t1": "h1"}, {"s1": "h1", "s2": "h1"}, ocr={"s2"}) == {"t1": ("s2", "exact")}
    assert plan({"t1": "h1"}, {"s1": "h1"}, ocr=set()) == {}


def test_pixel_match_must_be_strong_and_unambiguous():
    source = {"s1": "x"}
    assert plan({"t": "h"}, source, {"t": ("s1", 0.995, 0.5)}) == {"t": ("s1", "pixels")}
    assert plan({"t": "h"}, source, {"t": ("s1", 0.97, 0.5)}) == {}  # too weak
    assert plan({"t": "h"}, source, {"t": ("s1", 0.995, 0.99)}) == {}  # a look-alike page exists
    assert plan({"t": "h"}, source, {"t": ("s1", 0.995, 0.5)}, ocr=set()) == {}  # no OCR to reuse
    assert plan({"t": "h"}, source) == {}  # blank page, no candidate
