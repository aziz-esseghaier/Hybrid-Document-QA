"""Tests for comparing answer settings by judged accuracy."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compare_settings", ROOT / "scripts" / "compare_settings.py")
compare_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_settings)


def test_paired_comparison_against_the_best_setting():
    labels = {
        "5x640": {"a-en": "correct", "a-fr": "incorrect", "b-en": "partially correct", "c-fr": "incorrect"},
        "10x640": {"a-en": "correct", "a-fr": "correct", "b-en": "correct", "c-fr": "incorrect", "extra": "correct"},
    }
    languages = {"a-en": "en", "a-fr": "fr", "b-en": "en", "c-fr": "fr", "extra": "en"}
    clusters = {"a-en": ("p1",), "a-fr": ("p1",), "b-en": ("p2",), "c-fr": ("p3",), "extra": ("p4",)}
    result = compare_settings.compare(labels, languages, clusters, resamples=200)
    assert result["questions"] == 4  # only questions answered under every setting
    assert result["best"] == "10x640"
    assert result["settings"]["10x640"]["accuracy"]["mean"] == 0.75
    assert result["settings"]["5x640"]["accuracy"]["mean"] == 0.25
    assert result["settings"]["5x640"]["correct_or_partial"] == 0.5
    assert result["settings"]["5x640"]["en"] == {"questions": 2, "accuracy": 0.5}
    assert result["minus_best"]["5x640"]["mean"] == -0.5 and "10x640" not in result["minus_best"]


def test_no_shared_question_is_an_error():
    import pytest

    with pytest.raises(ValueError, match="every setting"):
        compare_settings.compare({"a": {"q1": "correct"}, "b": {"q2": "correct"}}, {}, {})
