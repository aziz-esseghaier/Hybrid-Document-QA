"""Tests for the garbled-text check."""

import importlib.util
import random
from pathlib import Path

import pytest
import yaml

from doc_qa.ingest.pdf import garbled_reasons, text_signals

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_garbled", ROOT / "scripts" / "check_garbled.py")
check_garbled = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_garbled)

TEXT = (
    "The company reduced the emissions of its stores by twelve percent in the last year.\n"
    "This progress comes from the energy plan that our teams have built with their suppliers.\n"
) * 8


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "configs" / "garbled.yaml").read_text(encoding="utf-8"))


def test_shift_codes():
    assert check_garbled.shift_codes("Contents 12", random.Random(0)) == "&RQWHQWV 12"


@pytest.mark.parametrize("name", list(check_garbled.CORRUPTIONS))
def test_default_rule_catches_every_corruption(name, cfg):
    corrupted = check_garbled.CORRUPTIONS[name](TEXT, random.Random(0))
    assert corrupted != TEXT
    assert garbled_reasons(text_signals(corrupted), cfg["rule"])
    assert not garbled_reasons(text_signals(TEXT), cfg["rule"])


def test_flag_rates_and_sweep(cfg):
    clean, bad = text_signals(TEXT), text_signals("�" * 100)
    groups = {"clean": [clean, clean], "mixed": [clean, bad], "empty": []}
    assert check_garbled.flag_rates(groups, cfg["rule"]) == {"clean": 0.0, "mixed": 0.5, "empty": 0.0}

    cfg["sweeps"] = {"bad_char_share": [0.5]}
    result = check_garbled.sweep(groups, cfg)
    assert result["bad_char_share"] == [{"threshold": 0.5, "flagged": {"clean": 0.0, "mixed": 0.5, "empty": 0.0}}]


def test_every_swept_signal_has_a_rule_key(cfg):
    assert set(cfg["sweeps"]) == set(check_garbled.RULE_KEYS)
    assert set(check_garbled.RULE_KEYS.values()) <= set(cfg["rule"])
