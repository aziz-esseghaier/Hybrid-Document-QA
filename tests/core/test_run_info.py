"""Tests for the run record: complete enough to reproduce a run, and safe to publish."""

import json

from doc_qa.core.config import REPO_ROOT
from doc_qa.core.run_info import collect_run_info, write_run_info


def test_run_info_has_no_secrets_or_paths(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value_123")
    info = collect_run_info(seed=0, extra={"datasets": {"v3_hr": "abc"}})
    dumped = json.dumps(info)
    assert "hf_secret_value_123" not in dumped
    assert str(REPO_ROOT) not in dumped
    assert str(REPO_ROOT).replace("\\", "/") not in dumped
    assert info["seed"] == 0 and info["datasets"] == {"v3_hr": "abc"}
    assert {"git_commit", "git_dirty", "packages", "gpus", "python"} <= info.keys()


def test_packages_have_no_local_paths():
    info = collect_run_info()
    assert info["packages"]
    assert all("==" in p and "file:" not in p and "@" not in p for p in info["packages"])


def test_write_run_info(tmp_path):
    write_run_info(tmp_path / "exp", seed=1)
    saved = json.loads((tmp_path / "exp" / "run_info.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 1
