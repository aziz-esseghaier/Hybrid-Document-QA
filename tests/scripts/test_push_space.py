"""Tests for assembling the demo Space."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("push_space", ROOT / "scripts" / "push_space.py")
push_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push_space)


def test_space_has_the_app_its_config_and_the_package(tmp_path):
    files = push_space.build_space(tmp_path)
    assert {"app.py", "requirements.txt", "README.md", "demo.yaml", "doc_qa/pipeline.py"} <= set(files)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in files)
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "sdk: gradio" in card
    [description] = [line for line in card.splitlines() if line.startswith("short_description:")]
    assert len(description.split(":", 1)[1].strip()) <= 60  # Hugging Face's limit
