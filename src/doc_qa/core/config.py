"""Settings from environment variables and YAML config files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/doc_qa/core/config.py -> the repo root is three levels above this file's folder.
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    """Machine-specific settings. They come from the environment, so the same code runs locally and on Kaggle.

    Attributes:
        data_root (Path): Folder for datasets and derived data (page hashes, page images, ...).
        results_root (Path): Folder holding one subfolder per experiment.
        qdrant_url (str): URL of the Qdrant server.
    """

    data_root: Path
    results_root: Path
    qdrant_url: str


def _resolve(path: str) -> Path:
    """Turn a path from the environment into an absolute path.

    Args:
        path (str): Absolute path, or path relative to the repo root.

    Returns:
        Path: The absolute path.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def get_settings() -> Settings:
    """Read the settings from environment variables, after loading the repo's `.env` file if present.

    Variables already set in the environment (for example by scripts/kaggle_setup.py) win over `.env`.

    Returns:
        Settings: The resolved settings.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    return Settings(
        data_root=_resolve(os.environ.get("DATA_ROOT", "data")),
        results_root=_resolve(os.environ.get("RESULTS_ROOT", "results")),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Args:
        path (Path): Absolute path, or path relative to the repo root.

    Returns:
        dict[str, Any]: The parsed content (empty dict for an empty file).
    """
    path = path if path.is_absolute() else REPO_ROOT / path
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
