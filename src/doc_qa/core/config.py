"""Settings from environment variables and YAML configs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    """Machine-specific settings.

    Attributes:
        data_root (Path): Datasets and derived data.
        results_root (Path): Experiment outputs.
        qdrant_url (str): Qdrant server URL.
    """

    data_root: Path
    results_root: Path
    qdrant_url: str


def _resolve(path: str) -> Path:
    """Resolve a path relative to the repo root.

    Args:
        path (str): Absolute or repo-relative path.

    Returns:
        Path: Absolute path.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def get_settings() -> Settings:
    """Read settings from the environment, falling back to `.env`.

    Returns:
        Settings: Resolved settings.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    return Settings(
        data_root=_resolve(os.environ.get("DATA_ROOT", "data")),
        results_root=_resolve(os.environ.get("RESULTS_ROOT", "results")),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        path (Path): Absolute or repo-relative path.

    Returns:
        dict[str, Any]: Parsed content ({} if empty).
    """
    path = path if path.is_absolute() else REPO_ROOT / path
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
