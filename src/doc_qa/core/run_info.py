"""Run metadata (run_info.json). Never records env vars, git remote, absolute paths or hostname."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doc_qa.core.config import REPO_ROOT


def _run(cmd: list[str]) -> str | None:
    """Run a command in the repo root.

    Args:
        cmd (list[str]): Command and arguments.

    Returns:
        str | None: Stripped stdout, or None on failure.
    """
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> tuple[str | None, bool | None]:
    """Return the HEAD commit and whether code differs from it (data/ and results/ ignored).

    Returns:
        tuple[str | None, bool | None]: (commit, dirty), or (None, None) outside git.
    """
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return None, None
    status = _run(["git", "status", "--porcelain", "--", ".", ":(exclude)data", ":(exclude)results"])
    return commit, bool(status)


def installed_packages() -> list[str]:
    """List installed packages as `name==version`.

    Returns:
        list[str]: Sorted pins.
    """
    pins = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pins.add(f"{name}=={dist.version}")
    return sorted(pins, key=str.lower)


def gpu_names() -> list[str]:
    """List visible GPUs.

    Returns:
        list[str]: GPU names; empty if none.
    """
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return [line.strip() for line in out.splitlines() if line.strip()] if out else []


def collect_run_info(seed: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect run metadata.

    Args:
        seed (int | None): Random seed.
        extra (dict[str, Any] | None): Run-specific fields, e.g. dataset revisions.

    Returns:
        dict[str, Any]: Run metadata.
    """
    commit, dirty = git_state()
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": seed,
        "python": platform.python_version(),
        "os": platform.system(),
        "gpus": gpu_names(),
        **(extra or {}),
        "packages": installed_packages(),
    }


def write_run_info(out_dir: Path, seed: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write run_info.json to `out_dir`. Warns on uncommitted changes.

    Args:
        out_dir (Path): Experiment folder.
        seed (int | None): Random seed.
        extra (dict[str, Any] | None): Run-specific fields.

    Returns:
        dict[str, Any]: Written metadata.
    """
    info = collect_run_info(seed=seed, extra=extra)
    if info["git_dirty"]:
        print("WARNING: uncommitted code changes; git_commit alone does not describe this run.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
