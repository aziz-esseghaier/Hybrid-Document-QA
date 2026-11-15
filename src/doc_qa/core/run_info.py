"""Record what produced an experiment's results, so the run can be reproduced.

The record must be safe to publish. It never contains environment variables (tokens), the git remote URL
(a private clone URL can embed a token), absolute paths (they contain user names) or the hostname.
"""

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
    """Run a command in the repo root and return its output.

    Args:
        cmd (list[str]): Command and arguments.

    Returns:
        str | None: Stripped standard output, or None if the command is missing or fails.
    """
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> tuple[str | None, bool | None]:
    """Get the current commit hash and whether tracked or new code differs from it.

    Changes under data/ and results/ are ignored: they are outputs, not code.

    Returns:
        tuple[str | None, bool | None]: Commit hash and dirty flag, or (None, None) outside a git repo.
    """
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return None, None
    status = _run(["git", "status", "--porcelain", "--", ".", ":(exclude)data", ":(exclude)results"])
    return commit, bool(status)


def installed_packages() -> list[str]:
    """List installed packages as `name==version`, like `pip freeze` but without local paths or URLs.

    Returns:
        list[str]: Sorted package pins.
    """
    pins = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pins.add(f"{name}=={dist.version}")
    return sorted(pins, key=str.lower)


def gpu_names() -> list[str]:
    """List the GPUs visible on this machine.

    Returns:
        list[str]: One name per GPU, empty when there is none.
    """
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return [line.strip() for line in out.splitlines() if line.strip()] if out else []


def collect_run_info(seed: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect the facts needed to reproduce a run.

    Args:
        seed (int | None): Random seed used by the run, if any.
        extra (dict[str, Any] | None): Run-specific versions, such as dataset or model revisions.

    Returns:
        dict[str, Any]: The run record.
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
    """Collect the run record and save it as `run_info.json` in the experiment folder.

    Prints a warning when the code has uncommitted changes, because the commit hash then doesn't describe it.

    Args:
        out_dir (Path): The experiment's results folder.
        seed (int | None): Random seed used by the run, if any.
        extra (dict[str, Any] | None): Run-specific versions, such as dataset or model revisions.

    Returns:
        dict[str, Any]: The run record that was written.
    """
    info = collect_run_info(seed=seed, extra=extra)
    if info["git_dirty"]:
        print("WARNING: uncommitted code changes; git_commit alone does not describe this run.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
