"""Kaggle session setup: data/results paths and secrets as environment variables.

Run with `%run scripts/kaggle_setup.py` (not `!python`) so the variables reach later `!python` calls.
Must not import doc_qa: the package is installed in the same cell.
"""

from __future__ import annotations

import os
from pathlib import Path

KAGGLE_TEMP = Path("/kaggle/temp")  # large, wiped at session end
KAGGLE_WORKING = Path("/kaggle/working")  # kept as notebook output (~20 GB)
SECRET_NAMES = ("HF_TOKEN",)


def set_paths() -> dict[str, str]:
    """Set DATA_ROOT, RESULTS_ROOT and HF_HOME (unless already set).

    Returns:
        dict[str, str]: Variables set.
    """
    paths = {
        "DATA_ROOT": KAGGLE_TEMP / "data",
        "RESULTS_ROOT": KAGGLE_WORKING / "results",
        "HF_HOME": KAGGLE_TEMP / "hf_home",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    return {name: os.environ[name] for name in paths}


def load_secrets(names: tuple[str, ...] = SECRET_NAMES) -> list[str]:
    """Copy Kaggle secrets into environment variables. Missing secrets are skipped.

    Args:
        names (tuple[str, ...]): Secret names.

    Returns:
        list[str]: Names found.
    """
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return []
    client, found = UserSecretsClient(), []
    for name in names:
        try:
            os.environ[name] = client.get_secret(name)
            found.append(name)
        except Exception:  # not attached to this notebook
            continue
    return found


def main() -> None:
    """Set up the session."""
    if not Path("/kaggle").exists():
        raise SystemExit("Not a Kaggle session. Locally, settings come from the .env file.")
    for name, value in set_paths().items():
        print(f"{name}={value}")
    found = load_secrets()
    print(f"Secrets loaded: {', '.join(found) if found else 'none'}")


if __name__ == "__main__":
    main()
