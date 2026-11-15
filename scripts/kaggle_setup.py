"""Prepare a Kaggle session: paths and secrets as environment variables.

Run it in the notebook with `%run scripts/kaggle_setup.py`, not `!python`: %run executes inside the
notebook's own process, so the variables it sets reach every later `!python ...` command.

This is the only Kaggle-specific code; the package reads everything from environment variables.
It must not import doc_qa: the package is installed in the same cell, before the kernel can see it.
"""

from __future__ import annotations

import os
from pathlib import Path

# Large disk, wiped at the end of the session.
KAGGLE_TEMP = Path("/kaggle/temp")
# Kept as notebook output when you save a version (about 20 GB max): small results only.
KAGGLE_WORKING = Path("/kaggle/working")
SECRET_NAMES = ("HF_TOKEN",)


def set_paths() -> dict[str, str]:
    """Point data, results and the Hugging Face cache to the right Kaggle disks.

    Returns:
        dict[str, str]: The variables set, by name.
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
    """Copy Kaggle Secrets into environment variables. Missing secrets are skipped.

    Args:
        names (tuple[str, ...]): Secret names, as added under Add-ons > Secrets.

    Returns:
        list[str]: Names of the secrets found (never their values).
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
        except Exception:  # secret not attached to this notebook
            continue
    return found


def main() -> None:
    """Set up the session and print what was configured."""
    if not Path("/kaggle").exists():
        raise SystemExit("Not a Kaggle session. Locally, settings come from the .env file.")
    for name, value in set_paths().items():
        print(f"{name}={value}")
    found = load_secrets()
    print(f"Secrets loaded: {', '.join(found) if found else 'none'}")


if __name__ == "__main__":
    main()
