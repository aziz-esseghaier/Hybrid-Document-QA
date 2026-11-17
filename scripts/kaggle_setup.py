"""Kaggle session setup: data/results paths and secrets as environment variables; optionally Qdrant.

Run with `%run scripts/kaggle_setup.py [--qdrant]` (not `!python`) so the variables reach later `!python` calls.
Must not import doc_qa: the package is installed in the same cell.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

KAGGLE_TEMP = Path("/kaggle/temp")  # large, wiped at session end
KAGGLE_WORKING = Path("/kaggle/working")  # kept as notebook output (~20 GB)
SECRET_NAMES = ("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "KAGGLE_KEY")  # KAGGLE_KEY: legacy key
QDRANT_DIR = KAGGLE_TEMP / "qdrant"
COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
# Static (musl) build: the glibc build can need a newer glibc than Kaggle's image has.
QDRANT_RELEASE = "https://github.com/qdrant/qdrant/releases/download/v{version}/qdrant-x86_64-unknown-linux-musl.tar.gz"


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


def qdrant_version() -> str:
    """Read the pinned Qdrant version from docker-compose.yml.

    Returns:
        str: Version, e.g. "1.20.1".
    """
    match = re.search(r"qdrant/qdrant:v([\d.]+)", COMPOSE_FILE.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"No qdrant/qdrant:vX.Y.Z image in {COMPOSE_FILE}")
    return match.group(1)


def _ready(url: str) -> bool:
    """Check whether Qdrant answers at `url`.

    Args:
        url (str): Base URL.

    Returns:
        bool: True if ready.
    """
    try:
        with urllib.request.urlopen(f"{url}/readyz", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def start_qdrant(port: int = 6333, timeout: float = 60.0) -> str:
    """Download the pinned Qdrant binary (once per session) and start it in the background.

    Args:
        port (int): HTTP port.
        timeout (float): Seconds to wait for it to be ready.

    Returns:
        str: Qdrant URL (also set as QDRANT_URL).
    """
    url = f"http://localhost:{port}"
    os.environ["QDRANT_URL"] = url
    if _ready(url):
        return url
    version = qdrant_version()
    binary_dir = QDRANT_DIR / f"v{version}-musl"
    binary = binary_dir / "qdrant"
    if not binary.exists():
        binary_dir.mkdir(parents=True, exist_ok=True)
        archive = binary_dir / "qdrant.tar.gz"
        urllib.request.urlretrieve(QDRANT_RELEASE.format(version=version), archive)
        with tarfile.open(archive) as tar:
            tar.extractall(binary_dir, filter="data")
        binary.chmod(0o755)
    env = {
        **os.environ,
        "QDRANT__STORAGE__STORAGE_PATH": str(QDRANT_DIR / "storage"),
        "QDRANT__STORAGE__SNAPSHOTS_PATH": str(QDRANT_DIR / "snapshots"),
        "QDRANT__SERVICE__HTTP_PORT": str(port),
        "QDRANT__TELEMETRY_DISABLED": "true",
    }
    log = (QDRANT_DIR / "qdrant.log").open("ab")
    subprocess.Popen([str(binary)], cwd=QDRANT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _ready(url):
            return url
        time.sleep(1)
    raise SystemExit(f"Qdrant did not start; see {QDRANT_DIR / 'qdrant.log'}")


def main() -> None:
    """Set up the session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant", action="store_true", help="Start the Qdrant server.")
    args = parser.parse_args()
    if not Path("/kaggle").exists():
        raise SystemExit("Not a Kaggle session. Locally, settings come from the .env file.")
    for name, value in set_paths().items():
        print(f"{name}={value}")
    found = load_secrets()
    print(f"Secrets loaded: {', '.join(found) if found else 'none'}")
    if args.qdrant:
        print(f"Qdrant {qdrant_version()} at {start_qdrant()}")


if __name__ == "__main__":
    main()
