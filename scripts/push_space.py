"""Assemble the demo Space and upload it to Hugging Face.

Usage:
    python scripts/push_space.py --space <user>/doc-qa-demo [--public]

Puts demo/space/ (app, requirements, Space card), configs/demo.yaml and the doc_qa package in one folder and uploads it
as a Gradio Space (created private unless --public). Choose the ZeroGPU hardware once in the Space settings. Needs a
Hugging Face token with write access (`hf auth login`, or HF_TOKEN).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_space(folder: Path) -> list[str]:
    """Copy the Space files into a folder.

    Args:
        folder (Path): Empty output folder.

    Returns:
        list[str]: Files written, relative to the folder.
    """
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(ROOT / "demo" / "space" / name, folder / name)
    shutil.copy2(ROOT / "configs" / "demo.yaml", folder / "demo.yaml")
    shutil.copytree(ROOT / "src" / "doc_qa", folder / "doc_qa", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


def git_commit() -> str:
    """Return the short commit of the repo, marked when there are uncommitted changes.

    Returns:
        str: E.g. "0e3c6c2" or "0e3c6c2-dirty".
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def main() -> None:
    """Build the Space folder and upload it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="<user>/<name> of the Space.")
    parser.add_argument("--public", action="store_true", help="Create the Space public (default: private).")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = build_space(Path(tmp))
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Demo from git {git_commit()}",
            delete_patterns=["doc_qa/**"],  # drop package files that no longer exist
        )
    print(f"Uploaded {len(files)} files to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
