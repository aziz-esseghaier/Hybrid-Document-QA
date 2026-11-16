"""Download the pinned ViDoRe test splits into <DATA_ROOT>/raw/<key>/. Resumable.

Usage:
    python scripts/download_data.py [--only KEY ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from doc_qa.core.config import get_settings, load_yaml


def folder_size_gb(path: Path) -> float:
    """Total size of a folder's files.

    Args:
        path (Path): Folder.

    Returns:
        float: Size in GB.
    """
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    """Download the datasets in configs/datasets.yaml."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="Dataset keys to download (default: all).")
    args = parser.parse_args()

    specs = load_yaml(Path("configs/datasets.yaml"))["datasets"]
    unknown = set(args.only or []) - specs.keys()
    if unknown:
        raise SystemExit(f"Unknown dataset keys: {sorted(unknown)}")

    raw_root = get_settings().data_root / "raw"
    for key, spec in specs.items():
        if args.only and key not in args.only:
            continue
        target = raw_root / key
        print(f"{key}: {spec['repo']}@{spec['revision'][:12]} -> {target}")
        snapshot_download(
            repo_id=spec["repo"],
            repo_type="dataset",
            revision=spec["revision"],
            local_dir=target,
            allow_patterns=["*/test-*.parquet"],
        )
        print(f"  {folder_size_gb(target):.2f} GB")


if __name__ == "__main__":
    main()
