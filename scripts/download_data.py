"""Download the pinned ViDoRe datasets into <DATA_ROOT>/raw/<dataset key>/.

Only the test split's parquet files are downloaded. Files already present are skipped, so the script
can be rerun after an interruption.

Usage:
    python scripts/download_data.py                 # all datasets in configs/datasets.yaml
    python scripts/download_data.py --only v3_hr    # some datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from doc_qa.core.config import get_settings, load_yaml


def folder_size_gb(path: Path) -> float:
    """Compute the total size of the files in a folder.

    Args:
        path (Path): The folder.

    Returns:
        float: Size in GB.
    """
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    """Download the datasets listed in configs/datasets.yaml."""
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
