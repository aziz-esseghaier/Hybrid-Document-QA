"""Save OCR output to a private Kaggle Dataset, or restore it from one.

Usage:
    python scripts/ocr_store.py push --dataset v2_esg_reports [--shards DIR] [--run DIR]
    python scripts/ocr_store.py pull --dataset v2_esg_reports

push uploads shard*.jsonl, manifest.json, run_info.json, config.yaml and summary.json to
<KAGGLE_USERNAME>/ocr-<dataset> (created private; a new version if it exists).
pull copies the shards to <DATA_ROOT>/derived/ocr/<dataset>/.
Needs KAGGLE_USERNAME and KAGGLE_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings
from doc_qa.core.parallel import read_jsonl

RUN_FILES = ("run_info.json", "config.yaml", "summary.json")


def dataset_handle(dataset: str) -> str:
    """Return the Kaggle Dataset handle for a dataset's OCR output.

    Args:
        dataset (str): Dataset key.

    Returns:
        str: `<username>/ocr-<dataset>`.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise SystemExit("KAGGLE_USERNAME is not set (attach the Kaggle secrets and run kaggle_setup.py).")
    return f"{username}/ocr-{dataset.replace('_', '-')}"


def build_manifest(dataset: str, shards: list[Path], run_dir: Path, allow_partial: bool = False) -> dict[str, Any]:
    """Describe an OCR run from its shards and run files.

    Args:
        dataset (str): Dataset key.
        shards (list[Path]): shard*.jsonl files.
        run_dir (Path): Folder with run_info.json, config.yaml and summary.json.
        allow_partial (bool): Accept a run that did not cover every page.

    Returns:
        dict[str, Any]: Manifest.
    """
    pages = {r["page_id"] for r in read_jsonl(shards)}
    total = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["pages_total"]
    if len(pages) < total and not allow_partial:
        raise SystemExit(f"Only {len(pages)}/{total} pages OCR'd. Use --allow-partial to save anyway.")
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return {
        "content": "ocr",
        "dataset": dataset,
        "source": run_info.get("dataset"),
        "pages": len(pages),
        "pages_total": total,
        "pipeline_version": config.get("pipeline_version"),
        "config": config,
        "git_commit": run_info.get("git_commit"),
        "git_dirty": run_info.get("git_dirty"),
        "packages": [p for p in run_info.get("packages", []) if p.lower().startswith("paddle")],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def push(dataset: str, shard_dir: Path, run_dir: Path, allow_partial: bool) -> None:
    """Upload OCR shards and run files as a new Kaggle Dataset version.

    Args:
        dataset (str): Dataset key.
        shard_dir (Path): Folder with shard*.jsonl.
        run_dir (Path): Folder with the run files.
        allow_partial (bool): Accept a run that did not cover every page.
    """
    import kagglehub

    shards = sorted(shard_dir.glob("shard*.jsonl"))
    if not shards:
        raise SystemExit(f"No shard*.jsonl in {shard_dir}")
    handle = dataset_handle(dataset)
    manifest = build_manifest(dataset, shards, run_dir, allow_partial)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for path in [*shards, *(run_dir / name for name in RUN_FILES)]:
            shutil.copy2(path, staging / path.name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        notes = f"{manifest['pages']}/{manifest['pages_total']} pages, git {str(manifest['git_commit'])[:7]}"
        kagglehub.dataset_upload(handle, str(staging), version_notes=notes)
    print(f"Uploaded {handle}: {notes}")


def pull(dataset: str, out_dir: Path, force: bool) -> None:
    """Download the latest OCR Dataset version and copy its shards to `out_dir`.

    Args:
        dataset (str): Dataset key.
        out_dir (Path): Shard folder.
        force (bool): Replace existing shards.
    """
    import kagglehub

    if any(out_dir.glob("shard*.jsonl")) and not force:
        raise SystemExit(f"{out_dir} already has shards. Use --force to replace them.")
    src = Path(kagglehub.dataset_download(dataset_handle(dataset)))
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("shard*.jsonl"):
        path.unlink()
    for path in sorted(src.glob("shard*.jsonl")):
        shutil.copy2(path, out_dir / path.name)
    print(f"Restored {manifest['pages']} pages (git {str(manifest['git_commit'])[:7]}) to {out_dir}")


def main() -> None:
    """Parse arguments and push or pull."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shards", type=Path, help="Shard folder. Default: <DATA_ROOT>/derived/ocr/<dataset>.")
    parser.add_argument("--run", type=Path, help="Run files folder. Default: <RESULTS_ROOT>/ocr/<dataset>.")
    parser.add_argument("--allow-partial", action="store_true", help="push: save a run that missed pages.")
    parser.add_argument("--force", action="store_true", help="pull: replace existing shards.")
    args = parser.parse_args()

    settings = get_settings()
    shard_dir = args.shards or settings.data_root / "derived" / "ocr" / args.dataset
    if args.action == "push":
        run_dir = args.run or settings.results_root / "ocr" / args.dataset
        push(args.dataset, shard_dir, run_dir, args.allow_partial)
    else:
        pull(args.dataset, shard_dir, args.force)


if __name__ == "__main__":
    main()
