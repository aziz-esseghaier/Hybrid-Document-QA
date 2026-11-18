"""Save a text collection to a private Kaggle Dataset, or restore it from one.

Usage:
    python scripts/qdrant_store.py push --dataset v2_esg_reports [--corpus NAME]
    python scripts/qdrant_store.py pull --dataset v2_esg_reports [--corpus NAME] [--force]

push uploads the collection snapshot, manifest.json, the index run files, pages.jsonl and the query embeddings to
<KAGGLE_USERNAME>/qdrant-text-<corpus> (created private; a new version if it exists). pull restores the collection
into the running Qdrant and copies the other inputs of eval_text.py back under <DATA_ROOT>/derived/, so evaluation
needs no chunking or embedding. Qdrant must be the version that made the snapshot.
Needs KAGGLE_USERNAME and KAGGLE_API_TOKEN (or a legacy KAGGLE_KEY).
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

from doc_qa.core.config import Settings, get_settings
from doc_qa.retrieval.qdrant import restore_snapshot, save_snapshot, text_collection

RUN_FILES = ("run_info.json", "config.yaml", "summary.json")
MAX_SLUG_LENGTH = 50  # Kaggle's limit


def dataset_handle(corpus: str) -> str:
    """Return the Kaggle Dataset handle for a corpus's collection.

    Args:
        corpus (str): Corpus name.

    Returns:
        str: `<username>/qdrant-text-<corpus>`.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise SystemExit("KAGGLE_USERNAME is not set (attach the Kaggle secrets and run kaggle_setup.py).")
    slug = f"qdrant-text-{corpus.replace('_', '-')}"
    if len(slug) > MAX_SLUG_LENGTH:
        raise SystemExit(f"Kaggle Dataset name too long: {slug}")
    return f"{username}/{slug}"


def local_paths(settings: Settings, corpus: str, config: dict[str, Any]) -> tuple[Path, Path]:
    """Locate the evaluation inputs stored next to a snapshot.

    Args:
        settings (Settings): Paths.
        corpus (str): Corpus name.
        config (dict[str, Any]): Text config.

    Returns:
        tuple[Path, Path]: pages.jsonl, and the folder of the query embeddings.
    """
    derived = settings.data_root / "derived"
    model = config["embedding_model"].split("/")[-1]
    return derived / "chunks" / corpus / "pages.jsonl", derived / "embeddings" / corpus / model


def build_manifest(
    corpus: str, collection: str, run_dir: Path, points: int, qdrant_version: str, snapshot: Path, sha256: str
) -> dict[str, Any]:
    """Describe a snapshot, checking that it matches the index run.

    Args:
        corpus (str): Corpus name.
        collection (str): Collection name.
        run_dir (Path): Folder with the run files of index_text.py.
        points (int): Points in the collection.
        qdrant_version (str): Version of the running Qdrant.
        snapshot (Path): Snapshot file.
        sha256 (str): SHA-256 of the snapshot.

    Returns:
        dict[str, Any]: Manifest.
    """
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["points"] != points or run_info["collection"] != collection:
        raise SystemExit(f"{collection} ({points} points) doesn't match the index run in {run_dir}.")
    if run_info["qdrant_version"] != qdrant_version:
        raise SystemExit(f"Indexed with Qdrant {run_info['qdrant_version']}, running {qdrant_version}.")
    return {
        "content": "qdrant_collection",
        "collection": collection,
        "corpus": corpus,
        "dataset": run_info["dataset"],
        "points": points,
        "dim": summary["dim"],
        "bm25_language": summary["bm25_language"],
        "qdrant_version": qdrant_version,
        "model": run_info["model"],
        "config": yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")),
        "git_commit": run_info["git_commit"],
        "git_dirty": run_info["git_dirty"],
        "snapshot": snapshot.name,
        "snapshot_bytes": snapshot.stat().st_size,
        "sha256": sha256,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def push(corpus: str, settings: Settings) -> None:
    """Upload a collection and its evaluation inputs as a new Kaggle Dataset version.

    Args:
        corpus (str): Corpus name.
        settings (Settings): Paths and Qdrant URL.
    """
    import kagglehub
    from qdrant_client import QdrantClient

    handle = dataset_handle(corpus)
    name = text_collection(corpus)
    client = QdrantClient(url=settings.qdrant_url, timeout=600)
    if not client.collection_exists(name):
        raise SystemExit(f"No collection {name} (run index_text.py first).")
    run_dir = settings.results_root / "index" / corpus
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    pages_file, embedding_dir = local_paths(settings, corpus, config)
    queries = sorted(embedding_dir.glob("queries_w*.npz"))
    if not pages_file.exists() or not queries:
        raise SystemExit(f"Missing pages.jsonl or query embeddings for {corpus}.")

    with tempfile.TemporaryDirectory(dir=settings.data_root) as tmp:  # the system temp folder can be small
        staging = Path(tmp)
        snapshot = staging / f"{name}.snapshot"
        sha256 = save_snapshot(client, settings.qdrant_url, name, snapshot)
        points = client.count(name, exact=True).count
        manifest = build_manifest(corpus, name, run_dir, points, client.info().version, snapshot, sha256)
        for path in [pages_file, *queries, *(run_dir / file for file in RUN_FILES)]:
            shutil.copy2(path, staging / path.name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        notes = f"{points} points, Qdrant {manifest['qdrant_version']}, git {str(manifest['git_commit'])[:7]}"
        kagglehub.dataset_upload(handle, str(staging), version_notes=notes)
    print(f"Uploaded {handle}: {notes}, {manifest['snapshot_bytes'] / 1e6:.0f} MB snapshot")


def pull(corpus: str, settings: Settings, force: bool) -> None:
    """Restore a collection and its evaluation inputs from the latest Kaggle Dataset version.

    Args:
        corpus (str): Corpus name.
        settings (Settings): Paths and Qdrant URL.
        force (bool): Replace an existing collection and existing files.
    """
    import kagglehub
    from qdrant_client import QdrantClient

    src = Path(kagglehub.dataset_download(dataset_handle(corpus)))
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["collection"]
    client = QdrantClient(url=settings.qdrant_url, timeout=600)
    version = client.info().version
    if manifest["qdrant_version"] != version:
        raise SystemExit(f"Snapshot made with Qdrant {manifest['qdrant_version']}, running {version}.")

    pages_file, embedding_dir = local_paths(settings, corpus, manifest["config"])
    old_queries = sorted(embedding_dir.glob("queries_w*.npz"))
    collection_exists = client.collection_exists(name)
    if (collection_exists or pages_file.exists() or old_queries) and not force:
        raise SystemExit(f"{corpus} already exists here. Use --force to replace it.")
    if collection_exists:
        client.delete_collection(name)
    restore_snapshot(settings.qdrant_url, name, src / manifest["snapshot"], manifest["sha256"])
    points = client.count(name, exact=True).count
    if points != manifest["points"]:
        raise SystemExit(f"{name} has {points} points after restoring, expected {manifest['points']}.")

    pages_file.parent.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for path in old_queries:
        path.unlink()
    shutil.copy2(src / pages_file.name, pages_file)
    for path in sorted(src.glob("queries_w*.npz")):
        shutil.copy2(path, embedding_dir / path.name)
    print(f"Restored {name}: {points} points (git {str(manifest['git_commit'])[:7]})")


def main() -> None:
    """Parse arguments and push or pull."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", help="Chunk folder name (default: the dataset key).")
    parser.add_argument("--force", action="store_true", help="pull: replace an existing collection and files.")
    args = parser.parse_args()

    corpus = args.corpus or args.dataset
    if args.action == "push":
        push(corpus, get_settings())
    else:
        pull(corpus, get_settings(), args.force)


if __name__ == "__main__":
    main()
