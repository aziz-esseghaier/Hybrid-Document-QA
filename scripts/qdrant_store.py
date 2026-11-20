"""Save a collection to a private Kaggle Dataset, or restore it from one.

Usage:
    python scripts/qdrant_store.py push --dataset v2_esg_reports [--corpus NAME]
    python scripts/qdrant_store.py push --branch visual [--config configs/visual.yaml] --dataset v2_esg_reports [--clean]
    python scripts/qdrant_store.py pull [--branch visual] [--config ...] --dataset v2_esg_reports [--corpus NAME] [--force]

push uploads the collection snapshot, manifest.json, the index run files and the query embeddings (plus pages.jsonl
for text) to <KAGGLE_USERNAME>/qdrant-<prefix>-<corpus> (created private; a new version if it exists). The prefix is
"text", or the visual config's `name`, so each visual retriever has its own collections and Datasets. pull restores
the collection into the running Qdrant and copies the other files back under <DATA_ROOT>/derived/, so search needs no
embedding. Qdrant must be the version that made the snapshot. Visual --clean deletes the collection and the dataset's
page and query vectors after the upload, to free disk.
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

from doc_qa.core.config import Settings, get_settings, load_yaml
from doc_qa.retrieval.qdrant import restore_snapshot, save_snapshot, text_collection, visual_collection
from doc_qa.retrieval.visual import visual_dir

RUN_FILES = ("run_info.json", "config.yaml", "summary.json")
MAX_SLUG_LENGTH = 50  # Kaggle's limit
SUMMARY_KEYS = {"text": ("dim", "bm25_language"), "visual": ("dim", "pool_factor", "vectors")}  # copied to manifest


def locate(branch: str, corpus: str, name: str = "visual") -> tuple[str, str, str]:
    """Name a corpus's collection, its index run folder and its Kaggle Dataset prefix.

    Args:
        branch (str): "text" or "visual".
        corpus (str): Corpus name.
        name (str): Visual only: config name, one per visual retriever.

    Returns:
        tuple[str, str, str]: (collection, run folder under the results root, Dataset prefix).
    """
    if branch == "text":
        return text_collection(corpus), "index", "text"
    return visual_collection(name, corpus), f"{name}_index", name


def dataset_handle(prefix: str, corpus: str) -> str:
    """Return the Kaggle Dataset handle for a corpus's collection.

    Args:
        prefix (str): "text", or the visual config name.
        corpus (str): Corpus name.

    Returns:
        str: `<username>/qdrant-<prefix>-<corpus>`, with dashes for underscores.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise SystemExit("KAGGLE_USERNAME is not set (attach the Kaggle secrets and run kaggle_setup.py).")
    slug = f"qdrant-{prefix}-{corpus}".replace("_", "-")
    if len(slug) > MAX_SLUG_LENGTH:
        raise SystemExit(f"Kaggle Dataset name too long: {slug}")
    return f"{username}/{slug}"


def stored_inputs(settings: Settings, branch: str, corpus: str, config: dict[str, Any]) -> list[tuple[Path, str]]:
    """List the files kept next to a snapshot, as local folders and file patterns.

    Args:
        settings (Settings): Paths.
        branch (str): "text" or "visual".
        corpus (str): Corpus name.
        config (dict[str, Any]): Config of the index run.

    Returns:
        list[tuple[Path, str]]: (folder, glob pattern) pairs; patterns don't overlap.
    """
    derived = settings.data_root / "derived"
    if branch == "visual":
        return [(visual_dir(settings.data_root, corpus, config["model"]) / "queries", "w*.npz")]
    model = config["embedding_model"].split("/")[-1]
    return [(derived / "chunks" / corpus, "pages.jsonl"), (derived / "embeddings" / corpus / model, "queries_w*.npz")]


def build_manifest(
    branch: str, corpus: str, collection: str, run_dir: Path, points: int, qdrant_version: str, snapshot: Path,
    sha256: str,
) -> dict[str, Any]:
    """Describe a snapshot, checking that it matches the index run.

    Args:
        branch (str): "text" or "visual".
        corpus (str): Corpus name.
        collection (str): Collection name.
        run_dir (Path): Folder with the run files of the index script.
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
        "branch": branch,
        "collection": collection,
        "corpus": corpus,
        "dataset": run_info["dataset"],
        "points": points,
        **{key: summary[key] for key in SUMMARY_KEYS[branch]},
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


def push(branch: str, corpus: str, settings: Settings, clean: bool = False, visual_name: str = "visual") -> None:
    """Upload a collection and its stored inputs as a new Kaggle Dataset version.

    Args:
        branch (str): "text" or "visual".
        corpus (str): Corpus name.
        settings (Settings): Paths and Qdrant URL.
        clean (bool): Visual only: then delete the collection and the dataset's vectors.
        visual_name (str): Visual only: config name.
    """
    import kagglehub
    from qdrant_client import QdrantClient

    name, run_folder, prefix = locate(branch, corpus, visual_name)
    handle = dataset_handle(prefix, corpus)
    client = QdrantClient(url=settings.qdrant_url, timeout=600)
    if not client.collection_exists(name):
        raise SystemExit(f"No collection {name} (run the index script first).")
    run_dir = settings.results_root / run_folder / corpus
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    inputs = stored_inputs(settings, branch, corpus, config)
    missing = [str(folder / pattern) for folder, pattern in inputs if not any(folder.glob(pattern))]
    if missing:
        raise SystemExit(f"Nothing to store for {corpus} in {missing}.")
    files = [path for folder, pattern in inputs for path in sorted(folder.glob(pattern))]

    with tempfile.TemporaryDirectory(dir=settings.data_root) as tmp:  # the system temp folder can be small
        staging = Path(tmp)
        snapshot = staging / f"{name}.snapshot"
        sha256 = save_snapshot(client, settings.qdrant_url, name, snapshot)
        points = client.count(name, exact=True).count
        manifest = build_manifest(branch, corpus, name, run_dir, points, client.info().version, snapshot, sha256)
        for path in [*files, *(run_dir / file for file in RUN_FILES)]:
            shutil.copy2(path, staging / path.name)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        notes = f"{points} points, Qdrant {manifest['qdrant_version']}, git {str(manifest['git_commit'])[:7]}"
        kagglehub.dataset_upload(handle, str(staging), version_notes=notes)
    print(f"Uploaded {handle}: {notes}, {manifest['snapshot_bytes'] / 1e6:.0f} MB snapshot")

    if clean:
        client.delete_collection(name)
        shutil.rmtree(visual_dir(settings.data_root, corpus, config["model"]))
        print(f"Deleted {name} and the local vectors of {corpus}")


def pull(branch: str, corpus: str, settings: Settings, force: bool, visual_name: str = "visual") -> None:
    """Restore a collection and its stored inputs from the latest Kaggle Dataset version.

    Args:
        branch (str): "text" or "visual".
        corpus (str): Corpus name.
        settings (Settings): Paths and Qdrant URL.
        force (bool): Replace an existing collection and existing files.
        visual_name (str): Visual only: config name.
    """
    import kagglehub
    from qdrant_client import QdrantClient

    _, _, prefix = locate(branch, corpus, visual_name)
    src = Path(kagglehub.dataset_download(dataset_handle(prefix, corpus)))
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["collection"]
    client = QdrantClient(url=settings.qdrant_url, timeout=600)
    version = client.info().version
    if manifest["qdrant_version"] != version:
        raise SystemExit(f"Snapshot made with Qdrant {manifest['qdrant_version']}, running {version}.")

    inputs = stored_inputs(settings, branch, corpus, manifest["config"])
    old_files = [path for folder, pattern in inputs for path in folder.glob(pattern)]
    collection_exists = client.collection_exists(name)
    if (collection_exists or old_files) and not force:
        raise SystemExit(f"{corpus} already exists here. Use --force to replace it.")
    if collection_exists:
        client.delete_collection(name)
    restore_snapshot(settings.qdrant_url, name, src / manifest["snapshot"], manifest["sha256"])
    points = client.count(name, exact=True).count
    if points != manifest["points"]:
        raise SystemExit(f"{name} has {points} points after restoring, expected {manifest['points']}.")

    for path in old_files:
        path.unlink()
    for folder, pattern in inputs:
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(src.glob(pattern)):
            shutil.copy2(path, folder / path.name)
    print(f"Restored {name}: {points} points (git {str(manifest['git_commit'])[:7]})")


def main() -> None:
    """Parse arguments and push or pull."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--branch", choices=list(SUMMARY_KEYS), default="text")
    parser.add_argument("--config", type=Path, help="Visual only: config whose name is used (configs/visual.yaml).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", help="Text only: chunk folder name (default: the dataset key).")
    parser.add_argument("--force", action="store_true", help="pull: replace an existing collection and files.")
    parser.add_argument("--clean", action="store_true", help="Visual push: then delete the collection and vectors.")
    args = parser.parse_args()
    if args.branch == "visual" and args.corpus:
        parser.error("--corpus is for the text branch.")
    if args.branch == "text" and args.config:
        parser.error("--config is for the visual branch.")
    if args.clean and (args.branch != "visual" or args.action != "push"):
        parser.error("--clean is for pushing the visual branch.")

    corpus = args.corpus or args.dataset
    visual_name = load_yaml(args.config or Path("configs/visual.yaml"))["name"] if args.branch == "visual" else ""
    if args.action == "push":
        push(args.branch, corpus, get_settings(), args.clean, visual_name)
    else:
        pull(args.branch, corpus, get_settings(), args.force, visual_name)


if __name__ == "__main__":
    main()
