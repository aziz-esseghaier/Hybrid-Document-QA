"""Tests for saving and restoring Qdrant collections."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qdrant_store.py"
spec = importlib.util.spec_from_file_location("qdrant_store", SCRIPT)
qdrant_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qdrant_store)

CORPUS = "v2_x"
COLLECTION = "text_v2_x"


class FakeClient:
    """In-memory stand-in for QdrantClient, shared through `collections`."""

    collections: dict[str, int] = {}
    version = "1.20.1"

    def __init__(self, url: str, timeout: int) -> None:
        pass

    def collection_exists(self, name):
        return name in self.collections

    def delete_collection(self, name):
        del self.collections[name]

    def count(self, name, exact):
        return SimpleNamespace(count=self.collections[name])

    def info(self):
        return SimpleNamespace(version=self.version)


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = SimpleNamespace(data_root=tmp_path / "data", results_root=tmp_path / "results", qdrant_url="http://q")
    run = settings.results_root / "index" / CORPUS
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"embedding_model": "org/model"}))
    (run / "summary.json").write_text(json.dumps({"points": 3, "dim": 4, "bm25_language": "english"}))
    (run / "run_info.json").write_text(json.dumps({
        "git_commit": "abc1234def", "git_dirty": False, "dataset": "v2_x", "collection": COLLECTION,
        "qdrant_version": "1.20.1", "model": {"repo": "org/model", "revision": "r1"},
    }))
    pages, embeddings = qdrant_store.local_paths(settings, CORPUS, {"embedding_model": "org/model"})
    pages.parent.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    pages.write_text('{"page_id": "p", "text": "t"}\n')
    (embeddings / "queries_w0.npz").write_bytes(b"queries")
    (embeddings / "chunks_w0.npz").write_bytes(b"chunks are not stored")

    store = tmp_path / "kaggle"
    kagglehub = ModuleType("kagglehub")
    kagglehub.dataset_upload = lambda handle, folder, version_notes: shutil.copytree(folder, store)
    kagglehub.dataset_download = lambda handle: str(store)
    qdrant_client = ModuleType("qdrant_client")
    qdrant_client.QdrantClient = FakeClient
    monkeypatch.setitem(sys.modules, "kagglehub", kagglehub)
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setattr(FakeClient, "collections", {COLLECTION: 3})
    monkeypatch.setattr(FakeClient, "version", "1.20.1")

    def fake_save(client, url, name, path):
        path.write_bytes(b"snapshot")
        return "sha"

    restored = []

    def fake_restore(url, name, path, sha256):
        assert path.read_bytes() == b"snapshot" and sha256 == "sha"
        restored.append(name)
        FakeClient.collections[name] = 3

    monkeypatch.setattr(qdrant_store, "save_snapshot", fake_save)
    monkeypatch.setattr(qdrant_store, "restore_snapshot", fake_restore)
    return SimpleNamespace(settings=settings, pages=pages, embeddings=embeddings, store=store, restored=restored)


def test_handle(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    assert qdrant_store.dataset_handle("v2_esg_reports-bridge-ocr") == "me/qdrant-text-v2-esg-reports-bridge-ocr"
    with pytest.raises(SystemExit, match="too long"):
        qdrant_store.dataset_handle("x" * 40)


def test_push_then_pull_in_a_new_session(env):
    qdrant_store.push(CORPUS, env.settings)
    manifest = json.loads((env.store / "manifest.json").read_text())
    assert (manifest["points"], manifest["sha256"], manifest["model"]["revision"]) == (3, "sha", "r1")
    assert manifest["config"] == {"embedding_model": "org/model"}
    assert not (env.store / "chunks_w0.npz").exists()

    shutil.rmtree(env.settings.data_root / "derived")
    FakeClient.collections.clear()
    qdrant_store.pull(CORPUS, env.settings, force=False)
    assert env.restored == [COLLECTION]
    assert env.pages.read_text() == '{"page_id": "p", "text": "t"}\n'
    assert (env.embeddings / "queries_w0.npz").read_bytes() == b"queries"


def test_pull_needs_force_to_replace(env):
    qdrant_store.push(CORPUS, env.settings)
    with pytest.raises(SystemExit, match="--force"):
        qdrant_store.pull(CORPUS, env.settings, force=False)
    assert env.restored == []
    (env.embeddings / "queries_w9.npz").write_bytes(b"stale")
    qdrant_store.pull(CORPUS, env.settings, force=True)
    assert env.restored == [COLLECTION]
    assert not (env.embeddings / "queries_w9.npz").exists()


def test_pull_rejects_another_qdrant_version(env, monkeypatch):
    qdrant_store.push(CORPUS, env.settings)
    monkeypatch.setattr(FakeClient, "version", "1.21.0")
    with pytest.raises(SystemExit, match="1.21.0"):
        qdrant_store.pull(CORPUS, env.settings, force=True)
    assert env.restored == []


def test_push_rejects_a_collection_that_differs_from_the_run(env):
    FakeClient.collections[COLLECTION] = 2
    with pytest.raises(SystemExit, match="doesn't match"):
        qdrant_store.push(CORPUS, env.settings)
    assert not env.store.exists()
