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
VISUAL_COLLECTION = "visual_v2_x"
VISUAL_CONFIG = {"model": "org/visual", "pool_factor": 2}


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


def write_run(run_dir: Path, config: dict, summary: dict, collection: str) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "summary.json").write_text(json.dumps({"points": 3, "dim": 4, **summary}))
    (run_dir / "run_info.json").write_text(json.dumps({
        "git_commit": "abc1234def", "git_dirty": False, "dataset": "v2_x", "collection": collection,
        "qdrant_version": "1.20.1", "model": {"repo": "org/model", "revision": "r1"},
    }))


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    settings = SimpleNamespace(data_root=tmp_path / "data", results_root=tmp_path / "results", qdrant_url="http://q")
    settings.data_root.mkdir()
    store = tmp_path / "kaggle"
    kagglehub = ModuleType("kagglehub")
    kagglehub.dataset_upload = lambda handle, folder, version_notes: shutil.copytree(folder, store)
    kagglehub.dataset_download = lambda handle: str(store)
    qdrant_client = ModuleType("qdrant_client")
    qdrant_client.QdrantClient = FakeClient
    monkeypatch.setitem(sys.modules, "kagglehub", kagglehub)
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    monkeypatch.setattr(FakeClient, "collections", {COLLECTION: 3, VISUAL_COLLECTION: 3})
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
    return SimpleNamespace(settings=settings, store=store, restored=restored)


@pytest.fixture
def env(fakes):
    config = {"embedding_model": "org/model"}
    write_run(fakes.settings.results_root / "index" / CORPUS, config, {"bm25_language": "english"}, COLLECTION)
    (pages, _), (embeddings, _) = qdrant_store.stored_inputs(fakes.settings, "text", CORPUS, config)
    pages.mkdir(parents=True)
    embeddings.mkdir(parents=True)
    (pages / "pages.jsonl").write_text('{"page_id": "p", "text": "t"}\n')
    (embeddings / "queries_w0.npz").write_bytes(b"queries")
    (embeddings / "chunks_w0.npz").write_bytes(b"chunks are not stored")
    fakes.pages, fakes.embeddings = pages / "pages.jsonl", embeddings
    return fakes


@pytest.fixture
def visual_env(fakes):
    summary = {"pool_factor": 2, "vectors": 900}
    write_run(fakes.settings.results_root / "visual_index" / CORPUS, VISUAL_CONFIG, summary, VISUAL_COLLECTION)
    [(queries, _)] = qdrant_store.stored_inputs(fakes.settings, "visual", CORPUS, VISUAL_CONFIG)
    for folder in ("queries", "pages", "pooled_2"):
        (queries.parent / folder).mkdir(parents=True)
        (queries.parent / folder / "w0_00000.npz").write_bytes(folder.encode())
    fakes.queries = queries
    return fakes


def test_handle(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "me")
    assert qdrant_store.dataset_handle("text", "v2_esg_reports-bridge-ocr") == "me/qdrant-text-v2-esg-reports-bridge-ocr"
    assert qdrant_store.dataset_handle("visual", "v2_esg_reports") == "me/qdrant-visual-v2-esg-reports"
    assert qdrant_store.dataset_handle("visual_tomoro", "v2_esg_reports_human") == (
        "me/qdrant-visual-tomoro-v2-esg-reports-human"
    )
    with pytest.raises(SystemExit, match="too long"):
        qdrant_store.dataset_handle("text", "x" * 40)


def test_locate_keeps_each_visual_retriever_apart():
    assert qdrant_store.locate("text", CORPUS) == (COLLECTION, "index", "text")
    assert qdrant_store.locate("visual", CORPUS) == (VISUAL_COLLECTION, "visual_index", "visual")
    assert qdrant_store.locate("visual", CORPUS, "visual_tomoro") == ("visual_tomoro_v2_x", "visual_tomoro_index",
                                                                       "visual_tomoro")


def test_push_then_pull_in_a_new_session(env):
    qdrant_store.push("text", CORPUS, env.settings)
    manifest = json.loads((env.store / "manifest.json").read_text())
    assert (manifest["points"], manifest["sha256"], manifest["model"]["revision"]) == (3, "sha", "r1")
    assert (manifest["branch"], manifest["bm25_language"]) == ("text", "english")
    assert manifest["config"] == {"embedding_model": "org/model"}
    assert not (env.store / "chunks_w0.npz").exists()

    shutil.rmtree(env.settings.data_root / "derived")
    FakeClient.collections.clear()
    qdrant_store.pull("text", CORPUS, env.settings, force=False)
    assert env.restored == [COLLECTION]
    assert env.pages.read_text() == '{"page_id": "p", "text": "t"}\n'
    assert (env.embeddings / "queries_w0.npz").read_bytes() == b"queries"


def test_pull_needs_force_to_replace(env):
    qdrant_store.push("text", CORPUS, env.settings)
    with pytest.raises(SystemExit, match="--force"):
        qdrant_store.pull("text", CORPUS, env.settings, force=False)
    assert env.restored == []
    (env.embeddings / "queries_w9.npz").write_bytes(b"stale")
    qdrant_store.pull("text", CORPUS, env.settings, force=True)
    assert env.restored == [COLLECTION]
    assert not (env.embeddings / "queries_w9.npz").exists()


def test_pull_rejects_another_qdrant_version(env, monkeypatch):
    qdrant_store.push("text", CORPUS, env.settings)
    monkeypatch.setattr(FakeClient, "version", "1.21.0")
    with pytest.raises(SystemExit, match="1.21.0"):
        qdrant_store.pull("text", CORPUS, env.settings, force=True)
    assert env.restored == []


def test_push_rejects_a_collection_that_differs_from_the_run(env):
    FakeClient.collections[COLLECTION] = 2
    with pytest.raises(SystemExit, match="doesn't match"):
        qdrant_store.push("text", CORPUS, env.settings)
    assert not env.store.exists()


def test_visual_push_with_clean_then_pull(visual_env):
    qdrant_store.push("visual", CORPUS, visual_env.settings, clean=True)
    manifest = json.loads((visual_env.store / "manifest.json").read_text())
    assert (manifest["branch"], manifest["collection"], manifest["pool_factor"]) == ("visual", VISUAL_COLLECTION, 2)
    assert (visual_env.store / "w0_00000.npz").read_bytes() == b"queries"  # page vectors are in the snapshot
    assert VISUAL_COLLECTION not in FakeClient.collections
    assert not visual_env.queries.parent.exists()

    qdrant_store.pull("visual", CORPUS, visual_env.settings, force=False)
    assert visual_env.restored == [VISUAL_COLLECTION]
    assert (visual_env.queries / "w0_00000.npz").read_bytes() == b"queries"


def test_visual_push_under_another_config_name(fakes):
    collection = "visual_tomoro_v2_x"
    FakeClient.collections[collection] = 3
    config = {"model": "org/other", "pool_factor": 1}
    write_run(fakes.settings.results_root / "visual_tomoro_index" / CORPUS, config, {"pool_factor": 1, "vectors": 9},
              collection)
    [(queries, _)] = qdrant_store.stored_inputs(fakes.settings, "visual", CORPUS, config)
    queries.mkdir(parents=True)
    (queries / "w0_00000.npz").write_bytes(b"queries")
    qdrant_store.push("visual", CORPUS, fakes.settings, visual_name="visual_tomoro")
    manifest = json.loads((fakes.store / "manifest.json").read_text())
    assert (manifest["collection"], manifest["pool_factor"]) == (collection, 1)
    assert FakeClient.collections[VISUAL_COLLECTION] == 3  # the other retriever's collection is untouched


def test_push_keeps_vectors_without_clean(visual_env):
    qdrant_store.push("visual", CORPUS, visual_env.settings)
    assert FakeClient.collections[VISUAL_COLLECTION] == 3
    assert (visual_env.queries.parent / "pages" / "w0_00000.npz").exists()
