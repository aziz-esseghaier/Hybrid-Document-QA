"""Load a dataset's chunks into a Qdrant collection with dense and BM25 vectors.

Usage:
    python scripts/index_text.py --dataset v2_esg_reports [--corpus NAME]

Creates (or replaces) collection text_<corpus>: one point per chunk, payload {chunk_id, page_id}. The corpus
(chunk folder) defaults to the dataset. Needs chunks (build_chunks.py), chunk embeddings (embed_text.py) and a
running Qdrant (QDRANT_URL). Writes <RESULTS_ROOT>/index/<corpus>/: config.yaml, run_info.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.retrieval.bm25 import load_bm25, mean_length
from doc_qa.retrieval.dense import load_vectors
from doc_qa.retrieval.qdrant import create_text_collection, sparse_vector, text_collection, upsert_chunks


def main() -> None:
    """Build the collection and check its point count."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/text.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", help="Chunk folder name (default: the dataset key).")
    args = parser.parse_args()
    corpus = args.corpus or args.dataset

    from qdrant_client import QdrantClient

    settings = get_settings()
    cfg = load_yaml(args.config)
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][args.dataset]
    language = cfg["bm25_languages"][spec["doc_language"]]
    name = text_collection(corpus)

    chunks = read_jsonl([settings.data_root / "derived" / "chunks" / corpus / "chunks.jsonl"])
    emb_dir = settings.data_root / "derived" / "embeddings" / corpus / cfg["embedding_model"].split("/")[-1]
    vectors = load_vectors(sorted(emb_dir.glob("chunks_w*.npz")))
    missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in vectors]
    if not chunks or missing:
        raise SystemExit(f"{len(chunks)} chunks, {len(missing)} without embeddings (run embed_text.py).")
    texts = [c["text"] for c in chunks]
    dense = np.stack([vectors[c["chunk_id"]] for c in chunks])

    avg_len = mean_length(texts, language)
    bm25 = load_bm25(language, cfg["bm25_k"], cfg["bm25_b"], avg_len)
    sparse = [sparse_vector(e.indices, e.values) for e in bm25.embed(texts, batch_size=256)]

    client = QdrantClient(url=settings.qdrant_url, timeout=120)
    results_dir = settings.results_root / "index" / corpus
    write_run_info(results_dir, extra={
        "dataset": args.dataset,
        "corpus": corpus,
        "collection": name,
        "qdrant_version": client.info().version,
        "model": {"repo": cfg["embedding_model"], "revision": cfg["embedding_revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    create_text_collection(client, name, dense.shape[1])
    upsert_chunks(client, name, [{"chunk_id": c["chunk_id"], "page_id": c["page_id"]} for c in chunks], dense, sparse)
    points = client.count(name, exact=True).count
    if points != len(chunks):
        raise SystemExit(f"Collection has {points} points, expected {len(chunks)}.")

    summary = {"collection": name, "points": points, "dim": int(dense.shape[1]),
               "bm25_language": language, "bm25_avg_len": round(avg_len, 1)}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
