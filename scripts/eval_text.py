"""Evaluate text retrieval on a dataset: BM25, dense, and their RRF merge.

Usage:
    python scripts/eval_text.py --dataset v2_esg_reports

Searches collection text_<dataset> (exact search), scores each page by its best chunk, and reports
NDCG@10 and Recall@100 per query language. Writes <RESULTS_ROOT>/text_eval/<dataset>/: metrics.json,
rankings_<method>.jsonl ({query_id, pages: [[page_id, score], ...]}), config.yaml, run_info.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import evaluate
from doc_qa.evaluation.vidore import load_benchmark
from doc_qa.retrieval.bm25 import load_bm25
from doc_qa.retrieval.dense import load_vectors
from doc_qa.retrieval.fusion import page_ranking, rrf
from doc_qa.retrieval.qdrant import DENSE, SPARSE, search, sparse_vector, text_collection

METHODS = ("bm25", "dense", "rrf")


def main() -> None:
    """Rank pages for every query with each method, then score the rankings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/text.yaml"))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    settings = get_settings()
    cfg = load_yaml(args.config)
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][args.dataset]
    bench = load_benchmark(args.dataset, spec, settings.data_root / "raw")
    name = text_collection(args.dataset)

    emb_dir = settings.data_root / "derived" / "embeddings" / args.dataset / cfg["embedding_model"].split("/")[-1]
    query_vectors = load_vectors(sorted(emb_dir.glob("queries_w*.npz")))
    query_ids = sorted(bench.queries)
    if any(q not in query_vectors for q in query_ids):
        raise SystemExit("Some queries have no embedding (run embed_text.py).")
    bm25 = load_bm25(cfg["bm25_languages"][spec["doc_language"]])  # query weights don't depend on k, b, avg_len
    bm25_queries = dict(zip(query_ids, bm25.query_embed([bench.queries[q].text for q in query_ids])))

    client = QdrantClient(url=settings.qdrant_url, timeout=120)
    results_dir = settings.results_root / "text_eval" / args.dataset
    write_run_info(results_dir, extra={
        "dataset": {"key": args.dataset, "repo": spec["repo"], "revision": spec["revision"]},
        "collection": name,
        "qdrant_version": client.info().version,
        "model": {"repo": cfg["embedding_model"], "revision": cfg["embedding_revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    limit, page_limit = cfg["chunk_limit"], cfg["page_limit"]
    rankings: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    for query_id in query_ids:
        sparse = bm25_queries[query_id]
        hits = {
            "bm25": search(client, name, SPARSE, sparse_vector(sparse.indices, sparse.values), limit)
            if len(sparse.indices) else [],  # only stopwords

            "dense": search(client, name, DENSE, query_vectors[query_id].tolist(), limit),
        }
        for method, method_hits in hits.items():
            pages = ((payload["page_id"], score) for payload, score in method_hits)
            rankings[method][query_id] = page_ranking(pages, page_limit)
        merged = [[p for p, _ in rankings[m][query_id]] for m in ("bm25", "dense")]
        rankings["rrf"][query_id] = rrf(merged, cfg["rrf_k"], page_limit)

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        qrels[qrel.query_id][qrel.page_id] = qrel.score
    languages = {q: bench.queries[q].language for q in query_ids}

    metrics = {}
    for method in METHODS:
        ranked_ids = {q: [p for p, _ in pages] for q, pages in rankings[method].items()}
        metrics[method] = evaluate(ranked_ids, qrels, languages)
        with (results_dir / f"rankings_{method}.jsonl").open("w", encoding="utf-8") as f:
            for query_id, pages in rankings[method].items():
                f.write(json.dumps({"query_id": query_id, "pages": [[p, round(s, 6)] for p, s in pages]}) + "\n")
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"{'method':<7} {'lang':<5} {'queries':>7} {'ndcg@10':>8} {'recall@100':>10}")
    for method in METHODS:
        for group, m in metrics[method].items():
            print(f"{method:<7} {group:<5} {m['queries']:>7} {m['ndcg@10']:>8.3f} {m['recall@100']:>10.3f}")


if __name__ == "__main__":
    main()
