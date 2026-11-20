"""Evaluate visual retrieval on a dataset, and what merging page vectors costs.

Usage:
    python scripts/eval_visual.py --dataset v2_esg_reports [--config configs/visual.yaml] [--local-only]

Ranks pages three ways, all exact MaxSim: `full` (every vector, computed here), `pooled` (merged vectors, computed
here; skipped when `pool_factor` is 1) and `qdrant` (the vectors Qdrant holds, searched in collection
visual_<dataset>; skipped with --local-only). Reports NDCG@10 and Recall@100 per query language. Writes
<RESULTS_ROOT>/<name>_eval/<dataset>/ (`name` from the config): metrics.json, summary.json (search time, agreement
between rankings; updated during the search), rankings_<method>.jsonl ({query_id, pages: [[page_id, score], ...]}),
config.yaml, run_info.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import evaluate
from doc_qa.evaluation.vidore import load_benchmark
from doc_qa.retrieval.qdrant import VISUAL, search, visual_collection
from doc_qa.retrieval.visual import iter_multivectors, maxsim_matrix, part_files, pooled_dir, visual_dir

Ranking = list[tuple[str, float]]


def rank_pages(page_ids: Sequence[str], scores: Sequence[float], limit: int) -> Ranking:
    """Keep the best pages; ties go to the smaller page id, so the order is reproducible.

    Args:
        page_ids (Sequence[str]): Pages.
        scores (Sequence[float]): Score per page.
        limit (int): Pages to keep.

    Returns:
        Ranking: (page id, score), best first.
    """
    best = heapq.nsmallest(limit, zip((-float(score) for score in scores), page_ids))
    return [(page_id, -negative) for negative, page_id in best]


def mean_overlap(first: dict[str, Ranking], second: dict[str, Ranking], k: int = 10) -> float:
    """Measure how much two sets of rankings agree on their top pages.

    Args:
        first (dict[str, Ranking]): Ranking per query.
        second (dict[str, Ranking]): Ranking per query, same queries.
        k (int): Top pages compared.

    Returns:
        float: Mean share of shared pages, from 0 to 1.
    """
    shares = [
        len({p for p, _ in first[query][:k]} & {p for p, _ in second[query][:k]}) / k
        for query in first
    ]
    return round(float(np.mean(shares)), 4)


def exact_rankings(
    query_ids: list[str], queries: dict[str, np.ndarray], paths: list[Path], limit: int
) -> dict[str, Ranking]:
    """Rank every saved page for every query, one file of pages at a time.

    Args:
        query_ids (list[str]): Queries.
        queries (dict[str, np.ndarray]): Vectors per query.
        paths (list[Path]): Files of page vectors.
        limit (int): Pages kept per query.

    Returns:
        dict[str, Ranking]: Ranking per query.
    """
    vectors = [queries[query_id] for query_id in query_ids]
    page_ids: list[str] = []
    blocks = []
    for ids, pages in iter_multivectors(paths):
        page_ids.extend(ids)
        blocks.append(maxsim_matrix(vectors, pages))
    scores = np.hstack(blocks)
    return {query_id: rank_pages(page_ids, row, limit) for query_id, row in zip(query_ids, scores)}


def main() -> None:
    """Rank pages with each method, then score the rankings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/visual.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--local-only", action="store_true", help="Skip the Qdrant search.")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][args.dataset]
    bench = load_benchmark(args.dataset, spec, settings.data_root / "raw")
    name = visual_collection(args.dataset)
    limit = cfg["page_limit"]

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        qrels[qrel.query_id][qrel.page_id] = qrel.score
    query_ids = sorted(qrels)
    languages = {query_id: bench.queries[query_id].language for query_id in query_ids}

    root = visual_dir(settings.data_root, args.dataset, cfg["model"])
    queries = {
        query_id: vector
        for ids, vectors in iter_multivectors(part_files(root / "queries"))
        for query_id, vector in zip(ids, vectors)
    }
    if any(query_id not in queries for query_id in query_ids):
        raise SystemExit("Some queries have no embedding (run embed_visual.py).")
    sources = {"full": root / "pages"}
    if cfg["pool_factor"] > 1:
        sources["pooled"] = pooled_dir(root, cfg["pool_factor"])
    stored = "pooled" if "pooled" in sources else "full"  # the vectors Qdrant holds

    extra: dict[str, Any] = {
        "dataset": {"key": args.dataset, "repo": spec["repo"], "revision": spec["revision"]},
        "model": {"repo": cfg["model"], "revision": cfg["revision"]},
    }
    client = None
    if not args.local_only:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=cfg["qdrant_timeout"])
        extra.update(collection=name, qdrant_version=client.info().version)
    results_dir = settings.results_root / f"{cfg['name']}_eval" / args.dataset
    write_run_info(results_dir, extra=extra)
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rankings: dict[str, dict[str, Ranking]] = {}
    summary: dict[str, Any] = {"queries": len(query_ids), "pool_factor": cfg["pool_factor"], "local_seconds": {}}

    def save() -> None:
        """Write the metrics and rankings of the finished methods, and the summary."""
        metrics = {}
        for method, ranked in rankings.items():
            ids_only = {query_id: [page for page, _ in pages] for query_id, pages in ranked.items()}
            metrics[method] = evaluate(ids_only, qrels, languages)
            with (results_dir / f"rankings_{method}.jsonl").open("w", encoding="utf-8") as f:
                for query_id, pages in ranked.items():
                    f.write(json.dumps({"query_id": query_id, "pages": [[p, round(s, 6)] for p, s in pages]}) + "\n")
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for method, folder in sources.items():
        start = time.perf_counter()
        rankings[method] = exact_rankings(query_ids, queries, part_files(folder), limit)
        summary["local_seconds"][method] = round(time.perf_counter() - start, 1)
    if "pooled" in rankings:
        summary["top10_overlap_full_vs_pooled"] = mean_overlap(rankings["full"], rankings["pooled"])
    save()
    print(f"Local search done: {summary['local_seconds']}", flush=True)

    if client is not None:
        found: dict[str, Ranking] = {}
        seconds: list[float] = []
        for count, query_id in enumerate(query_ids, 1):
            start = time.perf_counter()
            hits = search(client, name, VISUAL, queries[query_id].tolist(), limit)
            seconds.append(time.perf_counter() - start)
            found[query_id] = [(payload["page_id"], score) for payload, score in hits]
            if count % 10 == 0 or count == len(query_ids):
                summary["qdrant_search"] = {
                    "queries_done": count,
                    "median_seconds": round(statistics.median(seconds), 2),
                    "mean_seconds": round(statistics.fmean(seconds), 2),
                    "max_seconds": round(max(seconds), 2),
                }
                (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"{count}/{len(query_ids)} queries, {statistics.fmean(seconds):.1f} s each", flush=True)
        rankings["qdrant"] = found
        summary[f"top10_overlap_qdrant_vs_{stored}"] = mean_overlap(found, rankings[stored])
        save()

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    print(f"{'method':<7} {'lang':<5} {'queries':>7} {'ndcg@10':>8} {'recall@100':>10}")
    for method, groups in metrics.items():
        for group, m in groups.items():
            print(f"{method:<7} {group:<5} {m['queries']:>7} {m['ndcg@10']:>8.3f} {m['recall@100']:>10.3f}")


if __name__ == "__main__":
    main()
