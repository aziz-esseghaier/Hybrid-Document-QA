"""Rerank the top pages of a dataset's saved visual ranking with the text reranker, one worker per GPU. Resumable.

Usage:
    python scripts/rerank_pages.py --dataset v2_esg_reports [--config configs/rerank.yaml] [--workers 2]

Scores (question, page OCR text) pairs for the top `candidates` pages of every judged query, reading
<DATA_ROOT>/derived/chunks/<dataset>/pages.jsonl (build_chunks.py). Workers append to
<DATA_ROOT>/derived/<name>/<dataset>/<model>/w<worker>.jsonl. Writes <RESULTS_ROOT>/<name>/<dataset>/:
rankings_rerank.jsonl ({query_id, pages: [[page_id, score], ...]}, the candidates in reranker order; ties keep the
visual order), summary.json, config.yaml, run_info.json. No document text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml, result_path
from doc_qa.core.parallel import append_jsonl, batched, launch_workers, read_jsonl, write_jsonl
from doc_qa.core.run_info import gpu_names, write_run_info
from doc_qa.core.schemas import Benchmark
from doc_qa.evaluation.vidore import load_benchmark
from doc_qa.retrieval.rerank import load_reranker, pair_ids, score_pairs


def load_inputs(dataset: str, cfg: dict[str, Any]) -> tuple[Benchmark, dict[str, str], dict[str, list[str]]]:
    """Load a dataset's queries, page texts and candidate pages.

    Args:
        dataset (str): Dataset key.
        cfg (dict[str, Any]): Reranker config.

    Returns:
        tuple[Benchmark, dict[str, str], dict[str, list[str]]]: Dataset, text per page, candidates per query.
    """
    settings = get_settings()
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
    bench = load_benchmark(dataset, spec, settings.data_root / "raw")
    pages_file = settings.data_root / "derived" / "chunks" / dataset / "pages.jsonl"
    texts = {row["page_id"]: row["text"] for row in read_jsonl([pages_file])}
    if not texts:
        raise SystemExit(f"No page text for {dataset} (run ocr_store.py pull and build_chunks.py first).")
    saved = read_jsonl([result_path(cfg["candidates_from"].format(dataset=dataset))])
    if not saved:
        raise SystemExit(f"No visual ranking for {dataset} (run eval_visual.py first).")
    candidates = {row["query_id"]: [page for page, _ in row["pages"][: cfg["candidates"]]] for row in saved}
    return bench, texts, candidates


def rerank_order(candidates: list[str], scores: list[float]) -> list[list[Any]]:
    """Order candidates by reranker score; ties keep their original order.

    Args:
        candidates (list[str]): Pages in visual order.
        scores (list[float]): Reranker score per page.

    Returns:
        list[list[Any]]: [page_id, score] pairs, best first.
    """
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [[candidates[i], scores[i]] for i in order]


def run_worker(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    """Score this worker's share of the queries not done yet.

    Args:
        args (argparse.Namespace): CLI arguments (with worker index and count).
        cfg (dict[str, Any]): Reranker config.
        out_dir (Path): Folder of the workers' score files.
    """
    worker, tag = args.worker_index, f"[worker {args.worker_index}]"
    bench, texts, candidates = load_inputs(args.dataset, cfg)
    out_file = out_dir / f"w{worker}.jsonl"
    done = {row["query_id"] for row in read_jsonl([out_file])}
    todo = [query_id for query_id in sorted(candidates)[worker :: args.num_workers] if query_id not in done]
    print(f"{tag} {len(todo)} queries to rerank", flush=True)
    if not todo:
        return

    model, tokenizer = load_reranker(
        cfg["model"], cfg["revision"], cfg["dtype"], "cuda:0", cfg["attn_implementation"]
    )
    count = 0
    for batch in batched(todo, cfg["part_queries"]):
        start = time.perf_counter()
        pairs = [(query_id, page) for query_id in batch for page in candidates[query_id]]
        prompts = [
            pair_ids(tokenizer, cfg["instruction"], bench.queries[query].text, texts.get(page, ""), cfg["max_tokens"])
            for query, page in pairs
        ]
        scores = iter(round(float(score), 4) for score in score_pairs(model, tokenizer, prompts, cfg["batch_size"]))
        append_jsonl(out_file, [
            {"query_id": query_id, "pages": [[page, next(scores)] for page in candidates[query_id]]}
            for query_id in batch
        ])
        count += len(batch)
        print(f"{tag} {count}/{len(todo)} ({(time.perf_counter() - start) / len(pairs):.2f} s/pair)", flush=True)


def main() -> None:
    """Launch one worker per GPU, check that every query is scored, and write the reranked candidates."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/rerank.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, help="Default: number of GPUs.")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    out_dir = settings.data_root / "derived" / cfg["name"] / args.dataset / cfg["model"].split("/")[-1]
    if args.worker_index is not None:
        run_worker(args, cfg, out_dir)
        return

    workers = args.workers or len(gpu_names())
    if not workers:
        raise SystemExit("Reranking needs a GPU.")
    from huggingface_hub import snapshot_download

    snapshot_download(cfg["model"], revision=cfg["revision"])  # once, before the workers start
    results_dir = settings.results_root / cfg["name"] / args.dataset
    write_run_info(results_dir, extra={
        "dataset": args.dataset, "model": {"repo": cfg["model"], "revision": cfg["revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    start = time.perf_counter()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--dataset", args.dataset]
    launch_workers(cmd, workers)
    seconds = time.perf_counter() - start

    _, texts, candidates = load_inputs(args.dataset, cfg)
    scored = {row["query_id"]: row["pages"] for row in read_jsonl(sorted(out_dir.glob("w*.jsonl")))}
    missing = set(candidates) - set(scored)
    if missing:
        raise SystemExit(f"{len(missing)} queries have no scores; rerun to finish.")
    rows = []
    for query_id in sorted(candidates):
        score_of = dict(map(tuple, scored[query_id]))
        rows.append({
            "query_id": query_id,
            "pages": rerank_order(candidates[query_id], [score_of[page] for page in candidates[query_id]]),
        })
    write_jsonl(results_dir / "rankings_rerank.jsonl", rows)
    pairs = sum(len(pages) for pages in candidates.values())
    summary = {
        "workers": workers,
        "queries": len(candidates),
        "candidates": cfg["candidates"],
        "pairs": pairs,
        "seconds": round(seconds),
        "seconds_per_pair_two_gpus": round(seconds / pairs, 3),
        "candidates_without_text": sum(
            len(texts.get(page, "").strip()) < 20 for pages in candidates.values() for page in pages
        ),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
