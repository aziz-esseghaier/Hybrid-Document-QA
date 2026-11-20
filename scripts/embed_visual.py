"""Embed a dataset's page images and queries with the visual retriever, one worker per GPU. Resumable.

Usage:
    python scripts/embed_visual.py --dataset v2_esg_reports [--workers 2]

Saves every token vector, then merges each page's vectors so the page fits in one Qdrant point (unless
`pool_factor` is 1). Writes <DATA_ROOT>/derived/visual/<dataset>/<model>/{pages,queries,pooled_<factor>}/*.npz and
<RESULTS_ROOT>/<name>_embeddings/<dataset>/: config.yaml, run_info.json, summary.json (`name` from the config).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import batched, launch_workers
from doc_qa.core.run_info import gpu_names, write_run_info
from doc_qa.core.schemas import Benchmark
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark
from doc_qa.retrieval.visual import (
    decode_image,
    embed_images,
    embed_queries,
    fetch_visual_model,
    iter_multivectors,
    load_visual_model,
    multivector_lengths,
    part_files,
    pool_vectors,
    pooled_dir,
    save_multivectors,
    spread,
    visual_dir,
)


def load_dataset(dataset: str, cfg: dict[str, Any]) -> Benchmark:
    """Load a dataset.

    Args:
        dataset (str): Dataset key.
        cfg (dict[str, Any]): Visual config.

    Returns:
        Benchmark: Dataset.
    """
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
    return load_benchmark(dataset, spec, get_settings().data_root / "raw")


def todo_ids(ids: Iterable[str], done: Iterable[str], worker: int, workers: int) -> list[str]:
    """Return this worker's share of the items, without those already done.

    Args:
        ids (Iterable[str]): All items.
        done (Iterable[str]): Finished items.
        worker (int): Worker index.
        workers (int): Number of workers.

    Returns:
        list[str]: Items to process.
    """
    finished = set(done)
    return [item for item in sorted(ids)[worker::workers] if item not in finished]


def run_worker(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    """Embed this worker's share of the queries and pages not done yet.

    Args:
        args (argparse.Namespace): CLI arguments (with worker index and count).
        cfg (dict[str, Any]): Visual config.
        out_dir (Path): Embedding folder.
    """
    worker, tag = args.worker_index, f"[worker {args.worker_index}]"
    bench = load_dataset(args.dataset, cfg)
    page_dir, query_dir = out_dir / "pages", out_dir / "queries"
    pages = todo_ids(bench.pages, multivector_lengths(part_files(page_dir)), worker, args.num_workers)
    queries = todo_ids(bench.queries, multivector_lengths(part_files(query_dir)), worker, args.num_workers)
    print(f"{tag} {len(pages)} pages and {len(queries)} queries to embed", flush=True)
    if not pages and not queries:
        return

    model, processor, problems = load_visual_model(
        cfg["model"], cfg["revision"], cfg["dtype"], "cuda:0", cfg["attn_implementation"]
    )
    if problems:
        raise SystemExit(f"{len(problems)} weights did not load cleanly, e.g. {problems[0]}")

    if queries:
        vectors = embed_queries(model, processor, [bench.queries[q].text for q in queries], cfg["query_batch_size"])
        part = len(list(query_dir.glob(f"w{worker}_*.npz")))
        save_multivectors(query_dir / f"w{worker}_{part:05d}.npz", queries, vectors)

    part, count = len(list(page_dir.glob(f"w{worker}_*.npz"))), 0
    for batch in batched(iter_page_images(bench, pages), cfg["part_pages"]):
        start = time.perf_counter()
        images = [decode_image(data) for _, data in batch]
        vectors = embed_images(model, processor, images, cfg["page_batch_size"])
        save_multivectors(page_dir / f"w{worker}_{part:05d}.npz", [page_id for page_id, _ in batch], vectors)
        part, count = part + 1, count + len(batch)
        print(f"{tag} {count}/{len(pages)} ({(time.perf_counter() - start) / len(batch):.2f} s/page)", flush=True)


def pool_pages(page_dir: Path, out_dir: Path, factor: int) -> None:
    """Merge the vectors of every saved page, skipping files already merged.

    Args:
        page_dir (Path): Full page vectors.
        out_dir (Path): Output folder; with factor 1 it is `page_dir`, and nothing is written.
        factor (int): Reduction factor.
    """
    for path in part_files(page_dir):
        target = out_dir / path.name
        if not target.exists():
            for ids, vectors in iter_multivectors([path]):
                save_multivectors(target, ids, [pool_vectors(page, factor) for page in vectors])


def main() -> None:
    """Launch one embedding worker per GPU, check that nothing is missing, merge page vectors and summarize."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/visual.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, help="Default: number of GPUs.")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    out_dir = visual_dir(settings.data_root, args.dataset, cfg["model"])
    if args.worker_index is not None:
        run_worker(args, cfg, out_dir)
        return

    workers = args.workers or len(gpu_names())
    if not workers:
        raise SystemExit("Embedding pages needs a GPU.")
    fetch_visual_model(cfg["model"], cfg["revision"])
    results_dir = settings.results_root / f"{cfg['name']}_embeddings" / args.dataset
    write_run_info(results_dir, extra={
        "dataset": args.dataset, "model": {"repo": cfg["model"], "revision": cfg["revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    start = time.perf_counter()
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--dataset", args.dataset]
    launch_workers(cmd, workers)
    embed_seconds = time.perf_counter() - start

    bench = load_dataset(args.dataset, cfg)
    page_dir, query_dir = out_dir / "pages", out_dir / "queries"
    pages, queries = multivector_lengths(part_files(page_dir)), multivector_lengths(part_files(query_dir))
    missing = len(set(bench.pages) - set(pages)) + len(set(bench.queries) - set(queries))
    if missing:
        raise SystemExit(f"{missing} pages or queries have no embedding; rerun to finish.")

    start = time.perf_counter()
    merged_dir = pooled_dir(out_dir, cfg["pool_factor"])
    pool_pages(page_dir, merged_dir, cfg["pool_factor"])
    pooled = multivector_lengths(part_files(merged_dir))
    with np.load(part_files(page_dir)[0], allow_pickle=False) as data:
        dim = int(data["vectors"].shape[1])

    summary = {
        "workers": workers,
        "embed_seconds": round(embed_seconds),
        "pool_seconds": round(time.perf_counter() - start),
        "pages": len(pages),
        "queries": len(queries),
        "dim": dim,
        "vectors_per_page": spread(list(pages.values())),
        "vectors_per_query": spread(list(queries.values())),
        "pool_factor": cfg["pool_factor"],
        "pooled_vectors_per_page": spread(list(pooled.values())),
        "full_gb": round(sum(pages.values()) * dim * 2 / 1e9, 2),
        "pooled_gb": round(sum(pooled.values()) * dim * 2 / 1e9, 2),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
