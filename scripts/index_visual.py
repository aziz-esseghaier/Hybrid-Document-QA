"""Load a dataset's merged page vectors into a Qdrant multi-vector collection.

Usage:
    python scripts/index_visual.py --dataset v2_esg_reports

Creates (or replaces) collection visual_<dataset>: one point per page, payload {page_id}. Needs embed_visual.py's
output and a running Qdrant (QDRANT_URL). Writes <RESULTS_ROOT>/visual_index/<dataset>/: config.yaml, run_info.json,
summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.vidore import load_benchmark
from doc_qa.retrieval.qdrant import (
    create_visual_collection,
    fits_in_point,
    upsert_page,
    visual_collection,
    wait_until_ready,
)
from doc_qa.retrieval.visual import iter_multivectors, multivector_lengths, part_files, visual_dir


def main() -> None:
    """Build the collection and check its point count."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/visual.yaml"))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    settings = get_settings()
    cfg = load_yaml(args.config)
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][args.dataset]
    bench = load_benchmark(args.dataset, spec, settings.data_root / "raw")
    paths = part_files(visual_dir(settings.data_root, args.dataset, cfg["model"]) / f"pooled_{cfg['pool_factor']}")
    lengths = multivector_lengths(paths)
    if set(lengths) != set(bench.pages):
        raise SystemExit(f"{len(lengths)} merged pages for {len(bench.pages)} pages (run embed_visual.py).")

    _, first = next(iter_multivectors(paths[:1]))
    dim = int(first[0].shape[1])
    if not fits_in_point(max(lengths.values()), dim):
        raise SystemExit(f"A page has {max(lengths.values())} vectors of {dim} dimensions: too large for Qdrant.")

    name = visual_collection(args.dataset)
    client = QdrantClient(url=settings.qdrant_url, timeout=cfg["qdrant_timeout"])
    results_dir = settings.results_root / "visual_index" / args.dataset
    write_run_info(results_dir, extra={
        "dataset": args.dataset,
        "collection": name,
        "qdrant_version": client.info().version,
        "model": {"repo": cfg["model"], "revision": cfg["revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    start = time.perf_counter()
    create_visual_collection(client, name, dim)
    point_id = 0
    with ThreadPoolExecutor(cfg["upload_threads"]) as pool:
        for ids, vectors in iter_multivectors(paths):  # one file at a time, to keep memory low
            jobs = [(point_id + i, page_id, page) for i, (page_id, page) in enumerate(zip(ids, vectors))]
            list(pool.map(lambda job: upsert_page(client, name, *job), jobs))
            point_id += len(jobs)
            print(f"{point_id}/{len(lengths)} pages", flush=True)
    upload_seconds = time.perf_counter() - start
    wait_until_ready(client, name)
    points = client.count(name, exact=True).count
    if points != len(lengths):
        raise SystemExit(f"Collection has {points} points, expected {len(lengths)}.")

    summary = {
        "collection": name,
        "points": points,
        "dim": dim,
        "pool_factor": cfg["pool_factor"],
        "vectors": sum(lengths.values()),
        "upload_seconds_per_page": round(upload_seconds / points, 3),
        "total_seconds": round(time.perf_counter() - start),
        "segments": client.get_collection(name).segments_count,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
