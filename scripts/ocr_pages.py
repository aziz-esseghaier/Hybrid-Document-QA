"""OCR a dataset's page images with PaddleOCR-VL, one worker per GPU. Resumable.

Usage:
    python scripts/ocr_pages.py --dataset v2_esg_reports [--limit 20] [--workers 2]

Writes <DATA_ROOT>/derived/ocr/<dataset>/shard*.jsonl ({page_id, text, seconds}) and
<RESULTS_ROOT>/ocr/<dataset>/: config.yaml, run_info.json, summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import append_jsonl, batched, launch_workers, read_jsonl
from doc_qa.core.run_info import gpu_names, write_run_info
from doc_qa.core.schemas import Benchmark
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark
from doc_qa.ingest.ocr import load_pipeline, ocr_images


def select_pages(page_ids: list[str], limit: int | None) -> list[str]:
    """Return all page ids, or `limit` of them spread evenly, in sorted order.

    Args:
        page_ids (list[str]): Page ids.
        limit (int | None): Max pages; None = all.

    Returns:
        list[str]: Selected ids.
    """
    ids = sorted(page_ids)
    if not limit or limit >= len(ids):
        return ids
    step = len(ids) / limit
    return [ids[int(i * step)] for i in range(limit)]


def load(dataset: str, cfg: dict[str, Any]) -> tuple[Benchmark, dict[str, Any]]:
    """Load a dataset and its config entry.

    Args:
        dataset (str): Dataset key.
        cfg (dict[str, Any]): OCR config.

    Returns:
        tuple[Benchmark, dict[str, Any]]: Dataset and its datasets.yaml entry.
    """
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
    return load_benchmark(dataset, spec, get_settings().data_root / "raw"), spec


def run_worker(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> None:
    """OCR this worker's share of the pages not done yet.

    Args:
        args (argparse.Namespace): CLI arguments (with worker index and count).
        cfg (dict[str, Any]): OCR config.
        out_dir (Path): Shard folder.
    """
    tag = f"[worker {args.worker_index}]"
    bench, _ = load(args.dataset, cfg)
    done = {r["page_id"] for r in read_jsonl(out_dir.glob("shard*.jsonl"))}
    mine = select_pages(list(bench.pages), args.limit)[args.worker_index :: args.num_workers]
    todo = [p for p in mine if p not in done]
    print(f"{tag} {len(todo)} pages to OCR, {len(mine) - len(todo)} already done", flush=True)
    if not todo:
        return

    pipeline = load_pipeline(cfg["pipeline_version"], "gpu:0" if gpu_names() else "cpu", **cfg["pipeline_options"])
    shard = out_dir / f"shard{args.worker_index}.jsonl"
    count = 0
    for batch in batched(iter_page_images(bench, todo), cfg["batch_size"]):
        start = time.perf_counter()
        texts = ocr_images(pipeline, [data for _, data in batch], **cfg["predict_options"])
        seconds = (time.perf_counter() - start) / len(batch)
        append_jsonl(shard, [{"page_id": pid, "text": t, "seconds": round(seconds, 3)} for (pid, _), t in zip(batch, texts)])
        count += len(batch)
        print(f"{tag} {count}/{len(todo)} ({seconds:.1f} s/page)", flush=True)


def summarize(records: list[dict[str, Any]], total_pages: int, workers: int) -> dict[str, Any]:
    """Summarize OCR progress and speed.

    Args:
        records (list[dict[str, Any]]): OCR records, one per page.
        total_pages (int): Pages in the dataset.
        workers (int): Parallel workers.

    Returns:
        dict[str, Any]: Summary.
    """
    if not records:
        return {"pages_done": 0, "pages_total": total_pages}
    seconds = sum(r["seconds"] for r in records) / len(records)
    return {
        "pages_done": len(records),
        "pages_total": total_pages,
        "seconds_per_page_per_gpu": round(seconds, 2),
        "pages_per_hour": round(3600 * workers / seconds) if seconds else None,
        "hours_left": round((total_pages - len(records)) * seconds / workers / 3600, 2),
        "near_empty_pages": sum(1 for r in records if len(r["text"].strip()) < 20),
        "mean_chars": round(sum(len(r["text"]) for r in records) / len(records)),
    }


def main() -> None:
    """Launch one OCR worker per GPU, then summarize."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/ocr.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, help="OCR only this many pages, spread over the dataset.")
    parser.add_argument("--workers", type=int, help="Default: number of GPUs (1 on CPU).")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    out_dir = settings.data_root / "derived" / "ocr" / args.dataset
    if args.worker_index is not None:
        run_worker(args, cfg, out_dir)
        return

    bench, spec = load(args.dataset, cfg)
    results_dir = settings.results_root / "ocr" / args.dataset
    write_run_info(results_dir, extra={"dataset": {"repo": spec["repo"], "revision": spec["revision"]}})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    workers = args.workers or max(1, len(gpu_names()))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config), "--dataset", args.dataset]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    launch_workers(cmd, workers)

    records = {r["page_id"]: r for r in read_jsonl(sorted(out_dir.glob("shard*.jsonl")))}
    summary = summarize(list(records.values()), len(bench.pages), workers)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
