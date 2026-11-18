"""Reuse one dataset's OCR for another dataset that contains the same pages.

Usage:
    python scripts/reuse_ocr.py [--config configs/ocr_reuse.yaml]

A target page reuses a source page's OCR if their pixels are identical, or if it passes a strict pixel-correlation
check. Writes <DATA_ROOT>/derived/ocr/<target>/shard_reused.jsonl; ocr_pages.py then OCRs only the remaining pages.
<RESULTS_ROOT>/ocr_reuse/<target>/ gets matches.csv, summary.json, config.yaml and run_info.json (no document text).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_csv, write_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.overlap import best_matches, hash_pages, page_views
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark


def plan_reuse(
    target_hashes: dict[str, str],
    source_hashes: dict[str, str],
    matches: dict[str, tuple[str, float, float]],
    ocr_pages: set[str],
    min_correlation: float,
    min_margin: float,
) -> dict[str, tuple[str, str]]:
    """Choose the source page whose OCR each target page reuses.

    Args:
        target_hashes (dict[str, str]): Pixel hash per target page.
        source_hashes (dict[str, str]): Pixel hash per source page.
        matches (dict[str, tuple[str, float, float]]): Output of `best_matches` for target pages.
        ocr_pages (set[str]): Source pages that have OCR.
        min_correlation (float): Min pixel correlation.
        min_margin (float): Min gap between the best and second-best correlation.

    Returns:
        dict[str, tuple[str, str]]: {target page: (source page, "exact" or "pixels")}; other pages need OCR.
    """
    by_hash: dict[str, str] = {}
    for page_id in sorted(source_hashes):
        if page_id in ocr_pages:
            by_hash.setdefault(source_hashes[page_id], page_id)

    plan = {}
    for page_id, digest in target_hashes.items():
        if digest in by_hash:
            plan[page_id] = (by_hash[digest], "exact")
        elif page_id in matches:
            source, correlation, runner_up = matches[page_id]
            if source in ocr_pages and correlation >= min_correlation and correlation - runner_up >= min_margin:
                plan[page_id] = (source, "pixels")
    return plan


def main() -> None:
    """Match the target's pages to the source's and write the reused OCR."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/ocr_reuse.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    specs = load_yaml(Path(cfg["datasets_config"]))["datasets"]
    target, source = cfg["target"], cfg["source"]
    raw_root, ocr_root = settings.data_root / "raw", settings.data_root / "derived" / "ocr"
    target_bench = load_benchmark(target, specs[target], raw_root)
    source_bench = load_benchmark(source, specs[source], raw_root)

    records = {r["page_id"]: r for r in read_jsonl(sorted((ocr_root / source).glob("shard*.jsonl")))}
    if not records:
        raise SystemExit(f"No OCR output for {source} (run ocr_store.py pull first).")
    manifest_path = ocr_root / source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    results_dir = settings.results_root / "ocr_reuse" / target
    write_run_info(results_dir, extra={
        "target": {"key": target, "repo": specs[target]["repo"], "revision": specs[target]["revision"]},
        "source": {"key": source, "repo": specs[source]["repo"], "revision": specs[source]["revision"]},
        "source_ocr": {k: manifest.get(k) for k in ("git_commit", "pipeline_version", "created_at")},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    workers = cfg["workers"]
    target_hashes = {h.page_id: h.sha256 for h in hash_pages(iter_page_images(target_bench), workers)}
    source_hashes = {h.page_id: h.sha256 for h in hash_pages(iter_page_images(source_bench), workers)}
    exact = set(source_hashes.values())
    remaining = [p for p, digest in target_hashes.items() if digest not in exact]
    matches: dict[str, tuple[str, float, float]] = {}
    blank = 0
    if remaining:
        size = cfg["verify_size"]
        target_views = page_views(iter_page_images(target_bench, remaining), size, workers)
        source_views = page_views(iter_page_images(source_bench), size, workers)
        matches = best_matches(target_views, source_views, cfg["min_ink_share"])
        blank = len(target_views) - len(matches)

    plan = plan_reuse(
        target_hashes, source_hashes, matches, set(records), cfg["min_pixel_correlation"], cfg["min_margin"]
    )
    write_jsonl(ocr_root / target / "shard_reused.jsonl", [
        {**records[source_page], "page_id": page_id, "reused_from": source_page}
        for page_id, (source_page, _) in sorted(plan.items())
    ])

    rows: list[dict[str, Any]] = []
    for page_id in sorted(target_hashes):
        best, correlation, runner_up = matches.get(page_id, (None, None, None))
        source_page, method = plan.get(page_id, (None, "ocr"))
        rows.append({"page_id": page_id, "method": method, "source_page": source_page, "best_candidate": best,
                     "correlation": correlation, "runner_up": runner_up})
    write_csv(results_dir / "matches.csv", rows)

    to_ocr = {r["page_id"] for r in rows if r["method"] == "ocr"}
    summary = {
        "pages": len(rows),
        "reused_exact": sum(r["method"] == "exact" for r in rows),
        "reused_pixels": sum(r["method"] == "pixels" for r in rows),
        "pages_to_ocr": len(to_ocr),
        "blank_without_exact_match": blank,
        "near_misses": sum(1 for r in rows if r["method"] == "ocr" and (r["correlation"] or 0) >= 0.9),
        "relevant_pages": len({q.page_id for q in target_bench.qrels}),
        "relevant_pages_to_ocr": len({q.page_id for q in target_bench.qrels} & to_ocr),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
