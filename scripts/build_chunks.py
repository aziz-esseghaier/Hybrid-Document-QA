"""Clean a dataset's page text and split each page into token chunks.

Usage:
    python scripts/build_chunks.py --dataset v2_esg_reports [--source ocr|markdown]

The text comes from OCR, <DATA_ROOT>/derived/ocr/<dataset>/shard*.jsonl (default), or with --source markdown from the
text ViDoRe V3 ships with each page (download_data.py). Writes to <DATA_ROOT>/derived/chunks/<dataset>/: pages.jsonl
({page_id, text}, cleaned) and chunks.jsonl ({chunk_id, page_id, text, start, end, tokens}; start/end are character
offsets in the cleaned page text). Config, run_info.json and summary.json go to <RESULTS_ROOT>/chunks/<dataset>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.vidore import iter_page_markdown, load_benchmark
from doc_qa.ingest.text import chunk_pages, clean_ocr_text


def read_texts(dataset: str, source: str, cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Read a dataset's raw page text.

    Args:
        dataset (str): Dataset key.
        source (str): "ocr" or "markdown".
        cfg (dict[str, Any]): Text config.

    Returns:
        tuple[dict[str, str], dict[str, Any]]: Text per page id, and where it came from (for run_info).
    """
    settings = get_settings()
    if source == "markdown":
        spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
        bench = load_benchmark(dataset, spec, settings.data_root / "raw")
        return dict(iter_page_markdown(bench)), {"source": "markdown", "revision": spec["revision"]}
    ocr_dir = settings.data_root / "derived" / "ocr" / dataset
    records = {r["page_id"]: r["text"] for r in read_jsonl(sorted(ocr_dir.glob("shard*.jsonl")))}
    if not records:
        raise SystemExit(f"No OCR output in {ocr_dir} (run ocr_store.py pull first).")
    manifest_path = ocr_dir / "manifest.json"
    ocr = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return records, {"source": "ocr", **{k: ocr.get(k) for k in ("git_commit", "pipeline_version", "created_at")}}


def main() -> None:
    """Clean, chunk and summarize one dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/text.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", choices=["ocr", "markdown"], default="ocr")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    settings = get_settings()
    cfg = load_yaml(args.config)
    out_dir = settings.data_root / "derived" / "chunks" / args.dataset
    results_dir = settings.results_root / "chunks" / args.dataset

    records, origin = read_texts(args.dataset, args.source, cfg)
    write_run_info(results_dir, extra={
        "dataset": args.dataset,
        "tokenizer": {"repo": cfg["embedding_model"], "revision": cfg["embedding_revision"]},
        "text": origin,
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    tokenizer = Tokenizer.from_pretrained(cfg["embedding_model"], revision=cfg["embedding_revision"])
    texts = {page_id: clean_ocr_text(text) for page_id, text in records.items()}
    pages, chunks, summary = chunk_pages(texts, tokenizer, cfg["chunk_tokens"], cfg["chunk_overlap"])

    write_jsonl(out_dir / "pages.jsonl", pages)
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
