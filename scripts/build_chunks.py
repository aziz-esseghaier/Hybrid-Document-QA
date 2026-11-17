"""Clean a dataset's OCR text and split each page into token chunks.

Usage:
    python scripts/build_chunks.py --dataset v2_esg_reports

Reads <DATA_ROOT>/derived/ocr/<dataset>/shard*.jsonl. Writes to <DATA_ROOT>/derived/chunks/<dataset>/:
pages.jsonl ({page_id, text}, cleaned) and chunks.jsonl ({chunk_id, page_id, text, start, end, tokens};
start/end are character offsets in the cleaned page text). Config, run_info.json and summary.json go to
<RESULTS_ROOT>/chunks/<dataset>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.ingest.text import chunk_text, clean_ocr_text


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSONL file, replacing it.

    Args:
        path (Path): Output file.
        records (list[dict[str, Any]]): Records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(page_tokens: list[int], chunks_per_page: list[int], chunk_tokens: int) -> dict[str, Any]:
    """Summarize page lengths and chunk counts.

    Args:
        page_tokens (list[int]): Tokens per page.
        chunks_per_page (list[int]): Chunks per page.
        chunk_tokens (int): Max tokens per chunk.

    Returns:
        dict[str, Any]: Summary.
    """
    tokens = np.array(page_tokens)
    with_text = tokens[tokens > 0]
    return {
        "pages": len(tokens),
        "empty_pages": int((tokens == 0).sum()),
        "chunks": int(sum(chunks_per_page)),
        "page_tokens_p50": int(np.percentile(with_text, 50)) if with_text.size else 0,
        "page_tokens_p90": int(np.percentile(with_text, 90)) if with_text.size else 0,
        "page_tokens_max": int(with_text.max()) if with_text.size else 0,
        "pages_in_one_chunk": round(float((with_text <= chunk_tokens).mean()), 3) if with_text.size else None,
    }


def main() -> None:
    """Clean, chunk and summarize one dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/text.yaml"))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    settings = get_settings()
    cfg = load_yaml(args.config)
    ocr_dir = settings.data_root / "derived" / "ocr" / args.dataset
    out_dir = settings.data_root / "derived" / "chunks" / args.dataset
    results_dir = settings.results_root / "chunks" / args.dataset

    records = {r["page_id"]: r["text"] for r in read_jsonl(sorted(ocr_dir.glob("shard*.jsonl")))}
    if not records:
        raise SystemExit(f"No OCR output in {ocr_dir} (run ocr_store.py pull first).")
    manifest_path = ocr_dir / "manifest.json"
    ocr = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    write_run_info(results_dir, extra={
        "dataset": args.dataset,
        "tokenizer": {"repo": cfg["embedding_model"], "revision": cfg["embedding_revision"]},
        "ocr": {k: ocr.get(k) for k in ("git_commit", "pipeline_version", "created_at")},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    tokenizer = Tokenizer.from_pretrained(cfg["embedding_model"], revision=cfg["embedding_revision"])
    page_ids = sorted(records)
    texts = [clean_ocr_text(records[p]) for p in page_ids]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)

    pages, chunks, page_tokens, chunks_per_page = [], [], [], []
    for page_id, text, enc in zip(page_ids, texts, encodings):
        page_chunks = chunk_text(text, enc.offsets, cfg["chunk_tokens"], cfg["chunk_overlap"])
        pages.append({"page_id": page_id, "text": text})
        chunks += [{"chunk_id": f"{page_id}#{i}", "page_id": page_id, **c} for i, c in enumerate(page_chunks)]
        page_tokens.append(len(enc.offsets))
        chunks_per_page.append(len(page_chunks))

    write_jsonl(out_dir / "pages.jsonl", pages)
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    summary = summarize(page_tokens, chunks_per_page, cfg["chunk_tokens"])
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
