"""Bridge check inputs: the same dev pages with PDF text-layer text and with OCR text.

Usage:
    python scripts/build_bridge.py [--config configs/bridge.yaml]

Downloads the PDFs listed in <urls_dataset> (pdf_url), renders their pages and matches them to <dataset>'s page
images by pixel correlation. For the matched pages, writes two chunk corpora with the same pages:
<dataset>-bridge-ocr (cleaned OCR text) and <dataset>-bridge-pdf (text layer, or the OCR text where the layer has
too little text, as in the product). Then run embed_text, index_text and eval_text with --corpus on each.

PDFs go to <DATA_ROOT>/raw/pdfs/<urls_dataset>/, chunks to <DATA_ROOT>/derived/chunks/<corpus>/.
<RESULTS_ROOT>/bridge/<dataset>/ gets pdfs.csv, pages.csv, summary.json, config.yaml and run_info.json:
ids, public URLs and statistics, no document text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import token_f1
from doc_qa.evaluation.overlap import PageView, best_matches, image_view, page_views
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark
from doc_qa.ingest.pdf import clean_pdf_text, read_pdf, word_chars
from doc_qa.ingest.text import chunk_pages, clean_ocr_text


def download_pdf(url: str, path: Path, timeout: float) -> str:
    """Download a PDF unless already on disk.

    Args:
        url (str): PDF URL.
        path (Path): Output file.
        timeout (float): Seconds.

    Returns:
        str: "ok", "not a PDF", or the error.
    """
    if path.exists():
        return "ok"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return type(e).__name__
    if not data.startswith(b"%PDF"):
        return "not a PDF"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "ok"


def pdf_pages(path: Path, scale: float, size: int) -> list[tuple[int, PageView, str]] | None:
    """Render and read every page of a PDF (runs in a worker process).

    Args:
        path (Path): PDF file.
        scale (float): Render scale; 1 = 72 dpi.
        size (int): Comparison grid side.

    Returns:
        list[tuple[int, PageView, str]] | None: (page index from 0, view, raw text), or None if unreadable.
    """
    try:
        return [(index, image_view(image, size), text) for index, image, text in read_pdf(path, scale)]
    except Exception:  # PDFium raises its own error types for corrupt files
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, columns from the first row.

    Args:
        path (Path): Output file.
        rows (list[dict[str, Any]]): Rows.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Match PDF pages to dataset pages and write the two bridge corpora."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/bridge.yaml"))
    args = parser.parse_args()

    from tokenizers import Tokenizer

    settings = get_settings()
    cfg = load_yaml(args.config)
    text_cfg = load_yaml(Path(cfg["text_config"]))
    specs = load_yaml(Path(text_cfg["datasets_config"]))["datasets"]
    dataset, raw_root = cfg["dataset"], settings.data_root / "raw"
    results_dir = settings.results_root / "bridge" / dataset

    bench = load_benchmark(dataset, specs[dataset], raw_root)
    url_bench = load_benchmark(cfg["urls_dataset"], specs[cfg["urls_dataset"]], raw_root)
    urls = sorted({d.url for d in url_bench.documents.values() if d.url})
    ocr_dir = settings.data_root / "derived" / "ocr" / dataset
    ocr_raw = {r["page_id"]: r["text"] for r in read_jsonl(sorted(ocr_dir.glob("shard*.jsonl")))}
    if not ocr_raw:
        raise SystemExit(f"No OCR output in {ocr_dir} (run ocr_store.py pull first).")

    write_run_info(results_dir, extra={
        "dataset": {"key": dataset, "repo": specs[dataset]["repo"], "revision": specs[dataset]["revision"]},
        "urls_dataset": {k: specs[cfg["urls_dataset"]][k] for k in ("repo", "revision")},
        "tokenizer": {"repo": text_cfg["embedding_model"], "revision": text_cfg["embedding_revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 1. Download the PDFs.
    pdf_dir = raw_root / "pdfs" / cfg["urls_dataset"]
    pdfs = []
    for url in urls:
        path = pdf_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.pdf"
        status = download_pdf(url, path, cfg["download_timeout"])
        print(f"{status:<10} {url}", flush=True)
        pdfs.append({"url": url, "file": path.name, "status": status, "pages": 0, "matched": 0,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if status == "ok" else ""})

    # 2. Render the PDF pages and read their text layers; view the dataset's page images the same way.
    pdf_views: dict[str, PageView] = {}
    layer_raw: dict[str, str] = {}
    ok = [p for p in pdfs if p["status"] == "ok"]
    read = partial(pdf_pages, scale=cfg["render_scale"], size=cfg["verify_size"])
    with ProcessPoolExecutor(cfg["workers"]) as pool:
        for pdf, pages in zip(ok, pool.map(read, [pdf_dir / p["file"] for p in ok])):
            if pages is None:
                pdf["status"] = "unreadable"
                continue
            pdf["pages"] = len(pages)
            for index, view, text in pages:
                pdf_views[f"{pdf['file']}#{index}"] = view
                layer_raw[f"{pdf['file']}#{index}"] = text
    print(f"{len(pdf_views)} PDF pages from {sum(p['pages'] > 0 for p in pdfs)}/{len(urls)} PDFs", flush=True)
    dataset_views = page_views(iter_page_images(bench), cfg["verify_size"], cfg["workers"])

    # 3. Match each dataset page to its most correlated PDF page.
    matches = best_matches(dataset_views, pdf_views, cfg["min_ink_share"])
    matched = {p: m for p, m in matches.items() if m[1] >= cfg["min_pixel_correlation"]}
    url_of = {p["file"]: p["url"] for p in pdfs}

    # 4. Texts of the matched pages: OCR, and the text layer with the product's OCR fallback.
    ocr_texts, pdf_texts, rows = {}, {}, []
    for page_id in sorted(matched):
        key, correlation, runner_up = matched[page_id]
        file, index = key.rsplit("#", 1)
        ocr = clean_ocr_text(ocr_raw.get(page_id, ""))
        layer = clean_pdf_text(layer_raw[key])
        fallback = word_chars(layer) < cfg["min_word_chars"]
        ocr_texts[page_id], pdf_texts[page_id] = ocr, (ocr if fallback else layer)
        rows.append({
            "page_id": page_id, "url": url_of[file], "pdf_page": int(index) + 1,
            "correlation": correlation, "runner_up": runner_up,
            "layer_word_chars": word_chars(layer), "ocr_word_chars": word_chars(ocr),
            "fallback": fallback, "token_f1": round(token_f1(layer, ocr), 4),
        })
    for pdf in pdfs:
        pdf["matched"] = sum(r["url"] == pdf["url"] for r in rows)

    # 5. Chunk both versions with the main text config.
    tokenizer = Tokenizer.from_pretrained(text_cfg["embedding_model"], revision=text_cfg["embedding_revision"])
    corpora = {}
    for source, texts in (("ocr", ocr_texts), ("pdf", pdf_texts)):
        corpus = f"{dataset}-bridge-{source}"
        pages, chunks, corpora[corpus] = chunk_pages(
            texts, tokenizer, text_cfg["chunk_tokens"], text_cfg["chunk_overlap"]
        )
        out_dir = settings.data_root / "derived" / "chunks" / corpus
        write_jsonl(out_dir / "pages.jsonl", pages)
        write_jsonl(out_dir / "chunks.jsonl", chunks)

    f1 = np.array([r["token_f1"] for r in rows if not r["fallback"]])
    unmatched = [m[1] for p, m in matches.items() if p not in matched]
    summary = {
        "pdfs": len(urls),
        "pdfs_read": sum(p["pages"] > 0 for p in pdfs),
        "pdf_pages": len(pdf_views),
        "dataset_pages": len(bench.pages),
        "dataset_pages_blank": len(dataset_views) - len(matches),
        "matched_pages": len(matched),
        "near_misses": sum(c >= 0.8 for c in unmatched),  # best correlation in [0.8, threshold)
        "queries_with_matched_page": len({q.query_id for q in bench.qrels if q.page_id in matched}),
        "fallback_pages": sum(r["fallback"] for r in rows),
        "token_f1_p10": round(float(np.percentile(f1, 10)), 3) if f1.size else None,
        "token_f1_p50": round(float(np.percentile(f1, 50)), 3) if f1.size else None,
        "corpora": corpora,
    }
    write_csv(results_dir / "pdfs.csv", pdfs)
    write_csv(results_dir / "pages.csv", rows)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
