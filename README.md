# Document Q&A

Question answering over English and French PDFs (text, tables, charts, scans) with page citations. Hybrid retrieval: a text branch and a visual branch, fused and reranked. Open-source models only.

Evaluation: [ViDoRe V3](https://huggingface.co/vidore) (test), ViDoRe V2 + French ViDoRe V1 (dev).

## Layout

```
configs/      datasets (pinned revisions), experiment configs, dev exclusions
src/doc_qa/   package: core/ (settings, schemas, run metadata), evaluation/ (loaders, checks, metrics),
              ingest/ (OCR, PDF text, chunks), retrieval/ (embeddings, BM25, Qdrant, fusion)
scripts/      entry points
tests/        unit tests
```

## Local setup

```
conda env create -f environment.yml
conda activate doc-qa
copy .env.example .env      # Linux: cp
pytest
docker compose up -d        # Qdrant on http://localhost:6333
```

## Kaggle

Notebook settings: Internet on, environment pinned. Secrets: `HF_TOKEN` (optional), `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (or a legacy `KAGGLE_KEY`), for saving outputs to private Datasets.

```python
!git clone https://github.com/<user>/Hybrid-Document-QA.git /kaggle/temp/Hybrid-Document-QA
%cd /kaggle/temp/Hybrid-Document-QA
!pip install -q -e .
%run scripts/kaggle_setup.py
```

`kaggle_setup.py` must run with `%run` so its environment variables reach later cells. Data goes to `/kaggle/temp` (wiped at session end), results to `/kaggle/working/results`. `--qdrant` also downloads and starts the Qdrant binary, at the version pinned in `docker-compose.yml`.

## Page OCR

PaddleOCR-VL runs in its own virtual env so Paddle's CUDA libraries don't clash with PyTorch's. GPU session.

```python
!pip install -q virtualenv  # Kaggle's Python has no working ensurepip
!python -m virtualenv -q /kaggle/temp/ocr-env
!/kaggle/temp/ocr-env/bin/pip install -q paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!/kaggle/temp/ocr-env/bin/pip install -q "paddleocr[doc-parser]>=3.6.0,<4" "sentencepiece==0.2.1" -e .  # 0.2.3 needs protobuf 7
!python scripts/download_data.py --only v2_esg_reports
!/kaggle/temp/ocr-env/bin/python scripts/ocr_pages.py --dataset v2_esg_reports
```

One worker per GPU; output in `<DATA_ROOT>/derived/ocr/<dataset>/`, resumable. `--limit N` OCRs a sample.

Save the output to a private Kaggle Dataset (`<user>/ocr-<dataset>`, with a `manifest.json`), and restore it in later sessions:

```python
!python scripts/ocr_store.py push --dataset v2_esg_reports
!python scripts/ocr_store.py pull --dataset v2_esg_reports
```

`v2_esg_reports_human` has the same reports as `v2_esg_reports`, so it reuses their OCR where the pages match (identical pixels, or a strict pixel check). `ocr_pages.py` then handles the rest:

```python
!python scripts/reuse_ocr.py                                 # configs/ocr_reuse.yaml
!python scripts/ocr_pages.py --dataset v2_esg_reports_human  # from the OCR env if pages remain
```

## Text chunks

```python
!python scripts/build_chunks.py --dataset v2_esg_reports
```

Cleans the OCR text (drops image placeholders and HTML tags, turns tables into `cell | cell` rows) and splits each page into overlapping chunks counted in the embedding model's tokens (`configs/text.yaml`). Chunks never cross pages. Output in `<DATA_ROOT>/derived/chunks/<dataset>/`.

## Text retrieval

GPU session, `%run scripts/kaggle_setup.py --qdrant`.

```python
!python scripts/embed_text.py --dataset v2_esg_reports   # chunks + queries, one worker per GPU
!python scripts/index_text.py --dataset v2_esg_reports   # Qdrant collection text_<dataset>: dense + BM25
!python scripts/eval_text.py --dataset v2_esg_reports    # NDCG@10, Recall@100 per query language
```

Pages score as their best chunk. Dense search is exact (no HNSW) so rankings are reproducible. BM25 uses the document language's stemmer (`doc_language` in `configs/datasets.yaml`), with IDF computed by Qdrant. `rrf` merges the BM25 and dense page rankings. Output in `results/text_eval/<dataset>/`, with the rankings of each method and `coverage.json` (relevant pages with almost no text). `--corpus NAME` runs the three scripts on another chunk folder, scored on the dataset's queries.

Save a collection to a private Kaggle Dataset (`<user>/qdrant-text-<corpus>`, with a `manifest.json`), and restore it in a later session with the same Qdrant version:

```python
!python scripts/qdrant_store.py push --dataset v2_esg_reports
!python scripts/qdrant_store.py pull --dataset v2_esg_reports
```

The Dataset also holds `pages.jsonl` and the query embeddings, so `eval_text.py` can run right after `pull`.

## Visual retrieval

The retriever gives each page and query one vector per token and scores pages with MaxSim. Its model card uses bf16 and FlashAttention 2, which T4 GPUs lack, so this check runs it in fp16 with SDPA on a small sample (`configs/visual.yaml`). GPU session.

```python
!python scripts/download_data.py --only v2_esg_reports
!python scripts/check_visual_model.py
```

Reports memory, speed per batch size, vectors per page, whether batching changes the embeddings, how relevant pages rank, and the match with the model's bf16 output on CPU. Output in `results/visual_check/v2_esg_reports/`.

## Bridge check: PDF text vs OCR

Dev datasets ship page images only, so dev text comes from OCR, while the product reads the PDF text layer. This check compares both on the ESG pages whose PDF can still be downloaded (`configs/bridge.yaml`).

```python
!python scripts/download_data.py --only v2_esg_reports v2_esg_reports_human
!python scripts/ocr_store.py pull --dataset v2_esg_reports
!python scripts/build_bridge.py
for source in ("ocr", "pdf"):
    !python scripts/embed_text.py --dataset v2_esg_reports --corpus v2_esg_reports-bridge-{source}
    !python scripts/index_text.py --dataset v2_esg_reports --corpus v2_esg_reports-bridge-{source}
    !python scripts/eval_text.py --dataset v2_esg_reports --corpus v2_esg_reports-bridge-{source}
```

`build_bridge.py` downloads the PDFs, matches their pages to the dataset's page images by pixel correlation, reads the text layer with pypdfium2 (OCR text where the layer has almost none, as in the product), and writes two chunk folders with the same pages. Output in `results/bridge/v2_esg_reports/`: per-PDF and per-page match tables (ids, URLs, statistics; no document text).

`python scripts/analyze_bridge.py --input results` then compares the two runs per query, with bootstrap intervals, and shows which relevant pages each text source finds, grouped by how much text OCR and the PDF have. Output in `gaps/` under the same folder.

`python scripts/check_garbled.py --input results` measures the rule that sends unusable text layers to OCR (`configs/garbled.yaml`): false alarms on real English layers and French text, and detection on clean pages corrupted in memory. Output in `results/garbled/v2_esg_reports/`.

## Dev/test overlap check

```python
!python scripts/download_data.py
!python scripts/check_overlap.py
```

Finds dev pages that also appear in test: exact and perceptual hashes give candidate pairs, nearly blank pages are skipped, the rest are verified by pixel correlation. Output in `results/overlap/`. Confirmed overlapping documents go into `configs/dev_exclusions.yaml`.
