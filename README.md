# Document Q&A

Question answering over English and French PDFs (text, tables, charts, scans) with page citations. Hybrid retrieval: a visual retriever finds candidate pages, and a text reranker reorders them by reading their text. Open-source models only.

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

A Qdrant point holds at most 409 vectors of this size and a page has about 764, so each page's most similar vectors are merged (Ward clustering) down to half. GPU session, `%run scripts/kaggle_setup.py --qdrant`.

```python
!python scripts/embed_visual.py --dataset v2_esg_reports  # every vector, then merged; one worker per GPU, resumable
!python scripts/index_visual.py --dataset v2_esg_reports  # Qdrant collection visual_<dataset>: MaxSim, float16, no HNSW
!python scripts/eval_visual.py --dataset v2_esg_reports   # NDCG@10, Recall@100 per query language
```

`eval_visual.py` compares three exact rankings: all vectors and merged vectors, computed locally, and merged vectors searched in Qdrant. This shows what merging costs and whether Qdrant matches. Output in `results/visual_eval/<dataset>/`, with the search time in `summary.json`.

Save a collection to a private Kaggle Dataset (`<user>/qdrant-visual-<dataset>`: snapshot, query embeddings, `manifest.json`), and restore it in a later session:

```python
!python scripts/qdrant_store.py push --branch visual --dataset v2_esg_reports --clean
!python scripts/qdrant_store.py pull --branch visual --dataset v2_esg_reports
```

`--clean` then deletes the collection and the dataset's vectors (about 6 GB per 1,000 pages), so every dev dataset fits on the disk of one session. A restored collection can be searched right away; `eval_visual.py` also needs the local vectors, so it needs `embed_visual.py` again.

nemotron's license forbids commercial use. `configs/visual_tomoro.yaml` runs the same scripts with tomoro-colqwen3-embed-4b (Apache 2.0). Its pages have at most about 1,300 vectors of 320 dimensions, so each fits in one Qdrant point without merging (`pool_factor: 1`). Results go to `results/visual_tomoro_check/`, `visual_tomoro_embeddings/` and `visual_tomoro_eval/`. `--local-only` ranks without Qdrant, which is enough to compare models:

```python
!python scripts/check_visual_model.py --config configs/visual_tomoro.yaml
!python scripts/embed_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports
!python scripts/eval_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports --local-only
```

The config's `name` also names the Qdrant collection (`visual_tomoro_<dataset>`), the index results (`results/visual_tomoro_index/`) and the Kaggle Dataset (`<user>/qdrant-visual-tomoro-<dataset>`), so the two retrievers never overwrite each other. `--results-name` keeps an earlier evaluation in place:

```python
!python scripts/index_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports
!python scripts/eval_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports --results-name visual_tomoro_eval_qdrant
!python scripts/qdrant_store.py push --branch visual --config configs/visual_tomoro.yaml --dataset v2_esg_reports --clean
```

## Fusion and reranking

Fusion works on the saved rankings, so it runs locally once the dev judgments are exported (Kaggle, after `download_data.py`):

```python
!python scripts/export_qrels.py   # results/qrels/<dataset>.jsonl: ids and grades, no text
```

```
python scripts/eval_fusion.py     # local, configs/fusion.yaml
```

`eval_fusion.py` merges the visual, dense and BM25 rankings with weighted reciprocal rank fusion. It scores every weight combination in the grid (optionally without BM25 when the query and document languages differ) by mean NDCG@10 over the dataset-language groups, checks the choice by leaving each dataset out in turn, and compares the best setting with each branch alone (bootstrap intervals). It also recomputes the single-branch scores from the saved rankings and checks that they match their own evaluations. Output in `results/fusion/`, with the fused rankings per dataset.

The reranker reads a question and a page's OCR text and answers "yes" or "no"; the page scores as how much the model prefers "yes". Check it on ESG first (GPU session, both GPUs):

```python
!python scripts/ocr_store.py pull --dataset v2_esg_reports
!python scripts/build_chunks.py --dataset v2_esg_reports
!python scripts/check_reranker.py   # configs/rerank.yaml
```

Reports memory, speed per batch size, prompt lengths, whether batching changes the scores, how the visual ranking's top pages move, the match with fp32, and the time to rerank all dev queries. Output in `results/rerank_check/v2_esg_reports/`.

Rerank the top 30 pages of the saved visual ranking for every query of a dataset (GPU session, one worker per GPU, resumable):

```python
!python scripts/rerank_pages.py --dataset v2_esg_reports   # after ocr_store.py pull and build_chunks.py
```

Output in `results/rerank/<dataset>/`: the candidates in reranker order, with scores. Then, locally, `python scripts/eval_fusion.py --config configs/fusion_rerank.yaml` scores the reranker's order alone and blended with the visual ranking (output in `results/fusion_rerank/`).

The same steps with the Tomoro visual ranking use `configs/fusion_tomoro.yaml`, `configs/rerank_tomoro.yaml` (`rerank_pages.py --config configs/rerank_tomoro.yaml`) and `configs/fusion_rerank_tomoro.yaml`. Each config's name sets its results folder.

The retrieval setting chosen on dev is `configs/fusion_rerank_tomoro_top20.yaml`: the reranker reorders the visual ranking's top 20 pages, and its order is merged with the visual ranking at equal weight. Merging in text search (dense and BM25) scored lower on dev, so text search is kept only as an evaluation baseline.

## Answering

The answer model (Qwen3-VL-8B, `configs/answer.yaml`) reads the top pages of the chosen ranking, each labelled `[Page i]` with its image and text, and replies in JSON: `{"answer", "pages", "found"}`. Replies that break the rules (bad JSON, unknown labels, a found answer without a cited page) are sent back with the reason, up to `retries` times. Check it on ESG first (GPU session, split over both GPUs):

```python
!python scripts/ocr_store.py pull --dataset v2_esg_reports
!python scripts/build_chunks.py --dataset v2_esg_reports
!python scripts/check_answer_model.py   # configs/answer.yaml
```

Reports memory, prompt length, time per question, retries, cited pages and NaN/inf scores for each number of pages and image size. Output in `results/answer_check/v2_esg_reports/` (no document text); the replies go to the log and the data folder.

Answer every question that has a reference answer, under each setting in the config (`pages` x `page_tokens`), resumable:

```python
!python scripts/answer_questions.py --dataset v2_esg_reports --out /kaggle/working/answers
```

`results/answer_answers/<dataset>/` gets ids, cited pages, cost and a summary per setting. The answers contain document text, so they go to `--out` only (kept with the notebook output, never committed).

Answers are graded by an LLM judge against the reference answers: Correct, Partially correct or Incorrect; accuracy is the share graded Correct. The judge is chosen once, by agreement with hand grades on a sample, then never changed (`configs/judges.yaml`: 4-bit Gemma 4 31B and Mistral Small 3.2):

```
python scripts/label_sheet.py make --answers ANSWERS_DIR --out SHEET.md   # outside the repo; grade C/P/I by hand
python scripts/label_sheet.py ids --sheet SHEET.md --out results/human_labels/v2_esg_reports/sample.jsonl
python scripts/judge_answers.py --judge gemma4_31b --dataset v2_esg_reports --answers ANSWERS_DIR --out TEXT_DIR --only results/human_labels/v2_esg_reports/sample.jsonl
python scripts/label_sheet.py read --sheet SHEET.md --out results/human_labels/v2_esg_reports/labels.jsonl
python scripts/compare_judges.py --dataset v2_esg_reports --judges gemma4_31b mistral_small_32
```

The judge is `gemma4_31b` (`judge:` in `configs/judges.yaml`; comparison in `results/judge_comparison/`). Once it has graded every answer, `python scripts/compare_settings.py --dataset v2_esg_reports` compares the settings: accuracy per setting and language, and each setting's paired difference from the best, with 95% intervals that resample questions (translations together).

## Web demo

A page where you upload PDFs, ask a question, and get an answer with the pages it cites. The pipeline is `doc_qa.pipeline` (`configs/demo.yaml`): visual search over the uploaded pages, the reranker on the top 20, the two orders merged by reciprocal rank fusion, and the answer model on the top 5 pages. All three models (about 35 GB in fp16) share one GPU of at least 48 GB. Pages without a text layer are read from their image only (no OCR yet), and only the first 50 pages of a PDF are indexed.

It runs as a [Hugging Face ZeroGPU Space](https://huggingface.co/docs/hub/en/spaces-zerogpu) (`demo/space/`), free for accounts older than 30 days with a verified email: the GPU is attached only while a request runs, and each visitor gets a few minutes of GPU time a day. To publish it:

```
hf auth login                                          # a token with write access
python scripts/push_space.py --space <user>/doc-qa-demo  # private; add --public to share it
```

Then select ZeroGPU in the Space settings once. The Space builds and starts by itself (the models download from the Hub on first start).

`demo/modal_app.py` runs the same pipeline on [Modal](https://modal.com) (one L40S, scale to zero), which needs a payment method on the Modal account.

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
