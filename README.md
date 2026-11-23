# Document Q&A

Ask a question about English or French PDFs and get an answer that cites the pages it comes from. It handles charts, tables and scanned pages, because it reads every page as an image as well as text.

Try the live demo at https://huggingface.co/spaces/vziz/doc-qa-demo.

1. Upload one or more PDFs. The app reads the first 50 pages of each.
2. Wait until the document list shows the page count.
3. Type a question in English or French and press Ask.
4. Read the answer and its sources. Below the answer, the app shows the pages the model read, cited pages first.

Hugging Face gives each visitor 2 minutes of GPU time a day, or 5 minutes when logged in.

A visual retriever, tomoro-colqwen3-embed-4b, ranks the pages by their images. A text reranker, Qwen3-Reranker-4B, rereads the text of the best 20. Qwen3-VL-8B then answers from the top 5 pages, image and text. All the models are open source under Apache 2.0. Development uses ViDoRe V2 and the French ViDoRe V1 datasets, and the final test uses ViDoRe V3.

## Project layout

```
configs/             YAML settings: datasets at pinned revisions, models, experiments
src/doc_qa/          the Python package
  core/              settings, shared types, run records, attention for T4 GPUs
  ingest/            PDF text, OCR text cleaning, chunking
  retrieval/         visual search, reranking, text search baseline, Qdrant, fusion
  answer/            answer prompt, output checks, answer model
  evaluation/        dataset loaders, metrics, LLM judge
  pipeline.py        the full pipeline that the demo runs
scripts/             one script per step: download, OCR, embed, index, evaluate, answer, judge
demo/space/          the Hugging Face Space app
demo/modal_app.py    the same app for Modal, which needs a payment method
tests/               unit tests
results/             one folder per experiment: config, run record and scores, never document text
docker-compose.yml   Qdrant for local use
environment.yml      local conda environment
```

## How to run it

### Local setup

The local machine runs the tests and the analyses that only read saved results.

```
conda env create -f environment.yml
conda activate doc-qa
copy .env.example .env        # on Linux: cp
pytest
docker compose up -d          # Qdrant on http://localhost:6333
```

### Kaggle setup

Every step that runs a model runs on Kaggle, with two T4 GPUs. In the notebook settings, turn Internet on and add the secrets `HF_TOKEN`, `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`. The Kaggle secrets let the scripts save their outputs to private Kaggle Datasets. Start every session with this cell:

```python
!git clone https://github.com/<user>/Hybrid-Document-QA.git /kaggle/temp/Hybrid-Document-QA
%cd /kaggle/temp/Hybrid-Document-QA
!pip install -q -e .
%run scripts/kaggle_setup.py          # add --qdrant to start Qdrant
```

Run `kaggle_setup.py` with `%run` so that later cells see its settings. Data goes to `/kaggle/temp`, which Kaggle wipes when the session ends. Results go to `/kaggle/working/results`.

The commands below use the ESG reports. Any dataset key from `configs/datasets.yaml` works in place of `v2_esg_reports`.

### Data

```python
!python scripts/download_data.py --only v2_esg_reports   # without --only: every dev and test dataset
!python scripts/export_qrels.py                          # relevance judgments for the local analyses
!python scripts/check_overlap.py                         # needs every dataset; checks that no dev page is in the test set
```

### Page text

The dev datasets ship page images only, so PaddleOCR-VL reads every page. It gets its own environment because Paddle's CUDA libraries clash with PyTorch's.

```python
!pip install -q virtualenv
!python -m virtualenv -q /kaggle/temp/ocr-env
!/kaggle/temp/ocr-env/bin/pip install -q paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
!/kaggle/temp/ocr-env/bin/pip install -q "paddleocr[doc-parser]>=3.6.0,<4" "sentencepiece==0.2.1" -e .
!/kaggle/temp/ocr-env/bin/python scripts/ocr_pages.py --dataset v2_esg_reports
!python scripts/ocr_store.py push --dataset v2_esg_reports    # saves the text to a private Kaggle Dataset
```

Later sessions restore the text instead of running OCR again, then split it into pages and chunks. The reranking and answering steps need this.

```python
!python scripts/ocr_store.py pull --dataset v2_esg_reports
!python scripts/build_chunks.py --dataset v2_esg_reports
```

### Visual search

```python
%run scripts/kaggle_setup.py --qdrant
!python scripts/embed_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports
!python scripts/index_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports
!python scripts/eval_visual.py --config configs/visual_tomoro.yaml --dataset v2_esg_reports
!python scripts/qdrant_store.py push --branch visual --config configs/visual_tomoro.yaml --dataset v2_esg_reports --clean
```

`eval_visual.py` reports NDCG@10 and Recall@100 per query language. `qdrant_store.py push` saves the Qdrant collection to a private Kaggle Dataset, and `pull` restores it in a later session.

### Reranking

The reranker runs on Kaggle and reorders the top visual pages by reading their text:

```python
!python scripts/rerank_pages.py --config configs/rerank_tomoro.yaml --dataset v2_esg_reports
```

`eval_fusion.py` then scores the final ranking locally, from the saved rankings:

```
python scripts/eval_fusion.py --config configs/fusion_rerank_tomoro_top20.yaml
```

### Answering

The answer model reads the top 5 pages, image and text, and replies with an answer and the pages it cites.

```python
!python scripts/check_answer_model.py      # memory and speed on a few questions
!python scripts/answer_questions.py --dataset v2_esg_reports --out /kaggle/working/answers
```

The answers contain document text, so they go to `--out` and stay out of the repo. `results/` gets only ids, cited pages and timings.

### Grading answers

Gemma 4 31B grades each answer against the reference answers as correct, partially correct or incorrect. Accuracy is the share it grades correct.

```python
!pip install -q bitsandbytes==0.50.2
!python scripts/judge_answers.py --judge gemma4_31b --dataset v2_esg_reports --answers ANSWERS_DIR --out /kaggle/working/judgments
```

Then, locally, compare the answer settings with 95% intervals:

```
python scripts/compare_settings.py --dataset v2_esg_reports
```

We chose this judge with `label_sheet.py` and `compare_judges.py`, by its agreement with reference grades on 50 answers.

### Demo

```
hf auth login                                                # needs a token with write access
python scripts/push_space.py --space <user>/doc-qa-demo --public
```

Select ZeroGPU in the Space settings once. The Space then installs its packages, downloads the models and starts.

### Baselines and checks

These scripts build the baselines that the results report next to the final pipeline, and check the data.

- `embed_text.py`, `index_text.py` and `eval_text.py` build the text search baseline, with dense and BM25 search. `eval_fusion.py --config configs/fusion_tomoro.yaml` tests merging it with visual search.
- The visual scripts with `configs/visual.yaml` run nemotron-colembed-vl-4b-v2, the earlier visual retriever. Its license forbids commercial use, so it is only a reference.
- `build_bridge.py`, `analyze_bridge.py` and `check_garbled.py` compare text from the PDF layer with OCR text.
