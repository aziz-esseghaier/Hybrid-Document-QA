# Document Q&A

Question answering over English and French PDFs (text, tables, charts, scans) with page citations. Hybrid retrieval: a text branch and a visual branch, fused and reranked. Open-source models only.

Evaluation: [ViDoRe V3](https://huggingface.co/vidore) (test), ViDoRe V2 + French ViDoRe V1 (dev).

## Layout

```
configs/      datasets (pinned revisions), experiment configs, dev exclusions
src/doc_qa/   package: core/ (settings, schemas, run metadata), evaluation/ (loaders, checks)
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

Notebook settings: Internet on, environment pinned. Optional secret: `HF_TOKEN`.

```python
!git clone https://github.com/<user>/Hybrid-Document-QA.git /kaggle/temp/Hybrid-Document-QA
%cd /kaggle/temp/Hybrid-Document-QA
!pip install -q -e .
%run scripts/kaggle_setup.py
```

`kaggle_setup.py` must run with `%run` so its environment variables reach later cells. Data goes to `/kaggle/temp` (wiped at session end), results to `/kaggle/working/results`.

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

## Dev/test overlap check

```python
!python scripts/download_data.py
!python scripts/check_overlap.py
```

Finds dev pages that also appear in test: exact and perceptual hashes give candidate pairs, nearly blank pages are skipped, the rest are verified by pixel correlation. Output in `results/overlap/`. Confirmed overlapping documents go into `configs/dev_exclusions.yaml`.
