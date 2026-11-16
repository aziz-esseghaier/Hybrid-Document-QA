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

## Dev/test overlap check

```python
!python scripts/download_data.py
!python scripts/check_overlap.py
```

Finds dev pages that also appear in test: exact and perceptual hashes give candidate pairs, nearly blank pages are skipped, the rest are verified by pixel correlation. Output in `results/overlap/`. Confirmed overlapping documents go into `configs/dev_exclusions.yaml`.
