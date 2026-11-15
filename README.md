# Document Q&A

Answers questions about English and French PDFs, including charts, tables and scanned pages, and cites the pages each answer comes from. Retrieval is hybrid: a text branch and a visual branch, merged and reranked. Open-source models only.

Evaluated on [ViDoRe V3](https://huggingface.co/vidore) (test), with ViDoRe V2 and the French ViDoRe V1 datasets as dev.

## Layout

```
configs/          datasets (pinned revisions), experiments, dev exclusions
src/doc_qa/       the package: core/ (settings, schemas, run records), evaluation/ (loaders, checks)
scripts/          runnable scripts
tests/            unit tests
results/          one folder per experiment (large outputs are not tracked)
```

## Run on Kaggle

All data and models run on Kaggle. In the notebook settings: **Internet on**, **Environment: pin to original environment**. Optional: add an `HF_TOKEN` secret (Add-ons > Secrets); the ViDoRe datasets are public, so M0 doesn't need it.

One cell sets up the session:

```python
!git clone https://github.com/<you>/Hybrid-Document-QA.git /kaggle/temp/Hybrid-Document-QA
%cd /kaggle/temp/Hybrid-Document-QA
!pip install -q -e .
%run scripts/kaggle_setup.py
```

`kaggle_setup.py` must run with `%run`, so the variables it sets reach later commands. The repo is cloned to `/kaggle/temp` (wiped at session end); results go to `/kaggle/working/results` (kept as notebook output when you save a version).

### Overlap check (M0)

No GPU needed: a CPU session is enough.

```python
!python scripts/download_data.py
!python scripts/check_overlap.py --config configs/m0_overlap.yaml
```

Then download `/kaggle/working/results/m0_overlap/` and review `report.md` and `artifacts/pairs/`. Copy the confirmed document ids from `proposed_exclusions.yaml` into `configs/dev_exclusions.yaml`.

## Local setup

For editing and unit tests, in a conda environment (packages come from `pyproject.toml`):

```
conda env create -f environment.yml    # after a dependency change: conda env update -f environment.yml --prune
conda activate doc-qa
copy .env.example .env          # Linux: cp .env.example .env
pytest
```

Local Qdrant: `docker compose up -d` (REST on http://localhost:6333).
