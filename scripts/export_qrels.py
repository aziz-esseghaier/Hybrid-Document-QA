"""Save the dev datasets' relevance judgments and query languages, so saved rankings can be scored without the data.

Usage:
    python scripts/export_qrels.py [--only v2_esg_reports ...]

Writes <RESULTS_ROOT>/qrels/<dataset>.jsonl ({query_id, language, pages: {page_id: grade}}, one line per judged
query) and run_info.json. Ids and grades only, no query text. Needs download_data.py first.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import write_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.core.schemas import Benchmark
from doc_qa.evaluation.vidore import load_benchmark


def query_records(bench: Benchmark) -> list[dict[str, Any]]:
    """Group a dataset's judgments by query.

    Args:
        bench (Benchmark): Loaded dataset.

    Returns:
        list[dict[str, Any]]: {query_id, language, pages: {page_id: grade}} per judged query, sorted by id.
    """
    judged: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        judged[qrel.query_id][qrel.page_id] = qrel.score
    return [
        {"query_id": query_id, "language": bench.queries[query_id].language, "pages": dict(sorted(pages.items()))}
        for query_id, pages in sorted(judged.items())
    ]


def main() -> None:
    """Export the judgments of every dev dataset, or of the chosen ones."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--only", nargs="+", help="Dataset keys (default: every dev dataset).")
    args = parser.parse_args()

    settings = get_settings()
    specs = load_yaml(args.datasets_config)["datasets"]
    keys = args.only or [key for key, spec in specs.items() if spec["role"] == "dev"]
    results_dir = settings.results_root / "qrels"
    write_run_info(results_dir, extra={
        "datasets": {key: {"repo": specs[key]["repo"], "revision": specs[key]["revision"]} for key in keys},
    })
    for key in keys:
        records = query_records(load_benchmark(key, specs[key], settings.data_root / "raw"))
        write_jsonl(results_dir / f"{key}.jsonl", records)
        print(f"{key}: {len(records)} queries, {sum(len(r['pages']) for r in records)} judgments", flush=True)


if __name__ == "__main__":
    main()
