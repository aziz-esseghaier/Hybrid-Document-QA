"""Explain the bridge gap: which relevant pages OCR text loses against PDF text.

Usage:
    python scripts/analyze_bridge.py [--config configs/bridge.yaml] [--input results]

Reads a finished bridge run from --input (default: RESULTS_ROOT): bridge/<dataset>/pages.csv and the rankings in
text_eval/<dataset>-bridge-{ocr,pdf}/. Needs the dataset's qrels. Translations of a question share their relevant
pages, so intervals resample those groups rather than single queries.

Writes <RESULTS_ROOT>/bridge/<dataset>/gaps/: summary.json, relevant_pages.csv (one row per method, query and
relevant page; ids and statistics only), config.yaml, run_info.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_csv
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import cluster_bootstrap_ci, ndcg_at_k, recall_at_k
from doc_qa.evaluation.vidore import load_benchmark

METHODS = ("bm25", "dense", "rrf")
SOURCES = ("ocr", "pdf")
TEXT_GROUPS = ("same_text", "ocr_lacks_most", "ocr_lacks_some", "similar", "pdf_lacks_some")
METRICS = {
    "ndcg@10": lambda ranking, relevance: ndcg_at_k(ranking, relevance, 10),
    "recall@100": lambda ranking, relevance: recall_at_k(ranking, relevance, 100),
}


def text_group(page: dict[str, str]) -> str:
    """Classify a page by how much text each source has.

    Args:
        page (dict[str, str]): Row of pages.csv.

    Returns:
        str: One of TEXT_GROUPS; "same_text" means the PDF side fell back to the OCR text.
    """
    if page["fallback"] == "True":
        return "same_text"
    layer, ocr = int(page["layer_word_chars"]), int(page["ocr_word_chars"])
    if layer >= 2 * ocr:
        return "ocr_lacks_most"
    if layer >= 1.25 * ocr:
        return "ocr_lacks_some"
    if ocr >= 1.25 * layer:
        return "pdf_lacks_some"
    return "similar"


def relevant_rows(
    method: str,
    rankings: dict[str, dict[str, list[str]]],
    qrels: dict[str, dict[str, int]],
    languages: dict[str, str],
    pages: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """List every relevant page of every query with its rank under both text sources.

    Args:
        method (str): Retrieval method.
        rankings (dict[str, dict[str, list[str]]]): {source: {query id: ranked page ids}}.
        qrels (dict[str, dict[str, int]]): Relevance per query id, within the corpus.
        languages (dict[str, str]): Language per query id.
        pages (dict[str, dict[str, str]]): Rows of pages.csv per page id.

    Returns:
        list[dict[str, Any]]: One row per (query, relevant page); a rank is None if the page wasn't retrieved.
    """
    rows = []
    for query_id, relevance in sorted(qrels.items()):
        ranks = {s: {p: i + 1 for i, p in enumerate(rankings[s].get(query_id, []))} for s in SOURCES}
        for page_id in sorted(relevance):
            page = pages[page_id]
            rows.append({
                "method": method,
                "query_id": query_id,
                "language": languages[query_id],
                "page_id": page_id,
                "text_group": text_group(page),
                "ocr_rank": ranks["ocr"].get(page_id),
                "pdf_rank": ranks["pdf"].get(page_id),
                "layer_word_chars": int(page["layer_word_chars"]),
                "ocr_word_chars": int(page["ocr_word_chars"]),
                "token_f1": float(page["token_f1"]),
                "url": page["url"],
                "pdf_page": int(page["pdf_page"]),
            })
    return rows


def hit_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize how often relevant pages are found under each text source.

    Args:
        rows (list[dict[str, Any]]): Output of `relevant_rows`.

    Returns:
        dict[str, float]: Pair count, hit rates at 10 and 100, and pairs only one source finds in the top 100.
    """
    def found(row: dict[str, Any], source: str, k: int) -> bool:
        rank = row[f"{source}_rank"]
        return rank is not None and rank <= k

    summary: dict[str, float] = {"pairs": len(rows)}
    for k in (10, 100):
        for source in SOURCES:
            hits = sum(found(r, source, k) for r in rows)
            summary[f"{source}_hit@{k}"] = round(hits / len(rows), 4) if rows else 0.0
    summary["only_pdf@100"] = sum(found(r, "pdf", 100) and not found(r, "ocr", 100) for r in rows)
    summary["only_ocr@100"] = sum(found(r, "ocr", 100) and not found(r, "pdf", 100) for r in rows)
    return summary


def paired_scores(
    rankings: dict[str, dict[str, list[str]]],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Compare both text sources on the same queries.

    Args:
        rankings (dict[str, dict[str, list[str]]]): {source: {query id: ranked page ids}}.
        qrels (dict[str, dict[str, int]]): Relevance per query id.
        query_ids (list[str]): Queries to score.

    Returns:
        dict[str, dict[str, float]]: {metric: {"ocr", "pdf", "delta", "ci_low", "ci_high"}}; delta is PDF minus OCR.
    """
    clusters = [tuple(sorted(qrels[q])) for q in query_ids]
    result = {}
    for name, metric in METRICS.items():
        scores = {s: [metric(rankings[s].get(q, []), qrels[q]) for q in query_ids] for s in SOURCES}
        deltas = [p - o for p, o in zip(scores["pdf"], scores["ocr"])]
        delta, low, high = cluster_bootstrap_ci(deltas, clusters)
        result[name] = {
            "ocr": round(sum(scores["ocr"]) / len(query_ids), 4),
            "pdf": round(sum(scores["pdf"]) / len(query_ids), 4),
            "delta": round(delta, 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
        }
    return result


def analyze(
    input_dir: Path, dataset: str, qrels: dict[str, dict[str, int]], languages: dict[str, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare OCR and PDF retrieval over the bridge pages.

    Args:
        input_dir (Path): Results folder of the bridge run.
        dataset (str): Dataset key.
        qrels (dict[str, dict[str, int]]): Relevance per query id, for the whole dataset.
        languages (dict[str, str]): Language per query id.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: Summary, and the rows of all methods.
    """
    with (input_dir / "bridge" / dataset / "pages.csv").open(encoding="utf-8", newline="") as f:
        pages = {r["page_id"]: r for r in csv.DictReader(f)}
    corpus_qrels = {}
    for query_id, relevance in qrels.items():
        kept = {p: g for p, g in relevance.items() if p in pages and g > 0}
        if kept:
            corpus_qrels[query_id] = kept

    summary: dict[str, Any] = {
        "queries": len(corpus_qrels),
        "question_groups": len({tuple(sorted(r)) for r in corpus_qrels.values()}),
        "methods": {},
    }
    groups = {"all": sorted(corpus_qrels)}
    for query_id in sorted(corpus_qrels):
        groups.setdefault(languages[query_id], []).append(query_id)

    all_rows = []
    for method in METHODS:
        rankings, stored = {}, {}
        for source in SOURCES:
            run_dir = input_dir / "text_eval" / f"{dataset}-bridge-{source}"
            lines = read_jsonl([run_dir / f"rankings_{method}.jsonl"])
            rankings[source] = {r["query_id"]: [p for p, _ in r["pages"]] for r in lines}
            stored[source] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))[method]

        paired = {group: paired_scores(rankings, corpus_qrels, ids) for group, ids in groups.items()}
        for group, scores in paired.items():
            for metric, values in scores.items():
                for source in SOURCES:
                    if abs(values[source] - stored[source][group][metric]) > 1e-3:
                        raise SystemExit(f"{method}/{source}/{group}/{metric} doesn't match metrics.json.")

        rows = relevant_rows(method, rankings, corpus_qrels, languages, pages)
        all_rows.extend(rows)
        by_group = {g: hit_rates([r for r in rows if r["text_group"] == g]) for g in TEXT_GROUPS}
        summary["methods"][method] = {"scores": paired, "all_pairs": hit_rates(rows), "by_text_group": by_group}
    return summary, all_rows


def main() -> None:
    """Analyze the bridge run and write the summary and per-page table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/bridge.yaml"))
    parser.add_argument("--input", type=Path, help="Results folder of the bridge run (default: RESULTS_ROOT).")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    specs = load_yaml(Path(load_yaml(Path(cfg["text_config"]))["datasets_config"]))["datasets"]
    dataset = cfg["dataset"]
    input_dir = args.input or settings.results_root

    bench = load_benchmark(dataset, specs[dataset], settings.data_root / "raw")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        qrels[qrel.query_id][qrel.page_id] = qrel.score
    languages = {q: bench.queries[q].language for q in qrels}
    summary, rows = analyze(input_dir, dataset, dict(qrels), languages)

    source_run = json.loads(
        (input_dir / "text_eval" / f"{dataset}-bridge-pdf" / "run_info.json").read_text(encoding="utf-8")
    )
    results_dir = settings.results_root / "bridge" / dataset / "gaps"
    write_run_info(results_dir, extra={
        "dataset": {"key": dataset, "repo": specs[dataset]["repo"], "revision": specs[dataset]["revision"]},
        "bridge_run_commit": source_run["git_commit"],
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(results_dir / "relevant_pages.csv", rows)

    print(f"{summary['queries']} queries, {summary['question_groups']} question groups")
    print(f"{'method':<6} {'lang':<4} {'metric':<10} {'ocr':>6} {'pdf':>6} {'delta':>7}  95% interval")
    for method, result in summary["methods"].items():
        for group, scores in result["scores"].items():
            for metric, s in scores.items():
                print(f"{method:<6} {group:<4} {metric:<10} {s['ocr']:>6.3f} {s['pdf']:>6.3f} {s['delta']:>+7.3f}"
                      f"  [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
    print(f"\n{'method':<6} {'text group':<15} {'pairs':>5} {'ocr@100':>8} {'pdf@100':>8} {'only pdf':>8} {'only ocr':>8}")
    for method, result in summary["methods"].items():
        for group, s in result["by_text_group"].items():
            print(f"{method:<6} {group:<15} {s['pairs']:>5} {s['ocr_hit@100']:>8.3f} {s['pdf_hit@100']:>8.3f}"
                  f" {s['only_pdf@100']:>8} {s['only_ocr@100']:>8}")


if __name__ == "__main__":
    main()
