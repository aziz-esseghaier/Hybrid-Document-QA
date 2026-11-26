"""Report the test results next to dev and the published numbers: retrieval, answer accuracy, and a retriever check.

Usage:
    python scripts/final_report.py [--config configs/report.yaml]

Reads saved results only: the test retrieval summary (eval_fusion.py with configs/fusion_test.yaml), the test and dev
answer accuracy (answer_accuracy.py), the dev retrieval summary and text baselines, and the test judgments and
visual rankings, from which it recomputes the visual nDCG@5 of English queries to compare with the retriever's model
card. Writes <RESULTS_ROOT>/final_report/: report.md (the tables), summary.json, config.yaml, run_info.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml, result_path
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import ndcg_at_k

METHODS = {  # summary key -> column name, weakest branch first
    "bm25": "BM25", "dense": "Dense", "text": "Text RRF", "visual": "Visual", "rerank": "Reranker order",
    "best": "Pipeline",
}
TEXT_METHODS = {"bm25": "bm25", "dense": "dense", "rrf": "text"}  # text_eval.py method -> summary key


def read_json(pattern: str, **fields: str) -> dict[str, Any]:
    """Read a saved JSON result.

    Args:
        pattern (str): Repo-relative path, with {dataset} if needed.
        **fields (str): Values for the pattern's fields.

    Returns:
        dict[str, Any]: Content.
    """
    path = result_path(pattern.format(**fields))
    if not path.exists():
        raise SystemExit(f"Missing {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Format a Markdown table.

    Args:
        header (Sequence[str]): Column names.
        rows (Sequence[Sequence[Any]]): Cells, already formatted.

    Returns:
        str: The table.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def dev_text_means(datasets: Sequence[str], pattern: str) -> dict[str, float]:
    """Mean NDCG@10 of each text baseline over the dev dataset-language groups.

    Args:
        datasets (Sequence[str]): Dev dataset keys.
        pattern (str): Path of each dataset's text_eval.py metrics.

    Returns:
        dict[str, float]: {"bm25", "dense", "text": mean}.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for dataset in datasets:
        metrics = read_json(pattern, dataset=dataset)
        for method, key in TEXT_METHODS.items():
            values[key] += [m["ndcg@10"] for group, m in metrics[method].items() if group != "all"]
    return {key: round(float(np.mean(group_values)), 4) for key, group_values in values.items()}


def ndcg5_english(dataset: str, qrels_pattern: str, rankings_pattern: str) -> tuple[float, int]:
    """Mean nDCG@5 of a dataset's English queries in a saved ranking.

    Args:
        dataset (str): Dataset key.
        qrels_pattern (str): Path of the dataset's judgments (export_qrels.py).
        rankings_pattern (str): Path of the saved ranking.

    Returns:
        tuple[float, int]: Mean nDCG@5 and the number of English queries.
    """
    ranked = {
        row["query_id"]: [page for page, _ in row["pages"]]
        for row in read_jsonl([result_path(rankings_pattern.format(dataset=dataset))])
    }
    rows = [r for r in read_jsonl([result_path(qrels_pattern.format(dataset=dataset))]) if r["language"] == "en"]
    if not rows:
        raise SystemExit(f"No English judgments for {dataset}.")
    values = [ndcg_at_k(ranked.get(r["query_id"], []), r["pages"], 5) for r in rows]
    return round(float(np.mean(values)), 4), len(values)


def retrieval_section(test: dict[str, Any], dev_means: dict[str, float]) -> tuple[str, dict[str, Any]]:
    """Build the retrieval tables.

    Args:
        test (dict[str, Any]): Test retrieval summary (eval_fusion.py).
        dev_means (dict[str, float]): Dev mean NDCG@10 per method.

    Returns:
        tuple[str, dict[str, Any]]: Markdown, and the numbers for summary.json.
    """
    groups = list(test["ndcg@10"]["best"])
    rows = [[group, *(f"{test['ndcg@10'][m][group]:.3f}" for m in METHODS)] for group in groups]
    rows.append(["**Mean, test**", *(f"**{test['macro_ndcg@10'][m]:.3f}**" for m in METHODS)])
    rows.append(["Mean, dev", *(f"{dev_means[m]:.3f}" if m in dev_means else "" for m in METHODS)])
    gains = [
        [METHODS[name], f"{d['delta']:+.3f}", f"[{d['low']:+.3f}, {d['high']:+.3f}]"]
        for name, d in ((name, test["best_minus"][name]["all"]) for name in METHODS if name != "best")
    ]
    text = "\n\n".join([
        "## Retrieval\n\nNDCG@10 per dataset and query language. The pipeline is the visual ranking merged with the "
        "reranker's order of its top 20 pages.",
        markdown_table(["Group", *METHODS.values()], rows),
        "Pipeline minus each baseline, over all test queries, with 95% bootstrap intervals:",
        markdown_table(["Baseline", "Difference", "95% interval"], gains),
    ])
    numbers = {
        "test_mean": test["macro_ndcg@10"], "dev_mean": dev_means,
        "pipeline_minus": {name: test["best_minus"][name]["all"] for name in METHODS if name != "best"},
    }
    return text, numbers


def answer_section(test: dict[str, Any], dev: dict[str, Any], paper: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the answer accuracy table.

    Args:
        test (dict[str, Any]): Test answer accuracy summary (answer_accuracy.py).
        dev (dict[str, Any]): Dev answer accuracy summary.
        paper (dict[str, Any]): Published accuracies, in %.

    Returns:
        tuple[str, dict[str, Any]]: Markdown, and the numbers for summary.json.
    """
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group, row in test["groups"].items():
        dataset, language = group.split("/")
        by_dataset[dataset][language] = row

    def percent(value: float | None) -> str:
        """Format a share as a percentage."""
        return "" if value is None else f"{100 * value:.1f}"

    rows, per_dataset = [], {}
    for dataset, languages in sorted(by_dataset.items()):
        both = float(np.mean([row["accuracy"]["mean"] for row in languages.values()]))
        lenient = float(np.mean([row["correct_or_partial"] for row in languages.values()]))
        given = float(np.mean([row["relevant_page_given"] for row in languages.values()]))
        per_dataset[dataset] = {"accuracy": round(both, 4), "correct_or_partial": round(lenient, 4)}
        rows.append([
            dataset, percent(languages.get("en", {}).get("accuracy", {}).get("mean")),
            percent(languages.get("fr", {}).get("accuracy", {}).get("mean")), f"**{percent(both)}**",
            percent(lenient), percent(given), f"{paper['answer_accuracy'][dataset]:.1f}",
        ])

    def interval(summary: dict[str, Any], key: str) -> str:
        """Format a mean with its interval, in %."""
        m = summary["macro"].get(key)
        return "" if m is None else f"{100 * m['mean']:.1f} [{100 * m['low']:.1f}, {100 * m['high']:.1f}]"

    rows.append([
        "**Mean, test**", interval(test, "en"), interval(test, "fr"), f"**{interval(test, 'all')}**", "", "",
        f"**{paper['answer_accuracy_mean']:.1f}**",
    ])
    rows.append(["Mean, dev", interval(dev, "en"), interval(dev, "fr"), interval(dev, "all"), "", "", ""])
    text = "\n\n".join([
        "## Answer accuracy\n\nShare of answers the judge grades correct, in %. Both = the mean of the two "
        "languages. C or P = correct or partially correct. Relevant page = a relevant page was among the pages "
        "the answer model read. The means average the datasets, with 95% bootstrap intervals.",
        markdown_table(["Dataset", "English", "French", "Both", "C or P", "Relevant page", "Paper"], rows),
    ])
    numbers = {"per_dataset": per_dataset, "test_mean": test["macro"], "dev_mean": dev["macro"],
               "paper_mean": paper["answer_accuracy_mean"], "paper_hard": paper["answer_accuracy_hard"]}
    return text, numbers


def visual_check_section(ours: dict[str, tuple[float, int]], card: dict[str, float]) -> tuple[str, dict[str, Any]]:
    """Build the table that compares our visual retrieval with the model card.

    Args:
        ours (dict[str, tuple[float, int]]): Our English nDCG@5 and query count per dataset.
        card (dict[str, float]): The card's English nDCG@5 per dataset.

    Returns:
        tuple[str, dict[str, Any]]: Markdown, and the numbers for summary.json.
    """
    rows = [
        [dataset, count, f"{value:.3f}", f"{card[dataset]:.3f}", f"{value - card[dataset]:+.3f}"]
        for dataset, (value, count) in sorted(ours.items())
    ]
    mean_ours = float(np.mean([value for value, _ in ours.values()]))
    mean_card = float(np.mean([card[dataset] for dataset in ours]))
    rows.append(["**Mean**", "", f"{mean_ours:.3f}", f"{mean_card:.3f}", f"{mean_ours - mean_card:+.3f}"])
    text = "\n\n".join([
        "## Visual retriever check\n\nnDCG@5 of the visual ranking for English queries, next to the retriever's "
        "model card. Close values mean the retriever runs as its authors ran it.",
        markdown_table(["Dataset", "Queries", "Ours", "Model card", "Difference"], rows),
    ])
    return text, {dataset: {"ours": value, "card": card[dataset]} for dataset, (value, _) in ours.items()}


NOTES = """## How these numbers differ from the paper's

- Our pipeline answers with Qwen3-VL-8B from the top 5 pages, each as an image with its text. The paper's hybrid \
pipeline uses Gemini 3 Pro with the top 5 page images plus the top 5 texts from a text retriever.
- Gemma 4 31B judges our answers and only "correct" counts. GPT 5.2 judges the paper's. The paper doesn't say how it \
counts a partially correct answer, so the C or P column shows the most lenient reading.
- We grade a fixed random sample of 100 questions per dataset, each in English and French. The paper grades every \
question and doesn't say in which languages.
- The paper's 54.7% covers only its hard queries, which the dataset doesn't flag. The table uses its accuracy over \
all queries.
- Test page text is the text ViDoRe V3 ships with each page. Dev page text came from OCR.
"""


def main() -> None:
    """Read the saved results, build the tables and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/report.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    specs = load_yaml(Path(cfg["datasets_config"]))["datasets"]
    test_sets = [key for key, spec in specs.items() if spec["role"] == "test"]
    dev_sets = [key for key, spec in specs.items() if spec["role"] == "dev"]

    dev_retrieval = read_json(cfg["dev"]["retrieval"])
    dev_means = {**dev_text_means(dev_sets, cfg["dev"]["text"]), **dev_retrieval["macro_ndcg@10"]}
    retrieval, retrieval_numbers = retrieval_section(read_json(cfg["test"]["retrieval"]), dev_means)
    answers, answer_numbers = answer_section(
        read_json(cfg["test"]["answers"]), read_json(cfg["dev"]["answers"]), cfg["paper"]
    )
    ours = {d: ndcg5_english(d, cfg["test"]["qrels"], cfg["test"]["visual_rankings"]) for d in test_sets}
    check, check_numbers = visual_check_section(ours, cfg["model_card"]["ndcg@5_en"])

    results_dir = settings.results_root / "final_report"
    write_run_info(results_dir, extra={"datasets": test_sets})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    report = "\n\n".join(["# Test results", retrieval, answers, check, NOTES])
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    summary = {"retrieval": retrieval_numbers, "answers": answer_numbers, "visual_check": check_numbers}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
