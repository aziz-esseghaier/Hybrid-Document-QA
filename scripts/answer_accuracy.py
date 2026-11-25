"""Answer accuracy with the fixed judge and the chosen setting: per dataset and language, and overall.

Usage:
    python scripts/answer_accuracy.py [--role dev|test] [--datasets v2_esg_reports ...]
        [--judges-config configs/judges.yaml] [--answer-config configs/answer.yaml]

Accuracy is the share of questions graded correct. The overall score is the mean over dataset-language groups, so each
group weighs the same, with a 95% bootstrap interval that resamples questions within each dataset, keeping a
question's translations together. Accuracy is also split by whether a relevant page was among the pages the answer
model read, to separate retrieval misses from answering mistakes. Reads the judge's verdicts (judge_answers.py), the
answer records (answer_questions.py) and the judgments (export_qrels.py) of one dataset role, dev by default. Writes
<RESULTS_ROOT>/answer_accuracy/summary.json for dev, answer_accuracy_test/ for test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from doc_qa.core.config import get_settings, load_yaml, role_name
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import cluster_bootstrap_ci


def macro_interval(
    rows: Sequence[dict[str, Any]], groups: Sequence[str], resamples: int, seed: int
) -> dict[str, float]:
    """Mean over groups of the per-group mean, with a bootstrap interval resampling clusters within each dataset.

    Args:
        rows (Sequence[dict[str, Any]]): One per question: {dataset, group, cluster, value}.
        groups (Sequence[str]): Groups to average, e.g. every "<dataset>/<language>".
        resamples (int): Bootstrap samples.
        seed (int): Random seed.

    Returns:
        dict[str, float]: {mean, low, high}, rounded to 4 decimals.
    """
    rng = np.random.default_rng(seed)
    point, resampled = [], []
    for dataset in sorted({r["dataset"] for r in rows if r["group"] in groups}):
        subset = [r for r in rows if r["dataset"] == dataset]
        clusters = sorted({r["cluster"] for r in subset}, key=repr)
        index = {cluster: i for i, cluster in enumerate(clusters)}
        picks = rng.integers(len(clusters), size=(resamples, len(clusters)))  # shared by the dataset's languages
        for group in sorted({r["group"] for r in subset} & set(groups)):
            sums, counts = np.zeros(len(clusters)), np.zeros(len(clusters))
            for r in subset:
                if r["group"] == group:
                    sums[index[r["cluster"]]] += r["value"]
                    counts[index[r["cluster"]]] += 1
            point.append(sums.sum() / counts.sum())
            drawn_sums, drawn_counts = sums[picks].sum(axis=1), counts[picks].sum(axis=1)
            resampled.append(np.where(drawn_counts > 0, drawn_sums / np.maximum(drawn_counts, 1), np.nan))
    low, high = np.nanpercentile(np.nanmean(np.vstack(resampled), axis=0), [2.5, 97.5])
    return {"mean": round(float(np.mean(point)), 4), "low": round(float(low), 4), "high": round(float(high), 4)}


def group_summary(rows: Sequence[dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    """Summarize one dataset-language group.

    Args:
        rows (Sequence[dict[str, Any]]): The group's questions: {cluster, value, lenient, relevant_given}.
        resamples (int): Bootstrap samples.
        seed (int): Random seed.

    Returns:
        dict[str, Any]: Accuracy with its interval, correct-or-partial share, and accuracy with and without a relevant
            page given.
    """
    values, clusters = [r["value"] for r in rows], [r["cluster"] for r in rows]
    mean, low, high = cluster_bootstrap_ci(values, clusters, resamples, seed=seed)
    given = [r for r in rows if r["relevant_given"]]
    missed = [r for r in rows if not r["relevant_given"]]
    return {
        "questions": len(rows),
        "accuracy": {"mean": round(mean, 4), "low": round(low, 4), "high": round(high, 4)},
        "correct_or_partial": round(float(np.mean([r["lenient"] for r in rows])), 4),
        "relevant_page_given": round(len(given) / len(rows), 4),
        "accuracy_when_relevant_given": round(float(np.mean([r["value"] for r in given])), 4) if given else None,
        "accuracy_when_none_given": round(float(np.mean([r["value"] for r in missed])), 4) if missed else None,
    }


def load_rows(
    results_root: Path, dataset: str, judge: str, setting: str, qrels: str = "qrels"
) -> list[dict[str, Any]]:
    """Join a dataset's verdicts, answer records and judgments into one row per question.

    Args:
        results_root (Path): Results folder.
        dataset (str): Dataset key.
        judge (str): Judge name.
        setting (str): Answer setting, e.g. "5x1280".
        qrels (str): Folder of the judgments, "qrels" for dev or "qrels_test".

    Returns:
        list[dict[str, Any]]: {dataset, group, language, cluster, value, lenient, relevant_given} per judged question.
    """
    verdicts = read_jsonl([results_root / f"judge_{judge}" / dataset / f"verdicts_{setting}.jsonl"])
    given = {
        r["query_id"]: r["given"]
        for r in read_jsonl([results_root / "answer_answers" / dataset / f"records_{setting}.jsonl"])
    }
    relevance = {r["query_id"]: r["pages"] for r in read_jsonl([results_root / qrels / f"{dataset}.jsonl"])}
    rows = []
    for verdict in verdicts:
        query_id = verdict["query_id"]
        relevant = {page for page, grade in relevance[query_id].items() if grade > 0}
        rows.append({
            "dataset": dataset,
            "group": f"{dataset}/{verdict['language']}",
            "language": verdict["language"],
            "cluster": (dataset, tuple(sorted(relevant))),  # translations share their relevant pages
            "value": float(verdict["label"] == "correct"),
            "lenient": verdict["label"] in ("correct", "partially correct"),
            "relevant_given": bool(relevant & set(given.get(query_id, []))),
        })
    return rows


def main() -> None:
    """Join the verdicts, compute the accuracies and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["dev", "test"], default="dev")
    parser.add_argument("--datasets", nargs="+", help="Default: every dataset of the role graded for the setting.")
    parser.add_argument("--judges-config", type=Path, default=Path("configs/judges.yaml"))
    parser.add_argument("--answer-config", type=Path, default=Path("configs/answer.yaml"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    judge = load_yaml(args.judges_config)["judge"]
    answer_cfg = load_yaml(args.answer_config)
    specs = load_yaml(Path(answer_cfg["datasets_config"]))["datasets"]
    setting = f"{answer_cfg['pages'][0]}x{answer_cfg['page_tokens'][0]}"
    judge_dir = settings.results_root / f"judge_{judge}"
    judged = sorted(path.parent.name for path in judge_dir.glob(f"*/verdicts_{setting}.jsonl"))
    datasets = args.datasets or [d for d in judged if d in specs and specs[d]["role"] == args.role]
    if not datasets:
        raise SystemExit(f"No {args.role} verdicts of {judge} for setting {setting} (run judge_answers.py).")
    if any(specs[d]["role"] != args.role for d in datasets):
        raise SystemExit(f"Only {args.role} datasets can be summarized together.")

    qrels = role_name("qrels", args.role)
    rows = [row for dataset in datasets for row in load_rows(settings.results_root, dataset, judge, setting, qrels)]
    groups = sorted({r["group"] for r in rows})
    summary: dict[str, Any] = {
        "role": args.role, "judge": judge, "setting": setting, "datasets": datasets, "questions": len(rows),
    }
    summary["groups"] = {
        group: group_summary([r for r in rows if r["group"] == group], args.resamples, args.seed) for group in groups
    }
    summary["macro"] = {"all": macro_interval(rows, groups, args.resamples, args.seed)}
    for language in sorted({r["language"] for r in rows}):
        language_groups = [g for g in groups if g.endswith(f"/{language}")]
        summary["macro"][language] = macro_interval(rows, language_groups, args.resamples, args.seed)
    summary["micro_accuracy"] = round(float(np.mean([r["value"] for r in rows])), 4)

    results_dir = settings.results_root / role_name("answer_accuracy", args.role)
    write_run_info(results_dir, seed=args.seed, extra={"judge": judge, "setting": setting, "datasets": datasets})
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{args.role}: judge {judge}, setting {setting}, {len(rows)} questions")
    header = f"{'group':<30}{'n':>5}{'accuracy [95%]':>24}{'C or P':>8}{'rel. given':>11}"
    print(header + f"{'acc | given':>12}{'acc | none':>11}")
    for group, row in summary["groups"].items():
        acc = row["accuracy"]
        given, none = (
            float("nan") if v is None else v
            for v in (row["accuracy_when_relevant_given"], row["accuracy_when_none_given"])
        )
        print(f"{group:<30}{row['questions']:>5}{acc['mean']:>9.3f} [{acc['low']:.3f}, {acc['high']:.3f}]"
              f"{row['correct_or_partial']:>8.3f}{row['relevant_page_given']:>11.3f}{given:>12.3f}{none:>11.3f}")
    for name, macro in summary["macro"].items():
        print(f"{'mean of groups, ' + name:<35}{macro['mean']:>9.3f} [{macro['low']:.3f}, {macro['high']:.3f}]")


if __name__ == "__main__":
    main()
