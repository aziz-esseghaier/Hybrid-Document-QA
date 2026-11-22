"""Compare answer settings (pages given, image size) by judged accuracy, with bootstrap intervals.

Usage:
    python scripts/compare_settings.py --dataset v2_esg_reports [--judges-config configs/judges.yaml]

Reads the fixed judge's verdicts for every setting (judge_answers.py), the judgments' question languages, the dev
judgments for clusters (export_qrels.py) and the answer cost (answer_questions.py). Accuracy is the share graded
correct. Every question is answered under every setting, so differences are paired; intervals resample questions,
keeping a question's translations together (they share relevant pages). Writes
<RESULTS_ROOT>/answer_settings/<dataset>/summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Hashable

import numpy as np

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import cluster_bootstrap_ci


def interval(values: list[float], clusters: list[Hashable], resamples: int, seed: int) -> dict[str, float]:
    """Mean with a 95% cluster bootstrap interval.

    Args:
        values (list[float]): One value per question.
        clusters (list[Hashable]): Cluster per question.
        resamples (int): Bootstrap samples.
        seed (int): Random seed.

    Returns:
        dict[str, float]: {mean, low, high}, rounded to 4 decimals.
    """
    mean, low, high = cluster_bootstrap_ci(values, clusters, resamples, seed=seed)
    return {"mean": round(mean, 4), "low": round(low, 4), "high": round(high, 4)}


def compare(
    labels: dict[str, dict[str, str | None]],
    languages: dict[str, str],
    clusters: dict[str, Hashable],
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Accuracy per setting and language, and each setting's paired difference from the best one.

    Args:
        labels (dict[str, dict[str, str | None]]): {setting: {query id: judge label}}.
        languages (dict[str, str]): Language per query id.
        clusters (dict[str, Hashable]): Cluster per query id (translations share one).
        resamples (int): Bootstrap samples.
        seed (int): Random seed.

    Returns:
        dict[str, Any]: {questions, settings: {setting: {accuracy, correct_or_partial, en, fr, ...}}, best,
            minus_best: {setting: interval of (setting - best)}}.
    """
    questions = sorted(set.intersection(*(set(rows) for rows in labels.values())))
    if not questions:
        raise ValueError("No question was graded under every setting.")
    correct = {s: np.array([labels[s][q] == "correct" for q in questions], float) for s in labels}
    credited = ("correct", "partially correct")
    lenient = {s: np.array([labels[s][q] in credited for q in questions], float) for s in labels}
    groups = [clusters[q] for q in questions]
    result: dict[str, Any] = {"questions": len(questions), "settings": {}}
    for setting in labels:
        row = {
            "accuracy": interval(list(correct[setting]), groups, resamples, seed),
            "correct_or_partial": round(float(lenient[setting].mean()), 4),
        }
        for language in sorted(set(languages[q] for q in questions)):
            index = [i for i, q in enumerate(questions) if languages[q] == language]
            row[language] = {"questions": len(index), "accuracy": round(float(correct[setting][index].mean()), 4)}
        result["settings"][setting] = row
    best = max(labels, key=lambda s: correct[s].mean())
    result["best"] = best
    result["minus_best"] = {
        s: interval(list(correct[s] - correct[best]), groups, resamples, seed) for s in labels if s != best
    }
    return result


def main() -> None:
    """Read the verdicts, compare the settings and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--judges-config", type=Path, default=Path("configs/judges.yaml"))
    args = parser.parse_args()

    settings = get_settings()
    judge = load_yaml(args.judges_config)["judge"]
    folder = settings.results_root / f"judge_{judge}" / args.dataset
    verdicts = {
        path.stem.removeprefix("verdicts_"): read_jsonl([path]) for path in sorted(folder.glob("verdicts_*.jsonl"))
    }
    if not verdicts:
        raise SystemExit(f"No verdicts in {folder} (run judge_answers.py).")
    labels = {setting: {row["query_id"]: row["label"] for row in rows} for setting, rows in verdicts.items()}
    languages = {row["query_id"]: row["language"] for rows in verdicts.values() for row in rows}
    clusters = {
        row["query_id"]: tuple(sorted(page for page, grade in row["pages"].items() if grade > 0))
        for row in read_jsonl([settings.results_root / "qrels" / f"{args.dataset}.jsonl"])
    }
    missing = [q for q in languages if q not in clusters]
    if missing:
        raise SystemExit(f"{len(missing)} questions have no judgments in results/qrels (run export_qrels.py).")

    try:
        summary = compare(labels, languages, clusters)
    except ValueError as error:
        raise SystemExit(f"{error} (judge all answers first).") from None
    summary["judge"] = judge
    answers = json.loads((settings.results_root / "answer_answers" / args.dataset / "summary.json").read_text())
    for setting, row in summary["settings"].items():
        cost = answers["settings"].get(setting, {})
        row["seconds_per_question"] = (cost.get("seconds_per_question") or {}).get("mean")
        row["prompt_tokens"] = (cost.get("prompt_tokens") or {}).get("mean")

    results_dir = settings.results_root / "answer_settings" / args.dataset
    write_run_info(results_dir, extra={"dataset": args.dataset, "judge": judge})
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{summary['questions']} questions, judge {judge}, best {summary['best']}")
    print(f"{'setting':<10}{'accuracy [95%]':>26}{'en':>8}{'fr':>8}{'C or P':>8}{'s/q':>8}{'- best [95%]':>28}")
    for setting, row in summary["settings"].items():
        acc, diff = row["accuracy"], summary["minus_best"].get(setting)
        diff_text = f"{diff['mean']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]" if diff else "best"
        en, fr = (row.get(lang, {}).get("accuracy", float("nan")) for lang in ("en", "fr"))
        print(f"{setting:<10}{acc['mean']:>9.3f} [{acc['low']:.3f}, {acc['high']:.3f}]{en:>8.3f}{fr:>8.3f}"
              f"{row['correct_or_partial']:>8.3f}{row['seconds_per_question'] or 0:>8.1f}{diff_text:>28}")


if __name__ == "__main__":
    main()
