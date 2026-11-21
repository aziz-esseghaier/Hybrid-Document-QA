"""Compare judges with hand grades on the same answers, to choose the judge.

Usage:
    python scripts/compare_judges.py --dataset v2_esg_reports --judges gemma4_31b mistral_small_32
        [--labels results/human_labels/v2_esg_reports/labels.jsonl]

Matches hand grades (label_sheet.py read) with each judge's verdicts (judge_answers.py) on (query id, setting).
Reports exact agreement and Cohen's kappa on the three labels, the same on "correct or not" (what accuracy counts),
the accuracy each gives on the sample, and a confusion table. Writes <RESULTS_ROOT>/judge_comparison/<dataset>/
summary.json. Ids and labels only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from doc_qa.core.config import get_settings
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.judge import LABELS
from doc_qa.evaluation.metrics import cohen_kappa


def agreement(human: list[str], judge: list[str | None]) -> dict[str, Any]:
    """Measure how well a judge's labels match hand grades on the same answers.

    Args:
        human (list[str]): Hand grades.
        judge (list[str | None]): Judge labels, same order (None when the judge gave no valid reply).

    Returns:
        dict[str, Any]: Counts, agreement and kappa (three labels, and correct or not), accuracies, confusion table.
    """
    judge = [label or "unjudged" for label in judge]
    human_correct, judge_correct = [h == "correct" for h in human], [j == "correct" for j in judge]
    pairs = Counter(zip(human, judge))
    n = len(human)
    return {
        "answers": n,
        "agreement": round(sum(h == j for h, j in zip(human, judge)) / n, 4),
        "kappa": round(cohen_kappa(human, judge), 4),
        "agreement_correct_or_not": round(sum(h == j for h, j in zip(human_correct, judge_correct)) / n, 4),
        "kappa_correct_or_not": round(cohen_kappa(human_correct, judge_correct), 4),
        "accuracy_by_hand": round(sum(human_correct) / n, 4),
        "accuracy_by_judge": round(sum(judge_correct) / n, 4),
        "confusion": {h: {j: c for (hh, j), c in sorted(pairs.items()) if hh == h} for h in LABELS},  # hand -> judge
    }


def main() -> None:
    """Match the grades, score each judge, and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--judges", nargs="+", required=True)
    parser.add_argument("--labels", type=Path, help="Default: results/human_labels/<dataset>/labels.jsonl.")
    args = parser.parse_args()

    settings = get_settings()
    labels_file = args.labels or settings.results_root / "human_labels" / args.dataset / "labels.jsonl"
    human = {(row["query_id"], row["setting"]): row for row in read_jsonl([labels_file])}
    if not human:
        raise SystemExit(f"No hand grades in {labels_file} (run label_sheet.py read).")

    summary: dict[str, Any] = {}
    for judge in args.judges:
        folder = settings.results_root / f"judge_{judge}" / args.dataset
        rows = read_jsonl(sorted(folder.glob("verdicts_*.jsonl")))
        verdicts = {(row["query_id"], row["setting"]): row["label"] for row in rows}
        keys = [key for key in human if key in verdicts]
        if not keys:
            raise SystemExit(f"No verdicts of {judge} for the graded answers (run judge_answers.py --only ...).")
        result = agreement([human[k]["grade"] for k in keys], [verdicts[k] for k in keys])
        result["missing"] = len(human) - len(keys)
        for language in sorted({human[k]["language"] for k in keys}):
            subset = [k for k in keys if human[k]["language"] == language]
            scores = agreement([human[k]["grade"] for k in subset], [verdicts[k] for k in subset])
            result[language] = {key: scores[key] for key in ("answers", "agreement", "agreement_correct_or_not")}
        summary[judge] = result

    results_dir = settings.results_root / "judge_comparison" / args.dataset
    write_run_info(results_dir, extra={"dataset": args.dataset, "judges": args.judges})
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    columns = ["answers", "agreement", "kappa", "agreement_correct_or_not", "kappa_correct_or_not",
               "accuracy_by_hand", "accuracy_by_judge"]
    print(f"{'judge':<20}" + "".join(f"{c:>26}" for c in columns))
    for judge, result in summary.items():
        print(f"{judge:<20}" + "".join(f"{result[c]:>26}" for c in columns))


if __name__ == "__main__":
    main()
