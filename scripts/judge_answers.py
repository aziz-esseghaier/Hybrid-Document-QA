"""Grade saved answers with an LLM judge, against the reference answers. Resumable.

Usage:
    python scripts/judge_answers.py --judge gemma4_31b --dataset v2_esg_reports --answers ANSWERS_DIR --out TEXT_DIR
        [--only results/human_labels/v2_esg_reports/sample.jsonl] [--config configs/judges.yaml]

Reads ANSWERS_DIR/answers_<setting>.jsonl (answer_questions.py; the last row per question counts). Replies that failed
the checks are graded "incorrect" without calling the judge. --only grades just the listed {query_id, setting} pairs,
e.g. the hand-graded sample. Writes <RESULTS_ROOT>/judge_<judge>/<dataset>/: verdicts_<setting>.jsonl (ids, label,
cost; no text), summary.json (accuracy = share graded correct, per setting and language), config.yaml, run_info.json.
The explanations quote the answers, so they go to TEXT_DIR/judge_<judge>/<dataset>/explanations_<setting>.jsonl.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.parallel import append_jsonl, read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.judge import LABELS, JudgeReply, judge_messages, judge_with_retries, load_judge


def load_answers(folder: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read every setting's answers, keeping the last row per question.

    Args:
        folder (Path): Folder with answers_<setting>.jsonl files.

    Returns:
        dict[str, dict[str, dict[str, Any]]]: {setting: {query id: row}}.
    """
    return {
        path.stem.removeprefix("answers_"): {row["query_id"]: row for row in read_jsonl([path])}
        for path in sorted(folder.glob("answers_*.jsonl"))
    }


def summarize(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Count labels and accuracy (share graded correct), overall and per language.

    Args:
        verdicts (list[dict[str, Any]]): Rows of a verdicts file.

    Returns:
        dict[str, Any]: {"all" or language: {questions, accuracy, correct, partially correct, incorrect, unjudged}}.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for verdict in verdicts:
        groups["all"].append(verdict)
        groups[verdict["language"]].append(verdict)
    summary = {}
    for name, rows in groups.items():
        counts = Counter(row["label"] for row in rows)
        summary[name] = {
            "questions": len(rows),
            "accuracy": round(float(np.mean([row["label"] == "correct" for row in rows])), 4),
            **{label: counts[label] for label in LABELS},
            "unjudged": counts[None],
        }
    return summary


def main() -> None:
    """Grade every answer not graded yet and keep the summary up to date."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/judges.yaml"))
    parser.add_argument("--judge", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--answers", type=Path, required=True, help="Folder with answers_<setting>.jsonl.")
    parser.add_argument("--out", type=Path, required=True, help="Folder for the explanations (document text).")
    parser.add_argument("--only", type=Path, help="JSONL of {query_id, setting} pairs to grade.")
    args = parser.parse_args()

    settings = get_settings()
    cfg = load_yaml(args.config)
    spec = cfg["judges"][args.judge]
    answers = load_answers(args.answers)
    if not answers:
        raise SystemExit(f"No answers_<setting>.jsonl in {args.answers}.")
    wanted = {(row["query_id"], row["setting"]) for row in read_jsonl([args.only])} if args.only else None

    name = f"judge_{args.judge}"
    results_dir = settings.results_root / name / args.dataset
    text_dir = args.out / name / args.dataset
    model_info = {"repo": spec["model"], "revision": spec["revision"]}
    write_run_info(results_dir, extra={"dataset": args.dataset, "model": model_info})
    run_cfg = yaml.safe_dump({**cfg, "judge": args.judge}, sort_keys=False)
    (results_dir / "config.yaml").write_text(run_cfg, encoding="utf-8")

    def verdicts_file(setting: str) -> Path:
        """Return a setting's verdicts file."""
        return results_dir / f"verdicts_{setting}.jsonl"

    todo = []
    for setting, rows in answers.items():
        done = {row["query_id"] for row in read_jsonl([verdicts_file(setting)])}
        todo += [
            (setting, row) for q, row in sorted(rows.items())
            if q not in done and (wanted is None or (q, setting) in wanted)
        ]
    print(f"{len(todo)} answers to grade with {args.judge}", flush=True)

    def save_summary() -> None:
        """Rewrite the summary from the verdicts files."""
        summary = {setting: summarize(read_jsonl([verdicts_file(setting)])) for setting in answers}
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import torch

    chat = load_judge(spec, cfg["max_new_tokens"]) if any(row["found"] is not None for _, row in todo) else None
    for count, (setting, row) in enumerate(todo, 1):
        record = {"query_id": row["query_id"], "language": row["language"], "setting": setting, "label": "incorrect",
                  "auto": True, "attempts": 0, "seconds": 0.0, "prompt_tokens": 0, "finite": True}
        explanation, errors = "No usable reply from the answer model.", []
        if row["found"] is not None:
            replies: list[JudgeReply] = []

            def generate(messages: list[dict[str, Any]]) -> str:
                """Call the judge and keep the cost of the call."""
                replies.append(chat(messages))
                return replies[-1].text

            messages = judge_messages(
                row["question"], row["reference_answers"], row["answer"], cfg["max_reference_chars"]
            )
            try:
                verdict = judge_with_retries(generate, messages, cfg["retries"])
            except torch.cuda.OutOfMemoryError:  # recorded as unjudged; the run goes on
                gc.collect()
                torch.cuda.empty_cache()
                record.update(label=None, auto=False, error="out of memory")
                explanation, errors = "", ["out of memory"]
            else:
                record.update(
                    label=verdict.label, auto=False, attempts=verdict.attempts,
                    seconds=round(sum(r.seconds for r in replies), 2), prompt_tokens=replies[0].prompt_tokens,
                    finite=all(r.finite for r in replies),
                )
                explanation, errors = verdict.explanation, list(verdict.errors)
        append_jsonl(text_dir / f"explanations_{setting}.jsonl", [{
            "query_id": row["query_id"], "setting": setting, "label": record["label"], "explanation": explanation,
            "errors": errors,
        }])
        append_jsonl(verdicts_file(setting), [record])
        save_summary()
        print(f"{count}/{len(todo)} {setting} {row['query_id']}: {record['label']} ({record['seconds']} s)", flush=True)

    save_summary()
    print((results_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
