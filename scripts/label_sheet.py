"""Make a sheet of model answers to grade by hand, and read the grades back.

Usage:
    python scripts/label_sheet.py make --answers ANSWERS_DIR --out SHEET.md [--n 50] [--seed 0]
    python scripts/label_sheet.py ids --sheet SHEET.md --out results/human_labels/v2_esg_reports/sample.jsonl
    python scripts/label_sheet.py read --sheet SHEET.md --out results/human_labels/v2_esg_reports/labels.jsonl

`make` samples answers from ANSWERS_DIR/answers_<setting>.jsonl (answer_questions.py): distinct questions, half
English and half French, the settings in turn; replies that failed the checks are left out. The sheet shows each
question, its reference answers and the model's answer, so it contains document text: keep it out of the repo. Write
C, P or I after each "Label:". `ids` lists the sheet's answers (for judging the same ones), `read` the grades; both
keep only ids.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from doc_qa.core.parallel import read_jsonl, write_jsonl

GRADES = {"C": "correct", "P": "partially correct", "I": "incorrect"}
HEADER = re.compile(r"^### (\d+) · (\S+) · (\S+) · (\S+)$")

INSTRUCTIONS = """# Grade these answers

After each "Label:", write one letter:
- **C** (correct): the answer contains the core information of the reference answer.
- **P** (partially correct): part of the core information, or the right information plus a wrong claim.
- **I** (incorrect): wrong or unrelated, or "not in the pages" when the reference does answer the question.

Grade against the reference only; it may be in another language than the question. A note is optional.
"""


def load_answers(folder: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read every setting's answers, keeping the last row per question.

    Args:
        folder (Path): Folder with answers_<setting>.jsonl files.

    Returns:
        dict[str, dict[str, dict[str, Any]]]: {setting: {query id: row}}.
    """
    answers = {}
    for path in sorted(folder.glob("answers_*.jsonl")):
        answers[path.stem.removeprefix("answers_")] = {row["query_id"]: row for row in read_jsonl([path])}
    return answers


def sample(answers: dict[str, dict[str, dict[str, Any]]], n: int, seed: int) -> list[tuple[str, str]]:
    """Pick answers to grade: distinct questions, half per language, settings in turn, failed replies left out.

    Args:
        answers (dict[str, dict[str, dict[str, Any]]]): Output of `load_answers`.
        n (int): Answers to pick.
        seed (int): Random seed.

    Returns:
        list[tuple[str, str]]: (query id, setting) pairs, in sheet order.
    """
    rng = random.Random(seed)
    settings = sorted(answers)
    usable = {q for rows in answers.values() for q, row in rows.items() if row["found"] is not None}
    language = {q: row["language"] for rows in answers.values() for q, row in rows.items()}
    chosen: list[str] = []
    for lang in sorted({language[q] for q in usable}):
        pool = sorted(q for q in usable if language[q] == lang)
        rng.shuffle(pool)
        chosen += pool[: n // 2]
    rng.shuffle(chosen)
    picks = []
    for i, query_id in enumerate(chosen[:n]):
        for setting in settings[i % len(settings):] + settings[: i % len(settings)]:  # next setting with an answer
            row = answers[setting].get(query_id)
            if row and row["found"] is not None:
                picks.append((query_id, setting))
                break
    return picks


def make_sheet(answers: dict[str, dict[str, dict[str, Any]]], picks: list[tuple[str, str]]) -> str:
    """Write the grading sheet.

    Args:
        answers (dict[str, dict[str, dict[str, Any]]]): Output of `load_answers`.
        picks (list[tuple[str, str]]): Output of `sample`.

    Returns:
        str: Markdown sheet.
    """
    blocks = [INSTRUCTIONS]
    for number, (query_id, setting) in enumerate(picks, 1):
        row = answers[setting][query_id]
        references = "\n".join(f"- {ref}" for ref in row["reference_answers"])
        found = "yes" if row["found"] else "no (the model says the pages don't answer it)"
        blocks.append(
            f"---\n### {number} · {query_id} · {setting} · {row['language']}\n\n"
            f"**Question:** {row['question']}\n\n**Reference:**\n{references}\n\n"
            f"**Answer:** {row['answer']}\n\n**Found:** {found}\n\nLabel: \nNote: \n"
        )
    return "\n".join(blocks)


def sheet_ids(text: str) -> list[dict[str, Any]]:
    """List the answers on a sheet, graded or not.

    Args:
        text (str): Sheet content.

    Returns:
        list[dict[str, Any]]: {number, query_id, setting, language} per answer.
    """
    rows = []
    for line in text.splitlines():
        if header := HEADER.match(line.strip()):
            number, query_id, setting, language = header.groups()
            rows.append({"number": int(number), "query_id": query_id, "setting": setting, "language": language})
    return rows


def read_sheet(text: str) -> list[dict[str, Any]]:
    """Read the grades from a filled sheet.

    Args:
        text (str): Sheet content.

    Returns:
        list[dict[str, Any]]: {number, query_id, setting, language, grade} per graded answer.
    """
    labels, current = [], None
    for line in text.splitlines():
        if header := HEADER.match(line.strip()):
            number, query_id, setting, language = header.groups()
            current = {"number": int(number), "query_id": query_id, "setting": setting, "language": language}
        elif current and line.strip().lower().startswith("label:"):
            letter = line.split(":", 1)[1].strip().upper()[:1]
            if letter not in GRADES:
                raise ValueError(f"Answer {current['number']}: write C, P or I after 'Label:' (found {letter!r}).")
            labels.append({**current, "grade": GRADES[letter]})
            current = None
    return labels


def main() -> None:
    """Make a sheet or read one."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["make", "ids", "read"])
    parser.add_argument("--answers", type=Path, help="make: folder with answers_<setting>.jsonl.")
    parser.add_argument("--sheet", type=Path, help="read: the filled sheet.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.action == "make":
        answers = load_answers(args.answers)
        picks = sample(answers, args.n, args.seed)
        args.out.write_text(make_sheet(answers, picks), encoding="utf-8")
        print(f"{len(picks)} answers to grade in {args.out}")
    elif args.action == "ids":
        rows = sheet_ids(args.sheet.read_text(encoding="utf-8"))
        write_jsonl(args.out, rows)
        print(f"{len(rows)} answer ids -> {args.out}")
    else:
        labels = read_sheet(args.sheet.read_text(encoding="utf-8"))
        write_jsonl(args.out, labels)
        counts = {grade: sum(label["grade"] == grade for label in labels) for grade in GRADES.values()}
        print(f"{len(labels)} grades: {json.dumps(counts)} -> {args.out}")


if __name__ == "__main__":
    main()
