"""Answer every judged question of a dataset from its top retrieved pages, for each setting in the config. Resumable.

Usage:
    python scripts/answer_questions.py --dataset v2_esg_reports --out ANSWERS_DIR [--config configs/answer.yaml]

Settings are every combination of `pages` (pages given per question) and `page_tokens` (image size per page). Questions
without a reference answer are skipped, since they cannot be judged. On a test dataset, only `test_sample` questions
are answered, drawn at random with their translations. Each question is answered under all settings before the next,
so a cut-off run leaves the settings comparable. Pages are the top of the retrieval ranking for the dataset's role,
with page text from <DATA_ROOT>/derived/chunks/<dataset>/pages.jsonl (build_chunks.py).

Writes <RESULTS_ROOT>/<name>_answers/<dataset>/: records_<pages>x<tokens>.jsonl (ids, cited pages, cost; no text),
summary.json, config.yaml, run_info.json. The answers themselves contain document text, so they go to
<ANSWERS_DIR>/<name>/<dataset>/answers_<pages>x<tokens>.jsonl (question, reference answers, answer, cited pages).
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from doc_qa.answer.model import answer_question, fit_pixels, load_answer_model
from doc_qa.answer.output import cited_page_ids
from doc_qa.answer.prompt import PageInput
from doc_qa.core.config import get_settings, load_yaml, result_path
from doc_qa.core.parallel import append_jsonl, read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.answers import AnswerRecord, sample_clusters, summarize_answers
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark
from doc_qa.retrieval.visual import decode_image, spread


def setting_name(pages: int, page_tokens: int) -> str:
    """Name a setting in file names.

    Args:
        pages (int): Pages given per question.
        page_tokens (int): Image token budget per page.

    Returns:
        str: E.g. "5x640".
    """
    return f"{pages}x{page_tokens}"


def summarize_setting(records: list[dict[str, Any]], relevance: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Summarize one setting's saved records.

    Args:
        records (list[dict[str, Any]]): Rows of a records file.
        relevance (dict[str, dict[str, int]]): Graded relevance per query id.

    Returns:
        dict[str, Any]: Answer metrics per language, cost spreads, and error counts.
    """
    outcomes = [
        AnswerRecord(
            r["language"], tuple(r["given"]), relevance.get(r["query_id"], {}), r["found"], tuple(r["cited"]),
            r["attempts"], r["seconds"],
        )
        for r in records
    ]
    costed = [r for r in records if r["prompt_tokens"]]
    return {
        "answers": summarize_answers(outcomes),
        "prompt_tokens": spread([r["prompt_tokens"] for r in costed]) if costed else None,
        "seconds_per_question": spread([r["seconds"] for r in costed]) if costed else None,
        "new_tokens": spread([r["new_tokens"] for r in costed]) if costed else None,
        "replies_with_nan_or_inf": sum(not r["finite"] for r in records),
        "out_of_memory": sum(r["error"] == "out of memory" for r in records),
    }


def choose_questions(
    query_ids: list[str], relevance: dict[str, dict[str, int]], sample: dict[str, int] | None
) -> list[str]:
    """Keep every question, or a random sample of them with their translations.

    Args:
        query_ids (list[str]): Questions that can be judged.
        relevance (dict[str, dict[str, int]]): Graded relevance per query id; translations share their relevant pages.
        sample (dict[str, int] | None): {questions, seed}, or None for all.

    Returns:
        list[str]: Query ids to answer, sorted.
    """
    if not sample:
        return sorted(query_ids)
    clusters = {q: tuple(sorted(p for p, grade in relevance[q].items() if grade > 0)) for q in query_ids}
    return sample_clusters(clusters, sample["questions"], sample["seed"])


def main() -> None:
    """Answer the questions under every setting and keep the summary up to date."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/answer.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Folder for the answers (document text).")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Answering needs a GPU.")

    settings = get_settings()
    cfg = load_yaml(args.config)
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][args.dataset]
    bench = load_benchmark(args.dataset, spec, settings.data_root / "raw")
    pages_file = settings.data_root / "derived" / "chunks" / args.dataset / "pages.jsonl"
    texts = {row["page_id"]: row["text"] for row in read_jsonl([pages_file])}
    if not texts:
        raise SystemExit(f"No page text for {args.dataset} (run ocr_store.py pull and build_chunks.py first).")
    ranked = {
        row["query_id"]: [page for page, _ in row["pages"]]
        for row in read_jsonl([result_path(cfg["rankings"][spec["role"]].format(dataset=args.dataset))])
    }
    if not ranked:
        raise SystemExit(f"No {spec['role']} rankings for {args.dataset} (run eval_fusion.py first).")

    relevance: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        relevance[qrel.query_id][qrel.page_id] = qrel.score
    judged = sorted(q for q in relevance if q in bench.queries and q in ranked)
    answerable = [q for q in judged if bench.queries[q].answers]
    sample = cfg.get("test_sample") if spec["role"] == "test" else None
    query_ids = choose_questions(answerable, relevance, sample)
    combos = list(itertools.product(sorted(cfg["pages"]), sorted(cfg["page_tokens"])))
    most_pages = max(cfg["pages"])

    results_dir = settings.results_root / f"{cfg['name']}_answers" / args.dataset
    answers_dir = args.out / cfg["name"] / args.dataset
    model_info = {"repo": cfg["model"], "revision": cfg["revision"]}
    write_run_info(results_dir, extra={"dataset": args.dataset, "model": model_info})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    def records_file(combo: tuple[int, int]) -> Path:
        """Return a setting's records file."""
        return results_dir / f"records_{setting_name(*combo)}.jsonl"

    done = {combo: {r["query_id"] for r in read_jsonl([records_file(combo)])} for combo in combos}
    todo = [q for q in query_ids if any(q not in done[combo] for combo in combos)]
    print(f"{len(query_ids)} questions to answer ({len(judged) - len(answerable)} without a reference answer skipped, "
          f"sample {sample}), {len(todo)} still to answer, settings: {[setting_name(*c) for c in combos]}", flush=True)

    def save_summary() -> None:
        """Rewrite the summary from the records files."""
        summary = {
            "questions": len(query_ids),
            "skipped_without_reference_answer": len(judged) - len(answerable),
            "sample": {**sample, "queries_before_sampling": len(answerable)} if sample else None,
            "settings": {
                setting_name(*combo): summarize_setting(read_jsonl([records_file(combo)]), relevance)
                for combo in combos
            },
        }
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not todo:
        save_summary()
        return
    encoded = dict(iter_page_images(bench, {page for q in todo for page in ranked[q][:most_pages]}))
    model, processor = load_answer_model(
        cfg["model"], cfg["revision"], cfg["dtype"], cfg["attn_implementation"], cfg.get("max_memory")
    )

    for count, query_id in enumerate(todo, 1):
        query = bench.queries[query_id]
        images = {page_id: decode_image(encoded[page_id]) for page_id in ranked[query_id][:most_pages]}
        for combo in combos:
            if query_id in done[combo]:
                continue
            pages_kept, page_tokens = combo
            given = ranked[query_id][:pages_kept]
            pages = [PageInput(p, fit_pixels(images[p], page_tokens), texts.get(p, "")) for p in given]
            record: dict[str, Any] = {
                "query_id": query_id, "language": query.language, "given": given, "found": None, "cited": [],
                "attempts": 0, "seconds": 0.0, "prompt_tokens": 0, "new_tokens": 0, "finite": True, "error": None,
            }
            answer: dict[str, Any] = {"answer": None, "errors": []}
            try:
                result, generations = answer_question(
                    model, processor, query.text, pages, cfg["max_text_chars"], cfg["max_new_tokens"], cfg["retries"],
                    cfg.get("sdpa_backend"),
                )
            except torch.cuda.OutOfMemoryError:
                record["error"] = "out of memory"
                gc.collect()
                torch.cuda.empty_cache()
            else:
                cited = cited_page_ids(result.answer, given) if result.answer else []
                record.update(
                    found=result.answer.found if result.answer else None,
                    cited=cited,
                    attempts=result.attempts,
                    seconds=round(sum(g.seconds for g in generations), 2),
                    prompt_tokens=generations[0].prompt_tokens,
                    new_tokens=sum(g.new_tokens for g in generations),
                    finite=all(g.finite for g in generations),
                )
                answer.update(answer=result.answer.text if result.answer else None, errors=list(result.errors))
            append_jsonl(answers_dir / f"answers_{setting_name(*combo)}.jsonl", [{
                "query_id": query_id, "language": query.language, "question": query.text,
                "reference_answers": list(query.answers), "given": given, "found": record["found"],
                "cited": record["cited"], **answer,
            }])
            append_jsonl(records_file(combo), [record])
        save_summary()
        print(f"{count}/{len(todo)} {query_id} [{query.language}]", flush=True)

    save_summary()
    print((results_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
