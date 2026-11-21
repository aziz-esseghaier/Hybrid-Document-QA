"""Check that the answer model runs on this GPU setup, and measure what answering costs.

Usage:
    python scripts/check_answer_model.py [--config configs/answer.yaml]

For a few questions of one dataset, gives the model the top pages of the chosen retrieval ranking (page images, and
OCR text from <DATA_ROOT>/derived/chunks/<dataset>/pages.jsonl, built by build_chunks.py), for every number of pages
and image size in the config. Measures memory, prompt length and time per question, and checks the replies (valid
JSON, retries, cited pages, NaN or inf scores, the same reply when asked twice). Writes
<RESULTS_ROOT>/<name>_check/<dataset>/: summary.json (updated after each setting), config.yaml, run_info.json; no
document text. The replies go to the log and to <DATA_ROOT>/derived/<name>_check/<dataset>/replies.jsonl.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from doc_qa.answer.model import Generation, answer_question, fit_pixels, generate_reply, load_answer_model
from doc_qa.answer.output import cited_page_ids
from doc_qa.answer.prompt import PageInput, build_messages
from doc_qa.core.config import REPO_ROOT, get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl, write_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.core.schemas import Qrel, Query
from doc_qa.evaluation.answers import AnswerRecord, summarize_answers
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark
from doc_qa.retrieval.visual import decode_image, spread


def choose_questions(queries: dict[str, Query], qrels: Iterable[Qrel], questions: int, seed: int) -> list[str]:
    """Pick questions that have a relevant page, with all their translations.

    Args:
        queries (dict[str, Query]): Queries by id.
        qrels (Iterable[Qrel]): Relevance judgments.
        questions (int): Questions to pick; translations share relevant pages and count as one.
        seed (int): Random seed.

    Returns:
        list[str]: Query ids, sorted.
    """
    relevant: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        if qrel.score > 0 and qrel.query_id in queries:
            relevant[qrel.query_id].add(qrel.page_id)
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for query_id, pages in relevant.items():
        groups[tuple(sorted(pages))].append(query_id)
    keys = random.Random(seed).sample(sorted(groups), min(questions, len(groups)))
    return sorted(query_id for key in keys for query_id in groups[key])


def gpu_gb(values: Iterable[int]) -> list[float]:
    """Convert byte counts to GB.

    Args:
        values (Iterable[int]): Bytes per GPU.

    Returns:
        list[float]: GB per GPU, rounded to 0.01.
    """
    return [round(value / 1e9, 2) for value in values]


def main() -> None:
    """Answer the sample under each setting and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/answer.yaml"))
    args = parser.parse_args()

    import torch

    gpus = torch.cuda.device_count()
    if not gpus:
        raise SystemExit("This check needs a GPU.")

    settings = get_settings()
    cfg = load_yaml(args.config)
    check = cfg["check"]
    dataset = check["dataset"]
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
    bench = load_benchmark(dataset, spec, settings.data_root / "raw")
    pages_file = settings.data_root / "derived" / "chunks" / dataset / "pages.jsonl"
    texts = {row["page_id"]: row["text"] for row in read_jsonl([pages_file])}
    if not texts:
        raise SystemExit(f"No page text for {dataset} (run ocr_store.py pull and build_chunks.py first).")
    ranked = {
        row["query_id"]: [page for page, _ in row["pages"]]
        for row in read_jsonl([REPO_ROOT / cfg["rankings"].format(dataset=dataset)])
    }

    chosen = choose_questions(bench.queries, bench.qrels, check["questions"], check["seed"])
    query_ids = [query_id for query_id in chosen if query_id in ranked]
    relevance: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        relevance[qrel.query_id][qrel.page_id] = qrel.score
    most_pages = max(check["pages"])
    needed = {page for q in query_ids for page in ranked[q][:most_pages]}
    images = {page_id: decode_image(data) for page_id, data in iter_page_images(bench, needed)}

    results_dir = settings.results_root / f"{cfg['name']}_check" / dataset
    replies_file = settings.data_root / "derived" / f"{cfg['name']}_check" / dataset / "replies.jsonl"
    model_info = {"repo": cfg["model"], "revision": cfg["revision"]}
    write_run_info(results_dir, seed=check["seed"], extra={"dataset": dataset, "model": model_info})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    summary: dict[str, Any] = {"queries": len(query_ids), "settings": []}

    def save() -> None:
        """Write the summary so far."""
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    start = time.perf_counter()
    model, processor = load_answer_model(
        cfg["model"], cfg["revision"], cfg["dtype"], cfg["attn_implementation"], cfg.get("max_memory")
    )
    summary["load_seconds"] = round(time.perf_counter() - start)
    summary["gpu_gb_after_load"] = gpu_gb(torch.cuda.memory_allocated(i) for i in range(gpus))
    devices = list(model.hf_device_map.values())
    summary["modules_per_device"] = {str(d): devices.count(d) for d in sorted(set(devices), key=str)}
    save()
    if {"cpu", "disk"} & set(map(str, devices)):
        raise SystemExit("Some weights did not fit on the GPUs (raise max_memory or use a smaller model).")

    all_replies: list[dict[str, Any]] = []
    first_chat: tuple[list[dict[str, Any]], str] | None = None
    for pages_kept, page_tokens in itertools.product(sorted(check["pages"]), sorted(check["page_tokens"])):
        label = f"{pages_kept} pages x {page_tokens} tokens"
        print(f"--- {label}", flush=True)
        for i in range(gpus):
            torch.cuda.reset_peak_memory_stats(i)
        sized = {page_id: fit_pixels(image, page_tokens) for page_id, image in images.items()}
        records: list[AnswerRecord] = []
        costs: list[list[Generation]] = []
        status = "ok"
        for query_id in query_ids:
            query = bench.queries[query_id]
            given = ranked[query_id][:pages_kept]
            pages = [PageInput(page_id, sized[page_id], texts.get(page_id, "")) for page_id in given]
            try:
                result, generations = answer_question(
                    model, processor, query.text, pages, cfg["max_text_chars"], cfg["max_new_tokens"], cfg["retries"],
                    cfg.get("sdpa_backend"),
                )
            except torch.cuda.OutOfMemoryError:
                status = f"out of memory at query {len(records) + 1}"
                gc.collect()
                torch.cuda.empty_cache()
                break
            if first_chat is None:
                first_chat = (build_messages(query.text, pages, cfg["max_text_chars"]), generations[0].reply)
            cited = tuple(cited_page_ids(result.answer, given)) if result.answer else ()
            records.append(AnswerRecord(
                query.language, tuple(given), relevance[query_id], result.answer.found if result.answer else None,
                cited, result.attempts, sum(g.seconds for g in generations),
            ))
            costs.append(generations)
            text = result.answer.text if result.answer else "-"
            all_replies.append({
                "setting": label, "query_id": query_id, "attempts": result.attempts, "errors": result.errors,
                "found": records[-1].found, "cited": cited, "answer": text,
            })
            print(f"{query_id} [{query.language}] found={records[-1].found} cited={list(cited)} "
                  f"attempts={result.attempts} {records[-1].seconds:.1f}s :: {text[:200]}", flush=True)

        row: dict[str, Any] = {"pages": pages_kept, "page_tokens": page_tokens, "status": status}
        if records:
            row.update({
                "prompt_tokens": spread([g[0].prompt_tokens for g in costs]),
                "seconds_per_question": spread([round(r.seconds, 1) for r in records]),
                "new_tokens": spread([sum(x.new_tokens for x in g) for g in costs]),
                "replies_with_nan_or_inf": sum(not x.finite for g in costs for x in g),
                "answers": summarize_answers(records),
            })
        row["peak_gpu_gb"] = gpu_gb(torch.cuda.max_memory_allocated(i) for i in range(gpus))
        summary["settings"].append(row)
        save()

    if first_chat is not None:
        messages, reply = first_chat
        again = generate_reply(model, processor, messages, cfg["max_new_tokens"], cfg.get("sdpa_backend"))
        summary["same_reply_twice"] = again.reply == reply
    write_jsonl(replies_file, all_replies)
    save()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
