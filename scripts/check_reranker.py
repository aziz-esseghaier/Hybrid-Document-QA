"""Check that the text reranker runs correctly on this GPU, and measure what it costs.

Usage:
    python scripts/check_reranker.py [--config configs/rerank.yaml]

Reranks the top pages of the saved visual ranking for a sample of questions, reading each page's OCR text
(<DATA_ROOT>/derived/chunks/<dataset>/pages.jsonl, from build_chunks.py). Measures memory, speed per batch size and
prompt lengths. Checks that batching does not change the scores, that relevant pages move up, and that fp16 matches
fp32 (model spread over both GPUs). Writes <RESULTS_ROOT>/rerank_check/<dataset>/: summary.json (updated after each
step), config.yaml, run_info.json. No document text.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from doc_qa.core.config import REPO_ROOT, get_settings, load_yaml
from doc_qa.core.parallel import read_jsonl
from doc_qa.core.run_info import write_run_info
from doc_qa.evaluation.metrics import ndcg_at_k
from doc_qa.evaluation.vidore import load_benchmark
from doc_qa.retrieval.rerank import load_reranker, pair_ids, score_pairs
from doc_qa.retrieval.visual import spread


def pick_queries(qrels: dict[str, dict[str, int]], questions: int, seed: int) -> list[str]:
    """Pick questions at random, with all their translations.

    Args:
        qrels (dict[str, dict[str, int]]): Relevance per query id.
        questions (int): Questions to pick; translations share their relevant pages and count as one.
        seed (int): Random seed.

    Returns:
        list[str]: Query ids, sorted.
    """
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for query_id, pages in qrels.items():
        groups[tuple(sorted(page for page, grade in pages.items() if grade > 0))].append(query_id)
    keys = random.Random(seed).sample(sorted(groups), min(questions, len(groups)))
    return sorted(query_id for key in keys for query_id in groups[key])


def rerank_quality(
    query_ids: Sequence[str],
    languages: dict[str, str],
    candidates: dict[str, list[str]],
    scores: dict[tuple[str, str], float],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Compare NDCG@10 of the candidates in their original order and in reranked order, per language.

    Args:
        query_ids (Sequence[str]): Queries.
        languages (dict[str, str]): Language per query.
        candidates (dict[str, list[str]]): Candidate pages per query, in original order.
        scores (dict[tuple[str, str], float]): Reranker score per (query, page).
        qrels (dict[str, dict[str, int]]): Relevance per query.

    Returns:
        dict[str, dict[str, float]]: {language: queries, with_relevant_candidate, ndcg@10_before, ndcg@10_after}.
    """
    rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for query_id in query_ids:
        before = candidates[query_id]
        after = sorted(before, key=lambda page: -scores[query_id, page])  # stable: ties keep their order
        relevance = qrels[query_id]
        found = float(any(relevance.get(page, 0) > 0 for page in before))
        rows[languages[query_id]].append((found, ndcg_at_k(before, relevance), ndcg_at_k(after, relevance)))
    return {
        language: {
            "queries": len(values),
            "with_relevant_candidate": round(float(np.mean([v[0] for v in values])), 3),
            "ndcg@10_before": round(float(np.mean([v[1] for v in values])), 3),
            "ndcg@10_after": round(float(np.mean([v[2] for v in values])), 3),
        }
        for language, values in sorted(rows.items())
    }


def probe_batch_sizes(
    model: Any, tokenizer: Any, prompts: list[list[int]], sizes: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Score the same prompts at growing batch sizes, stopping at the first out-of-memory error.

    Args:
        model (Any): Reranker on GPU.
        tokenizer (Any): Its tokenizer.
        prompts (list[list[int]]): Token ids.
        sizes (Sequence[int]): Batch sizes, smallest first.

    Returns:
        tuple[list[dict[str, Any]], dict[int, np.ndarray]]: One row per size, scores per working size.
    """
    import torch

    rows: list[dict[str, Any]] = []
    outputs: dict[int, np.ndarray] = {}
    tokens = sum(len(prompt) for prompt in prompts)
    for size in sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            outputs[size] = score_pairs(model, tokenizer, prompts, size)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": size, "status": "out of memory"})
            break
        seconds = time.perf_counter() - start
        rows.append({
            "batch_size": size,
            "status": "ok",
            "seconds_per_pair": round(seconds / len(prompts), 3),
            "tokens_per_second": round(tokens / seconds),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        })
    gc.collect()
    torch.cuda.empty_cache()
    return rows, outputs


def main() -> None:
    """Run the checks and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/rerank.yaml"))
    args = parser.parse_args()

    import torch

    if torch.cuda.device_count() < 2:
        raise SystemExit("This check needs two GPUs (the fp32 reference is spread over both).")

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

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in bench.qrels:
        qrels[qrel.query_id][qrel.page_id] = qrel.score
    saved = read_jsonl([REPO_ROOT / check["rankings"]])
    candidates = {row["query_id"]: [page for page, _ in row["pages"][: check["candidates"]]] for row in saved}
    query_ids = pick_queries(qrels, check["questions"], check["seed"])
    pairs = [(query_id, page) for query_id in query_ids for page in candidates[query_id]]
    languages = {query_id: bench.queries[query_id].language for query_id in query_ids}

    results_dir = settings.results_root / "rerank_check" / dataset
    write_run_info(results_dir, seed=check["seed"], extra={
        "dataset": dataset, "model": {"repo": cfg["model"], "revision": cfg["revision"]},
    })
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    summary: dict[str, Any] = {"queries": len(query_ids), "pairs": len(pairs)}

    def save(step: str) -> None:
        """Write the summary so far.

        Args:
            step (str): Last finished step.
        """
        summary["last_step"] = step
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Done: {step}", flush=True)

    model, tokenizer = load_reranker(
        cfg["model"], cfg["revision"], cfg["dtype"], "cuda:0", cfg["attn_implementation"]
    )
    summary["gpu_gb_after_load"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    prompts = [
        pair_ids(tokenizer, cfg["instruction"], bench.queries[query_id].text, texts.get(page, ""), cfg["max_tokens"])
        for query_id, page in pairs
    ]
    overhead = len(pair_ids(tokenizer, cfg["instruction"], "", "", cfg["max_tokens"]))
    page_tokens = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts.values()]
    summary["prompt_tokens"] = spread([len(prompt) for prompt in prompts])
    summary["dataset_page_tokens"] = spread(page_tokens)
    summary["dataset_pages_cut"] = sum(tokens + overhead > cfg["max_tokens"] for tokens in page_tokens)
    save("load")

    score_pairs(model, tokenizer, prompts[:2], 1)  # the first pass is slower: keep it out of the timing
    summary["batches"], by_size = probe_batch_sizes(model, tokenizer, prompts, check["batch_sizes"])
    save("batch sizes")
    if 1 not in by_size:
        raise SystemExit("One prompt does not fit in GPU memory.")

    single, largest = by_size[1], max(by_size)
    summary["batching_changes_nothing"] = {
        "batch_size": largest, "max_abs_diff": round(float(np.max(np.abs(single - by_size[largest]))), 4),
    }
    summary["score_range"] = [round(float(single.min()), 2), round(float(single.max()), 2)]
    fastest = min(float(row["seconds_per_pair"]) for row in summary["batches"] if row["status"] == "ok")
    summary["dev_hours_two_gpus"] = {
        f"{n}_candidates": round(check["dev_queries"] * n * fastest / 2 / 3600, 2) for n in (20, 30, 50)
    }
    scores = {pair: float(score) for pair, score in zip(pairs, single)}
    summary["quality"] = rerank_quality(query_ids, languages, candidates, scores, qrels)
    save("scores")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    reference_model, _ = load_reranker(
        cfg["model"], cfg["revision"], check["reference_dtype"], "auto", cfg["attn_implementation"]
    )
    count = check["reference_pairs"]
    reference = score_pairs(reference_model, tokenizer, prompts[:count], 1)
    from scipy.stats import spearmanr

    summary["matches_reference"] = {
        "dtype": check["reference_dtype"],
        "pairs": count,
        "max_abs_diff": round(float(np.max(np.abs(single[:count] - reference))), 4),
        "mean_abs_diff": round(float(np.mean(np.abs(single[:count] - reference))), 4),
        "spearman": round(float(spearmanr(single[:count], reference).statistic), 4),
    }
    save("reference")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
