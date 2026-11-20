"""Check that the visual retriever runs correctly on this GPU, and measure what it costs.

Usage:
    python scripts/check_visual_model.py [--config configs/visual.yaml]

On a small sample of pages and questions, measures memory, speed per batch size and vectors per page. Checks that
batching does not change the embeddings, that relevant pages rank well, and that the results match the model's own
precision (run on CPU). Writes <RESULTS_ROOT>/<name>_check/<dataset>/ (`name` from the config): summary.json
(updated after each step), config.yaml, run_info.json. No document text.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from doc_qa.core.config import get_settings, load_yaml
from doc_qa.core.run_info import write_run_info
from doc_qa.core.schemas import Qrel, Query
from doc_qa.evaluation.vidore import iter_page_images, load_benchmark
from doc_qa.retrieval.visual import (
    decode_image,
    embed_images,
    embed_queries,
    load_visual_model,
    maxsim,
    spread,
    token_cosines,
)


def choose_sample(
    queries: dict[str, Query], qrels: Iterable[Qrel], page_ids: Iterable[str], questions: int, pages: int, seed: int
) -> tuple[list[str], list[str], dict[str, set[str]]]:
    """Pick questions with one of their relevant pages each, plus random other pages.

    Args:
        queries (dict[str, Query]): Queries by id.
        qrels (Iterable[Qrel]): Relevance judgments.
        page_ids (Iterable[str]): Available pages.
        questions (int): Questions to pick; translations share relevant pages and count as one.
        pages (int): Pages in the sample.
        seed (int): Random seed.

    Returns:
        tuple[list[str], list[str], dict[str, set[str]]]: Query ids, page ids, sampled relevant pages per query.
    """
    rng = random.Random(seed)
    available = set(page_ids)
    relevant: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        if qrel.score > 0 and qrel.query_id in queries and qrel.page_id in available:
            relevant[qrel.query_id].add(qrel.page_id)

    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for query_id, relevant_pages in relevant.items():
        groups[tuple(sorted(relevant_pages))].append(query_id)
    keys = rng.sample(sorted(groups), min(questions, len(groups)))

    chosen = {rng.choice(key) for key in keys}
    others = sorted(available - chosen)
    chosen.update(rng.sample(others, min(max(0, pages - len(chosen)), len(others))))
    query_ids = sorted(query_id for key in keys for query_id in groups[key])
    return query_ids, sorted(chosen), {q: relevant[q] & chosen for q in query_ids}


def retrieval_check(
    query_ids: Sequence[str],
    languages: Sequence[str],
    relevant: dict[str, set[str]],
    page_ids: Sequence[str],
    scores: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Measure how well the sample's relevant pages rank, per language.

    Args:
        query_ids (Sequence[str]): Queries, in the order of `scores` rows.
        languages (Sequence[str]): Language per query.
        relevant (dict[str, set[str]]): Relevant pages per query.
        page_ids (Sequence[str]): Pages, in the order of `scores` columns.
        scores (np.ndarray): (queries, pages).

    Returns:
        dict[str, dict[str, float]]: {language: queries, top1, mrr, top1_by_chance}.
    """
    rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for query_id, language, row in zip(query_ids, languages, scores):
        ranked = [page_ids[i] for i in np.argsort(-row, kind="stable")]
        rank = next(i for i, page in enumerate(ranked, 1) if page in relevant[query_id])
        rows[language].append((float(rank == 1), 1 / rank, len(relevant[query_id]) / len(page_ids)))
    return {
        language: {
            "queries": len(values),
            "top1": round(float(np.mean([v[0] for v in values])), 3),
            "mrr": round(float(np.mean([v[1] for v in values])), 3),
            "top1_by_chance": round(float(np.mean([v[2] for v in values])), 3),
        }
        for language, values in sorted(rows.items())
    }


def cosine_stats(first: list[np.ndarray], second: list[np.ndarray]) -> dict[str, float]:
    """Compare two sets of embeddings of the same sequences.

    Args:
        first (list[np.ndarray]): Embeddings.
        second (list[np.ndarray]): The same sequences, embedded another way.

    Returns:
        dict[str, float]: Mean and lowest cosine over all tokens.
    """
    cosines = np.concatenate([token_cosines(a, b) for a, b in zip(first, second)])
    return {"mean": round(float(cosines.mean()), 5), "min": round(float(cosines.min()), 5)}


def probe_batch_sizes(
    model: Any, processor: Any, images: list[Any], sizes: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[int, list[np.ndarray]]]:
    """Embed the same pages at growing batch sizes, stopping at the first out-of-memory error.

    Args:
        model (Any): Visual retriever on GPU.
        processor (Any): Its processor.
        images (list[Any]): Pages.
        sizes (Sequence[int]): Batch sizes, smallest first.

    Returns:
        tuple[list[dict[str, Any]], dict[int, list[np.ndarray]]]: One row per size, embeddings per working size.
    """
    import torch

    rows: list[dict[str, Any]] = []
    outputs: dict[int, list[np.ndarray]] = {}
    for size in sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            outputs[size] = embed_images(model, processor, images, size)
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": size, "status": "out of memory"})
            break
        seconds = (time.perf_counter() - start) / len(images)
        rows.append({
            "batch_size": size,
            "status": "ok",
            "seconds_per_page": round(seconds, 2),
            "pages_per_hour_two_gpus": round(2 * 3600 / seconds),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        })
    gc.collect()
    torch.cuda.empty_cache()
    return rows, outputs


def main() -> None:
    """Run the checks and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/visual.yaml"))
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This check needs a GPU.")

    settings = get_settings()
    cfg = load_yaml(args.config)
    check = cfg["check"]
    dataset = check["dataset"]
    spec = load_yaml(Path(cfg["datasets_config"]))["datasets"][dataset]
    bench = load_benchmark(dataset, spec, settings.data_root / "raw")

    model_info = {"repo": cfg["model"], "revision": cfg["revision"]}
    results_dir = settings.results_root / f"{cfg['name']}_check" / dataset
    write_run_info(results_dir, seed=check["seed"], extra={"dataset": dataset, "model": model_info})
    (results_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    query_ids, page_ids, relevant = choose_sample(
        bench.queries, bench.qrels, bench.page_locations, check["questions"], check["pages"], check["seed"]
    )
    decoded = {page_id: decode_image(data) for page_id, data in iter_page_images(bench, page_ids)}
    images = [decoded[page_id] for page_id in page_ids]
    texts = [bench.queries[query_id].text for query_id in query_ids]
    languages = [bench.queries[query_id].language for query_id in query_ids]

    summary: dict[str, Any] = {"pages": len(images), "queries": len(texts)}

    def save(step: str) -> None:
        """Write the summary so far.

        Args:
            step (str): Last finished step.
        """
        summary["last_step"] = step
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Done: {step}", flush=True)

    model, processor, problems = load_visual_model(
        cfg["model"], cfg["revision"], cfg["dtype"], "cuda:0", cfg["attn_implementation"]
    )
    summary["weights_not_loaded"] = {"count": len(problems), "first": problems[:10]}
    summary["gpu_gb_after_load"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    save("load")
    if problems:
        raise SystemExit(f"{len(problems)} weights did not load cleanly; see summary.json.")

    embed_images(model, processor, images[:1], 1)  # the first pass is slower: keep it out of the timing
    summary["batches"], by_size = probe_batch_sizes(model, processor, images, check["batch_sizes"])
    save("batch sizes")
    if 1 not in by_size:
        raise SystemExit("One page does not fit in GPU memory.")

    single = by_size[1]
    vectors = [len(page) for page in single]
    dim = int(single[0].shape[1])
    summary["dim"] = dim
    summary["vectors_per_page"] = spread(vectors)
    summary["fp16_gb_per_1000_pages"] = round(float(np.mean(vectors)) * dim * 2 * 1000 / 1e9, 2)

    queries_single = embed_queries(model, processor, texts, 1)
    queries_batched = embed_queries(model, processor, texts, cfg["query_batch_size"])
    summary["vectors_per_query"] = spread([len(query) for query in queries_single])
    largest = max(by_size)
    summary["batching_changes_nothing"] = {
        "pages_batch_size": largest,
        "pages": cosine_stats(single, by_size[largest]),
        "queries_batch_size": cfg["query_batch_size"],
        "queries": cosine_stats(queries_single, queries_batched),
    }
    scores = np.array([[maxsim(query, page) for page in single] for query in queries_single])
    summary["retrieval"] = retrieval_check(query_ids, languages, relevant, page_ids, scores)
    save("embeddings")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    pages_ref, queries_ref = images[: check["reference_pages"]], texts[: check["reference_queries"]]
    model, processor, problems = load_visual_model(
        cfg["model"], cfg["revision"], check["reference_dtype"], "cpu", cfg["attn_implementation"]
    )
    if problems:
        raise SystemExit(f"{len(problems)} weights did not load cleanly on CPU.")
    start = time.perf_counter()
    reference_pages = embed_images(model, processor, pages_ref, 1)
    seconds = (time.perf_counter() - start) / len(pages_ref)
    reference_queries = embed_queries(model, processor, queries_ref, 1)
    summary["matches_reference"] = {
        "dtype": check["reference_dtype"],
        "cpu_seconds_per_page": round(seconds, 1),
        "pages": cosine_stats(single, reference_pages),
        "queries": cosine_stats(queries_single, reference_queries),
    }
    save("reference")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
