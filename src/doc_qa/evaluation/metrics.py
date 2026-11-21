"""Retrieval metrics: NDCG@k (linear gain, as trec_eval), Recall@k, bootstrap intervals. Text overlap and coverage."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Hashable, Sequence

import numpy as np

_WORD = re.compile(r"\w+")


def cohen_kappa(first: Sequence[Hashable], second: Sequence[Hashable]) -> float:
    """Agreement between two raters beyond what their label frequencies give by chance.

    Args:
        first (Sequence[Hashable]): Labels from one rater.
        second (Sequence[Hashable]): Labels from the other, same items.

    Returns:
        float: Kappa: 1 for perfect agreement, 0 for chance level; 1 if both use a single label throughout.
    """
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    counts_first, counts_second = Counter(first), Counter(second)
    expected = sum(counts_first[label] * counts_second[label] for label in counts_first) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def token_f1(a: str, b: str) -> float:
    """F1 of the lowercased word multisets of two texts.

    Args:
        a (str): First text.
        b (str): Second text.

    Returns:
        float: F1 in [0, 1]; 1 if both are empty.
    """
    ca, cb = Counter(_WORD.findall(a.lower())), Counter(_WORD.findall(b.lower()))
    if not ca and not cb:
        return 1.0
    common = sum((ca & cb).values())
    return 2 * common / (sum(ca.values()) + sum(cb.values()))


def text_coverage(qrels: dict[str, dict[str, int]], page_texts: dict[str, str], min_chars: int = 20) -> dict[str, int]:
    """Count the pages and queries that have almost no text to search.

    Args:
        qrels (dict[str, dict[str, int]]): Relevance per query id.
        page_texts (dict[str, str]): Text per page id.
        min_chars (int): Pages with fewer characters count as near-empty.

    Returns:
        dict[str, int]: Page and relevant-page counts, and queries whose relevant pages are all near-empty.
    """
    near_empty = {p for p, text in page_texts.items() if len(text.strip()) < min_chars}
    relevant = {q: {p for p, g in pages.items() if g > 0} for q, pages in qrels.items()}
    relevant_pages = set().union(*relevant.values()) if relevant else set()
    return {
        "pages": len(page_texts),
        "near_empty_pages": len(near_empty),
        "relevant_pages": len(relevant_pages),
        "near_empty_relevant_pages": len(relevant_pages & near_empty),
        "queries": len(relevant),
        "queries_only_near_empty": sum(1 for pages in relevant.values() if pages and pages <= near_empty),
    }


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[Hashable],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap interval, resampling whole clusters.

    Args:
        values (Sequence[float]): One value per query.
        clusters (Sequence[Hashable]): Cluster of each value (e.g. translations of one question).
        n_resamples (int): Bootstrap samples.
        confidence (float): Interval coverage.
        seed (int): Random seed.

    Returns:
        tuple[float, float, float]: (mean, lower bound, upper bound).
    """
    index = {c: i for i, c in enumerate(sorted(set(clusters), key=repr))}
    sums, counts = np.zeros(len(index)), np.zeros(len(index))
    for value, cluster in zip(values, clusters):
        sums[index[cluster]] += value
        counts[index[cluster]] += 1
    picks = np.random.default_rng(seed).integers(len(index), size=(n_resamples, len(index)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def ndcg_at_k(ranking: list[str], relevance: dict[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain of the top `k`.

    Args:
        ranking (list[str]): Ranked page ids, best first.
        relevance (dict[str, int]): Graded relevance per page id.
        k (int): Cutoff.

    Returns:
        float: NDCG in [0, 1]; 0 if nothing is relevant.
    """
    dcg = sum(relevance.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranking[:k]))
    ideal = sorted((g for g in relevance.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: list[str], relevance: dict[str, int], k: int = 100) -> float:
    """Share of relevant pages found in the top `k`.

    Args:
        ranking (list[str]): Ranked page ids, best first.
        relevance (dict[str, int]): Graded relevance per page id.
        k (int): Cutoff.

    Returns:
        float: Recall in [0, 1]; 0 if nothing is relevant.
    """
    relevant = {p for p, g in relevance.items() if g > 0}
    return len(relevant & set(ranking[:k])) / len(relevant) if relevant else 0.0


def evaluate(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    groups: dict[str, str],
    recall_ks: Sequence[int] = (100,),
) -> dict[str, dict[str, float]]:
    """Mean NDCG@10 and Recall@k per group (e.g. query language) and over all queries.

    Args:
        rankings (dict[str, list[str]]): Ranked page ids per query id.
        qrels (dict[str, dict[str, int]]): Relevance per query id.
        groups (dict[str, str]): Group per query id.
        recall_ks (Sequence[int]): Recall cutoffs.

    Returns:
        dict[str, dict[str, float]]: {group or "all": {"queries", "ndcg@10", "recall@<k>", ...}}.
    """
    names = ["ndcg@10", *(f"recall@{k}" for k in recall_ks)]
    scores: dict[str, list[list[float]]] = defaultdict(list)
    for query_id, relevance in qrels.items():
        ranking = rankings.get(query_id, [])
        row = [ndcg_at_k(ranking, relevance, 10), *(recall_at_k(ranking, relevance, k) for k in recall_ks)]
        scores[groups[query_id]].append(row)
        scores["all"].append(row)
    return {
        group: {
            "queries": len(rows),
            **{name: round(sum(row[i] for row in rows) / len(rows), 4) for i, name in enumerate(names)},
        }
        for group, rows in sorted(scores.items())
    }
