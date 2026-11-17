"""Retrieval metrics: NDCG@k (linear gain, as trec_eval) and Recall@k."""

from __future__ import annotations

import math
from collections import defaultdict


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
    rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], groups: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Mean NDCG@10 and Recall@100 per group (e.g. query language) and over all queries.

    Args:
        rankings (dict[str, list[str]]): Ranked page ids per query id.
        qrels (dict[str, dict[str, int]]): Relevance per query id.
        groups (dict[str, str]): Group per query id.

    Returns:
        dict[str, dict[str, float]]: {group or "all": {"queries", "ndcg@10", "recall@100"}}.
    """
    scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for query_id, relevance in qrels.items():
        ranking = rankings.get(query_id, [])
        pair = (ndcg_at_k(ranking, relevance, 10), recall_at_k(ranking, relevance, 100))
        scores[groups[query_id]].append(pair)
        scores["all"].append(pair)
    return {
        group: {
            "queries": len(pairs),
            "ndcg@10": round(sum(p[0] for p in pairs) / len(pairs), 4),
            "recall@100": round(sum(p[1] for p in pairs) / len(pairs), 4),
        }
        for group, pairs in sorted(scores.items())
    }
