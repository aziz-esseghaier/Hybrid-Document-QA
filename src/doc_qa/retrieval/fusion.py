"""Chunk-to-page aggregation and reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def page_ranking(hits: Iterable[tuple[str, float]], limit: int | None = None) -> list[tuple[str, float]]:
    """Rank pages by their best chunk score.

    Args:
        hits (Iterable[tuple[str, float]]): (page id, chunk score).
        limit (int | None): Max pages; None = all.

    Returns:
        list[tuple[str, float]]: (page id, score), best first; ties broken by id.
    """
    best: dict[str, float] = {}
    for page_id, score in hits:
        if page_id not in best or score > best[page_id]:
            best[page_id] = score
    ranked = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit] if limit else ranked


def rrf(rankings: list[list[str]], k: int = 60, limit: int | None = None) -> list[tuple[str, float]]:
    """Merge rankings with reciprocal rank fusion: score = sum of 1 / (k + rank).

    Args:
        rankings (list[list[str]]): Ranked ids, best first.
        k (int): Damping constant; larger values flatten the rank weights.
        limit (int | None): Max ids; None = all.

    Returns:
        list[tuple[str, float]]: (id, fused score), best first; ties broken by id.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit] if limit else ranked
