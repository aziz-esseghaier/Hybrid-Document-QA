"""Answer metrics: how often the model answers, abstains or fails, citation precision, and judged accuracy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


def sample_clusters(clusters: dict[str, Hashable], count: int, seed: int) -> list[str]:
    """Draw questions at random by cluster, so that a question comes with its translations.

    Args:
        clusters (dict[str, Hashable]): Cluster per query id, e.g. the query's relevant pages.
        count (int): Clusters to draw; all of them when there are fewer.
        seed (int): Random seed.

    Returns:
        list[str]: Query ids of the drawn clusters, sorted.
    """
    names = sorted(set(clusters.values()), key=repr)
    picks = np.random.default_rng(seed).choice(len(names), size=min(count, len(names)), replace=False)
    chosen = {names[i] for i in picks}
    return sorted(query_id for query_id, cluster in clusters.items() if cluster in chosen)


@dataclass(frozen=True)
class AnswerRecord:
    """One question's answering outcome. `found` is None when no usable reply came back."""

    language: str
    given: tuple[str, ...]
    relevance: dict[str, int]
    found: bool | None
    cited: tuple[str, ...]
    attempts: int
    seconds: float
    correct: bool | None = None  # judge verdict, once judged


def citation_precision(cited: Sequence[str], relevance: dict[str, int]) -> float | None:
    """Share of cited pages that are relevant.

    Args:
        cited (Sequence[str]): Cited page ids.
        relevance (dict[str, int]): Graded relevance per page id.

    Returns:
        float | None: Precision in [0, 1]; None if nothing was cited.
    """
    if not cited:
        return None
    return sum(relevance.get(page, 0) > 0 for page in cited) / len(cited)


def _mean(values: Sequence[float]) -> float | None:
    """Mean rounded to 4 decimals, or None for no values."""
    return round(float(np.mean(values)), 4) if len(values) else None


def summarize_group(records: Sequence[AnswerRecord]) -> dict[str, float | int | None]:
    """Summarize the answering outcomes of a group of questions.

    Args:
        records (Sequence[AnswerRecord]): Outcomes.

    Returns:
        dict[str, float | int | None]: Counts and rates; rates over an empty subset are None.
    """
    given_relevant = [any(r.relevance.get(p, 0) > 0 for p in r.given) for r in records]
    missing = [r for r, has in zip(records, given_relevant) if not has]
    answered = [r for r in records if r.found]
    precisions = [p for r in answered if (p := citation_precision(r.cited, r.relevance)) is not None]
    judged = [r.correct for r in records if r.correct is not None]
    return {
        "questions": len(records),
        "relevant_page_given": _mean(given_relevant),  # upper bound set by retrieval
        "found": _mean([r.found is True for r in records]),
        "not_found": _mean([r.found is False for r in records]),
        "failed": _mean([r.found is None for r in records]),
        "not_found_when_no_relevant_page": _mean([r.found is False for r in missing]),
        "citation_precision": _mean(precisions),
        "retried": _mean([r.attempts > 1 for r in records]),
        "seconds_per_question": _mean([r.seconds for r in records]),
        "judged": len(judged),
        "accuracy": _mean(judged),
    }


def summarize_answers(records: Sequence[AnswerRecord]) -> dict[str, dict[str, float | int | None]]:
    """Summarize answering over all questions and per question language.

    Args:
        records (Sequence[AnswerRecord]): Outcomes.

    Returns:
        dict[str, dict[str, float | int | None]]: {"all" or language: summary}.
    """
    by_language: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        by_language[record.language].append(record)
    groups = {language: summarize_group(group) for language, group in sorted(by_language.items())}
    return {"all": summarize_group(records), **groups}
