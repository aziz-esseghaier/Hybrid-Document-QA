"""Data types shared by the whole pipeline.

Every id is prefixed with its dataset key (for example `v3_hr/123`), so ids stay unique across datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source PDF (or, for TabFQuAD, a single table image)."""

    doc_id: str
    title: str
    url: str | None = None


@dataclass(frozen=True)
class Page:
    """One page of a document. `page_number` is None when the dataset doesn't give it."""

    page_id: str
    doc_id: str
    page_number: int | None = None


@dataclass(frozen=True)
class Query:
    """A question with its reference answers (empty when the dataset has none)."""

    query_id: str
    text: str
    language: str
    answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Qrel:
    """A relevance judgment: page `page_id` is relevant to query `query_id`, with a graded score."""

    query_id: str
    page_id: str
    score: int


@dataclass
class Benchmark:
    """One loaded dataset: documents, pages, queries and relevance judgments.

    Page images are not held in memory. `page_locations` maps each page id to the parquet file and row
    that hold its image, so images can be streamed when needed. `warnings` lists data problems found
    while loading, for the reports.
    """

    name: str
    role: str
    documents: dict[str, Document]
    pages: dict[str, Page]
    queries: dict[str, Query]
    qrels: list[Qrel]
    page_locations: dict[str, tuple[Path, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
