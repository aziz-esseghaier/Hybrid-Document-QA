"""Shared data types. Ids are prefixed with the dataset key (e.g. `v3_hr/123`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Source document."""

    doc_id: str
    title: str
    url: str | None = None


@dataclass(frozen=True)
class Page:
    """Document page."""

    page_id: str
    doc_id: str
    page_number: int | None = None


@dataclass(frozen=True)
class Query:
    """Query with reference answers."""

    query_id: str
    text: str
    language: str
    answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Qrel:
    """Graded relevance judgment."""

    query_id: str
    page_id: str
    score: int


@dataclass
class Benchmark:
    """Loaded dataset. `page_locations` maps page id -> (parquet file, row) of its image."""

    name: str
    role: str
    documents: dict[str, Document]
    pages: dict[str, Page]
    queries: dict[str, Query]
    qrels: list[Qrel]
    page_locations: dict[str, tuple[Path, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
