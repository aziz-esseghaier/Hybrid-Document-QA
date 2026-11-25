"""ViDoRe loaders (V1 QA, V2, V3) into a common schema. Page images are streamed on demand."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

import pyarrow.parquet as pq

from doc_qa.core.schemas import Benchmark, Document, Page, Qrel, Query

LANGUAGE_CODES = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
}


# ---------------------------------------------------------------- parquet reading


def _files(raw_dir: Path, table: str) -> list[Path]:
    """List a table's test-split parquet files.

    Args:
        raw_dir (Path): Dataset folder.
        table (str): Table name.

    Returns:
        list[Path]: Sorted files.
    """
    files = sorted((raw_dir / table).glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for table '{table}' in {raw_dir}. Run scripts/download_data.py.")
    return files


def _iter_rows(raw_dir: Path, table: str, columns: list[str]) -> Iterator[tuple[dict[str, Any], Path, int]]:
    """Yield a table's rows with their file and row index. Missing columns are skipped.

    Args:
        raw_dir (Path): Dataset folder.
        table (str): Table name.
        columns (list[str]): Columns to read.

    Returns:
        Iterator[tuple[dict[str, Any], Path, int]]: (row, file, row index).
    """
    for path in _files(raw_dir, table):
        present = [c for c in columns if c in pq.read_schema(path).names]
        for i, row in enumerate(pq.read_table(path, columns=present).to_pylist()):
            yield row, path, i


def _read(raw_dir: Path, table: str, columns: list[str]) -> list[dict[str, Any]]:
    """Read a table's rows.

    Args:
        raw_dir (Path): Dataset folder.
        table (str): Table name.
        columns (list[str]): Columns to read.

    Returns:
        list[dict[str, Any]]: Rows.
    """
    return [row for row, _, _ in _iter_rows(raw_dir, table, columns)]


# ---------------------------------------------------------------- field parsing


def language_code(value: Any) -> str:
    """Map a language name to its ISO 639-1 code.

    Args:
        value (Any): Language name or code.

    Returns:
        str: Code, or the lowercased input if unknown.
    """
    text = str(value).strip().lower()
    return LANGUAGE_CODES.get(text, text)


def parse_answers(value: Any) -> tuple[str, ...]:
    """Parse an answer field (None, string, stringified list or list) into distinct non-empty answers.

    Args:
        value (Any): Raw answer field.

    Returns:
        tuple[str, ...]: Answers, in original order.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        items = [text]
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
            except (ValueError, SyntaxError):
                pass
    cleaned = (str(x).strip() for x in items if x is not None)
    return tuple(dict.fromkeys(s for s in cleaned if s))


def v1_ids(image_filename: str) -> tuple[str, str, int | None]:
    """Derive document key, page key and page number from a V1 image path.

    `<doc>.pdf/page_N.jpg` -> the PDF is the document; any other file is its own document.

    Args:
        image_filename (str): The `image_filename` field.

    Returns:
        tuple[str, str, int | None]: (document key, page key, page number).
    """
    path = PurePosixPath(image_filename.replace("\\", "/"))
    parent = path.parent.name
    if parent.lower().endswith(".pdf"):
        match = re.search(r"(\d+)$", path.stem)
        return parent, f"{parent}/{path.stem}", int(match.group(1)) if match else None
    return path.stem, path.stem, None


def _url_stem(url: str | None) -> str:
    """Return a URL's file name without extension.

    Args:
        url (str | None): URL.

    Returns:
        str: Stem, or "" if no URL.
    """
    if not url:
        return ""
    name = PurePosixPath(urlparse(url).path).name
    return name.rsplit(".", 1)[0] if "." in name else name


# ---------------------------------------------------------------- format adapters


def _load_v3(key: str, raw_dir: Path, spec: dict[str, Any]) -> Benchmark:
    """Load a V3 dataset (corpus, queries, qrels, documents_metadata).

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): Dataset folder.
        spec (dict[str, Any]): Dataset entry from datasets.yaml.

    Returns:
        Benchmark: Loaded dataset.
    """
    documents = {}
    for r in _read(raw_dir, "documents_metadata", ["doc_id", "file_name", "url"]):
        doc_id = f"{key}/{r['doc_id']}"
        documents[doc_id] = Document(doc_id, r.get("file_name") or r["doc_id"], r.get("url"))

    pages, locations = {}, {}
    for r, path, i in _iter_rows(raw_dir, "corpus", ["corpus_id", "doc_id", "page_number_in_doc"]):
        page_id = f"{key}/{r['corpus_id']}"
        pages[page_id] = Page(page_id, f"{key}/{r['doc_id']}", r.get("page_number_in_doc"))
        locations[page_id] = (path, i)

    queries = {}
    for r in _read(raw_dir, "queries", ["query_id", "query", "language", "answer"]):
        query_id = f"{key}/{r['query_id']}"
        queries[query_id] = Query(query_id, r["query"], language_code(r["language"]), parse_answers(r.get("answer")))

    qrels = [
        Qrel(f"{key}/{r['query_id']}", f"{key}/{r['corpus_id']}", int(r["score"]))
        for r in _read(raw_dir, "qrels", ["query_id", "corpus_id", "score"])
    ]
    return Benchmark(key, spec["role"], documents, pages, queries, qrels, locations)


def _load_v2(key: str, raw_dir: Path, spec: dict[str, Any]) -> Benchmark:
    """Load a V2 dataset (corpus, queries, qrels, docs). Answers come from qrels.

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): Dataset folder.
        spec (dict[str, Any]): Dataset entry from datasets.yaml.

    Returns:
        Benchmark: Loaded dataset.
    """
    documents = {}
    for r in _read(raw_dir, "docs", ["doc-id"]):
        doc_id = f"{key}/{r['doc-id']}"
        documents[doc_id] = Document(doc_id, r["doc-id"])

    pages, locations = {}, {}
    for r, path, i in _iter_rows(raw_dir, "corpus", ["corpus-id", "doc-id"]):
        page_id = f"{key}/{r['corpus-id']}"
        pages[page_id] = Page(page_id, f"{key}/{r['doc-id']}")
        locations[page_id] = (path, i)

    qrels, answers = [], defaultdict(list)
    for r in _read(raw_dir, "qrels", ["query-id", "corpus-id", "score", "answer"]):
        query_id = f"{key}/{r['query-id']}"
        qrels.append(Qrel(query_id, f"{key}/{r['corpus-id']}", int(r["score"])))
        answers[query_id].extend(parse_answers(r.get("answer")))

    queries = {}
    for r in _read(raw_dir, "queries", ["query-id", "query", "language"]):
        query_id = f"{key}/{r['query-id']}"
        language = language_code(r["language"]) if r.get("language") else spec["language"]
        queries[query_id] = Query(query_id, r["query"], language, tuple(dict.fromkeys(answers[query_id])))
    return Benchmark(key, spec["role"], documents, pages, queries, qrels, locations)


def _load_v2_human(key: str, raw_dir: Path, spec: dict[str, Any]) -> Benchmark:
    """Load the V2 human-labeled ESG dataset (documents identified by PDF URL).

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): Dataset folder.
        spec (dict[str, Any]): Dataset entry from datasets.yaml.

    Returns:
        Benchmark: Loaded dataset.
    """
    documents, pages, locations = {}, {}, {}
    url_to_doc: dict[str, str] = {}
    for r, path, i in _iter_rows(raw_dir, "corpus", ["corpus-id", "pdf_url", "company", "date"]):
        url = r.get("pdf_url") or ""
        if url not in url_to_doc:
            stem = _url_stem(url) or f"doc{len(url_to_doc)}"
            doc_id = f"{key}/{stem}"
            if doc_id in documents:  # same file name, different URL
                doc_id = f"{doc_id}-{len(url_to_doc)}"
            url_to_doc[url] = doc_id
            title = " ".join(str(r[c]) for c in ("company", "date") if r.get(c)) or stem
            documents[doc_id] = Document(doc_id, title, url or None)
        page_id = f"{key}/{r['corpus-id']}"
        pages[page_id] = Page(page_id, url_to_doc[url])
        locations[page_id] = (path, i)

    queries = {}
    for r in _read(raw_dir, "queries", ["query-id", "query", "answer"]):
        query_id = f"{key}/{r['query-id']}"
        queries[query_id] = Query(query_id, r["query"], spec["language"], parse_answers(r.get("answer")))

    qrels = [
        Qrel(f"{key}/{r['query-id']}", f"{key}/{r['corpus-id']}", int(r["score"]))
        for r in _read(raw_dir, "qrels", ["query-id", "corpus-id", "score"])
    ]
    return Benchmark(key, spec["role"], documents, pages, queries, qrels, locations)


def _load_v1_qa(key: str, raw_dir: Path, spec: dict[str, Any]) -> Benchmark:
    """Load a V1 dataset in QA format (single `data` table; has reference answers, unlike BEIR).

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): Dataset folder.
        spec (dict[str, Any]): Dataset entry from datasets.yaml.

    Returns:
        Benchmark: Loaded dataset.
    """
    documents, pages, locations, queries, qrels = {}, {}, {}, {}, []
    for n, (r, path, i) in enumerate(_iter_rows(raw_dir, "data", ["image_filename", "query", "answer"])):
        doc_key, page_key, page_number = v1_ids(r["image_filename"])
        doc_id, page_id = f"{key}/{doc_key}", f"{key}/{page_key}"
        documents.setdefault(doc_id, Document(doc_id, doc_key))
        if page_id not in pages:  # one row per query, so images repeat
            pages[page_id] = Page(page_id, doc_id, page_number)
            locations[page_id] = (path, i)
        if r.get("query"):
            query_id = f"{key}/q{n}"
            queries[query_id] = Query(query_id, r["query"], spec["language"], parse_answers(r.get("answer")))
            qrels.append(Qrel(query_id, page_id, 1))
    return Benchmark(key, spec["role"], documents, pages, queries, qrels, locations)


_ADAPTERS = {"v3": _load_v3, "v2": _load_v2, "v2_human": _load_v2_human, "v1_qa": _load_v1_qa}


# ---------------------------------------------------------------- filtering


def filter_languages(bench: Benchmark, languages: Iterable[str]) -> Benchmark:
    """Keep only queries (and their qrels) in the given languages.

    Args:
        bench (Benchmark): Dataset.
        languages (Iterable[str]): Language codes to keep.

    Returns:
        Benchmark: Filtered copy.
    """
    keep = set(languages)
    queries = {k: q for k, q in bench.queries.items() if q.language in keep}
    qrels = [q for q in bench.qrels if q.query_id in queries]
    return Benchmark(
        bench.name, bench.role, bench.documents, bench.pages, queries, qrels, bench.page_locations, bench.warnings
    )


def apply_exclusions(bench: Benchmark, excluded_doc_ids: Iterable[str]) -> Benchmark:
    """Remove documents, their pages, and every query with a relevant page among them.

    Args:
        bench (Benchmark): Dataset.
        excluded_doc_ids (Iterable[str]): Document ids to remove.

    Returns:
        Benchmark: Filtered copy.
    """
    excluded = set(excluded_doc_ids)
    removed_pages = {pid for pid, page in bench.pages.items() if page.doc_id in excluded}
    dropped = {q.query_id for q in bench.qrels if q.page_id in removed_pages}
    return Benchmark(
        bench.name,
        bench.role,
        {k: d for k, d in bench.documents.items() if k not in excluded},
        {k: p for k, p in bench.pages.items() if k not in removed_pages},
        {k: q for k, q in bench.queries.items() if k not in dropped},
        [q for q in bench.qrels if q.query_id not in dropped],
        {k: loc for k, loc in bench.page_locations.items() if k not in removed_pages},
        bench.warnings,
    )


def clean_qrels(bench: Benchmark) -> Benchmark:
    """Drop non-positive qrels, qrels to missing pages/queries, and unjudged queries; log to `warnings`.

    Args:
        bench (Benchmark): Dataset.

    Returns:
        Benchmark: Cleaned copy.
    """
    warnings = list(bench.warnings)
    qrels = [q for q in bench.qrels if q.score > 0]
    if len(qrels) < len(bench.qrels):
        warnings.append(f"dropped {len(bench.qrels) - len(qrels)} qrels with score <= 0")
    valid = [q for q in qrels if q.page_id in bench.pages and q.query_id in bench.queries]
    if len(valid) < len(qrels):
        warnings.append(f"dropped {len(qrels) - len(valid)} qrels pointing to missing pages or queries")
    judged = {q.query_id for q in valid}
    queries = {k: q for k, q in bench.queries.items() if k in judged}
    if len(queries) < len(bench.queries):
        warnings.append(f"dropped {len(bench.queries) - len(queries)} queries with no relevant page")
    return Benchmark(
        bench.name, bench.role, bench.documents, bench.pages, queries, valid, bench.page_locations, warnings
    )


# ---------------------------------------------------------------- public API


def load_benchmark(
    key: str,
    spec: dict[str, Any],
    raw_root: Path,
    languages: Iterable[str] = ("en", "fr"),
    excluded_doc_ids: Iterable[str] = (),
) -> Benchmark:
    """Load, clean and filter one dataset.

    Args:
        key (str): Dataset key.
        spec (dict[str, Any]): Dataset entry from datasets.yaml.
        raw_root (Path): Folder with one subfolder per dataset key.
        languages (Iterable[str]): Query languages to keep.
        excluded_doc_ids (Iterable[str]): Document ids to remove.

    Returns:
        Benchmark: Loaded dataset.
    """
    if spec["format"] not in _ADAPTERS:
        raise ValueError(f"Unknown format '{spec['format']}' for dataset '{key}'.")
    bench = _ADAPTERS[spec["format"]](key, raw_root / key, spec)
    bench = clean_qrels(bench)
    for warning in bench.warnings:
        print(f"WARNING {key}: {warning}")
    bench = filter_languages(bench, languages)
    return apply_exclusions(bench, excluded_doc_ids)


def load_benchmarks(
    specs: dict[str, dict[str, Any]],
    raw_root: Path,
    languages: Iterable[str] = ("en", "fr"),
    excluded_doc_ids: Iterable[str] = (),
    role: str | None = None,
) -> list[Benchmark]:
    """Load several datasets. Exclusions apply to dev only.

    Args:
        specs (dict[str, dict[str, Any]]): `datasets` section of datasets.yaml.
        raw_root (Path): Folder with one subfolder per dataset key.
        languages (Iterable[str]): Query languages to keep.
        excluded_doc_ids (Iterable[str]): Dev document ids to remove.
        role (str | None): "dev", "test" or None for both.

    Returns:
        list[Benchmark]: Datasets, in config order.
    """
    languages, excluded = tuple(languages), tuple(excluded_doc_ids)
    return [
        load_benchmark(key, spec, raw_root, languages, excluded if spec["role"] == "dev" else ())
        for key, spec in specs.items()
        if role is None or spec["role"] == role
    ]


def iter_page_images(
    bench: Benchmark, page_ids: Iterable[str] | None = None, batch_size: int = 32
) -> Iterator[tuple[str, bytes]]:
    """Stream encoded page images from the parquet files.

    Args:
        bench (Benchmark): Dataset.
        page_ids (Iterable[str] | None): Pages to yield; None = all.
        batch_size (int): Rows read at once.

    Returns:
        Iterator[tuple[str, bytes]]: (page id, image bytes), in file order.
    """
    for page_id, value in _iter_page_values(bench, "image", page_ids, batch_size):
        data = value["bytes"]
        if data is None:
            raise ValueError(f"{page_id}: image has no embedded bytes.")
        yield page_id, data


def iter_page_markdown(
    bench: Benchmark, page_ids: Iterable[str] | None = None, batch_size: int = 256
) -> Iterator[tuple[str, str]]:
    """Stream the text that ViDoRe V3 ships with each page (its `markdown` column).

    Args:
        bench (Benchmark): A V3 dataset.
        page_ids (Iterable[str] | None): Pages to yield; None = all.
        batch_size (int): Rows read at once.

    Returns:
        Iterator[tuple[str, str]]: (page id, text), in file order; a page without text gives "".
    """
    for page_id, value in _iter_page_values(bench, "markdown", page_ids, batch_size):
        yield page_id, value or ""


def _iter_page_values(
    bench: Benchmark, column: str, page_ids: Iterable[str] | None, batch_size: int
) -> Iterator[tuple[str, Any]]:
    """Stream one corpus column for the chosen pages.

    Args:
        bench (Benchmark): Dataset.
        column (str): Corpus column.
        page_ids (Iterable[str] | None): Pages to yield; None = all.
        batch_size (int): Rows read at once.

    Returns:
        Iterator[tuple[str, Any]]: (page id, value), in file order.
    """
    wanted = set(page_ids) if page_ids is not None else None
    rows_by_file: dict[Path, dict[int, str]] = defaultdict(dict)
    for page_id, (path, row) in bench.page_locations.items():
        if wanted is None or page_id in wanted:
            rows_by_file[path][row] = page_id

    for path, rows in rows_by_file.items():
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=[column]):
            values = batch.column(0)
            for j in range(batch.num_rows):
                page_id = rows.get(offset + j)
                if page_id is not None:
                    yield page_id, values[j].as_py()
            offset += batch.num_rows
