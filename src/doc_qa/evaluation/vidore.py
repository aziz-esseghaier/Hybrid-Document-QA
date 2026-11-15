"""Load the ViDoRe datasets (V1 QA format, V2, V3) into one common schema.

Each dataset is a folder of parquet files downloaded by scripts/download_data.py, with one subfolder per
table (`corpus`, `queries`, `qrels`, ...). Only metadata columns are read at load time; page images are
streamed on demand with `iter_page_images`, because ~25k decoded pages don't fit in RAM.
"""

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
    """List the parquet files of one table (test split only).

    Args:
        raw_dir (Path): The dataset's download folder.
        table (str): Table name, such as `corpus` or `queries`.

    Returns:
        list[Path]: Sorted parquet files.
    """
    files = sorted((raw_dir / table).glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files for table '{table}' in {raw_dir}. Run scripts/download_data.py.")
    return files


def _iter_rows(raw_dir: Path, table: str, columns: list[str]) -> Iterator[tuple[dict[str, Any], Path, int]]:
    """Yield the rows of one table, with the file and row index each row comes from.

    Columns missing from the files are skipped (their key is absent from the row).

    Args:
        raw_dir (Path): The dataset's download folder.
        table (str): Table name, such as `corpus` or `queries`.
        columns (list[str]): Columns to read.

    Returns:
        Iterator[tuple[dict[str, Any], Path, int]]: (row, parquet file, row index in that file).
    """
    for path in _files(raw_dir, table):
        present = [c for c in columns if c in pq.read_schema(path).names]
        for i, row in enumerate(pq.read_table(path, columns=present).to_pylist()):
            yield row, path, i


def _read(raw_dir: Path, table: str, columns: list[str]) -> list[dict[str, Any]]:
    """Read the rows of one table.

    Args:
        raw_dir (Path): The dataset's download folder.
        table (str): Table name, such as `queries`.
        columns (list[str]): Columns to read.

    Returns:
        list[dict[str, Any]]: One dict per row.
    """
    return [row for row, _, _ in _iter_rows(raw_dir, table, columns)]


# ---------------------------------------------------------------- field parsing


def language_code(value: Any) -> str:
    """Convert a language name such as `french` into its ISO code (`fr`).

    Args:
        value (Any): Language name or code from the dataset.

    Returns:
        str: Two-letter code, or the lowercased input when it is unknown.
    """
    text = str(value).strip().lower()
    return LANGUAGE_CODES.get(text, text)


def parse_answers(value: Any) -> tuple[str, ...]:
    """Turn a dataset's answer field into a tuple of distinct, non-empty answers.

    V1 stores lists as strings such as "['a', 'b']"; V2 and V3 store plain strings.

    Args:
        value (Any): The raw answer field (None, string, string-encoded list, or list).

    Returns:
        tuple[str, ...]: The answers, in their original order.
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
    """Derive the document key, page key and page number from a ViDoRe V1 image file name.

    Shift Project names look like `.../<report>.pdf/page_130.jpg`: the document is the PDF and the page
    number comes from the file name. TabFQuAD names look like `.../2315.png`: each image is its own document.

    Args:
        image_filename (str): The `image_filename` field.

    Returns:
        tuple[str, str, int | None]: (document key, page key, page number or None).
    """
    path = PurePosixPath(image_filename.replace("\\", "/"))
    parent = path.parent.name
    if parent.lower().endswith(".pdf"):
        match = re.search(r"(\d+)$", path.stem)
        return parent, f"{parent}/{path.stem}", int(match.group(1)) if match else None
    return path.stem, path.stem, None


def _url_stem(url: str | None) -> str:
    """Get the file name of a URL, without its extension.

    Args:
        url (str | None): A URL such as `https://x.com/reports/esg_2023.pdf`.

    Returns:
        str: The stem (`esg_2023`), or an empty string.
    """
    if not url:
        return ""
    name = PurePosixPath(urlparse(url).path).name
    return name.rsplit(".", 1)[0] if "." in name else name


# ---------------------------------------------------------------- format adapters


def _load_v3(key: str, raw_dir: Path, spec: dict[str, Any]) -> Benchmark:
    """Load a ViDoRe V3 dataset (tables: corpus, queries, qrels, documents_metadata).

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): The dataset's download folder.
        spec (dict[str, Any]): The dataset's entry in configs/datasets.yaml.

    Returns:
        Benchmark: The loaded dataset.
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
    """Load a ViDoRe V2 dataset (tables: corpus, queries, qrels, docs). Answers live in qrels.

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): The dataset's download folder.
        spec (dict[str, Any]): The dataset's entry in configs/datasets.yaml.

    Returns:
        Benchmark: The loaded dataset.
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
    """Load ViDoRe V2's human-labeled ESG dataset, whose corpus names documents by PDF URL, not doc id.

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): The dataset's download folder.
        spec (dict[str, Any]): The dataset's entry in configs/datasets.yaml.

    Returns:
        Benchmark: The loaded dataset.
    """
    documents, pages, locations = {}, {}, {}
    url_to_doc: dict[str, str] = {}
    for r, path, i in _iter_rows(raw_dir, "corpus", ["corpus-id", "pdf_url", "company", "date"]):
        url = r.get("pdf_url") or ""
        if url not in url_to_doc:
            stem = _url_stem(url) or f"doc{len(url_to_doc)}"
            doc_id = f"{key}/{stem}"
            if doc_id in documents:  # two URLs with the same file name
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
    """Load a ViDoRe V1 dataset in QA format: one table, one row per (image, optional query).

    The QA format is used instead of the BEIR one because only it has the reference answers.

    Args:
        key (str): Dataset key, used as id prefix.
        raw_dir (Path): The dataset's download folder.
        spec (dict[str, Any]): The dataset's entry in configs/datasets.yaml.

    Returns:
        Benchmark: The loaded dataset.
    """
    documents, pages, locations, queries, qrels = {}, {}, {}, {}, []
    for n, (r, path, i) in enumerate(_iter_rows(raw_dir, "data", ["image_filename", "query", "answer"])):
        doc_key, page_key, page_number = v1_ids(r["image_filename"])
        doc_id, page_id = f"{key}/{doc_key}", f"{key}/{page_key}"
        documents.setdefault(doc_id, Document(doc_id, doc_key))
        if page_id not in pages:  # the same image appears once per query
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
    """Keep only the queries in the given languages, and their relevance judgments.

    Args:
        bench (Benchmark): The dataset.
        languages (Iterable[str]): Language codes to keep, such as ["en", "fr"].

    Returns:
        Benchmark: A filtered copy. Documents and pages are unchanged.
    """
    keep = set(languages)
    queries = {k: q for k, q in bench.queries.items() if q.language in keep}
    qrels = [q for q in bench.qrels if q.query_id in queries]
    return Benchmark(
        bench.name, bench.role, bench.documents, bench.pages, queries, qrels, bench.page_locations, bench.warnings
    )


def apply_exclusions(bench: Benchmark, excluded_doc_ids: Iterable[str]) -> Benchmark:
    """Remove whole documents, their pages, and every query that points to one of those pages.

    A query is dropped if any of its relevant pages is removed: keeping it with fewer relevant pages
    would change what counts as a correct retrieval.

    Args:
        bench (Benchmark): The dataset.
        excluded_doc_ids (Iterable[str]): Document ids to remove (`<dataset key>/<doc id>`).

    Returns:
        Benchmark: A filtered copy.
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
    """Drop judgments that can't be used, and record why in `bench.warnings`.

    Judgments with score <= 0 mean "not relevant" and add nothing to NDCG. Judgments pointing to pages or
    queries missing from the dataset would make metrics wrong.

    Args:
        bench (Benchmark): The dataset.

    Returns:
        Benchmark: A copy with the usable judgments only.
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
    """Load one dataset, keep the queries in the given languages, and remove excluded documents.

    Args:
        key (str): Dataset key from configs/datasets.yaml.
        spec (dict[str, Any]): The dataset's entry in configs/datasets.yaml.
        raw_root (Path): Folder holding one download folder per dataset key.
        languages (Iterable[str]): Query language codes to keep.
        excluded_doc_ids (Iterable[str]): Document ids to remove.

    Returns:
        Benchmark: The loaded dataset.
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
    """Load several datasets. Exclusions apply to dev datasets only.

    Args:
        specs (dict[str, dict[str, Any]]): The `datasets` section of configs/datasets.yaml.
        raw_root (Path): Folder holding one download folder per dataset key.
        languages (Iterable[str]): Query language codes to keep.
        excluded_doc_ids (Iterable[str]): Dev document ids to remove.
        role (str | None): Load only `dev` or only `test` datasets; None loads both.

    Returns:
        list[Benchmark]: The loaded datasets, in config order.
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
    """Stream encoded page images (PNG or JPEG bytes) from the parquet files.

    Args:
        bench (Benchmark): The dataset.
        page_ids (Iterable[str] | None): Pages to yield; None yields every page.
        batch_size (int): Rows read at once. Bounds memory use.

    Returns:
        Iterator[tuple[str, bytes]]: (page id, image bytes), in file order.
    """
    wanted = set(page_ids) if page_ids is not None else None
    rows_by_file: dict[Path, dict[int, str]] = defaultdict(dict)
    for page_id, (path, row) in bench.page_locations.items():
        if wanted is None or page_id in wanted:
            rows_by_file[path][row] = page_id

    for path, rows in rows_by_file.items():
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["image"]):
            images = batch.column(0)
            for j in range(batch.num_rows):
                page_id = rows.get(offset + j)
                if page_id is not None:
                    data = images[j].as_py()["bytes"]
                    if data is None:
                        raise ValueError(f"{page_id}: image has no embedded bytes.")
                    yield page_id, data
            offset += batch.num_rows
