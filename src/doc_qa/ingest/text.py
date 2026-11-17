"""OCR text cleaning and page-level token chunking."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")
_BLANK_LINES = re.compile(r"\n{3,}")
# LaTeX spans: $$...$$, or $...$ on one line containing a LaTeX marker (so "$5 and $3" is left alone).
_MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]*[_^\\{][^$\n]*)\$", re.DOTALL)
_MATH_WRAPPER = re.compile(r"\\(?:mathrm|mathbf|mathit|text|operatorname)\b\s*")
_MATH_ESCAPE = re.compile(r"\\([%&#$])")
_MATH_MARKUP = re.compile(r"[_^{}]")


class _TableParser(HTMLParser):
    """Collect the cells of an HTML table, row by row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def _close_cell(self) -> None:
        if self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._close_cell()
            self.rows.append([])
        elif tag in ("td", "th"):
            self._close_cell()
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th", "tr", "table"):
            self._close_cell()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def table_to_text(table_html: str) -> str:
    """Convert an HTML table to one line per row, cells separated by ` | `.

    Args:
        table_html (str): `<table>...</table>` markup.

    Returns:
        str: Plain-text rows; empty rows dropped.
    """
    parser = _TableParser()
    parser.feed(table_html)
    parser.close()
    parser._close_cell()
    return "\n".join(" | ".join(row) for row in parser.rows if any(row))


def plain_math(expr: str) -> str:
    """Flatten simple LaTeX to text, e.g. `CO_{{2}}e` -> `CO2e`. Other commands are kept.

    Args:
        expr (str): LaTeX without the `$` delimiters.

    Returns:
        str: Plain text.
    """
    expr = _MATH_ESCAPE.sub(r"\1", _MATH_WRAPPER.sub("", expr))
    return " ".join(_MATH_MARKUP.sub("", expr).split())


def _plain_math_and_newlines(text: str) -> str:
    """Flatten LaTeX spans, and turn literal `\\n` outside them into line breaks.

    Args:
        text (str): Raw OCR Markdown.

    Returns:
        str: Text without LaTeX delimiters or literal `\\n`.
    """
    parts, last = [], 0
    for m in _MATH.finditer(text):
        parts.append(text[last : m.start()].replace("\\n", "\n"))
        parts.append(plain_math(m.group(1) or m.group(2)))
        last = m.end()
    parts.append(text[last:].replace("\\n", "\n"))
    return "".join(parts)


def clean_ocr_text(text: str) -> str:
    """Turn PaddleOCR-VL Markdown into plain text.

    Drops image placeholders and tags, flattens tables to rows and simple LaTeX to text.

    Args:
        text (str): Raw OCR Markdown.

    Returns:
        str: Cleaned text.
    """
    text = _plain_math_and_newlines(text)
    text = _TABLE.sub(lambda m: "\n" + table_to_text(m.group()) + "\n", text)
    text = _MD_IMG.sub("", _IMG.sub("", text))
    text = _TAG.sub("", _BR.sub("\n", text))
    text = html.unescape(text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def token_windows(num_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Split a token sequence into overlapping windows.

    Args:
        num_tokens (int): Sequence length.
        size (int): Max tokens per window.
        overlap (int): Tokens shared by consecutive windows.

    Returns:
        list[tuple[int, int]]: (start, end) token indices; the last window ends at `num_tokens`.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"Need 0 <= overlap < size, got overlap={overlap}, size={size}")
    windows, start = [], 0
    while start < num_tokens:
        end = min(start + size, num_tokens)
        windows.append((start, end))
        if end == num_tokens:
            break
        start += size - overlap
    return windows


def chunk_text(text: str, offsets: list[tuple[int, int]], size: int, overlap: int) -> list[dict]:
    """Split text into overlapping token windows.

    Args:
        text (str): Page text.
        offsets (list[tuple[int, int]]): Character span of each token in `text`.
        size (int): Max tokens per chunk.
        overlap (int): Tokens shared by consecutive chunks.

    Returns:
        list[dict]: Chunks with `text`, `start`, `end` (character offsets) and `tokens`.
    """
    chunks = []
    for s, e in token_windows(len(offsets), size, overlap):
        start, end = offsets[s][0], offsets[e - 1][1]
        chunks.append({"text": text[start:end], "start": start, "end": end, "tokens": e - s})
    return chunks
