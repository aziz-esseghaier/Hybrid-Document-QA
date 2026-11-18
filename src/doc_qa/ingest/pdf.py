"""PDF download, text layer, text-layer quality signals and page rendering with pypdfium2.

Requires `pypdfium2`. PDFium is not thread-safe: use processes, not threads.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

_HYPHEN_BREAK = re.compile(r"[\x02\xad]\s*")  # PDFium's end-of-line hyphen marker, soft hyphen
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f￾￿]")
_SPACES = re.compile(r"[ \t\xa0]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_WORD_CHAR = re.compile(r"\w")
_LETTER_RUN = re.compile(r"[^\W\d_]+")

_NORMAL_CONTROLS = frozenset("\t\n\r\x0b\x0c\x02")
_UNUSABLE_CATEGORIES = frozenset({"Co", "Cn", "Cs"})  # private use, unassigned, surrogate
_LONG_WORD = 25  # letters; longer runs are usually words merged by lost spaces
_MIN_SPACED_RUN = 4  # consecutive one-letter tokens
COMMON_WORDS = frozenset(
    # English
    "the of and to in is that for on with as are be by this it at from or an was we our has have not which their "
    "will its "
    # French
    "le la les de des du un une et en au aux est que qui dans pour sur par avec ce cette ces sont plus nous ou il "
    "elle ne pas se sa son ses à".split()
)


def download_pdf(url: str, path: Path, timeout: float) -> str:
    """Download a PDF unless already on disk.

    Args:
        url (str): PDF URL.
        path (Path): Output file.
        timeout (float): Seconds.

    Returns:
        str: "ok", "not a PDF", or the error.
    """
    if path.exists():
        return "ok"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return type(e).__name__
    if not data.startswith(b"%PDF"):
        return "not a PDF"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "ok"


def clean_pdf_text(text: str) -> str:
    """Normalize PDFium text: line breaks, hyphenation markers, control characters, spaces.

    Args:
        text (str): Raw text layer of one page.

    Returns:
        str: Cleaned text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", _HYPHEN_BREAK.sub("", text))
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def word_chars(text: str) -> int:
    """Count letters and digits, to tell a real text layer from an empty or image-only page.

    Args:
        text (str): Page text.

    Returns:
        int: Number of word characters.
    """
    return len(_WORD_CHAR.findall(text))


def _is_unusable(char: str) -> bool:
    """Tell whether a character means the PDF lost the real text.

    Args:
        char (str): One character.

    Returns:
        bool: True for replacement, private-use, unassigned and stray control characters.
    """
    category = unicodedata.category(char)
    return char == "�" or category in _UNUSABLE_CATEGORIES or (category == "Cc" and char not in _NORMAL_CONTROLS)


def _spaced_tokens(tokens: list[str]) -> int:
    """Count tokens inside runs of one-letter tokens, as in "S c o p e".

    Args:
        tokens (list[str]): Whitespace-separated tokens.

    Returns:
        int: Tokens in runs of at least `_MIN_SPACED_RUN`.
    """
    total = run = 0
    for token in [*tokens, ""]:
        if len(token) == 1 and token.isalpha():
            run += 1
            continue
        if run >= _MIN_SPACED_RUN:
            total += run
        run = 0
    return total


def text_signals(raw: str) -> dict[str, float]:
    """Measure signs that a page's text layer is unusable, from the text alone.

    Args:
        raw (str): Raw text layer of one page.

    Returns:
        dict[str, float]: Counts `word_chars`, `letters` and `distinct_words`, and the shares `bad_char_share`,
            `symbol_share`, `spaced_letter_share`, `long_word_share` (of letters, in very long words) and
            `common_word_share` (of words, English and French function words).
    """
    text = clean_pdf_text(raw)
    raw_chars = [c for c in raw if not c.isspace() and c != "\x02"]
    chars = [c for c in text if not c.isspace()]
    tokens = text.split()
    words = _LETTER_RUN.findall(text.lower())
    letters = sum(len(w) for w in words)

    def share(count: int, total: int) -> float:
        return round(count / total, 4) if total else 0.0

    return {
        "word_chars": word_chars(text),
        "letters": letters,
        "distinct_words": len(set(words)),
        "bad_char_share": share(sum(_is_unusable(c) for c in raw_chars), len(raw_chars)),
        "symbol_share": share(sum(unicodedata.category(c)[0] in "PS" for c in chars), len(chars)),
        "spaced_letter_share": share(_spaced_tokens(tokens), len(tokens)),
        "long_word_share": share(sum(len(w) for w in words if len(w) >= _LONG_WORD), letters),
        "common_word_share": share(sum(w in COMMON_WORDS for w in words), len(words)),
    }


def garbled_reasons(signals: dict[str, float], rule: dict[str, Any]) -> list[str]:
    """List the checks a page's text layer fails. Checks missing from `rule` are skipped.

    Args:
        signals (dict[str, float]): Output of `text_signals`.
        rule (dict[str, Any]): Thresholds: `max_bad_char_share`, `max_symbol_share`, `max_spaced_letter_share`,
            `max_long_word_share` (pages with at least `min_letters` letters) and `min_common_word_share` (pages with
            at least `min_distinct_words` different words, so tables of repeated labels pass).

    Returns:
        list[str]: Names of the failed signals; empty if the text looks usable.
    """
    reasons = []
    for signal in ("bad_char_share", "symbol_share", "spaced_letter_share"):
        limit = rule.get(f"max_{signal}")
        if limit is not None and signals[signal] > limit:
            reasons.append(signal)
    limit = rule.get("max_long_word_share")
    if limit is not None and signals["letters"] >= rule.get("min_letters", 0) and signals["long_word_share"] > limit:
        reasons.append("long_word_share")
    limit = rule.get("min_common_word_share")
    if (
        limit is not None
        and signals["distinct_words"] >= rule.get("min_distinct_words", 0)
        and signals["common_word_share"] < limit
    ):
        reasons.append("common_word_share")
    return reasons


def read_text_layers(path: Path) -> list[str]:
    """Extract the text layer of every page, without rendering.

    Args:
        path (Path): PDF file.

    Returns:
        list[str]: Raw text per page.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[index].get_textpage().get_text_range() for index in range(len(pdf))]
    finally:
        pdf.close()


def read_pdf(path: Path, scale: float = 1.0) -> Iterator[tuple[int, Image.Image, str]]:
    """Render each page and extract its text layer.

    Args:
        path (Path): PDF file.
        scale (float): Render scale; 1 = 72 dpi.

    Returns:
        Iterator[tuple[int, Image.Image, str]]: (page index from 0, grayscale image, raw text).
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            image = page.render(scale=scale, grayscale=True).to_pil().convert("L")
            text = page.get_textpage().get_text_range()
            yield index, image, text
    finally:
        pdf.close()
