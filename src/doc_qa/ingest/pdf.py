"""PDF text layer and page rendering with pypdfium2.

Requires `pypdfium2`. PDFium is not thread-safe: use processes, not threads.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from PIL import Image

_HYPHEN_BREAK = re.compile(r"[\x02\xad]\s*")  # PDFium's end-of-line hyphen marker, soft hyphen
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f￾￿]")
_SPACES = re.compile(r"[ \t\xa0]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_WORD_CHAR = re.compile(r"\w")


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
