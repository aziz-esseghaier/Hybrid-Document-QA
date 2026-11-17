"""Tests for PDF text-layer cleaning and reading."""

import pytest

from doc_qa.ingest.pdf import clean_pdf_text, read_pdf, word_chars


def test_clean_pdf_text():
    raw = "Scope 1 emis\x02\r\nsions fell\xa0 12%\r\n\r\n\r\n\r\nNet\xadzero  by 2040 \x00\r\n"
    assert clean_pdf_text(raw) == "Scope 1 emissions fell 12%\n\nNetzero by 2040"


def test_word_chars():
    assert word_chars("12 % – CO2e ...") == 6
    assert word_chars(" \n•\n") == 0


def test_read_pdf(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 300)
    pdf.new_page(300, 200)
    pdf.save(tmp_path / "doc.pdf")
    pdf.close()

    pages = list(read_pdf(tmp_path / "doc.pdf", scale=0.5))
    assert [(i, image.size, image.mode, text) for i, image, text in pages] == [
        (0, (100, 150), "L", ""),
        (1, (150, 100), "L", ""),
    ]
