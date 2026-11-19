"""Tests for PDF download, text-layer cleaning, quality signals and reading."""

import pytest
from PIL import Image

from doc_qa.ingest.pdf import (
    clean_pdf_text,
    download_pdf,
    garbled_reasons,
    read_pdf,
    read_text_layers,
    text_signals,
    word_chars,
)

ENGLISH = (
    "The company reduced the emissions of its stores by twelve percent in the last year. "
    "This progress comes from the energy plan that our teams have built with their suppliers, "
    "and we will report on it every year. "
) * 4
FRENCH = (
    "Le groupe a réduit les émissions de ses magasins de douze pour cent au cours de la dernière année. "
    "Cette baisse vient du plan pour l'énergie que nos équipes ont construit avec les fournisseurs, "
    "et nous en rendrons compte chaque année dans le rapport. "
) * 4
RULE = {
    "max_bad_char_share": 0.05,
    "max_symbol_share": 0.4,
    "max_spaced_letter_share": 0.3,
    "max_long_word_share": 0.2,
    "min_common_word_share": 0.03,
    "min_letters": 200,
    "min_distinct_words": 20,
}


def reasons(text: str) -> list[str]:
    return garbled_reasons(text_signals(text), RULE)


def test_clean_pdf_text():
    raw = "Scope 1 emis\x02\r\nsions fell\xa0 12%\r\n\r\n\r\n\r\nNet\xadzero  by 2040 \x00\r\n"
    assert clean_pdf_text(raw) == "Scope 1 emissions fell 12%\n\nNetzero by 2040"


def test_word_chars():
    assert word_chars("12 % – CO2e ...") == 6
    assert word_chars(" \n•\n") == 0


def test_clean_text_is_not_flagged():
    assert reasons(ENGLISH) == []
    assert reasons(FRENCH) == []


def test_numeric_table_is_not_flagged():
    table = "\n".join(f"Scope {i} 2023 {i * 1200} tCO2e {i}.5% 2022 {i * 900}" for i in range(1, 40))
    assert reasons(table) == []


def test_lost_characters_are_flagged():
    assert "bad_char_share" in reasons("�" * 40 + ENGLISH[:300])
    assert "bad_char_share" in reasons(" " * 30)
    assert "bad_char_share" in reasons("\x01\x03\x04 " * 30)


def test_normal_line_marks_do_not_count_as_bad_characters():
    assert text_signals("Scope 1 emis\x02\r\nsions\r\n\tfell\x0c")["bad_char_share"] == 0.0


def test_scrambled_text_is_flagged():
    shifted = "".join(chr(ord(c) - 29) if 62 <= ord(c) <= 126 else c for c in ENGLISH)
    assert reasons(shifted)
    table = str.maketrans("abcdefghijklmnopqrstuvwxyz", "qwertzuiopasdfghjklyxcvbnm")
    assert "common_word_share" in reasons(ENGLISH.translate(table))


def test_spacing_problems_are_flagged():
    assert "spaced_letter_share" in reasons(" ".join(ENGLISH.replace(" ", "")))
    assert "long_word_share" in reasons(ENGLISH.replace(". ", ".\n").replace(" ", ""))


def test_rule_skips_missing_checks_and_short_pages():
    signals = text_signals("Totally unrelated words here, see averyveryveryverylongwebaddressname")
    assert garbled_reasons(signals, {"min_common_word_share": 0.5, "min_distinct_words": 20}) == []
    assert garbled_reasons(signals, {"min_common_word_share": 0.5}) == ["common_word_share"]
    assert garbled_reasons(signals, {"max_long_word_share": 0.2, "min_letters": 200}) == []
    assert garbled_reasons(signals, {"max_long_word_share": 0.2}) == ["long_word_share"]
    assert garbled_reasons(signals, {}) == []


def test_download_pdf(tmp_path):
    good, bad = tmp_path / "good.pdf", tmp_path / "bad.pdf"
    good.write_bytes(b"%PDF-1.7 data")
    bad.write_bytes(b"<html>")
    out = tmp_path / "out" / "a.pdf"
    assert download_pdf(good.as_uri(), out, 5) == "ok"
    assert out.read_bytes() == b"%PDF-1.7 data"
    assert download_pdf(bad.as_uri(), tmp_path / "b.pdf", 5) == "not a PDF"
    assert download_pdf((tmp_path / "missing.pdf").as_uri(), tmp_path / "c.pdf", 5) == "URLError"


def test_read_text_layers(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 300)
    pdf.save(tmp_path / "doc.pdf")
    pdf.close()
    assert read_text_layers(tmp_path / "doc.pdf") == [""]


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


def test_read_pdf_in_color_keeps_channel_order(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(200, 300)
    picture = pdfium.PdfImage.new(pdf)
    picture.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (8, 8), (230, 20, 40))))
    picture.set_matrix(pdfium.PdfMatrix().scale(200, 300))  # fill the page
    page.insert_obj(picture)
    page.gen_content()
    pdf.save(tmp_path / "doc.pdf")
    pdf.close()

    [(_, image, _)] = read_pdf(tmp_path / "doc.pdf", scale=0.5, color=True)
    assert (image.mode, image.size) == ("RGB", (100, 150))
    red, green, blue = image.getpixel((50, 75))
    assert red > 200 and green < 60 and blue < 80  # red stays red, not swapped with blue
