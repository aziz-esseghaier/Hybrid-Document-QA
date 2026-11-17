"""Tests for OCR text cleaning and chunking."""

import re

import pytest

from doc_qa.ingest.text import chunk_text, clean_ocr_text, table_to_text, token_windows


def test_clean_drops_image_placeholders_and_tags():
    raw = (
        '<div style="text-align: center;"><img src="imgs/a.jpg" alt="Image" width="67%" /></div>\n'
        "# COMMUNITY\n\n\n\n![](imgs/b.jpg)\n"
        '<div style="text-align: center;">Our people &amp; values</div>  \n'
        "Emissions fell <5% in 2023."
    )
    assert clean_ocr_text(raw) == "# COMMUNITY\n\nOur people & values\nEmissions fell <5% in 2023."


def test_table_to_rows():
    table = (
        "<table border=1><tr><th>Year</th><th>CO<sub>2</sub> (t)</th></tr>"
        "<tr><td>2021</td><td>1,234<br>est.</td></tr><tr><td></td><td></td></tr>"
        "<tr><td>R&amp;D<td>5</tr></table>"
    )
    assert table_to_text(table) == "Year | CO2 (t)\n2021 | 1,234 est.\nR&D | 5"
    assert clean_ocr_text(f"Before\n{table}\nAfter") == "Before\n\nYear | CO2 (t)\n2021 | 1,234 est.\nR&D | 5\n\nAfter"


def test_clean_flattens_latex_and_literal_newlines():
    raw = (
        "Metric Tonnes ( $ CO_{2}e $)\n"
        "<table><tr><td>2022\\nMt  $ CO_{{2}}e $</td><td>574 $ ^{1} $</td><td>$ 12.5\\% $</td></tr></table>\n"
        "Costs were $5 million and $3 million. $$\\mathrm{E} = m c^{2}$$ \\nu stays in $ \\nu_{0} $."
    )
    assert clean_ocr_text(raw) == (
        "Metric Tonnes ( CO2e)\n\n2022 Mt CO2e | 574 1 | 12.5%\n\n"
        "Costs were $5 million and $3 million. E = m c2\nu stays in \\nu0."
    )


def test_token_windows():
    assert token_windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert token_windows(7, 4, 1) == [(0, 4), (3, 7)]
    assert token_windows(3, 4, 1) == [(0, 3)]
    assert token_windows(0, 4, 1) == []
    with pytest.raises(ValueError):
        token_windows(10, 4, 4)


def test_chunk_text_maps_tokens_to_characters():
    text = "a bb ccc dd e"
    offsets = [m.span() for m in re.finditer(r"\S+", text)]
    chunks = chunk_text(text, offsets, size=2, overlap=1)
    assert [c["text"] for c in chunks] == ["a bb", "bb ccc", "ccc dd", "dd e"]
    assert chunks[1] == {"text": "bb ccc", "start": 2, "end": 8, "tokens": 2}
    assert chunk_text("", [], 2, 1) == []
