"""Tests for overlap detection."""

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doc_qa.core.schemas import Document
from doc_qa.evaluation.overlap import (
    PageHash,
    find_exact_matches,
    find_phash_matches,
    find_title_matches,
    hash_image_bytes,
    hash_pages,
    load_hashes,
    normalize_title,
    page_view,
    page_views,
    save_hashes,
    verify_pair,
)

WORDS = "the energy report shows growth of renewable capacity in europe while demand fell across sectors".split()


def make_page(layout: str, size: tuple[int, int] = (850, 1100)) -> Image.Image:
    """Synthetic page with a "report" or "chart" block layout."""
    w, h = size
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)
    if layout == "report":
        draw.rectangle([0.1 * w, 0.05 * h, 0.9 * w, 0.12 * h], fill="black")
        for i in range(8):
            y = 0.2 * h + i * 0.08 * h
            draw.rectangle([0.1 * w, y, 0.9 * w, y + 0.03 * h], fill="gray")
    else:  # "chart": one big dark block bottom-right
        draw.rectangle([0.45 * w, 0.5 * h, 0.95 * w, 0.95 * h], fill="black")
    return page


def make_text_page(seed: int, size: tuple[int, int] = (850, 1100)) -> Image.Image:
    """Synthetic text page. Same layout for every seed, different text."""
    rng = random.Random(seed)
    w, h = size
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=14)
    draw.text((0.1 * w, 0.06 * h), f"Chapter {seed}", fill="black", font=ImageFont.load_default(size=28))
    for i in range(48):
        line = ""
        while draw.textlength(line + "longest ", font=font) < 0.8 * w:
            line += rng.choice(WORDS) + " "
        draw.text((0.1 * w, 0.14 * h + i * 0.78 * h / 48), line, fill="black", font=font)
    draw.text((w / 2, 0.96 * h), "12", fill="black", font=font)
    return page


def make_blank_page(number: str, size: tuple[int, int] = (850, 1100)) -> Image.Image:
    """Blank page with only a page number."""
    page = Image.new("RGB", size, "white")
    ImageDraw.Draw(page).text((size[0] / 2, 0.96 * size[1]), number, fill="black", font=ImageFont.load_default(size=14))
    return page


def encode(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    """Encode an image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def rerender(page: Image.Image, scale: float, border: int = 0, crop: int = 0) -> bytes:
    """Re-render a page with optional crop/border, rescaled, as JPEG."""
    if crop:
        page = page.crop((crop, crop, page.width, page.height))
    if border:
        framed = Image.new("RGB", (page.width + 2 * border, page.height + 2 * border), "white")
        framed.paste(page, (border, border))
        page = framed
    resized = page.resize((round(page.width * scale), round(page.height * scale)), Image.LANCZOS)
    return encode(resized, "JPEG", quality=75)


def distance(a: str, b: str) -> int:
    """Hamming distance between two hex pHashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def test_exact_hash_ignores_encoding_but_not_pixels():
    page = make_page("report")
    fast, small = encode(page, compress_level=1), encode(page, compress_level=9)
    assert fast != small  # different files...
    assert hash_image_bytes(fast)[0] == hash_image_bytes(small)[0]  # ...same pixels
    resized = encode(page.resize((425, 550)))
    assert hash_image_bytes(fast)[0] != hash_image_bytes(resized)[0]


def test_phash_matches_resized_page_but_not_other_page():
    page = make_page("report")
    original = hash_image_bytes(encode(page))[1]
    rerendered = hash_image_bytes(encode(page.resize((1275, 1650)), "JPEG", quality=80))[1]
    other = hash_image_bytes(encode(make_page("chart")))[1]
    assert distance(original, rerendered) <= 8
    assert distance(original, other) > 8


def test_find_matches():
    dev = [PageHash("dev/1", "aa", "ffff000000000000", 100, 130), PageHash("dev/2", "bb", "0000000000000000", 100, 130)]
    test = [
        PageHash("test/1", "aa", "ffff000000000001", 100, 130),  # same pixels, 1 bit apart
        PageHash("test/2", "cc", "0000000000000000", 160, 90),  # same pHash but landscape
    ]
    assert find_exact_matches(dev, test) == [("dev/1", "test/1")]
    assert find_phash_matches(dev, test, max_distance=8, max_aspect_ratio_diff=0.05) == [("dev/1", "test/1", 1)]


def test_hash_pages_in_parallel_matches_serial(tmp_path):
    images = [(f"p/{i}", encode(make_page("report" if i % 2 else "chart"))) for i in range(4)]
    serial = hash_pages(images, workers=1)
    assert hash_pages(images, workers=2, batch_size=3) == serial
    save_hashes(serial, tmp_path / "h.csv")
    assert load_hashes(tmp_path / "h.csv") == serial


def test_ink_share_separates_blank_and_text_pages():
    assert page_view(encode(Image.new("RGB", (850, 1100), "white"))).ink_share == 0
    assert page_view(encode(make_blank_page("12"))).ink_share < 0.01
    assert page_view(encode(make_text_page(1))).ink_share > 0.01


def test_verify_pair_confirms_rerendered_page_only():
    page = make_text_page(1)
    original = page_view(encode(page))
    for scale, border, crop in [(0.6, 0, 0), (1.5, 0, 0), (0.8, 40, 0), (0.7, 0, 10)]:
        status, correlation = verify_pair(original, page_view(rerender(page, scale, border, crop)), 0.01, 0.9)
        assert status == "confirmed", (scale, border, crop, correlation)

    # Same layout, other text: the pHash can match, the pixels must not.
    status, correlation = verify_pair(original, page_view(encode(make_text_page(2))), 0.01, 0.9)
    assert status == "rejected" and correlation < 0.85

    blank_12, blank_37 = page_view(encode(make_blank_page("12"))), page_view(encode(make_blank_page("37")))
    assert verify_pair(blank_12, blank_37, 0.01, 0.9) == ("blank", None)


def test_page_views_in_parallel_match_serial():
    images = [(f"p/{i}", encode(make_text_page(i))) for i in range(3)]
    serial = page_views(images, size=64, workers=1)
    parallel = page_views(images, size=64, workers=2, batch_size=2)
    assert list(parallel) == list(serial)
    for page_id, view in serial.items():
        assert parallel[page_id].ink_share == view.ink_share
        np.testing.assert_array_equal(parallel[page_id].pixels, view.pixels)
        assert view.pixels.shape == (64, 64)


def test_title_matching():
    assert normalize_title("Rapport_Shift-Project_2021.PDF") == "rapport shift project 2021"
    dev = [
        Document("v1/a", "Rapport_Shift-Project_Mai-2021.pdf"),
        Document("v1/2315", "2315"),
        Document("v2/b", "ESG report", "https://www.x.com/esg.pdf"),
    ]
    test = [
        Document("v3/a", "rapport-shift-project-mai-2021.pdf"),
        Document("v3/2315", "2315"),
        Document("v3/b", "Other title", "http://x.com/esg.pdf/"),
    ]
    found = {(d, t, k) for d, t, _, k in find_title_matches(dev, test, min_ratio=0.85)}
    assert found == {("v1/a", "v3/a", "title"), ("v2/b", "v3/b", "url")}
