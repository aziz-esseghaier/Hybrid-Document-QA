"""Tests for page hashing and dev/test matching."""

import io

from PIL import Image, ImageDraw

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
    save_hashes,
)


def make_page(layout: str, size: tuple[int, int] = (850, 1100)) -> Image.Image:
    """Draw a synthetic page: a title bar plus text-like blocks placed by `layout`."""
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


def encode(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    """Encode an image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


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
