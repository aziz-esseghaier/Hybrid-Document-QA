"""Find dev pages and documents that also appear in the test set.

Two page fingerprints find candidate pairs:
- exact hash: SHA-256 of the decoded RGB pixels. Catches identical renders, whatever the file encoding.
- perceptual hash (pHash): a 64-bit fingerprint of the page's coarse layout. Catches the same page rendered
  at another resolution, which the exact hash misses (each ViDoRe version rendered its own images).

Each candidate pair is then verified, because both fingerprints also match pages that merely look alike:
- nearly blank pages (end pages, section breaks) look the same in every document, so they are skipped;
- the other pairs are compared pixel by pixel, after cropping each page to its content and shrinking both
  to the same small grid. Only pairs with a high correlation count as the same page.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass
from difflib import SequenceMatcher
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import imagehash
import numpy as np
from PIL import Image

from doc_qa.core.schemas import Document

# Number of set bits in each byte value, used to count differing bits between hashes.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class PageHash:
    """Fingerprints of one page image."""

    page_id: str
    sha256: str
    phash: str  # 16 hex characters = 64 bits
    width: int
    height: int


# ---------------------------------------------------------------- hashing


def hash_image_bytes(data: bytes) -> tuple[str, str, int, int]:
    """Compute the exact and perceptual hashes of an encoded image.

    Args:
        data (bytes): PNG or JPEG bytes.

    Returns:
        tuple[str, str, int, int]: (SHA-256 of size + RGB pixels, pHash as hex, width, height).
    """
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    return digest, str(imagehash.phash(rgb)), rgb.width, rgb.height


def _batched(items: Iterable, size: int) -> Iterator[list]:
    """Split an iterable into lists of at most `size` items.

    Args:
        items (Iterable): Items to split.
        size (int): Maximum list length.

    Returns:
        Iterator[list]: Consecutive batches.
    """
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def _map_images(
    fn: Callable[[bytes], Any], images: Iterable[tuple[str, bytes]], workers: int | None, batch_size: int
) -> Iterator[tuple[str, Any]]:
    """Apply a function to page images in parallel on CPU processes.

    Images are sent in bounded batches, so memory stays small however many pages there are.

    Args:
        fn (Callable[[bytes], Any]): Function of the encoded image. Must be picklable (module level).
        images (Iterable[tuple[str, bytes]]): (page id, image bytes) pairs.
        workers (int | None): Number of processes; None uses all CPU cores, 1 runs in this process.
        batch_size (int): Images in memory at once.

    Returns:
        Iterator[tuple[str, Any]]: (page id, result) pairs, in input order.
    """
    if workers == 1:
        for page_id, data in images:
            yield page_id, fn(data)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in _batched(images, batch_size):
            ids = [page_id for page_id, _ in batch]
            yield from zip(ids, pool.map(fn, [data for _, data in batch], chunksize=8))


def hash_pages(images: Iterable[tuple[str, bytes]], workers: int | None = None, batch_size: int = 256) -> list[PageHash]:
    """Hash page images in parallel on CPU processes.

    Args:
        images (Iterable[tuple[str, bytes]]): (page id, image bytes) pairs.
        workers (int | None): Number of processes; None uses all CPU cores, 1 runs in this process.
        batch_size (int): Images in memory at once.

    Returns:
        list[PageHash]: One entry per image, in input order.
    """
    return [PageHash(page_id, *h) for page_id, h in _map_images(hash_image_bytes, images, workers, batch_size)]


def save_hashes(hashes: list[PageHash], path: Path) -> None:
    """Save page hashes to a CSV file.

    Args:
        hashes (list[PageHash]): The hashes.
        path (Path): Output CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_id", "sha256", "phash", "width", "height"])
        writer.writerows(astuple(h) for h in hashes)


def load_hashes(path: Path) -> list[PageHash]:
    """Load page hashes saved by `save_hashes`.

    Args:
        path (Path): CSV file.

    Returns:
        list[PageHash]: The hashes.
    """
    with path.open(newline="", encoding="utf-8") as f:
        return [PageHash(r["page_id"], r["sha256"], r["phash"], int(r["width"]), int(r["height"])) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- page matching


def find_exact_matches(dev: list[PageHash], test: list[PageHash]) -> list[tuple[str, str]]:
    """Find dev and test pages with identical pixels.

    Args:
        dev (list[PageHash]): Dev page hashes.
        test (list[PageHash]): Test page hashes.

    Returns:
        list[tuple[str, str]]: (dev page id, test page id) pairs.
    """
    test_by_sha = defaultdict(list)
    for h in test:
        test_by_sha[h.sha256].append(h.page_id)
    return [(d.page_id, t) for d in dev for t in test_by_sha.get(d.sha256, [])]


def find_phash_matches(
    dev: list[PageHash],
    test: list[PageHash],
    max_distance: int,
    max_aspect_ratio_diff: float,
    chunk_size: int = 256,
) -> list[tuple[str, str, int]]:
    """Find dev and test pages whose perceptual hashes differ by at most `max_distance` bits.

    Compares every dev page with every test page, a block of dev pages at a time, with vectorized XOR and
    bit counting. Pairs with clearly different shapes (width/height ratio) are skipped, because pHash
    squeezes every image into a square and can confuse a slide with a portrait page.

    Args:
        dev (list[PageHash]): Dev page hashes.
        test (list[PageHash]): Test page hashes.
        max_distance (int): Largest Hamming distance (0 to 64) counted as a match.
        max_aspect_ratio_diff (float): Largest relative difference of width/height ratios.
        chunk_size (int): Dev pages compared at once. Bounds memory use.

    Returns:
        list[tuple[str, str, int]]: (dev page id, test page id, distance), closest first.
    """
    if not dev or not test:
        return []
    test_bits = np.array([int(h.phash, 16) for h in test], dtype=np.uint64)
    test_ratio = np.array([h.width / h.height for h in test])
    matches = []
    for start in range(0, len(dev), chunk_size):
        block = dev[start : start + chunk_size]
        dev_bits = np.array([int(h.phash, 16) for h in block], dtype=np.uint64)
        dev_ratio = np.array([h.width / h.height for h in block])

        xor = np.bitwise_xor(dev_bits[:, None], test_bits[None, :])
        distance = _POPCOUNT[xor.view(np.uint8)].reshape(len(block), len(test), 8).sum(axis=2)
        same_shape = np.abs(dev_ratio[:, None] - test_ratio[None, :]) <= max_aspect_ratio_diff * test_ratio[None, :]

        for i, j in zip(*np.nonzero((distance <= max_distance) & same_shape)):
            matches.append((block[i].page_id, test[j].page_id, int(distance[i, j])))
    return sorted(matches, key=lambda m: (m[2], m[0], m[1]))


# ---------------------------------------------------------------- verification of candidate pairs


@dataclass(frozen=True, eq=False)
class PageView:
    """A page reduced for pixel comparison."""

    ink_share: float  # share of pixels that stand out from the background (0 = blank page)
    pixels: np.ndarray  # (size, size) uint8 grayscale image of the page's content box


def page_view(data: bytes, size: int = 128, contrast: int = 40) -> PageView:
    """Measure how much content a page has, and shrink its content box to a fixed grid.

    The background is the most common gray level. Pixels differing from it by more than `contrast` are ink.
    Cropping to the box around the ink removes margin and border differences between two renders of the
    same page; shrinking to a small grid removes resolution and compression differences.

    Args:
        data (bytes): PNG or JPEG bytes.
        size (int): Side of the output grid in pixels.
        contrast (int): Gray-level difference (0 to 255) from the background that counts as ink.

    Returns:
        PageView: Ink share and the shrunk content box.
    """
    with Image.open(io.BytesIO(data)) as image:
        gray = image.convert("L")
    pixels = np.asarray(gray)
    background = int(np.bincount(pixels.ravel(), minlength=256).argmax())
    ink = np.abs(pixels.astype(np.int16) - background) > contrast
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    if rows.size:
        gray = gray.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
    return PageView(float(ink.mean()), np.asarray(gray.resize((size, size), Image.BOX)))


def page_views(
    images: Iterable[tuple[str, bytes]], size: int = 128, workers: int | None = None, batch_size: int = 64
) -> dict[str, PageView]:
    """Compute `page_view` for many pages in parallel on CPU processes.

    Args:
        images (Iterable[tuple[str, bytes]]): (page id, image bytes) pairs.
        size (int): Side of the output grid in pixels.
        workers (int | None): Number of processes; None uses all CPU cores, 1 runs in this process.
        batch_size (int): Images in memory at once.

    Returns:
        dict[str, PageView]: View of each page, by page id.
    """
    return dict(_map_images(partial(page_view, size=size), images, workers, batch_size))


def pixel_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Pearson correlation of two images of the same shape.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Returns:
        float: Correlation from -1 to 1; 1 means the same image up to brightness and contrast. 0 if either
            image is uniform.
    """
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / denominator) if denominator > 0 else 0.0


def verify_pair(dev: PageView, test: PageView, min_ink_share: float, min_correlation: float) -> tuple[str, float | None]:
    """Decide whether a candidate pair is really the same page.

    Args:
        dev (PageView): The dev page.
        test (PageView): The test page.
        min_ink_share (float): Pages with less ink than this are nearly blank: no evidence either way.
        min_correlation (float): Smallest pixel correlation counted as the same page.

    Returns:
        tuple[str, float | None]: Status ("blank", "rejected" or "confirmed") and the correlation (None for
            blank pages).
    """
    if min(dev.ink_share, test.ink_share) < min_ink_share:
        return "blank", None
    correlation = pixel_correlation(dev.pixels, test.pixels)
    return ("confirmed" if correlation >= min_correlation else "rejected"), correlation


# ---------------------------------------------------------------- document matching


def normalize_title(title: str) -> str:
    """Normalize a document title or file name for comparison.

    Args:
        title (str): Title or file name, such as `Rapport_Shift-Project_2021.pdf`.

    Returns:
        str: Lowercase words separated by single spaces, without the file extension.
    """
    text = re.sub(r"\.(pdf|png|jpe?g)$", "", title.strip().lower())
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison (no scheme, no `www.`, no trailing slash, lowercase).

    Args:
        url (str): The URL.

    Returns:
        str: The normalized URL.
    """
    url = re.sub(r"^https?://(www\.)?", "", url.strip().lower())
    return url.rstrip("/")


def find_title_matches(dev: list[Document], test: list[Document], min_ratio: float) -> list[tuple[str, str, float, str]]:
    """Find dev and test documents with the same URL or similar titles.

    Titles made only of digits (TabFQuAD image numbers) or shorter than 5 characters are skipped: they
    carry no meaning and would match by chance.

    Args:
        dev (list[Document]): Dev documents.
        test (list[Document]): Test documents.
        min_ratio (float): Smallest difflib similarity (0 to 1) counted as a match.

    Returns:
        list[tuple[str, str, float, str]]: (dev doc id, test doc id, similarity, "url" or "title").
    """
    def usable(title: str) -> bool:
        return len(title) >= 5 and not title.replace(" ", "").isdigit()

    test_norm = [(t, normalize_title(t.title), _normalize_url(t.url) if t.url else None) for t in test]
    matches = []
    for d in dev:
        d_title = normalize_title(d.title)
        d_url = _normalize_url(d.url) if d.url else None
        for t, t_title, t_url in test_norm:
            if d_url and t_url and d_url == t_url:
                matches.append((d.doc_id, t.doc_id, 1.0, "url"))
            elif usable(d_title) and usable(t_title):
                ratio = SequenceMatcher(None, d_title, t_title).ratio()
                if ratio >= min_ratio:
                    matches.append((d.doc_id, t.doc_id, round(ratio, 3), "title"))
    return sorted(matches, key=lambda m: -m[2])


# ---------------------------------------------------------------- review images


def side_by_side(left: bytes, right: bytes, height: int = 900, gap: int = 20) -> Image.Image:
    """Put two page images next to each other at the same height, for manual review.

    Args:
        left (bytes): Encoded image shown on the left (the dev page).
        right (bytes): Encoded image shown on the right (the test page).
        height (int): Height of the output in pixels.
        gap (int): White space between the two pages, in pixels.

    Returns:
        Image.Image: The combined image.
    """
    pages = []
    for data in (left, right):
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
        pages.append(rgb.resize((max(1, round(rgb.width * height / rgb.height)), height)))
    canvas = Image.new("RGB", (pages[0].width + gap + pages[1].width, height), "white")
    canvas.paste(pages[0], (0, 0))
    canvas.paste(pages[1], (pages[0].width + gap, 0))
    return canvas
