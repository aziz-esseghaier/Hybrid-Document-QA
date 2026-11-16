"""Dev/test overlap detection: page hashes, pixel verification, title matching."""

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
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import imagehash
import numpy as np
from PIL import Image

from doc_qa.core.parallel import batched
from doc_qa.core.schemas import Document

# Set bits per byte value.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class PageHash:
    """Fingerprints of one page image."""

    page_id: str
    sha256: str
    phash: str  # 64 bits as 16 hex chars
    width: int
    height: int


# ---------------------------------------------------------------- hashing


def hash_image_bytes(data: bytes) -> tuple[str, str, int, int]:
    """Compute the exact and perceptual hashes of an image.

    Args:
        data (bytes): Encoded image.

    Returns:
        tuple[str, str, int, int]: SHA-256 of size + RGB pixels, pHash (hex), width, height.
    """
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    return digest, str(imagehash.phash(rgb)), rgb.width, rgb.height


def _map_images(
    fn: Callable[[bytes], Any], images: Iterable[tuple[str, bytes]], workers: int | None, batch_size: int
) -> Iterator[tuple[str, Any]]:
    """Apply `fn` to images in a process pool, in bounded batches.

    Args:
        fn (Callable[[bytes], Any]): Picklable function of the encoded image.
        images (Iterable[tuple[str, bytes]]): (page id, image bytes).
        workers (int | None): Processes; None = all cores, 1 = in-process.
        batch_size (int): Images in memory at once.

    Returns:
        Iterator[tuple[str, Any]]: (page id, result), in input order.
    """
    if workers == 1:
        for page_id, data in images:
            yield page_id, fn(data)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in batched(images, batch_size):
            ids = [page_id for page_id, _ in batch]
            yield from zip(ids, pool.map(fn, [data for _, data in batch], chunksize=8))


def hash_pages(images: Iterable[tuple[str, bytes]], workers: int | None = None, batch_size: int = 256) -> list[PageHash]:
    """Hash page images in parallel.

    Args:
        images (Iterable[tuple[str, bytes]]): (page id, image bytes).
        workers (int | None): Processes; None = all cores, 1 = in-process.
        batch_size (int): Images in memory at once.

    Returns:
        list[PageHash]: One per image, in input order.
    """
    return [PageHash(page_id, *h) for page_id, h in _map_images(hash_image_bytes, images, workers, batch_size)]


def save_hashes(hashes: list[PageHash], path: Path) -> None:
    """Save page hashes to CSV.

    Args:
        hashes (list[PageHash]): Hashes.
        path (Path): Output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_id", "sha256", "phash", "width", "height"])
        writer.writerows(astuple(h) for h in hashes)


def load_hashes(path: Path) -> list[PageHash]:
    """Load page hashes from CSV.

    Args:
        path (Path): File written by `save_hashes`.

    Returns:
        list[PageHash]: Hashes.
    """
    with path.open(newline="", encoding="utf-8") as f:
        return [PageHash(r["page_id"], r["sha256"], r["phash"], int(r["width"]), int(r["height"])) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- candidate pairs


def find_exact_matches(dev: list[PageHash], test: list[PageHash]) -> list[tuple[str, str]]:
    """Find dev/test pages with identical pixels.

    Args:
        dev (list[PageHash]): Dev hashes.
        test (list[PageHash]): Test hashes.

    Returns:
        list[tuple[str, str]]: (dev page id, test page id).
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
    """Find dev/test pages within `max_distance` pHash bits and with similar aspect ratios.

    Args:
        dev (list[PageHash]): Dev hashes.
        test (list[PageHash]): Test hashes.
        max_distance (int): Max Hamming distance (0-64).
        max_aspect_ratio_diff (float): Max relative difference of width/height.
        chunk_size (int): Dev pages compared per block.

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


# ---------------------------------------------------------------- pixel verification


@dataclass(frozen=True, eq=False)
class PageView:
    """Downscaled page used for pixel comparison."""

    ink_share: float  # fraction of non-background pixels
    pixels: np.ndarray  # (size, size) uint8, content box only


def page_view(data: bytes, size: int = 128, contrast: int = 40) -> PageView:
    """Measure a page's ink and shrink its content box to a `size` x `size` grayscale grid.

    Args:
        data (bytes): Encoded image.
        size (int): Output side in pixels.
        contrast (int): Min gray-level difference from the background to count as ink.

    Returns:
        PageView: Ink share and downscaled content box.
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
    """Compute `page_view` for many pages in parallel.

    Args:
        images (Iterable[tuple[str, bytes]]): (page id, image bytes).
        size (int): Output side in pixels.
        workers (int | None): Processes; None = all cores, 1 = in-process.
        batch_size (int): Images in memory at once.

    Returns:
        dict[str, PageView]: View per page id.
    """
    return dict(_map_images(partial(page_view, size=size), images, workers, batch_size))


def pixel_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two same-shape images.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Returns:
        float: Correlation in [-1, 1]; 0 if either image is uniform.
    """
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / denominator) if denominator > 0 else 0.0


def verify_pair(dev: PageView, test: PageView, min_ink_share: float, min_correlation: float) -> tuple[str, float | None]:
    """Classify a candidate pair as blank, rejected or confirmed.

    Args:
        dev (PageView): Dev page.
        test (PageView): Test page.
        min_ink_share (float): Pages below this are treated as blank.
        min_correlation (float): Min pixel correlation for a match.

    Returns:
        tuple[str, float | None]: Status and correlation (None if blank).
    """
    if min(dev.ink_share, test.ink_share) < min_ink_share:
        return "blank", None
    correlation = pixel_correlation(dev.pixels, test.pixels)
    return ("confirmed" if correlation >= min_correlation else "rejected"), correlation


# ---------------------------------------------------------------- document matching


def normalize_title(title: str) -> str:
    """Lowercase a title, drop the file extension and collapse punctuation to spaces.

    Args:
        title (str): Title or file name.

    Returns:
        str: Normalized title.
    """
    text = re.sub(r"\.(pdf|png|jpe?g)$", "", title.strip().lower())
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _normalize_url(url: str) -> str:
    """Lowercase a URL and strip scheme, `www.` and trailing slash.

    Args:
        url (str): URL.

    Returns:
        str: Normalized URL.
    """
    url = re.sub(r"^https?://(www\.)?", "", url.strip().lower())
    return url.rstrip("/")


def find_title_matches(dev: list[Document], test: list[Document], min_ratio: float) -> list[tuple[str, str, float, str]]:
    """Find dev/test documents with the same URL or similar titles.

    Titles that are numeric or shorter than 5 chars are ignored.

    Args:
        dev (list[Document]): Dev documents.
        test (list[Document]): Test documents.
        min_ratio (float): Min difflib similarity (0-1).

    Returns:
        list[tuple[str, str, float, str]]: (dev doc id, test doc id, similarity, "url" | "title").
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
    """Place two images side by side at the same height.

    Args:
        left (bytes): Encoded left image (dev page).
        right (bytes): Encoded right image (test page).
        height (int): Output height in pixels.
        gap (int): Gap in pixels.

    Returns:
        Image.Image: Combined image.
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
