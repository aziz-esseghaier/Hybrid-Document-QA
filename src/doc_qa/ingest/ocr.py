"""Page OCR with PaddleOCR-VL. Requires `paddleocr` (not a base dependency)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


def load_pipeline(pipeline_version: str = "v1.6", device: str = "gpu:0", **options: Any) -> Any:
    """Create a PaddleOCR-VL pipeline.

    Args:
        pipeline_version (str): PaddleOCR-VL version.
        device (str): Paddle device, e.g. "gpu:0" or "cpu".
        **options (Any): Extra `PaddleOCRVL` arguments.

    Returns:
        Any: Pipeline.
    """
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(pipeline_version=pipeline_version, device=device, **options)


def result_markdown(result: Any) -> str:
    """Extract the Markdown text of one pipeline result.

    Args:
        result (Any): Item returned by `pipeline.predict`.

    Returns:
        str: Markdown text.
    """
    markdown = result.markdown
    if isinstance(markdown, dict):
        return markdown.get("markdown_texts") or ""
    return str(markdown or "")


def ocr_images(pipeline: Any, images: list[bytes], **predict_options: Any) -> list[str]:
    """OCR encoded page images to Markdown.

    Args:
        pipeline (Any): Pipeline from `load_pipeline`.
        images (list[bytes]): Encoded images (JPEG or PNG).
        **predict_options (Any): Extra `predict` arguments.

    Returns:
        list[str]: Markdown per image, in input order.
    """
    if not images:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, data in enumerate(images):
            path = Path(tmp) / f"{i}{'.jpg' if data[:3] == b'\xff\xd8\xff' else '.png'}"
            path.write_bytes(data)
            paths.append(str(path))
        texts = [result_markdown(r) for r in pipeline.predict(paths, **predict_options)]
    if len(texts) != len(images):
        raise RuntimeError(f"Expected {len(images)} OCR results, got {len(texts)}.")
    return texts
