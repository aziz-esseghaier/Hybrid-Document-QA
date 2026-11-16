"""Tests for the OCR wrapper (with a fake pipeline)."""

import io
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from doc_qa.ingest.ocr import ocr_images, result_markdown


class FakePipeline:
    """Returns each input's file name as its Markdown."""

    def __init__(self):
        self.seen = []

    def predict(self, paths, **options):
        self.seen = [(Path(p).suffix, Path(p).exists()) for p in paths]
        return [SimpleNamespace(markdown={"markdown_texts": Path(p).name}) for p in paths]


def encode(fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format=fmt)
    return buffer.getvalue()


def test_ocr_images_keeps_order_and_format():
    pipeline = FakePipeline()
    texts = ocr_images(pipeline, [encode("JPEG"), encode("PNG")])
    assert texts == ["0.jpg", "1.png"]
    assert pipeline.seen == [(".jpg", True), (".png", True)]
    assert ocr_images(pipeline, []) == []


def test_result_markdown():
    assert result_markdown(SimpleNamespace(markdown={"markdown_texts": "# Title"})) == "# Title"
    assert result_markdown(SimpleNamespace(markdown={})) == ""
    assert result_markdown(SimpleNamespace(markdown="text")) == "text"
