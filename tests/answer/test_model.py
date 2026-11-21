"""Tests for sizing page images for the answer model."""

from PIL import Image

from doc_qa.answer.model import PIXELS_PER_TOKEN, fit_pixels


def test_large_pages_shrink_to_the_token_budget_and_keep_their_shape():
    page = Image.new("RGB", (1700, 2200))
    small = fit_pixels(page, 640)
    assert small.width * small.height <= 640 * PIXELS_PER_TOKEN
    assert abs(small.width / small.height - 1700 / 2200) < 0.01


def test_small_pages_are_kept():
    page = Image.new("RGB", (300, 400))
    assert fit_pixels(page, 640) is page
