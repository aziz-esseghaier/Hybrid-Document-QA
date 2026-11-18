"""Tests for the visual retriever's helpers."""

import io

import numpy as np
import pytest
from PIL import Image

from doc_qa.retrieval.visual import decode_image, maxsim, token_cosines, trim_padding


def test_decode_image_gives_rgb():
    buffer = io.BytesIO()
    Image.new("L", (4, 3), 200).save(buffer, "PNG")
    image = decode_image(buffer.getvalue())
    assert (image.mode, image.size) == ("RGB", (4, 3))


def test_trim_padding_handles_either_side():
    vectors = np.arange(12.0).reshape(2, 3, 2)
    mask = np.array([[1, 1, 0], [0, 1, 1]])
    first, second = trim_padding(vectors, mask)
    assert first.tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert second.tolist() == [[8.0, 9.0], [10.0, 11.0]]


def test_maxsim_sums_each_query_tokens_best_match():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    page = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])
    assert maxsim(query, page) == pytest.approx(1.0 + 0.8)


def test_token_cosines():
    first = np.array([[1.0, 0.0], [0.0, 1.0]])
    second = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert token_cosines(first, second).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError):
        token_cosines(first, second[:1])
