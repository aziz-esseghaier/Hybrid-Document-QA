"""Tests for reranker prompts and batching."""

import numpy as np
import pytest

from doc_qa.retrieval.rerank import PREFIX, SUFFIX, pair_ids, split_on_error


class CharTokenizer:
    """One token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def decode(ids):
    return "".join(chr(i) for i in ids)


def test_prompt_follows_the_model_card_format():
    text = decode(pair_ids(CharTokenizer(), "Judge.", "Q?", "page text", 10_000))
    assert text == PREFIX + "<Instruct>: Judge.\n<Query>: Q?\n<Document>: page text" + SUFFIX


def test_long_pages_lose_their_end_but_keep_the_answer_slot():
    ids = pair_ids(CharTokenizer(), "Judge.", "Q?", "x" * 5000, 400)
    text = decode(ids)
    assert len(ids) == 400
    assert text.startswith(PREFIX + "<Instruct>: Judge.\n<Query>: Q?\n<Document>: x")
    assert text.endswith("x" + SUFFIX)  # the model still answers right after the assistant tag


def test_a_batch_that_runs_out_of_memory_is_split_and_keeps_its_order():
    calls, cleanups = [], []

    def forward(batch):
        calls.append(list(batch))
        if len(batch) > 1 and 7 in batch:  # index 7 is too long to share a batch
            raise MemoryError
        return np.array(batch, dtype=float) * 10

    scores = split_on_error([5, 6, 7, 8], forward, MemoryError, lambda: cleanups.append(1))
    assert scores.tolist() == [50, 60, 70, 80]
    assert calls == [[5, 6, 7, 8], [5, 6], [7, 8], [7], [8]]
    assert len(cleanups) == 2


def test_a_single_item_that_runs_out_of_memory_still_fails():
    def forward(batch):
        raise MemoryError

    with pytest.raises(MemoryError):
        split_on_error([3], forward, MemoryError)
