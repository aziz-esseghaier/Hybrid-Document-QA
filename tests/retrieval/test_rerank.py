"""Tests for reranker prompts."""

from doc_qa.retrieval.rerank import PREFIX, SUFFIX, pair_ids


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
