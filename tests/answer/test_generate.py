"""Tests for the retry loop, with a fake answer model."""

from doc_qa.answer.generate import answer_with_retries
from doc_qa.answer.output import Answer
from doc_qa.answer.prompt import PageInput, build_messages

GOOD = '{"answer": "42", "pages": [1], "found": true}'


class FakeModel:
    """Replies from a script, and records the chats it was given."""

    def __init__(self, replies):
        self.replies, self.chats = list(replies), []

    def __call__(self, messages):
        self.chats.append(messages)
        return self.replies.pop(0)


def messages():
    return build_messages("Q?", [PageInput("d/1", "img", "t"), PageInput("d/2", "img", "t")], 100)


def test_first_usable_reply_is_kept():
    model = FakeModel([GOOD])
    result = answer_with_retries(model, messages(), 2, retries=2)
    assert result.answer == Answer("42", (1,), True)
    assert (result.attempts, result.errors) == (1, ())


def test_retry_sees_the_rejected_reply_and_the_reason():
    model = FakeModel(["no json here", GOOD])
    result = answer_with_retries(model, messages(), 2, retries=2)
    assert result.answer == Answer("42", (1,), True)
    assert result.attempts == 2 and result.errors == ("it contains no JSON object",)
    retry_chat = model.chats[1]
    assert retry_chat[-2]["content"][0]["text"] == "no json here"
    assert "no JSON object" in retry_chat[-1]["content"][0]["text"]


def test_gives_up_after_the_retries():
    model = FakeModel(["x", '{"answer": "a", "pages": [9], "found": true}', "y"])
    result = answer_with_retries(model, messages(), 2, retries=2)
    assert result.answer is None and result.attempts == 3
    assert len(result.errors) == 3 and "do not exist" in result.errors[1]
    assert not model.replies  # exactly 3 calls
